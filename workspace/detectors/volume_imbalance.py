"""volume_imbalance.py — volume imbalance and opening gaps (Phase 1).

Pure / stateless. list[dict] in, list[dict] out. No I/O, no broker, no DB.

Volume imbalance: a gap between consecutive candle BODIES where the wicks
(ranges) overlap but the bodies do not. Distinct from a full FVG (where the
entire ranges do not overlap). The body gap is the imbalance zone.

Opening gaps (NWOG/NDOG/session): the difference between a session's first
open and the prior session's close, detected from timestamp boundaries.
TIMEZONE-DEPENDENT: `tz` is a REQUIRED parameter (ICT uses New York time;
the boundary definition is meaningless without an explicit tz).

    NDOG (New Day Opening Gap): prior calendar-day close vs current-day open.
    NWOG (New Week Opening Gap): prior ISO-week close vs current-week open.
    session gap: prior session close vs current session open (uses sessions.py).

NWOG/NDOG count ("last N gaps") is a usage convention, not a detector rule —
the detector emits all; the consumer selects.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from .candles import _to_datetime
from .sessions import detect_sessions, _resolve_tz

__all__ = ["detect_volume_imbalance", "detect_opening_gaps"]


def detect_volume_imbalance(candles: list[dict]) -> list[dict]:
    """Detect volume imbalances between consecutive candles.

    A volume imbalance at index i (between candle i-1 and i) requires:
        - bodies do NOT overlap: max(open[i-1],close[i-1]) < min(open[i],close[i])
          OR min(open[i-1],close[i-1]) > max(open[i],close[i])
        - ranges DO overlap: high[i-1] >= low[i] and high[i] >= low[i-1]
    The imbalance zone is the body gap: [top, bottom].

    Each dict:
        type   : "volume_imbalance"
        top    : float (upper bound of body gap)
        bottom : float (lower bound of body gap)
        index  : int (candle i)
        timestamp : timestamp of candle i

    Edge cases: <2 candles -> []; identical open/close (touching bodies) ->
    no imbalance (requires a strict body gap).
    """
    if not candles:
        return []
    n = len(candles)
    if n < 2:
        return []

    results: list[dict] = []
    for i in range(1, n):
        a = candles[i - 1]
        b = candles[i]

        a_body_high = max(a["open"], a["close"])
        a_body_low = min(a["open"], a["close"])
        b_body_high = max(b["open"], b["close"])
        b_body_low = min(b["open"], b["close"])

        # Ranges must overlap.
        ranges_overlap = a["high"] >= b["low"] and b["high"] >= a["low"]
        if not ranges_overlap:
            continue

        # Bodies must NOT overlap (strict gap).
        if a_body_high < b_body_low:
            # Up gap between bodies.
            bottom = a_body_high
            top = b_body_low
        elif a_body_low > b_body_high:
            # Down gap between bodies.
            bottom = b_body_high
            top = a_body_low
        else:
            continue  # bodies overlap -> not an imbalance

        results.append({
            "type": "volume_imbalance",
            "top": top,
            "bottom": bottom,
            "index": i,
            "timestamp": b["timestamp"],
        })

    return results


def detect_opening_gaps(
    candles: list[dict],
    boundary: Literal["day", "week", "session"],
    tz: str,
) -> list[dict]:
    """Detect opening gaps at day/week/session boundaries (in `tz`).

    `boundary`:
        "day"     -> NDOG: prior calendar-day close vs current-day open.
        "week"    -> NWOG: prior ISO-week close vs current-week open.
        "session" -> prior session close vs current session open.

    Each dict:
        type         : "ndog" | "nwog" | "opening_gap"
        top          : float (max of prior_close, current_open)
        bottom       : float (min of prior_close, current_open)
        prior_close  : float
        current_open : float
        index        : int (first candle of the new period/session)
        timestamp    : timestamp of that candle

    Edge cases: no boundary crossing -> []; tz required; insufficient history
    for a prior period -> that boundary skipped.
    """
    zone = _resolve_tz(tz)
    if not candles:
        return []

    if boundary == "session":
        return _session_gaps(candles, zone)

    results: list[dict] = []
    type_name = "ndog" if boundary == "day" else "nwog"

    def _key(dt: datetime):
        if boundary == "day":
            return dt.date()
        # ISO week: (year, week number).
        iso = dt.isocalendar()
        return (iso[0], iso[1])

    prev_key = None
    prev_close = None
    for i, c in enumerate(candles):
        dt = _to_datetime(c["timestamp"]).astimezone(zone)
        key = _key(dt)
        if prev_key is not None and key != prev_key:
            # Boundary crossed: gap between prev_close and this open.
            current_open = c["open"]
            results.append({
                "type": type_name,
                "top": max(prev_close, current_open),
                "bottom": min(prev_close, current_open),
                "prior_close": prev_close,
                "current_open": current_open,
                "index": i,
                "timestamp": c["timestamp"],
            })
        prev_key = key
        prev_close = c["close"]

    return results


def _session_gaps(candles: list[dict], zone: ZoneInfo) -> list[dict]:
    """Opening gaps between consecutive session instances."""
    sessions = detect_sessions(candles, tz=str(zone))
    results: list[dict] = []
    for k in range(1, len(sessions)):
        prev = sessions[k - 1]
        cur = sessions[k]
        prev_close = candles[prev["end_index"]]["close"]
        current_open = candles[cur["start_index"]]["open"]
        results.append({
            "type": "opening_gap",
            "top": max(prev_close, current_open),
            "bottom": min(prev_close, current_open),
            "prior_close": prev_close,
            "current_open": current_open,
            "index": cur["start_index"],
            "timestamp": candles[cur["start_index"]]["timestamp"],
        })
    return results
