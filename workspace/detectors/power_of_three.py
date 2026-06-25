"""power_of_three.py — AMD (Accumulation/Manipulation/Distribution) (Phase 1).

Pure / stateless. list[dict] in, list[dict] out. No I/O, no broker, no DB.

Detects the Power of Three / AMD profile within a chosen period (day default),
tz-explicit:
    Accumulation : early consolidation range (first `accum_bars` candles of the
                   period). Zone = [min low, max high] over those candles.
    Manipulation : a sweep beyond the accumulation zone (high above zone high
                   = manipulation up; low below zone low = manipulation down)
                   that then reverses. The FIRST such sweep in the period.
    Distribution : a sustained move OPPOSITE the manipulation, confirmed by a
                   displacement candle in the opposite direction after the
                   manipulation candle.

PO3 is the most narrative/subjective ICT concept; deterministic encoding
requires hard choices for each leg (flagged HIGH ambiguity). Each leg's rule
is parameterized and documented. Requires a manipulation leg to qualify.

tz is REQUIRED (no default). period in {"day","week"}.
"""

from __future__ import annotations

from typing import Literal

from .candles import _to_datetime
from .displacement import detect_displacement
from .sessions import _resolve_tz

__all__ = ["detect_power_of_three"]


def detect_power_of_three(
    candles: list[dict],
    period: Literal["day", "week"] = "day",
    tz: str = None,
    accum_bars: int = 5,
) -> list[dict]:
    """Detect Power of Three (AMD) profiles per period.

    `tz` is REQUIRED (no default). Calling without it raises ``TypeError``.

    Each dict:
        type                    : "po3"
        period_start            : timestamp of the first candle of the period
        period_open             : float (open of the first candle)
        accumulation_zone       : [low, high] over the first `accum_bars` candles
        manipulation_index      : int | None
        manipulation_direction  : "up" | "down" | None
        distribution_direction  : "up" | "down" | None
        timestamp               : timestamp of the period's first candle

    Edge cases: incomplete period (no manipulation found) -> skipped; no
    manipulation -> not PO3 (skipped); tz required.
    """
    if tz is None:
        raise TypeError("detect_power_of_three() missing required argument: 'tz'")
    zone = _resolve_tz(tz)
    if not candles:
        return []

    def _key(dt):
        if period == "day":
            return dt.date()
        iso = dt.isocalendar()
        return (iso[0], iso[1])

    # Group candle indices by period key.
    periods: list[tuple[object, list[int]]] = []
    for i, c in enumerate(candles):
        dt = _to_datetime(c["timestamp"]).astimezone(zone)
        key = _key(dt)
        if not periods or periods[-1][0] != key:
            periods.append((key, [i]))
        else:
            periods[-1][1].append(i)

    # Precompute displacement indices for the distribution leg.
    disp_by_index = {d["index"]: d for d in detect_displacement(candles)}

    results: list[dict] = []
    for _key, idxs in periods:
        if len(idxs) < accum_bars + 1:
            continue
        accum_idxs = idxs[:accum_bars]
        accum_low = min(candles[j]["low"] for j in accum_idxs)
        accum_high = max(candles[j]["high"] for j in accum_idxs)
        period_open = candles[idxs[0]]["open"]
        period_start_ts = candles[idxs[0]]["timestamp"]

        manip_index = None
        manip_dir = None
        for j in idxs[accum_bars:]:
            cj = candles[j]
            if cj["high"] > accum_high and cj["close"] < accum_high:
                manip_index = j
                manip_dir = "up"
                break
            if cj["low"] < accum_low and cj["close"] > accum_low:
                manip_index = j
                manip_dir = "down"
                break

        if manip_index is None:
            continue  # no manipulation -> not PO3

        # Distribution: a displacement candle AFTER manipulation, opposite dir.
        dist_dir = "down" if manip_dir == "up" else "up"
        dist_found = False
        for j in idxs:
            if j <= manip_index:
                continue
            d = disp_by_index.get(j)
            if d is None:
                continue
            if dist_dir == "up" and d["type"] == "displacement_bullish":
                dist_found = True
                break
            if dist_dir == "down" and d["type"] == "displacement_bearish":
                dist_found = True
                break

        results.append({
            "type": "po3",
            "period_start": period_start_ts,
            "period_open": period_open,
            "accumulation_zone": [accum_low, accum_high],
            "manipulation_index": manip_index,
            "manipulation_direction": manip_dir,
            "distribution_direction": dist_dir if dist_found else None,
            "timestamp": period_start_ts,
        })

    return results
