"""sessions.py — session classification and ICT time windows (Phase 1).

Pure / stateless. list[dict] in, list[dict] out. No I/O, no broker, no DB.

All windows are defined in an EXPLICIT reference timezone (`tz`), which is a
REQUIRED parameter (no default) — ICT uses New York time, but the boundary
definition is meaningless without an explicit tz, so we force the caller to
state it. Uses the stdlib `zoneinfo` tz database (DST-correct, not fixed
UTC offsets).

Methodological note (flagged): exact kill-zone / Silver Bullet clock times
vary across ICT sources and the knowledge base contains CONFLICTING times
(e.g. one graphic lists NY AM 7am-9am; alerts repeatedly cite 9:30-11am).
These are CONVENTIONS, not authoritative. Defaults below are parameterizable
and stated explicitly; they were NOT adopted from KB authority.

Default window times (clock, in `tz`):
    Sessions:
        asian   : 20:00-00:00 (8pm-midnight, wraps midnight via start>end)
        london  : 02:00-05:00
        ny_am   : 07:00-10:00  (covers NY AM kill zone)
        ny_pm   : 13:00-16:00
    Kill zones (parameterizable via killzone_windows):
        london_kz : 02:00-05:00
        ny_am_kz  : 09:30-11:00
        ny_pm_kz  : 13:30-15:00
    Silver Bullet windows (parameterizable via silver_bullet_windows):
        london_sb : 03:00-04:00
        ny_am_sb  : 10:00-11:00
        ny_pm_sb  : 14:00-15:00

These produce time-anchored zones, NOT trade triggers.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .candles import _to_datetime

__all__ = ["detect_sessions", "detect_kill_zones", "detect_session_levels"]

# Default session windows (clock times in `tz`). (start, end) with inclusive
# start and exclusive end. Each is a (name, start_time, end_time) tuple.
# The Asian session wraps midnight: start=20:00 > end=00:00, so `_in_window`
# uses its wrap branch (t >= start OR t < end), covering 20:00-23:59:59.
_DEFAULT_SESSIONS = [
    ("asian", time(20, 0), time(0, 0)),
    ("london", time(2, 0), time(5, 0)),
    ("ny_am", time(7, 0), time(10, 0)),
    ("ny_pm", time(13, 0), time(16, 0)),
]

_DEFAULT_KILLZONES = [
    ("london_kz", time(2, 0), time(5, 0)),
    ("ny_am_kz", time(9, 30), time(11, 0)),
    ("ny_pm_kz", time(13, 30), time(15, 0)),
]

_DEFAULT_SILVER_BULLETS = [
    ("london_sb", time(3, 0), time(4, 0)),
    ("ny_am_sb", time(10, 0), time(11, 0)),
    ("ny_pm_sb", time(14, 0), time(15, 0)),
]


def _resolve_tz(tz: str) -> ZoneInfo:
    if not isinstance(tz, str) or not tz:
        raise ValueError("tz must be a non-empty IANA timezone string")
    try:
        return ZoneInfo(tz)
    except Exception as exc:
        raise ValueError(f"invalid timezone: {tz!r}") from exc


def _in_window(local_dt: datetime, start: time, end: time) -> bool:
    """True if local_dt's clock time is in [start, end).

    When start > end the window wraps midnight (e.g. asian 20:00-00:00):
    a time is in-window if it is >= start OR < end.
    """
    t = local_dt.time()
    if start <= end:
        return start <= t < end
    # Wraps midnight (e.g. asian 20:00-00:00).
    return t >= start or t < end


def detect_sessions(
    candles: list[dict],
    tz: str,
    session_windows: list[tuple[str, time, time]] | None = None,
) -> list[dict]:
    """Assign candles to sessions and aggregate high/low per session instance.

    A session instance is a contiguous run of candles assigned to the same
    session name on the same calendar date (in `tz`). Each instance emits:
        type          : "session"
        session_name  : str
        start_index   : int (first candle of the instance)
        end_index     : int (last candle of the instance)
        session_high  : float
        session_low   : float
        timestamp     : timestamp of the first candle

    Edge cases: candles in a gap (no session) are skipped; empty session ->
    skipped; DST handled by zoneinfo (the local clock is computed per candle).
    """
    zone = _resolve_tz(tz)
    windows = session_windows if session_windows is not None else _DEFAULT_SESSIONS
    if not candles:
        return []

    # Assign each candle to a session name (or None).
    assignments: list[str | None] = []
    for c in candles:
        dt = _to_datetime(c["timestamp"]).astimezone(zone)
        name = None
        for sname, start, end in windows:
            if _in_window(dt, start, end):
                name = sname
                break
        assignments.append(name)

    results: list[dict] = []
    i = 0
    n = len(candles)
    while i < n:
        name = assignments[i]
        if name is None:
            i += 1
            continue
        # Extend the run while same session name AND same calendar date.
        start_idx = i
        run_date = _to_datetime(candles[i]["timestamp"]).astimezone(zone).date()
        j = i
        while j < n and assignments[j] == name:
            jd = _to_datetime(candles[j]["timestamp"]).astimezone(zone).date()
            if jd != run_date:
                break
            j += 1
        end_idx = j - 1
        run = candles[start_idx:end_idx + 1]
        results.append({
            "type": "session",
            "session_name": name,
            "start_index": start_idx,
            "end_index": end_idx,
            "session_high": max(c["high"] for c in run),
            "session_low": min(c["low"] for c in run),
            "timestamp": candles[start_idx]["timestamp"],
        })
        i = j

    return results


def detect_kill_zones(
    candles: list[dict],
    tz: str,
    killzone_windows: list[tuple[str, time, time]] | None = None,
    silver_bullet_windows: list[tuple[str, time, time]] | None = None,
) -> list[dict]:
    """Emit kill-zone and Silver Bullet window instances covering the candles.

    Each window dict:
        type         : "killzone" | "silver_bullet"
        window_name  : str
        start_index  : int (first candle inside the window)
        end_index    : int (last candle inside the window)
        timestamp    : timestamp of the first candle

    A window instance is a contiguous run of candles whose local clock falls
    in the window on a single calendar date. Windows with no candles are
    skipped. DST is handled by zoneinfo.
    """
    zone = _resolve_tz(tz)
    kz = killzone_windows if killzone_windows is not None else _DEFAULT_KILLZONES
    sb = silver_bullet_windows if silver_bullet_windows is not None else _DEFAULT_SILVER_BULLETS
    if not candles:
        return []

    results: list[dict] = []
    for window_list, wtype in ((kz, "killzone"), (sb, "silver_bullet")):
        for wname, start, end in window_list:
            # Walk candles, grouping contiguous in-window candles by date.
            i = 0
            n = len(candles)
            while i < n:
                dt = _to_datetime(candles[i]["timestamp"]).astimezone(zone)
                if not _in_window(dt, start, end):
                    i += 1
                    continue
                run_date = dt.date()
                start_idx = i
                j = i
                while j < n:
                    jdt = _to_datetime(candles[j]["timestamp"]).astimezone(zone)
                    if jdt.date() != run_date or not _in_window(jdt, start, end):
                        break
                    j += 1
                end_idx = j - 1
                results.append({
                    "type": wtype,
                    "window_name": wname,
                    "start_index": start_idx,
                    "end_index": end_idx,
                    "timestamp": candles[start_idx]["timestamp"],
                })
                i = j

    # Sort by start_index for deterministic output.
    results.sort(key=lambda r: (r["start_index"], r["window_name"]))
    return results


# ── detect_session_levels ─────────────────────────────────────────────────────

# (clock time in `tz`, exact hour:minute match)
_OPEN_TARGETS: list[tuple[str, time]] = [
    ("midnight_open", time(0, 0)),
    ("true_day_open", time(18, 0)),
    ("london_open", time(2, 0)),
    ("open_830", time(8, 30)),
    ("open_930", time(9, 30)),
]

# (key prefix, start, end) — same inclusive-start / exclusive-end semantics as
# _DEFAULT_SESSIONS; the asia window wraps midnight via _in_window.
_LEVEL_SESSIONS: list[tuple[str, time, time]] = [
    ("asia", time(20, 0), time(0, 0)),
    ("london", time(2, 0), time(5, 0)),
    ("nyam", time(7, 0), time(10, 0)),
    ("nylunch", time(12, 0), time(13, 0)),
    ("nypm", time(13, 0), time(16, 0)),
]


def detect_session_levels(
    candles: list[dict],
    tz: str,
) -> dict:
    """Compute key ICT price levels from `candles`.

    Designed for a SINGLE trading day's worth of candles. Multi-day input is
    accepted but open prices return the first global match and high/low pairs
    aggregate across all matching candles regardless of date.

    Returns a flat dict with:
      - Five open prices (first candle whose local hour:minute equals the
        target): midnight_open (00:00), true_day_open (18:00, Globex day),
        london_open (02:00), open_830 (08:30), open_930 (09:30).
      - Five session high/low pairs (ten keys) aggregated over all candles
        in each window: asia (20:00-00:00), london (02:00-05:00),
        nyam (07:00-10:00), nylunch (12:00-13:00), nypm (13:00-16:00).

    All times are interpreted in `tz` (required, non-empty IANA string).
    Timestamps are assumed open-stamped (candle timestamp = bar open time).
    Any level not present in `candles` is None.
    """
    zone = _resolve_tz(tz)
    result: dict[str, Any] = {
        "midnight_open": None,
        "true_day_open": None,
        "london_open": None,
        "open_830": None,
        "open_930": None,
        "asia_high": None,
        "asia_low": None,
        "london_high": None,
        "london_low": None,
        "nyam_high": None,
        "nyam_low": None,
        "nylunch_high": None,
        "nylunch_low": None,
        "nypm_high": None,
        "nypm_low": None,
    }
    if not candles:
        return result

    local_dts = [_to_datetime(c["timestamp"]).astimezone(zone) for c in candles]

    for key, target in _OPEN_TARGETS:
        for c, dt in zip(candles, local_dts):
            if dt.hour == target.hour and dt.minute == target.minute:
                result[key] = c["open"]
                break

    for prefix, start, end in _LEVEL_SESSIONS:
        in_session = [c for c, dt in zip(candles, local_dts) if _in_window(dt, start, end)]
        if in_session:
            result[f"{prefix}_high"] = max(c["high"] for c in in_session)
            result[f"{prefix}_low"] = min(c["low"] for c in in_session)

    return result
