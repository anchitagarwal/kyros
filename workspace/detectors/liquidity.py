"""liquidity.py — BSL/SSL, equal highs/lows, prior levels, sweeps (Phase 1).

Pure / stateless. list[dict] in, list[dict] out. No I/O, no broker, no DB.

    BSL (buy-side liquidity) = resting liquidity above highs / equal highs.
    SSL (sell-side liquidity) = resting liquidity below lows / equal lows.

    Equal highs/lows: >=2 SWING extremes within a tolerance band. Tolerance is
    a FRACTION of a reference range (average candle range), NOT an absolute
    price — this is the most abuse-prone parameter and is range-relative by
    design. Knowledge-base hardcoded pip values are rejected without
    methodological basis.

    PDH/PDL/PWH/PWL: prior day/week high/low from timestamp grouping in `tz`.
    `tz` is REQUIRED.

    Liquidity sweep: price trades beyond a known pool level then closes back
    inside (reversal). A clean breakout (no return) is NOT a sweep; it may be
    a BOS instead. Both can be emitted; the consumer disambiguates.

Lookahead-safety: sweeps confirmed at the candle that closes back inside.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from .candles import _to_datetime
from .market_structure import detect_swings
from .sessions import _resolve_tz

__all__ = ["detect_equal_levels", "detect_prior_levels", "detect_liquidity_sweeps"]


def _avg_range(candles: list[dict]) -> float:
    """Average (high - low) across candles; used as the reference range."""
    if not candles:
        return 0.0
    return sum(c["high"] - c["low"] for c in candles) / len(candles)


def detect_equal_levels(
    candles: list[dict],
    tolerance: float = 0.1,
    lookback: int = 2,
) -> list[dict]:
    """Detect equal highs / equal lows clusters among swing extremes.

    `tolerance` is a FRACTION of the average candle range (reference range).
    Two swing highs are "equal" if |price_a - price_b| <= tolerance * avg_range.
    Clusters are formed greedily: a swing joins the most recent cluster whose
    level is within tolerance, else starts a new cluster. Only clusters with
    >=2 members are emitted.

    Each dict:
        type            : "equal_highs" | "equal_lows"
        level           : float (mean of member prices)
        member_indices  : list[int]
        count           : int
        timestamp       : timestamp of the last member

    Edge cases: flat market -> not "equal highs everywhere" (we use SWING
    extremes, not adjacent candles); single extreme -> none; <2 swings -> [].
    """
    if not candles or tolerance < 0:
        return []
    swings = detect_swings(candles, lookback=lookback)
    if len(swings) < 2:
        return []

    ref = _avg_range(candles)
    if ref <= 0:
        return []
    band = tolerance * ref

    results: list[dict] = []
    for swing_type, out_type in (("swing_high", "equal_highs"),
                                 ("swing_low", "equal_lows")):
        ext = [s for s in swings if s["type"] == swing_type]
        clusters: list[list[dict]] = []
        for s in ext:
            placed = False
            # Try to join the most recent cluster within tolerance.
            for cl in reversed(clusters):
                level = sum(m["price"] for m in cl) / len(cl)
                if abs(s["price"] - level) <= band:
                    cl.append(s)
                    placed = True
                    break
            if not placed:
                clusters.append([s])
        for cl in clusters:
            if len(cl) >= 2:
                level = sum(m["price"] for m in cl) / len(cl)
                results.append({
                    "type": out_type,
                    "level": level,
                    "member_indices": [m["index"] for m in cl],
                    "count": len(cl),
                    "timestamp": cl[-1]["timestamp"],
                })

    return results


def detect_prior_levels(
    candles: list[dict],
    period: Literal["day", "week"],
    tz: str,
) -> list[dict]:
    """Emit prior day/week high/low (PDH/PDL/PWH/PWL) anchored at new periods.

    Groups candles by calendar day or ISO week in `tz`. When a new period
    begins, emits the HIGH and LOW of the just-completed prior period,
    anchored at the first candle of the new period.

    Each dict:
        type                : "pdh"|"pdl"|"pwh"|"pwl"
        level               : float
        source_period_start : timestamp of the first candle of the source period
        index               : int (first candle of the new period)
        timestamp           : timestamp of that anchor candle

    Edge cases: insufficient history for a prior period -> skipped; tz required.
    """
    zone = _resolve_tz(tz)
    if not candles:
        return []

    def _key(dt: datetime):
        if period == "day":
            return dt.date()
        iso = dt.isocalendar()
        return (iso[0], iso[1])

    high_name, low_name = ("pdh", "pdl") if period == "day" else ("pwh", "pwl")

    results: list[dict] = []
    cur_key = None
    cur_start_ts = None
    cur_high = None
    cur_low = None

    for i, c in enumerate(candles):
        dt = _to_datetime(c["timestamp"]).astimezone(zone)
        key = _key(dt)
        if cur_key is None:
            cur_key = key
            cur_start_ts = c["timestamp"]
            cur_high = c["high"]
            cur_low = c["low"]
        elif key != cur_key:
            # New period: emit prior period's high/low anchored here.
            results.append({
                "type": high_name,
                "level": cur_high,
                "source_period_start": cur_start_ts,
                "index": i,
                "timestamp": c["timestamp"],
            })
            results.append({
                "type": low_name,
                "level": cur_low,
                "source_period_start": cur_start_ts,
                "index": i,
                "timestamp": c["timestamp"],
            })
            cur_key = key
            cur_start_ts = c["timestamp"]
            cur_high = c["high"]
            cur_low = c["low"]
        else:
            cur_high = max(cur_high, c["high"])
            cur_low = min(cur_low, c["low"])

    return results


def detect_liquidity_sweeps(
    candles: list[dict],
    tolerance: float = 0.1,
    lookback: int = 2,
) -> list[dict]:
    """Detect liquidity sweeps (stop runs) with reversal.

    A BSL sweep: a candle's high exceeds a prior swing-high pool level (by any
    amount) and the candle closes back BELOW that level (reversal).
    A SSL sweep: a candle's low breaks a prior swing-low pool level and closes
    back ABOVE it.

    `tolerance` is a fraction of the average candle range; a level is
    considered "exceeded" when price goes strictly beyond it. The reversal
    requirement (close back inside) distinguishes a sweep from a clean BOS.

    Each dict:
        type                : "sweep_bsl" | "sweep_ssl"
        swept_level         : float
        sweep_index         : int (the sweeping candle)
        reversal_confirmed  : bool (True: closed back inside same candle)
        timestamp           : timestamp of the sweeping candle

    Edge cases: clean breakout (no return) -> NOT a sweep; flat market -> [].
    """
    if not candles or tolerance < 0:
        return []
    swings = detect_swings(candles, lookback=lookback)
    if not swings:
        return []

    ref = _avg_range(candles)
    if ref <= 0:
        return []

    highs = [s for s in swings if s["type"] == "swing_high"]
    lows = [s for s in swings if s["type"] == "swing_low"]

    results: list[dict] = []
    swept_high_idx: set[int] = set()
    swept_low_idx: set[int] = set()

    for i, c in enumerate(candles):
        # BSL sweep: high exceeds a prior swing high, close back below it.
        for sh in highs:
            if sh["index"] >= i:
                break  # swings are index-sorted; no prior pool beyond i
            if sh["index"] in swept_high_idx:
                continue
            level = sh["price"]
            if c["high"] > level and c["close"] < level:
                results.append({
                    "type": "sweep_bsl",
                    "swept_level": level,
                    "sweep_index": i,
                    "reversal_confirmed": True,
                    "timestamp": c["timestamp"],
                })
                swept_high_idx.add(sh["index"])
                break

        # SSL sweep: low breaks a prior swing low, close back above it.
        for sl in lows:
            if sl["index"] >= i:
                break
            if sl["index"] in swept_low_idx:
                continue
            level = sl["price"]
            if c["low"] < level and c["close"] > level:
                results.append({
                    "type": "sweep_ssl",
                    "swept_level": level,
                    "sweep_index": i,
                    "reversal_confirmed": True,
                    "timestamp": c["timestamp"],
                })
                swept_low_idx.add(sl["index"])
                break

    return results
