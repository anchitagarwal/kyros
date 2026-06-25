"""premium_discount.py — equilibrium, premium/discount, OTE (Phase 1).

Pure / stateless. list[dict] in, list[dict] out. No I/O, no broker, no DB.

Given a dealing range (a confirmed swing-high / swing-low pair), computes:
    equilibrium    = 50% of the range
    premium zone   = upper half [equilibrium, range_high]
    discount zone  = lower half [range_low, equilibrium]
    OTE band       = 0.62-0.79 retracement of the range (default; parameterized)

Direction: "up" if the swing low precedes the swing high (low is the origin,
high is the target); "down" if the high precedes the low. The 0% / 100%
anchors depend on direction:
    up   : 0% = range_low (origin), 100% = range_high (target). OTE is the
           retracement FROM the high back toward the low, i.e. levels at
           range_high - 0.62*R .. range_high - 0.79*R (the discount retracement).
    down : 0% = range_high (origin), 100% = range_low (target). OTE retraces
           from the low back toward the high: range_low + 0.62*R ..
           range_low + 0.79*R (the premium retracement).

Methodological note (flagged): OTE exact band varies in ICT literature
(0.62-0.79 vs a 0.705 sweet spot). Default 0.62-0.79, parameterized. Which
swing pair defines "the" dealing range is ambiguous; default = most recent
confirmed pair (last swing high + last swing low), parameterizable via
lookback.
"""

from __future__ import annotations

from .market_structure import detect_swings

__all__ = ["detect_premium_discount"]


def detect_premium_discount(
    candles: list[dict],
    lookback: int = 2,
    ote_lower: float = 0.62,
    ote_upper: float = 0.79,
) -> list[dict]:
    """Compute equilibrium / premium / discount / OTE for the most recent range.

    Uses the most recent confirmed swing high and swing low (from
    detect_swings). The range is [range_low, range_high]; direction is "up" if
    the low's index < the high's index, else "down".

    Each dict:
        type          : "dealing_range"
        range_high    : float
        range_low     : float
        equilibrium   : float (50%)
        premium_zone  : [low, high]
        discount_zone : [low, high]
        ote_zone      : [low, high]
        direction     : "up" | "down"
        index         : int (index of the later of the two swings = confirmation)
        timestamp     : timestamp of the confirming candle

    Edge cases: no valid swing pair -> []; degenerate range (high==low) -> [].
    """
    if not candles:
        return []
    swings = detect_swings(candles, lookback=lookback)
    highs = [s for s in swings if s["type"] == "swing_high"]
    lows = [s for s in swings if s["type"] == "swing_low"]
    if not highs or not lows:
        return []

    last_high = highs[-1]
    last_low = lows[-1]
    range_high = last_high["price"]
    range_low = last_low["price"]
    if range_high == range_low:
        return []  # degenerate

    R = range_high - range_low
    equilibrium = range_low + R / 2.0

    # Direction: up if low precedes high, down if high precedes low.
    direction = "up" if last_low["index"] < last_high["index"] else "down"

    if direction == "up":
        # Retracement from the high (target) back toward the low (origin).
        ote_high = range_high - ote_lower * R
        ote_low = range_high - ote_upper * R
    else:
        # Retracement from the low (target) back toward the high (origin).
        ote_low = range_low + ote_lower * R
        ote_high = range_low + ote_upper * R

    confirm_index = max(last_high["index"], last_low["index"])

    return [{
        "type": "dealing_range",
        "range_high": range_high,
        "range_low": range_low,
        "equilibrium": equilibrium,
        "premium_zone": [equilibrium, range_high],
        "discount_zone": [range_low, equilibrium],
        "ote_zone": [ote_low, ote_high],
        "direction": direction,
        "index": confirm_index,
        "timestamp": candles[confirm_index]["timestamp"],
    }]
