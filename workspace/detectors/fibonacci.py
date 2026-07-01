"""fibonacci.py — direction-aware Fibonacci grid (Phase 2B).

Pure / stateless. list[dict] in, list[dict] out. No I/O, no broker, no DB.

Given the most-recent confirmed dealing range (a swing-high / swing-low pair),
emits the direction-aware Fibonacci grid:
    equilibrium        = price @ f=0.5 (premium/discount split)
    golden_pocket      = prices @ (0.618, 0.66)
    ote                = grid @ (0.5, 0.62, 0.705, 0.79) with a primary (0.705)
                         and a zone derived from explicit ratios 0.62 and 0.79
    retracement_target = price @ 0.382
    extensions         = prices @ negative ratios (-0.5, -1.0, -1.5, -2.0,
                         -2.5) — expansion / DOL targets beyond the origin
                         extreme.

Direction & level math REUSE premium_discount's convention byte-for-byte:
    direction = "up" if the swing low's index < the swing high's index, else
                "down" (identical to detect_premium_discount).
    R = range_high - range_low
    up   → price(f) = range_high - f * R
    down → price(f) = range_low  + f * R

Methodological note (flagged, verbatim from premium_discount): Which swing
pair defines "the" dealing range is ambiguous; default = most recent confirmed
pair (last swing high + last swing low), parameterizable via lookback.
"""

from __future__ import annotations

from .market_structure import detect_swings

__all__ = ["detect_fibonacci"]


def _ratio_key(f: float) -> str:
    return repr(float(f))


def detect_fibonacci(
    candles: list[dict],
    lookback: int = 2,
    retracements: tuple[float, ...] = (0.382, 0.5, 0.618, 0.66, 0.705, 0.79),
    ote_grid: tuple[float, ...] = (0.5, 0.62, 0.705, 0.79),
    ote_primary: float = 0.705,
    golden_pocket: tuple[float, float] = (0.618, 0.66),
    retracement_target: float = 0.382,
    extensions: tuple[float, ...] = (-0.5, -1.0, -1.5, -2.0, -2.5),
    swings: list[dict] | None = None,
) -> list[dict]:
    if not candles:
        return []

    swings = swings if swings is not None else detect_swings(candles, lookback=lookback)
    highs = [s for s in swings if s["type"] == "swing_high"]
    lows = [s for s in swings if s["type"] == "swing_low"]
    if not highs or not lows:
        return []

    last_high = highs[-1]
    last_low = lows[-1]
    range_high = last_high["price"]
    range_low = last_low["price"]
    if range_high == range_low:
        return []

    R = range_high - range_low
    direction = "up" if last_low["index"] < last_high["index"] else "down"

    def price(f: float) -> float:
        if direction == "up":
            return range_high - f * R
        return range_low + f * R

    equilibrium = price(0.5)

    gp_prices = [price(f) for f in golden_pocket]
    golden_pocket_prices = [min(gp_prices), max(gp_prices)]

    ote: dict[str, float | list] = {}
    for f in ote_grid:
        ote[_ratio_key(f)] = price(f)
    ote["primary"] = price(ote_primary)

    # Repair #3: zone derived from explicit ratios, not positional indexing.
    z0 = price(0.62)
    z1 = price(0.79)
    ote["zone"] = [min(z0, z1), max(z0, z1)]

    retracements_dict = {_ratio_key(f): price(f) for f in retracements}
    retracement_target_price = price(retracement_target)
    extensions_dict = {_ratio_key(f): price(f) for f in extensions}

    confirm_index = max(last_high["index"], last_low["index"])
    current_price = candles[-1]["close"]
    premium_array = current_price > equilibrium

    return [{
        "type": "fibonacci",
        "range_high": range_high,
        "range_low": range_low,
        "direction": direction,
        "equilibrium": equilibrium,
        "golden_pocket": golden_pocket_prices,
        "ote": ote,
        "retracements": retracements_dict,
        "retracement_target": retracement_target_price,
        "extensions": extensions_dict,
        "premium_array": premium_array,
        "index": confirm_index,
        "timestamp": candles[confirm_index]["timestamp"],
    }]
