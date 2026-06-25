"""displacement.py — displacement detection (Phase 1).

Pure / stateless. list[dict] in, list[dict] out. No I/O, no broker, no DB.

Displacement = a strong momentum candle whose body is large relative to
recent volatility. ICT "displacement" is qualitative; any numeric threshold
is a modeling choice. Here:

    strength = body[i] / ATR(window)
    where ATR(window) = mean of (high - low) over the trailing `window`
    candles [i-window, i-1] (simplified ATR using pure range; documented).

A candle is displacement if strength > k (default 1.5). Direction from
close vs open. `leaves_fvg` cross-checked against FVG presence at i.

Warmup behavior: for i < window, the trailing average uses whatever history
is available (at least 1 candle); i == 0 is skipped (no trailing history).
If the trailing ATR is 0 (flat market), no displacement is emitted (avoid
division by zero; a flat market has no displacement by definition).

Parameterized: window (default 14), k (default 1.5). Defaults stated, not
authoritative. Knowledge-base profitability claims were NOT used to set k.
"""

from __future__ import annotations

from .fair_value_gaps import detect_fvg

__all__ = ["detect_displacement"]


def detect_displacement(
    candles: list[dict],
    window: int = 14,
    k: float = 1.5,
) -> list[dict]:
    """Detect displacement candles (body > k * trailing average range).

    Each displacement dict:
        type        : "displacement_bullish" | "displacement_bearish"
        index       : int
        timestamp   : timestamp of the candle
        strength    : float (body / trailing_avg_range)
        leaves_fvg  : bool (True if an FVG is confirmed at this candle)

    Edge cases: empty -> []; flat market (ATR 0) -> []; i=0 skipped (warmup).
    """
    if not candles or window < 1:
        return []
    n = len(candles)
    if n < 2:
        return []

    # Precompute FVG confirmation indices for the leaves_fvg cross-check.
    fvg_indices = {f["index"] for f in detect_fvg(candles)}

    results: list[dict] = []
    for i in range(1, n):
        c = candles[i]
        body = abs(c["close"] - c["open"])

        # Trailing window [max(0, i-window), i-1].
        start = max(0, i - window)
        trailing = candles[start:i]
        if not trailing:
            continue
        ranges = [t["high"] - t["low"] for t in trailing]
        avg_range = sum(ranges) / len(ranges)
        if avg_range <= 0:
            continue  # flat market: no displacement

        strength = body / avg_range
        if strength <= k:
            continue

        direction = "bullish" if c["close"] > c["open"] else "bearish"
        results.append({
            "type": f"displacement_{direction}",
            "index": i,
            "timestamp": c["timestamp"],
            "strength": strength,
            "leaves_fvg": i in fvg_indices,
        })

    return results
