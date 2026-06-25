"""inducement.py — IDM (inducement) and turtle soup (Phase 1).

Pure / stateless. list[dict] in, list[dict] out. No I/O, no broker, no DB.

    Turtle soup: price breaks a prior `lookback`-bar extreme (high/low), fails
        to hold, and reverses back through it. Closely related to a liquidity
        sweep; the distinction is that turtle soup is the reversal pattern on a
        prior-PERIOD extreme (a rolling N-bar high/low), whereas a liquidity
        sweep targets a known swing pool. Both are kept; the relationship is
        documented, not merged.

    IDM (inducement): a minor liquidity pool (swing) taken out immediately
        prior to a confirmed BOS/ChoCH. "Immediately prior" = within a bounded
        window before the structural break. IDM is heavily discretionary in
        ICT; the "minor pool before structure" rule is a modeling choice
        (flagged HIGH). Here: a swing high/low that is swept (price exceeds it
        then closes back) within `idm_window` candles before a BOS in the
        opposite direction.

Methodological note: IDM overlaps liquidity sweep and turtle soup. We keep all
three and document the relationship. KB descriptions are treated as
supplementary, not authoritative.
"""

from __future__ import annotations

from .market_structure import detect_bos, detect_swings

__all__ = ["detect_turtle_soup", "detect_inducement"]


def detect_turtle_soup(
    candles: list[dict],
    lookback: int = 20,
    tolerance: float = 0.0,
) -> list[dict]:
    """Detect turtle-soup failed-breakout reversals on rolling extremes.

    For each candle i, compute the prior `lookback`-bar high and low
    (candles [i-lookback, i-1]). A bullish turtle soup fires when candle i's
    low breaks below that prior low (by > tolerance) and candle i closes back
    ABOVE the prior low (reversal). A bearish turtle soup fires when the high
    breaks above the prior high and closes back BELOW it.

    `tolerance` is a fraction of the average range over the lookback window
    (range-relative, not absolute). Default 0.0 (any strict break counts).

    Each dict:
        type          : "turtle_soup_bullish" | "turtle_soup_bearish"
        broken_level  : float (the prior extreme)
        break_index   : int (the reversing candle)
        reversal_index: int (same as break_index; closes back inside)
        timestamp     : timestamp of the reversing candle

    Edge cases: clean breakout continuation (no reversal) -> excluded;
    insufficient lookback history -> skipped.
    """
    if not candles or lookback < 1:
        return []
    n = len(candles)
    if n < 2:
        return []

    results: list[dict] = []
    for i in range(1, n):
        start = max(0, i - lookback)
        window = candles[start:i]
        if len(window) < 2:
            continue
        prior_high = max(c["high"] for c in window)
        prior_low = min(c["low"] for c in window)
        avg_rng = sum(c["high"] - c["low"] for c in window) / len(window)
        band = tolerance * avg_rng if avg_rng > 0 else 0.0

        c = candles[i]
        # Bearish turtle soup: break above prior high, close back below.
        if c["high"] > prior_high + band and c["close"] < prior_high:
            results.append({
                "type": "turtle_soup_bearish",
                "broken_level": prior_high,
                "break_index": i,
                "reversal_index": i,
                "timestamp": c["timestamp"],
            })
        # Bullish turtle soup: break below prior low, close back above.
        if c["low"] < prior_low - band and c["close"] > prior_low:
            results.append({
                "type": "turtle_soup_bullish",
                "broken_level": prior_low,
                "break_index": i,
                "reversal_index": i,
                "timestamp": c["timestamp"],
            })

    return results


def detect_inducement(
    candles: list[dict],
    lookback: int = 2,
    idm_window: int = 5,
) -> list[dict]:
    """Detect inducement (IDM) — a minor pool swept just before a BOS.

    A swing high/low that is swept (price exceeds it then closes back inside)
    within `idm_window` candles before a confirmed BOS in the opposite
    direction. The swept swing is the "induced" liquidity; the BOS is the
    "real" move.

    Each dict:
        type                    : "idm"
        induced_level           : float (the swept swing price)
        induced_index           : int (the swept swing's index)
        related_structure_index : int (the BOS break_index)
        timestamp               : timestamp of the BOS candle

    Edge cases: structural break with no prior minor pool swept -> no IDM;
    no BOS -> [].
    """
    if not candles:
        return []
    swings = detect_swings(candles, lookback=lookback)
    bos_events = detect_bos(candles, lookback=lookback, confirm="close")
    if not bos_events or not swings:
        return []

    results: list[dict] = []
    for bos in bos_events:
        bos_idx = bos["break_index"]
        is_bull_bos = bos["type"] == "bos_bullish"
        # For a bullish BOS, the induced pool is a swing HIGH swept just before
        # (buy-side liquidity taken before the real move up). For a bearish BOS,
        # the induced pool is a swing LOW swept just before.
        pool_type = "swing_high" if is_bull_bos else "swing_low"
        pools = [s for s in swings if s["type"] == pool_type and s["index"] < bos_idx]

        for pool in pools:
            # Look for a sweep of this pool within [bos_idx - idm_window, bos_idx-1].
            lo = max(0, bos_idx - idm_window)
            for k in range(lo, bos_idx):
                ck = candles[k]
                level = pool["price"]
                if is_bull_bos:
                    # BSL sweep: high exceeds level, close back below.
                    if ck["high"] > level and ck["close"] < level:
                        results.append({
                            "type": "idm",
                            "induced_level": level,
                            "induced_index": pool["index"],
                            "related_structure_index": bos_idx,
                            "timestamp": candles[bos_idx]["timestamp"],
                        })
                        break
                else:
                    # SSL sweep: low breaks level, close back above.
                    if ck["low"] < level and ck["close"] > level:
                        results.append({
                            "type": "idm",
                            "induced_level": level,
                            "induced_index": pool["index"],
                            "related_structure_index": bos_idx,
                            "timestamp": candles[bos_idx]["timestamp"],
                        })
                        break

    return results
