"""market_structure.py — swings, BOS, ChoCH, HH/HL/LH/LL (Phase 1).

Pure / stateless. list[dict] in, list[dict] out. No I/O, no broker, no DB.

Methodological splits (flagged, not silently resolved):
    - Swing lookback: default 2 (5-candle fractal). Parameterized.
    - Tie rule: a swing high requires high[i] STRICTLY > all neighbors within
      lookback on both sides. Equal adjacent highs (plateau) do NOT form a
      swing at the equal candle. Documented.
    - BOS confirm: default "close" (close beyond reference swing); alt "wick".
    - "Protected swing" / reference swing: the most recent confirmed swing of
      the relevant type that has not yet been broken.
    - BOS vs ChoCH: BOS is a CONTINUATION break (fires only in the direction
      of the prevailing trend); ChoCH is a REVERSAL break (fires against the
      prevailing trend). Prevailing trend is inferred from swing labels
      (HH/HL = up, LH/LL = down). Before a trend is established, a break in
      either direction is treated as a BOS (no prior trend to reverse). This
      keeps BOS and ChoCH from firing on the same candle/direction.

Lookahead-safety: a swing at index i is only EMITTED after i+lookback candles
exist (full bilateral confirmation). BOS/ChoCH fire at the confirming candle.
The prevailing trend used to classify a break is inferred ONLY from swings
already confirmed strictly before the break candle (index + lookback <=
break_index - 1), so a break's BOS-vs-ChoCH type never depends on a swing
whose confirmation lies beyond the break candle.
"""

from __future__ import annotations

from typing import Literal

__all__ = ["detect_swings", "detect_bos", "detect_choch"]


def detect_swings(candles: list[dict], lookback: int = 2) -> list[dict]:
    """Detect swing highs/lows via fractal pivots.

    A swing high at i: high[i] strictly > high[j] for all j in
    [i-lookback, i+lookback], j != i. Symmetric for swing low (strict <).
    Emitted only once `lookback` future candles confirm it (no lookahead leak).

    Each swing dict:
        type        : "swing_high" | "swing_low"
        price       : high (swing_high) or low (swing_low)
        index       : int (position of the pivot candle)
        timestamp   : timestamp of the pivot candle
        label       : "HH"|"LH" (highs) or "HL"|"LL" (lows), or None for the
                      first of its type (not yet classifiable)

    Edge cases: < 2*lookback+1 candles -> []; flat series -> no swings;
    plateau (equal highs) -> strict > rule, no swing on equal candle.
    """
    if not candles or lookback < 1:
        return []
    n = len(candles)
    if n < 2 * lookback + 1:
        return []

    swings: list[dict] = []
    last_high: float | None = None
    last_low: float | None = None

    for i in range(lookback, n - lookback):
        hi = candles[i]["high"]
        lo = candles[i]["low"]

        is_high = True
        is_low = True
        for j in range(i - lookback, i + lookback + 1):
            if j == i:
                continue
            if not (hi > candles[j]["high"]):
                is_high = False
            if not (lo < candles[j]["low"]):
                is_low = False
            if not is_high and not is_low:
                break

        if is_high:
            label = None
            if last_high is not None:
                label = "HH" if hi > last_high else "LH"
            swings.append({
                "type": "swing_high",
                "price": hi,
                "index": i,
                "timestamp": candles[i]["timestamp"],
                "label": label,
            })
            last_high = hi

        if is_low:
            label = None
            if last_low is not None:
                label = "HL" if lo > last_low else "LL"
            swings.append({
                "type": "swing_low",
                "price": lo,
                "index": i,
                "timestamp": candles[i]["timestamp"],
                "label": label,
            })
            last_low = lo

    return swings


def _reference_swing(swings: list[dict], swing_type: str, before_index: int) -> dict | None:
    """Most recent confirmed swing of `swing_type` with index < before_index."""
    cand = [s for s in swings if s["type"] == swing_type and s["index"] < before_index]
    if not cand:
        return None
    return cand[-1]


def _trend_at(swings: list[dict], idx: int, lookback: int) -> str | None:
    """Infer the prevailing trend from swing labels as of index `idx`.

    Returns "up" if the most recent same-type swing pair is HH (highs) or HL
    (lows); "down" if LH (highs) or LL (lows); None if no labeled pair exists
    yet (trend not established). Requires at least two swings of a type with a
    non-None label on the most recent one.

    Lookahead-safety: only swings already CONFIRMED strictly before the break
    candle `idx` are considered. A swing at index `s` is confirmed only once
    candle `s + lookback` prints, so it is usable for trend inference at `idx`
    only when `s + lookback <= idx - 1` (i.e. `s <= idx - lookback - 1`). This
    guarantees a break's BOS-vs-ChoCH classification never depends on a swing
    whose confirmation lies beyond the break candle.
    """
    # A swing at index s is confirmed at s + lookback; it is "known" at idx
    # only if s + lookback < idx, i.e. s + lookback <= idx - 1.
    confirmed = [s for s in swings if s["index"] + lookback <= idx - 1]
    highs = [s for s in confirmed if s["type"] == "swing_high"]
    lows = [s for s in confirmed if s["type"] == "swing_low"]
    if len(highs) >= 2 and highs[-1]["label"] is not None:
        return "up" if highs[-1]["label"] == "HH" else "down"
    if len(lows) >= 2 and lows[-1]["label"] is not None:
        return "up" if lows[-1]["label"] == "HL" else "down"
    return None


def detect_bos(
    candles: list[dict],
    lookback: int = 2,
    confirm: Literal["close", "wick"] = "close",
) -> list[dict]:
    """Detect Break of Structure (trend continuation).

    A bullish BOS fires when price closes (confirm="close") or trades
    (confirm="wick") above the most recent confirmed swing high, while the
    prevailing structure is up (or no trend is established yet). A bearish
    BOS is symmetric (below most recent swing low, prevailing structure down
    or unestablished). A break AGAINST an established trend is a ChoCH, not a
    BOS, and is therefore excluded here.

    Prevailing structure is inferred from the last two same-type swings:
    up = last swing high is HH (and/or last swing low is HL); down = LH/LL.
    Before a trend is established, breaks in either direction count as BOS.

    Each BOS dict:
        type                : "bos_bullish" | "bos_bearish"
        broken_swing_index  : int
        break_index         : int (confirming candle)
        break_price         : float (close or high/low that broke)
        timestamp           : timestamp of confirming candle

    BOS and ChoCH never fire on the same candle for the same direction.
    """
    if not candles or lookback < 1:
        return []
    n = len(candles)
    if n < 2 * lookback + 1:
        return []

    swings = detect_swings(candles, lookback=lookback)
    results: list[dict] = []

    # Track which swing highs/lows have already been broken to avoid repeats.
    broken_high_idx: set[int] = set()
    broken_low_idx: set[int] = set()

    for i in range(n):
        c = candles[i]
        trend = _trend_at(swings, i, lookback)

        # Bullish BOS: break most recent swing high not yet broken.
        # Allowed when trend is up OR trend is unestablished (None).
        if trend in ("up", None):
            ref_high = _reference_swing(swings, "swing_high", i)
            if ref_high is not None and ref_high["index"] not in broken_high_idx:
                broke = False
                break_price = None
                if confirm == "close":
                    if c["close"] > ref_high["price"]:
                        broke = True
                        break_price = c["close"]
                else:  # wick
                    if c["high"] > ref_high["price"]:
                        broke = True
                        break_price = c["high"]
                if broke:
                    results.append({
                        "type": "bos_bullish",
                        "broken_swing_index": ref_high["index"],
                        "break_index": i,
                        "break_price": break_price,
                        "timestamp": c["timestamp"],
                    })
                    broken_high_idx.add(ref_high["index"])

        # Bearish BOS: break most recent swing low not yet broken.
        # Allowed when trend is down OR trend is unestablished (None).
        if trend in ("down", None):
            ref_low = _reference_swing(swings, "swing_low", i)
            if ref_low is not None and ref_low["index"] not in broken_low_idx:
                broke = False
                break_price = None
                if confirm == "close":
                    if c["close"] < ref_low["price"]:
                        broke = True
                        break_price = c["close"]
                else:
                    if c["low"] < ref_low["price"]:
                        broke = True
                        break_price = c["low"]
                if broke:
                    results.append({
                        "type": "bos_bearish",
                        "broken_swing_index": ref_low["index"],
                        "break_index": i,
                        "break_price": break_price,
                        "timestamp": c["timestamp"],
                    })
                    broken_low_idx.add(ref_low["index"])

    return results


def detect_choch(
    candles: list[dict],
    lookback: int = 2,
    confirm: Literal["close", "wick"] = "close",
) -> list[dict]:
    """Detect Change of Character (trend reversal).

    ChoCH fires when, within an established trend, the most recent protected
    swing on the COUNTER side is broken by close (or wick). I.e. in an uptrend
    (HH/HL sequence), a close below the most recent swing low = bearish ChoCH.
    In a downtrend (LH/LL), a close above the most recent swing high = bullish
    ChoCH.

    "Established trend" requires at least two same-type swings so a label
    exists (avoids spamming ChoCH on alternating chop with no real trend).

    Each ChoCH dict:
        type                : "choch_bullish" | "choch_bearish"
        broken_swing_index  : int
        break_index         : int (confirming candle)
        break_price         : float
        timestamp           : timestamp of confirming candle

    Edge cases: no established trend -> []; BOS/ChoCH never same candle/dir.
    """
    if not candles or lookback < 1:
        return []
    n = len(candles)
    if n < 2 * lookback + 1:
        return []

    swings = detect_swings(candles, lookback=lookback)
    results: list[dict] = []

    broken_high_idx: set[int] = set()
    broken_low_idx: set[int] = set()

    for i in range(n):
        c = candles[i]
        trend = _trend_at(swings, i, lookback)

        # Bearish ChoCH: in an uptrend, break most recent swing low.
        if trend == "up":
            ref_low = _reference_swing(swings, "swing_low", i)
            if ref_low is not None and ref_low["index"] not in broken_low_idx:
                broke = False
                break_price = None
                if confirm == "close":
                    if c["close"] < ref_low["price"]:
                        broke = True
                        break_price = c["close"]
                else:
                    if c["low"] < ref_low["price"]:
                        broke = True
                        break_price = c["low"]
                if broke:
                    results.append({
                        "type": "choch_bearish",
                        "broken_swing_index": ref_low["index"],
                        "break_index": i,
                        "break_price": break_price,
                        "timestamp": c["timestamp"],
                    })
                    broken_low_idx.add(ref_low["index"])

        # Bullish ChoCH: in a downtrend, break most recent swing high.
        elif trend == "down":
            ref_high = _reference_swing(swings, "swing_high", i)
            if ref_high is not None and ref_high["index"] not in broken_high_idx:
                broke = False
                break_price = None
                if confirm == "close":
                    if c["close"] > ref_high["price"]:
                        broke = True
                        break_price = c["close"]
                else:
                    if c["high"] > ref_high["price"]:
                        broke = True
                        break_price = c["high"]
                if broke:
                    results.append({
                        "type": "choch_bullish",
                        "broken_swing_index": ref_high["index"],
                        "break_index": i,
                        "break_price": break_price,
                        "timestamp": c["timestamp"],
                    })
                    broken_high_idx.add(ref_high["index"])

    return results
