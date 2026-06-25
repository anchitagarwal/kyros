"""fair_value_gaps.py — FVG and Inverse FVG (Phase 1).

Pure / stateless. list[dict] in, list[dict] out. No I/O, no broker, no DB.

Definitions (first-principles ICT, cross-checked against the legacy ATLAS
read-only reference ``atlas/detectors/fvg.py``):

    A Fair Value Gap is a 3-candle imbalance. For three consecutive candles
    at positions ``first`` (i-2), ``middle`` (i-1, the displacement candle),
    and ``third`` (i, the confirmation candle):

    Bullish FVG: ``first.high < third.low`` (strict). The gap between
        ``first.high`` and ``third.low`` is unfilled; ``middle`` is the
        displacement candle. ``bottom = first.high``, ``top = third.low``.
    Bearish FVG: ``first.low > third.high`` (strict). ``bottom = third.high``,
        ``top = first.low``.

    In both cases ``top`` is the upper price boundary and ``bottom`` the
    lower, so ``top > bottom`` and ``size = top - bottom > 0`` for every
    reported gap. Direction is carried by the ``type`` token.

    Consequent encroachment = 50% of the gap (midpoint). Standard; flagged
    if a knowledge-base source redefines it (none found that does).

Index semantics (ATLAS-aligned — see contract split #14):
    ``index``       — the MIDDLE (displacement) candle. This is the candle
                      that "leaves" the gap. (ATLAS ``FVG.index``.)
    ``start_index`` — the FIRST candle of the 3-candle pattern.
    ``end_index``   — the THIRD (confirmation) candle — the first candle at
                      which the gap is detectable without lookahead. (ATLAS
                      ``FVG.detection_index``.)
    ``timestamp``   — the MIDDLE candle's timestamp (ATLAS convention).

    The gap is *located at* the middle candle but is only *confirmed*
    (usable without lookahead) once the third candle closes. Both labels are
    reported so callers cannot accidentally treat the middle candle as the
    detection timestamp.

IFVG (Inverse FVG): a previously formed FVG whose zone is closed through on
    the FAR side, inverting polarity. A bullish FVG (support zone) closed
    BELOW its bottom becomes a bearish IFVG (resistance). A bearish FVG
    closed ABOVE its top becomes a bullish IFVG. Requires a CLOSE through;
    partial fill (close inside the zone) is NOT inversion. The close-through
    scan starts at ``end_index + 1`` (after confirmation) — matching ATLAS
    ``check_inversion`` (``range(detection_index + 1, n)``) — so the
    confirmation candle itself is never treated as an inversion.

Lookahead-safety: FVG confirmed at the third candle (``end_index``); IFVG
confirmed at the candle that closes through the zone (uses only data up to
that candle).
"""

from __future__ import annotations

import math
from typing import Any, Literal

__all__ = ["detect_fvg", "detect_ifvg"]


# OHLC fields read for the FVG boundary comparison. ``open``/``close`` are
# not used by detect_fvg itself but are required candle keys per problem.md
# (candle dicts have open, high, low, close, volume, timestamp); validating
# them here keeps detect_fvg self-contained and gives a clear ValueError
# instead of a downstream KeyError.
_REQUIRED_FIELDS = ("open", "high", "low", "close")


def _is_finite_number(value: Any) -> bool:
    """Return True iff ``value`` is a finite number (int/float, not NaN/inf)."""
    if isinstance(value, bool):
        # bool is a subclass of int; reject it as a non-numeric boundary.
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _validate_candles(candles: Any) -> list[dict]:
    """Validate the ``candles`` argument and return it as a list of dicts.

    Raises ``ValueError`` with a clear message for: non-list input,
    non-dict elements, missing required OHLC keys, or non-finite OHLC
    values. (NaN/inf in a boundary are rejected up-front rather than
    silently producing a confusing comparison result.)
    """
    if candles is None:
        raise ValueError("candles must be a list of dicts, got None")
    if not isinstance(candles, list):
        raise ValueError(
            f"candles must be a list of dicts, got {type(candles).__name__}"
        )
    for idx, c in enumerate(candles):
        if not isinstance(c, dict):
            raise ValueError(
                f"candles[{idx}] must be a dict, got {type(c).__name__}"
            )
        for field in _REQUIRED_FIELDS:
            if field not in c:
                raise ValueError(
                    f"candles[{idx}] missing required field {field!r}"
                )
            if not _is_finite_number(c[field]):
                raise ValueError(
                    f"candles[{idx}].{field} must be a finite number, "
                    f"got {c[field]!r}"
                )
    return candles


def detect_fvg(candles: list[dict]) -> list[dict]:
    """Detect Fair Value Gaps (3-candle imbalances).

    Parameters
    ----------
    candles : list[dict]
        OHLCV candle dicts with keys ``open, high, low, close, volume,
        timestamp`` (extra keys ignored). Required OHLC fields must be
        finite numbers.

    Returns
    -------
    list[dict]
        One dict per detected FVG, in chronological order (by confirmation
        candle). Each dict:

            type        : "fvg_bullish" | "fvg_bearish"
            top         : float (upper bound of the gap; >= bottom)
            bottom      : float (lower bound of the gap; <= top)
            midpoint    : float (consequent encroachment = 50%)
            size        : float (top - bottom; > 0)
            index       : int (MIDDLE / displacement candle)
            start_index : int (FIRST candle)
            end_index   : int (THIRD / confirmation candle)
            timestamp   : timestamp of the MIDDLE candle

        Bullish: ``first.high < third.low`` -> ``bottom=first.high``,
        ``top=third.low``.
        Bearish: ``first.low > third.high`` -> ``bottom=third.high``,
        ``top=first.low``.

    Raises
    ------
    ValueError
        If ``candles`` is not a list of dicts, or any candle is missing a
        required OHLC field, or any OHLC value is non-finite (NaN/inf).

    Notes
    -----
    Edge cases: empty / <3 candles -> []. A zero-width gap (touching, ``==``)
    is NOT an FVG (strict inequality required, so ``size > 0`` always).
    """
    candles = _validate_candles(candles)
    n = len(candles)
    if n < 3:
        return []

    results: list[dict] = []
    # ``i`` is the THIRD (confirmation) candle; first = i-2, middle = i-1.
    for i in range(2, n):
        c1 = candles[i - 2]  # first
        c2 = candles[i - 1]  # middle (displacement)
        c3 = candles[i]      # third (confirmation)

        # Bullish FVG: first.high < third.low
        if c1["high"] < c3["low"]:
            bottom = c1["high"]
            top = c3["low"]
            results.append({
                "type": "fvg_bullish",
                "top": top,
                "bottom": bottom,
                "midpoint": (top + bottom) / 2.0,
                "size": top - bottom,
                "index": i - 1,        # middle (displacement) candle
                "start_index": i - 2,  # first candle
                "end_index": i,        # third (confirmation) candle
                "timestamp": c2["timestamp"],
            })

        # Bearish FVG: first.low > third.high
        if c1["low"] > c3["high"]:
            bottom = c3["high"]
            top = c1["low"]
            results.append({
                "type": "fvg_bearish",
                "top": top,
                "bottom": bottom,
                "midpoint": (top + bottom) / 2.0,
                "size": top - bottom,
                "index": i - 1,        # middle (displacement) candle
                "start_index": i - 2,  # first candle
                "end_index": i,        # third (confirmation) candle
                "timestamp": c2["timestamp"],
            })

    return results


def detect_ifvg(
    candles: list[dict],
    confirm: Literal["close"] = "close",
) -> list[dict]:
    """Detect Inverse FVGs — FVGs closed through on the far side.

    A bullish FVG (zone ``[bottom, top]``) is inverted when a later candle
    CLOSES below its bottom -> becomes an IFVG with new polarity "bearish".
    A bearish FVG is inverted when a later candle closes above its top ->
    becomes an IFVG with new polarity "bullish".

    The close-through scan starts at ``end_index + 1`` (the candle AFTER the
    FVG's confirmation candle), matching the legacy ATLAS ``check_inversion``
    (``range(detection_index + 1, n)``). The confirmation candle itself is
    never treated as an inversion.

    Parameters
    ----------
    candles : list[dict]
        OHLCV candle dicts (same requirements as :func:`detect_fvg`).
    confirm : {"close"}
        Confirmation mode. Only ``"close"`` is supported (a CLOSE through the
        far side is required; a mere wick through is not inversion).

    Returns
    -------
    list[dict]
        One dict per inverted FVG. Each dict:

            type                : "ifvg_bullish" | "ifvg_bearish" (NEW polarity)
            original_fvg_index  : int (MIDDLE candle of the original FVG)
            original_type       : "fvg_bullish" | "fvg_bearish"
            inversion_index     : int (candle that closed through)
            top, bottom         : floats (original zone bounds)
            timestamp           : timestamp of the inversion candle

    Raises
    ------
    ValueError
        If ``candles`` is invalid (see :func:`detect_fvg`), or ``confirm``
        is not ``"close"``.

    Notes
    -----
    Edge cases: FVG never traded -> not an IFVG; partial fill (close inside
    the zone but not beyond the far side) -> not inversion. Only the FIRST
    close-through per FVG is emitted.
    """
    if confirm != "close":
        raise ValueError(
            f"confirm must be 'close', got {confirm!r}"
        )
    candles = _validate_candles(candles)
    n = len(candles)
    if n < 3:
        return []

    fvgs = detect_fvg(candles)
    results: list[dict] = []

    for fvg in fvgs:
        # ``index`` is the middle candle; the FVG is only actionable after
        # its confirmation candle (``end_index``) closes, so the inversion
        # scan starts at end_index + 1 — never at index + 1 (which would
        # include the confirmation candle itself).
        end_index = fvg["end_index"]
        top = fvg["top"]
        bottom = fvg["bottom"]
        original_type = fvg["type"]
        original_fvg_index = fvg["index"]

        for j in range(end_index + 1, n):
            close_j = candles[j]["close"]
            inverted = False
            new_polarity = None

            if original_type == "fvg_bullish":
                # Bullish FVG acts as support; close below bottom inverts to bearish.
                if close_j < bottom:
                    inverted = True
                    new_polarity = "ifvg_bearish"
            else:  # fvg_bearish
                # Bearish FVG acts as resistance; close above top inverts to bullish.
                if close_j > top:
                    inverted = True
                    new_polarity = "ifvg_bullish"

            if inverted:
                results.append({
                    "type": new_polarity,
                    "original_fvg_index": original_fvg_index,
                    "original_type": original_type,
                    "inversion_index": j,
                    "top": top,
                    "bottom": bottom,
                    "timestamp": candles[j]["timestamp"],
                })
                break  # only first close-through per FVG

    return results
