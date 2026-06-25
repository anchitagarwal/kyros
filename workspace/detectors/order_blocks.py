"""order_blocks.py — OB, breaker, mitigation (Phase 1).

Pure / stateless. list[dict] in, list[dict] out. No I/O, no broker, no DB.

    Bullish OB: the last DOWN candle immediately before an up-displacement
        that breaks structure (BOS bullish). Zone = OB candle's range (default)
        or body (parameterized).
    Bearish OB: the last UP candle immediately before a down-displacement that
        breaks structure (BOS bearish).

    Mitigation: a later candle returns into the OB zone (high/low touches the
        zone). Tagged with mitigated=True and mitigation_index.

    Breaker: an OB whose zone is violated by close, then (optionally) retested
        from the opposite side. Role flips. Default: emit breaker on violation
        with retest_index nullable.

Methodological splits (flagged, not silently resolved):
    - OB zone = full range (default) vs body only. Parameterized `zone`.
    - "Displacement" qualification uses displacement.py's definition (single
      source of truth): body > k * trailing avg range.
    - KB may assert OB requires an FVG present ("ICT OB"). Exposed as optional
      `require_fvg=False` default; not silently imposed.

Lookahead-safety: OB confirmed at the displacement/BOS candle; mitigation
confirmed at the candle that re-enters the zone; breaker confirmed at the
violating close. Critically, a breaker can only fire AFTER the OB is
confirmed — i.e. the violation scan starts at displacement_index + 1, NOT at
ob_index + 1. A candle between ob_index and displacement_index closes through
the zone before the displacement that validates the OB has occurred; acting
on such a "breaker" would use future information (the not-yet-occurred
displacement). See the cross-cutting lookahead invariant #2.
"""

from __future__ import annotations

from typing import Literal

from .displacement import detect_displacement
from .fair_value_gaps import detect_fvg
from .market_structure import detect_bos

__all__ = ["detect_order_blocks", "detect_breaker_blocks"]


def _ob_zone(candle: dict, zone: Literal["range", "body"]) -> tuple[float, float]:
    if zone == "body":
        return min(candle["open"], candle["close"]), max(candle["open"], candle["close"])
    return candle["low"], candle["high"]


def detect_order_blocks(
    candles: list[dict],
    lookback: int = 2,
    zone: Literal["range", "body"] = "range",
    require_fvg: bool = False,
    disp_window: int = 14,
    disp_k: float = 1.5,
) -> list[dict]:
    """Detect bullish/bearish order blocks with mitigation state.

    An OB is the last opposing-direction candle immediately before a
    displacement candle that also produces a BOS in the new direction.

    Each dict:
        type              : "ob_bullish" | "ob_bearish"
        top, bottom       : floats (zone bounds)
        ob_index          : int (the OB candle)
        displacement_index: int (the displacement/BOS candle)
        mitigated         : bool
        mitigation_index  : int | None
        timestamp         : timestamp of the displacement candle

    Edge cases: displacement without a prior opposing candle -> no OB;
    consecutive same-direction candles -> the LAST opposing one is selected.
    """
    if not candles:
        return []
    n = len(candles)
    if n < 3:
        return []

    displacements = detect_displacement(candles, window=disp_window, k=disp_k)
    bos_events = detect_bos(candles, lookback=lookback, confirm="close")
    bos_indices = {b["break_index"] for b in bos_events}
    fvg_indices = {f["index"] for f in detect_fvg(candles)} if require_fvg else None

    results: list[dict] = []
    for d in displacements:
        d_idx = d["index"]
        # Require a BOS at the same candle (displacement + structure break).
        if d_idx not in bos_indices:
            continue
        is_bull = d["type"] == "displacement_bullish"
        # Walk backward from d_idx-1 to find the last opposing-direction candle.
        ob_idx = None
        for j in range(d_idx - 1, -1, -1):
            cj = candles[j]
            c_dir = "bullish" if cj["close"] > cj["open"] else "bearish"
            if is_bull and c_dir == "bearish":
                ob_idx = j
                break
            if not is_bull and c_dir == "bullish":
                ob_idx = j
                break
        if ob_idx is None:
            continue
        if require_fvg and d_idx not in fvg_indices:
            continue

        ob_candle = candles[ob_idx]
        bottom, top = _ob_zone(ob_candle, zone)

        # Mitigation: a later candle (after displacement) re-enters the zone.
        mitigated = False
        mitigation_index = None
        for m in range(d_idx + 1, n):
            cm = candles[m]
            if cm["low"] <= top and cm["high"] >= bottom:
                mitigated = True
                mitigation_index = m
                break

        results.append({
            "type": "ob_bullish" if is_bull else "ob_bearish",
            "top": top,
            "bottom": bottom,
            "ob_index": ob_idx,
            "displacement_index": d_idx,
            "mitigated": mitigated,
            "mitigation_index": mitigation_index,
            "timestamp": candles[d_idx]["timestamp"],
        })

    return results


def detect_breaker_blocks(
    candles: list[dict],
    lookback: int = 2,
    zone: Literal["range", "body"] = "range",
    disp_window: int = 14,
    disp_k: float = 1.5,
) -> list[dict]:
    """Detect breaker blocks — OBs violated by close, role flipped.

    A bullish OB (support zone) whose bottom is closed below becomes a bearish
    breaker. A bearish OB (resistance zone) whose top is closed above becomes a
    bullish breaker. Retest (a later candle returning into the zone from the
    new side) is tracked but nullable.

    Each dict:
        type            : "breaker_bullish" | "breaker_bearish" (NEW role)
        origin_ob_index : int
        break_index     : int (candle that closed through)
        retest_index    : int | None
        timestamp       : timestamp of the break candle

    Edge cases: violated but never retested -> emitted with retest_index=None.

    Lookahead-safety: the violation scan starts at displacement_index + 1, not
    ob_index + 1. The OB only EXISTS (is confirmed) at its displacement_index;
    a close-through before that point would use the not-yet-occurred
    displacement as future information.
    """
    obs = detect_order_blocks(
        candles, lookback=lookback, zone=zone,
        disp_window=disp_window, disp_k=disp_k,
    )
    if not obs:
        return []
    n = len(candles)

    results: list[dict] = []
    for ob in obs:
        ob_idx = ob["ob_index"]
        # The OB is only confirmed at its displacement_index. A breaker (zone
        # violation) can only be recognized AFTER confirmation, so the scan
        # starts at displacement_index + 1 — never at ob_index + 1, which would
        # allow a candle between the OB and its displacement to fire a breaker
        # using future information.
        scan_start = ob["displacement_index"] + 1
        top = ob["top"]
        bottom = ob["bottom"]
        is_bull_ob = ob["type"] == "ob_bullish"

        # Scan candles after OB confirmation for a close-through of the far side.
        for j in range(scan_start, n):
            close_j = candles[j]["close"]
            violated = False
            new_role = None
            if is_bull_ob and close_j < bottom:
                violated = True
                new_role = "breaker_bearish"
            elif not is_bull_ob and close_j > top:
                violated = True
                new_role = "breaker_bullish"
            if violated:
                # Look for a retest: a later candle returning into the zone.
                retest_index = None
                for r in range(j + 1, n):
                    cr = candles[r]
                    if cr["low"] <= top and cr["high"] >= bottom:
                        retest_index = r
                        break
                results.append({
                    "type": new_role,
                    "origin_ob_index": ob_idx,
                    "break_index": j,
                    "retest_index": retest_index,
                    "timestamp": candles[j]["timestamp"],
                })
                break  # first violation per OB

    return results
