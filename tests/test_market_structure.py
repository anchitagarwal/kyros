"""Tests for market_structure.py — swings, BOS, ChoCH."""

import pytest

from detectors.market_structure import detect_swings, detect_bos, detect_choch
from tests._fixtures import mkseries, mkc


# ── detect_swings ─────────────────────────────────────────────────────────────

def test_swings_empty():
    assert detect_swings([]) == []


def test_swings_insufficient_length():
    # need >= 2*lookback+1 = 5 candles for lookback=2
    assert detect_swings(mkseries([(1, 2, 0, 1), (1, 2, 0, 1), (1, 2, 0, 1)])) == []


def test_swings_monotonic_up_leg():
    # Strictly increasing highs and lows -> no interior candle is a swing high
    # (right neighbor higher) nor swing low (left neighbor lower). The ends
    # lack bilateral confirmation. So no swings in a pure monotonic rise.
    rows = [(i, i + 1, i - 1, i) for i in range(10)]
    swings = detect_swings(mkseries(rows))
    assert swings == []


def test_swings_clear_peak_and_trough():
    # Down to a trough at index 3, up to a peak at index 7. All OHLC valid and
    # all swing extremes strictly distinct from their lookback neighbors.
    rows = [
        (10, 11, 9, 9),    # 0
        (9, 10, 8, 8),     # 1
        (8, 9, 7, 7),      # 2
        (7, 8, 6, 7),      # 3 trough (low=6)
        (8, 9, 7, 9),      # 4
        (9, 11, 8, 11),    # 5
        (11, 13, 10, 13),  # 6
        (13, 15, 12, 14),  # 7 peak (high=15)
        (14, 14, 13, 13),  # 8
        (13, 14, 12, 12),  # 9
    ]
    swings = detect_swings(mkseries(rows))
    highs = [s for s in swings if s["type"] == "swing_high"]
    lows = [s for s in swings if s["type"] == "swing_low"]
    assert any(s["index"] == 7 and s["price"] == 15 for s in highs)
    assert any(s["index"] == 3 and s["price"] == 6 for s in lows)


def test_swing_high_strict_greater_tie_rule():
    # Plateau: two equal highs at index 3 and 4. Neither should be a swing high
    # because strict > fails against the equal neighbor.
    rows = [
        (10, 12, 9, 10),    # 0
        (10, 12, 9, 10),    # 1
        (10, 12, 9, 10),    # 2
        (10, 14, 9, 14),    # 3 high=14
        (10, 14, 9, 14),    # 4 high=14 (equal to 3)
        (10, 12, 9, 10),    # 5
        (10, 12, 9, 10),    # 6
    ]
    swings = detect_swings(mkseries(rows))
    highs = [s for s in swings if s["type"] == "swing_high"]
    # Neither 3 nor 4 is strictly greater than the other -> no swing high.
    assert highs == []


def test_swing_label_hh_hl():
    # Two peaks with the second higher -> HH. All OHLC valid, extremes distinct.
    rows = [
        (10, 12, 9, 10),    # 0
        (10, 11, 8, 9),     # 1
        (9, 11, 8, 11),     # 2
        (11, 13, 10, 12),   # 3 peak1 high=13
        (12, 12, 11, 11),   # 4
        (11, 12, 9, 10),    # 5
        (10, 11, 7, 8),     # 6 trough low=7
        (8, 10, 8, 10),     # 7
        (10, 12, 9, 12),    # 8
        (12, 15, 11, 14),   # 9 peak2 high=15 (HH)
        (14, 14, 13, 13),   # 10
        (13, 14, 12, 12),   # 11
    ]
    swings = detect_swings(mkseries(rows))
    highs = [s for s in swings if s["type"] == "swing_high"]
    # First high has label None, second (higher) is HH.
    assert highs[0]["label"] is None
    assert highs[1]["label"] == "HH"


def test_lookahead_swing_only_after_confirmation():
    # A swing at index i needs i+lookback to exist. With lookback=2, a peak at
    # index 2 in a 6-candle series (indices 0..5) is confirmed at index 4.
    rows = [
        (10, 11, 9, 10),    # 0
        (10, 11, 9, 10),    # 1
        (10, 14, 9, 14),    # 2 peak high=14
        (13, 13, 12, 12),   # 3
        (12, 13, 11, 12),   # 4
        (11, 12, 10, 11),   # 5
    ]
    swings = detect_swings(mkseries(rows), lookback=2)
    highs = [s for s in swings if s["type"] == "swing_high"]
    assert len(highs) == 1
    assert highs[0]["index"] == 2


def test_flat_series_no_swings():
    rows = [(10, 10, 10, 10)] * 7
    assert detect_swings(mkseries(rows)) == []


# ── detect_bos ────────────────────────────────────────────────────────────────

def test_bos_empty():
    assert detect_bos([]) == []


def test_bos_bullish_on_close_break():
    # Build an up-trend: trough, peak, then a close above the peak.
    rows = [
        (10, 11, 9, 10),    # 0
        (10, 11, 8, 9),     # 1
        (9, 11, 8, 11),     # 2
        (11, 13, 10, 12),   # 3 peak high=13
        (12, 12, 11, 11),   # 4
        (11, 12, 10, 10),   # 5
        (10, 12, 9, 11),    # 6
        (11, 14, 10, 14),   # 7 close=14 > 13 -> BOS bullish
        (14, 15, 13, 14),   # 8
        (14, 15, 13, 14),   # 9
    ]
    candles = mkseries(rows)
    bos = detect_bos(candles, lookback=2, confirm="close")
    bull = [b for b in bos if b["type"] == "bos_bullish"]
    assert len(bull) >= 1
    assert bull[0]["break_index"] == 7
    assert bull[0]["break_price"] == 14


def test_bos_wick_vs_close_difference():
    # A candle that wicks above the swing high but closes below it.
    rows = [
        (10, 11, 9, 10),    # 0
        (10, 11, 8, 9),     # 1
        (9, 11, 8, 11),     # 2
        (11, 13, 10, 12),   # 3 peak high=13
        (12, 12, 11, 11),   # 4
        (11, 12, 10, 10),   # 5
        (10, 12, 9, 11),    # 6
        (11, 14, 10, 12),   # 7 high=14>13 but close=12<13 -> wick only
        (12, 13, 11, 12),   # 8
        (12, 13, 11, 12),   # 9
    ]
    candles = mkseries(rows)
    # close-confirm: no BOS at 7 (close 12 < 13)
    bos_close = detect_bos(candles, lookback=2, confirm="close")
    assert not any(b["break_index"] == 7 for b in bos_close if b["type"] == "bos_bullish")
    # wick-confirm: BOS at 7 (high 14 > 13)
    bos_wick = detect_bos(candles, lookback=2, confirm="wick")
    assert any(b["break_index"] == 7 for b in bos_wick if b["type"] == "bos_bullish")


def test_bos_bearish_on_close_break():
    rows = [
        (14, 15, 13, 14),   # 0
        (14, 15, 13, 14),   # 1
        (14, 15, 12, 13),   # 2
        (13, 14, 11, 12),   # 3 trough low=11
        (12, 13, 12, 13),   # 4
        (13, 14, 12, 14),   # 5
        (14, 15, 13, 14),   # 6
        (14, 15, 9, 10),    # 7 close=10 < 11 -> BOS bearish
        (10, 11, 9, 10),    # 8
        (10, 11, 9, 10),    # 9
    ]
    candles = mkseries(rows)
    bos = detect_bos(candles, lookback=2, confirm="close")
    bear = [b for b in bos if b["type"] == "bos_bearish"]
    assert len(bear) >= 1
    assert bear[0]["break_index"] == 7


# ── detect_choch ──────────────────────────────────────────────────────────────

def test_choch_empty():
    assert detect_choch([]) == []


def test_choch_no_established_trend():
    # Only one swing high/low -> no label -> no trend -> no ChoCH.
    rows = [
        (10, 11, 9, 10),    # 0
        (10, 11, 8, 9),     # 1
        (9, 11, 8, 11),     # 2
        (11, 13, 10, 12),   # 3 peak
        (12, 12, 11, 11),   # 4
        (11, 12, 10, 10),   # 5
        (10, 12, 9, 11),    # 6
    ]
    candles = mkseries(rows)
    assert detect_choch(candles, lookback=2) == []


def test_choch_bearish_after_uptrend():
    # Establish uptrend (HH, HL), then break below the last swing low.
    rows = [
        (10, 11, 9, 10),    # 0
        (10, 11, 8, 9),     # 1
        (9, 11, 8, 11),     # 2
        (11, 13, 10, 12),   # 3 peak1 high=13
        (12, 12, 11, 11),   # 4
        (11, 12, 9, 10),    # 5
        (10, 11, 7, 8),     # 6 trough1 low=7
        (8, 10, 8, 10),     # 7
        (10, 12, 9, 12),    # 8
        (12, 15, 11, 14),   # 9 peak2 high=15 (HH)
        (14, 14, 13, 13),   # 10
        (13, 14, 12, 12),   # 11
        (12, 13, 11, 11),   # 12
        (11, 12, 10, 10),   # 13
        (10, 11, 5, 6),     # 14 close=6 < 7 (last swing low) -> ChoCH bearish
        (6, 7, 5, 6),       # 15
        (6, 7, 5, 6),       # 16
    ]
    candles = mkseries(rows)
    choch = detect_choch(candles, lookback=2, confirm="close")
    bear = [c for c in choch if c["type"] == "choch_bearish"]
    assert len(bear) >= 1
    assert bear[0]["break_index"] == 14


def test_choch_bullish_after_downtrend():
    # Establish downtrend (LH, LL), then break above the last swing high.
    rows = [
        (18, 19, 17, 18),   # 0
        (18, 19, 17, 18),   # 1
        (18, 19, 17, 18),   # 2
        (18, 20, 17, 19),   # 3 peak1 high=20
        (19, 19, 18, 18),   # 4
        (18, 19, 16, 17),   # 5
        (17, 18, 15, 16),   # 6 trough1 low=15
        (16, 17, 16, 17),   # 7
        (17, 17, 16, 16),   # 8
        (17, 18, 16, 17),   # 9
        (17, 17, 14, 15),   # 10
        (15, 16, 13, 14),   # 11
        (14, 15, 12, 13),   # 12 trough2 low=12 (LL)
        (13, 14, 13, 14),   # 13
        (14, 15, 13, 15),   # 14
        (15, 16, 14, 16),   # 15
        (16, 19, 15, 19),   # 16 close=19 > 18 (last swing high) -> ChoCH bullish
        (19, 20, 18, 19),   # 17
        (19, 20, 18, 19),   # 18
    ]
    candles = mkseries(rows)
    choch = detect_choch(candles, lookback=2, confirm="close")
    bull = [c for c in choch if c["type"] == "choch_bullish"]
    assert len(bull) >= 1
    assert bull[0]["break_index"] == 16


def test_bos_and_choch_not_same_candle_same_direction():
    # In the ChoCH bearish scenario, ensure no BOS bearish AND ChoCH bearish
    # fire on the same candle index.
    rows = [
        (10, 11, 9, 10), (10, 11, 8, 9), (9, 11, 8, 11), (11, 13, 10, 12),
        (12, 12, 11, 11), (11, 12, 9, 10), (10, 11, 7, 8), (8, 10, 8, 10),
        (10, 12, 9, 12), (12, 15, 11, 14), (14, 14, 13, 13), (13, 14, 12, 12),
        (12, 13, 11, 11), (11, 12, 10, 10), (10, 11, 5, 6), (6, 7, 5, 6),
        (6, 7, 5, 6),
    ]
    candles = mkseries(rows)
    bos = detect_bos(candles, lookback=2, confirm="close")
    choch = detect_choch(candles, lookback=2, confirm="close")
    bos_bear_idx = {b["break_index"] for b in bos if b["type"] == "bos_bearish"}
    choch_bear_idx = {c["break_index"] for c in choch if c["type"] == "choch_bearish"}
    assert not (bos_bear_idx & choch_bear_idx)


# ── lookahead-safety: _trend_at uses only confirmed swings ────────────────────

def test_trend_at_ignores_unconfirmed_swing_label():
    # Regression for the Round-2 HIGH finding: a break's BOS-vs-ChoCH `type`
    # must not depend on a swing whose bilateral confirmation lies beyond the
    # break candle.
    #
    # lookback=2. Swings produced:
    #   swing_high@2  (label None, confirmed @4)
    #   swing_low @6  (label None, confirmed @8)
    #   swing_high@9  (label "HH", confirmed only @11)
    # Break candle idx 10 closes below swing_low@6 (a CONFIRMED swing low), so
    # it is a valid bearish break. idx 10 lies in [9, 9+2): swing_high@9 is NOT
    # yet confirmed at idx 10 (needs idx 11).
    #
    # OLD (buggy) _trend_at read swing_high@9's "HH" label -> trend "up" ->
    # the bearish break was misclassified as choch_bearish (a reversal).
    # FIXED _trend_at excludes swing_high@9 (9+2=11 > 10-1=9) -> trend None ->
    # the bearish break is bos_bearish (continuation in an unestablished trend).
    rows = [
        (12, 15, 11, 13),  # 0
        (13, 16, 12, 14),  # 1
        (14, 20, 13, 19),  # 2  swing_high high=20 (label None)
        (18, 19, 14, 15),  # 3
        (15, 17, 10, 16),  # 4
        (14, 16, 9, 15),   # 5
        (13, 15, 8, 14),   # 6  swing_low low=8 (label None)
        (14, 16, 9, 15),   # 7
        (15, 17, 10, 16),  # 8  <- swing_low@6 confirmed
        (16, 25, 15, 24),  # 9  swing_high high=25 (HH; confirmed only @11)
        (12, 12, 7, 7),    # 10 break: close=7 < 8 (breaks confirmed swing_low@6)
        (8, 10, 6, 9),     # 11 <- swing_high@9 confirmed
        (8, 10, 6, 9),     # 12
    ]
    candles = mkseries(rows)

    # Sanity: the swings are exactly as described.
    swings = detect_swings(candles, lookback=2)
    highs = [s for s in swings if s["type"] == "swing_high"]
    lows = [s for s in swings if s["type"] == "swing_low"]
    assert [(s["index"], s["label"]) for s in highs] == [(2, None), (9, "HH")]
    assert [(s["index"], s["label"]) for s in lows] == [(6, None)]

    bos = detect_bos(candles, lookback=2, confirm="close")
    choch = detect_choch(candles, lookback=2, confirm="close")

    # The bearish break at idx 10 must be a BOS (continuation), NOT a ChoCH,
    # because the only trend-defining swing (swing_high@9) is unconfirmed at
    # idx 10, so the prevailing trend is None.
    bear_bos = [b for b in bos if b["type"] == "bos_bearish" and b["break_index"] == 10]
    bear_choch = [c for c in choch if c["type"] == "choch_bearish" and c["break_index"] == 10]
    assert len(bear_bos) == 1
    assert bear_bos[0]["broken_swing_index"] == 6
    assert bear_choch == []


def test_trend_at_ignores_unconfirmed_swing_label_bullish_mirror():
    # Symmetric (bullish) regression for the Round-2 HIGH finding: the fix must
    # hold in both polarities. A bullish break whose trend would only be
    # established by an UNCONFIRMED same-type swing must be classified BOS
    # (continuation in an unestablished trend), not ChoCH (reversal).
    #
    # lookback=2. Swings produced:
    #   swing_high@2 (label None, confirmed @4)
    #   swing_low @6 (label None, confirmed @8)
    #   swing_low @9 (label "LL", confirmed only @11)
    # Break candle idx 10 closes above swing_high@2 (a CONFIRMED swing high), so
    # it is a valid bullish break. idx 10 lies in [9, 9+2): swing_low@9 is NOT
    # yet confirmed at idx 10 (needs idx 11). idx 10 is itself NOT a swing high
    # (its right neighbor at idx 12 has high=23 >= high@10=22).
    #
    # A buggy _trend_at reading swing_low@9's "LL" label -> trend "down" ->
    # misclassifies the bullish break as choch_bullish (a reversal). The fixed
    # _trend_at excludes swing_low@9 (9+2=11 > 10-1=9) -> trend None -> the
    # bullish break is bos_bullish (continuation in an unestablished trend).
    rows = [
        (12, 13, 11, 12),  # 0
        (12, 14, 11, 13),  # 1
        (13, 20, 12, 19),  # 2  swing_high high=20 (label None)
        (18, 19, 14, 15),  # 3
        (15, 17, 10, 16),  # 4  <- swing_high@2 confirmed
        (14, 16, 9, 15),   # 5
        (13, 15, 8, 14),   # 6  swing_low low=8 (label None)
        (14, 16, 9, 15),   # 7
        (15, 17, 10, 16),  # 8  <- swing_low@6 confirmed
        (9, 10, 5, 6),     # 9  swing_low low=5 (LL; confirmed only @11)
        (21, 22, 20, 21),  # 10 break: close=21 > 20 (breaks confirmed swing_high@2)
        (6, 7, 6, 7),      # 11 <- swing_low@9 confirmed (low=6 > 5)
        (6, 23, 6, 7),     # 12  high=23 >= 22 -> idx 10 NOT a swing high
    ]
    candles = mkseries(rows)

    # Sanity: the swings are exactly as described.
    swings = detect_swings(candles, lookback=2)
    highs = [s for s in swings if s["type"] == "swing_high"]
    lows = [s for s in swings if s["type"] == "swing_low"]
    assert [(s["index"], s["label"]) for s in highs] == [(2, None)]
    assert [(s["index"], s["label"]) for s in lows] == [(6, None), (9, "LL")]

    bos = detect_bos(candles, lookback=2, confirm="close")
    choch = detect_choch(candles, lookback=2, confirm="close")

    # The bullish break at idx 10 must be a BOS (continuation), NOT a ChoCH,
    # because the only trend-defining swing (swing_low@9) is unconfirmed at
    # idx 10, so the prevailing trend is None.
    bull_bos = [b for b in bos if b["type"] == "bos_bullish" and b["break_index"] == 10]
    bull_choch = [c for c in choch if c["type"] == "choch_bullish" and c["break_index"] == 10]
    assert len(bull_bos) == 1
    assert bull_bos[0]["broken_swing_index"] == 2
    assert bull_choch == []
