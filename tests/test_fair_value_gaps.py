"""Tests for fair_value_gaps.py — FVG and IFVG.

Index semantics (ATLAS-aligned; see ../atlas/atlas/detectors/fvg.py and
../atlas/tests/test_fvg.py lines 215/229/592-593):
    index       = the MIDDLE / displacement candle  (ATLAS FVG.index)
    start_index = the FIRST  candle
    end_index   = the THIRD  / confirmation candle  (ATLAS detection_index)
A gap is only *confirmed* (usable without lookahead) at the third candle.
"""

import pytest

from detectors.fair_value_gaps import detect_fvg, detect_ifvg
from tests._fixtures import mkseries, mkc


# ── detect_fvg ────────────────────────────────────────────────────────────────

def test_fvg_empty():
    assert detect_fvg([]) == []


def test_fvg_too_few_candles():
    assert detect_fvg(mkseries([(1, 2, 0, 1), (1, 2, 0, 1)])) == []


def test_fvg_bullish_textbook():
    # candle1.high < candle3.low creates a bullish gap.
    # c0: high=10, c1: displacement (middle), c2: low=12 -> gap [10, 12]
    candles = mkseries([
        (8, 10, 7, 9),    # 0  high=10   (first)
        (9, 13, 9, 13),   # 1  displacement up (MIDDLE -> index)
        (13, 15, 12, 14), # 2  low=12    (third -> end_index / confirmation)
    ])
    fvgs = detect_fvg(candles)
    bull = [f for f in fvgs if f["type"] == "fvg_bullish"]
    assert len(bull) == 1
    f = bull[0]
    assert f["bottom"] == 10
    assert f["top"] == 12
    assert f["midpoint"] == 11
    assert f["index"] == 1          # MIDDLE (displacement) candle
    assert f["start_index"] == 0    # FIRST candle
    assert f["end_index"] == 2      # THIRD (confirmation) candle


def test_fvg_bearish_textbook():
    # candle1.low > candle3.high creates a bearish gap.
    # c0: low=15, c1: displacement (middle), c2: high=13 -> gap [13, 15]
    candles = mkseries([
        (16, 18, 15, 17), # 0  low=15    (first)
        (17, 17, 12, 12), # 1  displacement down (MIDDLE -> index)
        (12, 13, 10, 11), # 2  high=13   (third -> end_index / confirmation)
    ])
    fvgs = detect_fvg(candles)
    bear = [f for f in fvgs if f["type"] == "fvg_bearish"]
    assert len(bear) == 1
    f = bear[0]
    assert f["bottom"] == 13
    assert f["top"] == 15
    assert f["midpoint"] == 14
    assert f["index"] == 1          # MIDDLE (displacement) candle
    assert f["start_index"] == 0    # FIRST candle
    assert f["end_index"] == 2      # THIRD (confirmation) candle


def test_fvg_near_miss_touching_not_fvg():
    # candle1.high == candle3.low (touching) -> zero-width, NOT an FVG.
    candles = mkseries([
        (8, 10, 7, 9),    # 0  high=10
        (9, 13, 9, 13),   # 1
        (13, 15, 10, 14), # 2  low=10 == c0.high
    ])
    fvgs = detect_fvg(candles)
    assert fvgs == []


def test_fvg_no_gap_when_overlapping():
    candles = mkseries([
        (8, 12, 7, 9),    # 0  high=12
        (9, 13, 9, 13),   # 1
        (13, 15, 11, 14), # 2  low=11 < c0.high=12 -> overlap, no bullish FVG
    ])
    fvgs = detect_fvg(candles)
    assert [f for f in fvgs if f["type"] == "fvg_bullish"] == []


def test_fvg_multiple_in_series():
    candles = mkseries([
        (8, 10, 7, 9),    # 0
        (9, 13, 9, 13),   # 1  middle of FVG #1
        (13, 15, 12, 14), # 2  bullish FVG [10,12] confirmed
        (14, 16, 13, 15), # 3
        (15, 19, 15, 19), # 4  middle of FVG #2
        (19, 21, 18, 20), # 5  bullish FVG [16,18] confirmed
    ])
    fvgs = detect_fvg(candles)
    bull = [f for f in fvgs if f["type"] == "fvg_bullish"]
    assert len(bull) == 2
    assert bull[0]["index"] == 1     # MIDDLE candle of FVG #1
    assert bull[0]["end_index"] == 2
    assert bull[1]["index"] == 4     # MIDDLE candle of FVG #2
    assert bull[1]["end_index"] == 5


# ── detect_ifvg ───────────────────────────────────────────────────────────────

def test_ifvg_empty():
    assert detect_ifvg([]) == []


def test_ifvg_bullish_fvg_closed_below_becomes_bearish():
    # Bullish FVG [10,12] with MIDDLE candle at index 1 (candles 0,1,2);
    # a later candle closes below 10.
    candles = mkseries([
        (8, 10, 7, 9),    # 0
        (9, 13, 9, 13),   # 1  middle of bullish FVG [10,12]
        (13, 15, 12, 14), # 2  FVG confirmed (end_index)
        (14, 16, 13, 15), # 3
        (15, 16, 9, 9),   # 4  close=9 < 10 -> inversion to bearish
    ])
    ifvgs = detect_ifvg(candles)
    assert len(ifvgs) == 1
    inv = ifvgs[0]
    assert inv["type"] == "ifvg_bearish"
    assert inv["original_type"] == "fvg_bullish"
    assert inv["original_fvg_index"] == 1   # MIDDLE candle of the original FVG
    assert inv["inversion_index"] == 4


def test_ifvg_bearish_fvg_closed_above_becomes_bullish():
    # Bearish FVG [13,15] with MIDDLE candle at index 1 (candles 0,1,2);
    # a later candle closes above 15.
    candles = mkseries([
        (16, 18, 15, 17), # 0
        (17, 17, 12, 12), # 1  middle of bearish FVG [13,15]
        (12, 13, 10, 11), # 2  FVG confirmed (end_index)
        (11, 12, 9, 10),  # 3
        (10, 16, 10, 16), # 4  close=16 > 15 -> inversion to bullish
    ])
    ifvgs = detect_ifvg(candles)
    assert len(ifvgs) == 1
    inv = ifvgs[0]
    assert inv["type"] == "ifvg_bullish"
    assert inv["original_type"] == "fvg_bearish"
    assert inv["inversion_index"] == 4


def test_ifvg_partial_fill_not_inversion():
    # Bullish FVG [10,12]; a candle closes INSIDE the zone (e.g. 11) but not
    # below 10 -> not an inversion.
    candles = mkseries([
        (8, 10, 7, 9),    # 0
        (9, 13, 9, 13),   # 1
        (13, 15, 12, 14), # 2  bullish FVG [10,12]
        (14, 16, 11, 11), # 3  close=11 inside zone, not below 10
    ])
    ifvgs = detect_ifvg(candles)
    assert ifvgs == []


def test_ifvg_never_traded_not_inversion():
    # Bullish FVG [10,12]; no later candle closes below 10.
    candles = mkseries([
        (8, 10, 7, 9),    # 0
        (9, 13, 9, 13),   # 1
        (13, 15, 12, 14), # 2  bullish FVG [10,12]
        (14, 16, 13, 15), # 3  stays above
        (15, 17, 14, 16), # 4  stays above
    ])
    ifvgs = detect_ifvg(candles)
    assert ifvgs == []


def test_ifvg_only_first_close_through_emitted():
    # Bullish FVG; two candles close below. Only the first inversion emitted.
    candles = mkseries([
        (8, 10, 7, 9),    # 0
        (9, 13, 9, 13),   # 1
        (13, 15, 12, 14), # 2  bullish FVG [10,12]
        (14, 16, 9, 9),   # 3  close=9 < 10 -> inversion
        (9, 10, 8, 8),    # 4  also below
    ])
    ifvgs = detect_ifvg(candles)
    assert len(ifvgs) == 1
    assert ifvgs[0]["inversion_index"] == 3
