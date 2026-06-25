"""Tests for order_blocks.py — OB, breaker, mitigation."""

import pytest

from detectors.order_blocks import detect_order_blocks, detect_breaker_blocks
from tests._fixtures import mkseries


def test_ob_empty():
    assert detect_order_blocks([]) == []


def test_ob_bullish_before_bos():
    # Last down-candle before an up-displacement that breaks structure.
    # Build a swing high, then a down candle (the OB), then displacement up
    # through the swing high.
    rows = [
        (10, 12, 9, 11),    # 0
        (11, 16, 10, 15),   # 1
        (15, 20, 14, 19),   # 2  swing high 20
        (19, 19, 13, 14),   # 3
        (14, 14, 10, 11),   # 4
        (11, 11, 9, 10),    # 5
        (10, 10, 8, 9),     # 6
        (9, 11, 8, 10),     # 7
        (10, 10, 9, 9),     # 8  down candle (close<open) -> OB candidate
        (9, 30, 9, 30),     # 9  big up displacement, close=30>20 -> BOS bullish
        (30, 31, 29, 30),   # 10
        (30, 31, 29, 30),   # 11
    ]
    candles = mkseries(rows)
    obs = detect_order_blocks(candles, lookback=2, disp_window=5, disp_k=1.0)
    bull = [o for o in obs if o["type"] == "ob_bullish"]
    assert len(bull) >= 1
    ob = bull[0]
    assert ob["ob_index"] == 8
    assert ob["displacement_index"] == 9
    # zone = full range of candle 8: [9, 10]
    assert ob["bottom"] == 9
    assert ob["top"] == 10


def test_ob_bearish_before_bos():
    rows = [
        (20, 21, 19, 20),    # 0
        (20, 21, 18, 19),    # 1
        (19, 20, 11, 12),    # 2  swing low 11
        (12, 13, 12, 13),    # 3
        (13, 14, 12, 14),    # 4
        (14, 15, 13, 14),    # 5
        (14, 15, 13, 14),    # 6
        (14, 15, 13, 14),    # 7
        (14, 15, 13, 15),    # 8  up candle (close>open) -> OB candidate
        (15, 15, 0, 0),      # 9  big down displacement, close=0<11 -> BOS bearish
        (0, 1, -1, 0),       # 10
        (0, 1, -1, 0),       # 11
    ]
    candles = mkseries(rows)
    obs = detect_order_blocks(candles, lookback=2, disp_window=5, disp_k=1.0)
    bear = [o for o in obs if o["type"] == "ob_bearish"]
    assert len(bear) >= 1
    ob = bear[0]
    assert ob["ob_index"] == 8
    assert ob["displacement_index"] == 9


def test_ob_zone_body_vs_range():
    rows = [
        (10, 12, 9, 11), (11, 16, 10, 15), (15, 20, 14, 19), (19, 19, 13, 14),
        (14, 14, 10, 11), (11, 11, 9, 10), (10, 10, 8, 9), (9, 11, 8, 10),
        (10, 12, 7, 9),    # 8  down candle: body [9,10], range [7,12]
        (9, 30, 9, 30),    # 9  displacement + BOS
        (30, 31, 29, 30), (30, 31, 29, 30),
    ]
    candles = mkseries(rows)
    ob_range = detect_order_blocks(candles, lookback=2, zone="range",
                                   disp_window=5, disp_k=1.0)
    ob_body = detect_order_blocks(candles, lookback=2, zone="body",
                                  disp_window=5, disp_k=1.0)
    bull_r = [o for o in ob_range if o["type"] == "ob_bullish"][0]
    bull_b = [o for o in ob_body if o["type"] == "ob_bullish"][0]
    assert (bull_r["bottom"], bull_r["top"]) == (7, 12)   # range
    assert (bull_b["bottom"], bull_b["top"]) == (9, 10)   # body


def test_ob_mitigation_tagged():
    # OB at index 8; a later candle re-enters the zone [9,10].
    rows = [
        (10, 12, 9, 11), (11, 16, 10, 15), (15, 20, 14, 19), (19, 19, 13, 14),
        (14, 14, 10, 11), (11, 11, 9, 10), (10, 10, 8, 9), (9, 11, 8, 10),
        (10, 10, 9, 9),    # 8  OB zone [9,10]
        (9, 30, 9, 30),    # 9  displacement + BOS
        (30, 31, 29, 30),  # 10
        (30, 31, 9, 30),   # 11  low=9 <= top=10 -> mitigation
    ]
    candles = mkseries(rows)
    obs = detect_order_blocks(candles, lookback=2, disp_window=5, disp_k=1.0)
    bull = [o for o in obs if o["type"] == "ob_bullish"][0]
    assert bull["mitigated"] is True
    assert bull["mitigation_index"] == 11


def test_ob_no_mitigation():
    rows = [
        (10, 12, 9, 11), (11, 16, 10, 15), (15, 20, 14, 19), (19, 19, 13, 14),
        (14, 14, 10, 11), (11, 11, 9, 10), (10, 10, 8, 9), (9, 11, 8, 10),
        (10, 10, 9, 9),    # 8  OB zone [9,10]
        (9, 30, 9, 30),    # 9  displacement + BOS
        (30, 31, 29, 30),  # 10  stays above zone
        (30, 31, 29, 30),  # 11  stays above zone
    ]
    candles = mkseries(rows)
    obs = detect_order_blocks(candles, lookback=2, disp_window=5, disp_k=1.0)
    bull = [o for o in obs if o["type"] == "ob_bullish"][0]
    assert bull["mitigated"] is False
    assert bull["mitigation_index"] is None


def test_ob_require_fvg_filters():
    # require_fvg=True keeps an OB only when an FVG's MIDDLE (displacement)
    # candle coincides with the displacement candle. Here the displacement at
    # index 9 does NOT leave an FVG: the 3-candle window (8,9,10) has
    # c8.high=10 and c10.low=10 (touching, not strict), so no FVG forms with
    # middle=9 -> the OB is filtered out.
    rows = [
        (10, 12, 9, 11), (11, 16, 10, 15), (15, 20, 14, 19), (19, 19, 13, 14),
        (14, 14, 10, 11), (11, 11, 9, 10), (10, 10, 8, 9), (9, 11, 8, 10),
        (10, 10, 9, 9),    # 8  OB
        (9, 30, 9, 30),    # 9  displacement; no FVG (c8.high=10 not < c10.low=10)
        (30, 31, 10, 30),  # 10 low=10 touches c8.high -> no strict FVG at middle 9
        (30, 31, 29, 30),
    ]
    candles = mkseries(rows)
    obs_nofvg = detect_order_blocks(candles, lookback=2, require_fvg=False,
                                    disp_window=5, disp_k=1.0)
    obs_fvg = detect_order_blocks(candles, lookback=2, require_fvg=True,
                                  disp_window=5, disp_k=1.0)
    assert len([o for o in obs_nofvg if o["type"] == "ob_bullish"]) >= 1
    # No FVG with middle==9 -> filtered out.
    assert [o for o in obs_fvg if o["type"] == "ob_bullish"] == []


def test_ob_no_opposing_candle_no_ob():
    # Displacement with no prior opposing candle (all up candles) -> no OB.
    rows = [
        (10, 12, 9, 11), (11, 13, 10, 12), (12, 14, 11, 13),
        (13, 30, 12, 30),  # 3 displacement up, but prior candle 2 is also up
        (30, 31, 29, 30), (30, 31, 29, 30),
    ]
    candles = mkseries(rows)
    obs = detect_order_blocks(candles, lookback=2, disp_window=5, disp_k=1.0)
    # No swing high established for BOS with only 6 candles/lookback 2? Check
    # that no bullish OB is emitted regardless.
    assert [o for o in obs if o["type"] == "ob_bullish"] == []


# ── detect_breaker_blocks ─────────────────────────────────────────────────────

def test_breaker_empty():
    assert detect_breaker_blocks([]) == []


def test_breaker_bullish_ob_violated_becomes_bearish():
    # Bullish OB zone [9,10]; later candle closes below 9 -> breaker_bearish.
    rows = [
        (10, 12, 9, 11), (11, 16, 10, 15), (15, 20, 14, 19), (19, 19, 13, 14),
        (14, 14, 10, 11), (11, 11, 9, 10), (10, 10, 8, 9), (9, 11, 8, 10),
        (10, 10, 9, 9),    # 8  OB zone [9,10]
        (9, 30, 9, 30),    # 9  displacement + BOS
        (30, 31, 29, 30),  # 10
        (30, 31, 8, 8),    # 11  close=8 < 9 -> violation -> breaker_bearish
        (8, 9, 7, 8),      # 12
    ]
    candles = mkseries(rows)
    breakers = detect_breaker_blocks(candles, lookback=2, disp_window=5, disp_k=1.0)
    bear = [b for b in breakers if b["type"] == "breaker_bearish"]
    assert len(bear) >= 1
    assert bear[0]["origin_ob_index"] == 8
    assert bear[0]["break_index"] == 11


def test_breaker_retest_nullable():
    # Violation but no retest -> retest_index None.
    rows = [
        (10, 12, 9, 11), (11, 16, 10, 15), (15, 20, 14, 19), (19, 19, 13, 14),
        (14, 14, 10, 11), (11, 11, 9, 10), (10, 10, 8, 9), (9, 11, 8, 10),
        (10, 10, 9, 9),    # 8  OB zone [9,10]
        (9, 30, 9, 30),    # 9
        (30, 31, 29, 30),  # 10
        (30, 31, 8, 8),    # 11  violation
        (8, 8, 7, 7),      # 12  stays below zone (no retest into [9,10])
        (7, 7, 6, 6),      # 13
    ]
    candles = mkseries(rows)
    breakers = detect_breaker_blocks(candles, lookback=2, disp_window=5, disp_k=1.0)
    bear = [b for b in breakers if b["type"] == "breaker_bearish"]
    assert len(bear) >= 1
    assert bear[0]["retest_index"] is None


def test_breaker_with_retest():
    rows = [
        (10, 12, 9, 11), (11, 16, 10, 15), (15, 20, 14, 19), (19, 19, 13, 14),
        (14, 14, 10, 11), (11, 11, 9, 10), (10, 10, 8, 9), (9, 11, 8, 10),
        (10, 10, 9, 9),    # 8  OB zone [9,10]
        (9, 30, 9, 30),    # 9
        (30, 31, 29, 30),  # 10
        (30, 31, 8, 8),    # 11  violation
        (8, 10, 7, 10),    # 12  high=10 >= bottom=9 -> retest
    ]
    candles = mkseries(rows)
    breakers = detect_breaker_blocks(candles, lookback=2, disp_window=5, disp_k=1.0)
    bear = [b for b in breakers if b["type"] == "breaker_bearish"]
    assert len(bear) >= 1
    assert bear[0]["retest_index"] == 12


def test_breaker_no_lookahead_before_displacement():
    # Regression test for lookahead bias: a candle that closes through the OB
    # zone BEFORE the displacement that confirms the OB must NOT trigger a
    # breaker. The OB only exists at its displacement_index; a close-through
    # between ob_index and displacement_index uses future information.
    #
    # idx2: swing high (high=130)
    # idx4: down candle -> OB candidate, zone [100, 122]
    # idx5: close=90 < 100 -> would-be violation, but OB not yet confirmed
    # idx6: big up displacement, close=132 > 130 -> BOS -> OB confirmed HERE
    rows = [
        (120, 125, 118, 122),  # 0
        (122, 128, 120, 126),  # 1
        (126, 130, 124, 128),  # 2  swing high 130
        (128, 129, 110, 112),  # 3
        (121, 122, 100, 102),  # 4  down candle -> OB candidate, zone [100,122]
        (80, 95, 78, 90),      # 5  close=90 < 100 -> pre-confirmation violation
        (90, 135, 88, 132),    # 6  big up displacement, close=132>130 -> BOS
        (132, 134, 130, 132),  # 7
    ]
    candles = mkseries(rows)
    obs = detect_order_blocks(candles, lookback=2, disp_window=5, disp_k=1.0)
    bull = [o for o in obs if o["type"] == "ob_bullish"]
    assert len(bull) == 1
    assert bull[0]["ob_index"] == 4
    assert bull[0]["displacement_index"] == 6
    # The pre-confirmation close-through at idx5 must NOT produce a breaker.
    breakers = detect_breaker_blocks(candles, lookback=2, disp_window=5, disp_k=1.0)
    assert breakers == []

    # Sanity: adding a post-confirmation violation DOES fire a breaker.
    rows2 = rows + [(132, 134, 80, 80)]  # 8: close=80 < 100 -> post-confirmation
    candles2 = mkseries(rows2)
    breakers2 = detect_breaker_blocks(candles2, lookback=2, disp_window=5, disp_k=1.0)
    assert len(breakers2) == 1
    assert breakers2[0]["break_index"] == 8
    assert breakers2[0]["break_index"] > bull[0]["displacement_index"]
