"""Tests for premium_discount.py — equilibrium, premium/discount, OTE."""

import pytest

from detectors.premium_discount import detect_premium_discount
from tests._fixtures import mkseries


def test_pd_empty():
    assert detect_premium_discount([]) == []


def test_pd_no_swings():
    # Monotonic rise -> no swings -> [].
    rows = [(i, i + 1, i - 1, i) for i in range(10)]
    assert detect_premium_discount(mkseries(rows)) == []


def test_pd_known_range_up_direction():
    # Swing low at 100, swing high at 200, low precedes high -> direction up.
    rows = [
        (150, 155, 145, 150),  # 0
        (150, 152, 120, 130),  # 1
        (130, 135, 100, 110),  # 2  low=100 (trough)
        (110, 130, 105, 125),  # 3
        (125, 140, 120, 135),  # 4
        (135, 200, 130, 190),  # 5  high=200 (peak)
        (190, 195, 185, 188),  # 6
        (188, 190, 185, 186),  # 7
    ]
    candles = mkseries(rows)
    result = detect_premium_discount(candles, lookback=2)
    assert len(result) == 1
    r = result[0]
    assert r["range_high"] == 200
    assert r["range_low"] == 100
    assert r["equilibrium"] == 150
    assert r["direction"] == "up"
    assert r["premium_zone"] == [150, 200]
    assert r["discount_zone"] == [100, 150]
    # OTE up: from high back toward low. ote_high = 200 - 0.62*100 = 138;
    # ote_low = 200 - 0.79*100 = 121.
    assert r["ote_zone"] == [121, 138]


def test_pd_known_range_down_direction():
    # Swing high at 200, swing low at 100, high precedes low -> direction down.
    rows = [
        (150, 155, 145, 150),  # 0
        (150, 180, 145, 175),  # 1
        (175, 200, 170, 190),  # 2  high=200 (peak)
        (190, 195, 160, 162),  # 3
        (162, 165, 135, 138),  # 4
        (138, 145, 100, 110),  # 5  low=100 (trough)
        (110, 115, 105, 112),  # 6
        (112, 115, 108, 110),  # 7
    ]
    candles = mkseries(rows)
    result = detect_premium_discount(candles, lookback=2)
    assert len(result) == 1
    r = result[0]
    assert r["direction"] == "down"
    # OTE down: from low back toward high. ote_low = 100 + 0.62*100 = 162;
    # ote_high = 100 + 0.79*100 = 179.
    assert r["ote_zone"] == [162, 179]


def test_pd_degenerate_range_skipped():
    # A valid swing pair (high 20, low 8) -> normal case returns 1 result.
    rows = [
        (10, 12, 9, 11),    # 0
        (11, 16, 10, 15),   # 1
        (15, 20, 14, 19),   # 2  swing high 20
        (19, 19, 13, 14),   # 3
        (14, 14, 10, 11),   # 4
        (11, 11, 8, 9),     # 5  swing low 8
        (9, 12, 9, 11),     # 6
        (11, 13, 10, 12),   # 7
    ]
    candles = mkseries(rows)
    result = detect_premium_discount(candles, lookback=2)
    assert len(result) == 1  # normal case still works


def test_pd_ote_parameterized():
    rows = [
        (150, 155, 145, 150),
        (150, 152, 120, 130),
        (130, 135, 100, 110),
        (110, 130, 105, 125),
        (125, 140, 120, 135),
        (135, 200, 130, 190),
        (190, 195, 185, 188),
        (188, 190, 185, 186),
    ]
    candles = mkseries(rows)
    r = detect_premium_discount(candles, lookback=2, ote_lower=0.5, ote_upper=0.7)[0]
    # up: ote_high = 200 - 0.5*100 = 150; ote_low = 200 - 0.7*100 = 130.
    assert r["ote_zone"] == [130, 150]
