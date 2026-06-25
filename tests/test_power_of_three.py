"""Tests for power_of_three.py — AMD (Accumulation/Manipulation/Distribution)."""

from zoneinfo import ZoneInfo

import pytest

from detectors.power_of_three import detect_power_of_three
from tests._fixtures import mkc


def _candle_at(year, month, day, hour, minute, o, h, l, c, tz="America/New_York"):
    from datetime import datetime
    zone = ZoneInfo(tz)
    dt = datetime(year, month, day, hour, minute, tzinfo=zone)
    return mkc(o, h, l, c, int(dt.timestamp()))


def test_po3_empty():
    assert detect_power_of_three([], tz="America/New_York") == []


def test_po3_tz_required():
    with pytest.raises(TypeError):
        detect_power_of_three([mkc(10, 11, 9, 10, 1000)])


def test_po3_textbook_amd_day():
    # Accumulation (first 5 candles range ~100-102), manipulation up (sweep
    # above 102 then close back below), distribution down (big bearish candle).
    candles = [
        _candle_at(2024, 3, 4, 9, 30, 100, 102, 99, 101),   # 0 accum
        _candle_at(2024, 3, 4, 9, 45, 101, 102, 100, 101),  # 1 accum
        _candle_at(2024, 3, 4, 10, 0, 101, 102, 100, 101),  # 2 accum
        _candle_at(2024, 3, 4, 10, 15, 101, 102, 100, 101), # 3 accum
        _candle_at(2024, 3, 4, 10, 30, 101, 102, 100, 101), # 4 accum
        # manipulation up: high > 102, close < 102
        _candle_at(2024, 3, 4, 10, 45, 101, 105, 100, 101), # 5 manip up
        # distribution down: big bearish displacement
        _candle_at(2024, 3, 4, 11, 0, 101, 101, 80, 80),    # 6 disp bearish
        _candle_at(2024, 3, 4, 11, 15, 80, 81, 79, 80),     # 7
    ]
    po3 = detect_power_of_three(candles, period="day", tz="America/New_York",
                                accum_bars=5)
    assert len(po3) == 1
    p = po3[0]
    assert p["type"] == "po3"
    assert p["accumulation_zone"] == [99, 102]
    assert p["manipulation_index"] == 5
    assert p["manipulation_direction"] == "up"
    assert p["distribution_direction"] == "down"


def test_po3_no_manipulation_skipped():
    # Trending day with no sweep beyond accumulation -> not PO3.
    candles = [
        _candle_at(2024, 3, 4, 9, 30, 100, 102, 99, 101),
        _candle_at(2024, 3, 4, 9, 45, 101, 102, 100, 101),
        _candle_at(2024, 3, 4, 10, 0, 101, 102, 100, 101),
        _candle_at(2024, 3, 4, 10, 15, 101, 102, 100, 101),
        _candle_at(2024, 3, 4, 10, 30, 101, 102, 100, 101),
        # no manipulation: stays within [99,102]
        _candle_at(2024, 3, 4, 10, 45, 101, 102, 100, 101),
        _candle_at(2024, 3, 4, 11, 0, 101, 102, 100, 101),
    ]
    po3 = detect_power_of_three(candles, period="day", tz="America/New_York",
                                accum_bars=5)
    assert po3 == []


def test_po3_partial_period_skipped():
    # Fewer than accum_bars+1 candles -> skipped.
    candles = [
        _candle_at(2024, 3, 4, 9, 30, 100, 102, 99, 101),
        _candle_at(2024, 3, 4, 9, 45, 101, 102, 100, 101),
    ]
    po3 = detect_power_of_three(candles, period="day", tz="America/New_York",
                                accum_bars=5)
    assert po3 == []


def test_po3_manipulation_down_distribution_up():
    # Accumulation, manipulation down (sweep below then close back above),
    # distribution up (big bullish displacement).
    candles = [
        _candle_at(2024, 3, 4, 9, 30, 100, 102, 99, 101),
        _candle_at(2024, 3, 4, 9, 45, 101, 102, 100, 101),
        _candle_at(2024, 3, 4, 10, 0, 101, 102, 100, 101),
        _candle_at(2024, 3, 4, 10, 15, 101, 102, 100, 101),
        _candle_at(2024, 3, 4, 10, 30, 101, 102, 100, 101),
        # manipulation down: low < 99, close > 99
        _candle_at(2024, 3, 4, 10, 45, 101, 102, 95, 101),  # 5 manip down
        # distribution up: big bullish displacement
        _candle_at(2024, 3, 4, 11, 0, 101, 130, 101, 130),  # 6 disp bullish
        _candle_at(2024, 3, 4, 11, 15, 130, 131, 129, 130),
    ]
    po3 = detect_power_of_three(candles, period="day", tz="America/New_York",
                                accum_bars=5)
    assert len(po3) == 1
    p = po3[0]
    assert p["manipulation_direction"] == "down"
    assert p["distribution_direction"] == "up"
