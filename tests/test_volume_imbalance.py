"""Tests for volume_imbalance.py — volume imbalance and opening gaps."""

from datetime import time
from zoneinfo import ZoneInfo

import pytest

from detectors.volume_imbalance import detect_volume_imbalance, detect_opening_gaps
from tests._fixtures import mkseries, mkc


def _candle_at(year, month, day, hour, minute, o, h, l, c, tz="America/New_York"):
    from datetime import datetime
    zone = ZoneInfo(tz)
    dt = datetime(year, month, day, hour, minute, tzinfo=zone)
    return mkc(o, h, l, c, int(dt.timestamp()))


# ── detect_volume_imbalance ───────────────────────────────────────────────────

def test_vi_empty():
    assert detect_volume_imbalance([]) == []


def test_vi_single_candle():
    assert detect_volume_imbalance(mkseries([(1, 2, 0, 1)])) == []


def test_vi_textbook_up_gap():
    # Bodies gap up but ranges overlap.
    # c0: body [10,12] (open 10 close 12), range [9,13]
    # c1: body [14,16] (open 14 close 16), range [13,17]
    # bodies: 12 < 14 (gap), ranges overlap (13<=13). -> imbalance [12,14]
    candles = mkseries([
        (10, 13, 9, 12),
        (14, 17, 13, 16),
    ])
    vi = detect_volume_imbalance(candles)
    assert len(vi) == 1
    assert vi[0]["type"] == "volume_imbalance"
    assert vi[0]["bottom"] == 12
    assert vi[0]["top"] == 14
    assert vi[0]["index"] == 1


def test_vi_textbook_down_gap():
    # c0: body [14,16], range [13,17]
    # c1: body [10,12], range [9,13]
    # bodies: 14 > 12 (gap down), ranges overlap. -> imbalance [12,14]
    candles = mkseries([
        (14, 17, 13, 16),
        (10, 13, 9, 12),
    ])
    vi = detect_volume_imbalance(candles)
    assert len(vi) == 1
    assert vi[0]["bottom"] == 12
    assert vi[0]["top"] == 14


def test_vi_full_fvg_not_volume_imbalance():
    # Ranges do NOT overlap -> this is a full FVG, not a volume imbalance.
    candles = mkseries([
        (10, 12, 9, 11),   # range [9,12]
        (15, 18, 14, 17),  # range [14,18] -> no overlap with [9,12]
    ])
    assert detect_volume_imbalance(candles) == []


def test_vi_touching_bodies_no_imbalance():
    # Bodies touch (c0 close == c1 open) -> no strict gap.
    candles = mkseries([
        (10, 13, 9, 12),   # body [10,12]
        (12, 16, 11, 15),  # body [12,15] -> 12 == 12, no gap
    ])
    assert detect_volume_imbalance(candles) == []


def test_vi_overlapping_bodies_no_imbalance():
    candles = mkseries([
        (10, 13, 9, 12),   # body [10,12]
        (11, 16, 10, 15),  # body [11,15] -> overlaps [10,12]
    ])
    assert detect_volume_imbalance(candles) == []


# ── detect_opening_gaps ───────────────────────────────────────────────────────

def test_opening_gaps_empty():
    assert detect_opening_gaps([], boundary="day", tz="America/New_York") == []


def test_opening_gaps_tz_required():
    with pytest.raises(TypeError):
        detect_opening_gaps([mkc(10, 11, 9, 10, 1000)], boundary="day")


def test_opening_gaps_ndog_day_boundary():
    # Day 1 closes at 100, Day 2 opens at 105 -> NDOG gap.
    candles = [
        _candle_at(2024, 3, 4, 15, 0, 100, 101, 99, 100),  # day1 close=100
        _candle_at(2024, 3, 5, 9, 30, 105, 106, 104, 106),  # day2 open=105
    ]
    gaps = detect_opening_gaps(candles, boundary="day", tz="America/New_York")
    assert len(gaps) == 1
    g = gaps[0]
    assert g["type"] == "ndog"
    assert g["prior_close"] == 100
    assert g["current_open"] == 105
    assert g["top"] == 105
    assert g["bottom"] == 100


def test_opening_gaps_nwog_week_boundary():
    # Week 1 (Friday) closes at 200, Week 2 (Monday) opens at 210 -> NWOG.
    # 2024-03-08 is a Friday; 2024-03-11 is a Monday (different ISO week).
    candles = [
        _candle_at(2024, 3, 8, 16, 0, 200, 201, 199, 200),  # Fri close=200
        _candle_at(2024, 3, 11, 9, 30, 210, 211, 209, 211),  # Mon open=210
    ]
    gaps = detect_opening_gaps(candles, boundary="week", tz="America/New_York")
    assert len(gaps) == 1
    assert gaps[0]["type"] == "nwog"
    assert gaps[0]["prior_close"] == 200
    assert gaps[0]["current_open"] == 210


def test_opening_gaps_no_boundary_returns_empty():
    # All candles on the same day -> no day boundary.
    candles = [
        _candle_at(2024, 3, 4, 9, 0, 100, 101, 99, 100),
        _candle_at(2024, 3, 4, 10, 0, 101, 102, 100, 101),
        _candle_at(2024, 3, 4, 11, 0, 101, 102, 100, 101),
    ]
    assert detect_opening_gaps(candles, boundary="day", tz="America/New_York") == []


def test_opening_gaps_session_boundary():
    # Two sessions: asian then ny_am. Gap between asian close and ny_am open.
    candles = [
        _candle_at(2024, 3, 4, 20, 0, 100, 101, 99, 100),   # asian
        _candle_at(2024, 3, 4, 21, 0, 100, 101, 99, 95),    # asian close=95
        _candle_at(2024, 3, 5, 7, 0, 98, 99, 97, 98),       # ny_am open=98
    ]
    gaps = detect_opening_gaps(candles, boundary="session", tz="America/New_York")
    assert len(gaps) == 1
    g = gaps[0]
    assert g["type"] == "opening_gap"
    assert g["prior_close"] == 95
    assert g["current_open"] == 98
