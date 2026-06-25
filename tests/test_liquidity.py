"""Tests for liquidity.py — equal levels, prior levels, sweeps."""

from zoneinfo import ZoneInfo

import pytest

from detectors.liquidity import (
    detect_equal_levels,
    detect_prior_levels,
    detect_liquidity_sweeps,
)
from tests._fixtures import mkseries, mkc


def _candle_at(year, month, day, hour, minute, o, h, l, c, tz="America/New_York"):
    from datetime import datetime
    zone = ZoneInfo(tz)
    dt = datetime(year, month, day, hour, minute, tzinfo=zone)
    return mkc(o, h, l, c, int(dt.timestamp()))


# ── detect_equal_levels ───────────────────────────────────────────────────────

def test_equal_levels_empty():
    assert detect_equal_levels([]) == []


def test_equal_levels_cluster():
    # Two swing highs at ~same price (within tolerance). All OHLC valid and
    # swing extremes strictly distinct from their lookback neighbors.
    rows = [
        (10, 12, 9, 11),    # 0
        (11, 16, 10, 15),   # 1
        (15, 20, 14, 19),   # 2  swing high 20
        (19, 19, 13, 14),   # 3
        (14, 14, 10, 11),   # 4
        (11, 11, 9, 10),    # 5
        (10, 10, 8, 9),     # 6  swing low 8
        (9, 11, 8, 10),     # 7
        (10, 14, 9, 13),    # 8
        (13, 17, 12, 16),   # 9
        (16, 20, 15, 19),   # 10 swing high 20 (equal to index 2)
        (19, 19, 13, 14),   # 11
        (14, 14, 10, 11),   # 12
    ]
    candles = mkseries(rows)
    eq = detect_equal_levels(candles, tolerance=0.1, lookback=2)
    highs = [e for e in eq if e["type"] == "equal_highs"]
    assert len(highs) >= 1
    assert highs[0]["count"] >= 2
    assert set(highs[0]["member_indices"]) >= {2, 10}


def test_equal_levels_single_extreme_none():
    # Only one swing high -> no cluster.
    rows = [
        (10, 12, 9, 11), (11, 16, 10, 15), (15, 20, 14, 19),
        (19, 19, 13, 14), (14, 14, 10, 11), (11, 11, 9, 10),
        (10, 10, 8, 9),
    ]
    candles = mkseries(rows)
    eq = detect_equal_levels(candles, tolerance=0.1, lookback=2)
    highs = [e for e in eq if e["type"] == "equal_highs"]
    assert highs == []


def test_equal_levels_flat_market_not_everywhere():
    # Flat market -> avg range 0 -> returns [] (no false equal highs).
    rows = [(10, 10, 10, 10)] * 9
    candles = mkseries(rows)
    assert detect_equal_levels(candles, tolerance=0.1) == []


def test_equal_levels_tolerance_range_relative():
    # With a tiny tolerance, two highs 1 apart should NOT cluster if avg range
    # is large; with a large tolerance they should.
    rows = [
        (10, 25, 5, 11),    # 0
        (11, 28, 6, 12),    # 1
        (12, 30, 7, 13),    # 2  swing high 30
        (13, 25, 8, 14),    # 3
        (14, 20, 9, 12),    # 4
        (12, 15, 10, 10),   # 5
        (10, 12, 8, 8),     # 6
        (8, 15, 8, 14),     # 7
        (14, 28, 13, 15),   # 8
        (15, 29, 14, 16),   # 9
        (16, 31, 15, 17),   # 10 swing high 31 (1 apart from 30)
        (17, 25, 16, 18),   # 11
        (18, 20, 16, 16),   # 12
    ]
    candles = mkseries(rows)
    # avg range ~15; tolerance 0.1 -> band ~1.5; |31-30|=1 <= 1.5 -> cluster.
    eq = detect_equal_levels(candles, tolerance=0.1, lookback=2)
    highs = [e for e in eq if e["type"] == "equal_highs"]
    assert len(highs) >= 1
    # tolerance 0.01 -> band ~0.15; |31-30|=1 > 0.15 -> no cluster.
    eq2 = detect_equal_levels(candles, tolerance=0.01, lookback=2)
    highs2 = [e for e in eq2 if e["type"] == "equal_highs"]
    assert highs2 == []


# ── detect_prior_levels ───────────────────────────────────────────────────────

def test_prior_levels_empty():
    assert detect_prior_levels([], period="day", tz="America/New_York") == []


def test_prior_levels_tz_required():
    with pytest.raises(TypeError):
        detect_prior_levels([mkc(10, 11, 9, 10, 1000)], period="day")


def test_prior_levels_pdh_pdl_across_day():
    # Day 1: high=120, low=80. Day 2 first candle anchors PDH=120, PDL=80.
    candles = [
        _candle_at(2024, 3, 4, 10, 0, 100, 120, 80, 100),
        _candle_at(2024, 3, 4, 11, 0, 100, 110, 90, 100),
        _candle_at(2024, 3, 5, 9, 30, 100, 105, 95, 100),  # new day
    ]
    levels = detect_prior_levels(candles, period="day", tz="America/New_York")
    pdh = [l for l in levels if l["type"] == "pdh"]
    pdl = [l for l in levels if l["type"] == "pdl"]
    assert len(pdh) == 1 and pdh[0]["level"] == 120
    assert len(pdl) == 1 and pdl[0]["level"] == 80
    assert pdh[0]["index"] == 2  # anchored at first candle of new day


def test_prior_levels_pwh_pwl_across_week():
    # Week 1 high=200 low=150; Week 2 first candle anchors PWH/PWL.
    candles = [
        _candle_at(2024, 3, 7, 10, 0, 180, 200, 150, 180),   # Thu week1
        _candle_at(2024, 3, 11, 9, 30, 180, 190, 170, 180),  # Mon week2
    ]
    levels = detect_prior_levels(candles, period="week", tz="America/New_York")
    pwh = [l for l in levels if l["type"] == "pwh"]
    pwl = [l for l in levels if l["type"] == "pwl"]
    assert len(pwh) == 1 and pwh[0]["level"] == 200
    assert len(pwl) == 1 and pwl[0]["level"] == 150


def test_prior_levels_insufficient_history():
    # Single day -> no prior period -> [].
    candles = [_candle_at(2024, 3, 4, 10, 0, 100, 120, 80, 100)]
    assert detect_prior_levels(candles, period="day", tz="America/New_York") == []


# ── detect_liquidity_sweeps ───────────────────────────────────────────────────

def test_sweeps_empty():
    assert detect_liquidity_sweeps([]) == []


def test_sweep_bsl_classic():
    # A swing high at 20; later candle wicks above 20 and closes below.
    rows = [
        (10, 12, 9, 11),    # 0
        (11, 16, 10, 15),   # 1
        (15, 20, 14, 19),   # 2  swing high 20
        (19, 19, 13, 14),   # 3
        (14, 14, 10, 11),   # 4
        (11, 11, 9, 10),    # 5
        (10, 10, 8, 9),     # 6
        (9, 11, 8, 10),     # 7
        (10, 21, 9, 11),    # 8  high=21>20, close=11<20 -> sweep_bsl
        (11, 12, 10, 11),   # 9
        (11, 12, 10, 11),   # 10
    ]
    candles = mkseries(rows)
    sweeps = detect_liquidity_sweeps(candles, lookback=2)
    bsl = [s for s in sweeps if s["type"] == "sweep_bsl"]
    assert len(bsl) == 1
    assert bsl[0]["sweep_index"] == 8
    assert bsl[0]["swept_level"] == 20
    assert bsl[0]["reversal_confirmed"] is True


def test_sweep_ssl_classic():
    # A swing low at 11; later candle wicks below 11 and closes above.
    rows = [
        (14, 15, 13, 14),   # 0
        (13, 14, 12, 13),   # 1
        (12, 13, 11, 13),   # 2  swing low 11
        (13, 14, 12, 14),   # 3
        (14, 15, 13, 14),   # 4
        (14, 15, 13, 14),   # 5
        (14, 15, 13, 14),   # 6
        (14, 15, 13, 14),   # 7
        (14, 15, 10, 15),   # 8  low=10<11, close=15>11 -> sweep_ssl
        (15, 16, 14, 15),   # 9
        (15, 16, 14, 15),   # 10
    ]
    candles = mkseries(rows)
    sweeps = detect_liquidity_sweeps(candles, lookback=2)
    ssl = [s for s in sweeps if s["type"] == "sweep_ssl"]
    assert len(ssl) == 1
    assert ssl[0]["sweep_index"] == 8
    assert ssl[0]["swept_level"] == 11


def test_sweep_clean_breakout_not_sweep():
    # Candle breaks above swing high and closes ABOVE (no reversal) -> not sweep.
    rows = [
        (10, 12, 9, 11), (11, 16, 10, 15), (15, 20, 14, 19),
        (19, 19, 13, 14), (14, 14, 10, 11), (11, 11, 9, 10),
        (10, 10, 8, 9), (9, 11, 8, 10),
        (10, 25, 9, 24),  # high=25>20, close=24>20 -> breakout, not sweep
        (24, 25, 23, 24), (24, 25, 23, 24),
    ]
    candles = mkseries(rows)
    sweeps = detect_liquidity_sweeps(candles, lookback=2)
    assert [s for s in sweeps if s["type"] == "sweep_bsl"] == []


def test_sweep_flat_market_none():
    rows = [(10, 10, 10, 10)] * 9
    candles = mkseries(rows)
    assert detect_liquidity_sweeps(candles) == []
