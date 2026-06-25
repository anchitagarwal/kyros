"""Tests for inducement.py — turtle soup and IDM."""

import pytest

from detectors.inducement import detect_turtle_soup, detect_inducement
from tests._fixtures import mkseries


# ── detect_turtle_soup ────────────────────────────────────────────────────────

def test_turtle_soup_empty():
    assert detect_turtle_soup([]) == []


def test_turtle_soup_bullish_reversal():
    # Prior lookback low at ~9; candle breaks below and closes back above.
    rows = [(10, 12, 9, 11)] * 10   # prior lows ~9
    rows.append((11, 12, 7, 11))    # low=7<9, close=11>9 -> bullish turtle soup
    candles = mkseries(rows)
    ts = detect_turtle_soup(candles, lookback=10, tolerance=0.0)
    bull = [t for t in ts if t["type"] == "turtle_soup_bullish"]
    assert len(bull) == 1
    assert bull[0]["break_index"] == 10
    assert bull[0]["broken_level"] == 9


def test_turtle_soup_bearish_reversal():
    rows = [(10, 12, 9, 11)] * 10   # prior highs ~12
    rows.append((11, 15, 9, 11))    # high=15>12, close=11<12 -> bearish turtle soup
    candles = mkseries(rows)
    ts = detect_turtle_soup(candles, lookback=10, tolerance=0.0)
    bear = [t for t in ts if t["type"] == "turtle_soup_bearish"]
    assert len(bear) == 1
    assert bear[0]["break_index"] == 10
    assert bear[0]["broken_level"] == 12


def test_turtle_soup_clean_breakout_excluded():
    # Breaks below prior low and closes BELOW (continuation) -> not turtle soup.
    rows = [(10, 12, 9, 11)] * 10
    rows.append((11, 12, 5, 6))     # low=5<9, close=6<9 -> no reversal
    candles = mkseries(rows)
    ts = detect_turtle_soup(candles, lookback=10, tolerance=0.0)
    assert [t for t in ts if t["type"] == "turtle_soup_bullish"] == []


def test_turtle_soup_insufficient_lookback():
    # Only 1 candle in window -> skipped.
    candles = mkseries([(10, 12, 9, 11), (11, 12, 5, 11)])
    ts = detect_turtle_soup(candles, lookback=10, tolerance=0.0)
    assert ts == []


def test_turtle_soup_tolerance_applied():
    # With tolerance, a break must exceed level + band.
    rows = [(10, 12, 9, 11)] * 10   # avg range 3
    # low=8.5; band = 0.5*3=1.5; need low < 9 - 1.5 = 7.5. 8.5 not < 7.5 -> no.
    rows.append((11, 12, 8.5, 11))
    candles = mkseries(rows)
    ts = detect_turtle_soup(candles, lookback=10, tolerance=0.5)
    assert [t for t in ts if t["type"] == "turtle_soup_bullish"] == []


# ── detect_inducement ─────────────────────────────────────────────────────────

def test_inducement_empty():
    assert detect_inducement([]) == []


def test_inducement_idm_before_bos():
    # Establish a swing high (BSL), sweep it, then BOS bullish.
    rows = [
        (10, 12, 9, 11),    # 0
        (11, 16, 10, 15),   # 1
        (15, 20, 14, 19),   # 2  swing high 20
        (19, 19, 13, 14),   # 3
        (14, 14, 10, 11),   # 4
        (11, 11, 9, 10),    # 5
        (10, 10, 8, 9),     # 6
        (9, 11, 8, 10),     # 7
        (10, 21, 9, 11),    # 8  sweep BSL: high=21>20, close=11<20
        (11, 30, 11, 30),   # 9  BOS bullish: close=30>20
        (30, 31, 29, 30),   # 10
        (30, 31, 29, 30),   # 11
    ]
    candles = mkseries(rows)
    idm = detect_inducement(candles, lookback=2, idm_window=5)
    assert len(idm) >= 1
    assert idm[0]["type"] == "idm"
    assert idm[0]["induced_level"] == 20
    assert idm[0]["induced_index"] == 2
    assert idm[0]["related_structure_index"] == 9


def test_inducement_no_prior_pool_no_idm():
    # BOS with no swept pool before it -> no IDM.
    rows = [
        (10, 12, 9, 11), (11, 16, 10, 15), (15, 20, 14, 19), (19, 19, 13, 14),
        (14, 14, 10, 11), (11, 11, 9, 10), (10, 10, 8, 9), (9, 11, 8, 10),
        (10, 12, 9, 11),   # 8  no sweep
        (11, 30, 11, 30),  # 9  BOS bullish
        (30, 31, 29, 30), (30, 31, 29, 30),
    ]
    candles = mkseries(rows)
    idm = detect_inducement(candles, lookback=2, idm_window=5)
    assert idm == []


def test_inducement_no_bos_no_idm():
    # Sweep but no BOS -> no IDM.
    rows = [
        (10, 12, 9, 11), (11, 16, 10, 15), (15, 20, 14, 19), (19, 19, 13, 14),
        (14, 14, 10, 11), (11, 11, 9, 10), (10, 10, 8, 9), (9, 11, 8, 10),
        (10, 21, 9, 11),   # 8  sweep BSL
        (11, 12, 9, 11),   # 9  no BOS (close 11 < 20)
        (11, 12, 9, 11), (11, 12, 9, 11),
    ]
    candles = mkseries(rows)
    idm = detect_inducement(candles, lookback=2, idm_window=5)
    assert idm == []
