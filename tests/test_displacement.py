"""Tests for displacement.py.

`leaves_fvg` semantics (ATLAS-aligned; see ../atlas/atlas/models/fvg_scalp.py:270
and ict_22.py:560): an FVG's `index` is its MIDDLE / displacement candle, so a
displacement candle "leaves an FVG" iff its own index equals an FVG's middle
index. The FVG is only *confirmed* at the third candle, but the candle that
*leaves* (creates) the gap is the middle one.
"""

import pytest

from detectors.displacement import detect_displacement
from tests._fixtures import mkseries


def test_displacement_empty():
    assert detect_displacement([]) == []


def test_displacement_single_candle():
    assert detect_displacement(mkseries([(1, 2, 0, 1)])) == []


def test_displacement_spike_flagged():
    # Normal candles with range ~2, then a big body candle (body=10).
    rows = [(10, 12, 10, 11)] * 5  # range 2, body 1
    rows.append((11, 21, 11, 21))  # body 10, range 10 -> strength 10/2 = 5 > 1.5
    candles = mkseries(rows)
    disp = detect_displacement(candles, window=5, k=1.5)
    assert len(disp) == 1
    d = disp[0]
    assert d["type"] == "displacement_bullish"
    assert d["index"] == 5
    assert d["strength"] == pytest.approx(5.0)
    assert d["leaves_fvg"] in (True, False)


def test_displacement_bearish_direction():
    rows = [(10, 12, 10, 11)] * 5
    rows.append((11, 11, 1, 1))  # body 10 down
    candles = mkseries(rows)
    disp = detect_displacement(candles, window=5, k=1.5)
    assert len(disp) == 1
    assert disp[0]["type"] == "displacement_bearish"


def test_displacement_normal_candle_not_flagged():
    rows = [(10, 12, 10, 11)] * 6  # body 1, range 2 -> strength 0.5
    candles = mkseries(rows)
    assert detect_displacement(candles, window=5, k=1.5) == []


def test_displacement_leaves_fvg_true():
    # The displacement candle at index 1 IS the middle candle of an FVG
    # (candles 0,1,2: c0.high=10 < c2.low=12 -> bullish FVG, middle=1).
    # The candle that leaves the gap is the middle/displacement candle, so
    # leaves_fvg must be True here.
    rows = [
        (8, 10, 7, 9),     # 0  range 3, high=10
        (9, 20, 9, 20),    # 1  body 11, range 11 -> strength 11/3 ~ 3.67 (displacement)
        (20, 22, 12, 21),  # 2  low=12 > c0.high=10 -> FVG confirmed, middle=1
    ]
    candles = mkseries(rows)
    disp = detect_displacement(candles, window=5, k=1.0)
    assert any(d["index"] == 1 for d in disp)
    d1 = [d for d in disp if d["index"] == 1][0]
    assert d1["leaves_fvg"] is True


def test_displacement_confirmation_candle_does_not_leave_fvg():
    # The displacement candle at index 2 is the THIRD / confirmation candle of
    # the FVG (candles 0,1,2: c0.high=10 < c2.low=12 -> bullish FVG, middle=1).
    # The confirmation candle does NOT leave the gap (the middle candle does),
    # so leaves_fvg must be False here.
    rows = [
        (8, 10, 7, 9),     # 0  high=10
        (9, 11, 9, 10),    # 1  small (FVG middle)
        (10, 25, 12, 25),  # 2  body 15, low=12 > c0.high=10 -> FVG confirmed, disp at 2
    ]
    candles = mkseries(rows)
    disp = detect_displacement(candles, window=5, k=1.0)
    d2 = [d for d in disp if d["index"] == 2]
    assert len(d2) == 1
    assert d2[0]["leaves_fvg"] is False


def test_displacement_warmup_uses_available_history():
    # window=14 but only 3 candles; index 1 should still evaluate using 1 candle.
    rows = [
        (10, 12, 10, 11),  # 0  range 2
        (11, 30, 11, 30),  # 1  body 19, strength 19/2 = 9.5 > 1.5
    ]
    candles = mkseries(rows)
    disp = detect_displacement(candles, window=14, k=1.5)
    assert len(disp) == 1
    assert disp[0]["index"] == 1


def test_displacement_flat_market_no_displacement():
    rows = [(10, 10, 10, 10)] * 5
    candles = mkseries(rows)
    assert detect_displacement(candles, window=5, k=1.5) == []


def test_displacement_threshold_respected():
    # body/atr exactly at k should NOT trigger (uses >).
    rows = [(10, 12, 10, 11)] * 5  # range 2
    # body = 3 -> strength 3/2 = 1.5; with k=1.5, 1.5 > 1.5 is False.
    rows.append((11, 14, 11, 14))  # body 3
    candles = mkseries(rows)
    assert detect_displacement(candles, window=5, k=1.5) == []
    # with k=1.0, 1.5 > 1.0 True -> flagged
    assert len(detect_displacement(candles, window=5, k=1.0)) == 1
