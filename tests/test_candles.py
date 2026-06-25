"""Tests for candles.py — ingestion and validation."""

import math
import pytest

from detectors.candles import validate_candles, candle_metrics, _to_epoch, _to_datetime
from tests._fixtures import mkc


# ── validate_candles: happy path ──────────────────────────────────────────────

def test_empty_returns_empty_list():
    assert validate_candles([]) == []


def test_none_returns_empty_list():
    assert validate_candles(None) == []


def test_single_valid_candle_returned():
    c = mkc(10, 12, 9, 11, 1000)
    out = validate_candles([c])
    assert len(out) == 1
    assert out[0]["open"] == 10.0
    assert out[0]["high"] == 12.0
    assert out[0]["low"] == 9.0
    assert out[0]["close"] == 11.0
    assert out[0]["volume"] == 1000.0
    assert out[0]["duplicate_timestamp"] is False


def test_valid_passthrough_values_preserved():
    candles = [mkc(10, 12, 9, 11, 1000), mkc(11, 13, 10, 12, 1001)]
    out = validate_candles(candles)
    assert len(out) == 2
    # Values preserved (coerced to float).
    assert out[0]["open"] == 10.0 and out[1]["close"] == 12.0
    # Timestamps preserved (not mutated).
    assert out[0]["timestamp"] == 1000
    assert out[1]["timestamp"] == 1001


def test_ohlcv_coerced_to_float():
    out = validate_candles([mkc("10", "12", "9", "11", 1000, "100")])
    assert isinstance(out[0]["open"], float)
    assert out[0]["volume"] == 100.0


# ── validate_candles: rejection paths ─────────────────────────────────────────

def test_missing_key_raises():
    c = {"open": 10, "high": 12, "low": 9, "close": 11, "timestamp": 1000}
    with pytest.raises(ValueError, match="missing required key"):
        validate_candles([c])


def test_non_numeric_ohlcv_raises():
    with pytest.raises(ValueError, match="non-numeric"):
        validate_candles([mkc("abc", 12, 9, 11, 1000)])


def test_none_ohlcv_raises():
    c = mkc(None, 12, 9, 11, 1000)
    with pytest.raises(ValueError, match="open is None"):
        validate_candles([c])


def test_nan_raises():
    with pytest.raises(ValueError, match="NaN"):
        validate_candles([mkc(float("nan"), 12, 9, 11, 1000)])


def test_high_less_than_low_raises():
    with pytest.raises(ValueError, match="high < low"):
        validate_candles([mkc(10, 8, 9, 11, 1000)])


def test_high_less_than_body_raises():
    # open=10, close=15 -> body high 15, but high=14
    with pytest.raises(ValueError, match="high < body high"):
        validate_candles([mkc(10, 14, 9, 15, 1000)])


def test_low_greater_than_body_raises():
    # open=10, close=8 -> body low 8, but low=9
    with pytest.raises(ValueError, match="low > body low"):
        validate_candles([mkc(10, 12, 9, 8, 1000)])


def test_negative_volume_raises():
    with pytest.raises(ValueError, match="volume is negative"):
        validate_candles([mkc(10, 12, 9, 11, 1000, -5)])


def test_zero_volume_allowed():
    out = validate_candles([mkc(10, 12, 9, 11, 1000, 0)])
    assert out[0]["volume"] == 0.0


def test_unsorted_timestamps_raises():
    candles = [mkc(10, 12, 9, 11, 1000), mkc(11, 13, 10, 12, 999)]
    with pytest.raises(ValueError, match="unsorted"):
        validate_candles(candles)


def test_non_list_raises():
    with pytest.raises(ValueError, match="must be a list"):
        validate_candles("not a list")


def test_non_dict_candle_raises():
    with pytest.raises(ValueError, match="not a dict"):
        validate_candles(["not a dict"])


# ── duplicate timestamps ──────────────────────────────────────────────────────

def test_duplicate_timestamps_permitted_and_flagged():
    candles = [mkc(10, 12, 9, 11, 1000), mkc(11, 13, 10, 12, 1000)]
    out = validate_candles(candles)
    assert len(out) == 2
    assert out[0]["duplicate_timestamp"] is False
    assert out[1]["duplicate_timestamp"] is True


def test_equal_then_greater_timestamp_ok():
    candles = [mkc(10, 12, 9, 11, 1000), mkc(11, 13, 10, 12, 1000),
               mkc(12, 14, 11, 13, 1001)]
    out = validate_candles(candles)
    assert [c["duplicate_timestamp"] for c in out] == [False, True, False]


# ── timestamp normalization helpers ───────────────────────────────────────────

def test_to_epoch_int_passthrough():
    assert _to_epoch(1000) == 1000


def test_to_epoch_iso_string():
    # 2024-01-01T00:00:00Z == 1704067200
    assert _to_epoch("2024-01-01T00:00:00Z") == 1704067200


def test_to_epoch_iso_with_offset():
    # 2024-01-01T05:00:00+05:00 == 2024-01-01T00:00:00Z
    assert _to_epoch("2024-01-01T05:00:00+05:00") == 1704067200


def test_to_epoch_naive_iso_assumed_utc():
    assert _to_epoch("2024-01-01T00:00:00") == 1704067200


def test_to_epoch_bad_string_raises():
    with pytest.raises(ValueError, match="unparseable"):
        _to_epoch("not a date")


def test_to_epoch_bool_rejected():
    with pytest.raises(ValueError, match="bool"):
        _to_epoch(True)


def test_to_datetime_int():
    from datetime import datetime, timezone
    dt = _to_datetime(1704067200)
    assert dt == datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_to_datetime_iso():
    from datetime import datetime, timezone
    dt = _to_datetime("2024-01-01T00:00:00Z")
    assert dt == datetime(2024, 1, 1, tzinfo=timezone.utc)


# ── candle_metrics ────────────────────────────────────────────────────────────

def test_metrics_bull_candle():
    m = candle_metrics(mkc(10, 15, 8, 13, 0))
    assert m["body"] == 3.0
    assert m["range"] == 7.0
    assert m["upper_wick"] == 2.0   # 15 - max(10,13)=13
    assert m["lower_wick"] == 2.0   # min(10,13)=10 - 8
    assert m["midpoint"] == 11.5
    assert m["direction"] == "bull"


def test_metrics_bear_candle():
    m = candle_metrics(mkc(13, 15, 8, 10, 0))
    assert m["body"] == 3.0
    assert m["upper_wick"] == 2.0   # 15 - max(13,10)=13
    assert m["lower_wick"] == 2.0   # min(13,10)=10 - 8
    assert m["direction"] == "bear"


def test_metrics_doji_flat():
    # open == close, high == low -> range 0
    m = candle_metrics(mkc(10, 10, 10, 10, 0))
    assert m["body"] == 0.0
    assert m["range"] == 0.0
    assert m["upper_wick"] == 0.0
    assert m["lower_wick"] == 0.0
    assert m["direction"] == "doji"


def test_metrics_doji_with_wicks():
    # open == close but with wicks
    m = candle_metrics(mkc(10, 12, 8, 10, 0))
    assert m["body"] == 0.0
    assert m["range"] == 4.0
    assert m["upper_wick"] == 2.0
    assert m["lower_wick"] == 2.0
    assert m["direction"] == "doji"
