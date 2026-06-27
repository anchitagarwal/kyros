"""test_candle_window.py — sliding window behaviour."""

import pytest

from trading.candle_window import CandleWindow, DEFAULT_SIZES, TIMEFRAMES


def _candle(i, tf="1m"):
    from datetime import datetime, timedelta
    return {
        "open": float(i), "high": float(i) + 1, "low": float(i) - 1,
        "close": float(i), "volume": 1000.0,
        "timestamp": datetime(2026, 6, 15, 9, 30) + timedelta(minutes=i),
    }


def test_default_sizes_match_spec():
    assert DEFAULT_SIZES == {"4h": 60, "1h": 100, "15m": 200, "5m": 300, "1m": 500}


def test_update_appends_correctly():
    w = CandleWindow({"1m": 5, "5m": 3})
    w.update({"1m": _candle(0), "5m": _candle(0)})
    assert len(w.to_list("1m")) == 1
    assert len(w.to_list("5m")) == 1


def test_maxlen_eviction_fifo():
    w = CandleWindow({"1m": 3})
    for i in range(13):  # push 13 into a size-3 window
        w.update({"1m": _candle(i)})
    lst = w.to_list("1m")
    assert len(lst) == 3
    # FIFO: last 3 candles (indices 10, 11, 12).
    assert [c["open"] for c in lst] == [10.0, 11.0, 12.0]


def test_to_list_returns_list_not_deque():
    w = CandleWindow({"1m": 5})
    w.update({"1m": _candle(0)})
    lst = w.to_list("1m")
    assert isinstance(lst, list)
    # Mutation of the returned list must not affect the window.
    lst.append(_candle(99))
    assert len(w.to_list("1m")) == 1


def test_partial_update_leaves_others_unchanged():
    w = CandleWindow({"1m": 5, "5m": 5})
    w.update({"1m": _candle(0)})
    w.update({"1m": _candle(1)})  # only 1m advances
    assert len(w.to_list("1m")) == 2
    assert len(w.to_list("5m")) == 0


def test_to_list_chronological_order():
    w = CandleWindow({"1m": 5})
    for i in range(5):
        w.update({"1m": _candle(i)})
    lst = w.to_list("1m")
    assert [c["open"] for c in lst] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_unknown_timeframe_raises():
    w = CandleWindow({"1m": 5})
    with pytest.raises(KeyError):
        w.update({"2m": _candle(0)})
    with pytest.raises(KeyError):
        w.to_list("2m")


def test_is_warm():
    w = CandleWindow({"1m": 3})
    assert not w.is_warm("1m")
    w.update({"1m": _candle(0)})
    w.update({"1m": _candle(1)})
    assert not w.is_warm("1m")
    w.update({"1m": _candle(2)})
    assert w.is_warm("1m")
    # One more — still warm (eviction keeps it full).
    w.update({"1m": _candle(3)})
    assert w.is_warm("1m")
