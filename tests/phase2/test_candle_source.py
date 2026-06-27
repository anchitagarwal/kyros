"""test_candle_source.py — MockCandleSource + ReplayCandleSource."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from trading.candle_source import (
    CandleSource,
    MockCandleSource,
    ReplayCandleSource,
    TIMEFRAMES,
    TF_MINUTES,
)

_NY = ZoneInfo("America/New_York")


# ── MockCandleSource ──────────────────────────────────────────────────────────


def test_mock_is_candle_source():
    assert isinstance(MockCandleSource("flat"), CandleSource)


def test_mock_emits_all_five_timeframes():
    src = MockCandleSource("flat", n_bars=5)
    batch = src.next()
    assert set(batch.keys()) == set(TIMEFRAMES)
    for tf in TIMEFRAMES:
        c = batch[tf]
        assert set(c.keys()) == {"open", "high", "low", "close", "volume", "timestamp"}
        ts = c["timestamp"]
        assert isinstance(ts, str)
        # Must be a parseable, tz-aware ISO-8601 string.
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None


def test_mock_is_done_false_until_exhausted():
    src = MockCandleSource("flat", n_bars=3)
    assert src.is_done() is False
    src.next()
    src.next()
    src.next()
    assert src.is_done() is True
    assert src.next() is None


def test_mock_determinism_same_seed_identical():
    a = MockCandleSource("sweep_and_fvg", seed=42, n_bars=20)
    b = MockCandleSource("sweep_and_fvg", seed=42, n_bars=20)
    while not a.is_done():
        ca = a.next()
        cb = b.next()
        for tf in TIMEFRAMES:
            assert ca[tf]["open"] == cb[tf]["open"]
            assert ca[tf]["close"] == cb[tf]["close"]
            assert ca[tf]["high"] == cb[tf]["high"]
            assert ca[tf]["low"] == cb[tf]["low"]


def test_mock_different_seed_different_series():
    a = MockCandleSource("flat", seed=1, n_bars=20)
    b = MockCandleSource("flat", seed=2, n_bars=20)
    ca = a.next()
    cb = b.next()
    # At least one TF differs (noise is seed-dependent).
    assert any(ca[tf]["close"] != cb[tf]["close"] for tf in TIMEFRAMES)


@pytest.mark.parametrize("scenario", MockCandleSource.SCENARIOS)
def test_all_scenarios_produce_enough_candles(scenario):
    """Each scenario must emit enough bars to fill the largest window (4h→60)."""
    src = MockCandleSource(scenario, n_bars=60)
    count = 0
    while not src.is_done():
        src.next()
        count += 1
    assert count == 60


def test_trending_up_produces_uptrend():
    src = MockCandleSource("trending_up", n_bars=50)
    bars = [src.next()["5m"] for _ in range(50)]
    # Overall close should rise.
    assert bars[-1]["close"] > bars[0]["close"]


def test_trending_down_produces_downtrend():
    src = MockCandleSource("trending_down", n_bars=50)
    bars = [src.next()["5m"] for _ in range(50)]
    assert bars[-1]["close"] < bars[0]["close"]


def test_trending_up_down_symmetric():
    """trending_down is the mirror of trending_up: same noise, opposite drift.

    The close-to-open delta of the down scenario must be the exact negation
    of the up scenario's delta for the same bar index (same seed → same noise).
    """
    up = MockCandleSource("trending_up", seed=7, n_bars=30)
    down = MockCandleSource("trending_down", seed=7, n_bars=30)
    for _ in range(30):
        u = up.next()["5m"]
        d = down.next()["5m"]
        up_delta = u["close"] - u["open"]
        down_delta = d["close"] - d["open"]
        assert up_delta == pytest.approx(-down_delta)


def test_killzone_active_timestamp_in_killzone():
    """The first 1m timestamp should fall inside the NY AM killzone (09:30-11:00)."""
    src = MockCandleSource("killzone_active", n_bars=5)
    c = src.next()["1m"]
    local = datetime.fromisoformat(c["timestamp"]).astimezone(_NY)
    assert 9 <= local.hour <= 10  # 09:30-10:59 window


def test_unknown_scenario_raises():
    with pytest.raises(ValueError):
        MockCandleSource("bogus")


# ── ReplayCandleSource ────────────────────────────────────────────────────────


def _write_1m_csv(tmp_path, n=120):
    """Write a small 1m OHLCV CSV starting at 09:30 ET."""
    import csv as _csv

    path = tmp_path / "1m.csv"
    start = datetime(2026, 6, 15, 9, 30, tzinfo=_NY)
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        price = 20000.0
        for i in range(n):
            ts = (start + timedelta(minutes=i)).isoformat()
            o = price
            c = price + (i % 5) - 2
            h = max(o, c) + 1
            l = min(o, c) - 1
            w.writerow([ts, o, h, l, c, 1000])
            price = c
    return path


def test_replay_resamples_and_feeds(tmp_path):
    path = _write_1m_csv(tmp_path, n=120)
    src = ReplayCandleSource(str(path))
    seen = {tf: 0 for tf in TIMEFRAMES}
    while not src.is_done():
        batch = src.next()
        if batch:
            for tf in batch:
                seen[tf] += 1
    # 1m should have ~119 bars (120 minus the dropped partial trailing bar).
    assert seen["1m"] >= 100
    # 5m should have ~23 bars (120/5 = 24, minus partial).
    assert seen["5m"] >= 15
    # 15m should have ~7 bars.
    assert seen["15m"] >= 4


def test_replay_timestamps_chronological(tmp_path):
    path = _write_1m_csv(tmp_path, n=60)
    src = ReplayCandleSource(str(path))
    prev_ts = None
    while not src.is_done():
        batch = src.next()
        if batch and "1m" in batch:
            ts = batch["1m"]["timestamp"]
            if prev_ts is not None:
                assert ts >= prev_ts  # ISO strings compare chronologically
            prev_ts = ts


# ── ReplayCandleSource: lookahead-bias regression ─────────────────────────────


def _write_contiguous_1m_csv(tmp_path, n=300):
    """Write a contiguous 1m CSV with a strictly rising price (price = 20000+i).

    A monotonic price makes the no-lookahead invariant easy to assert: at any
    decision timestamp ``now``, the highest 1m price seen so far is exactly
    ``20000 + (minutes elapsed)``. If a higher-TF bar is emitted before it has
    closed, its ``high`` will exceed that running maximum — a detectable leak.
    """
    import csv as _csv

    path = tmp_path / "1m_contig.csv"
    start = datetime(2026, 6, 15, 9, 30, tzinfo=_NY)
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for i in range(n):
            ts = (start + timedelta(minutes=i)).isoformat()
            o = 20000.0 + i
            c = o + 0.5
            h = o + 1.0
            l = o - 1.0
            w.writerow([ts, o, h, l, c, 1000])
    return path


def test_replay_no_lookahead_higher_tf_emits_only_after_close(tmp_path):
    """A higher-TF bar must be emitted only once it has fully closed.

    Regression for the lookahead-bias bug where a resampled bar (left-labeled
    at its open time) was emitted as soon as ``now`` reached the bar's OPEN,
    exposing the bar's fully-formed OHLC (high/low/close) before the period
    had elapsed. The fix gates emission on ``now >= bar_open + tf_duration``
    (the bar's close time).

    For every emitted higher-TF bar we assert:
      (1) now_dt >= bar_open + tf_duration  (the bar has closed), and
      (2) the bar's high/low never exceed the 1m extremes seen up to now_dt
          (no future price is visible at the decision timestamp).
    """
    path = _write_contiguous_1m_csv(tmp_path, n=300)
    src = ReplayCandleSource(str(path))

    # Collect the 1m series to compute running extremes at each decision time.
    one_m_rows: list[tuple[datetime, float, float]] = []
    while not src.is_done():
        batch = src.next()
        if not batch or "1m" not in batch:
            continue
        bar = batch["1m"]
        now_dt = datetime.fromisoformat(bar["timestamp"])
        one_m_rows.append((now_dt, bar["high"], bar["low"]))

    # Re-run a fresh source to inspect higher-TF emissions against the
    # running 1m extremes computed above.
    src2 = ReplayCandleSource(str(path))
    violations: list[str] = []
    idx = 0
    while not src2.is_done():
        batch = src2.next()
        if not batch or "1m" not in batch:
            continue
        now_dt = datetime.fromisoformat(batch["1m"]["timestamp"])
        # Running 1m extremes up to and including now_dt.
        seen = one_m_rows[: idx + 1]
        running_high = max(r[1] for r in seen)
        running_low = min(r[2] for r in seen)
        for tf in ("4h", "1h", "15m", "5m"):
            if tf not in batch:
                continue
            bar = batch[tf]
            bar_open = datetime.fromisoformat(bar["timestamp"])
            bar_close = bar_open + timedelta(minutes=TF_MINUTES[tf])
            # (1) The bar must have closed before/at the decision time.
            if now_dt < bar_close:
                violations.append(
                    f"{tf} bar open={bar_open} emitted at now={now_dt} "
                    f"before close={bar_close}"
                )
            # (2) The bar's OHLC must not exceed the 1m extremes seen so far.
            if bar["high"] > running_high + 1e-9:
                violations.append(
                    f"{tf} bar high={bar['high']} > running 1m high={running_high} "
                    f"at now={now_dt} (future leak)"
                )
            if bar["low"] < running_low - 1e-9:
                violations.append(
                    f"{tf} bar low={bar['low']} < running 1m low={running_low} "
                    f"at now={now_dt} (future leak)"
                )
        idx += 1

    assert not violations, "lookahead violations:\n  " + "\n  ".join(violations)


def test_replay_first_1h_bar_emitted_at_close_not_open(tmp_path):
    """The first 1h bar (open 09:00, covering 09:00-09:59) must emit at 10:00.

    This is the exact failure mode the evaluator probed: with the bug, the 1h
    bar was emitted at 09:30 (its open) carrying high/close from 09:59. After
    the fix it emits at 10:00 (its close), so no future price is visible.
    """
    path = _write_contiguous_1m_csv(tmp_path, n=300)
    src = ReplayCandleSource(str(path))

    first_1h = None
    while not src.is_done():
        batch = src.next()
        if batch and "1h" in batch:
            first_1h = batch
            break

    assert first_1h is not None, "no 1h bar was ever emitted"
    now_dt = datetime.fromisoformat(first_1h["1m"]["timestamp"])
    # The 1h bar's own timestamp is its open time.
    h_open = datetime.fromisoformat(first_1h["1h"]["timestamp"])
    # The 1h bar covers [h_open, h_open+1h); it must emit at h_open+1h.
    expected_emit = h_open + timedelta(minutes=TF_MINUTES["1h"])
    assert now_dt >= expected_emit, (
        f"1h bar (open {h_open}) emitted at now={now_dt}, but should not "
        f"appear until its close at {expected_emit}"
    )
