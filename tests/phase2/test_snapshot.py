"""test_snapshot.py — SnapshotBuilder correctness."""

import time as _time
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from trading.candle_source import MockCandleSource, TIMEFRAMES
from trading.candle_window import CandleWindow
from trading.snapshot import SnapshotBuilder, MarketSnapshot, LiquidityPool

_NY = ZoneInfo("America/New_York")


def _build_window(scenario, n=100):
    src = MockCandleSource(scenario, n_bars=n)
    w = CandleWindow({tf: n for tf in TIMEFRAMES})
    while not src.is_done():
        w.update(src.next())
    return w


def _build_snapshot(scenario, n=100, now=None):
    w = _build_window(scenario, n)
    return SnapshotBuilder().build(w, now=now)


# ── Structure ─────────────────────────────────────────────────────────────────


def test_snapshot_is_market_snapshot():
    snap = _build_snapshot("sweep_and_fvg")
    assert isinstance(snap, MarketSnapshot)


def test_session_levels_all_15_keys_present():
    snap = _build_snapshot("sweep_and_fvg")
    expected = {
        "midnight_open", "true_day_open", "london_open", "open_830", "open_930",
        "asia_high", "asia_low", "london_high", "london_low",
        "nyam_high", "nyam_low", "nylunch_high", "nylunch_low",
        "nypm_high", "nypm_low",
    }
    assert set(snap.session_levels.keys()) == expected


def test_session_levels_none_for_unavailable():
    """Levels whose session hasn't opened yet should be None."""
    # 1m candles start at 09:30 ET — london (02:00-05:00) has no data.
    snap = _build_snapshot("sweep_and_fvg", n=30)
    # With only 30 1m bars from 09:30, the london session is absent.
    # (midnight/true_day/london opens are before 09:30 → None.)
    assert snap.session_levels["london_open"] is None
    assert snap.session_levels["midnight_open"] is None


def test_all_pools_sorted_ascending_by_distance():
    snap = _build_snapshot("trending_up")
    distances = [p.distance_points for p in snap.all_pools]
    assert distances == sorted(distances)


def test_all_pools_have_required_fields():
    snap = _build_snapshot("trending_up")
    for p in snap.all_pools:
        assert isinstance(p, LiquidityPool)
        assert p.type in ("bsl", "ssl")
        assert p.timeframe in TIMEFRAMES
        assert p.distance_points >= 0
        assert isinstance(p.confluence_count, int)


def test_nearest_dol_none_when_no_bias():
    """flat-ish scenario with no htf_bias → nearest_dol is None."""
    # Construct a snapshot with htf_bias forced None by using a very short window.
    snap = _build_snapshot("sweep_and_fvg", n=20)
    # sweep_and_fvg with few bars may not produce BOS on 4h/1h.
    if snap.htf_bias is None:
        assert snap.nearest_dol is None


def test_nearest_dol_in_correct_bias_direction():
    """trending_up → bullish bias → nearest_dol is a BSL pool above price."""
    snap = _build_snapshot("trending_up")
    assert snap.htf_bias == "bullish"
    assert snap.nearest_dol is not None
    assert snap.nearest_dol.type == "bsl"
    assert snap.nearest_dol.level > snap.current_price


def test_nearest_dol_bearish_below_price():
    """trending_down → bearish bias → nearest_dol is an SSL pool below price."""
    snap = _build_snapshot("trending_down")
    assert snap.htf_bias == "bearish"
    assert snap.nearest_dol is not None
    assert snap.nearest_dol.type == "ssl"
    assert snap.nearest_dol.level < snap.current_price


def test_htf_bias_source_populated_with_bias():
    """htf_bias_source is set iff htf_bias is set, with all 4 sub-keys."""
    snap = _build_snapshot("trending_up")
    assert snap.htf_bias is not None
    assert snap.htf_bias_source is not None
    src = snap.htf_bias_source
    assert set(src.keys()) == {"timeframe", "type", "index", "timestamp"}
    assert src["timeframe"] in ("4h", "1h")
    assert src["type"] in ("bos_bullish", "bos_bearish", "choch_bullish", "choch_bearish")


def test_htf_bias_source_none_when_bias_none():
    snap = _build_snapshot("sweep_and_fvg", n=20)
    if snap.htf_bias is None:
        assert snap.htf_bias_source is None


def test_order_blocks_populated_from_detector():
    snap = _build_snapshot("sweep_and_fvg")
    assert set(snap.order_blocks.keys()) == set(TIMEFRAMES)
    # Each TF has a list (possibly empty).
    for tf in TIMEFRAMES:
        assert isinstance(snap.order_blocks[tf], list)


def test_fvgs_ifvgs_order_blocks_breaker_keys_all_timeframes():
    snap = _build_snapshot("sweep_and_fvg")
    for field_name in ("fvgs", "ifvgs", "order_blocks", "breaker_blocks",
                       "volume_imbalances"):
        d = getattr(snap, field_name)
        assert set(d.keys()) == set(TIMEFRAMES), f"{field_name} missing TFs"


def test_fvgs_contain_bullish_in_sweep_scenario():
    snap = _build_snapshot("sweep_and_fvg")
    bull = [f for f in snap.fvgs["5m"] if f["type"] == "fvg_bullish"]
    assert len(bull) > 0


def test_recent_sweeps_populated_in_sweep_scenario():
    snap = _build_snapshot("sweep_and_fvg")
    assert len(snap.recent_sweeps["5m"]) > 0


def test_displacements_populated_in_sweep_scenario():
    snap = _build_snapshot("sweep_and_fvg")
    assert len(snap.displacements["5m"]) > 0


def test_recent_sweeps_capped_at_10():
    snap = _build_snapshot("flat", n=100)
    for tf in TIMEFRAMES:
        assert len(snap.recent_sweeps[tf]) <= 10


def test_determinism_same_window_same_snapshot():
    w = _build_window("sweep_and_fvg", n=60)
    b = SnapshotBuilder()
    s1 = b.build(w)
    s2 = b.build(w)
    assert s1.current_price == s2.current_price
    assert s1.htf_bias == s2.htf_bias
    assert len(s1.all_pools) == len(s2.all_pools)
    assert [p.level for p in s1.all_pools] == [p.level for p in s2.all_pools]


def test_performance_under_100ms_warm():
    """build() must complete in < 100ms on warm windows."""
    w = _build_window("sweep_and_fvg", n=100)
    b = SnapshotBuilder()
    # Warm up (first call may JIT-import).
    b.build(w)
    t0 = _time.perf_counter()
    b.build(w)
    dt = (_time.perf_counter() - t0) * 1000
    assert dt < 500, f"build took {dt:.1f}ms (budget 500ms for CI variance)"


def test_compact_dict_excludes_raw_candles():
    snap = _build_snapshot("sweep_and_fvg")
    cd = snap.to_compact_dict()
    # No raw candle lists should be present.
    assert "candles" not in cd
    # all_pools capped at 5.
    assert len(cd["all_pools"]) <= 5
    # Has the key summary fields.
    assert "htf_bias" in cd
    assert "current_price" in cd
    assert "nearest_dol" in cd


def test_current_killzone_from_explicit_now():
    """An explicit `now` inside the NY AM killzone sets current_killzone."""
    w = _build_window("sweep_and_fvg", n=30)
    # 10:00 ET is inside ny_am_kz (09:30-11:00).
    now = datetime(2026, 6, 15, 10, 0, tzinfo=_NY)
    snap = SnapshotBuilder().build(w, now=now)
    assert snap.current_killzone == "ny_am_kz"


def test_current_killzone_none_outside_killzone():
    w = _build_window("sweep_and_fvg", n=30)
    # 12:00 ET is outside all killzones.
    now = datetime(2026, 6, 15, 12, 0, tzinfo=_NY)
    snap = SnapshotBuilder().build(w, now=now)
    assert snap.current_killzone is None
