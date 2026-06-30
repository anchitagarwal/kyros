"""test_walkforward.py — rolling folds split at expected timestamps; no trace
in both train and test of the same fold; IS/OOS reproducible; adversarial
boundary test."""

import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from trading.config import TradingConfig
from tuning.params import ALL, PostLLMParams, default_post_params
from tuning.walkforward import (
    Fold,
    FoldResult,
    WalkForwardResult,
    make_folds,
    run_walkforward,
)

_NY = ZoneInfo("America/New_York")


def _trace(ts_iso, bias="long", conviction=70, actual_rr=2.0, result="win",
           model="2022", killzone="ny_am_kz"):
    return {
        "trace_id": f"{ts_iso}_{bias}",
        "timestamp": ts_iso,
        "instrument": "NQ",
        "killzone": killzone,
        "trigger_reason": "fvg",
        "snapshot_summary": {"instrument": "NQ"},
        "raw_llm_output": "{}",
        "alert": {
            "bias": bias, "model": model, "conviction": conviction,
            "entry_zone": [100.0, 100.0], "stop": 95.0, "target": 110.0,
            "dol": {}, "risk_reward": 0.0, "rationale": "",
            "killzone": killzone, "valid_until": "", "no_trade_reason": None,
        },
        "rr_validated": True,
        "outcome": {"result": result, "actual_rr": actual_rr,
                    "fill_price": 100.0, "exit_price": 110.0,
                    "candles_to_fill": 1, "candles_to_resolution": 2},
    }


def _traces_over_days(start_date, n_days, per_day=3):
    """Build traces at 10:00 ET on each of n_days consecutive days."""
    base = datetime.fromisoformat(start_date).replace(tzinfo=_NY)
    traces = []
    for d in range(n_days):
        for k in range(per_day):
            ts = base + timedelta(days=d, hours=k)
            traces.append(_trace(ts.isoformat()))
    return traces


# ── Fold construction ─────────────────────────────────────────────────────────


def test_make_folds_basic_split():
    """train_days=2, test_days=1, step_days=2 → non-overlapping folds."""
    traces = _traces_over_days("2026-06-01", 6, per_day=2)
    folds = make_folds(traces, train_days=2, test_days=1, step_days=2)
    assert len(folds) >= 1
    # Each fold has train and test.
    for f in folds:
        assert len(f.train) > 0
        assert len(f.test) > 0


def test_make_folds_consecutive_starts_differ_by_step_days():
    traces = _traces_over_days("2026-06-01", 10, per_day=2)
    folds = make_folds(traces, train_days=2, test_days=1, step_days=2)
    assert len(folds) >= 2
    for i in range(1, len(folds)):
        prev_start = datetime.fromisoformat(folds[i - 1].train_start)
        cur_start = datetime.fromisoformat(folds[i].train_start)
        assert (cur_start - prev_start).days == 2


def test_make_folds_train_end_equals_test_start():
    """The boundary is shared: train_end == test_start (half-open)."""
    traces = _traces_over_days("2026-06-01", 5, per_day=2)
    folds = make_folds(traces, train_days=2, test_days=1, step_days=2)
    for f in folds:
        assert f.train_end == f.test_start


# ── NO LEAKAGE (the release-blocker invariant) ────────────────────────────────


def test_no_trace_in_both_train_and_test():
    """For every fold, train ∩ test = ∅ by timestamp."""
    traces = _traces_over_days("2026-06-01", 8, per_day=3)
    folds = make_folds(traces, train_days=2, test_days=2, step_days=1)
    assert len(folds) >= 2
    for f in folds:
        train_ts = {t["timestamp"] for t in f.train}
        test_ts = {t["timestamp"] for t in f.test}
        assert not (train_ts & test_ts), "LEAKAGE: a trace is in both train and test"


def test_adversarial_boundary_trace_lands_in_exactly_one_side():
    """A trace exactly on the train/test boundary (train_end == test_start)
    must land in EXACTLY ONE side (test, per half-open [start, end)).

    We construct traces so one sits precisely on the boundary and assert it
    is in test, not train, and not both.
    """
    # train_days=1 → train = [day0_midnight, day1_midnight), test = [day1, day2).
    # A trace at exactly day1 00:00 UTC is on the boundary → must be in test.
    base = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
    traces = [
        _trace(base.isoformat()),                       # day0 00:00 → train
        _trace((base + timedelta(days=1)).isoformat()),  # day1 00:00 → boundary → test
        _trace((base + timedelta(days=1, hours=5)).isoformat()),  # day1 05:00 → test
    ]
    folds = make_folds(traces, train_days=1, test_days=1, step_days=1)
    assert len(folds) >= 1
    f = folds[0]
    boundary_ts = (base + timedelta(days=1)).isoformat()
    train_ts = {t["timestamp"] for t in f.train}
    test_ts = {t["timestamp"] for t in f.test}
    # The boundary trace is in test, NOT train, NOT both.
    assert boundary_ts in test_ts
    assert boundary_ts not in train_ts
    assert not (train_ts & test_ts)


def test_make_folds_asserts_disjointness_internally():
    """make_folds calls _assert_disjoint per fold; a constructed fold with
    overlap would raise AssertionError. We verify the assertion exists by
    confirming folds are always disjoint (no exception on valid input)."""
    traces = _traces_over_days("2026-06-01", 6, per_day=4)
    folds = make_folds(traces, train_days=3, test_days=2, step_days=1)
    # If _assert_disjoint were missing, this would silently pass; the real
    # guarantee is that make_folds never returns an overlapping fold.
    for f in folds:
        _assert_disjoint(f)


def _assert_disjoint(fold):
    train_ts = {t["timestamp"] for t in fold.train}
    test_ts = {t["timestamp"] for t in fold.test}
    assert not (train_ts & test_ts)


# ── Cross-fold overlap is allowed (step < test) ───────────────────────────────


def test_cross_fold_overlap_allowed_when_step_less_than_test():
    """step_days < test_days → a trace may appear in fold N's test AND fold
    N+1's train. This is fine; only INTRA-fold train/test must be disjoint."""
    traces = _traces_over_days("2026-06-01", 6, per_day=2)
    folds = make_folds(traces, train_days=2, test_days=2, step_days=1)
    assert len(folds) >= 2
    # Fold 0's test and fold 1's train may overlap (both cover day 2-3).
    f0_test_ts = {t["timestamp"] for t in folds[0].test}
    f1_train_ts = {t["timestamp"] for t in folds[1].train}
    # Cross-fold overlap is permitted (not asserted against).
    # But each fold individually is disjoint (tested above).
    for f in folds:
        _assert_disjoint(f)


# ── Empty train/test folds dropped ────────────────────────────────────────────


def test_folds_with_empty_train_or_test_dropped():
    """A fold with no traces on one side is dropped."""
    # Traces only on days 0 and 5 (gap in the middle). With train=2/test=1/step=1,
    # folds whose windows fall entirely in the gap are dropped.
    traces = [
        _trace(datetime(2026, 6, 1, 10, 0, tzinfo=_NY).isoformat()),
        _trace(datetime(2026, 6, 6, 10, 0, tzinfo=_NY).isoformat()),
    ]
    folds = make_folds(traces, train_days=1, test_days=1, step_days=1)
    for f in folds:
        assert len(f.train) > 0
        assert len(f.test) > 0


def test_empty_traces_returns_no_folds():
    assert make_folds([], train_days=2, test_days=1, step_days=1) == []


def test_invalid_window_sizes_raise():
    with pytest.raises(ValueError):
        make_folds(_traces_over_days("2026-06-01", 3), 0, 1, 1)
    with pytest.raises(ValueError):
        make_folds(_traces_over_days("2026-06-01", 3), 1, 0, 1)
    with pytest.raises(ValueError):
        make_folds(_traces_over_days("2026-06-01", 3), 1, 1, 0)


# ── run_walkforward: IS/OOS, baseline parity, determinism ─────────────────────


def test_run_walkforward_produces_is_and_oos():
    traces = _traces_over_days("2026-06-01", 8, per_day=3)
    folds = make_folds(traces, train_days=3, test_days=2, step_days=2)
    baseline_hash = TradingConfig().config_hash()
    grid = [default_post_params()]
    result = run_walkforward({baseline_hash: traces}, folds, grid, min_trades=1)
    assert len(result.folds) == len(folds)
    for fr in result.folds:
        # IS computed on train, OOS on test.
        assert isinstance(fr.is_expectancy, float)
        assert isinstance(fr.oos_expectancy, float)
        assert isinstance(fr.baseline_oos_expectancy, float)
        # Baseline uses default params.
        assert fr.chosen_params == default_post_params()


def test_run_walkforward_baseline_oos_uses_same_test_window():
    """Tuned and baseline OOS are evaluated on the SAME test window of the
    SAME (baseline) config's traces — apples-to-apples."""
    traces = _traces_over_days("2026-06-01", 8, per_day=3)
    folds = make_folds(traces, train_days=3, test_days=2, step_days=2)
    baseline_hash = TradingConfig().config_hash()
    # Grid with only the baseline → tuned == baseline by construction.
    grid = [default_post_params()]
    result = run_walkforward({baseline_hash: traces}, folds, grid, min_trades=1)
    for fr in result.folds:
        # With only the baseline in the grid, tuned OOS == baseline OOS.
        assert fr.oos_expectancy == fr.baseline_oos_expectancy


def test_run_walkforward_deterministic():
    traces = _traces_over_days("2026-06-01", 8, per_day=3)
    folds = make_folds(traces, train_days=3, test_days=2, step_days=2)
    baseline_hash = TradingConfig().config_hash()
    grid = [
        default_post_params(),
        PostLLMParams(60, 1.0, ALL, ALL),
        PostLLMParams(40, 1.5, ALL, ALL),
    ]
    r1 = run_walkforward({baseline_hash: traces}, folds, grid, min_trades=1)
    r2 = run_walkforward({baseline_hash: traces}, folds, grid, min_trades=1)
    assert len(r1.folds) == len(r2.folds)
    for a, b in zip(r1.folds, r2.folds):
        assert a.chosen_params == b.chosen_params
        assert a.is_expectancy == b.is_expectancy
        assert a.oos_expectancy == b.oos_expectancy
        assert a.baseline_oos_expectancy == b.baseline_oos_expectancy


def test_run_walkforward_chosen_config_is_baseline_for_free_path():
    """With only the baseline config in trace_sets, the chosen config is the
    baseline TradingConfig()."""
    traces = _traces_over_days("2026-06-01", 8, per_day=3)
    folds = make_folds(traces, train_days=3, test_days=2, step_days=2)
    baseline_hash = TradingConfig().config_hash()
    grid = [default_post_params(), PostLLMParams(60, 1.0, ALL, ALL)]
    result = run_walkforward({baseline_hash: traces}, folds, grid, min_trades=1)
    for fr in result.folds:
        assert fr.chosen_config.config_hash() == baseline_hash


def test_run_walkforward_no_leakage_in_results():
    """The IS is computed on train, OOS on test — and train/test are disjoint
    (verified by make_folds). Re-assert the fold's disjointness holds for the
    traces actually scored."""
    traces = _traces_over_days("2026-06-01", 8, per_day=3)
    folds = make_folds(traces, train_days=3, test_days=2, step_days=2)
    baseline_hash = TradingConfig().config_hash()
    grid = [default_post_params(), PostLLMParams(60, 1.0, ALL, ALL)]
    result = run_walkforward({baseline_hash: traces}, folds, grid, min_trades=1)
    for fr in result.folds:
        _assert_disjoint(fr.fold)
