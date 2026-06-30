"""test_objective.py — expectancy matches PerformanceReport; below MIN_TRADES
returns -inf; metrics always populated."""

import math

import pytest

from backtesting.report import PerformanceReport
from tuning.objective import MIN_TRADES, evaluate
from tuning.params import ALL, PostLLMParams, default_post_params
from tuning.rescore import rescore_traces


def _trace(timestamp, bias="long", model="2022", killzone="ny_am_kz",
           result="win", actual_rr=2.0, conviction=70,
           entry_zone=(100.0, 100.0), stop=95.0, target=110.0):
    return {
        "trace_id": f"{timestamp}_{bias}",
        "timestamp": timestamp,
        "instrument": "NQ",
        "killzone": killzone,
        "trigger_reason": "fvg",
        "snapshot_summary": {"instrument": "NQ"},
        "raw_llm_output": "{}",
        "alert": {
            "bias": bias, "model": model, "conviction": conviction,
            "entry_zone": list(entry_zone), "stop": stop, "target": target,
            "dol": {}, "risk_reward": 0.0, "rationale": "",
            "killzone": killzone, "valid_until": "", "no_trade_reason": None,
        },
        "rr_validated": True,
        "outcome": {"result": result, "actual_rr": actual_rr,
                    "fill_price": 100.0, "exit_price": 110.0,
                    "candles_to_fill": 1, "candles_to_resolution": 2},
    }


KEEPING = PostLLMParams(conviction_min=40, rr_min=1.0, allowed_models=ALL, allowed_killzones=ALL)


# ── Expectancy matches PerformanceReport ──────────────────────────────────────


def test_expectancy_matches_performance_report():
    """evaluate's score == PerformanceReport._overall_metrics(rescored)['expectancy']."""
    traces = [
        _trace("2026-06-01T10:00:00-04:00", result="win", actual_rr=2.0),
        _trace("2026-06-01T11:00:00-04:00", result="win", actual_rr=1.5),
        _trace("2026-06-01T12:00:00-04:00", result="loss", actual_rr=-1.0),
        _trace("2026-06-01T13:00:00-04:00", result="no_trade", actual_rr=None),
    ]
    score, metrics = evaluate(traces, KEEPING, min_trades=1)
    # PerformanceReport's expectancy over the re-scored set (keeping → unchanged).
    pr_expectancy = PerformanceReport()._overall_metrics(
        rescore_traces(traces, KEEPING))["expectancy"]
    assert score == pr_expectancy
    # Hand check: (2.0 + 1.5 - 1.0 + 0) / 4 = 0.625
    assert score == 0.625


def test_expectancy_with_filtering():
    """Filtering a win to no_trade (R=0) lowers expectancy."""
    traces = [
        _trace("2026-06-01T10:00:00-04:00", result="win", actual_rr=2.0, conviction=70),
        _trace("2026-06-01T11:00:00-04:00", result="win", actual_rr=1.5, conviction=50),
    ]
    # Keeping: (2.0 + 1.5)/2 = 1.75
    score_keep, _ = evaluate(traces, KEEPING, min_trades=1)
    assert score_keep == 1.75
    # Filter out the conviction-50 trade: (2.0 + 0)/2 = 1.0
    p = PostLLMParams(60, 1.0, ALL, ALL)
    score_filt, _ = evaluate(traces, p, min_trades=1)
    assert score_filt == 1.0


def test_taken_trades_counted_on_rescored_set():
    """taken_trades (filled_count) is on the RE-SCORED set, not the raw set."""
    traces = [
        _trace("2026-06-01T10:00:00-04:00", result="win", actual_rr=2.0, conviction=70),
        _trace("2026-06-01T11:00:00-04:00", result="win", actual_rr=1.5, conviction=50),
        _trace("2026-06-01T12:00:00-04:00", result="loss", actual_rr=-1.0, conviction=30),
    ]
    # Raw: 3 taken. Filter conviction>=40 → 2 taken (the conviction-30 loss is
    # downgraded to no_trade).
    p = PostLLMParams(40, 1.0, ALL, ALL)
    _, metrics = evaluate(traces, p, min_trades=1)
    assert metrics["filled_count"] == 2


# ── MIN_TRADES guard ──────────────────────────────────────────────────────────


def test_below_min_trades_returns_neg_inf():
    traces = [
        _trace("2026-06-01T10:00:00-04:00", result="win", actual_rr=2.0),
        _trace("2026-06-01T11:00:00-04:00", result="win", actual_rr=1.5),
    ]
    # 2 taken trades < min_trades=5 → -inf.
    score, metrics = evaluate(traces, KEEPING, min_trades=5)
    assert score == -math.inf
    # Metrics still populated.
    assert metrics["filled_count"] == 2
    assert "expectancy" in metrics


def test_at_min_trades_is_finite():
    """A score is finite at EXACTLY min_trades (not -inf)."""
    traces = [
        _trace("2026-06-01T10:00:00-04:00", result="win", actual_rr=2.0),
        _trace("2026-06-01T11:00:00-04:00", result="win", actual_rr=1.5),
    ]
    score, _ = evaluate(traces, KEEPING, min_trades=2)
    assert score != -math.inf
    assert score == 1.75


def test_metrics_populated_even_on_neg_inf():
    traces = [_trace("2026-06-01T10:00:00-04:00", result="win", actual_rr=2.0)]
    score, metrics = evaluate(traces, KEEPING, min_trades=10)
    assert score == -math.inf
    assert metrics["total_traces"] == 1
    assert metrics["filled_count"] == 1


def test_default_min_trades_constant():
    assert MIN_TRADES == 10


# ── Determinism ───────────────────────────────────────────────────────────────


def test_evaluate_deterministic():
    traces = [
        _trace("2026-06-01T10:00:00-04:00", result="win", actual_rr=2.0),
        _trace("2026-06-01T11:00:00-04:00", result="loss", actual_rr=-1.0),
    ]
    s1, m1 = evaluate(traces, KEEPING, min_trades=1)
    s2, m2 = evaluate(traces, KEEPING, min_trades=1)
    assert s1 == s2
    assert m1 == m2


# ── Default params sanity tie-out ─────────────────────────────────────────────


def test_default_params_baseline_no_filter():
    """default_post_params() is a no-op re-score (ALL on both axes, config
    defaults) — expectancy equals the raw PerformanceReport expectancy."""
    traces = [
        _trace("2026-06-01T10:00:00-04:00", result="win", actual_rr=2.0, conviction=70),
        _trace("2026-06-01T11:00:00-04:00", result="loss", actual_rr=-1.0, conviction=60),
        _trace("2026-06-01T12:00:00-04:00", result="no_trade", actual_rr=None),
    ]
    p = default_post_params()
    score, _ = evaluate(traces, p, min_trades=1)
    raw_expectancy = PerformanceReport()._overall_metrics(traces)["expectancy"]
    assert score == raw_expectancy
