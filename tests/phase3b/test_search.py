"""test_search.py — best_params picks the argmax; deterministic tie-break;
all-below-min_trades fallback."""

import math

import pytest

from tuning.params import ALL, PostLLMParams, default_post_params
from tuning.search import best_params


def _trace(timestamp, bias="long", conviction=70, actual_rr=2.0, result="win",
           model="2022", killzone="ny_am_kz"):
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
            "entry_zone": [100.0, 100.0], "stop": 95.0, "target": 110.0,
            "dol": {}, "risk_reward": 0.0, "rationale": "",
            "killzone": killzone, "valid_until": "", "no_trade_reason": None,
        },
        "rr_validated": True,
        "outcome": {"result": result, "actual_rr": actual_rr,
                    "fill_price": 100.0, "exit_price": 110.0,
                    "candles_to_fill": 1, "candles_to_resolution": 2},
    }


# ── Argmax ────────────────────────────────────────────────────────────────────


def test_best_params_picks_highest_objective():
    """One param set filters out a loss → higher expectancy → chosen."""
    traces = [
        _trace("2026-06-01T10:00:00-04:00", conviction=70, actual_rr=2.0, result="win"),
        _trace("2026-06-01T11:00:00-04:00", conviction=50, actual_rr=-1.0, result="loss"),
        _trace("2026-06-01T12:00:00-04:00", conviction=80, actual_rr=1.5, result="win"),
    ]
    # p_keep (cv>=40): keeps all 3 → (2.0 - 1.0 + 1.5)/3 = 0.8333
    # p_strict (cv>=60): drops the conviction-50 loss → (2.0 + 0 + 1.5)/3 = 1.1667
    p_keep = PostLLMParams(40, 1.0, ALL, ALL)
    p_strict = PostLLMParams(60, 1.0, ALL, ALL)
    grid = [p_keep, p_strict]
    best, score, metrics = best_params(traces, grid, min_trades=1)
    assert best is p_strict
    assert score == pytest.approx((2.0 + 0.0 + 1.5) / 3, abs=1e-4)


def test_best_params_returns_score_and_metrics():
    traces = [
        _trace("2026-06-01T10:00:00-04:00", conviction=70, actual_rr=2.0),
        _trace("2026-06-01T11:00:00-04:00", conviction=70, actual_rr=1.5),
    ]
    p = PostLLMParams(40, 1.0, ALL, ALL)
    best, score, metrics = best_params(traces, [p], min_trades=1)
    assert best is p
    assert score == 1.75
    assert metrics["filled_count"] == 2


# ── Deterministic tie-break (first in grid order) ─────────────────────────────


def test_tie_break_first_in_grid_order():
    """Two params with equal score → the FIRST in grid order wins."""
    traces = [
        _trace("2026-06-01T10:00:00-04:00", conviction=70, actual_rr=2.0),
        _trace("2026-06-01T11:00:00-04:00", conviction=70, actual_rr=2.0),
    ]
    # Both params keep all trades (cv 40 and 60 both <= 70) → same expectancy.
    p_a = PostLLMParams(40, 1.0, ALL, ALL)
    p_b = PostLLMParams(60, 1.0, ALL, ALL)
    best, score, _ = best_params(traces, [p_a, p_b], min_trades=1)
    assert best is p_a  # first in grid order
    best_rev, _, _ = best_params(traces, [p_b, p_a], min_trades=1)
    assert best_rev is p_b  # order matters for ties


def test_tie_break_deterministic_across_runs():
    traces = [
        _trace("2026-06-01T10:00:00-04:00", conviction=70, actual_rr=2.0),
        _trace("2026-06-01T11:00:00-04:00", conviction=70, actual_rr=2.0),
    ]
    p_a = PostLLMParams(40, 1.0, ALL, ALL)
    p_b = PostLLMParams(60, 1.0, ALL, ALL)
    grid = [p_a, p_b]
    b1, s1, _ = best_params(traces, grid, min_trades=1)
    b2, s2, _ = best_params(traces, grid, min_trades=1)
    assert b1 is b2
    assert s1 == s2


# ── All-below-min_trades fallback ─────────────────────────────────────────────


def test_all_below_min_trades_returns_baseline():
    """When every grid point is below min_trades, return the baseline params."""
    traces = [
        _trace("2026-06-01T10:00:00-04:00", conviction=70, actual_rr=2.0),
    ]
    # Only 1 taken trade; min_trades=5 → every grid point is -inf.
    p1 = PostLLMParams(40, 1.0, ALL, ALL)
    p2 = PostLLMParams(60, 1.0, ALL, ALL)
    best, score, metrics = best_params(traces, [p1, p2], min_trades=5)
    # Fallback returns the baseline (default_post_params).
    assert best == default_post_params()
    assert score == -math.inf
    assert metrics["filled_count"] == 1


def test_all_below_min_trades_baseline_in_grid():
    """If the baseline is in the grid and all are -inf, the fallback still
    returns the baseline explicitly (with its metrics)."""
    traces = [_trace("2026-06-01T10:00:00-04:00", conviction=70, actual_rr=2.0)]
    baseline = default_post_params()
    best, score, _ = best_params(traces, [baseline], min_trades=5)
    assert best == baseline
    assert score == -math.inf


# ── Pure function ─────────────────────────────────────────────────────────────


def test_best_params_never_returns_unseen_params():
    """The returned params must be from the grid (or the documented baseline
    fallback)."""
    traces = [
        _trace("2026-06-01T10:00:00-04:00", conviction=70, actual_rr=2.0),
        _trace("2026-06-01T11:00:00-04:00", conviction=70, actual_rr=1.5),
    ]
    p = PostLLMParams(40, 1.0, ALL, ALL)
    best, _, _ = best_params(traces, [p], min_trades=1)
    assert best is p


def test_empty_grid_returns_baseline():
    """An empty grid falls back to the baseline (never None)."""
    traces = [_trace("2026-06-01T10:00:00-04:00", conviction=70, actual_rr=2.0)]
    best, _, _ = best_params(traces, [], min_trades=1)
    assert best == default_post_params()
