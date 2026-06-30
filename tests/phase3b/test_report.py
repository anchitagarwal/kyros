"""test_report.py — honesty diagnostics: overfitting warning, tuned-vs-baseline,
parameter stability, disclaimers, degenerate folds, deterministic markdown."""

import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from trading.config import TradingConfig
from tuning.params import ALL, PostLLMParams, default_post_params
from tuning.report import WalkForwardReport
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
    base = datetime.fromisoformat(start_date).replace(tzinfo=_NY)
    traces = []
    for d in range(n_days):
        for k in range(per_day):
            ts = base + timedelta(days=d, hours=k)
            traces.append(_trace(ts.isoformat()))
    return traces


# ── Mandatory sections present ────────────────────────────────────────────────


def test_report_has_all_six_sections(tmp_path):
    traces = _traces_over_days("2026-06-01", 8, per_day=3)
    folds = make_folds(traces, train_days=3, test_days=2, step_days=2)
    baseline_hash = TradingConfig().config_hash()
    grid = [default_post_params()]
    result = run_walkforward({baseline_hash: traces}, folds, grid, min_trades=1)
    md = WalkForwardReport.generate(result, out_path=str(tmp_path / "wf.md"))

    assert "# Kyros Walk-Forward Tuning Report" in md
    assert "## 1. Per-Fold Results" in md
    assert "## 2. Aggregate OOS: Tuned vs Baseline" in md
    assert "## 3. Parameter Stability Across Folds" in md
    assert "## 4. Overfitting Assessment" in md
    assert "## 5. Disclaimers" in md
    assert "## 6. Degenerate Folds" in md


def test_report_written_to_file(tmp_path):
    traces = _traces_over_days("2026-06-01", 8, per_day=3)
    folds = make_folds(traces, train_days=3, test_days=2, step_days=2)
    baseline_hash = TradingConfig().config_hash()
    grid = [default_post_params()]
    result = run_walkforward({baseline_hash: traces}, folds, grid, min_trades=1)
    out = tmp_path / "walkforward_report.md"
    md = WalkForwardReport.generate(result, out_path=str(out))
    assert out.exists()
    assert out.read_text() == md


# ── Overfitting warning: tuned OOS ≤ baseline ─────────────────────────────────


def test_overfitting_warning_when_tuned_le_baseline(tmp_path):
    """When tuned OOS ≤ baseline OOS, the report says 'Tuning added nothing'."""
    # Construct a result directly where tuned OOS == baseline OOS (grid is
    # baseline-only → tuned == baseline → tuned ≤ baseline holds).
    traces = _traces_over_days("2026-06-01", 8, per_day=3)
    folds = make_folds(traces, train_days=3, test_days=2, step_days=2)
    baseline_hash = TradingConfig().config_hash()
    grid = [default_post_params()]
    result = run_walkforward({baseline_hash: traces}, folds, grid, min_trades=1)
    md = WalkForwardReport.generate(result, out_path=str(tmp_path / "wf.md"))
    assert "Tuning added nothing; use baseline." in md


def test_overfitting_warning_when_is_much_greater_than_oos(tmp_path):
    """When IS ≫ OOS (gap > threshold), the overfitting warning fires."""
    # Build a synthetic WalkForwardResult with IS=2.0, OOS=0.0 (gap 2.0 > 0.5).
    fold = Fold(
        train=[_trace("2026-06-01T10:00:00-04:00")],
        test=[_trace("2026-06-04T10:00:00-04:00")],
        train_start="2026-06-01T00:00:00+00:00",
        train_end="2026-06-04T00:00:00+00:00",
        test_start="2026-06-04T00:00:00+00:00",
        test_end="2026-06-05T00:00:00+00:00",
    )
    fr = FoldResult(
        fold=fold,
        chosen_config=TradingConfig(),
        chosen_params=default_post_params(),
        is_expectancy=2.0,
        oos_expectancy=0.0,
        oos_metrics={"filled_count": 1, "win_rate": 0.0, "profit_factor": "n/a",
                     "max_drawdown_r": 0.0},
        baseline_oos_expectancy=0.0,
        baseline_oos_metrics={"filled_count": 1, "win_rate": 0.0, "profit_factor": "n/a",
                              "max_drawdown_r": 0.0},
    )
    result = WalkForwardResult(folds=[fr], baseline_config_hash=TradingConfig().config_hash())
    md = WalkForwardReport.generate(result, out_path=str(tmp_path / "wf.md"))
    assert "OVERFITTING WARNING" in md


def test_no_overfitting_warning_when_tuned_beats_baseline(tmp_path):
    """When tuned OOS > baseline OOS and IS-OOS gap is small, no warning."""
    fold = Fold(
        train=[_trace("2026-06-01T10:00:00-04:00")],
        test=[_trace("2026-06-04T10:00:00-04:00")],
        train_start="2026-06-01T00:00:00+00:00",
        train_end="2026-06-04T00:00:00+00:00",
        test_start="2026-06-04T00:00:00+00:00",
        test_end="2026-06-05T00:00:00+00:00",
    )
    fr = FoldResult(
        fold=fold,
        chosen_config=TradingConfig(),
        chosen_params=default_post_params(),
        is_expectancy=0.5,
        oos_expectancy=0.4,   # tuned
        oos_metrics={"filled_count": 1, "win_rate": 1.0, "profit_factor": "inf",
                     "max_drawdown_r": 0.0},
        baseline_oos_expectancy=0.1,  # baseline < tuned
        baseline_oos_metrics={"filled_count": 1, "win_rate": 0.5, "profit_factor": 1.0,
                              "max_drawdown_r": 0.0},
    )
    result = WalkForwardResult(folds=[fr], baseline_config_hash=TradingConfig().config_hash())
    md = WalkForwardReport.generate(result, out_path=str(tmp_path / "wf.md"))
    assert "No overfitting warning triggered" in md
    assert "Tuning added nothing" not in md


# ── Parameter stability ───────────────────────────────────────────────────────


def test_parameter_stability_unstable_flag(tmp_path):
    """When chosen params differ wildly across folds, the UNSTABLE flag fires."""
    fold = Fold(
        train=[_trace("2026-06-01T10:00:00-04:00")],
        test=[_trace("2026-06-04T10:00:00-04:00")],
        train_start="2026-06-01T00:00:00+00:00",
        train_end="2026-06-04T00:00:00+00:00",
        test_start="2026-06-04T00:00:00+00:00",
        test_end="2026-06-05T00:00:00+00:00",
    )
    # Two folds with different conviction_min winners → unstable.
    fr1 = FoldResult(fold, TradingConfig(),
                     PostLLMParams(40, 1.0, ALL, ALL), 0.5, 0.4,
                     {"filled_count": 1, "win_rate": 1.0, "profit_factor": "inf", "max_drawdown_r": 0.0},
                     0.1, {"filled_count": 1, "win_rate": 0.5, "profit_factor": 1.0, "max_drawdown_r": 0.0})
    fr2 = FoldResult(fold, TradingConfig(),
                     PostLLMParams(80, 1.0, ALL, ALL), 0.5, 0.4,
                     {"filled_count": 1, "win_rate": 1.0, "profit_factor": "inf", "max_drawdown_r": 0.0},
                     0.1, {"filled_count": 1, "win_rate": 0.5, "profit_factor": 1.0, "max_drawdown_r": 0.0})
    result = WalkForwardResult(folds=[fr1, fr2], baseline_config_hash=TradingConfig().config_hash())
    md = WalkForwardReport.generate(result, out_path=str(tmp_path / "wf.md"))
    assert "UNSTABLE" in md


def test_parameter_stability_stable_when_consistent(tmp_path):
    """When the same params win every fold, no UNSTABLE flag."""
    traces = _traces_over_days("2026-06-01", 8, per_day=3)
    folds = make_folds(traces, train_days=3, test_days=2, step_days=2)
    baseline_hash = TradingConfig().config_hash()
    grid = [default_post_params()]  # only one choice → always stable
    result = run_walkforward({baseline_hash: traces}, folds, grid, min_trades=1)
    md = WalkForwardReport.generate(result, out_path=str(tmp_path / "wf.md"))
    # conviction_min is 40 in every fold → stable.
    assert "UNSTABLE" not in md


# ── Disclaimers always present ────────────────────────────────────────────────


def test_disclaimer_present(tmp_path):
    fold = Fold(
        train=[_trace("2026-06-01T10:00:00-04:00")],
        test=[_trace("2026-06-04T10:00:00-04:00")],
        train_start="2026-06-01T00:00:00+00:00",
        train_end="2026-06-04T00:00:00+00:00",
        test_start="2026-06-04T00:00:00+00:00",
        test_end="2026-06-05T00:00:00+00:00",
    )
    fr = FoldResult(fold, TradingConfig(), default_post_params(), 0.5, 0.4,
                    {"filled_count": 1, "win_rate": 1.0, "profit_factor": "inf", "max_drawdown_r": 0.0},
                    0.1, {"filled_count": 1, "win_rate": 0.5, "profit_factor": 1.0, "max_drawdown_r": 0.0})
    result = WalkForwardResult(folds=[fr], baseline_config_hash=TradingConfig().config_hash())
    md = WalkForwardReport.generate(result, out_path=str(tmp_path / "wf.md"))
    assert "## 5. Disclaimers" in md
    assert "SIMULATION" in md
    assert "Past performance" in md


def test_leakage_note_present(tmp_path):
    fold = Fold(
        train=[_trace("2026-06-01T10:00:00-04:00")],
        test=[_trace("2026-06-04T10:00:00-04:00")],
        train_start="2026-06-01T00:00:00+00:00",
        train_end="2026-06-04T00:00:00+00:00",
        test_start="2026-06-04T00:00:00+00:00",
        test_end="2026-06-05T00:00:00+00:00",
    )
    fr = FoldResult(fold, TradingConfig(), default_post_params(), 0.5, 0.4,
                    {"filled_count": 1, "win_rate": 1.0, "profit_factor": "inf", "max_drawdown_r": 0.0},
                    0.1, {"filled_count": 1, "win_rate": 0.5, "profit_factor": 1.0, "max_drawdown_r": 0.0})
    result = WalkForwardResult(folds=[fr], baseline_config_hash=TradingConfig().config_hash())
    md = WalkForwardReport.generate(result, out_path=str(tmp_path / "wf.md"))
    assert "LLM Training-Data Leakage Note" in md
    assert "training data overlaps" in md


# ── Degenerate folds ──────────────────────────────────────────────────────────


def test_degenerate_fold_flagged(tmp_path):
    """A fold with -inf OOS (below min_trades) is flagged in section 6."""
    fold = Fold(
        train=[_trace("2026-06-01T10:00:00-04:00")],
        test=[_trace("2026-06-04T10:00:00-04:00")],
        train_start="2026-06-01T00:00:00+00:00",
        train_end="2026-06-04T00:00:00+00:00",
        test_start="2026-06-04T00:00:00+00:00",
        test_end="2026-06-05T00:00:00+00:00",
    )
    fr = FoldResult(fold, TradingConfig(), default_post_params(),
                    0.5, -math.inf,  # OOS degenerate
                    {"filled_count": 0, "win_rate": 0.0, "profit_factor": "n/a", "max_drawdown_r": 0.0},
                    0.1, {"filled_count": 1, "win_rate": 0.5, "profit_factor": 1.0, "max_drawdown_r": 0.0})
    result = WalkForwardResult(folds=[fr], baseline_config_hash=TradingConfig().config_hash())
    md = WalkForwardReport.generate(result, out_path=str(tmp_path / "wf.md"))
    assert "below min_trades" in md


def test_no_degenerate_note_when_all_folds_healthy(tmp_path):
    traces = _traces_over_days("2026-06-01", 8, per_day=3)
    folds = make_folds(traces, train_days=3, test_days=2, step_days=2)
    baseline_hash = TradingConfig().config_hash()
    grid = [default_post_params()]
    result = run_walkforward({baseline_hash: traces}, folds, grid, min_trades=1)
    md = WalkForwardReport.generate(result, out_path=str(tmp_path / "wf.md"))
    assert "No degenerate folds" in md


# ── Per-fold table completeness ───────────────────────────────────────────────


def test_every_fold_in_table(tmp_path):
    traces = _traces_over_days("2026-06-01", 10, per_day=3)
    folds = make_folds(traces, train_days=3, test_days=2, step_days=2)
    baseline_hash = TradingConfig().config_hash()
    grid = [default_post_params(), PostLLMParams(60, 1.0, ALL, ALL)]
    result = run_walkforward({baseline_hash: traces}, folds, grid, min_trades=1)
    md = WalkForwardReport.generate(result, out_path=str(tmp_path / "wf.md"))
    # Each fold appears as a numbered row.
    for i in range(1, len(result.folds) + 1):
        assert f"| {i} |" in md


# ── Deterministic markdown ────────────────────────────────────────────────────


def test_deterministic_markdown(tmp_path):
    """The same result produces byte-identical markdown across two runs."""
    traces = _traces_over_days("2026-06-01", 8, per_day=3)
    folds = make_folds(traces, train_days=3, test_days=2, step_days=2)
    baseline_hash = TradingConfig().config_hash()
    grid = [default_post_params(), PostLLMParams(60, 1.0, ALL, ALL)]
    result = run_walkforward({baseline_hash: traces}, folds, grid, min_trades=1)
    md1 = WalkForwardReport.generate(result, out_path=str(tmp_path / "wf1.md"))
    md2 = WalkForwardReport.generate(result, out_path=str(tmp_path / "wf2.md"))
    assert md1 == md2


# ── Aggregate reports both mean-of-folds and trade-weighted ───────────────────


def test_aggregate_reports_both_methods(tmp_path):
    traces = _traces_over_days("2026-06-01", 8, per_day=3)
    folds = make_folds(traces, train_days=3, test_days=2, step_days=2)
    baseline_hash = TradingConfig().config_hash()
    grid = [default_post_params()]
    result = run_walkforward({baseline_hash: traces}, folds, grid, min_trades=1)
    md = WalkForwardReport.generate(result, out_path=str(tmp_path / "wf.md"))
    assert "mean-of-folds" in md
    assert "trade-weighted" in md
