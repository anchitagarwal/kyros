"""test_run_tuning.py — integration test for the default (zero-LLM) path and
unit tests for the Tier-2 cost gate / per-config routing.

The default path must make ZERO LLM calls and work with NO API key. We assert
this by running run_tuning.run_default over a fixture trade_traces.jsonl and
confirming the report is produced and no engine/loader/router is invoked.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

# Ensure workspace/ is importable.
_WORKSPACE = Path(__file__).resolve().parent.parent.parent / "workspace"
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from trading.config import TradingConfig
from tuning.params import default_post_params, PostLLMParams, ALL

# scripts/ is not a package; import run_tuning by path.
_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_tuning

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


def _write_fixture_traces(path: Path, n_days: int = 10, per_day: int = 4):
    """Write a fixture trade_traces.jsonl with traces over n_days."""
    base = datetime(2026, 6, 1, 10, 0, tzinfo=_NY)
    traces = []
    for d in range(n_days):
        for k in range(per_day):
            ts = base + timedelta(days=d, hours=k)
            # Alternate win/loss/no_trade for variety.
            if k % 3 == 0:
                traces.append(_trace(ts.isoformat(), result="win", actual_rr=2.0, conviction=70))
            elif k % 3 == 1:
                traces.append(_trace(ts.isoformat(), result="loss", actual_rr=-1.0, conviction=50))
            else:
                traces.append(_trace(ts.isoformat(), result="no_trade", actual_rr=None, conviction=0))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")
    return traces


# ── Default path: zero LLM calls, no API key ──────────────────────────────────


def test_default_path_produces_report_no_api_key(tmp_path, monkeypatch):
    """The default path produces a walkforward_report.md with NO API key set
    and ZERO LLM calls."""
    # Strip all API keys.
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ZAI_API_KEY", "ALPACA_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    traces_path = tmp_path / "trade_traces.jsonl"
    _write_fixture_traces(traces_path, n_days=10, per_day=4)

    out_path = tmp_path / "walkforward_report.md"
    md = run_tuning.run_default(
        traces_path=traces_path,
        train_days=3,
        test_days=2,
        step_days=2,
        min_trades=1,
        out_path=out_path,
    )
    # Report written.
    assert out_path.exists()
    assert out_path.read_text() == md
    # Mandatory sections present.
    assert "# Kyros Walk-Forward Tuning Report" in md
    assert "## 1. Per-Fold Results" in md
    assert "## 5. Disclaimers" in md


def test_default_path_makes_zero_llm_calls(tmp_path, monkeypatch):
    """The default path never imports/invokes the LLM router, loader, or engine.

    We snapshot sys.modules before and after run_default and assert that NO
    LLM-related module (model_router, agent_loader, reasoning_agent,
    BacktestEngine, TriggerCalibrator, openai, anthropic) is newly imported
    during the default path. Completion with an empty LLM-import set proves
    zero LLM calls — the default path is purely offline arithmetic.
    """
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ZAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    traces_path = tmp_path / "trade_traces.jsonl"
    _write_fixture_traces(traces_path, n_days=10, per_day=4)

    # Snapshot modules present before the default path runs.
    before = set(sys.modules.keys())

    out_path = tmp_path / "wf.md"
    md = run_tuning.run_default(traces_path, 3, 2, 2, 1, out_path)
    assert "Kyros Walk-Forward" in md

    # Modules newly imported during the default path.
    after = set(sys.modules.keys())
    new_mods = after - before

    # The default path must not touch any LLM / Tier-2 recording module.
    llm_markers = (
        "model_router", "agent_loader", "reasoning_agent",
        "backtesting.engine", "backtesting.calibrator",
        "openai", "anthropic",
    )
    llm_imports = [m for m in new_mods if any(marker in m for marker in llm_markers)]
    assert llm_imports == [], (
        f"Default path imported LLM/Tier-2 modules (should be zero): {llm_imports}"
    )


def test_default_path_offline_with_real_traces(tmp_path, monkeypatch):
    """Integration: run_default over the repo's real trade_traces.jsonl
    produces a report with no API key (the canonical offline gate)."""
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ZAI_API_KEY", "ALPACA_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    real_traces = _WORKSPACE / "trade_traces.jsonl"
    if not real_traces.exists():
        pytest.skip("workspace/trade_traces.jsonl not present")

    out_path = tmp_path / "walkforward_report.md"
    md = run_tuning.run_default(
        traces_path=real_traces,
        train_days=2,
        test_days=1,
        step_days=1,
        min_trades=1,
        out_path=out_path,
    )
    assert out_path.exists()
    assert "Kyros Walk-Forward" in md


# ── Tier-2 cost gate ──────────────────────────────────────────────────────────


def test_tier2_decline_aborts_zero_spend(tmp_path, monkeypatch):
    """Declining the spend gate aborts with zero spend (no recording)."""
    # Mock the cost estimate to avoid running the calibrator over real data.
    monkeypatch.setattr(run_tuning, "estimate_recording_cost", lambda configs, data: 1.23)
    # Mock record_config to raise if called (proving zero spend).
    def _boom(cfg, data, runs, agent=None):
        raise AssertionError("record_config must not be called after decline")
    monkeypatch.setattr(run_tuning, "record_config", _boom)
    # Non-interactive, no --yes → should exit(1).
    monkeypatch.setattr(sys, "stdin", type("S", (), {"isatty": lambda self: False})())

    with pytest.raises(SystemExit) as exc:
        run_tuning.run_tier2(
            configs=[TradingConfig()],
            data_path=tmp_path / "data.parquet",
            runs_dir=tmp_path / "runs",
            train_days=3, test_days=2, step_days=2, min_trades=1,
            out_path=tmp_path / "wf.md",
            yes=False,
        )
    assert exc.value.code == 1


def test_tier2_cost_math(tmp_path, monkeypatch):
    """estimate_recording_cost = sum over configs of (total_fires × $0.003).

    We mock TriggerCalibrator (so no real data is read) to return a known fire
    count per config, then assert the real estimate_recording_cost function
    computes the correct total. This exercises the actual function, not a
    re-derived formula.
    """
    class FakeReport:
        def __init__(self, fires):
            self.total_fires = fires

    class FakeCal:
        """A fake calibrator that reports a fixed fire count per config."""
        def __init__(self, *a, **kw):
            pass

        def run(self, src):
            return FakeReport(100)

    class FakeSource:
        def __init__(self, *a, **kw):
            pass

    # Patch the calibrator + source that estimate_recording_cost imports
    # lazily inside the function body.
    monkeypatch.setattr("backtesting.calibrator.TriggerCalibrator", FakeCal)
    monkeypatch.setattr("trading.candle_source.ReplayCandleSource", FakeSource)

    configs = [
        TradingConfig(),
        TradingConfig(rr_min=2.0),
        TradingConfig(conviction_min=50),
    ]
    cost = run_tuning.estimate_recording_cost(configs, tmp_path / "data.parquet")
    # 3 configs × 100 fires × $0.003 = $0.90
    assert cost == pytest.approx(0.9, abs=1e-9)


def test_tier2_cost_math_scales_with_fires(tmp_path, monkeypatch):
    """Cost scales linearly with the fire count (the $0.003 multiplier)."""
    fire_counts = []

    class FakeReport:
        def __init__(self, fires):
            self.total_fires = fires

    class FakeCal:
        def __init__(self, *a, **kw):
            pass

        def run(self, src):
            # Return an incrementing fire count per config.
            n = 50 * (len(fire_counts) + 1)
            fire_counts.append(n)
            return FakeReport(n)

    class FakeSource:
        def __init__(self, *a, **kw):
            pass

    monkeypatch.setattr("backtesting.calibrator.TriggerCalibrator", FakeCal)
    monkeypatch.setattr("trading.candle_source.ReplayCandleSource", FakeSource)

    configs = [TradingConfig(), TradingConfig(rr_min=2.0)]
    cost = run_tuning.estimate_recording_cost(configs, tmp_path / "data.parquet")
    # Config 1: 50 fires, config 2: 100 fires → (50 + 100) × 0.003 = 0.45
    assert fire_counts == [50, 100]
    assert cost == pytest.approx((50 + 100) * 0.003, abs=1e-9)


def test_tier2_per_config_run_dirs_isolated(tmp_path):
    """Two configs → two disjoint run dirs keyed by config_hash."""
    cfg_a = TradingConfig()
    cfg_b = TradingConfig(rr_min=2.0)
    assert cfg_a.config_hash() != cfg_b.config_hash()
    runs_dir = tmp_path / "runs"
    dir_a = runs_dir / cfg_a.short_hash()
    dir_b = runs_dir / cfg_b.short_hash()
    assert dir_a != dir_b


def test_load_trace_sets_reads_per_config_dirs(tmp_path):
    """load_trace_sets reads runs/{hash}/trade_traces.jsonl into a dict."""
    runs = tmp_path / "runs"
    h1 = "aaaa1111"
    h2 = "bbbb2222"
    (runs / h1).mkdir(parents=True)
    (runs / h2).mkdir(parents=True)
    (runs / h1 / "trade_traces.jsonl").write_text(
        json.dumps(_trace("2026-06-01T10:00:00-04:00")) + "\n")
    (runs / h2 / "trade_traces.jsonl").write_text(
        json.dumps(_trace("2026-06-02T10:00:00-04:00")) + "\n")
    # A non-dir file and a dir without traces should be skipped.
    (runs / "stray.txt").write_text("x")
    (runs / "empty").mkdir()

    trace_sets = run_tuning.load_trace_sets(runs)
    assert set(trace_sets.keys()) == {h1, h2}
    assert len(trace_sets[h1]) == 1
    assert len(trace_sets[h2]) == 1


# ── Idempotent resume (Tier-2) ────────────────────────────────────────────────


def test_tier2_idempotent_resume_no_duplicate_fires(tmp_path, monkeypatch):
    """A second --record run with a fully-recorded config performs zero new
    fires (BacktestEngine resume skips already-ledgered timestamps)."""
    # This is a unit test of the resume contract: we mock BacktestEngine.run to
    # count calls and assert the ledger grows by 0 on the second call.
    import scripts.run_tuning as rt

    run_calls = {"n": 0}

    class FakeEngine:
        def __init__(self, loop, sim, output_path):
            self.output_path = output_path
            self.alerts_path = output_path.with_name("trade_alerts.jsonl")

        def run(self, src):
            run_calls["n"] += 1
            # Simulate resume: if the ledger exists, return existing traces.
            if self.output_path.exists():
                return [json.loads(l) for l in self.output_path.read_text().splitlines() if l.strip()]
            # First run: write a ledger + traces.
            self.alerts_path.parent.mkdir(parents=True, exist_ok=True)
            self.alerts_path.write_text(json.dumps({"timestamp": "t1", "alert": {}}) + "\n")
            self.output_path.write_text(json.dumps(_trace("2026-06-01T10:00:00-04:00")) + "\n")
            return [_trace("2026-06-01T10:00:00-04:00")]

    # Patch the BacktestEngine import inside record_config.
    monkeypatch.setattr("backtesting.engine.BacktestEngine", FakeEngine)
    monkeypatch.setattr(rt, "_build_reasoning_agent", lambda: object())
    # Patch the other imports inside record_config to no-ops.
    monkeypatch.setattr("trading.candle_source.ReplayCandleSource", lambda *a, **k: None)
    monkeypatch.setattr("trading.candle_window.CandleWindow", lambda: None)
    monkeypatch.setattr("trading.snapshot.SnapshotBuilder", lambda **k: None)
    monkeypatch.setattr("trading.trigger.TriggerEngine", lambda *a, **k: None)
    monkeypatch.setattr("trading.cooldown.CooldownState", lambda **k: None)
    monkeypatch.setattr("trading.trading_loop.TradingLoop", lambda **k: None)
    monkeypatch.setattr("backtesting.outcome.OutcomeSimulator", lambda: None)

    cfg = TradingConfig()
    runs_dir = tmp_path / "runs"
    data_path = tmp_path / "data.parquet"

    # First record.
    rt.record_config(cfg, data_path, runs_dir)
    assert run_calls["n"] == 1
    first_ledger = (runs_dir / cfg.short_hash() / "trade_alerts.jsonl").read_text()

    # Second record (resume) — should call run() once more but the FakeEngine
    # returns existing traces without appending to the ledger.
    rt.record_config(cfg, data_path, runs_dir)
    assert run_calls["n"] == 2
    second_ledger = (runs_dir / cfg.short_hash() / "trade_alerts.jsonl").read_text()
    # Ledger unchanged (no duplicate fires).
    assert first_ledger == second_ledger
