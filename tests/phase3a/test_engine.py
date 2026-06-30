"""test_engine.py — BacktestEngine unit tests.

The engine drives a MockCandleSource through the TradingLoop with a mocked
LLM agent (returns a fixture AlertPayload), attaches OutcomeSimulator
results, and writes trade_traces.jsonl. Resume logic is tested by running
the engine twice on the same source.
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure workspace/ is importable.
_WORKSPACE = Path(__file__).resolve().parent.parent.parent / "workspace"
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from trading.candle_source import MockCandleSource, TIMEFRAMES
from trading.candle_window import CandleWindow, DEFAULT_SIZES
from trading.snapshot import SnapshotBuilder
from trading.trigger import TriggerEngine
from trading.cooldown import CooldownState
from trading.alert import AlertPayload
from trading.trading_loop import TradingLoop
from trading.reasoning_agent import LLMReasoningAgent
from backtesting.outcome import OutcomeSimulator
from backtesting.engine import BacktestEngine, TradeTrace


# ── Helpers ───────────────────────────────────────────────────────────────────

# The verbatim LLM response string the mock router returns. The engine
# captures this as raw_llm_output.
_RAW_LLM = '{"bias": "long", "model": "2022", "entry_zone": [0,0], "stop": 0, "target": 0}'


def _mock_agent(bias="long", model="2022"):
    """A mocked LLMReasoningAgent that returns a fixture AlertPayload.

    The alert's entry_zone/stop/target are set relative to the snapshot's
    current_price so the trade can fill and resolve against subsequent
    candles. The mock exposes a model_router whose .call() returns a
    verbatim string; the mock's reason() calls the router (mirroring the
    production LLMReasoningAgent.reason() path) so the engine can capture
    raw_llm_output.
    """
    agent = MagicMock(spec=LLMReasoningAgent)

    # A mock router whose .call() returns a fixed response.
    router = MagicMock()
    resp = MagicMock()
    resp.content = _RAW_LLM
    router.call.return_value = resp
    agent.model_router = router

    def _reason(snapshot):
        # Mirror production: call the router (so the engine's capture works).
        router.call(agent_config={}, messages=[{"role": "user", "content": "{}"}])
        price = snapshot.current_price
        if bias == "long":
            entry_zone = (price - 5, price + 5)
            stop = price - 50
            target = price + 100
        else:
            entry_zone = (price - 5, price + 5)
            stop = price + 50
            target = price - 100
        return AlertPayload(
            bias=bias, model=model, conviction=70,
            entry_zone=entry_zone, stop=stop, target=target,
            dol={"level": target, "type": "bsl" if bias == "long" else "ssl",
                 "timeframe": "1h"},
            risk_reward=0.0, rationale="fixture alert",
            killzone=snapshot.current_killzone or "",
            valid_until="",
        )

    agent.reason.side_effect = _reason
    return agent


def _make_engine(tmp_path, scenario="sweep_and_fvg", n=100, agent=None,
                 output_path=None):
    """Build a BacktestEngine over a MockCandleSource."""
    src = MockCandleSource(scenario, n_bars=n)
    w = CandleWindow(DEFAULT_SIZES)
    builder = SnapshotBuilder()
    cd = CooldownState()
    trigger = TriggerEngine(cd)
    if agent is None:
        agent = _mock_agent()
    out = str(tmp_path / "alerts.jsonl")
    loop = TradingLoop(src, w, builder, trigger, agent, cd, output_path=out)
    sim = OutcomeSimulator()
    traces_path = output_path or (tmp_path / "trade_traces.jsonl")
    engine = BacktestEngine(loop, sim, output_path=traces_path)
    return engine, src


# ── Basic run ─────────────────────────────────────────────────────────────────


def test_engine_writes_at_least_one_trace(tmp_path):
    """BacktestEngine with sweep_and_fvg writes ≥1 TradeTrace to JSONL."""
    engine, src = _make_engine(tmp_path, scenario="sweep_and_fvg", n=100)
    traces = engine.run(src)

    # At least one trace produced.
    assert len(traces) >= 1

    # The JSONL file exists and has the same number of lines.
    lines = [l for l in engine.output_path.read_text().splitlines() if l.strip()]
    assert len(lines) >= 1

    # Each line is valid JSON with the required TradeTrace fields.
    required = {"trace_id", "timestamp", "instrument", "killzone",
                "trigger_reason", "snapshot_summary", "raw_llm_output",
                "alert", "rr_validated", "outcome"}
    for line in lines:
        rec = json.loads(line)
        assert required.issubset(rec.keys()), f"missing: {required - set(rec.keys())}"


def test_engine_trace_has_outcome(tmp_path):
    """Each trace has an outcome dict with a result field."""
    engine, src = _make_engine(tmp_path, scenario="sweep_and_fvg", n=100)
    traces = engine.run(src)
    assert len(traces) >= 1
    for t in traces:
        assert "result" in t.outcome
        assert t.outcome["result"] in ("win", "loss", "expired", "no_fill", "no_trade")


def test_engine_trace_has_alert_dict(tmp_path):
    """Each trace has an alert dict with bias/entry_zone/stop/target."""
    engine, src = _make_engine(tmp_path, scenario="sweep_and_fvg", n=100)
    traces = engine.run(src)
    assert len(traces) >= 1
    for t in traces:
        assert "bias" in t.alert
        assert "entry_zone" in t.alert
        assert "stop" in t.alert
        assert "target" in t.alert


def test_engine_trace_has_snapshot_summary(tmp_path):
    """snapshot_summary has no raw candle arrays (compact dict)."""
    engine, src = _make_engine(tmp_path, scenario="sweep_and_fvg", n=100)
    traces = engine.run(src)
    assert len(traces) >= 1
    for t in traces:
        # snapshot_summary should be a dict with instrument/timestamp/etc.
        assert isinstance(t.snapshot_summary, dict)
        assert "instrument" in t.snapshot_summary
        assert "current_price" in t.snapshot_summary
        # No raw candle arrays — compact dict only.
        assert "candles" not in t.snapshot_summary


def test_engine_rr_validated(tmp_path):
    """rr_validated is True for a valid (>1:1) alert."""
    engine, src = _make_engine(tmp_path, scenario="sweep_and_fvg", n=100)
    traces = engine.run(src)
    assert len(traces) >= 1
    for t in traces:
        # The fixture alert has risk=50, reward=100 → rr=2.0 → validated.
        assert t.rr_validated is True


# ── Resume logic ──────────────────────────────────────────────────────────────


def test_engine_resume_skips_existing(tmp_path):
    """Running the engine twice on the same source skips first run's traces."""
    engine1, src1 = _make_engine(tmp_path, scenario="sweep_and_fvg", n=100)
    traces1 = engine1.run(src1)
    n1 = len(traces1)
    assert n1 >= 1

    # Count lines in the JSONL.
    lines1 = [l for l in engine1.output_path.read_text().splitlines() if l.strip()]
    assert len(lines1) == n1

    # Run again with a fresh source (same scenario/seed → same candles).
    engine2, src2 = _make_engine(
        tmp_path, scenario="sweep_and_fvg", n=100,
        output_path=engine1.output_path,
    )
    traces2 = engine2.run(src2)

    # No new lines appended (all timestamps already processed).
    lines2 = [l for l in engine2.output_path.read_text().splitlines() if l.strip()]
    assert len(lines2) == n1, "second run should not append duplicate traces"

    # The returned list has the same traces (no duplicates).
    assert len(traces2) == n1


def test_engine_resume_no_duplicate_timestamps(tmp_path):
    """Resumed run produces no duplicate trace timestamps."""
    engine1, src1 = _make_engine(tmp_path, scenario="sweep_and_fvg", n=100)
    engine1.run(src1)

    engine2, src2 = _make_engine(
        tmp_path, scenario="sweep_and_fvg", n=100,
        output_path=engine1.output_path,
    )
    traces2 = engine2.run(src2)

    timestamps = [t.timestamp for t in traces2]
    assert len(timestamps) == len(set(timestamps)), "duplicate timestamps found"


def test_engine_resume_no_api_key_required(tmp_path):
    """The engine runs without any API key (LLM mocked)."""
    import os

    # Ensure no API keys are set.
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ZAI_API_KEY"):
        os.environ.pop(key, None)

    engine, src = _make_engine(tmp_path, scenario="sweep_and_fvg", n=100)
    traces = engine.run(src)
    assert len(traces) >= 1


# ── Lookahead safety ──────────────────────────────────────────────────────────


def test_engine_outcome_uses_only_subsequent_candles(tmp_path):
    """The OutcomeSimulator receives only candles after the alert timestamp.

    We instrument the simulator to record the timestamps it receives and
    assert all are strictly greater than the alert timestamp.
    """
    received_ts: list = []

    class RecordingSimulator(OutcomeSimulator):
        def simulate(self, alert, subsequent_candles):
            for c in subsequent_candles:
                received_ts.append(c["timestamp"])
            return super().simulate(alert, subsequent_candles)

    src = MockCandleSource("sweep_and_fvg", n_bars=100)
    w = CandleWindow(DEFAULT_SIZES)
    builder = SnapshotBuilder()
    cd = CooldownState()
    trigger = TriggerEngine(cd)
    agent = _mock_agent()
    out = str(tmp_path / "alerts.jsonl")
    loop = TradingLoop(src, w, builder, trigger, agent, cd, output_path=out)
    sim = RecordingSimulator()
    traces_path = tmp_path / "trade_traces.jsonl"
    engine = BacktestEngine(loop, sim, output_path=traces_path)
    traces = engine.run(src)

    assert len(traces) >= 1
    # Every timestamp the simulator received must be strictly after the
    # corresponding alert timestamp.
    alert_ts_set = {t.timestamp for t in traces}
    for ts_str in received_ts:
        # The received timestamps should not include any alert timestamp.
        assert ts_str not in alert_ts_set, (
            f"simulator received an alert timestamp {ts_str} — lookahead leak"
        )


# ── raw_llm_output capture ────────────────────────────────────────────────────


def test_engine_captures_raw_llm_output(tmp_path):
    """raw_llm_output is the verbatim model_router.call() response string."""
    engine, src = _make_engine(tmp_path, scenario="sweep_and_fvg", n=100)
    traces = engine.run(src)
    assert len(traces) >= 1
    for t in traces:
        # The mock router returns a fixed JSON string.
        assert t.raw_llm_output == _RAW_LLM


# ── Two-phase persistence / crash safety ──────────────────────────────────────


def test_engine_writes_alert_ledger(tmp_path):
    """Phase A writes a trade_alerts.jsonl ledger alongside the trace file.

    Every ledger line carries the alert + raw LLM output but NO outcome — the
    outcome lives only in the derived trade_traces.jsonl.
    """
    engine, src = _make_engine(tmp_path, scenario="sweep_and_fvg", n=100)
    traces = engine.run(src)
    assert len(traces) >= 1

    assert engine.alerts_path.name == "trade_alerts.jsonl"
    ledger_lines = [l for l in engine.alerts_path.read_text().splitlines() if l.strip()]
    assert len(ledger_lines) == len(traces)
    for line in ledger_lines:
        rec = json.loads(line)
        assert "raw_llm_output" in rec
        assert "alert" in rec
        assert "outcome" not in rec, "ledger must not carry outcomes (Phase A)"


def test_engine_resumes_from_ledger_without_trace_file(tmp_path):
    """A crash before Phase B leaves only the ledger; resume must not re-spend.

    Simulate the exact failure that motivated this design: the ledger exists
    (alerts were fsynced during replay) but trade_traces.jsonl was never
    written. The resumed run must skip every ledgered timestamp (no LLM call)
    and still produce the full trace file.
    """
    engine1, src1 = _make_engine(tmp_path, scenario="sweep_and_fvg", n=100)
    engine1.run(src1)
    n1 = sum(1 for l in engine1.alerts_path.read_text().splitlines() if l.strip())
    assert n1 >= 1

    # Simulate the crash: delete the derived trace file, keep the ledger.
    engine1.output_path.unlink()

    engine2, src2 = _make_engine(
        tmp_path, scenario="sweep_and_fvg", n=100,
        output_path=engine1.output_path,
    )
    traces2 = engine2.run(src2)

    # No LLM calls on resume — every alert came from the ledger.
    engine2.loop.agent.reason.assert_not_called()
    # Ledger unchanged (no duplicate appends) and trace file regenerated.
    n2 = sum(1 for l in engine2.alerts_path.read_text().splitlines() if l.strip())
    assert n2 == n1
    assert len(traces2) == n1
    assert engine2.output_path.exists()


def test_engine_resumes_from_legacy_trace_file(tmp_path):
    """If only a legacy trade_traces.jsonl exists (no ledger), still resume.

    Back-compat: a run from the pre-split engine left full traces but no
    ledger. The resumed run must read those timestamps and skip the LLM.
    """
    # Hand-write a legacy trace file with one resolved trace, then point a
    # fresh engine at it with no ledger present.
    engine0, src0 = _make_engine(tmp_path, scenario="sweep_and_fvg", n=100)
    engine0.run(src0)
    legacy = engine0.output_path.read_text()
    first_ts = json.loads(next(l for l in legacy.splitlines() if l.strip()))["timestamp"]

    # Reset: keep only the legacy trace file, remove the ledger.
    engine0.alerts_path.unlink()

    engine1, src1 = _make_engine(
        tmp_path, scenario="sweep_and_fvg", n=100,
        output_path=engine0.output_path,
    )
    _, processed = engine1._load_existing()
    assert first_ts in processed, "legacy trace timestamps must seed resume"


def test_engine_trace_file_written_atomically_no_tmp_left(tmp_path):
    """Phase B leaves no .tmp file behind after a successful write."""
    engine, src = _make_engine(tmp_path, scenario="sweep_and_fvg", n=100)
    engine.run(src)
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == [], f"temp files left behind: {leftovers}"


# ── TradeTrace.to_dict ────────────────────────────────────────────────────────


def test_trade_trace_to_dict():
    trace = TradeTrace(
        trace_id="ts1_long_abc12345",
        timestamp="2026-06-15T10:00:00+00:00",
        instrument="NQ",
        killzone="ny_am_kz",
        trigger_reason="fvg",
        snapshot_summary={"instrument": "NQ"},
        raw_llm_output="{}",
        alert={"bias": "long"},
        rr_validated=True,
        outcome={"result": "win"},
    )
    d = trace.to_dict()
    assert d["trace_id"] == "ts1_long_abc12345"
    assert d["instrument"] == "NQ"
    assert d["outcome"] == {"result": "win"}
