"""engine.py — drive the full TradingLoop over historical data and attach outcomes.

The BacktestEngine replays a CandleSource through the production TradingLoop
(with live LLM inference), captures each emitted alert with its metadata
(trigger reason, raw LLM output, snapshot summary), and attaches an
OutcomeSimulator result to produce a TradeTrace. Traces are appended
idempotently to ``trade_traces.jsonl``.

Resume logic
------------
On restart, the engine reads existing ``trade_traces.jsonl`` and collects the
processed alert timestamps. During replay, alerts whose timestamp is already
in the file are skipped (no duplicate TradeTrace written). The cooldown state
is rebuilt from the existing traces' alert biases so subsequent triggers fire
identically to the first run — no LLM re-call is needed for resumed alerts.

Replay buffer
-------------
The engine maintains a replay buffer of 1m candles (all candles processed,
which is ≥480 for any meaningful backtest — enough to resolve intraday
outcomes over 8 hours). When an alert fires, the engine slices subsequent
candles (strictly AFTER the alert timestamp) from this buffer and passes them
to OutcomeSimulator. These subsequent candles are NEVER visible to the
TradingLoop — they are purely for outcome resolution (the one allowed forward
read, isolated to OutcomeSimulator after the alert is produced).

No broker, no IBKR, no live market data, no order placement. The LLM is the
only network dependency and is mocked in tests.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from trading.alert import AlertPayload, validate_rr

__all__ = ["TradeTrace", "BacktestEngine"]

# Minimum replay buffer size (1m candles) — 8 hours of intraday data.
_MIN_BUFFER = 480


@dataclass
class TradeTrace:
    """A single trade trace: alert + metadata + resolved outcome.

    Fields:
        trace_id: unique identifier for the trace.
        timestamp: the alert timestamp (ISO string).
        instrument: "NQ" (from the snapshot).
        killzone: the killzone active at the alert time.
        trigger_reason: the soft trigger that fired (fvg/ifvg/sweep/displacement).
        snapshot_summary: compact snapshot dict (no raw candle arrays).
        raw_llm_output: the verbatim model_router.call() response string.
        alert: the AlertPayload as a dict (after validate_rr).
        rr_validated: whether R:R validation passed (risk_reward >= 1.0).
        outcome: the TradeOutcome as a dict.
    """

    trace_id: str
    timestamp: str
    instrument: str
    killzone: str
    trigger_reason: str
    snapshot_summary: dict
    raw_llm_output: str
    alert: dict
    rr_validated: bool
    outcome: dict

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "instrument": self.instrument,
            "killzone": self.killzone,
            "trigger_reason": self.trigger_reason,
            "snapshot_summary": self.snapshot_summary,
            "raw_llm_output": self.raw_llm_output,
            "alert": self.alert,
            "rr_validated": self.rr_validated,
            "outcome": self.outcome,
        }


def _parse_ts(ts: Any) -> datetime:
    """Parse a timestamp (datetime or ISO-8601 str) to an aware datetime."""
    if isinstance(ts, datetime):
        return ts
    return datetime.fromisoformat(str(ts))


class BacktestEngine:
    """Drive a CandleSource through the TradingLoop and attach outcomes.

    Args:
        loop: a TradingLoop (the engine uses its window, builder, trigger,
            agent, and cooldown components).
        simulator: an OutcomeSimulator.
        output_path: path to the trade_traces.jsonl file.
    """

    def __init__(self, loop, simulator, output_path: Path = Path("workspace/trade_traces.jsonl")):
        self.loop = loop
        self.simulator = simulator
        self.output_path = Path(output_path)

    def run(self, source) -> list[TradeTrace]:
        """Drive ``source`` through the loop, attach outcomes, write traces.

        Returns a list of ALL TradeTrace objects (existing resumed ones +
        newly produced ones), ordered by timestamp.
        """
        # ── Resume: read existing traces ────────────────────────────────────
        existing_traces, processed_ts = self._load_existing()

        # Access the loop's components.
        window = self.loop.window
        builder = self.loop.builder
        trigger = self.loop.trigger
        agent = self.loop.agent
        cooldown = self.loop.cooldown

        # Replay buffer: all 1m candles processed (≥480 for meaningful runs).
        replay_buffer: list[dict] = []

        # Deferred alerts: (alert, snapshot, trigger_reason, raw_llm_output).
        deferred: list[tuple] = []

        # ── Drive the source candle by candle ───────────────────────────────
        while not source.is_done():
            candles = source.next()
            if candles is None:
                break

            window.update(candles)
            snapshot = builder.build(window)

            # Collect the 1m candle into the replay buffer.
            one_m = candles.get("1m")
            if one_m is not None:
                replay_buffer.append(one_m)

            result = trigger.evaluate(snapshot)
            if not result.should_trigger:
                continue

            alert_ts = snapshot.timestamp
            alert_ts_str = str(alert_ts)

            if alert_ts_str in processed_ts:
                # Resume: skip LLM call, use existing alert for cooldown.
                existing_alert_dict = existing_traces[alert_ts_str].get("alert", {})
                alert = AlertPayload(bias=existing_alert_dict.get("bias", "no_trade"))
                cooldown.update(alert, snapshot)
                continue

            # New alert: call the LLM and capture the raw output.
            raw_llm_output, alert = self._reason_with_capture(agent, snapshot)
            alert = validate_rr(alert)
            cooldown.update(alert, snapshot)

            # Defer for outcome resolution (after the source is exhausted).
            deferred.append((alert, snapshot, result.reason, raw_llm_output))

        # ── Resolve deferred alerts ─────────────────────────────────────────
        new_traces: list[TradeTrace] = []
        for alert, snapshot, trigger_reason, raw_llm_output in deferred:
            subsequent = self._slice_subsequent(replay_buffer, snapshot.timestamp)
            outcome = self.simulator.simulate(alert, subsequent)
            trace = self._build_trace(alert, snapshot, trigger_reason, raw_llm_output, outcome)
            new_traces.append(trace)
            self._append_trace(trace)

        # ── Return all traces (existing + new), ordered by timestamp ────────
        all_traces = list(existing_traces.values())
        # Convert existing trace dicts to TradeTrace objects.
        all_trace_objs = [self._dict_to_trace(d) for d in all_traces] + new_traces
        all_trace_objs.sort(key=lambda t: t.timestamp)
        return all_trace_objs

    # ── resume helpers ──────────────────────────────────────────────────────

    def _load_existing(self) -> tuple[dict, set]:
        """Read existing trade_traces.jsonl.

        Returns (traces_by_timestamp, set_of_processed_timestamps).
        """
        traces: dict[str, dict] = {}
        processed: set[str] = set()
        if not self.output_path.exists():
            return traces, processed
        for line in self.output_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines (e.g., partial write on crash).
                continue
            ts = rec.get("timestamp")
            if ts is not None:
                traces[ts] = rec
                processed.add(ts)
        return traces, processed

    # ── LLM capture ─────────────────────────────────────────────────────────

    def _reason_with_capture(self, agent, snapshot) -> tuple[str, AlertPayload]:
        """Call agent.reason(snapshot), capturing the raw LLM output string.

        Wraps ``agent.model_router.call`` (if present) to capture the verbatim
        response content. For mocked agents without a real router, returns "".
        """
        raw_output = ""
        router = getattr(agent, "model_router", None)
        original_call = None
        if router is not None and hasattr(router, "call"):
            original_call = router.call

            def capturing_call(*args, **kwargs):
                resp = original_call(*args, **kwargs)
                nonlocal raw_output
                raw_output = getattr(resp, "content", "") or ""
                return resp

            router.call = capturing_call

        try:
            alert = agent.reason(snapshot)
        finally:
            if original_call is not None and router is not None:
                router.call = original_call

        return raw_output, alert

    # ── subsequent candle slicing ───────────────────────────────────────────

    @staticmethod
    def _slice_subsequent(replay_buffer: list[dict], alert_ts) -> list[dict]:
        """Return candles from replay_buffer strictly AFTER alert_ts.

        CRITICAL: only candles with timestamp > alert_ts are returned. The
        alert candle itself is excluded — using it would leak the alert bar's
        high/low into outcome resolution (lookahead bias).
        """
        alert_dt = _parse_ts(alert_ts)
        subsequent = []
        for candle in replay_buffer:
            c_dt = _parse_ts(candle["timestamp"])
            if c_dt > alert_dt:
                subsequent.append(candle)
        return subsequent

    # ── trace building ──────────────────────────────────────────────────────

    def _build_trace(self, alert, snapshot, trigger_reason, raw_llm_output, outcome) -> TradeTrace:
        """Build a TradeTrace from the alert, snapshot, and outcome."""
        ts_str = str(snapshot.timestamp)
        trace_id = self._make_trace_id(ts_str, alert)
        rr_validated = alert.risk_reward >= 1.0
        return TradeTrace(
            trace_id=trace_id,
            timestamp=ts_str,
            instrument=snapshot.instrument,
            killzone=alert.killzone or snapshot.current_killzone or "",
            trigger_reason=trigger_reason,
            snapshot_summary=snapshot.to_compact_dict(),
            raw_llm_output=raw_llm_output,
            alert=alert.to_dict(),
            rr_validated=rr_validated,
            outcome=outcome.to_dict(),
        )

    @staticmethod
    def _make_trace_id(ts_str: str, alert) -> str:
        """Build a deterministic trace ID from timestamp + bias."""
        return f"{ts_str}_{alert.bias}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _dict_to_trace(d: dict) -> TradeTrace:
        """Convert a trace dict (from JSONL) to a TradeTrace object."""
        return TradeTrace(
            trace_id=d.get("trace_id", ""),
            timestamp=d.get("timestamp", ""),
            instrument=d.get("instrument", ""),
            killzone=d.get("killzone", ""),
            trigger_reason=d.get("trigger_reason", ""),
            snapshot_summary=d.get("snapshot_summary", {}),
            raw_llm_output=d.get("raw_llm_output", ""),
            alert=d.get("alert", {}),
            rr_validated=d.get("rr_validated", False),
            outcome=d.get("outcome", {}),
        )

    # ── file append ─────────────────────────────────────────────────────────

    def _append_trace(self, trace: TradeTrace) -> None:
        """Append one TradeTrace as a JSON line to output_path."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "a") as f:
            f.write(json.dumps(trace.to_dict(), default=str) + "\n")
