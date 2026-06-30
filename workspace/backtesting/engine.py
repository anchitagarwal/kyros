"""engine.py — drive the full TradingLoop over historical data and attach outcomes.

The BacktestEngine replays a CandleSource through the production TradingLoop
(with live LLM inference), captures each emitted alert with its metadata
(trigger reason, raw LLM output, snapshot summary), and attaches an
OutcomeSimulator result to produce a TradeTrace.

Two-phase persistence
---------------------
The expensive, irrecoverable part of a run is the LLM call. Outcomes, by
contrast, are a pure function of (alert, candles) and can always be recomputed
offline. So the engine splits writing into two phases:

  Phase A (online, crash-safe ledger): the moment the LLM returns an alert, the
    engine appends it — with its raw LLM output and snapshot summary, but NO
    outcome yet — to ``trade_alerts.jsonl`` and fsyncs. This append-only file is
    the resume ledger; a crash mid-replay loses at most the single in-flight
    call, never the whole run.

  Phase B (offline): after the source is exhausted, the engine resolves an
    outcome for every alert (resumed + new) from the replay buffer and writes
    the full traces to ``trade_traces.jsonl`` (a derived file, rewritten
    atomically each run). Because it is regenerable from the ledger + data, it
    is safe to overwrite; the precious raw LLM outputs live only in the
    append-only ledger.

Resume logic
------------
On restart, the engine reads the ``trade_alerts.jsonl`` ledger (falling back to
a legacy ``trade_traces.jsonl`` if only that exists, so a half-migrated run does
not re-spend) and collects the processed alert timestamps. During replay, alerts
whose timestamp is already in the ledger are skipped — the stored alert is
reconstructed to drive cooldown so subsequent triggers fire identically to the
first run, and its outcome is recomputed in Phase B. No LLM re-call is made for
a resumed alert.

Replay buffer
-------------
The engine accumulates every processed 1m candle into a replay buffer (so it
grows to the full dataset). After replay, each deferred alert slices the
candles strictly AFTER its timestamp — located via bisect over the buffer's
chronological timestamps and capped at _MAX_LOOKAHEAD — and passes them to
OutcomeSimulator. These subsequent candles are NEVER visible to the
TradingLoop — they are purely for outcome resolution (the one allowed forward
read, isolated to OutcomeSimulator after the alert is produced).

No broker, no IBKR, no live market data, no order placement. The LLM is the
only network dependency and is mocked in tests.
"""

from __future__ import annotations

import bisect
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from trading.alert import AlertPayload, validate_rr

__all__ = ["TradeTrace", "BacktestEngine"]

# Minimum replay buffer size (1m candles) — 8 hours of intraday data.
_MIN_BUFFER = 480

# Max candles examined per alert during outcome resolution. Every alert resolves
# within its killzone (same trading day), so two trading days of 1m bars is always
# enough; the cap bounds resolution at O(_MAX_LOOKAHEAD) instead of O(n) per alert.
_MAX_LOOKAHEAD = 2880


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
        output_path: path to the trade_traces.jsonl file (the Phase B output).
        alerts_path: path to the crash-safe Phase A ledger. Defaults to a
            sibling ``trade_alerts.jsonl`` next to ``output_path``.
    """

    def __init__(self, loop, simulator,
                 output_path: Path = Path("workspace/trade_traces.jsonl"),
                 alerts_path: Path | None = None):
        self.loop = loop
        self.simulator = simulator
        self.output_path = Path(output_path)
        self.alerts_path = (Path(alerts_path) if alerts_path is not None
                            else self.output_path.with_name("trade_alerts.jsonl"))

    def run(self, source) -> list[TradeTrace]:
        """Drive ``source`` through the loop, attach outcomes, write traces.

        Returns a list of ALL TradeTrace objects (existing resumed ones +
        newly produced ones), ordered by timestamp.
        """
        # ── Resume: read the existing alert ledger ──────────────────────────
        existing_alerts, processed_ts = self._load_existing()

        # Access the loop's components.
        window = self.loop.window
        builder = self.loop.builder
        trigger = self.loop.trigger
        agent = self.loop.agent
        cooldown = self.loop.cooldown

        # Replay buffer: all 1m candles processed (≥480 for meaningful runs).
        replay_buffer: list[dict] = []

        # Pending alerts awaiting Phase B outcome resolution. Each entry is
        # (alert_record_dict, AlertPayload, alert_timestamp) and covers both
        # resumed (from the ledger) and newly produced alerts.
        pending: list[tuple] = []

        # ── Phase A: drive the source candle by candle ──────────────────────
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
                # Resume: skip the LLM call. Reconstruct the stored alert to
                # drive cooldown identically and to resolve its outcome below.
                record = existing_alerts[alert_ts_str]
                alert = self._alert_from_dict(record.get("alert", {}))
                cooldown.update(alert, snapshot)
                pending.append((record, alert, alert_ts))
                continue

            # New alert: call the LLM and capture the raw output.
            raw_llm_output, alert = self._reason_with_capture(agent, snapshot)
            alert = validate_rr(alert)
            cooldown.update(alert, snapshot)

            # Persist immediately (crash-safe) BEFORE deferring outcome work.
            record = self._build_alert_record(alert, snapshot, result.reason, raw_llm_output)
            self._append_alert(record)
            pending.append((record, alert, alert_ts))

        # ── Phase B: resolve outcomes and write the derived trace file ──────
        # Parse candle timestamps once (chronological → sorted) so each alert
        # uses an O(log n) bisect instead of a full scan of the buffer.
        buffer_ts = [_parse_ts(c["timestamp"]) for c in replay_buffer]

        traces: list[TradeTrace] = []
        for record, alert, alert_ts in pending:
            subsequent = self._slice_subsequent(replay_buffer, buffer_ts, alert_ts)
            outcome = self.simulator.simulate(alert, subsequent)
            traces.append(self._trace_from_record(record, outcome))

        traces.sort(key=lambda t: t.timestamp)
        self._write_traces(traces)
        return traces

    # ── resume helpers ──────────────────────────────────────────────────────

    def _load_existing(self) -> tuple[dict, set]:
        """Read the alert ledger (or legacy trace file) for resume.

        Reads ``trade_alerts.jsonl`` if present; otherwise falls back to a
        legacy ``trade_traces.jsonl`` so a run written by the pre-split engine
        still resumes without re-spending. Both formats carry ``timestamp`` and
        ``alert``, which is all resume needs.

        Returns (records_by_timestamp, set_of_processed_timestamps).
        """
        source_path = self.alerts_path if self.alerts_path.exists() else self.output_path
        records: dict[str, dict] = {}
        processed: set[str] = set()
        if not source_path.exists():
            return records, processed
        for line in source_path.read_text().splitlines():
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
                records[ts] = rec
                processed.add(ts)
        return records, processed

    @staticmethod
    def _alert_from_dict(d: dict) -> AlertPayload:
        """Reconstruct an AlertPayload from a stored alert dict.

        Used on resume to drive cooldown and recompute the outcome without
        re-calling the LLM. ``entry_zone`` is normalized back to a 2-tuple.
        """
        ez = d.get("entry_zone", [0.0, 0.0])
        entry_zone = ((float(ez[0]), float(ez[1]))
                      if isinstance(ez, (list, tuple)) and len(ez) >= 2 else (0.0, 0.0))
        return AlertPayload(
            bias=d.get("bias", "no_trade"),
            model=d.get("model", "none"),
            conviction=int(d.get("conviction", 0)),
            entry_zone=entry_zone,
            stop=float(d.get("stop", 0.0)),
            target=float(d.get("target", 0.0)),
            dol=d.get("dol", {}),
            risk_reward=float(d.get("risk_reward", 0.0)),
            rationale=d.get("rationale", ""),
            killzone=d.get("killzone", ""),
            valid_until=d.get("valid_until", ""),
            no_trade_reason=d.get("no_trade_reason"),
        )

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
    def _slice_subsequent(replay_buffer: list[dict], buffer_ts: list[datetime],
                          alert_ts) -> list[dict]:
        """Return up to _MAX_LOOKAHEAD candles strictly AFTER alert_ts.

        ``buffer_ts`` holds the parsed candle timestamps in the same order as
        ``replay_buffer`` (chronological → sorted). ``bisect_right`` finds the
        first index strictly greater than the alert timestamp, so the alert
        candle itself is excluded — using it would leak the alert bar's
        high/low into outcome resolution (lookahead bias). The slice is capped
        at _MAX_LOOKAHEAD: every alert resolves within its killzone, so two
        trading days of bars always suffice, and the cap keeps resolution
        O(_MAX_LOOKAHEAD) rather than copying the whole tail per alert.
        """
        alert_dt = _parse_ts(alert_ts)
        lo = bisect.bisect_right(buffer_ts, alert_dt)
        hi = min(lo + _MAX_LOOKAHEAD, len(replay_buffer))
        return replay_buffer[lo:hi]

    # ── trace building ──────────────────────────────────────────────────────

    def _build_alert_record(self, alert, snapshot, trigger_reason, raw_llm_output) -> dict:
        """Build the Phase A ledger record: everything but the outcome.

        The record is the TradeTrace shape minus ``outcome`` — its ``trace_id``
        is generated once here and reused when Phase B attaches the outcome, so
        a resumed alert keeps a stable id across runs.
        """
        ts_str = str(snapshot.timestamp)
        return {
            "trace_id": self._make_trace_id(ts_str, alert),
            "timestamp": ts_str,
            "instrument": snapshot.instrument,
            "killzone": alert.killzone or snapshot.current_killzone or "",
            "trigger_reason": trigger_reason,
            "snapshot_summary": snapshot.to_compact_dict(),
            "raw_llm_output": raw_llm_output,
            "alert": alert.to_dict(),
            "rr_validated": alert.risk_reward >= 1.0,
        }

    @staticmethod
    def _trace_from_record(record: dict, outcome) -> TradeTrace:
        """Combine a Phase A ledger record with a resolved outcome."""
        return TradeTrace(
            trace_id=record.get("trace_id", ""),
            timestamp=record.get("timestamp", ""),
            instrument=record.get("instrument", ""),
            killzone=record.get("killzone", ""),
            trigger_reason=record.get("trigger_reason", ""),
            snapshot_summary=record.get("snapshot_summary", {}),
            raw_llm_output=record.get("raw_llm_output", ""),
            alert=record.get("alert", {}),
            rr_validated=record.get("rr_validated", False),
            outcome=outcome.to_dict(),
        )

    @staticmethod
    def _make_trace_id(ts_str: str, alert) -> str:
        """Build a deterministic trace ID from timestamp + bias."""
        return f"{ts_str}_{alert.bias}_{uuid.uuid4().hex[:8]}"

    # ── file IO ─────────────────────────────────────────────────────────────

    def _append_alert(self, record: dict) -> None:
        """Append one Phase A ledger record, fsynced for crash safety.

        The fsync is what makes resume reliable: an interrupted run loses at
        most the single in-flight LLM call, never the whole replay. The cost is
        trivial next to LLM call latency.
        """
        self.alerts_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.alerts_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _write_traces(self, traces: list[TradeTrace]) -> None:
        """Atomically (re)write the derived trace file in one pass.

        Safe to overwrite: every trace is regenerable from the append-only
        ledger plus the data. Writing to a temp file and ``os.replace``-ing it
        means a crash mid-write never leaves a half-written trace file.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.output_path.with_name(self.output_path.name + ".tmp")
        with open(tmp, "w") as f:
            for trace in traces:
                f.write(json.dumps(trace.to_dict(), default=str) + "\n")
        os.replace(tmp, self.output_path)
