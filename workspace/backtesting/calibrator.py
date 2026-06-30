"""calibrator.py — run TriggerEngine over the full replay period with zero LLM calls.

The TriggerCalibrator maps gate-block distribution, soft-trigger breakdown,
and firing rate so miscalibration (and API cost) is caught before any backtest
spend. It makes NO LLM calls — the TriggerEngine is evaluated in isolation.

It mirrors production semantics:
  - Builds a snapshot from the CandleWindow on every candle (same as
    TradingLoop.run()).
  - Evaluates the TriggerEngine (hard gates short-circuit on first failure,
    then any one soft trigger fires).
  - Calls ``cooldown.update()`` on every fire with a directional alert (bias
    derived from ``htf_bias``), so ``cooldown_active`` blocks reflect the
    same tier behavior the BacktestEngine would experience.

Gate-block keys (mirror TriggerEngine gate order, first failure wins):
    no_killzone, no_htf_bias, no_dol, cooldown_active
    (+ no_soft_trigger when all gates pass but no soft trigger fires)

Soft-trigger keys (mirror TriggerEngine):
    fvg, ifvg, sweep, displacement

Writes ``workspace/calibration_report.json`` on completion.
``estimated_llm_cost_usd = total_fires * 0.003``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from trading.alert import AlertPayload

__all__ = ["CalibrationReport", "TriggerCalibrator"]

# Gate-block keys in TriggerEngine gate order (first failure wins).
_GATE_KEYS = ("no_killzone", "no_htf_bias", "no_dol", "cooldown_active", "no_soft_trigger")
# Soft-trigger keys in TriggerEngine evaluation order.
_SOFT_KEYS = ("fvg", "ifvg", "sweep", "displacement")
# Killzone keys.
_KZ_KEYS = ("london_kz", "ny_am_kz", "ny_pm_kz")
# Session-level keys — matches detect_session_levels output.
_SESSION_LEVEL_KEYS = (
    "midnight_open", "true_day_open", "london_open", "open_830", "open_930",
    "asia_high", "asia_low", "london_high", "london_low",
    "nyam_high", "nyam_low", "nylunch_high", "nylunch_low", "nypm_high", "nypm_low",
)


@dataclass
class CalibrationReport:
    """Aggregated calibration statistics over a replay period."""

    period: dict = field(default_factory=lambda: {"start": "", "end": ""})
    total_1m_candles: int = 0
    gate_blocks: dict = field(default_factory=lambda: {k: 0 for k in _GATE_KEYS})
    soft_triggers: dict = field(default_factory=lambda: {k: 0 for k in _SOFT_KEYS})
    # Counts how often each structure is present on candles that cleared all hard
    # gates, regardless of which structure fired first. Useful because the first-
    # match priority in TriggerEngine means sweep/displacement can never win once
    # FVGs have accumulated — this counter shows whether the detectors are working.
    structures_present: dict = field(default_factory=lambda: {k: 0 for k in _SOFT_KEYS})
    fires_by_killzone: dict = field(default_factory=lambda: {k: 0 for k in _KZ_KEYS})
    # How often a sweep structure (in recent_sweeps["5m"] or ["15m"]) is present
    # when that specific session level appears in session_levels, on candles that
    # cleared all hard gates. Tracks which price levels are most swept.
    sweeps_by_session: dict = field(default_factory=lambda: {k: 0 for k in _SESSION_LEVEL_KEYS})
    fires_by_month: dict = field(default_factory=dict)
    total_fires: int = 0
    estimated_llm_cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "period": self.period,
            "total_1m_candles": self.total_1m_candles,
            "gate_blocks": self.gate_blocks,
            "soft_triggers": self.soft_triggers,
            "structures_present": self.structures_present,
            "sweeps_by_session": self.sweeps_by_session,
            "fires_by_killzone": self.fires_by_killzone,
            "fires_by_month": self.fires_by_month,
            "total_fires": self.total_fires,
            "estimated_llm_cost_usd": self.estimated_llm_cost_usd,
        }


class TriggerCalibrator:
    """Run TriggerEngine over a CandleSource with zero LLM calls.

    Args:
        window: a CandleWindow (updated per candle).
        builder: a SnapshotBuilder.
        trigger: a TriggerEngine.
        cooldown: a CooldownState (updated on fires, mirroring production).
    """

    def __init__(self, window, builder, trigger, cooldown):
        self.window = window
        self.builder = builder
        self.trigger = trigger
        self.cooldown = cooldown

    def run(self, source) -> CalibrationReport:
        """Iterate ``source``, evaluate the trigger, return a CalibrationReport.

        Also writes ``workspace/calibration_report.json``.
        """
        report = CalibrationReport()
        first_ts = None
        last_ts = None

        while not source.is_done():
            candles = source.next()
            if candles is None:
                break
            self.window.update(candles)
            snapshot = self.builder.build(self.window)

            # Track the 1m candle timestamp for period + month bucketing.
            ts = snapshot.timestamp
            ts_str = str(ts)
            if first_ts is None:
                first_ts = ts_str
            last_ts = ts_str
            report.total_1m_candles += 1

            result = self.trigger.evaluate(snapshot)
            if result.should_trigger:
                self._record_fire(report, snapshot, result.reason)
            else:
                # Gate block (first failing gate, or no_soft_trigger).
                report.gate_blocks[result.reason] = (
                    report.gate_blocks.get(result.reason, 0) + 1
                )

            # Count each structure independently on candles that cleared all
            # hard gates (should_trigger OR no_soft_trigger). This is separate
            # from soft_triggers, which only records the winning trigger due to
            # first-match priority — so sweep/displacement would appear as 0
            # once FVGs accumulate even though the detectors are finding them.
            if result.should_trigger or result.reason == "no_soft_trigger":
                self._count_structures(report, snapshot)

        report.period = {"start": first_ts or "", "end": last_ts or ""}
        report.estimated_llm_cost_usd = round(report.total_fires * 0.003, 4)

        # Write the report JSON.
        out_path = Path("workspace/calibration_report.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report.to_dict(), indent=2, default=str))

        return report

    # ── helpers ─────────────────────────────────────────────────────────────

    def _record_fire(self, report, snapshot, reason: str) -> None:
        """Record a fire: increment counters and update cooldown (production parity)."""
        report.total_fires += 1
        # Soft-trigger breakdown.
        if reason in report.soft_triggers:
            report.soft_triggers[reason] += 1
        # Killzone bucket.
        kz = snapshot.current_killzone
        if kz in report.fires_by_killzone:
            report.fires_by_killzone[kz] += 1
        # Month bucket (YYYY-MM).
        month_key = self._month_key(snapshot.timestamp)
        report.fires_by_month[month_key] = report.fires_by_month.get(month_key, 0) + 1

        # Update cooldown with a directional alert (mirrors production).
        # The calibrator has no LLM, so it synthesizes a directional alert
        # whose bias is derived from htf_bias — exactly what the production
        # TradingLoop would emit (the LLM's bias follows htf_bias). This keeps
        # cooldown_active blocks comparable to the BacktestEngine.
        bias = "long" if snapshot.htf_bias == "bullish" else "short"
        alert = AlertPayload(bias=bias, killzone=kz or "")
        self.cooldown.update(alert, snapshot)

    def _count_structures(self, report: CalibrationReport, snapshot) -> None:
        """Increment structures_present for every soft trigger condition that
        holds, independent of priority order, and tally sweeps_by_session.

        Uses TriggerEngine.soft_triggers_present as the single source of truth,
        so a new soft trigger added to the engine is counted here automatically
        (no silent drift)."""
        present = self.trigger.soft_triggers_present(snapshot)
        for name, is_present in present.items():
            if is_present and name in report.structures_present:
                report.structures_present[name] += 1
        if present.get("sweep"):
            self._count_sweeps_by_session(report, snapshot)

    @staticmethod
    def _count_sweeps_by_session(report: CalibrationReport, snapshot) -> None:
        """Tally which session levels were actually swept on this candle.

        For each sweep in the recent 5m/15m window, match its ``swept_level``
        price against the snapshot's session levels (within a relative
        tolerance) and increment the matching session-level keys. This counts
        the level the price ran — not merely which levels happened to be
        defined — so the breakdown reflects real sweep targets."""
        sl = snapshot.session_levels or {}
        sweeps = (snapshot.recent_sweeps.get("15m") or []) + (snapshot.recent_sweeps.get("5m") or [])
        if not sweeps:
            return
        matched: set[str] = set()  # one increment per key per candle
        for sweep in sweeps:
            level = sweep.get("swept_level")
            if level is None:
                continue
            for key in _SESSION_LEVEL_KEYS:
                val = sl.get(key)
                if val is None or key in matched:
                    continue
                # Relative tolerance (0.05% of the level), mirroring the
                # confluence band used in SnapshotBuilder._build_pools.
                if abs(level - val) <= abs(val) * 0.0005:
                    report.sweeps_by_session[key] = report.sweeps_by_session.get(key, 0) + 1
                    matched.add(key)

    @staticmethod
    def _month_key(ts) -> str:
        """Return 'YYYY-MM' for a timestamp (datetime or ISO str)."""
        from datetime import datetime

        if isinstance(ts, datetime):
            dt = ts
        else:
            dt = datetime.fromisoformat(str(ts))
        return dt.strftime("%Y-%m")
