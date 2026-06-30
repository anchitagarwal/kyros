"""trigger.py — hard-gate + soft-trigger evaluation before any LLM call.

The TriggerEngine prevents wasteful/incoherent LLM invocations. All hard
gates must pass (short-circuit on first failure), then any one soft trigger
fires. ``reason`` always names the blocking gate or the firing trigger.

Hard gates (in order):
  a. current_killzone is not None            → fail "no_killzone"
  b. htf_bias is not None                    → fail "no_htf_bias"
  c. nearest_dol is not None                 → fail "no_dol"
  d. not cooldown.is_cooling_down(snapshot)  → fail "cooldown_active"

Soft triggers (any one sufficient, after all gates pass):
  - FVG in fvgs["5m"] or fvgs["15m"]         → "fvg"
  - iFVG in ifvgs["5m"] or ifvgs["15m"]      → "ifvg"
  - sweep in recent_sweeps["15m"] or ["5m"]  → "sweep"
  - displacement in displacements["5m"] or ["1m"] → "displacement"

Config threading (Phase 3B): ``TriggerEngine`` accepts an optional
``config: TradingConfig = TradingConfig()``. The soft-trigger evaluation order
(today's dict insertion order) and the per-trigger timeframe ``or``-chains
(today's ``get("5m") or get("15m")`` literals) are read from config. With the
default config, behavior is byte-identical to pre-change code.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import TradingConfig

__all__ = ["TriggerResult", "TriggerEngine"]


@dataclass
class TriggerResult:
    """Outcome of a trigger evaluation."""

    should_trigger: bool
    reason: str


class TriggerEngine:
    """Evaluate hard gates then soft triggers against a snapshot.

    Args:
        cooldown: a CooldownState.
        config: trading config (defaults reproduce today's behavior exactly).
            The soft-trigger evaluation order and per-trigger timeframe
            or-chains are read from config.
    """

    def __init__(self, cooldown, config: TradingConfig = TradingConfig()):
        self.cooldown = cooldown
        self.config = config

    def soft_triggers_present(self, snapshot) -> dict[str, bool]:
        """Map each soft-trigger name → whether its condition holds.

        Single source of truth for the soft-trigger predicates. ``evaluate``
        fires the first present trigger (config order = priority); the
        calibrator uses the full map to count every present structure
        independent of priority. Adding a trigger here updates both at once.

        The evaluation order and per-trigger timeframe or-chains are read from
        ``self.config`` (default == today's literals).
        """
        tf_map = self.config.soft_trigger_tf_map_dict()
        out: dict[str, bool] = {}
        for name in self.config.soft_trigger_order:
            tfs = tf_map.get(name, ())
            if name == "fvg":
                present = bool(snapshot.fvgs.get(tfs[0]) or snapshot.fvgs.get(tfs[1])) if len(tfs) >= 2 else False
            elif name == "ifvg":
                present = bool(snapshot.ifvgs.get(tfs[0]) or snapshot.ifvgs.get(tfs[1])) if len(tfs) >= 2 else False
            elif name == "sweep":
                present = bool(snapshot.recent_sweeps.get(tfs[0]) or snapshot.recent_sweeps.get(tfs[1])) if len(tfs) >= 2 else False
            elif name == "displacement":
                present = bool(snapshot.displacements.get(tfs[0]) or snapshot.displacements.get(tfs[1])) if len(tfs) >= 2 else False
            else:
                # Unknown trigger name: conservatively absent.
                present = False
            out[name] = present
        return out

    def evaluate(self, snapshot) -> TriggerResult:
        # ── Hard gates (short-circuit on first failure) ────────────────────
        if snapshot.current_killzone is None:
            return TriggerResult(False, "no_killzone")
        if snapshot.htf_bias is None:
            return TriggerResult(False, "no_htf_bias")
        if snapshot.nearest_dol is None:
            return TriggerResult(False, "no_dol")
        if self.cooldown.is_cooling_down(snapshot):
            return TriggerResult(False, "cooldown_active")

        # ── Soft triggers (first present wins; order = priority) ────────────
        for name, present in self.soft_triggers_present(snapshot).items():
            if present:
                return TriggerResult(True, name)

        return TriggerResult(False, "no_soft_trigger")
