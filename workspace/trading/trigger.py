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
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TriggerResult", "TriggerEngine"]


@dataclass
class TriggerResult:
    """Outcome of a trigger evaluation."""

    should_trigger: bool
    reason: str


class TriggerEngine:
    """Evaluate hard gates then soft triggers against a snapshot."""

    def __init__(self, cooldown):
        self.cooldown = cooldown

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

        # ── Soft triggers (any one sufficient) ─────────────────────────────
        if snapshot.fvgs.get("5m") or snapshot.fvgs.get("15m"):
            return TriggerResult(True, "fvg")
        if snapshot.ifvgs.get("5m") or snapshot.ifvgs.get("15m"):
            return TriggerResult(True, "ifvg")
        if snapshot.recent_sweeps.get("15m") or snapshot.recent_sweeps.get("5m"):
            return TriggerResult(True, "sweep")
        if snapshot.displacements.get("5m") or snapshot.displacements.get("1m"):
            return TriggerResult(True, "displacement")

        return TriggerResult(False, "no_soft_trigger")
