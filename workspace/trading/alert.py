"""alert.py — AlertPayload model and the authoritative R:R validator.

``validate_rr`` runs AFTER the LLM produces an AlertPayload and BEFORE the
loop emits it. Python recomputes the risk-reward from the geometry
(entry_zone, stop, target) and overrides any LLM-supplied value. Sub-1:1 or
degenerate setups are downgraded to ``no_trade``.

The LLM's arithmetic is NEVER trusted — Python owns the truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

__all__ = ["AlertPayload", "validate_rr"]

_VALID_BIAS = {"long", "short", "no_trade"}
_VALID_MODEL = {"2022", "unicorn", "ifvg", "silver_bullet", "breaker", "none"}


@dataclass
class AlertPayload:
    """A single trade alert (or no_trade decision) emitted by the pipeline."""

    bias: str = "no_trade"
    model: str = "none"
    conviction: int = 0
    entry_zone: tuple[float, float] = (0.0, 0.0)
    stop: float = 0.0
    target: float = 0.0
    dol: dict = field(default_factory=lambda: {"level": 0.0, "type": "", "timeframe": ""})
    risk_reward: float = 0.0
    rationale: str = ""
    killzone: str = ""
    valid_until: str = ""
    no_trade_reason: str | None = None

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict for alerts.jsonl."""
        return {
            "bias": self.bias,
            "model": self.model,
            "conviction": self.conviction,
            "entry_zone": list(self.entry_zone),
            "stop": self.stop,
            "target": self.target,
            "dol": self.dol,
            "risk_reward": self.risk_reward,
            "rationale": self.rationale,
            "killzone": self.killzone,
            "valid_until": self.valid_until,
            "no_trade_reason": self.no_trade_reason,
        }


def validate_rr(alert: AlertPayload) -> AlertPayload:
    """Recompute R:R from geometry; downgrade sub-1:1 / degenerate to no_trade.

    - entry_mid = (entry_zone[0] + entry_zone[1]) / 2
    - risk = abs(entry_mid - stop)
    - risk == 0 → no_trade, reason "degenerate_stop" (no division)
    - rr = abs(target - entry_mid) / risk
    - risk_reward is ALWAYS overwritten with round(rr, 2)
    - rr < 1.0 → no_trade, reason "rr_below_1"
    """
    entry_mid = (alert.entry_zone[0] + alert.entry_zone[1]) / 2.0
    risk = abs(entry_mid - alert.stop)

    if risk == 0:
        return replace(
            alert,
            bias="no_trade",
            no_trade_reason="degenerate_stop",
            risk_reward=0.0,
        )

    rr = abs(alert.target - entry_mid) / risk
    updated = replace(alert, risk_reward=round(rr, 2))

    if rr < 1.0:
        return replace(
            updated,
            bias="no_trade",
            no_trade_reason="rr_below_1",
        )
    return updated


def parse_llm_json(data: dict) -> AlertPayload:
    """Parse an LLM JSON dict into an AlertPayload with safe defaults.

    On any structural problem (missing keys, wrong types, invalid enums),
    returns a no_trade AlertPayload with no_trade_reason="llm_parse_error".
    Never raises.
    """
    try:
        if not isinstance(data, dict):
            return AlertPayload(bias="no_trade", no_trade_reason="llm_parse_error")

        bias = data.get("bias", "no_trade")
        model = data.get("model", "none")
        if bias not in _VALID_BIAS:
            bias = "no_trade"
        if model not in _VALID_MODEL:
            model = "none"

        # entry_zone is REQUIRED: a 2-element list of numbers.
        if "entry_zone" not in data:
            return AlertPayload(bias="no_trade", no_trade_reason="llm_parse_error")
        ez = data["entry_zone"]
        if not isinstance(ez, (list, tuple)) or len(ez) != 2:
            return AlertPayload(bias="no_trade", no_trade_reason="llm_parse_error")
        entry_zone = (float(ez[0]), float(ez[1]))

        # stop and target are REQUIRED numeric fields.
        if "stop" not in data or "target" not in data:
            return AlertPayload(bias="no_trade", no_trade_reason="llm_parse_error")
        stop = float(data["stop"])
        target = float(data["target"])
        conviction = int(data.get("conviction", 0))
        dol = data.get("dol", {})
        if not isinstance(dol, dict):
            dol = {"level": 0.0, "type": "", "timeframe": ""}
        rationale = str(data.get("rationale", ""))
        killzone = str(data.get("killzone", ""))
        valid_until = str(data.get("valid_until", ""))
        ntr = data.get("no_trade_reason")
        if ntr is not None:
            ntr = str(ntr)

        return AlertPayload(
            bias=bias,
            model=model,
            conviction=conviction,
            entry_zone=entry_zone,
            stop=stop,
            target=target,
            dol=dol,
            risk_reward=float(data.get("risk_reward", 0.0)),
            rationale=rationale,
            killzone=killzone,
            valid_until=valid_until,
            no_trade_reason=ntr,
        )
    except (TypeError, ValueError):
        return AlertPayload(bias="no_trade", no_trade_reason="llm_parse_error")
