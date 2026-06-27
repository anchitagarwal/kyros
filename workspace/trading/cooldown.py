"""cooldown.py — tiered alert cooldown (ICT "one setup per session" discipline).

Cooldown is NOT a flat timer. It is tiered:
  - After a ``no_trade`` alert: cool down for 5 minutes (measured against
    ``snapshot.timestamp``, never wall clock), then clear.
  - After a ``long``/``short`` alert: cool down for the ENTIRE same killzone
    session — clears only when ``snapshot.current_killzone`` differs from
    ``last_alert_killzone`` (including transition to None or a new killzone).
  - Fresh state (no prior alert) → never cooling down.

Time comparison uses ``snapshot.timestamp`` for determinism during replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

__all__ = ["CooldownState", "NO_TRADE_COOLDOWN_MINUTES"]

NO_TRADE_COOLDOWN_MINUTES = 5


@dataclass
class CooldownState:
    """Mutable cooldown bookkeeping for the trading loop."""

    last_alert_time: datetime | None = None
    last_alert_bias: str = "no_trade"
    last_alert_killzone: str | None = None

    def is_cooling_down(self, snapshot) -> bool:
        """True if a new alert should be suppressed for ``snapshot``."""
        if self.last_alert_time is None:
            return False
        elapsed_min = (snapshot.timestamp - self.last_alert_time).total_seconds() / 60.0
        if self.last_alert_bias == "no_trade":
            return elapsed_min < NO_TRADE_COOLDOWN_MINUTES
        if self.last_alert_bias in ("long", "short"):
            # Directional alert: block for the entire same killzone session.
            return self.last_alert_killzone == snapshot.current_killzone
        return False

    def update(self, alert, snapshot) -> None:
        """Record the emitted alert's bias, time, and killzone."""
        self.last_alert_time = snapshot.timestamp
        self.last_alert_bias = alert.bias
        self.last_alert_killzone = snapshot.current_killzone
