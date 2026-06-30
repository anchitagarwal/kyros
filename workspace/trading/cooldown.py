"""cooldown.py — tiered alert cooldown (ICT "one setup per session" discipline).

Cooldown is NOT a flat timer. It is tiered:
  - After a ``no_trade`` alert: cool down for 5 minutes (measured against
    ``snapshot.timestamp``, never wall clock), then clear.
  - After a ``long``/``short`` alert: cool down for the ENTIRE same killzone
    session — clears only when ``snapshot.current_killzone`` differs from
    ``last_alert_killzone`` (including transition to None or a new killzone).
  - Fresh state (no prior alert) → never cooling down.

Time comparison uses ``snapshot.timestamp`` for determinism during replay.

Config threading (Phase 3B): ``CooldownState`` accepts an optional
``config: TradingConfig = TradingConfig()``. The no_trade cooldown duration
(today's hardcoded ``NO_TRADE_COOLDOWN_MINUTES = 5``) is read from
``config.no_trade_cooldown_minutes``. With the default config, behavior is
byte-identical to pre-change code. ``NO_TRADE_COOLDOWN_MINUTES`` is kept as a
module constant for backward compatibility (it equals the default).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .config import TradingConfig

__all__ = ["CooldownState", "NO_TRADE_COOLDOWN_MINUTES"]

# Backward-compatible module constant (equals the default config value).
NO_TRADE_COOLDOWN_MINUTES = 5


@dataclass
class CooldownState:
    """Mutable cooldown bookkeeping for the trading loop.

    Args:
        config: trading config (defaults reproduce today's behavior exactly).
            The no_trade cooldown duration is read from
            ``config.no_trade_cooldown_minutes``.
    """

    last_alert_time: datetime | None = None
    last_alert_bias: str = "no_trade"
    last_alert_killzone: str | None = None
    config: TradingConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = TradingConfig()

    def is_cooling_down(self, snapshot) -> bool:
        """True if a new alert should be suppressed for ``snapshot``."""
        if self.last_alert_time is None:
            return False
        elapsed_min = (snapshot.timestamp - self.last_alert_time).total_seconds() / 60.0
        if self.last_alert_bias == "no_trade":
            return elapsed_min < self.config.no_trade_cooldown_minutes
        if self.last_alert_bias in ("long", "short"):
            # Directional alert: block for the entire same killzone session.
            return self.last_alert_killzone == snapshot.current_killzone
        return False

    def update(self, alert, snapshot) -> None:
        """Record the emitted alert's bias, time, and killzone."""
        self.last_alert_time = snapshot.timestamp
        self.last_alert_bias = alert.bias
        self.last_alert_killzone = snapshot.current_killzone
