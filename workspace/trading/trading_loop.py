"""trading_loop.py — orchestrate the full alert pipeline per candle.

Strict ordering per candle:
  update → build → evaluate → (reason → validate_rr → cooldown.update → emit)

The LLM is called ONLY when the trigger engine says should_trigger. validate_rr
runs before every emit. cooldown.update is called even for no_trade alerts
(drives the 5-min tier). emit appends one JSON line to alerts.jsonl + stdout.

No broker, no Telegram, no orders. All file writes confined to workspace/.
"""

from __future__ import annotations

import json
import sys

from .alert import AlertPayload, validate_rr

__all__ = ["TradingLoop"]


class TradingLoop:
    """Drive the candle source through the full pipeline and emit alerts."""

    def __init__(
        self,
        source,
        window,
        builder,
        trigger,
        agent,
        cooldown,
        output_path: str = "workspace/alerts.jsonl",
    ):
        self.source = source
        self.window = window
        self.builder = builder
        self.trigger = trigger
        self.agent = agent
        self.cooldown = cooldown
        self.output_path = output_path

    def run(self) -> None:
        """Consume the source until exhausted, emitting alerts as triggered."""
        while not self.source.is_done():
            candles = self.source.next()
            if candles is None:
                break
            self.window.update(candles)
            snapshot = self.builder.build(self.window)
            result = self.trigger.evaluate(snapshot)
            if not result.should_trigger:
                continue
            alert = self.agent.reason(snapshot)
            alert = validate_rr(alert)
            self.cooldown.update(alert, snapshot)
            self._emit(alert, snapshot)

    def _emit(self, alert: AlertPayload, snapshot) -> None:
        """Append one JSON line to output_path and print to stdout."""
        record = alert.to_dict()
        record["timestamp"] = str(snapshot.timestamp)
        record["instrument"] = snapshot.instrument
        record["current_price"] = round(snapshot.current_price, 2)
        line = json.dumps(record, default=str)
        with open(self.output_path, "a") as f:
            f.write(line + "\n")
        print(line)
