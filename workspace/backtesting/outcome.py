"""outcome.py — deterministically resolve an AlertPayload into a trade outcome.

The OutcomeSimulator walks strictly-subsequent candles (candles AFTER the
alert timestamp) to determine whether a trade filled, won, lost, expired, or
never filled. It uses the LLM's own entry_zone/stop/target without
modification — Python owns the truth.

CRITICAL LOOKAHEAD RULE
-----------------------
``simulate()`` does NOT validate that ``subsequent_candles`` are strictly
after ``alert.timestamp`` — that is the CALLER's responsibility (the spec is
explicit: "No validation of this inside simulate() — caller's responsibility").
The BacktestEngine slices candles strictly after the alert timestamp before
passing them here. Passing the alert candle itself as the first subsequent
candle would leak the alert bar's high/low into fill/resolution detection,
producing an incorrect (and optimistic) outcome. The test
``test_alert_candle_as_subsequent_gives_wrong_outcome`` documents this.

Resolution rules
----------------
1. no_trade → immediate return, all numeric fields None.
2. Pre-fill invalidation (cancel): while waiting for a fill, if price reaches
   the target (the targeted move happened without us) or breaches the stop
   (the invalidation level traded through) before the entry_zone fills, the
   setup is cancelled — no trade is taken. Conservative: a candle that tags
   target/stop AND overlaps the entry_zone in the same bar cancels rather than
   fills, so this check PRECEDES the fill check. Long: high >= target or
   low <= stop. Short: low <= target or high >= stop.
3. Fill: iterate subsequent_candles until the candle's price range overlaps
   the entry_zone. fill_price = entry_mid = (entry_zone[0]+entry_zone[1])/2,
   clamped to the fill candle's [low, high] so an edge-graze fill never uses a
   price the bar never traded.
4. Resolution begins on the candle AFTER the fill candle (the fill candle
   itself never resolves). Long win: high >= target. Long loss: low <= stop.
   Short win: low <= target. Short loss: high >= stop. Both in the same
   candle → loss (conservative).
5. Killzone expiry: compare each candle's timestamp against alert.valid_until.
   No fill by valid_until → no_fill. Filled but unresolved by valid_until →
   expired.

actual_rr
---------
- Win:  (exit_price - fill_price) / abs(fill_price - stop) for long;
        (fill_price - exit_price) / abs(fill_price - stop) for short.
        exit_price = alert.target (nominal — wins never assume better-than-target fills).
- Loss: negative of the same formula. exit_price is the REALIZED stop-out: a
        clean intrabar stop fills at alert.stop (exactly -1.0); a gap through
        the stop fills at the candle open (worse than -1.0).
- no_fill / no_trade / expired / cancelled → None.

Pure function of (alert, candles): no I/O, no clock, no randomness, no LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

__all__ = ["TradeOutcome", "OutcomeSimulator"]


@dataclass
class TradeOutcome:
    """The resolved outcome of a single alert.

    Fields:
        result: "win" | "loss" | "expired" | "cancelled" | "no_fill" | "no_trade"
        candles_to_fill: 1-based count of candles from the alert until fill
            (None if no fill / no_trade).
        candles_to_resolution: 1-based count of candles from the alert until
            resolution (None if not resolved).
        fill_price: the price at which the trade filled — entry_mid clamped to
            the fill candle's [low, high] range (None if no fill).
        exit_price: the price at which the trade resolved (target for win,
            realized stop for loss; None if not resolved).
        actual_rr: realized risk-reward (positive for win, negative for loss;
            None for no_fill / no_trade / expired / cancelled).
    """

    result: str
    candles_to_fill: int | None = None
    candles_to_resolution: int | None = None
    fill_price: float | None = None
    exit_price: float | None = None
    actual_rr: float | None = None

    def to_dict(self) -> dict:
        return {
            "result": self.result,
            "candles_to_fill": self.candles_to_fill,
            "candles_to_resolution": self.candles_to_resolution,
            "fill_price": self.fill_price,
            "exit_price": self.exit_price,
            "actual_rr": self.actual_rr,
        }


def _parse_ts(ts: Any) -> datetime:
    """Parse a timestamp (datetime or ISO-8601 str) to a datetime (maybe naive)."""
    if isinstance(ts, datetime):
        return ts
    dt = datetime.fromisoformat(str(ts))
    return dt


# Project display timezone. Candle timestamps arrive tz-aware ET; an LLM-produced
# valid_until may be naive — interpret naive datetimes as ET (the framing the LLM is
# given) before comparing, so aware and naive values never collide.
_NY = ZoneInfo("America/New_York")


def _to_utc(dt: datetime) -> datetime:
    """Coerce a datetime to aware UTC; naive inputs are assumed to be ET."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_NY)
    return dt.astimezone(timezone.utc)


class OutcomeSimulator:
    """Resolve an AlertPayload into a TradeOutcome using subsequent candles."""

    def simulate(self, alert, subsequent_candles: list[dict]) -> TradeOutcome:
        """Resolve ``alert`` against ``subsequent_candles``.

        Args:
            alert: an AlertPayload (or compatible object with .bias,
                .entry_zone, .stop, .target, .valid_until).
            subsequent_candles: candles strictly after alert.timestamp.
                PRECONDITION (caller's responsibility): every candle.timestamp
                > alert.timestamp. Not validated here per spec.

        Returns:
            A TradeOutcome.
        """
        # no_trade: immediate return, no candle iteration.
        if alert.bias == "no_trade":
            return TradeOutcome(result="no_trade")

        entry_low, entry_high = float(alert.entry_zone[0]), float(alert.entry_zone[1])
        stop = float(alert.stop)
        target = float(alert.target)
        entry_mid = (entry_low + entry_high) / 2.0
        risk = abs(entry_mid - stop)
        is_long = alert.bias == "long"

        # Parse valid_until once (empty string → no expiry).
        valid_until_dt = self._parse_valid_until(alert.valid_until)

        # ── Step 1: fill detection ──────────────────────────────────────────
        fill_index = None
        for i, candle in enumerate(subsequent_candles):
            # Killzone expiry check BEFORE fill: if this candle is at/after
            # valid_until and we haven't filled, it's no_fill.
            if valid_until_dt is not None:
                c_ts = _parse_ts(candle["timestamp"])
                if _to_utc(c_ts) >= _to_utc(valid_until_dt):
                    return TradeOutcome(result="no_fill")

            c_low = float(candle["low"])
            c_high = float(candle["high"])

            # Pre-fill invalidation (cancel): if price reaches the target (the
            # targeted move happened without us) or breaches the stop (the
            # invalidation level traded through) before the entry_zone fills,
            # the setup is cancelled — no trade taken. Conservative: a candle
            # that tags target/stop AND overlaps the entry_zone in the same bar
            # cancels rather than fills, so this precedes the fill check.
            if is_long:
                invalidated = c_high >= target or c_low <= stop
            else:
                invalidated = c_low <= target or c_high >= stop
            if invalidated:
                return TradeOutcome(result="cancelled")

            # Fill: price range overlaps entry_zone.
            # Long fill: candle.low <= entry_zone[1] and candle.high >= entry_zone[0]
            # Short fill: same condition (range overlap).
            if c_low <= entry_high and c_high >= entry_low:
                fill_index = i
                break

        if fill_index is None:
            # No fill encountered before the end of subsequent_candles.
            # If valid_until was set and we ran out of candles before it,
            # it's still no_fill (we never filled). If valid_until was not
            # set, also no_fill (ran out of data without filling).
            return TradeOutcome(result="no_fill")

        # Fill at entry_mid, but clamp to the fill candle's actual range: if the
        # candle only grazes a zone edge, entry_mid can lie outside [low, high]
        # — a price the market never traded on that bar. Clamping yields the
        # worst achievable fill within the candle (conservative in both
        # directions: a higher entry for longs, a lower entry for shorts).
        fill_candle = subsequent_candles[fill_index]
        fill_price = max(float(fill_candle["low"]),
                         min(float(fill_candle["high"]), entry_mid))
        candles_to_fill = fill_index + 1  # 1-based count from alert

        # ── Step 2: resolution (from candle AFTER fill candle) ──────────────
        for j in range(fill_index + 1, len(subsequent_candles)):
            candle = subsequent_candles[j]
            c_ts = _parse_ts(candle["timestamp"])

            # Killzone expiry check: if this candle is at/after valid_until
            # and we haven't resolved, it's expired.
            if valid_until_dt is not None and _to_utc(c_ts) >= _to_utc(valid_until_dt):
                return TradeOutcome(
                    result="expired",
                    candles_to_fill=candles_to_fill,
                    candles_to_resolution=None,
                    fill_price=fill_price,
                    exit_price=None,
                    actual_rr=None,
                )

            c_high = float(candle["high"])
            c_low = float(candle["low"])

            if is_long:
                win = c_high >= target
                loss = c_low <= stop
            else:
                win = c_low <= target
                loss = c_high >= stop

            if win and loss:
                # Ambiguous: both hit in the same candle → loss (conservative).
                exit_price = self._realized_loss_exit(is_long, stop, float(candle["open"]))
                actual_rr = self._compute_rr(alert.bias, fill_price, exit_price, stop, risk)
                return TradeOutcome(
                    result="loss",
                    candles_to_fill=candles_to_fill,
                    candles_to_resolution=j + 1,
                    fill_price=fill_price,
                    exit_price=exit_price,
                    actual_rr=actual_rr,
                )
            if win:
                exit_price = target
                actual_rr = self._compute_rr(alert.bias, fill_price, exit_price, stop, risk)
                return TradeOutcome(
                    result="win",
                    candles_to_fill=candles_to_fill,
                    candles_to_resolution=j + 1,
                    fill_price=fill_price,
                    exit_price=exit_price,
                    actual_rr=actual_rr,
                )
            if loss:
                exit_price = self._realized_loss_exit(is_long, stop, float(candle["open"]))
                actual_rr = self._compute_rr(alert.bias, fill_price, exit_price, stop, risk)
                return TradeOutcome(
                    result="loss",
                    candles_to_fill=candles_to_fill,
                    candles_to_resolution=j + 1,
                    fill_price=fill_price,
                    exit_price=exit_price,
                    actual_rr=actual_rr,
                )

        # Filled but neither stop nor target hit before the end of
        # subsequent_candles (and no valid_until expiry encountered).
        return TradeOutcome(
            result="expired",
            candles_to_fill=candles_to_fill,
            candles_to_resolution=None,
            fill_price=fill_price,
            exit_price=None,
            actual_rr=None,
        )

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_valid_until(valid_until: Any) -> datetime | None:
        """Parse alert.valid_until into an aware datetime, or None if unset."""
        if valid_until is None or valid_until == "":
            return None
        try:
            return _parse_ts(valid_until)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _realized_loss_exit(is_long: bool, stop: float, candle_open: float) -> float:
        """Realized stop-out price for a losing candle.

        A clean intrabar stop fills at ``stop`` (exactly -1.0R). A gap through
        the stop (the candle opens beyond it) fills at the open, giving a loss
        worse than -1.0R. Long fills at min(stop, open); short at max(stop, open).
        """
        if is_long:
            return min(stop, candle_open)
        return max(stop, candle_open)

    @staticmethod
    def _compute_rr(bias: str, fill_price: float, exit_price: float,
                    stop: float, risk: float) -> float:
        """Compute realized risk-reward, signed by direction.

        Win: positive. Loss: negative. risk == 0 → 0.0 (degenerate, avoids
        division by zero; should not occur for validated alerts).
        """
        if risk == 0:
            return 0.0
        if bias == "long":
            return (exit_price - fill_price) / risk
        else:  # short
            return (fill_price - exit_price) / risk
