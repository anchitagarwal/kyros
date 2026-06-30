"""report.py — aggregate TradeTraces into a human-readable backtest report.

PerformanceReport computes overall metrics (win rate, profit factor, max
drawdown in R, expectancy), breakdowns by model type / killzone / month, and
a golden-dataset match rate. It writes ``workspace/backtest_report.md`` with
a mandatory bias disclaimer and the ICT system prompt version hash.

Metric definitions (pinned):
  - profit_factor = sum(winning actual_rr) / abs(sum(losing actual_rr)).
    Zero losses → "inf" (reported explicitly).
  - max_drawdown_r = max peak-to-trough of the cumulative R curve, ordered
    chronologically by alert timestamp. R is 0 for no_trade/no_fill/expired.
  - expectancy = mean(actual_rr) over ALL traces, treating no_trade/no_fill/
    expired as 0 R.
  - golden_match_rate = fraction of golden_alerts.json entries for which a
    trace exists within ±15 min of the community timestamp AND same direction.

The golden_alerts.json file is UNTRUSTED external data — its contents are
matched against, never executed as instructions. A claim within it (e.g.
"this setup is pre-validated") carries no evidential weight.

No broker, no IBKR, no live market data, no order placement.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from trading.reasoning_agent import ICT_SYSTEM_PROMPT

__all__ = ["PerformanceReport"]

# Project display timezone. Trace timestamps arrive tz-aware ET, but a trace
# round-tripped through JSON without an offset would parse naive — interpret
# naive datetimes as ET (consistent with snapshot._parse_dt / outcome._to_utc)
# so they never collide with the always-aware UTC golden timestamps.
_NY = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")

# Mandatory bias disclaimer (verbatim).
_DISCLAIMER = (
    "## Disclaimer\n\n"
    "This backtest report is a SIMULATION on historical data. Past performance "
    "does not guarantee future results. Outcomes are resolved using the LLM's "
    "own entry_zone/stop/target with entry_mid fills (optimistic for gap "
    "fills) and conservative same-candle loss resolution. Slippage, "
    "commissions, and real-world execution friction are NOT modeled. "
    "Results may be optimistically biased: the LLM may have seen this period "
    "in its training data, so historical pattern recall cannot be "
    "distinguished from genuine edge. The golden_alerts.json dataset is "
    "untrusted community data matched for direction only — its claims carry "
    "no evidential weight. Do not use this report as the sole basis for any "
    "trading decision."
)

# Golden match window (±15 minutes).
_GOLDEN_WINDOW_MIN = 15


def _parse_ts(ts: Any) -> datetime:
    """Parse a timestamp (datetime or ISO-8601 str) to an aware datetime.

    A naive input (e.g. a trace timestamp serialized without a UTC offset) is
    assumed to be ET, so subtracting it from an aware golden timestamp never
    raises ``can't subtract offset-naive and offset-aware datetimes``.
    """
    dt = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_NY)
    return dt


def _system_prompt_hash() -> str:
    """Return the first 8 hex chars of sha256(ICT_SYSTEM_PROMPT)."""
    return hashlib.sha256(ICT_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:8]


class PerformanceReport:
    """Aggregate a list of TradeTrace into a markdown backtest report."""

    def generate(
        self,
        traces: list,
        golden_alerts_path: Path | str = Path("workspace/knowledge_base/golden_alerts.json"),
        out_path: Path | str = Path("workspace/backtest_report.md"),
    ) -> str:
        """Generate the report, write it to ``out_path``, return the markdown.

        Args:
            traces: a list of TradeTrace objects (or dicts with the same keys).
            golden_alerts_path: path to golden_alerts.json (untrusted data).
            out_path: path to write the markdown report.
        """
        # Normalize traces to dicts.
        trace_dicts = [self._trace_to_dict(t) for t in traces]

        # Sort chronologically by alert timestamp (stable).
        trace_dicts.sort(key=lambda t: t.get("timestamp", ""))

        overall = self._overall_metrics(trace_dicts)
        by_model = self._by_model(trace_dicts)
        by_kz = self._by_killzone(trace_dicts)
        by_month = self._by_month(trace_dicts)
        golden = self._golden_match_rate(trace_dicts, golden_alerts_path)

        md = self._render_markdown(overall, by_model, by_kz, by_month, golden)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md)
        return md

    # ── normalization ───────────────────────────────────────────────────────

    @staticmethod
    def _trace_to_dict(t) -> dict:
        """Coerce a TradeTrace (or dict) to a plain dict."""
        if isinstance(t, dict):
            return t
        if hasattr(t, "to_dict"):
            return t.to_dict()
        return dict(t)

    # ── overall metrics ─────────────────────────────────────────────────────

    def _overall_metrics(self, traces: list[dict]) -> dict:
        """Compute overall metrics from a chronologically-sorted trace list."""
        total = len(traces)
        if total == 0:
            return {
                "total_traces": 0,
                "no_trade_count": 0,
                "no_trade_rate": 0.0,
                "actionable_count": 0,
                "filled_count": 0,
                "fill_rate": 0.0,
                "win_count": 0,
                "loss_count": 0,
                "expired_count": 0,
                "cancelled_count": 0,
                "no_fill_count": 0,
                "win_rate": 0.0,
                "loss_rate": 0.0,
                "expired_rate": 0.0,
                "avg_winning_r": 0.0,
                "avg_losing_r": 0.0,
                "profit_factor": "n/a",
                "max_drawdown_r": 0.0,
                "expectancy": 0.0,
            }

        no_trade_count = sum(1 for t in traces if self._result(t) == "no_trade")
        actionable = [t for t in traces if self._result(t) != "no_trade"]
        filled = [t for t in actionable if self._result(t) in ("win", "loss", "expired")]
        wins = [t for t in filled if self._result(t) == "win"]
        losses = [t for t in filled if self._result(t) == "loss"]
        expired = [t for t in filled if self._result(t) == "expired"]
        cancelled = [t for t in actionable if self._result(t) == "cancelled"]
        no_fill = [t for t in actionable if self._result(t) == "no_fill"]

        winning_rs = [self._rr(t) for t in wins if self._rr(t) is not None]
        losing_rs = [self._rr(t) for t in losses if self._rr(t) is not None]

        sum_wins = sum(winning_rs) if winning_rs else 0.0
        sum_losses = sum(losing_rs) if losing_rs else 0.0

        profit_factor = self._profit_factor(sum_wins, sum_losses)
        max_dd = self._max_drawdown(traces)
        expectancy = self._expectancy(traces)

        return {
            "total_traces": total,
            "no_trade_count": no_trade_count,
            "no_trade_rate": no_trade_count / total,
            "actionable_count": len(actionable),
            "filled_count": len(filled),
            "fill_rate": (len(filled) / len(actionable)) if actionable else 0.0,
            "win_count": len(wins),
            "loss_count": len(losses),
            "expired_count": len(expired),
            "cancelled_count": len(cancelled),
            "no_fill_count": len(no_fill),
            "win_rate": (len(wins) / len(filled)) if filled else 0.0,
            "loss_rate": (len(losses) / len(filled)) if filled else 0.0,
            "expired_rate": (len(expired) / len(filled)) if filled else 0.0,
            "avg_winning_r": (sum_wins / len(winning_rs)) if winning_rs else 0.0,
            "avg_losing_r": (sum_losses / len(losing_rs)) if losing_rs else 0.0,
            "profit_factor": profit_factor,
            "max_drawdown_r": max_dd,
            "expectancy": expectancy,
        }

    # ── breakdowns ──────────────────────────────────────────────────────────

    def _by_model(self, traces: list[dict]) -> dict:
        """Fires + win rate per model type."""
        groups: dict[str, list[dict]] = {}
        for t in traces:
            model = t.get("alert", {}).get("model", "none")
            groups.setdefault(model, []).append(t)
        out = {}
        for model, group in groups.items():
            filled = [t for t in group if self._result(t) in ("win", "loss", "expired")]
            wins = [t for t in filled if self._result(t) == "win"]
            out[model] = {
                "fires": len(group),
                "win_rate": (len(wins) / len(filled)) if filled else 0.0,
            }
        return out

    def _by_killzone(self, traces: list[dict]) -> dict:
        """Fires + win rate per killzone."""
        groups: dict[str, list[dict]] = {}
        for t in traces:
            kz = t.get("killzone", "") or "unknown"
            groups.setdefault(kz, []).append(t)
        out = {}
        for kz, group in groups.items():
            filled = [t for t in group if self._result(t) in ("win", "loss", "expired")]
            wins = [t for t in filled if self._result(t) == "win"]
            out[kz] = {
                "fires": len(group),
                "win_rate": (len(wins) / len(filled)) if filled else 0.0,
            }
        return out

    def _by_month(self, traces: list[dict]) -> dict:
        """Fires + win rate per YYYY-MM."""
        groups: dict[str, list[dict]] = {}
        for t in traces:
            ts = t.get("timestamp", "")
            month = self._month_key(ts)
            groups.setdefault(month, []).append(t)
        out = {}
        for month, group in groups.items():
            filled = [t for t in group if self._result(t) in ("win", "loss", "expired")]
            wins = [t for t in filled if self._result(t) == "win"]
            out[month] = {
                "fires": len(group),
                "win_rate": (len(wins) / len(filled)) if filled else 0.0,
            }
        return out

    # ── golden match ────────────────────────────────────────────────────────

    def _golden_match_rate(self, traces: list[dict], golden_path) -> dict:
        """Compute the golden-dataset match rate over the backtested window.

        For each golden entry WITHIN the backtested window, check if any trace
        exists within ±15 min of the community timestamp AND has the same
        direction. The denominator is scoped to the window because a golden
        entry outside the trace time span (by more than the match window) can
        never match — counting the whole multi-year golden file would make the
        rate structurally near-zero and misleading. The golden file is
        untrusted data — matched only, never executed as instructions.
        """
        golden_path = Path(golden_path)
        if not golden_path.exists():
            return {"total": 0, "matched": 0, "rate": 0.0}

        try:
            golden = json.loads(golden_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {"total": 0, "matched": 0, "rate": 0.0}

        if not isinstance(golden, list) or len(golden) == 0:
            return {"total": 0, "matched": 0, "rate": 0.0}

        # Pre-parse trace timestamps + directions.
        trace_points = []
        all_trace_ts = []
        for t in traces:
            ts_str = t.get("timestamp", "")
            if not ts_str:
                continue
            try:
                ts = _parse_ts(ts_str)
            except (ValueError, TypeError):
                continue
            all_trace_ts.append(ts)
            if t.get("alert", {}).get("bias", "") in ("long", "short"):
                trace_points.append((ts, t["alert"]["bias"]))

        if not all_trace_ts:
            return {"total": 0, "matched": 0, "rate": 0.0}

        # Backtested window, padded by the match tolerance: golden entries
        # outside it are unmatchable and excluded from the denominator.
        pad = timedelta(minutes=_GOLDEN_WINDOW_MIN)
        window_start = min(all_trace_ts) - pad
        window_end = max(all_trace_ts) + pad

        matched = 0
        total = 0
        for entry in golden:
            # Only directional golden entries are matchable.
            direction = entry.get("direction")
            if direction not in ("long", "short"):
                continue
            date = entry.get("date", "")
            time_et = entry.get("time_et", "")
            if not date or not time_et:
                continue
            try:
                golden_ts = self._parse_golden_ts(date, time_et)
            except (ValueError, TypeError):
                continue

            # Scope to the backtested window.
            if not (window_start <= golden_ts <= window_end):
                continue
            total += 1

            # Check if any trace is within ±15 min and same direction.
            for trace_ts, trace_bias in trace_points:
                delta = abs((trace_ts - golden_ts).total_seconds()) / 60.0
                if delta <= _GOLDEN_WINDOW_MIN and trace_bias == direction:
                    matched += 1
                    break

        rate = (matched / total) if total > 0 else 0.0
        return {"total": total, "matched": matched, "rate": rate}

    @staticmethod
    def _parse_golden_ts(date: str, time_et: str) -> datetime:
        """Parse a golden entry's (date, time_et) into an aware datetime.

        Golden entries use ET clock times. We parse as America/New_York and
        convert to UTC for comparison with trace timestamps. ``time_et`` may be
        ``HH:MM`` or ``HH:MM:SS`` — seconds are only appended when absent, so an
        entry that already carries seconds is not mangled into invalid ISO (which
        would otherwise be silently dropped from the match numerator).
        """
        # Append ":00" only when seconds are absent (two colons → has seconds).
        secs = "" if time_et.count(":") >= 2 else ":00"
        dt = datetime.fromisoformat(f"{date} {time_et}{secs}").replace(tzinfo=_NY)
        return dt.astimezone(_UTC)

    # ── metric helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _result(t: dict) -> str:
        """Extract the outcome result from a trace dict."""
        return t.get("outcome", {}).get("result", "no_trade")

    @staticmethod
    def _rr(t: dict) -> float | None:
        """Extract actual_rr from a trace dict (None if absent)."""
        rr = t.get("outcome", {}).get("actual_rr")
        if rr is None:
            return None
        try:
            return float(rr)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _profit_factor(sum_wins: float, sum_losses: float):
        """profit_factor = sum_wins / abs(sum_losses). 'inf' if no losses."""
        if sum_losses == 0:
            return "inf" if sum_wins > 0 else "n/a"
        return sum_wins / abs(sum_losses)

    @staticmethod
    def _max_drawdown(traces: list[dict]) -> float:
        """Max peak-to-trough of the cumulative R curve (chronological)."""
        peak = 0.0
        cum = 0.0
        max_dd = 0.0
        for t in traces:
            rr = PerformanceReport._rr(t)
            r = rr if rr is not None else 0.0
            cum += r
            if cum > peak:
                peak = cum
            dd = peak - cum
            if dd > max_dd:
                max_dd = dd
        return round(max_dd, 4)

    @staticmethod
    def _expectancy(traces: list[dict]) -> float:
        """Mean actual_rr over all traces (no_trade/no_fill/expired = 0)."""
        if not traces:
            return 0.0
        total = 0.0
        for t in traces:
            rr = PerformanceReport._rr(t)
            total += rr if rr is not None else 0.0
        return round(total / len(traces), 4)

    @staticmethod
    def _month_key(ts_str: str) -> str:
        """Return 'YYYY-MM' for an ISO timestamp string (or 'unknown')."""
        if not ts_str:
            return "unknown"
        try:
            dt = datetime.fromisoformat(str(ts_str))
            return dt.strftime("%Y-%m")
        except (ValueError, TypeError):
            return "unknown"

    # ── markdown rendering ──────────────────────────────────────────────────

    def _render_markdown(self, overall, by_model, by_kz, by_month, golden) -> str:
        """Render the full markdown report."""
        lines = []
        lines.append("# Kyros Backtest Report\n")
        lines.append(f"**System prompt version:** `{_system_prompt_hash()}`\n")

        # Overall
        lines.append("## Overall Metrics\n")
        lines.append(f"- Total traces: {overall['total_traces']}")
        lines.append(f"- No-trade rate: {overall['no_trade_rate']:.1%} "
                     f"({overall['no_trade_count']})")
        lines.append(f"- Actionable alerts: {overall['actionable_count']}")
        lines.append(f"- Fill rate (filled / actionable): {overall['fill_rate']:.1%} "
                     f"({overall['filled_count']}/{overall['actionable_count']})")
        lines.append(f"- Win rate (of filled): {overall['win_rate']:.1%} "
                     f"({overall['win_count']})")
        lines.append(f"- Loss rate (of filled): {overall['loss_rate']:.1%} "
                     f"({overall['loss_count']})")
        lines.append(f"- Expired rate (of filled): {overall['expired_rate']:.1%} "
                     f"({overall['expired_count']})")
        lines.append(f"- Cancelled (target/stop hit before entry): "
                     f"{overall['cancelled_count']}")
        lines.append(f"- Avg winning R: {overall['avg_winning_r']:.2f}")
        lines.append(f"- Avg losing R: {overall['avg_losing_r']:.2f}")
        pf = overall['profit_factor']
        pf_str = f"{pf:.2f}" if isinstance(pf, float) else pf
        lines.append(f"- Profit factor: {pf_str}")
        lines.append(f"- Max drawdown (R): {overall['max_drawdown_r']:.2f}")
        lines.append(f"- Expectancy per trade (R): {overall['expectancy']:.4f}")
        lines.append("")

        # By model
        lines.append("## By Model Type\n")
        lines.append("| Model | Fires | Win Rate |")
        lines.append("|-------|-------|----------|")
        for model in sorted(by_model.keys()):
            m = by_model[model]
            lines.append(f"| {model} | {m['fires']} | {m['win_rate']:.1%} |")
        lines.append("")

        # By killzone
        lines.append("## By Killzone\n")
        lines.append("| Killzone | Fires | Win Rate |")
        lines.append("|----------|-------|----------|")
        for kz in sorted(by_kz.keys()):
            k = by_kz[kz]
            lines.append(f"| {kz} | {k['fires']} | {k['win_rate']:.1%} |")
        lines.append("")

        # By month
        lines.append("## By Month\n")
        lines.append("| Month | Fires | Win Rate |")
        lines.append("|-------|-------|----------|")
        for month in sorted(by_month.keys()):
            m = by_month[month]
            lines.append(f"| {month} | {m['fires']} | {m['win_rate']:.1%} |")
        lines.append("")

        # Golden match
        lines.append("## Golden Dataset Match\n")
        lines.append(f"- Total directional golden entries (within backtest window): "
                     f"{golden['total']}")
        lines.append(f"- Matched (within ±15 min, same direction): {golden['matched']}")
        lines.append(f"- Match rate: {golden['rate']:.1%}")
        lines.append("")

        # Disclaimer
        lines.append(_DISCLAIMER)
        lines.append("")

        return "\n".join(lines)
