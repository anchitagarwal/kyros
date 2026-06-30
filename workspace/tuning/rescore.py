"""rescore.py — apply PostLLMParams to recorded traces by pure arithmetic.

Re-scoring NEVER re-simulates. A recorded alert that fails any post-LLM filter
is downgraded to ``no_trade`` (R=0); a surviving alert keeps its recorded
outcome verbatim. This is the Tier-1 (free) operation: no LLM calls, no
OutcomeSimulator, no candle data needed.

Filters (a taken/directional alert is downgraded to no_trade if ANY holds):
  - ``alert.conviction < p.conviction_min``
  - recomputed ``risk_reward < p.rr_min`` (recomputed from recorded
    entry/stop/target, NOT read from the possibly-stale stored field)
  - ``alert.model not in p.allowed_models`` (unless allowed_models == ALL)
  - ``alert.killzone not in p.allowed_killzones`` (unless == ALL)

R:R recompute mirrors ``validate_rr`` EXACTLY (the single source of truth in
the trading layer):
    entry_mid = (entry_zone[0] + entry_zone[1]) / 2
    risk = abs(entry_mid - stop)
    risk == 0 → degenerate (downgrade to no_trade, matching validate_rr's
                degenerate_stop branch)
    rr = abs(target - entry_mid) / risk
A test (test_rescore.py::test_rr_recompute_matches_validate_rr) pins this
agreement so the rr_min filter is internally consistent with the trading layer.

MANDATORY LIMITATION (cooldown is NOT a post-LLM filter):
    Re-scoring treats alerts as independent and ignores cooldown
    re-interaction — filtering a trade would in reality free a cooldown slot
    the LLM was never queried about. Therefore cooldown is a TradingConfig
    (recording-time, Tier-2) knob, NOT a post-LLM re-scored filter.
    ``PostLLMParams`` deliberately has no cooldown field. This is documented
    here and surfaced in the walk-forward report.

Trace schema (pinned against workspace/trade_traces.jsonl):
    trace = {
      "timestamp": str, "killzone": str, "alert": {
        "bias", "model", "conviction", "entry_zone": [lo, hi],
        "stop", "target", "risk_reward", "killzone", "no_trade_reason", ...
      }, "outcome": {"result", "actual_rr", ...}, ...
    }
The top-level ``killzone`` and ``alert.killzone`` are both present and equal
in recorded traces; rescore reads ``alert.killzone`` (falling back to the
top-level) for the killzone filter, and ``alert.model``/``alert.conviction``
for the others.
"""

from __future__ import annotations

from typing import Iterable

from .params import ALL, PostLLMParams

__all__ = ["rescore_trace", "rescore_traces", "compute_rr"]


# Outcomes that are NOT taken trades — filters never touch them. A filter only
# ever DOWNGRADES a taken (directional, filled/resolved) trade to no_trade; it
# never promotes. (A no_trade stays no_trade; a no_fill/expired/cancelled
# stays as-is — these are not "taken" directional trades whose outcome could
# be zeroed by a conviction/rr/model/kz filter.)
_NON_TAKEN_RESULTS = frozenset({"no_trade", "no_fill", "expired", "cancelled"})


def compute_rr(entry_zone, stop: float, target: float) -> float:
    """Recompute risk-reward from geometry, mirroring ``validate_rr`` exactly.

    entry_mid = (entry_zone[0] + entry_zone[1]) / 2
    risk = abs(entry_mid - stop)
    risk == 0 → returns 0.0 (degenerate; caller treats as a downgrade trigger)
    rr = abs(target - entry_mid) / risk

    This is the SAME formula validate_rr uses (workspace/trading/alert.py).
    Shared here (not duplicated from the frozen layer) because the only
    permitted trading-layer change is config threading — we cannot add a
    helper to alert.py. A dedicated test pins the agreement.
    """
    lo, hi = float(entry_zone[0]), float(entry_zone[1])
    entry_mid = (lo + hi) / 2.0
    risk = abs(entry_mid - float(stop))
    if risk == 0:
        return 0.0
    return abs(float(target) - entry_mid) / risk


def _no_trade_outcome() -> dict:
    """The canonical no_trade outcome dict (R=0, all resolution fields None).

    Matches the shape OutcomeSimulator emits for a no_trade alert and that
    PerformanceReport._result reads (result == "no_trade", actual_rr None →
    counted as 0 in expectancy).
    """
    return {
        "result": "no_trade",
        "candles_to_fill": None,
        "candles_to_resolution": None,
        "fill_price": None,
        "exit_price": None,
        "actual_rr": None,
    }


def _is_taken(trace: dict) -> bool:
    """True if the trace is a taken directional trade that filters can downgrade.

    A directional alert (long/short) whose outcome is a resolved fill (win/loss)
    is "taken". no_trade/no_fill/expired/cancelled are NOT taken — filters
    leave them untouched (a filter never promotes, and zeroing an already-
    non-trade outcome is a no-op). We key off the OUTCOME result (the source
    of truth for what actually happened), not the alert bias, so a directional
    alert that expired is left as expired (its actual_rr is already None).
    """
    result = trace.get("outcome", {}).get("result", "no_trade")
    return result not in _NON_TAKEN_RESULTS


def rescore_trace(trace: dict, p: PostLLMParams) -> dict:
    """Apply post-LLM params to a single trace. Returns a NEW dict (no mutation).

    A taken trade failing ANY filter is downgraded to no_trade (R=0). A
    surviving taken trade is returned unchanged (recorded outcome reused
    verbatim — no re-simulation). A non-taken trace is returned unchanged.
    """
    if not _is_taken(trace):
        return dict(trace)

    alert = trace.get("alert", {}) or {}

    # ── Filter 1: conviction ──────────────────────────────────────────────
    conviction = alert.get("conviction", 0)
    try:
        conviction = int(conviction)
    except (TypeError, ValueError):
        conviction = 0
    if conviction < p.conviction_min:
        return _downgrade(trace, "conviction_below_min")

    # ── Filter 2: recomputed R:R ──────────────────────────────────────────
    entry_zone = alert.get("entry_zone", [0.0, 0.0])
    stop = alert.get("stop", 0.0)
    target = alert.get("target", 0.0)
    rr = compute_rr(entry_zone, stop, target)
    if rr < p.rr_min:
        return _downgrade(trace, "rr_below_min")

    # ── Filter 3: allowed models ──────────────────────────────────────────
    model = alert.get("model", "none")
    if p.allowed_models is not ALL and model not in p.allowed_models:
        return _downgrade(trace, "model_filtered")

    # ── Filter 4: allowed killzones ───────────────────────────────────────
    killzone = alert.get("killzone") or trace.get("killzone") or ""
    if p.allowed_killzones is not ALL and killzone not in p.allowed_killzones:
        return _downgrade(trace, "killzone_filtered")

    # All filters passed → keep the recorded outcome verbatim.
    return dict(trace)


def rescore_traces(traces: Iterable[dict], p: PostLLMParams) -> list[dict]:
    """Apply ``p`` to every trace, preserving chronological order.

    Order is preserved exactly as given (the caller is expected to pass
    chronologically-sorted traces; rescore does not re-sort, to avoid masking
    a caller ordering bug).
    """
    return [rescore_trace(t, p) for t in traces]


def _downgrade(trace: dict, reason: str) -> dict:
    """Return a copy of ``trace`` downgraded to a no_trade (R=0).

    The alert's bias/no_trade_reason and the outcome are rewritten to the
    no_trade form. All other fields (timestamp, killzone, snapshot_summary,
    raw_llm_output, etc.) are preserved so the trace remains identifiable and
    the report can attribute the downgrade.
    """
    out = dict(trace)
    alert = dict(out.get("alert", {}) or {})
    alert["bias"] = "no_trade"
    alert["no_trade_reason"] = reason
    # risk_reward is left as-recorded (the true geometry value); only the
    # outcome is zeroed. This mirrors validate_rr, which sets risk_reward to
    # the true value before downgrading on rr_below_1.
    out["alert"] = alert
    out["outcome"] = _no_trade_outcome()
    return out
