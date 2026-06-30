"""report.py — WalkForwardReport: the human-readable honesty report.

The deliverable that tells the user whether tuning is real or noise. It reads
aggregates from the WalkForwardResult / FoldResult metrics (which came from
PerformanceReport) — it does NOT recompute trade math a third way.

Mandatory sections:
  1. Per-fold table: fold dates, chosen config_hash + chosen params, IS
     expectancy, OOS expectancy, OOS win rate / profit factor / max_drawdown_r,
     OOS taken_trades.
  2. Aggregate OOS vs baseline: mean OOS expectancy (tuned) vs mean OOS
     expectancy (baseline), pooled across folds; same for win rate / PF / max_dd.
     BOTH mean-of-folds AND trade-weighted (pooled) are reported so
     small-sample folds can't hide.
  3. Parameter stability: per-field distribution of chosen params across folds;
     flag when winners differ wildly fold-to-fold ("unstable → likely noise").
  4. OVERFITTING WARNING: emit when aggregate IS ≫ aggregate OOS (gap >
     threshold) OR tuned OOS ≤ baseline OOS. If tuned OOS ≤ baseline OOS,
     state plainly: "Tuning added nothing; use baseline."
  5. Disclaimers: the Phase 3A optimism disclaimer (verbatim-in-spirit) PLUS a
     leakage note: the LLM's training data overlaps the backtest period, so
     even OOS numbers here are optimistic relative to truly unseen future data.
  6. Degenerate-fold note for folds below min_trades.

Aggregation rule (documented): we report BOTH mean-of-folds (each fold weighted
equally) AND trade-weighted (pooled — each trade weighted equally). The
headline uses mean-of-folds; trade-weighted is shown alongside so a single
large fold can't dominate or a single small fold can't be hidden.

Stability threshold (documented, conservative): a field is "unstable" if the
most common value across folds accounts for < 60% of folds (i.e. the winner
changes on > 40% of folds). This is heuristic and intentionally conservative.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Iterable

from trading.config import TradingConfig
from .params import ALL, PostLLMParams
from .walkforward import FoldResult, WalkForwardResult

__all__ = ["WalkForwardReport"]

# Overfitting gap threshold (R): if (mean IS − mean OOS) exceeds this, warn.
# Conservative: 0.5R is a large IS→OOS decay for an expectancy-per-trade metric.
_OVERFIT_GAP_R = 0.5

# Stability: a field is "unstable" if its modal value covers < this fraction of
# folds. Conservative: requires > 60% agreement to call a field stable.
_STABLE_FRACTION = 0.60


# Phase 3A optimism disclaimer, carried forward verbatim-in-spirit (the same
# content as backtesting/report.py's _DISCLAIMER, adapted to the tuning context).
# Rendered under the "## 5. Disclaimers" header (section 5 of the report).
_DISCLAIMER_BODY = (
    "This walk-forward report is a SIMULATION on historical data. Past performance "
    "does not guarantee future results. Re-scored outcomes reuse each recorded "
    "trade's outcome verbatim (no re-simulation); filtered trades become no_trade "
    "(R=0). Slippage, commissions, and real-world execution friction are NOT "
    "modeled. Results may be optimistically biased: the LLM may have seen this "
    "period in its training data, so historical pattern recall cannot be "
    "distinguished from genuine edge. Do not use this report as the sole basis "
    "for any trading decision."
)

# Leakage note: even OOS folds are optimistic because the LLM's training data
# overlaps the backtest period.
_LEAKAGE_NOTE = (
    "### LLM Training-Data Leakage Note\n\n"
    "The reasoning LLM's training data overlaps the backtest period. "
    "Consequently, even the out-of-sample (OOS) fold numbers here are "
    "optimistic relative to truly unseen future data: the model may recall "
    "historical price action it was trained on, inflating apparent edge. "
    "Treat OOS expectancy as an upper bound on realizable forward performance, "
    "not a prediction of live results."
)


class WalkForwardReport:
    """Generate the walk-forward honesty report (markdown)."""

    @staticmethod
    def generate(
        result: WalkForwardResult,
        out_path: str = "workspace/walkforward_report.md",
    ) -> str:
        """Generate the report, write it to ``out_path``, return the markdown.

        Pure function of ``result``: no recomputation of trade math (reads
        metrics already in FoldResult, which came from PerformanceReport).
        """
        lines: list[str] = []
        lines.append("# Kyros Walk-Forward Tuning Report\n")
        lines.append(
            f"**Baseline config hash:** `{result.baseline_config_hash[:12]}`\n"
        )

        folds = result.folds

        # ── Section 1: Per-fold table ─────────────────────────────────────
        lines.append("## 1. Per-Fold Results\n")
        if not folds:
            lines.append("_No folds produced (insufficient data for the requested "
                         "train/test/step window sizes)._\n")
        else:
            lines.append(
                "| Fold | Train | Test | Chosen Config | Chosen Params | "
                "IS Exp (R) | OOS Exp (R) | OOS Win% | OOS PF | OOS MaxDD (R) | "
                "OOS Trades |"
            )
            lines.append(
                "|------|-------|------|---------------|---------------|"
                "------------|-------------|----------|--------|----------------|"
                "------------|"
            )
            for i, fr in enumerate(folds, 1):
                lines.append(_fold_row(i, fr))
            lines.append("")

        # ── Section 2: Aggregate OOS vs baseline ──────────────────────────
        lines.append("## 2. Aggregate OOS: Tuned vs Baseline\n")
        if folds:
            lines.append(_aggregate_section(folds))
        else:
            lines.append("_No folds to aggregate._\n")

        # ── Section 3: Parameter stability ────────────────────────────────
        lines.append("## 3. Parameter Stability Across Folds\n")
        if folds:
            lines.append(_stability_section(folds))
        else:
            lines.append("_No folds to assess._\n")

        # ── Section 4: Overfitting warning ────────────────────────────────
        lines.append("## 4. Overfitting Assessment\n")
        if folds:
            lines.append(_overfit_section(folds))
        else:
            lines.append("_No folds to assess._\n")

        # ── Section 5: Disclaimers ────────────────────────────────────────
        lines.append("## 5. Disclaimers\n")
        lines.append(_DISCLAIMER_BODY)
        lines.append("")
        lines.append(_LEAKAGE_NOTE)
        lines.append("")

        # ── Section 6: Degenerate-fold note ───────────────────────────────
        lines.append("## 6. Degenerate Folds\n")
        if folds:
            lines.append(_degenerate_section(folds))
        else:
            lines.append("_No folds._\n")

        md = "\n".join(lines)

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)
        return md


# ── Section renderers ─────────────────────────────────────────────────────────


def _fmt_float(v) -> str:
    """Format a float for the table; -inf → '-inf'; None/str → str."""
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    if isinstance(v, float):
        if math.isinf(v):
            return "-inf" if v < 0 else "inf"
        if math.isnan(v):
            return "nan"
        return f"{v:.4f}"
    return str(v)


def _fmt_pct(v) -> str:
    if v is None or isinstance(v, str):
        return "n/a"
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _params_str(p: PostLLMParams) -> str:
    """Compact one-line representation of chosen params."""
    models = "ALL" if p.allowed_models is ALL else ",".join(sorted(p.allowed_models))
    kzs = "ALL" if p.allowed_killzones is ALL else ",".join(sorted(p.allowed_killzones))
    return f"cv≥{p.conviction_min} rr≥{p.rr_min} M={models} KZ={kzs}"


def _fold_row(i: int, fr: FoldResult) -> str:
    m = fr.oos_metrics or {}
    pf = m.get("profit_factor", "n/a")
    pf_str = f"{pf:.2f}" if isinstance(pf, float) else str(pf)
    return (
        f"| {i} | {fr.fold.train_start[:10]}→{fr.fold.train_end[:10]} | "
        f"{fr.fold.test_start[:10]}→{fr.fold.test_end[:10]} | "
        f"`{fr.chosen_config.config_hash()[:8]}` | {_params_str(fr.chosen_params)} | "
        f"{_fmt_float(fr.is_expectancy)} | {_fmt_float(fr.oos_expectancy)} | "
        f"{_fmt_pct(m.get('win_rate'))} | {pf_str} | "
        f"{_fmt_float(m.get('max_drawdown_r'))} | "
        f"{m.get('filled_count', 0)} |"
    )


def _finite_expectancies(folds: Iterable[FoldResult], key: str) -> list[float]:
    """Collect finite OOS expectancies (drop -inf) for aggregation."""
    out = []
    for fr in folds:
        v = getattr(fr, key)
        if isinstance(v, (int, float)) and not math.isinf(v) and not math.isnan(v):
            out.append(float(v))
    return out


def _aggregate_section(folds: list[FoldResult]) -> str:
    """Mean-of-folds AND trade-weighted (pooled) for tuned vs baseline OOS."""
    tuned = _finite_expectancies(folds, "oos_expectancy")
    base = _finite_expectancies(folds, "baseline_oos_expectancy")

    mean_tuned = sum(tuned) / len(tuned) if tuned else float("nan")
    mean_base = sum(base) / len(base) if base else float("nan")

    # Trade-weighted (pooled): weight each fold's OOS expectancy by its
    # filled_count (taken trades). Folds with -inf expectancy (degenerate) are
    # excluded from the pooled mean (they have no meaningful per-trade R).
    pooled_tuned_num = 0.0
    pooled_tuned_den = 0
    pooled_base_num = 0.0
    pooled_base_den = 0
    for fr in folds:
        tv = fr.oos_expectancy
        bv = fr.baseline_oos_expectancy
        tn = (fr.oos_metrics or {}).get("filled_count", 0)
        bn = (fr.baseline_oos_metrics or {}).get("filled_count", 0)
        if isinstance(tv, (int, float)) and not math.isinf(tv) and not math.isnan(tv) and tn:
            pooled_tuned_num += tv * tn
            pooled_tuned_den += tn
        if isinstance(bv, (int, float)) and not math.isinf(bv) and not math.isnan(bv) and bn:
            pooled_base_num += bv * bn
            pooled_base_den += bn
    pooled_tuned = pooled_tuned_num / pooled_tuned_den if pooled_tuned_den else float("nan")
    pooled_base = pooled_base_num / pooled_base_den if pooled_base_den else float("nan")

    lines = [
        "Aggregation reports BOTH mean-of-folds (each fold weighted equally) "
        "AND trade-weighted/pooled (each trade weighted equally), so a single "
        "large or small fold cannot hide.\n",
        "| Metric | Tuned (mean-of-folds) | Baseline (mean-of-folds) | "
        "Tuned (trade-weighted) | Baseline (trade-weighted) |",
        "|--------|----------------------|--------------------------|"
        "------------------------|---------------------------|",
        f"| OOS expectancy (R) | {_fmt_float(mean_tuned)} | {_fmt_float(mean_base)} | "
        f"{_fmt_float(pooled_tuned)} | {_fmt_float(pooled_base)} |",
        "",
    ]
    return "\n".join(lines)


def _stability_section(folds: list[FoldResult]) -> str:
    """Per-field distribution of chosen params; flag instability."""
    # Collect each param field's values across folds.
    cv_vals = [fr.chosen_params.conviction_min for fr in folds]
    rr_vals = [fr.chosen_params.rr_min for fr in folds]
    model_vals = [
        "ALL" if fr.chosen_params.allowed_models is ALL
        else ",".join(sorted(fr.chosen_params.allowed_models))
        for fr in folds
    ]
    kz_vals = [
        "ALL" if fr.chosen_params.allowed_killzones is ALL
        else ",".join(sorted(fr.chosen_params.allowed_killzones))
        for fr in folds
    ]

    def _dist(vals):
        c = Counter(str(v) for v in vals)
        total = len(vals)
        modal, modal_n = c.most_common(1)[0]
        frac = modal_n / total if total else 0.0
        stable = frac >= _STABLE_FRACTION
        detail = ", ".join(f"{k}×{v}" for k, v in sorted(c.items()))
        return modal, frac, stable, detail

    lines = [
        f"A field is flagged **unstable** when its most common value covers "
        f"< {int(_STABLE_FRACTION * 100)}% of folds (winners change too often "
        f"→ likely noise).\n",
        "| Field | Modal value | Modal coverage | Stable? | Distribution |",
        "|-------|-------------|----------------|---------|--------------|",
    ]
    for name, vals in (("conviction_min", cv_vals), ("rr_min", rr_vals),
                       ("allowed_models", model_vals), ("allowed_killzones", kz_vals)):
        modal, frac, stable, detail = _dist(vals)
        flag = "yes" if stable else "**UNSTABLE**"
        lines.append(f"| {name} | {modal} | {frac:.0%} | {flag} | {detail} |")
    lines.append("")
    return "\n".join(lines)


def _overfit_section(folds: list[FoldResult]) -> str:
    """Overfitting warning: IS≫OOS gap OR tuned OOS ≤ baseline OOS."""
    is_vals = _finite_expectancies(folds, "is_expectancy")
    oos_vals = _finite_expectancies(folds, "oos_expectancy")
    base_vals = _finite_expectancies(folds, "baseline_oos_expectancy")

    mean_is = sum(is_vals) / len(is_vals) if is_vals else float("nan")
    mean_oos = sum(oos_vals) / len(oos_vals) if oos_vals else float("nan")
    mean_base = sum(base_vals) / len(base_vals) if base_vals else float("nan")

    lines = [
        f"- Mean IS expectancy (R): {_fmt_float(mean_is)}",
        f"- Mean OOS expectancy (R), tuned: {_fmt_float(mean_oos)}",
        f"- Mean OOS expectancy (R), baseline: {_fmt_float(mean_base)}",
        f"- IS→OOS gap (R): {_fmt_float(mean_is - mean_oos if not math.isnan(mean_is) and not math.isnan(mean_oos) else float('nan'))} "
        f"(warning threshold: > {_OVERFIT_GAP_R})",
        "",
    ]

    warned = False
    # Condition 1: IS ≫ OOS (gap exceeds threshold).
    if (not math.isnan(mean_is) and not math.isnan(mean_oos)
            and (mean_is - mean_oos) > _OVERFIT_GAP_R):
        lines.append(
            f"⚠️ **OVERFITTING WARNING:** mean IS expectancy exceeds mean OOS "
            f"expectancy by {(mean_is - mean_oos):.4f}R (> {_OVERFIT_GAP_R}R "
            f"threshold). The tuned parameters fit the training slices far "
            f"better than they generalize — treat OOS results with skepticism."
        )
        warned = True

    # Condition 2: tuned OOS ≤ baseline OOS.
    if (not math.isnan(mean_oos) and not math.isnan(mean_base)
            and mean_oos <= mean_base):
        lines.append(
            "⚠️ **Tuning added nothing; use baseline.** Mean tuned OOS "
            f"expectancy ({_fmt_float(mean_oos)}) is ≤ mean baseline OOS "
            f"expectancy ({_fmt_float(mean_base)}). The post-LLM grid search "
            f"did not improve on the default configuration out-of-sample."
        )
        warned = True

    if not warned:
        lines.append(
            f"No overfitting warning triggered: the IS→OOS gap is within the "
            f"{_OVERFIT_GAP_R}R threshold and tuned OOS exceeds baseline OOS. "
            f"(This is not a guarantee of real edge — see the disclaimers and "
            f"leakage note below.)"
        )
    lines.append("")
    return "\n".join(lines)


def _degenerate_section(folds: list[FoldResult]) -> str:
    """Flag folds whose OOS (or IS) is -inf (below min_trades)."""
    degenerate = []
    for i, fr in enumerate(folds, 1):
        is_deg = math.isinf(fr.is_expectancy) and fr.is_expectancy < 0
        oos_deg = math.isinf(fr.oos_expectancy) and fr.oos_expectancy < 0
        if is_deg or oos_deg:
            tags = []
            if is_deg:
                tags.append("IS")
            if oos_deg:
                tags.append("OOS")
            degenerate.append(
                f"- Fold {i} ({fr.fold.test_start[:10]}→{fr.fold.test_end[:10]}): "
                f"{'/'.join(tags)} below min_trades (too few taken trades to score)."
            )
    if not degenerate:
        return "No degenerate folds (every fold met min_trades on both IS and OOS).\n"
    return "\n".join(degenerate) + "\n"
