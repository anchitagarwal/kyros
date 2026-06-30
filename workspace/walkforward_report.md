# Kyros Walk-Forward Tuning Report

**Baseline config hash:** `fcf94ad26ad4`

## 1. Per-Fold Results

| Fold | Train | Test | Chosen Config | Chosen Params | IS Exp (R) | OOS Exp (R) | OOS Win% | OOS PF | OOS MaxDD (R) | OOS Trades |
|------|-------|------|---------------|---------------|------------|-------------|----------|--------|----------------|------------|
| 1 | 2026-06-01→2026-06-03 | 2026-06-03→2026-06-04 | `fcf94ad2` | cv≥40 rr≥1.5 M=ALL KZ=ALL | 0.0430 | -inf | 0.0% | n/a | 0.0000 | 0 |
| 2 | 2026-06-02→2026-06-04 | 2026-06-04→2026-06-05 | `fcf94ad2` | cv≥40 rr≥1.0 M=ALL KZ=ALL | 0.0201 | 0.0000 | 0.0% | n/a | 0.0000 | 1 |

## 2. Aggregate OOS: Tuned vs Baseline

Aggregation reports BOTH mean-of-folds (each fold weighted equally) AND trade-weighted/pooled (each trade weighted equally), so a single large or small fold cannot hide.

| Metric | Tuned (mean-of-folds) | Baseline (mean-of-folds) | Tuned (trade-weighted) | Baseline (trade-weighted) |
|--------|----------------------|--------------------------|------------------------|---------------------------|
| OOS expectancy (R) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## 3. Parameter Stability Across Folds

A field is flagged **unstable** when its most common value covers < 60% of folds (winners change too often → likely noise).

| Field | Modal value | Modal coverage | Stable? | Distribution |
|-------|-------------|----------------|---------|--------------|
| conviction_min | 40 | 100% | yes | 40×2 |
| rr_min | 1.5 | 50% | **UNSTABLE** | 1.0×1, 1.5×1 |
| allowed_models | ALL | 100% | yes | ALL×2 |
| allowed_killzones | ALL | 100% | yes | ALL×2 |

## 4. Overfitting Assessment

- Mean IS expectancy (R): 0.0315
- Mean OOS expectancy (R), tuned: 0.0000
- Mean OOS expectancy (R), baseline: 0.0000
- IS→OOS gap (R): 0.0315 (warning threshold: > 0.5)

⚠️ **Tuning added nothing; use baseline.** Mean tuned OOS expectancy (0.0000) is ≤ mean baseline OOS expectancy (0.0000). The post-LLM grid search did not improve on the default configuration out-of-sample.

## 5. Disclaimers

This walk-forward report is a SIMULATION on historical data. Past performance does not guarantee future results. Re-scored outcomes reuse each recorded trade's outcome verbatim (no re-simulation); filtered trades become no_trade (R=0). Slippage, commissions, and real-world execution friction are NOT modeled. Results may be optimistically biased: the LLM may have seen this period in its training data, so historical pattern recall cannot be distinguished from genuine edge. Do not use this report as the sole basis for any trading decision.

### LLM Training-Data Leakage Note

The reasoning LLM's training data overlaps the backtest period. Consequently, even the out-of-sample (OOS) fold numbers here are optimistic relative to truly unseen future data: the model may recall historical price action it was trained on, inflating apparent edge. Treat OOS expectancy as an upper bound on realizable forward performance, not a prediction of live results.

## 6. Degenerate Folds

- Fold 1 (2026-06-03→2026-06-04): OOS below min_trades (too few taken trades to score).
