"""tuning — Phase 3B offline tuning & walk-forward harness.

Tier 1 (free): post-LLM re-scoring over already-recorded traces. Zero LLM
calls. Modules: params, rescore, objective, search, walkforward, report.

Tier 2 (cost-gated): pre-LLM recording variants, driven by scripts/run_tuning.py
--record. Each TradingConfig variant is recorded once over the full span
(reusing BacktestEngine's idempotent resume), then sliced per fold by timestamp.
"""
