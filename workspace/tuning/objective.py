"""objective.py — score a (traces, params) pair by re-scoring then computing
expectancy via the existing report engine.

Pipeline:
    rescore_traces(traces, p)  →  PerformanceReport._overall_metrics  →  expectancy

The objective is ``expectancy_per_trade`` = mean(actual_rr) over ALL re-scored
traces, treating no_trade/no_fill/expired as 0 R. This is EXACTLY
PerformanceReport's expectancy definition (the single source of truth — we
REUSE it, never recompute the mean independently). ``taken_trades`` is the
filled count (win/loss/expired) on the RE-SCORED set.

MIN_TRADES guard: when ``taken_trades < min_trades``, returns ``(-inf,
metrics)``. ``metrics`` is ALWAYS returned (even on -inf) so the report can
show why a fold was rejected. ``-inf`` is strictly below ``min_trades``; a
score is finite at exactly ``min_trades``.
"""

from __future__ import annotations

import math
from typing import Sequence

from backtesting.report import PerformanceReport
from .params import PostLLMParams
from .rescore import rescore_traces

__all__ = ["MIN_TRADES", "evaluate"]

# Default minimum taken trades for a score to be meaningful. Below this, the
# objective is -inf (the fold is "degenerate" — too few trades to trust).
# Callers (search, walkforward) pass min_trades explicitly; this is the default.
MIN_TRADES: int = 10


def evaluate(
    traces: Sequence[dict],
    p: PostLLMParams,
    min_trades: int = MIN_TRADES,
) -> tuple[float, dict]:
    """Score (traces, p) by re-scoring then computing expectancy.

    Args:
        traces: recorded traces (chronological; not mutated).
        p: post-LLM re-scoring params.
        min_trades: minimum taken trades for a finite score.

    Returns:
        (score, metrics) where score is expectancy (float) or -inf, and
        metrics is the full PerformanceReport overall-metrics dict over the
        re-scored set (always populated, even when score is -inf, so the
        report can explain a rejected fold).
    """
    rescored = rescore_traces(traces, p)
    metrics = PerformanceReport()._overall_metrics(rescored)

    taken_trades = metrics.get("filled_count", 0)
    expectancy = metrics.get("expectancy", 0.0)

    if taken_trades < min_trades:
        return (-math.inf, metrics)
    return (float(expectancy), metrics)
