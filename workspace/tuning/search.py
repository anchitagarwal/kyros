"""search.py — grid search over PostLLMParams on a training slice.

``best_params`` scores every grid point on the train slice (via
objective.evaluate) and returns the argmax. Ties are broken by grid order
(first wins) — deterministic and documented so walk-forward stability
analysis is meaningful.

All-below-min_trades fallback: if EVERY grid point scores -inf (too few taken
trades on the train slice), the fold is degenerate. Rather than producing no
choice, ``best_params`` returns the baseline (default) params with their
(score, metrics) so the fold still records a comparable result. This is
documented and tested.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from .objective import evaluate
from .params import PostLLMParams, default_post_params

__all__ = ["best_params"]


def best_params(
    train_traces: Sequence[dict],
    grid: Iterable[PostLLMParams],
    min_trades: int,
) -> tuple[PostLLMParams, float, dict]:
    """Return the params with max objective on ``train_traces``.

    Args:
        train_traces: the training slice (chronological).
        grid: an iterable of PostLLMParams to evaluate. Materialized into a
            list so grid order is the deterministic tie-break.
        min_trades: minimum taken trades for a finite score (passed to
            objective.evaluate).

    Returns:
        (best_params, best_score, best_metrics).

    Tie-break: the FIRST grid point achieving the max score wins (grid order).
    This is deterministic given a fixed grid.

    Fallback: if every grid point scores -inf (all below min_trades), returns
    the baseline (default_post_params) with its (score, metrics) so the fold
    still produces a comparable record. The baseline is guaranteed to be in
    the grid by convention (callers include it); if it is not, we still return
    it as the documented degenerate-fold fallback.
    """
    grid_list = list(grid)
    best: PostLLMParams | None = None
    best_score = -math.inf
    best_metrics: dict = {}

    for p in grid_list:
        score, metrics = evaluate(train_traces, p, min_trades=min_trades)
        # Strictly-greater comparison → first-in-grid-order wins ties.
        if score > best_score:
            best = p
            best_score = score
            best_metrics = metrics

    if best is None:
        # Empty grid: fall back to baseline (should not happen with a
        # well-formed grid, but never return None).
        best = default_post_params()
        best_score, best_metrics = evaluate(train_traces, best, min_trades=min_trades)
        return (best, best_score, best_metrics)

    # All-below-min_trades fallback: every grid point was -inf. Return the
    # baseline so the fold records a comparable (degenerate) result rather
    # than an arbitrary -inf grid point. We re-evaluate the baseline to get
    # its own metrics (it may also be -inf, but its metrics are the canonical
    # degenerate-fold record).
    if best_score == -math.inf:
        baseline = default_post_params()
        # If the baseline is in the grid, ``best`` is already it (first -inf
        # in grid order). Re-evaluate to be explicit and return baseline.
        b_score, b_metrics = evaluate(train_traces, baseline, min_trades=min_trades)
        return (baseline, b_score, b_metrics)

    return (best, best_score, best_metrics)
