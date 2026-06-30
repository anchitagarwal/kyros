"""walkforward.py — rolling train/test split by timestamp; pick best (config,
params) on each train slice, evaluate on the next unseen test slice, and
compare to baseline on the same test windows.

Fold construction (rolling, by timestamp, NO LEAKAGE):
    Sort traces by ISO timestamp. The window start advances by ``step_days``.
    For each window:
        train = [start, start + train_days)        (half-open)
        test  = [start + train_days, start + train_days + test_days)  (half-open)
    A trace falls in AT MOST ONE of {train, test} of a given fold (half-open
    intervals; no boundary double-count). The critical invariant — asserted
    per fold — is that train and test are DISJOINT by timestamp. A dedicated
    test tries to break this with a trace exactly on the boundary.

    Consecutive folds' starts differ by exactly ``step_days``. Cross-fold
    overlap (step_days < test_days) is allowed — a trace may appear in fold N's
    test and fold N+1's train — but INTRA-fold train/test never overlap.

Per fold:
    Over the product (pre-LLM configs × post-LLM grid), score each (config,
    params) on that fold's TRAIN slice of that config's traces; pick the
    argmax → (chosen_config, chosen_params, is_expectancy). Evaluate that SAME
    choice on the fold's TEST slice of that config's traces → oos. Separately
    evaluate BASELINE (default config + default params) on the same TEST window
    → baseline_oos. Apples-to-apples: tuned and baseline use the SAME test
    window of the SAME (baseline) config's traces.

Timestamp normalization:
    Trace timestamps are parsed once to aware datetimes (naive → ET, matching
    report.py / snapshot._parse_dt). Mixed tz offsets are normalized via
    astimezone so ordering is correct. Fold date bounds are computed in UTC
    (date arithmetic at midnight UTC) and compared against the aware trace
    timestamps — both aware, so the comparison is well-defined.

Folds with empty train or empty test are DROPPED (documented). A fold whose
train slice has too few taken trades still produces a result via search's
all-below-min_trades fallback (degenerate, flagged in the report).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from trading.config import TradingConfig
from .objective import evaluate
from .params import PostLLMParams, default_post_params
from .search import best_params

__all__ = ["Fold", "FoldResult", "WalkForwardResult", "make_folds", "run_walkforward"]

_NY = ZoneInfo("America/New_York")


def _parse_ts(ts) -> datetime:
    """Parse a timestamp to an aware datetime (naive → ET, matching report.py)."""
    if isinstance(ts, datetime):
        dt = ts
    else:
        dt = datetime.fromisoformat(str(ts))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_NY)
    return dt


@dataclass(frozen=True)
class Fold:
    """A single rolling train/test split.

    train_start/train_end/test_start/test_end are ISO-8601 strings (UTC
    midnight bounds). train = [train_start, train_end); test = [test_start,
    test_end). train_end == test_start (the boundary is half-open: a trace
    exactly at train_end falls in test, not train).
    """

    train: list[dict]
    test: list[dict]
    train_start: str
    train_end: str
    test_start: str
    test_end: str


@dataclass(frozen=True)
class FoldResult:
    """The outcome of one fold: the chosen (config, params) and its IS/OOS."""

    fold: Fold
    chosen_config: TradingConfig
    chosen_params: PostLLMParams
    is_expectancy: float
    oos_expectancy: float
    oos_metrics: dict
    baseline_oos_expectancy: float
    baseline_oos_metrics: dict


@dataclass(frozen=True)
class WalkForwardResult:
    """All fold results + the baseline config hash (for the report)."""

    folds: list[FoldResult]
    baseline_config_hash: str


def make_folds(
    traces: Sequence[dict],
    train_days: int,
    test_days: int,
    step_days: int,
) -> list[Fold]:
    """Build rolling half-open train/test folds over ``traces`` by timestamp.

    The fold date bounds are derived from the BASELINE traces' timestamp span
    (the earliest trace timestamp defines the first window start, aligned to
    its UTC midnight). Per-config traces are sliced to these SAME bounds in
    run_walkforward, so all configs share identical fold windows.

    Folds with empty train OR empty test are dropped (a fold must have at
    least one trace on each side to be meaningful).

    Args:
        traces: the baseline traces (sorted or unsorted; sorted internally).
        train_days: train window length in days.
        test_days: test window length in days.
        step_days: window advance per fold in days.

    Returns:
        A list of Fold objects. Each fold's train and test are DISJOINT by
        timestamp (asserted). Consecutive folds' starts differ by step_days.
    """
    if train_days <= 0 or test_days <= 0 or step_days <= 0:
        raise ValueError("train_days, test_days, step_days must all be positive")

    if not traces:
        return []

    # Sort by parsed timestamp (stable). Carry the original trace dict.
    indexed = sorted(((_parse_ts(t.get("timestamp")), t) for t in traces),
                     key=lambda x: x[0])
    parsed_ts = [ts for ts, _ in indexed]
    trace_list = [t for _, t in indexed]

    # Align the first window start to the UTC midnight of the earliest trace.
    first = parsed_ts[0].astimezone(timezone.utc)
    window_start = first.replace(hour=0, minute=0, second=0, microsecond=0)
    last = parsed_ts[-1].astimezone(timezone.utc)

    folds: list[Fold] = []
    cur = window_start
    while True:
        train_start = cur
        train_end = cur + timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + timedelta(days=test_days)

        # If the train window starts beyond the last trace, we're done.
        if train_start > last:
            break

        # Slice by half-open intervals [start, end). A trace exactly at a
        # boundary falls in the LATER window (train_end == test_start → test).
        train = [t for ts, t in zip(parsed_ts, trace_list) if train_start <= ts < train_end]
        test = [t for ts, t in zip(parsed_ts, trace_list) if test_start <= ts < test_end]

        # Drop folds with empty train or empty test.
        if train and test:
            folds.append(Fold(
                train=train,
                test=test,
                train_start=train_start.isoformat(),
                train_end=train_end.isoformat(),
                test_start=test_start.isoformat(),
                test_end=test_end.isoformat(),
            ))

        cur = cur + timedelta(days=step_days)

    # ── Release-blocker invariant: train ∩ test = ∅ per fold (by timestamp). ──
    for f in folds:
        _assert_disjoint(f)

    return folds


def _assert_disjoint(fold: Fold) -> None:
    """Assert no trace timestamp appears in both train and test of this fold.

    This is the phase's most important statistical invariant. A trace on the
    train/test boundary (train_end == test_start) must land in exactly one
    side (the half-open interval puts it in test). We check by timestamp
    string set intersection — robust and explicit.
    """
    train_ts = {t.get("timestamp") for t in fold.train}
    test_ts = {t.get("timestamp") for t in fold.test}
    overlap = train_ts & test_ts
    assert not overlap, (
        f"LEAKAGE: fold train/test overlap on {len(overlap)} timestamp(s): "
        f"{sorted(overlap)[:5]}"
    )


def _slice_to_bounds(traces: Sequence[dict], start: str, end: str) -> list[dict]:
    """Return traces whose timestamp is in [start, end) (half-open).

    ``start``/``end`` are ISO-8601 UTC bounds (from Fold). Trace timestamps
    are parsed to aware datetimes for the comparison (naive → ET).
    """
    start_dt = _parse_ts(start)
    end_dt = _parse_ts(end)
    out = []
    for t in traces:
        ts = _parse_ts(t.get("timestamp"))
        if start_dt <= ts < end_dt:
            out.append(t)
    return out


def run_walkforward(
    trace_sets: Mapping[str, list[dict]],
    folds: list[Fold],
    grid: Iterable[PostLLMParams],
    min_trades: int,
    configs: Mapping[str, TradingConfig] | None = None,
) -> WalkForwardResult:
    """Run walk-forward over per-config trace sets and shared folds.

    Args:
        trace_sets: config_hash → that config's full-span recorded traces.
            For the free path this is ``{baseline_hash: baseline_traces}``.
        folds: folds from make_folds (built on the baseline traces' span).
        grid: the post-LLM grid (must include default_post_params for the
            baseline comparison and the all-below-min_trades fallback).
        min_trades: minimum taken trades for a finite objective.
        configs: optional registry config_hash → TradingConfig, used to recover
            the chosen config object for FoldResult. Defaults to
            ``{baseline_hash: TradingConfig()}`` — sufficient for the free
            path (baseline only). Tier-2 callers pass the full registry so
            non-baseline chosen configs are recorded correctly.

    Returns:
        A WalkForwardResult with one FoldResult per fold.

    Per fold:
        1. For each config in trace_sets: slice its traces to the fold's train
           and test bounds. Score every (config, params) on the train slice;
           pick the argmax over the product (configs × grid).
        2. Evaluate the chosen (config, params) on that config's TEST slice → oos.
        3. Evaluate BASELINE (default config + default params) on the baseline
           config's TEST slice → baseline_oos.
    """
    grid_list = list(grid)
    baseline_cfg = TradingConfig()
    baseline_hash = baseline_cfg.config_hash()

    # Build the config registry: caller-supplied configs take precedence; the
    # baseline is always available.
    registry: dict[str, TradingConfig] = {baseline_hash: baseline_cfg}
    if configs is not None:
        for h, c in configs.items():
            registry[h] = c

    # Identify the baseline trace set. Prefer the true baseline hash; if a
    # caller omitted it, use the first available set (defensive).
    if baseline_hash not in trace_sets and trace_sets:
        baseline_hash = next(iter(trace_sets))

    fold_results: list[FoldResult] = []
    for fold in folds:
        # ── 1. Pick best (config, params) on the TRAIN slice ──────────────
        chosen_config: TradingConfig | None = None
        chosen_params: PostLLMParams | None = None
        chosen_is = -math.inf
        chosen_metrics: dict = {}

        for cfg_hash, cfg_traces in trace_sets.items():
            cfg = registry.get(cfg_hash, baseline_cfg)
            train_slice = _slice_to_bounds(cfg_traces, fold.train_start, fold.train_end)
            params, score, metrics = best_params(train_slice, grid_list, min_trades)
            if score > chosen_is:
                chosen_config = cfg
                chosen_params = params
                chosen_is = score
                chosen_metrics = metrics

        if chosen_config is None or chosen_params is None:
            # No config had any traces in this fold's train window (degenerate).
            chosen_config = baseline_cfg
            chosen_params = default_post_params()
            chosen_is = -math.inf
            chosen_metrics = {}

        # ── 2. Evaluate the chosen (config, params) on its TEST slice ─────
        chosen_hash = chosen_config.config_hash()
        chosen_test_traces = trace_sets.get(chosen_hash, trace_sets.get(baseline_hash, []))
        test_slice = _slice_to_bounds(chosen_test_traces, fold.test_start, fold.test_end)
        oos_score, oos_metrics = evaluate(test_slice, chosen_params, min_trades=min_trades)

        # ── 3. Evaluate BASELINE on the baseline config's TEST slice ──────
        baseline_traces = trace_sets.get(baseline_hash, [])
        baseline_test_slice = _slice_to_bounds(baseline_traces, fold.test_start, fold.test_end)
        baseline_params = default_post_params()
        b_score, b_metrics = evaluate(baseline_test_slice, baseline_params, min_trades=min_trades)

        fold_results.append(FoldResult(
            fold=fold,
            chosen_config=chosen_config,
            chosen_params=chosen_params,
            is_expectancy=chosen_is,
            oos_expectancy=oos_score,
            oos_metrics=oos_metrics,
            baseline_oos_expectancy=b_score,
            baseline_oos_metrics=b_metrics,
        ))

    return WalkForwardResult(folds=fold_results, baseline_config_hash=baseline_hash)
