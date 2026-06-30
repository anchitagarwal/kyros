"""params.py — typed search space for post-LLM re-scoring (Tier 1) and
pre-LLM recording variants (Tier 2).

PostLLMParams are the FREE re-scoring knobs applied to already-recorded traces
by pure arithmetic (see rescore.py): conviction_min, rr_min, allowed_models,
allowed_killzones. No LLM calls, no re-simulation.

PreLLMGrid is the COSTLY recording-variant axis: a list of TradingConfig
instances, each of which would be recorded once over the full span (Tier 2,
cost-gated). The default grid is a single baseline config → zero cost.

"Allow all" sentinel
--------------------
``allowed_models`` / ``allowed_killzones`` use frozenset semantics where an
EMPTY frozenset means "reject everything" (testable as such). "Allow all" is
represented by the module-level sentinel ``ALL = frozenset({"*"})``: rescore
treats ``{"*"}`` as "no filter on this axis". This avoids the empty-set
ambiguity (does empty mean "allow all" or "allow nothing"?). ``default_post_params``
uses ``ALL`` for both axes so the baseline re-score is a no-op (today's
behavior).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator, Sequence

from trading.config import TradingConfig

__all__ = [
    "PostLLMParams",
    "ALL",
    "param_grid",
    "default_post_params",
    "PreLLMGrid",
]

# Sentinel frozenset meaning "allow all" on a filter axis. Distinct from the
# empty frozenset, which means "reject everything". rescore.py checks for this
# sentinel (membership of "*" short-circuits the filter).
ALL = frozenset({"*"})


@dataclass(frozen=True)
class PostLLMParams:
    """The Tier-1 (free) re-scoring knobs.

    Fields:
        conviction_min: alerts with conviction < this are downgraded to
            no_trade. (Conviction is enforced in the LLM prompt at 40 today;
            this knob lets the re-score raise the floor.)
        rr_min: alerts whose recomputed risk-reward < this are downgraded to
            no_trade. Recomputed from recorded entry/stop/target via the same
            formula validate_rr uses.
        allowed_models: alerts whose model is NOT in this set are downgraded.
            ``ALL`` (frozenset({"*"})) = no filter; empty frozenset = reject all.
        allowed_killzones: alerts whose killzone is NOT in this set are
            downgraded. Same sentinel semantics as allowed_models.
    """

    conviction_min: int
    rr_min: float
    allowed_models: frozenset[str]
    allowed_killzones: frozenset[str]


def param_grid(
    conviction_mins: Sequence[int],
    rr_mins: Sequence[float],
    model_sets: Sequence[frozenset[str]],
    killzone_sets: Sequence[frozenset[str]],
) -> Iterator[PostLLMParams]:
    """Yield the full cartesian product of the four axes in deterministic order.

    Order: conviction_mins (outer) × rr_mins × model_sets × killzone_sets
    (inner). Deterministic across runs (no hash randomization — frozensets are
    yielded in the order given, not sorted).
    """
    for cv, rr, models, kzs in product(conviction_mins, rr_mins, model_sets, killzone_sets):
        yield PostLLMParams(
            conviction_min=cv,
            rr_min=rr,
            allowed_models=models,
            allowed_killzones=kzs,
        )


def default_post_params() -> PostLLMParams:
    """The post-LLM projection of ``TradingConfig()`` (the baseline).

    Baseline tuning == baseline config: conviction_min and rr_min mirror the
    TradingConfig defaults, and both filter axes are ``ALL`` (no filtering) so
    the baseline re-score is a no-op — every taken trade keeps its recorded
    outcome. This is the apples-to-apples baseline the walk-forward compares
    tuned choices against.
    """
    cfg = TradingConfig()
    return PostLLMParams(
        conviction_min=cfg.conviction_min,
        rr_min=cfg.rr_min,
        allowed_models=ALL,
        allowed_killzones=ALL,
    )


def PreLLMGrid(configs: Sequence[TradingConfig] | None = None) -> list[TradingConfig]:
    """The Tier-2 (costly) recording-variant axis.

    With no args (or None), yields exactly ``[TradingConfig()]`` — the single
    baseline config. This is the default path: Tier-2 sweep is OFF, so the
    recording loop records nothing new (zero cost) and tuning runs over the
    existing baseline traces only.

    With an explicit list, returns it as-is (callers supply distinct configs;
    duplicates are the caller's responsibility — they would collide on
    config_hash run dirs).
    """
    if configs is None:
        return [TradingConfig()]
    return list(configs)
