# Phase 2B Executor Contract (update)

## Resume check (current repo state)
- `.kyros_state.json` exists at repo root (NOT under `workspace/`). Current phase: `phase_2b`. `evaluator_round=2`, `max_evaluator_rounds=3`.
- Test status: `uv run pytest` currently **fails** with 1 failure:
  - `tests/test_agent_loader.py::test_evaluator_prompt_includes_current_round_context` expects evaluator prompt to include `"round 1 of 3"`.
- Phase 2B continuation items (weighted DOL scoring) are **not implemented** in `workspace/trading/`:
  - No `_score_pool`, `_rank_dols`, `ranked_dols`, `clarity_score`, `score_breakdown`.
  - `_dol_target` still uses `_nearest_internal/_nearest_external` first-match logic.
  - `_build_pools` dedup is by `(level,type,role)` not `(level,type)`.
  - `reasoning_agent.py` prompt still uses the forbidden `1, 2, 1.5, 3` ordering.

## Frozen boundaries (must remain true)
- `workspace/detectors/`: ONLY allowed changes are adding `fibonacci.py` (already present) and its single export line in `__init__.py` (already present). No other detector edits.
- `workspace/trading/`: only additive changes per Phase 2B continuation spec. `_nearest_dol` must remain byte-identical.
- `workspace/backtesting/` and `workspace/tuning/`: off-limits.

## Planned changes (mapped to review findings)

### A) Fix failing test: evaluator round context (tests/test_agent_loader.py)
- **Why**: suite currently failing; must be green before Phase 2B work can be validated.
- **Change**: update `config/prompts.yaml` evaluator prompt to include a literal `ROUND CONTEXT` section containing `"round 1 of 3"` (lowercased match) and ensure it does **not** include `"FINAL allowed round"` by default.
- **Note**: This is outside `workspace/`, but required to restore baseline green tests.

### B) Implement weighted DOL scoring continuation (review HIGH #1-#3)
Files: `workspace/trading/config.py`, `workspace/trading/snapshot.py`, `workspace/trading/reasoning_agent.py`, plus new tests under `tests/phase2/`.

1) `workspace/trading/config.py`
- Add defaults + accessors:
  - `_DEFAULT_DOL_WEIGHTS`, `_DEFAULT_TF_WEIGHTS`, `_DEFAULT_ROLE_WEIGHTS` (tuple-of-tuples)
  - `dol_weights_dict()`, `tf_weights_dict()`, `role_weights_dict()`
  - `ranked_dols_to_llm: int = 5`
- Extend `config_hash()` canonical dict with these new knobs.

2) `workspace/trading/snapshot.py`
- Extend `LiquidityPool` (defaulted): `clarity_score: float = 0.0`, `score_breakdown: dict = field(default_factory=dict)`.
- Extend `LiquidityPool.to_dict()` to emit `clarity_score` and `score_breakdown`.
- Extend `MarketSnapshot` (defaulted): `ranked_dols: list[LiquidityPool] = field(default_factory=list)`.
- Add `_score_pool` (pure) and `_rank_dols` (direction-filter then score then sort desc; stable distance tiebreak).
- Rewrite `_dol_target`:
  - If `cycle is None` or `dol_use_cycle` False → call `_nearest_dol` over **ERL-only** pools (`scope=="external"`) (repair #1) while keeping `_nearest_dol` unchanged.
  - Else → return `_rank_dols(...)[0]` (or None).
  - Delete `_nearest_internal` and `_nearest_external`.
- Add `ranked_dols` population in `build()` and ensure `dol_target == ranked_dols[0]` when cycle active.
- Update `_compact_dict` to include `ranked_dols` (top-N via config) and serialize `dol_target` accordingly.

3) `workspace/trading/snapshot.py` dedup repair (review MEDIUM #6 / repair #5)
- Change dedup to merge on `(level,type)` keeping richest role by priority and recompute confluence on merged set.

4) `workspace/trading/reasoning_agent.py` prompt repair (review HIGH #5 / repair #7)
- Reorder steps so cycle read precedes direction selection; remove fractional numbering.
- Update DOL step to use `ranked_dols` and default to top `clarity_score`.
- Keep OUTPUT JSON schema unchanged.

### C) Add/extend tests for scoring (review HIGH #3)
- Add tests under `tests/phase2/`:
  - `_score_pool` monotonicity for key factors.
  - `_rank_dols` wrong-side exclusion.
  - Core behavior: farther high-TF external pool beats nearer low-TF pool (proximity does not dominate).
  - Determinism: same inputs → identical ranked list + breakdown.
  - Fallback: cycle None / dol_use_cycle False equals `_nearest_dol` over ERL-only set.

## Verification steps
- Run `uv run pytest` after each file change.
- Confirm `_nearest_dol` remains byte-identical (manual diff / grep).
- Ensure only golden updates are serializer additions (scope/role/clarity_score/ranked_dols) where applicable.
