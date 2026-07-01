# Phase 2B Contract (Executor)

State source: `.kyros_state.json` indicates `current_phase=phase_2b`, `last_review_status=BLOCK`, evaluator round 1/3 with a HIGH finding.

## What exists (baseline)
- `workspace/detectors/fibonacci.py` exists and `detect_fibonacci()` is implemented and tested.
- `workspace/trading/snapshot.py` contains:
  - `LiquidityPool` with new fields `scope`, `role`, `clarity_score`, `score_breakdown`.
  - `MarketSnapshot` with `fib_levels`, `liquidity_cycle`, `ranked_dols`, `dol_target`.
  - Weighted DOL scoring: `_score_pool`, `_rank_dols`, `_dol_target`.
  - `_nearest_dol` present and must remain byte-identical.
- `workspace/trading/config.py` contains fib knobs, IRL gating, cycle toggle, and weight maps.
- Tests exist under `tests/test_fibonacci.py` and `tests/phase2/test_fib_liquidity_cycle.py`.

## Review.md / evaluator findings to address
### [HIGH] Repair #8 not landed — `_build_pools` ignores `precomputed`
- File: `workspace/trading/snapshot.py`
- Required fix: `_build_pools` must consume precomputed detector outputs passed from `build()` so detectors are not re-run on identical candles.
  - Use raw `detect_swings` output per TF for swing pools.
  - Use HTF `detect_fvg` and `detect_order_blocks` outputs for IRL pools.

### [MEDIUM] Missing scoring behavior tests
- Add tests in `tests/phase2/` to pin:
  - proximity does not dominate,
  - monotonicity of key factors,
  - wrong-side exclusion,
  - clean_path penalty,
  - breakdown sums to score,
  - and a call-count spy test for repair #8.

## Planned changes (mapping to blueprint components)
1. **snapshot.py**
   - Thread raw detector outputs through `build()` into `_build_pools(precomputed=...)`.
   - Update `_build_pools` to use `precomputed` when provided, falling back to running detectors only when missing.
   - Keep `_nearest_dol` byte-identical.

2. **tests/phase2/**
   - Extend `tests/phase2/test_fib_liquidity_cycle.py` with the promised scoring tests.
   - Add a spy/call-count test using `unittest.mock.patch` to ensure `detect_swings` is not called inside `_build_pools` when precomputed swings are supplied, and HTF `detect_fvg`/`detect_order_blocks` are not re-run.

No changes are planned outside `workspace/` except additive/adjustment tests under `tests/`.
