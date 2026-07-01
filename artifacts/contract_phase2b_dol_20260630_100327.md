# Phase 2B Contract — Fibonacci Levels & ERL→IRL→ERL Liquidity Cycle / DOL

## State at resume
- `.kyros_state.json`: phase_2b, branch phase-2b-fib-liquidity-cycle, last_test_status=pass, last_review_status=APPROVE (Phase 3B), evaluator_round=0.
- `uv run pytest` (pre): 516 passed (Phases 1+2+3A+3B fully green). No review.md for Phase 2B yet (first round).
- `uv run pytest` (post): 578 passed (516 + 21 fibonacci + 41 phase2 cycle). No regressions.

## What exists (read, not modified)
- `detectors/premium_discount.py`: anchor = most-recent confirmed swing high + low via `detect_swings`; `direction="up"` iff `last_low["index"] < last_high["index"]`; `R = range_high - range_low`; up → `price(f)=range_high - f*R`, down → `price(f)=range_low + f*R`; equilibrium = `range_low + R/2`; confirm_index = `max(high_idx, low_idx)`; timestamp = `candles[confirm_index]["timestamp"]`. Edge cases → `[]`: empty, no high/low pair, `range_high == range_low`. Ambiguity note in module docstring.
- `detectors/market_structure.py`: `detect_swings` (fractal pivots, strict >/<, lookahead-safe). `detect_bos`/`detect_choch` emit `break_index`/`timestamp`.
- `detectors/fair_value_gaps.py`: `detect_fvg` emits `midpoint` = `(top+bottom)/2` (CE).
- `detectors/order_blocks.py`: `detect_order_blocks` emits `top`/`bottom`; centre = `(top+bottom)/2`.
- `detectors/liquidity.py`: `detect_liquidity_sweeps` emits `type` ∈ {`sweep_bsl`,`sweep_ssl`}, `swept_level`, `sweep_index`, `timestamp`.
- `detectors/displacement.py`: `detect_displacement` emits `type` ∈ {`displacement_bullish`,`displacement_bearish`}, `index`, `timestamp`.
- `trading/snapshot.py`: `LiquidityPool` dataclass; `MarketSnapshot` dataclass; `SnapshotBuilder.build()`; `_build_pools`; `_nearest_dol`; `_compact_dict`; per-detector `_xxx_dict` mappers.
- `trading/config.py`: frozen `TradingConfig`; `_DEFAULT_RECENCY_CAPS`; `config_hash()` (no test asserts a literal hash — only stability/differs-on-change, confirmed across tests/phase3b).
- `trading/reasoning_agent.py`: `ICT_SYSTEM_PROMPT`; `LLMReasoningAgent.reason()` → `parse_llm_json`.
- `trading/trigger.py`: hard gates include `nearest_dol is not None`.
- Tests: no test asserts exact compact-dict key set or pool `to_dict` golden (verified by grep).

## What I changed (mapped to blueprint components) — IMPLEMENTED

### 1. `detectors/fibonacci.py` (NEW) — detect_fibonacci ✓
Pure detector. Reuses `detect_swings` (anchor) + premium_discount's `price(f)` convention. Returns `[]` or `[dict]`. Canonical numbers (100→200, R=100) asserted exactly in tests/test_fibonacci.py (both directions). Edge cases → `[]`: empty, no swing pair, degenerate range. Carries premium_discount's ambiguity note verbatim. Key formatting: extension keys `"-1.0"` etc.; OTE keys `"0.5"` etc. via `_ratio_key` (repr-based). ote.zone ordered ascending [lo, hi] in BOTH directions (canonical table: up [121.0,138.0], down [162.0,179.0]).

### 2. `detectors/__init__.py` (EDIT) ✓
Added `from .fibonacci import detect_fibonacci` + `"detect_fibonacci"` in `__all__` + docstring line. Nothing else. (git diff confirms only these additions.)

### 3. `trading/config.py` (EDIT, additive) ✓
New knobs: `fib_retracements`, `fib_golden_pocket`, `fib_ote_grid`, `fib_ote_primary`, `fib_retracement_target`, `fib_extensions`, `fib_anchor_lookback`, `irl_sources`, `dol_use_cycle`. Added `("fib_levels", 1)` to `_DEFAULT_RECENCY_CAPS`. Added all new knobs to `config_hash()` canonical dict. Existing defaults byte-identical (verified by diff). All new fields tuples/immutables. Added `irl_sources_set()` accessor.

### 4. `trading/snapshot.py` (EDIT, additive + 1 non-additive serializer surface) ✓
- `LiquidityPool`: added `scope: str = "external"`, `role: str = ""` (defaulted → backward compatible). `to_dict()` appends `scope`/`role`.
- `MarketSnapshot`: added `fib_levels`, `liquidity_cycle`, `dol_target` (defaulted).
- `build()`: per-TF `fib_levels[tf] = detect_fibonacci(...)`; `_htf_fib()` selects first non-empty HTF TF; `_build_pools` extended for IRL sources (FVG midpoint→`fvg_ce`, OB centre→`ob_ce`, fib equilibrium→`equilibrium`, fib OTE 0.705→`ote`) gated by `config.irl_sources`, plus swing ERL (`role="swing"`); scope refinement via HTF dealing range (strictly inside → internal, at/beyond → external). Then `_derive_liquidity_cycle`, then `_dol_target`.
- `_derive_liquidity_cycle`: most-recent sweep across TFs by timestamp (coarser TF wins ties); `last_swept_erl_side`; `current_leg` = `expand_to_erl` if reversal-direction displacement/BOS strictly after sweep else `seek_irl`; `target_erl_side` = opposite; `agrees_with_htf_bias`. None if no sweep. Stateless, documented.
- `_dol_target`: augments `_nearest_dol` (unchanged). cycle None / `dol_use_cycle False` → `_nearest_dol`. seek_irl → nearest unswept internal pool in reversal direction. expand_to_erl → nearest unswept external pool on target side beyond equilibrium. Bias conflict → prefer htf_bias, set `agrees_with_htf_bias=False`. Falls back to `_nearest_dol` when cycle selection finds nothing.
- `_compact_dict`: added `fib_levels` (per-TF latest via `_fib_dict`), `liquidity_cycle`, `dol_target`. `_fib_dict(fib)` 2dp: direction/equilibrium/golden_pocket/ote_primary/ote_zone/retracement_target/extensions/premium_array.
- `_nearest_dol` byte-identical (verified by diff). Existing field defaults byte-identical.

### 5. `trading/reasoning_agent.py` (EDIT, prompt text only) ✓
`ICT_SYSTEM_PROMPT`: added Step 1.5 (read `liquidity_cycle`), Step 3 DOL prefer `dol_target`, Step 4 intermediate-liquidity (only unswept opposing EXTERNAL blocks), Step 5 OTE modifier (+15 golden_pocket, +10 at ote_primary), entry logic (premium→short/discount→long, golden_pocket/OTE entries, 0.382 partial + negative extensions). Output JSON schema UNCHANGED (verified by test). KB stays untrusted (no `knowledge_base`/`authoritative` directive). Existing model definitions + DOL-FIRST + INTERMEDIATE LIQUIDITY CHECK preserved.

### 6. Tests (NEW) ✓
- `tests/test_fibonacci.py` (21 tests): exact up/down canonical numbers; edge cases (empty/no-pair/degenerate); purity (no mutation, same-input-same-output); convention parity with premium_discount; parameterization honored; output shape.
- `tests/phase2/test_fib_liquidity_cycle.py` (41 tests): ERL/IRL classifier (defaults, gating, scope refinement inside/at-boundary); cycle derivation (seek_irl/expand_to_erl, agrees flag, no-sweep→None, latest-sweep-wins, stateless); DOL selector (cycle-None→nearest_dol, dol_use_cycle False, seek_irl internal, expand_to_erl external beyond equilibrium, bias-conflict flip, empty→None, fallback); snapshot integration (fib_levels/scope/role/liquidity_cycle/dol_target in build() + to_compact_dict()); mocked-LLM e2e (fib/cycle alert → valid AlertPayload, parse-failure → no_trade); prompt references new fields + unchanged schema + KB untrusted; determinism (snapshot + compact dict byte-identical).
- Regression: full suite green. No pre-existing pool `to_dict` golden existed to update (verified by grep); new tests assert scope/role.

## Scope verification (git diff)
- Modified code files: `detectors/__init__.py`, `trading/config.py`, `trading/reasoning_agent.py`, `trading/snapshot.py`.
- New code files: `detectors/fibonacci.py`, `tests/test_fibonacci.py`, `tests/phase2/test_fib_liquidity_cycle.py`.
- No other detector modified (only `__init__.py` + new `fibonacci.py`).
- `_nearest_dol` byte-identical (diff-verified).
- `backtesting/`, `tuning/`: untouched (diff-verified).
- config_hash: no test asserts a literal hash; adding keys is safe. Tuning uses `config_hash()` at runtime (not a frozen literal).
- Note: `workspace/blueprint.md`, `workspace/problem.md`, `workspace/review.md` show as modified/deleted in git status — these are pre-existing planner-phase artifacts (dated Jun 29 23:44-23:46, before execution), NOT my changes. My scope is confined to the code files above.

## Cross-cutting validation gates
1. Scope: only whitelist files touched (verified). ✓
2. Frozen byte-equality: every detector except fibonacci.py/__init__.py unchanged; _nearest_dol unchanged; existing config/snapshot defaults unchanged (diff-verified). ✓
3. Fib canonical numbers: exact assertions pass (both directions). ✓
4. Suite green: 578 passed; no pre-existing golden broken. ✓
5. Offline: no network, no API key, mock/replay sources only. ✓
6. Determinism: double-build equality on snapshot + compact dict (tested). ✓
7. KB untrusted: no code or prompt treats knowledge_base/ as authoritative. ✓
