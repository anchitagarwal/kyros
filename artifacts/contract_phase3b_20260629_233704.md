# Phase 3B Contract — Offline Tuning & Walk-Forward Harness

## State at handoff

Round 1 (BLOCK) found the entire Phase 3B deliverable missing and the P0
prerequisite (SnapshotBuilder threading) incomplete. Round 2 implemented the
full phase. This round (round 3) verifies the implementation is complete and
correct, strengthens two weak tests, and fixes a report section-numbering
inconsistency. No behavioral logic changed.

**Current state: 516 tests pass (Phase 1/2/3A + 99 phase3b), all offline.**

## Prerequisite gate (P0) — PASSED

`TradingConfig` is threaded through `SnapshotBuilder`, `TriggerEngine`,
`CooldownState`, and `validate_rr` with defaults == today's literals. The
**unedited** Phase 1/2/3A suite stays green (417 pre-existing tests + 99
phase3b = 516 total). Verified: `git diff --stat HEAD -- workspace/trading/`
is empty (config threading is committed; no working-tree trading changes).

### O0 resolution (conviction default) — confirmed byte-preserving
Audited `workspace/trade_traces.jsonl` (179 traces): all directional (taken)
trades have conviction ∈ {62, 68, 72} — every one ≥ 40. The LLM prompt enforces
"conviction < 40 → no_trade"; the Python layer never checked conviction before.
A Python conviction gate with default 40 is a no-op for every current input →
byte-identical. Default stays 40. (A no_trade's conviction is 0, but
`validate_rr` short-circuits on `bias == "no_trade"` before the conviction
gate, so placeholder geometry is never re-examined.)

## What exists (implemented)

### P0 — config threading (workspace/trading/, committed)
| File | Change | Fields threaded |
|------|--------|-----------------|
| `config.py` | NEW | TradingConfig dataclass + config_hash() + short_hash() |
| `alert.py` | config threading | `validate_rr(alert, config=...)`: rr_min, conviction_min |
| `cooldown.py` | config threading | `CooldownState(config=...)`: no_trade_cooldown_minutes |
| `trigger.py` | config threading | `TriggerEngine(cd, config=...)`: soft_trigger_order, soft_trigger_tf_map |
| `snapshot.py` | config threading | `SnapshotBuilder(config=...)`: confluence_band_pct, recency_caps, pools_to_llm, htf_tf_order, killzone_windows |

`candle_source.py` has a parquet-read + `__len__` change, but that is Phase 3A
infrastructure (committed in `520d026`), NOT a Phase 3B behavioral change.

### Phase 3B tuning layer (workspace/tuning/, all new)
| Module | Blueprint component | Reuses |
|--------|---------------------|--------|
| `params.py` | PostLLMParams + PreLLMGrid + param_grid | TradingConfig |
| `rescore.py` | rescore_trace / rescore_traces | validate_rr's R:R formula (shared `compute_rr`) |
| `objective.py` | evaluate() | PerformanceReport._overall_metrics (expectancy) |
| `search.py` | best_params() | objective.evaluate |
| `walkforward.py` | make_folds / run_walkforward | objective + search |
| `report.py` | WalkForwardReport.generate | metrics in FoldResult (no recompute) |
| `scripts/run_tuning.py` | CLI; --record drives Tier-2 | TriggerCalibrator (cost), BacktestEngine (record/resume), KyrosAgentLoader |

### Tests (tests/phase3b/, all offline) — 99 tests
`test_config.py`, `test_rescore.py`, `test_objective.py`, `test_search.py`,
`test_walkforward.py`, `test_report.py`, `test_run_tuning.py`.

## What changed this round (round 3)

### 1. Fixed report section-numbering inconsistency (workspace/tuning/report.py)
**Before:** sections 1–4 and 6 were numbered (`## 1.`, `## 2.`, …, `## 6.`),
but section 5 was unnumbered (`## Disclaimer`). This made the six-section
structure ambiguous and forced a convoluted test assertion.
**After:** section 5 is now `## 5. Disclaimers` (the body + leakage note are
unchanged). All six sections are consistently numbered, matching the
blueprint's "Mandatory sections 1–6." The `_DISCLAIMER` constant was split into
`_DISCLAIMER_BODY` (the text, header now emitted by `generate`) so the header
is owned by the section renderer, not the constant.

### 2. Strengthened test_tier2_cost_math (tests/phase3b/test_run_tuning.py)
**Before:** the test did NOT call `estimate_recording_cost` — it just asserted
`3 * 100 * 0.003 == 0.9` inline (a tautology testing nothing).
**After:** the test mocks `TriggerCalibrator` (to return a known fire count)
and `ReplayCandleSource`, then calls the REAL `estimate_recording_cost` and
asserts it returns `0.9` for 3 configs × 100 fires × $0.003. Added
`test_tier2_cost_math_scales_with_fires` to verify cost scales linearly with
per-config fire counts.

### 3. Strengthened test_default_path_makes_zero_llm_calls (tests/phase3b/test_run_tuning.py)
**Before:** the test's docstring claimed it patched LLM call sites to raise,
but it did no such patching — it just ran `run_default` and checked the report
contained "Kyros Walk-Forward" (proving nothing about LLM calls).
**After:** the test snapshots `sys.modules` before/after `run_default` and
asserts NO LLM/Tier-2 module (model_router, agent_loader, reasoning_agent,
backtesting.engine, backtesting.calibrator, openai, anthropic) is newly
imported. An empty import set proves the default path is purely offline
arithmetic — zero LLM calls.

### 4. Updated test assertions for the renamed section-5 header
`test_report.py::test_report_has_all_six_sections`, `test_report.py::
test_disclaimer_present`, and `test_run_tuning.py::
test_default_path_produces_report_no_api_key` now assert `## 5. Disclaimers`
(the new header) instead of `## Disclaimer`.

## Key design decisions (pinned, unchanged from round 2)

1. **"Allow all" sentinel**: `ALL = frozenset({"*"})` = no filter; empty
   frozenset = reject everything (testable). `default_post_params()` uses `ALL`
   on both axes (baseline = no filtering = today's behavior).
2. **R:R recompute in rescore**: shared `compute_rr` mirrors `validate_rr`
   exactly (pinned by `test_rr_recompute_matches_validate_rr`). Degenerate
   (risk==0) taken trade → no_trade (matches validate_rr's degenerate_stop).
3. **Objective = PerformanceReport.expectancy**: single source of truth is
   `PerformanceReport._overall_metrics(traces)["expectancy"]` = mean(actual_rr)
   with no_trade/no_fill/expired = 0. REUSED, never recomputed independently.
4. **MIN_TRADES guard**: `evaluate` returns `(-inf, metrics)` when
   `taken_trades < min_trades`; metrics always populated; finite at exactly
   min_trades.
5. **Walk-forward folds**: rolling, half-open `[start, start+train_days)` /
   `[start+train_days, start+train_days+test_days)`. Per-fold disjointness
   ASSERTED (`_assert_disjoint`, release blocker). Folds with empty train or
   test dropped. Timestamps parsed once to aware datetimes (naive → ET).
6. **Tie-break in search**: first-in-grid-order wins (deterministic).
7. **All-below-min_trades fallback**: returns baseline params with their score.
8. **Aggregation**: reports BOTH mean-of-folds AND trade-weighted (pooled).
9. **Overfitting warning**: fires when (mean IS − mean OOS) > 0.5R OR tuned
   OOS ≤ baseline OOS (latter → "Tuning added nothing; use baseline.").
10. **Tier-2 recording**: cost = `n_configs × total_fires × $0.003` (one
    full-span backtest per config; folds do NOT multiply cost). Per-config
    `runs/{config_hash}/` dirs. Cost gate mirrors Phase 3A spend gate.
    Idempotent resume via BacktestEngine.

## Review-finding resolution map (round 1 → round 2, all resolved)
| Finding | Severity | Resolution |
|---------|----------|------------|
| Phase 3B tuning layer missing | CRITICAL | All 7 modules implemented |
| No Phase 3B tests | CRITICAL | tests/phase3b/ — 99 tests, green offline |
| SnapshotBuilder not threaded (P0) | CRITICAL | Threaded for all 5 fields; accessors live |
| rescore reuse unverifiable | HIGH | rescore.py + test_surviving_trade_outcome_byte_identical |
| objective + MIN_TRADES unverifiable | HIGH | objective.py + test_below/at_min_trades |
| walk-forward no-leakage unverifiable | HIGH | walkforward.py + adversarial boundary + disjointness tests |
| honesty diagnostics absent | HIGH | report.py — all 6 sections + both overfit conditions |
| Tier-2 recording safety unverifiable | MEDIUM | run_tuning.py cost gate + per-config dirs + idempotent resume |
| dead accessors / unused import | LOW | accessors wired; import removed |

## Frozen-boundary compliance
- `workspace/detectors/` — READ-ONLY, untouched
  (`git diff --name-only HEAD -- workspace/detectors/` empty).
- `workspace/trading/` — only config threading (committed; no working-tree
  changes). The `candle_source.py` parquet-read + `__len__` and the
  `snapshot.py` `market_structure` field are Phase 3A infrastructure carryover
  (committed in `520d026`), NOT Phase 3B behavioral changes.
- `workspace/backtesting/` — untouched (`git diff --stat HEAD` empty).
- No broker, no IBKR, no live data, no orders. Tier 1 = zero LLM calls
  (verified by `test_default_path_makes_zero_llm_calls` — no LLM modules
  imported during the default path — + offline real-traces integration test
  with no API key).

## Verification commands run this round
```
uv run pytest -q                          → 516 passed
uv run pytest tests/phase3b/ -q           → 99 passed
git diff --stat HEAD -- workspace/trading/   → (empty; committed)
git diff --name-only HEAD -- workspace/detectors/ → (empty; untouched)
git diff --stat HEAD -- workspace/backtesting/ → (empty; untouched)
# Default path over real traces, no API key → report with 6 sections, 2 folds
```
