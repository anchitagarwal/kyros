# Phase 3B Evaluation — Offline Tuning & Walk-Forward Harness

**Evaluator round:** 1 of 3 (per orchestrator). The contract self-describes as
"round 3"; I grade independently of that claim.
**Verdict basis:** independent re-run of pytest + direct invariant probing
(not the Executor's self-report).

## Independent verification performed

| Check | Method | Result |
|-------|--------|--------|
| Full suite | `ALPACA/ANTHROPIC/ZAI_API_KEY="" uv run pytest -q` | **516 passed in 52s** |
| Phase 3B offline | keys blanked, `pytest tests/phase3b/ -q` | **99 passed** |
| Frozen detectors | `git diff --name-only HEAD -- workspace/detectors/` | empty |
| Frozen trading (working tree) | `git diff --stat HEAD -- workspace/trading/` | empty |
| Frozen backtesting | `git diff --stat HEAD -- workspace/backtesting/` | empty |
| rescore reuse | crafted fixture, kept vs filtered | byte-identical / no_trade |
| objective MIN_TRADES | direct call | -inf below, finite at |
| walk-forward leakage | crafted dated traces + boundary | disjoint, boundary→test |
| config defaults | direct construction | all literals matched |

## Invariant-by-invariant findings

### a. Frozen boundaries — PASS
`workspace/detectors/`, `workspace/trading/`, and `workspace/backtesting/` have
empty working-tree diffs vs HEAD. The trading-layer config threading is
committed (per `git log`, in the Phase 2/3A history), and the only behavioral
content is reading config fields whose defaults equal the old literals. No
behavioral change to the trading layer beyond optional-config threading.

### b. TradingConfig behavior-preserving — PASS
- The **unedited** Phase 1/2/3A suite (417 pre-existing tests) is green inside
  the 516-total run; no test was edited to accept new behavior.
- Direct construction confirms every documented literal:
  `rr_min=1.0`, `conviction_min=40`, `no_trade_cooldown_minutes=5`,
  `confluence_band_pct=0.001`, `pools_to_llm=5`, `htf_tf_order=("4h","1h")`,
  recency caps 10 (sweeps/displacements/inducements) / 5 (swings/fvg/ifvg/ob/
  breaker/volume_imbalance) / 3 (opening_gaps/po3).
- O0 (conviction default 40) is justified: `validate_rr` short-circuits on
  `bias=="no_trade"` before the conviction gate, and the audit shows all taken
  trades ≥ 40, so default-40 is a no-op on current inputs. Defensible.
- `config_hash()` is sha256 over a json-canonicalized, key-sorted blob (never
  builtin `hash()`): stable across constructions, sensitive to field changes.
  Verified directly.

### c. Re-scoring reuses recorded outcomes — PASS
Direct fixture probe (`/tmp/audit.py`):
- KEEP params → recorded `win` outcome returned **byte-identical**
  (`actual_rr==2.5`), input not mutated.
- FILTER params (conviction_min 70 > 65) → outcome becomes `no_trade`,
  `actual_rr=None` (counted 0). Input unchanged.
- Idempotent. `OutcomeSimulator` never imported on this path.
Filters only downgrade taken trades; non-taken (no_trade/no_fill/expired/
cancelled) are untouched. R:R recomputed from geometry via shared `compute_rr`
mirroring `validate_rr`.

### d. Objective + MIN_TRADES guard — PASS
`evaluate()` re-scores then calls `PerformanceReport()._overall_metrics`,
reading `expectancy` (= `_expectancy`, mean actual_rr, no_trade=0) and
`filled_count`. It does **not** reimplement the metric. Below `min_trades` →
`(-inf, metrics)` with metrics still populated; finite at exactly min_trades
(verified: 5 traces, min_trades=10 → -inf; min_trades=5 → 2.5).

### e. Walk-forward rolling + no leakage — PASS (most important)
Independent probe (`/tmp/wf.py`) on 12 daily traces, train=3/test=2/step=2:
- 5 folds; consecutive starts differ by exactly step_days (2).
- Every fold: `train_end == test_start`, train precedes test, **train∩test=∅**.
- Adversarial boundary trace at `train_end` lands in **test only** (half-open).
- `_assert_disjoint` runs per fold as a release-blocker assertion.
Baseline OOS is evaluated on the baseline config's same test window —
apples-to-apples.

### f. Honesty diagnostics — PASS
`report.py` emits all six numbered sections: (1) per-fold IS/OOS table,
(2) tuned-vs-baseline OOS (both mean-of-folds and trade-weighted),
(3) parameter stability with an explicit <60% instability flag, (4) overfitting
warning firing on BOTH conditions (IS−OOS > 0.5R, and tuned OOS ≤ baseline →
"Tuning added nothing; use baseline."), (5) Phase 3A optimism disclaimer +
LLM-leakage note, (6) degenerate-fold note. test_report.py (14 tests) green.

### g. Tier-2 recording safety — PASS
`scripts/run_tuning.py`: `estimate_recording_cost` uses `TriggerCalibrator`
(no LLM) and gates spend (`--yes` or interactive `y`; non-interactive without
`--yes` → exit 1, zero spend). Each config records to
`workspace/tuning/runs/{short_hash}/` — no shared ledger path across configs.
LLM agent built lazily only inside `record_config`/`_build_reasoning_agent`.

### h. Offline — PASS
`tests/phase3b/` (99 tests) pass with all keys blanked; tests `delenv` keys
rather than requiring them. `test_default_path_makes_zero_llm_calls` snapshots
`sys.modules` and asserts no model_router/agent_loader/reasoning_agent/
backtesting.engine/calibrator/openai/anthropic import on the default path —
a genuine zero-LLM proof, not a tautology. No live feed, no broker, no orders.

## Minor / non-blocking

[LOW] Doc path drift
File: workspace/contract.md
Issue: contract references `workspace/scripts/run_tuning.py`; the actual file
is `scripts/run_tuning.py` (project root). No functional impact — the default
path imports `tuning.*` from `workspace/` correctly via sys.path setup.
Fix: update the contract path reference.

[LOW] Round-number discrepancy
Issue: orchestrator says round 1 of 3; contract narrates "round 3". Cosmetic;
graded independently on merits.

## Review Summary

| Invariant | Severity if failed | Status |
|-----------|--------------------|--------|
| a. Frozen boundaries | CRITICAL | PASS |
| b. Config behavior-preserving | CRITICAL | PASS |
| c. Rescore reuses outcomes | HIGH | PASS |
| d. Objective + MIN_TRADES | HIGH | PASS |
| e. Walk-forward no leakage | CRITICAL | PASS |
| f. Honesty diagnostics | HIGH/MEDIUM | PASS |
| g. Tier-2 recording safety | HIGH | PASS |
| h. Offline / no live key | CRITICAL | PASS |
| Doc path / round labels | LOW | noted |

All CRITICAL and HIGH invariants verified independently and pass. The two
findings are LOW (documentation). 516 tests green offline. No blocking issues.

VERDICT: APPROVE
