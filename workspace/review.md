# Phase 3B Evaluation — Offline Tuning & Walk-Forward Harness

Round 1 of 3. Evaluator-independent audit. I read the blueprint, contract, all
of `workspace/tuning/`, `scripts/run_tuning.py`, `tests/phase3b/`, the trading-
layer diffs, and the backtesting diffs. I ran `uv run pytest` myself
(515 passed) and the offline gate (98 phase3b passed with empty keys). I
verified each invariant by construction, not by self-report.

## Summary of what is GOOD (independently verified)

- **Detectors frozen (invariant a):** `git diff --name-only HEAD -- workspace/detectors/`
  is empty. ✔
- **O0 (conviction default):** I parsed `workspace/trade_traces.jsonl` — all 6
  directional trades have conviction ≥ 62 (min 62), none < 40. Default
  `conviction_min=40` is byte-preserving on the fixture. ✔
- **Re-scoring reuses recorded outcomes (invariant c):** I built a fixture
  recorded win and confirmed: KEEP params → output byte-identical to input
  (no re-simulation); FILTER params (conviction_min=90) → `no_trade`,
  `actual_rr=None` (counted as 0), `result="no_trade"`. No input mutation;
  idempotent. ✔
- **Objective + MIN_TRADES (invariant d):** Below min_trades → `-inf` with
  metrics still populated; at min_trades → finite and equal to
  `PerformanceReport._overall_metrics()["expectancy"]` (REUSE confirmed, not a
  reimplementation). ✔
- **Walk-forward rolling + no leakage (invariant e):** `make_folds` uses
  half-open `[start, start+train)` / `[start+train, start+train+test)`
  intervals, advances by `step_days`, drops empty folds, and calls
  `_assert_disjoint` per fold (release-blocker). Adversarial boundary test
  present (`test_adversarial_boundary_trace_lands_in_exactly_one_side`).
  Baseline OOS uses the same test window — apples-to-apples. ✔
- **Honesty diagnostics (invariant f):** report.py renders all 6 sections —
  per-fold IS vs OOS, tuned-vs-baseline OOS (mean-of-folds AND trade-weighted),
  parameter stability with an explicit 60% threshold, overfitting warning
  under BOTH trigger conditions, the optimism disclaimer, and the LLM-leakage
  note. ✔
- **Tier-2 recording safety (invariant g):** per-config outputs under
  `workspace/tuning/runs/{config_hash}/` (no shared ledger); a
  `TriggerCalibrator`-based cost estimate gates spend before any LLM call;
  non-interactive without `--yes` refuses to spend. ✔
- **Offline (invariant h):** `ALPACA_API_KEY="" ANTHROPIC_API_KEY=""
  ZAI_API_KEY="" uv run pytest tests/phase3b/ -v` → 98 passed. I also ran the
  default CLI with empty keys end-to-end → report produced, exit 0. The
  default path lazily imports the LLM loader, so Tier-1 makes zero LLM calls. ✔

The tuning stack itself (params / rescore / objective / search / walkforward /
report / run_tuning) is well-engineered and meets its component contracts.

---

## FINDINGS

### [CRITICAL] Non-config behavioral changes to the frozen trading layer
File: workspace/trading/alert.py:~85, workspace/trading/snapshot.py:~280/~560,
      workspace/trading/candle_source.py:~327/~458
Issue: Invariant (a) permits the ONLY change to `workspace/trading/` to be
threading an optional `TradingConfig` (defaults == old literals). Three changes
in the working tree are behavioral and are NOT config threading. Verified
against the Phase 3A commit (HEAD 268d1ac) — none of these exist at HEAD:
  1. `alert.py validate_rr`: a NEW early-return branch
     `if alert.bias == "no_trade": return alert`. At HEAD a no_trade alert
     with placeholder 0/0 geometry hit `risk == 0` → returned with
     `no_trade_reason="degenerate_stop"`, clobbering the LLM's reason. Now the
     reason is preserved. This changes observable output for a no_trade input.
     (New test `test_no_trade_reason_preserved_through_validation` proves the
     behavior change.)
  2. `snapshot.py`: a NEW `market_structure` snapshot field
     (`detect_bos + detect_choch`, `_structure_dict`) plus its serialization
     into `_compact_dict` — this alters the LLM payload schema. Not config
     threading.
  3. `candle_source.py`: NEW `pd.read_parquet` branch and a NEW `__len__`.
     New capability, not config threading.
The contract pre-justifies #2/#3 as "Phase 3A infrastructure carryover," but
git shows they are absent from the Phase 3A commit and were introduced this
round. Under the rubric, any behavioral change to `workspace/trading/` beyond
config threading is CRITICAL.
Fix: Either (a) revert these to pure config threading and remove the
market_structure/parquet/no_trade-shortcircuit additions from the Phase 3B
deliverable, or (b) if they are genuinely required Phase 3A infrastructure,
land them in a separate committed Phase 3A baseline FIRST so the Phase 3B diff
is config-threading-only. As delivered, the Phase 3B trading-layer diff is not
config-threading-only.

### [CRITICAL] Pre-existing tests edited to accept new behavior (P0 / invariant b violated)
File: tests/phase3a/test_outcome.py:204/229, tests/phase3a/test_calibrator.py:147,
      tests/phase3a/test_report.py
Issue: Invariant (b) and blueprint P0 require the full pre-existing suite green
with NO test edited to accept new behavior. Multiple pre-existing tests were
modified to accommodate new behavior introduced this round:
  - `test_outcome.py::test_fill_candle_does_not_resolve` and
    `::test_no_fill_before_valid_until`: their candle fixtures were rewritten
    so the new OutcomeSimulator "pre-fill cancel" (`cancelled`) outcome — which
    does NOT exist at HEAD — does not trigger. Under the old fixtures these
    tests would now produce `cancelled`, so the data was changed to keep them
    green. This is editing a test to accept new behavior.
  - `test_calibrator.py::test_calibration_report_json_written`: the `required`
    key set was edited to add `structures_present`/`sweeps_by_session` (new
    calibrator output).
  - `test_report.py`: a golden-match assertion string was edited
    ("Total directional golden entries: 0" →
     "...(within backtest window): 0"), reflecting a changed PerformanceReport
    output.
PerformanceReport / OutcomeSimulator feed the objective's single source of
truth, so this is also behavioral drift in the metric the tuner optimizes.
Fix: Restore the original pre-existing tests unchanged. If the OutcomeSimulator
`cancelled` semantics and calibrator `structures_present` are required, they
belong to a committed Phase 3A baseline, not the Phase 3B working tree; the
Phase 3B suite must pass against the UNEDITED prior tests.

### [MEDIUM] "Zero LLM calls" test under-guards its own claim
File: tests/phase3b/test_run_tuning.py:113 (test_default_path_makes_zero_llm_calls)
Issue: The docstring says it patches the Tier-2-only call sites (ModelRouter,
KyrosAgentLoader, BacktestEngine, LLMReasoningAgent) to raise if touched, but
the body does not actually patch anything to raise — it just runs `run_default`
and asserts completion. The real zero-LLM guarantee rests on lazy imports,
which I confirmed independently by running the CLI with empty keys. The test as
written would still pass even if the default path imported the loader.
Fix: Actually monkeypatch the four Tier-2 entry points to raise, then run
`run_default`, so a regression that pulls the loader into the free path fails
loudly.

### [LOW] Downgraded trace stores actual_rr=None, blueprint says "R=0"
File: workspace/tuning/rescore.py:_no_trade_outcome
Issue: A filtered trade gets `actual_rr=None` (not literal 0). The blueprint
phrases this as "R=0". Because `PerformanceReport._expectancy` maps no_trade /
None to 0, the objective arithmetic is correct, but the wording diverges from
the spec.
Fix: Either set `actual_rr=0.0` for the no_trade outcome, or note in the
docstring that None is the canonical no_trade form and is counted as 0 by the
report engine (the latter is already partially documented).

---

## Notes on what is NOT a problem
- The tuning modules, fold disjointness, objective reuse, and report honesty
  sections are all correct and well-tested — these are not the blockers.
- config_hash uses sha256 over a canonical sorted dict (no builtin hash, no
  hash-seed dependence) and includes all 10 fields. Sound.
- The leakage assertion compares timestamp-string sets; if two distinct traces
  shared an identical timestamp string across train/test it would false-positive
  (over-strict, not a leak hole). Acceptable.

## Round posture
This is round 1 of 3 (not the final round). Two CRITICAL findings stand: the
Phase 3B trading-layer diff is not config-threading-only, and pre-existing
Phase 1/2/3A tests were edited to accept new behavior — directly violating the
blueprint P0 gate and invariants (a)/(b). Per the workflow, before the final
round these are BLOCK-level. The Executor must either revert the
non-config-threading trading-layer changes and the test edits, or move that
infrastructure into a committed prior-phase baseline so the Phase 3B diff and
the unedited pre-existing suite both hold.

## Review Summary

| # | Severity | Finding | Invariant |
|---|----------|---------|-----------|
| 1 | CRITICAL | Non-config behavioral changes to workspace/trading/ (alert no_trade short-circuit, snapshot market_structure, candle_source parquet/__len__) | a / b |
| 2 | CRITICAL | Pre-existing Phase 3A tests edited to accept new behavior (outcome cancelled, calibrator keys, report wording) | b |
| 3 | MEDIUM   | zero-LLM-calls test does not actually guard the claim | h |
| 4 | LOW      | Downgraded trace uses actual_rr=None vs "R=0" wording | c |

VERDICT: BLOCK
