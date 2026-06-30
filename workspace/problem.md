## Problem Statement


Build the Phase 3B Offline Tuning & Walk-Forward Harness for Project Kyros: an
offline parameter optimizer that learns better trading parameters from realized
outcomes and validates them honestly with walk-forward analysis. The LLM weights
are never touched. Online/live adaptation is OUT OF SCOPE for this phase.

Phases 1, 2 and 3A are complete and validated. Phase 3A already produces
workspace/trade_traces.jsonl — one line per LLM-triggered alert with a
deterministically simulated outcome (win/loss/expired/no_fill + actual_rr). That
file is the ground-truth reward signal this phase tunes against.

THE CORE INSIGHT that makes this cheap:
Post-LLM gates (conviction floor, R:R floor, allowed model types, allowed
killzones) decide only WHETHER TO TAKE an already-recorded trade. They never
change entry_zone/stop/target, so they do not change a taken trade's outcome.
Re-scoring them is therefore pure arithmetic over the existing trade_traces.jsonl:
filter → reuse the recorded outcome → recompute metrics. Zero LLM calls, zero
re-simulation (Tier 1). Pre-LLM params (killzone windows, confluence %, recency
caps, HTF timeframe order) change WHAT fires and WHAT the LLM sees, so they need a
fresh LLM backtest per config — Tier 2, an outer loop, explicitly cost-gated.

The harness has two tiers plus a config-injection prerequisite:

0. TradingConfig (prerequisite) — a dataclass holding every currently-hardcoded
   knob, threaded through SnapshotBuilder/TriggerEngine/CooldownState/validate_rr
   with defaults equal to today's literals. This is the ONLY permitted change to
   the otherwise-frozen workspace/trading/ layer, and it must be behavior-preserving.

1. Tier 1 — post-LLM re-scoring (free): re-score recorded traces under candidate
   PostLLMParams and recompute metrics by reusing PerformanceReport.

2. Tier 2 — pre-LLM config sweep (cost-gated): for a small grid of TradingConfig
   variants, run one full-span backtest each (reusing BacktestEngine), gated by a
   TriggerCalibrator cost estimate before any spend.

3. Walk-forward — rolling train/test split by timestamp: pick the best params on
   each training slice, evaluate out-of-sample on the next unseen slice, and
   compare against the baseline (default params) on the same windows.


## End Goal


A tested, runnable offline tuning harness that:

1. workspace/trading/config.py — TradingConfig dataclass + config_hash(), threaded
   through the trading layer with defaults that reproduce current behavior exactly
   (the full pre-existing Phase 1/2/3A test suite stays green with no behavioral edits).

2. workspace/tuning/rescore.py — rescore_trace/rescore_traces: convert filtered
   trades to no_trade (R=0), REUSE the recorded outcome for taken trades (never
   re-simulate). Documented limitation: ignores cooldown re-interaction.

3. workspace/tuning/objective.py — evaluate(traces, params) → (expectancy, metrics),
   reusing PerformanceReport's metric computation; returns -inf below MIN_TRADES.

4. workspace/tuning/search.py — best_params(train_traces, grid, min_trades).

5. workspace/tuning/walkforward.py — make_folds (rolling, by timestamp, no
   train/test leakage) and run_walkforward producing per-fold IS and OOS results
   plus a baseline comparison.

6. workspace/tuning/report.py — WalkForwardReport.generate → workspace/walkforward_report.md
   with per-fold IS vs OOS, aggregate OOS vs baseline, parameter-stability, an
   overfitting warning, the Phase 3A optimism disclaimer, and an LLM-leakage note.

7. scripts/run_tuning.py — CLI. Default (post-LLM only, default config) makes ZERO
   LLM calls over an existing trade_traces.jsonl. --record drives the cost-gated
   Tier-2 sweep first.

Full test suite (tests/phase3b/) runs offline — no API key, Tier-1 path makes no
LLM calls. All Phase 3A components (PerformanceReport, OutcomeSimulator,
TriggerCalibrator, BacktestEngine) are imported and REUSED, not reimplemented.


## Constraints


HARD CONSTRAINTS:
- workspace/detectors/ is READ-ONLY — never modify.
- workspace/trading/ is SEMI-FROZEN — the ONLY permitted change is threading an
  optional `config: TradingConfig = TradingConfig()` through SnapshotBuilder,
  TriggerEngine, CooldownState, and validate_rr, replacing hardcoded literals with
  config reads. Defaults MUST be byte-identical to today's behavior; the existing
  test suite must stay green with no behavioral test edits. Any other change to
  workspace/trading/ is a critical scope violation.
- No broker, no IBKR, no live market data, no order placement.
- All tuning tests run OFFLINE — no API key. The Tier-1 (post-LLM) path makes ZERO
  LLM calls.
- Re-scoring REUSES each recorded trade's outcome and MUST NOT re-simulate a
  filtered post-LLM trade. Filtered trades become no_trade (R=0).
- Walk-forward is ROLLING and splits strictly by timestamp; a trace may appear in a
  fold's test OR its train, never both (no leakage). This invariant is critical —
  leakage silently inflates out-of-sample results.

REUSE (do not reimplement):
- PerformanceReport (workspace/backtesting/report.py) — objective + report metrics.
- OutcomeSimulator (workspace/backtesting/outcome.py) — only when entry/stop/target
  change (never in Tier 1).
- TriggerCalibrator (workspace/backtesting/calibrator.py) — Tier-2 cost gate.
- BacktestEngine (workspace/backtesting/engine.py) — Tier-2 per-config recording,
  idempotent resume.
- KyrosAgentLoader.get_model_engine("trading", fallback_role="executor") — model
  selection.

MODULE LAYOUT:
  workspace/trading/config.py        — TradingConfig dataclass + config_hash()
  workspace/tuning/
    __init__.py
    params.py        — PostLLMParams, PreLLMGrid, param_grid()
    rescore.py       — rescore_trace(), rescore_traces()
    objective.py     — evaluate()
    search.py        — best_params()
    walkforward.py   — make_folds(), run_walkforward()
    report.py        — WalkForwardReport
  scripts/run_tuning.py              — CLI; --record drives cost-gated Tier-2
  workspace/tuning/runs/{config_hash}/trade_{alerts,traces}.jsonl  — per-config
  workspace/walkforward_report.md    — human-readable walk-forward output

OBJECTIVE & WALK-FORWARD:
- Objective: expectancy_per_trade (mean actual_rr, no_trade=0), subject to a
  MIN_TRADES guard that rejects param sets with too few taken trades.
- Rolling window with configurable --train-days/--test-days/--step-days.
- Tier-2 LLM cost = n_pre_llm_configs × one full-span backtest (folds do NOT
  multiply cost — record each config once over the whole span, slice by timestamp).

HONESTY DIAGNOSTICS (mandatory in the report):
- Only walk-forward OOS numbers are trustworthy; report IS vs OOS AND tuned vs
  baseline OOS. If tuned OOS <= baseline OOS, tuning added nothing — state it plainly.
- Parameter-stability across folds: wildly different per-fold winners = noise.
- Carry forward the Phase 3A optimism disclaimer and add a leakage note (the LLM's
  training data overlaps the backtest period).

PERFORMANCE / VALIDITY:
- Tier 1 must run in milliseconds over an existing trade_traces.jsonl.
- Determinism: same traces + same params → same metrics and same fold splits.
