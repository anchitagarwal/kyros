"""
main.py — Phase 3A: Evaluation Harness & Backtesting Engine

Run this to kick off the Planner → Executor → Evaluator cycle for Phase 3A.

Usage:
    uv run --env-file .env python main.py

Resets:
    rm workspace/blueprint.md workspace/contract.md workspace/review.md
"""

from kyros.core.orchestrator import Orchestrator, EscalationRequired

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3A TASK DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

PROBLEM_STATEMENT = """
Build the Phase 3A Evaluation Harness for Project Kyros: a backtesting engine
that runs the Phase 2 TradingLoop over historical NQ replay data, calls the LLM
to produce real AlertPayloads, simulates trade outcomes deterministically against
subsequent candles, and generates a structured performance report.

Phase 2 is complete and validated (101 tests, APPROVE). workspace/trading/ is
READ-ONLY — the evaluation harness calls it as a library. The backtest answers
one question before live deployment: does this system generate tradeable ICT
setups with positive expectancy?

The harness has four components:

1. TriggerCalibrator (no LLM) — diagnostic that maps TriggerEngine gate
   distribution and firing rate over the full historical period without spending
   API budget. Run this before BacktestEngine to catch miscalibration early.

2. BacktestEngine (LLM in the loop) — drives ReplayCandleSource through
   historical data, runs the complete TradingLoop including LLM inference,
   collects AlertPayload per trigger, passes to OutcomeSimulator.

3. OutcomeSimulator (no LLM, deterministic) — given an AlertPayload and
   subsequent candles, determines win/loss/expired by walking candles forward
   from alert timestamp. Uses the LLM's entry_zone/stop/target directly — no
   rules about what these should be. The LLM already decided; the simulator
   checks whether the market obliged.

4. PerformanceReport — aggregates outcomes into human-readable metrics and
   writes backtest_report.md. Also produces calibration_report.json from
   TriggerCalibrator output.
"""

END_GOAL = """
A tested, runnable evaluation harness that:

1. TriggerCalibrator.run(source) → CalibrationReport:
   - Processes full replay period without any LLM calls
   - Reports fires per killzone, gate block distribution, soft trigger breakdown
   - Writes workspace/calibration_report.json

2. BacktestEngine.run(source) → list[TradeTrace]:
   - Drives ReplayCandleSource end-to-end through historical data
   - Calls TradingLoop (including live LLM inference) per candle
   - Attaches OutcomeSimulator result to each AlertPayload
   - Writes each trace as a JSON line to workspace/trade_traces.jsonl
   - Resumes from last written trace on restart (idempotent)

3. OutcomeSimulator.simulate(alert, subsequent_candles) → TradeOutcome:
   - Lookahead-safe: uses ONLY candles with timestamp > alert.timestamp
   - Fill logic: first candle where price enters entry_zone → filled at entry_mid
   - Win: subsequent candle high >= target (long) or low <= target (short)
   - Loss: subsequent candle low <= stop (long) or high >= stop (short)
   - Ambiguous (both in same candle): conservative → loss
   - Expired: killzone ends with neither hit → no_fill

4. PerformanceReport.generate(traces) → str:
   - Writes workspace/backtest_report.md
   - Metrics: total alerts, no_trade rate, win rate, avg R won, avg R lost,
     profit factor, max drawdown (in R), by model type, by killzone,
     by timeframe of primary FVG/OB, golden dataset hit rate

Full test suite runnable offline — BacktestEngine tests mock the LLM and
OutcomeSimulator, TriggerCalibrator tests use MockCandleSource, all Phase 2
modules are imported from workspace/trading/ as-is (READ-ONLY).
"""

CONSTRAINTS = """
HARD CONSTRAINTS:
- workspace/trading/ is READ-ONLY — import as a library, never modify
- workspace/detectors/ is READ-ONLY — same rule as Phase 2
- No broker, no IBKR, no live market data, no order placement
- All tests must run offline — LLM calls mocked in tests, no API key required
- OutcomeSimulator MUST NOT use the alert candle itself for outcome resolution
  (only candles strictly after alert.timestamp); this is a CRITICAL correctness
  invariant — lookahead here invalidates the entire backtest

MODULE LAYOUT:
  workspace/backtesting/
    __init__.py
    data_loader.py     — load + cache NQ historical data; backends:
                         yfinance/alpaca (download), csv (local file)
    calibrator.py      — TriggerCalibrator, CalibrationReport
    engine.py          — BacktestEngine
    outcome.py         — OutcomeSimulator, TradeOutcome
    report.py          — PerformanceReport

  workspace/
    trade_traces.jsonl      — one JSON line per LLM-triggered alert + outcome
    calibration_report.json — TriggerCalibrator output
    backtest_report.md      — human-readable PerformanceReport output

TRADE TRACE SCHEMA (one JSON line per trace in trade_traces.jsonl):
{
  "trace_id":         str,          // uuid4
  "timestamp":        str,          // ISO — when alert fired
  "instrument":       "NQ",
  "killzone":         str,
  "trigger_reason":   str,          // which soft trigger fired
  "snapshot_summary": dict,         // compact snapshot — no raw candles
  "raw_llm_output":   str,          // raw string returned by model_router.call()
  "alert":            dict,         // AlertPayload as dict
  "rr_validated":     bool,         // did Python R:R validator run?
  "outcome": {
    "result":                "win" | "loss" | "expired" | "no_fill" | "no_trade",
    "candles_to_fill":       int | null,
    "candles_to_resolution": int | null,
    "fill_price":            float | null,
    "exit_price":            float | null,
    "actual_rr":             float | null   // realized R, negative for loss
  }
}

CALIBRATION REPORT SCHEMA (workspace/calibration_report.json):
{
  "period":           {"start": str, "end": str},
  "total_1m_candles": int,
  "gate_blocks": {
    "no_killzone":    int,
    "no_htf_bias":    int,
    "no_dol":         int,
    "cooldown_active": int
  },
  "soft_triggers": {
    "active_fvg":     int,
    "ifvg":           int,
    "sweep":          int,
    "displacement":   int
  },
  "fires_by_killzone": {"london_kz": int, "ny_am_kz": int, "ny_pm_kz": int},
  "fires_by_month":    {"2024-01": int, ...},
  "total_fires":       int,
  "estimated_llm_cost_usd": float
}

DATA LOADER:
  DataLoader supports three backends (configured via env var KYROS_DATA_BACKEND):
    "yfinance"  — downloads /NQ=F in 7-day 1m chunks, stitches, caches to parquet
                  Use for dev/recent data testing only
    "alpaca"    — uses Alpaca Markets API (ALPACA_API_KEY, ALPACA_SECRET_KEY env vars)
                  for 1m NQ futures data up to 5 years back
                  Use for full historical backtest
    "csv"       — local 1m CSV exported offline by an external tool (path via
                  KYROS_CSV_PATH, default workspace/data/nq_1min_data.csv).
                  Columns: date (UTC), open, high, low, close, volume; extra columns
                  (e.g. contract) are ignored. Normalizes date→timestamp (UTC),
                  filters to [start, end], caches to the same parquet format.
                  No network access — never imports a broker/IBKR client.
  BacktestEngine and TriggerCalibrator consume cached parquet — never re-download

  Active for this phase: KYROS_DATA_BACKEND=csv → workspace/data/nq_1min_data.csv
  (1m bars, 2024-06-09 → present, exported offline from TWS). Base timeframe stays
  1m — ReplayCandleSource in workspace/trading/ is used as-is (READ-ONLY, no edits).

PERFORMANCE REPORT must include:
  - LLM contamination disclaimer: "Backtest results may be optimistically biased.
    The LLM's training data includes market commentary from the backtest period.
    Treat metrics as directional signal for comparing system versions, not as
    prediction of live performance."
  - System prompt hash (first 8 chars of sha256 of ICT_SYSTEM_PROMPT) for
    version tracking — allows comparison across prompt iterations

BACKTESTING VALIDITY:
  - BacktestEngine must pass the same cooldown logic as production TradingLoop
    (tiered CooldownState from workspace/trading/cooldown.py)
  - No look-ahead: ReplayCandleSource in workspace/trading/candle_source.py
    already enforces this — BacktestEngine must not bypass it
  - resume_from argument: BacktestEngine reads existing trade_traces.jsonl
    and skips already-processed timestamps on restart
"""


def main() -> None:
    orch = Orchestrator()
    try:
        result = orch.run(
            problem_statement=PROBLEM_STATEMENT,
            end_goal=END_GOAL,
            constraints=CONSTRAINTS,
        )
        print(f"\nPhase 3A complete in {result.rounds_taken} round(s).")
        print(f"Total tokens : {result.total_tokens:,}")
        print(f"Blueprint    : {result.blueprint_path}")
    except EscalationRequired as e:
        print(f"\nNeeds human review: {e.review_path}")
        raise


if __name__ == "__main__":
    main()