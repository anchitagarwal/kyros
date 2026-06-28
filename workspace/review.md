# Phase 3A Evaluation Review — Round 3 of 3 (FINAL)

Evaluator: independent grader (did not author this code).
Verdict basis: all invariants verified independently; `uv run pytest` executed
by the evaluator (not self-reported).

## Test Execution (independent)
```
ALPACA_API_KEY="" ANTHROPIC_API_KEY="" ZAI_API_KEY="" uv run pytest tests/phase3a/ -v
→ 67 passed in 9.43s
```
No test failed for a missing API key. DataLoader yfinance/alpaca paths are
mocked (yfinance not installed → proves no real HTTP). Engine tests mock the
LLM via `MagicMock(spec=LLMReasoningAgent)`. Invariant (h) satisfied.

---

## Invariant-by-invariant findings

### (a) Phase 2 / Phase 1 READ-ONLY — PASS
`git diff --name-only HEAD -- workspace/trading/ workspace/detectors/` → empty.
No tracked modifications to either tree. New Phase 3A code lives under
`workspace/backtesting/` and `tests/phase3a/` (untracked). No CRITICAL.

### (b) OutcomeSimulator lookahead safety — PASS (most critical)
- **Step 1**: `OutcomeSimulator.simulate()` does NOT internally filter
  `subsequent_candles` by timestamp — per spec the caller owns slicing
  (outcome.py docstring + body confirm). Acceptable per spec.
- **Step 2**: `BacktestEngine._slice_subsequent` (engine.py:~230) assembles
  subsequent candles with strict `if c_dt > alert_dt:` — the alert candle is
  EXCLUDED. The alert candle is not included. No CRITICAL.
- **Step 3 (manual probe)** — alert long, entry (100,101), stop 99, target 105,
  candles T50(alert)..T53:
  - Including alert candle `[T50,T51,T52,T53]` → result "win", ctf=1, ctr=3.
    Note: even when the alert candle is wrongly fed in, T50's high=105 does NOT
    produce an instant win because the fill candle never resolves (resolution
    begins on the candle AFTER fill). The leak only shifts timing, and the
    engine never feeds the alert candle anyway.
  - Correct `[T51,T52,T53]` → "win", ctf=1, ctr=2.
  Documented; the engine's strict `>` slice is the protective barrier.
- **Step 4 (ambiguous)**: single post-fill candle with high≥target AND low≤stop
  → result "loss", actual_rr -1.0. Conservative resolution confirmed.
- **Step 5 (no_trade short-circuit)**: `alert.bias=="no_trade"` →
  `TradeOutcome(result="no_trade")` with all numeric fields None, zero candle
  iteration. Confirmed by probe and code (outcome.py first branch).

### (c) TriggerCalibrator is LLM-free — PASS
`grep model_router|call_agentic|LLMReasoningAgent|anthropic|openai|reason(`
on calibrator.py → no hits. The calibrator takes `(window, builder, trigger,
cooldown)` and never constructs or calls an agent. It drives `TriggerEngine`
directly. `test_no_llm_calls` (LLM mock that raises if called) passes. No CRITICAL.

### (d) BacktestEngine uses Phase 2 CooldownState unchanged — PASS
Engine reads `cooldown = self.loop.cooldown` (engine.py:127) — the production
`CooldownState` instance owned by `TradingLoop`. Tests import `CooldownState`
from `trading.cooldown` and pass it into both `TriggerEngine` and
`TradingLoop`. No reimplementation. No HIGH.

### (e) Resume idempotent — PASS
Independent two-run probe (MockCandleSource sweep_and_fvg, n=100, mocked LLM):
- Run 1: 1 line written, 1 trace returned.
- Run 2 (same output file): file still 1 line, 1 trace returned.
- `Idempotent (file not doubled): True`; `No duplicate timestamps: True`.
`_load_existing` collects processed timestamps; new alerts at already-written
timestamps are skipped (LLM not re-called); cooldown is replayed from the
existing alert bias. No CRITICAL.

### (f) PerformanceReport arithmetic — PASS
Fixture (2 wins 2.0/1.5, 1 loss -1.0, 1 no_trade None) run through
`PerformanceReport`:
- win_rate = 0.667 ✓
- profit_factor = 3.5 ✓
- expectancy = 0.625 ✓
- max_drawdown_r = 1.0 ✓
All four exact. No HIGH.

### (g) LLM contamination disclaimer + prompt hash — PASS
`grep "optimistically biased"|"training data" workspace/backtest_report.md` →
present (line 52: "Results may be optimistically biased: the LLM may have seen
this period in its training data..."). System prompt hash present:
`**System prompt version:** ` + `b64522da` (first 8 hex of
`sha256(ICT_SYSTEM_PROMPT)`, imported from READ-ONLY `trading.reasoning_agent`).
`test_system_prompt_hash_matches_sha256` passes. No MEDIUM.

### (h) Offline test suite — PASS
All 67 tests pass with all three API keys empty. DataLoader tests mock the
download helper / use synthetic CSV (no real yfinance/Alpaca HTTP). Engine
tests mock TradingLoop's agent. No CRITICAL.

### (i) trade_traces.jsonl schema — PASS
No committed `workspace/trade_traces.jsonl`, so I generated one via an
integration probe and parsed the first line:
all required top-level fields present (trace_id, timestamp, instrument,
killzone, trigger_reason, snapshot_summary, raw_llm_output, alert,
rr_validated, outcome); outcome contains result, candles_to_fill,
candles_to_resolution, fill_price, exit_price, actual_rr. No HIGH.

### (j) No broker / live data / order placement — PASS
`grep -rni "ibkr|ib_insync|place_order|submit_order|create_order"
workspace/backtesting/` → only documentation/comment strings ("No broker...",
"never imports a broker/IBKR client"). No actual broker client import or order
call. The alpaca backend uses `requests.get` for a market-data bars endpoint
only (read-only, mocked in CI, never exercised this phase). No CRITICAL.

### (k) calibration_report.json keys — PASS
`workspace/calibration_report.json` parsed; keys present: total_1m_candles,
gate_blocks, soft_triggers, fires_by_killzone, fires_by_month, total_fires,
estimated_llm_cost_usd (+ period). Calibrator `_GATE_KEYS` /`_SOFT_KEYS` /
`_KZ_KEYS` exactly match the production `TriggerEngine` reason vocabulary and
gate order (verified against trigger.py: no_killzone → no_htf_bias → no_dol →
cooldown_active → no_soft_trigger; soft fvg/ifvg/sweep/displacement). Gate
blocks are mutually exclusive per candle (single short-circuit return). No HIGH.

---

## Round-1/Round-2 findings — confirmed addressed
- M1 (disclaimer LLM-contamination caveat): present.
- M2 (`backtest_report.md` artifact): committed.
- L1 (Alpaca env vars in `.env.example`): out-of-scope of this review's
  invariants; report.py/data_loader read from env, no hardcoded secrets.
- L2 (cache path backend collision): `_cache_path` includes `{backend}`.

## Documented deviations (acceptable, not defects)
1. **Expired exit_price/actual_rr = None** (outcome.py) rather than blueprint's
   "exit_price = close of last in-session candle". Contract D4 pins this
   explicitly; the report treats None as 0R for expectancy. Internally
   consistent and disclosed. LOW/none.
2. **OutcomeSimulator does not raise on `timestamp <= alert.timestamp`**
   (blueprint suggested a defensive in-component assertion). Per spec/contract
   D3 this is the caller's responsibility, and `BacktestEngine._slice_subsequent`
   enforces it with strict `>`. A defensive assertion would be belt-and-suspenders
   but its absence is explicitly spec-permitted ("If it does filter internally,
   that's fine too — flag as deviation, not a bug"). LOW.

---

## Review Summary

| Invariant | Result |
|-----------|--------|
| (a) Phase 1/2 READ-ONLY | PASS |
| (b) OutcomeSimulator lookahead safety | PASS |
| (c) TriggerCalibrator LLM-free | PASS |
| (d) CooldownState reused unchanged | PASS |
| (e) Resume idempotent | PASS |
| (f) PerformanceReport arithmetic | PASS |
| (g) Disclaimer + prompt hash | PASS |
| (h) Offline test suite (67 passed) | PASS |
| (i) trade_traces.jsonl schema | PASS |
| (j) No broker/live/order | PASS |
| (k) calibration_report.json keys | PASS |

No CRITICAL, HIGH, or MEDIUM findings. Two LOW deviations, both explicitly
spec/contract-sanctioned. All 67 tests pass offline with empty API keys.

VERDICT: APPROVE
