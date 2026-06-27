# Phase 3A Contract — Backtesting & Evaluation Harness

## Scope

Phase 3A implements the offline backtesting/calibration/evaluation harness under
`workspace/backtesting/`. It imports `workspace/trading/` and
`workspace/detectors/` as READ-ONLY libraries (zero edits). No broker, no IBKR,
no live market data, no order placement — not even as a stub.

## Modules

| Module | Responsibility |
|--------|----------------|
| `data_loader.py` | Materialize 1m NQ data into a canonical UTC parquet (csv/yfinance/alpaca backends). |
| `calibrator.py` | Run `TriggerEngine` over replay with ZERO LLM calls; emit `calibration_report.json`. |
| `outcome.py` | Deterministically resolve an `AlertPayload` → win/loss/expired/no_fill/no_trade from strictly-subsequent candles. |
| `engine.py` | Drive `TradingLoop` over replay, attach outcomes, append idempotently to `trade_traces.jsonl`. |
| `report.py` | Aggregate `TradeTrace` list → `backtest_report.md` with metrics + disclaimer + prompt hash. |

## Pinned Design Decisions

### D1 — Range semantics (DataLoader)
`start`/`end` are ISO date strings (YYYY-MM-DD). The range is INCLUSIVE of both
bounds at date granularity (the full `end` day is included). Internally the
download loops treat end as exclusive (`end + 1 day`); `_normalize` filters to
`[start, end+1day)`.

### D2 — Cache key includes backend (DataLoader)
Cache path: `workspace/data/NQ_1m_{start}_{end}_{backend}.parquet`. The backend
segment is part of the cache key so a csv-derived cache and a yfinance/alpaca-
derived cache for the same date range never alias to the same file. (Evaluator
finding L2 — addressed.)

### D3 — OutcomeSimulator lookahead safety (CRITICAL)
`simulate()` does NOT internally filter `subsequent_candles` by timestamp; the
docstring states this is the caller's responsibility per spec. The
`BacktestEngine._slice_subsequent` assembles subsequent candles with strict
`if c_dt > alert_dt:` — the alert candle is in the replay buffer but excluded
by the strict greater-than. Resolution begins on the candle AFTER the fill
candle (the fill candle itself never resolves). Ambiguous same-candle
stop+target → loss (conservative). `no_trade` short-circuits with all-None
numeric fields.

### D4 — actual_rr formula
- Win: `(exit_price - fill_price) / abs(fill_price - stop)` for long;
  `(fill_price - exit_price) / abs(fill_price - stop)` for short → positive.
- Loss: negative of the same formula (realized, may be worse than -1R on a gap).
- no_fill / no_trade / expired: `actual_rr = None` (expired reports None per the
  engine's outcome dict; the report treats None as 0R for expectancy).
- Fill price: `entry_mid = (entry_zone[0] + entry_zone[1]) / 2` (optimistic for
  gap fills — disclosed in the report disclaimer).

### D5 — BacktestEngine subsequent-candle sourcing
The engine maintains a rolling replay buffer of recent candles per timeframe
(≥480 1m candles = 8 hours). When an alert fires, it slices candles strictly
after `alert.timestamp` from the buffer and passes them to OutcomeSimulator.
This is the ONE allowed forward read, isolated to OutcomeSimulator — the
TradingLoop itself only ever advances via ReplayCandleSource and never sees
future candles.

### D6 — Resume / idempotency (BacktestEngine)
On restart, `_load_existing` reads `trade_traces.jsonl`, collects processed
alert timestamps into a set, and skips candles whose alert timestamp is already
present. Malformed/partial lines are tolerated (JSONDecodeError skipped). A
completed backtest re-run appends nothing and produces no duplicate timestamps.

### D7 — CooldownState reuse
`BacktestEngine` reads `cooldown = self.loop.cooldown` — the production
`CooldownState` instance owned by `TradingLoop`. Not reimplemented. The resume
path reconstructs only `AlertPayload(bias=...)` to drive `cooldown.update`,
sufficient because `CooldownState` keys on `alert.bias` and
`snapshot.current_killzone`.

### D8 — Golden match (PerformanceReport)
For each directional golden entry (`direction ∈ {long, short}`), check if any
trace exists within ±15 min of the community timestamp AND same direction.
`golden_alerts.json` is UNTRUSTED data — matched only, never executed as
instructions. A claim within it (e.g. "pre-validated") carries no evidential
weight.

### D9 — Metric definitions (PerformanceReport)
- `profit_factor = sum(winning actual_rr) / abs(sum(losing actual_rr))`; zero
  losses → `"inf"`; no wins and no losses → `"n/a"`.
- `max_drawdown_r` = max peak-to-trough of cumulative R curve, ordered
  chronologically by alert timestamp; R=0 for no_trade/no_fill/expired.
- `expectancy` = mean(actual_rr) over ALL traces, None→0.
- win/loss/expired rates are over FILLED trades only (denominators stated in
  the report).

### D10 — Disclaimer (PerformanceReport)
The mandatory bias disclaimer discloses: SIMULATION on historical data; entry-
mid fill optimism; conservative same-candle loss; unmodeled slippage/
commissions; **LLM training-data contamination** ("Results may be optimistically
biased: the LLM may have seen this period in its training data, so historical
pattern recall cannot be distinguished from genuine edge."); untrusted golden
dataset. (Evaluator finding M1 — addressed.)

### D11 — System prompt hash
`sha256(ICT_SYSTEM_PROMPT).hexdigest()[:8]`, imported from the READ-ONLY
`trading.reasoning_agent` symbol. Rendered as `**System prompt version:** \`<hash>\``.

## Evaluator Round-1 Findings — Addressed

| ID | Severity | Finding | Resolution |
|----|----------|---------|------------|
| M1 | MEDIUM | Disclaimer omits LLM-training-data contamination caveat | Added contamination sentence to `_DISCLAIMER` in `report.py` (contains "optimistically biased" + "training data"). |
| M2 | MEDIUM | `backtest_report.md` not produced as artifact | Generated via `generate_sample_report.py`; `workspace/backtest_report.md` committed. |
| L1 | LOW | Alpaca env vars not in `.env.example` | Added `ALPACA_API_KEY=` / `ALPACA_SECRET_KEY=` (empty) to `.env.example`. |
| L2 | LOW | Cache path drops backend (collision risk) | `_cache_path` now includes `{backend}` segment; docstring updated. |

## Test Strategy
All tests offline. DataLoader uses synthetic CSV + `unittest.mock.patch`;
yfinance not installed (proves no real HTTP). Engine tests mock
`LLMReasoningAgent` via `MagicMock(spec=...)`. No API key required for any test.
`ALPACA_API_KEY="" ANTHROPIC_API_KEY="" ZAI_API_KEY="" uv run pytest tests/phase3a/`
→ 67 passed.

## Out-of-Scope / Open Questions (surfaced, not silently resolved)
- CSV `date` timezone: spec says UTC; the active csv backend parses as UTC. If a
  real TWS export is ET, killzone classification would be wrong — verify the
  real file's tz before a production backtest.
- `actual_rr` for a gap-through-stop loss could be worse than -1R; the simulator
  reports realized R (conservative), not a nominal -1R cap.
- Golden match is defined against trace alerts (LLM outputs), not raw
  TriggerEngine fires; the report's `generate()` takes only `traces`, so the
  fire-timeline source for golden matching is the trace list itself.
