Before producing the blueprint, let me note what I can and cannot verify from the problem statement, and surface key ambiguities rather than silently resolving them.

I'm treating the Phase 2 interfaces (`TradingLoop`, `TriggerEngine`, `ReplayCandleSource`, `CooldownState`, `AlertPayload`, `model_router.call()`, `ICT_SYSTEM_PROMPT`) as fixed contracts I import but cannot inspect here. Where I assume a signature, I flag it as an open question for the Executor to verify against the real READ-ONLY modules.

---

# Component: DataLoader

## Purpose
Materialize 1m NQ historical data into a canonical parquet file, abstracting three backends (yfinance, alpaca, csv) behind one interface. Backtest/calibration consumers read only the parquet — never re-download.

## Interface
```
class DataLoader:
    @staticmethod
    def from_env() -> "DataLoader"
        # reads KYROS_DATA_BACKEND ∈ {"yfinance","alpaca","csv"}; default → error, not silent fallback

    def load(self, start: str, end: str) -> Path
        # start/end: ISO date strings (inclusive start, exclusive-or-inclusive end — see Risks)
        # returns Path to parquet conforming to CANONICAL SCHEMA below
```

CANONICAL PARQUET SCHEMA (all backends normalize to this):
```
timestamp : datetime64[ns, UTC]   # tz-aware, UTC, sorted ascending, unique
open      : float64
high      : float64
low       : float64
close     : float64
volume    : int64                 # 0 if backend lacks volume
```

Cache path: `workspace/data/cache/nq_1m_{start}_{end}_{backend}.parquet`
(backend in key so csv-derived and yfinance-derived caches never collide).

Backend contracts:
- `csv`: read `KYROS_CSV_PATH` (default `workspace/data/nq_1min_data.csv`); columns `date,open,high,low,close,volume`; drop extras (e.g. `contract`); rename `date`→`timestamp`; assert UTC; filter to `[start, end]`. **No network imports whatsoever.**
- `yfinance`: download `NQ=F` in ≤7-day 1m chunks, stitch, dedup overlaps; **hard-fail if requested range exceeds 7 days lookback** (do not silently truncate).
- `alpaca`: `ALPACA_API_KEY`/`ALPACA_SECRET_KEY` from env; 1m bars up to 5y; never hardcode keys.

## Correctness Criteria
- Output parquet is UTC tz-aware, strictly ascending by timestamp, zero duplicate timestamps.
- `csv` backend never imports yfinance/alpaca/IBKR/any broker client (verifiable by import-graph test).
- Second `load()` of same (start,end,backend) returns cached parquet without re-reading source (idempotent, observable via mtime / no network).
- `yfinance` raises on >7-day lookback rather than returning a short series.
- Rows outside `[start, end]` are absent from output.
- No `KYROS_DATA_BACKEND` → explicit error, not a default download.

## Test Strategy
- Unit (csv): synthetic CSV with `contract` extra column + out-of-range rows → assert schema, dropped column, range filter, UTC.
- Unit (csv): malformed CSV (missing column, non-monotonic timestamps, duplicate rows) → assert raises / dedups deterministically.
- Unit (import isolation): import `data_loader` with backend=csv under a sys.modules guard that fails if `yfinance`/`alpaca`/`ib_insync` is imported.
- Unit (cache): call `load()` twice, assert source read once.
- yfinance/alpaca: network paths **not** run in CI; mock the client and assert chunking/lookahead-limit logic only.

## Dependencies
- env: `KYROS_DATA_BACKEND`, `KYROS_CSV_PATH`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`.
- pandas/pyarrow. Update `.env.example` with the (empty-valued) Alpaca keys.

## Risks & Open Questions
- **`end` inclusive vs exclusive** is unspecified — must be fixed once and documented; affects last-candle/session-expiry edge cases in OutcomeSimulator.
- CSV `date` timezone: spec says UTC, but TWS exports are often exchange-local (ET). If the file is actually ET, every killzone classification is wrong. **Executor must verify the real file's tz, not assume.**
- Does the CSV represent regular-trading-hours only or 23h futures session? Gaps affect `candles_to_resolution` counting and session-end detection.
- "present" as an end bound is non-deterministic; backtests should pin an explicit end date for reproducibility.

---

# Component: TriggerCalibrator

## Purpose
Run `TriggerEngine` over the full replay period with **zero LLM calls** to map gate-block distribution, soft-trigger breakdown, and firing rate, so miscalibration (and API cost) is caught before any backtest spend.

## Interface
```
@dataclass(frozen=True)
class CalibrationReport:
    period: dict                  # {"start": str, "end": str}
    total_1m_candles: int
    gate_blocks: dict             # {"no_killzone","no_htf_bias","no_dol","cooldown_active": int}
    soft_triggers: dict           # {"active_fvg","ifvg","sweep","displacement": int}
    fires_by_killzone: dict       # {"london_kz","ny_am_kz","ny_pm_kz": int}
    fires_by_month: dict          # {"YYYY-MM": int}
    total_fires: int
    estimated_llm_cost_usd: float # total_fires * 0.003

class TriggerCalibrator:
    def __init__(self, trigger_engine, cooldown_state): ...
    def run(self, source: CandleSource) -> CalibrationReport
        # also writes workspace/calibration_report.json
```

## Correctness Criteria
- Makes **zero** calls to `model_router.call()` / any LLM (verifiable via mock that raises if called).
- `total_fires == sum(fires_by_killzone.values()) == sum(fires_by_month.values())`.
- `estimated_llm_cost_usd == round(total_fires * 0.003, …)` — fixed precision documented.
- Gate-block counts are **mutually exclusive per candle at the point a gate short-circuits** (a candle blocked at `no_killzone` is not also counted under `no_dol`); ordering must mirror production `TriggerEngine` gate order.
- Uses the **same** `CooldownState` semantics as production so `cooldown_active` counts match what BacktestEngine would experience.
- Iterates candles strictly through the `CandleSource` (no direct lookahead).

## Test Strategy
- Unit: MockCandleSource with hand-built candles → assert each gate-block category increments exactly when expected; assert sum invariants.
- Unit: candle sequence crafted to fire in each killzone → assert `fires_by_killzone` / `fires_by_month` bucketing (incl. month-boundary candle).
- Unit: LLM mock that raises on call → `run()` completes (proves no-LLM contract).
- Unit: cooldown scenario → consecutive eligible candles produce `cooldown_active` blocks matching production cooldown tier behavior.
- Integration: small real CSV slice via DataLoader → JSON written, schema-valid, invariants hold.

## Dependencies
- `workspace/trading/`: `TriggerEngine`, `CooldownState`, `CandleSource`/`ReplayCandleSource`, killzone definitions.
- DataLoader (parquet → CandleSource adapter).

## Risks & Open Questions
- The four `soft_triggers` keys and four `gate_blocks` keys must **exactly match** the real `TriggerEngine`'s internal categories. If Phase 2 has more/different triggers, the schema is lossy. **Executor must confirm against the real enum, not invent mappings.**
- Are gate blocks single-reason (first failing gate) or multi-reason per candle? Spec implies short-circuit; must be verified against real gate code.
- Cooldown depends on prior *fires*; calibrator has no LLM outcomes, so `cooldown_active` reflects fire-triggered cooldown only — confirm production cooldown isn't outcome-dependent in a way the no-LLM path can't reproduce.

---

# Component: OutcomeSimulator

## Purpose
Deterministically resolve an `AlertPayload` into win/loss/expired/no_fill/no_trade by walking strictly-subsequent candles, using the LLM's own entry_zone/stop/target without modification. The single most safety-critical component (lookahead invalidates everything).

## Interface
```
@dataclass(frozen=True)
class TradeOutcome:
    result: str                    # "win"|"loss"|"expired"|"no_fill"|"no_trade"
    candles_to_fill: Optional[int]
    candles_to_resolution: Optional[int]
    fill_price: Optional[float]
    exit_price: Optional[float]
    actual_rr: Optional[float]     # negative for loss

class OutcomeSimulator:
    def simulate(self,
                 alert: AlertPayload,
                 subsequent_candles: list[dict]) -> TradeOutcome
        # subsequent_candles: each {"timestamp","open","high","low","close",...}
        # PRECONDITION: every candle.timestamp > alert.timestamp
```

Resolution rules (as specified, restated precisely):
- **no_trade**: alert has no actionable setup (no entry_zone/stop/target) → all numeric fields None.
- **Fill (Step 1)**, scanning in order:
  - long fill when `low <= entry_zone[1] and high >= entry_zone[0]`
  - short fill when `high >= entry_zone[0] and low <= entry_zone[1]`
  - `fill_price = (entry_zone[0] + entry_zone[1]) / 2`; `candles_to_fill` = index+1 (1-based count from alert).
- **Resolution (Step 2)** begins on the candle **after** the fill candle (fill candle itself never resolves):
  - long: win if `high >= target`; loss if `low <= stop`
  - short: win if `low <= target`; loss if `high >= stop`
  - both in same candle → **loss** (conservative).
- **Expiry (Step 3)**:
  - session_end passes with no fill → `no_fill`
  - filled, neither stop nor target before session_end → `expired` (exit_price = close of last in-session candle; document this choice).
- `actual_rr`: win → `+|target-fill|/|fill-stop|`; loss → `-|fill-stop|/|fill-stop|` = `-1.0` only if exit==stop, else computed from exit; expired → realized `(exit-fill)/risk` signed by direction. **(see Risks — RR convention must be pinned.)**

## Correctness Criteria
- **CRITICAL**: if any candle in `subsequent_candles` has `timestamp <= alert.timestamp`, `simulate` raises (defensive assertion) — never silently uses it. This is enforced in-component even though the caller is responsible for slicing.
- Resolution never reads the fill candle's high/low for win/loss.
- Ambiguous same-candle stop+target → `loss`, deterministically.
- `candles_to_fill` / `candles_to_resolution` are None exactly when fill / resolution did not occur.
- Pure function of (alert, candles): no I/O, no clock, no randomness, no LLM.
- session_end source is the alert's killzone end, derived identically to production.

## Test Strategy
- Unit (lookahead guard): pass a candle with `timestamp == alert.timestamp` → raises.
- Unit (fill-candle-not-resolving): target sits inside the fill candle's range but resolution must wait for next candle → assert not an instant win.
- Unit (long win, long loss, short win, short loss): minimal 2-candle sequences.
- Unit (ambiguous): single post-fill candle straddling both stop and target → `loss`.
- Unit (no_fill): candles never enter entry_zone before session_end.
- Unit (expired): fills, then drifts sideways to session_end.
- Unit (no_trade): alert lacking entry/stop/target.
- Unit (rr signs): assert win RR>0, loss RR<0, expired RR sign matches drift.
- Property: shuffling does not apply (order matters) — instead, assert idempotence on repeated calls.

## Dependencies
- `workspace/trading/`: `AlertPayload`, killzone/session-end definitions.
- No DataLoader, no LLM, no network.

## Risks & Open Questions
- **`actual_rr` exact formula is underspecified.** Spec gives `-1.0` example for loss but loss exit could be a gap through stop (worse than -1R). Must decide: report nominal -1R or realized gap-fill R? Conservative-loss elsewhere argues for realized worse-than-1R, but that's an assumption. **Surfaced, not resolved.**
- **Gap fills**: if a candle opens beyond entry_zone (price gapped through), is fill at entry_mid (optimistic) or at open (realistic)? Spec mandates entry_mid — this is optimistic and should be noted in the report's bias disclaimer.
- **session_end with no subsequent candles** (alert at end of data): no_fill vs undefined? Need explicit handling when `subsequent_candles` is empty.
- Direction field name on `AlertPayload` ("long"/"short" vs "buy"/"sell") must be confirmed.
- entry_zone ordering ([low,high] vs [near,far]) must be confirmed against AlertPayload contract.

---

# Component: BacktestEngine

## Purpose
Drive `ReplayCandleSource` end-to-end through historical parquet, run the full production `TradingLoop` (with live LLM inference) per candle, attach an `OutcomeSimulator` result to each resulting alert, and append idempotently to `trade_traces.jsonl`.

## Interface
```
@dataclass(frozen=True)
class TradeTrace:
    trace_id: str
    timestamp: str
    instrument: str               # "NQ"
    killzone: str
    trigger_reason: str
    snapshot_summary: dict
    raw_llm_output: str
    alert: dict
    rr_validated: bool
    outcome: dict                 # TradeOutcome as dict

class BacktestEngine:
    def __init__(self,
                 trading_loop,                 # production TradingLoop
                 outcome_simulator: OutcomeSimulator,
                 traces_path: Path = workspace/trade_traces.jsonl): ...

    def run(self,
            source: CandleSource,
            resume_from: Optional[str] = None) -> list[TradeTrace]
        # appends one JSON line per trace; resumes past already-written timestamps
```

## Correctness Criteria
- **Resume/idempotent**: on restart, reads existing `trade_traces.jsonl`, skips candles at/below the last written alert timestamp; re-running a completed backtest appends nothing and produces no duplicate `trace_id` for the same alert timestamp.
- Uses production `CooldownState` (`workspace/trading/cooldown.py`) — same tiers as live; cooldown affects which candles fire identically to production.
- **No lookahead**: only ever advances via `ReplayCandleSource`; the OutcomeSimulator is fed candles strictly after the alert timestamp, sliced from already-replayed-or-future data **without** revealing future candles to the TradingLoop. (Outcome resolution may legitimately read future candles — the model has already produced its alert; this is the one allowed forward read, isolated to OutcomeSimulator.)
- Each fired alert → exactly one TradeTrace line; `snapshot_summary` contains **no raw candle arrays** (top-5 pools, latest swing per TF only).
- `raw_llm_output` is the verbatim `model_router.call()` string.
- LLM is the only nondeterministic dependency; everything else reproducible.

## Test Strategy
- Unit (resume): pre-seed `trade_traces.jsonl` with N lines → `run()` skips those timestamps, appends only new ones; assert no duplicate (timestamp).
- Unit (idempotent re-run): run twice on same data with mocked LLM → identical line count.
- Unit (cooldown parity): scenario where back-to-back fires would occur; assert cooldown suppresses the second exactly as production TradingLoop would (compare against direct TradingLoop call).
- Unit (snapshot hygiene): assert `snapshot_summary` has no key containing raw candle lists; ≤5 pools.
- Integration (MockCandleSource + mocked LLM + mocked OutcomeSimulator): full loop produces well-formed JSONL matching TRADE TRACE SCHEMA; validate every field present and typed.
- Integration (lookahead audit): instrument OutcomeSimulator mock to record the timestamps it received; assert all > alert.timestamp.

## Dependencies
- `workspace/trading/`: `TradingLoop`, `ReplayCandleSource`, `CooldownState`, `model_router`, `AlertPayload`, R:R validator, snapshot builder, `ICT_SYSTEM_PROMPT`.
- OutcomeSimulator, DataLoader.
- LLM API key at runtime (real backtest only); tests mock it — **no key required for tests**.

## Risks & Open Questions
- **How does BacktestEngine obtain "subsequent candles" for OutcomeSimulator without violating ReplayCandleSource's lookahead guard?** Two viable patterns: (a) two-pass — full parquet held separately for outcome resolution while ReplayCandleSource feeds the loop; (b) deferred resolution after replay completes. Each has different resume semantics. **Must be decided explicitly; (b) is cleaner for idempotency but delays outcomes.**
- **Resume granularity**: keyed on alert timestamp assumes ≤1 alert per timestamp/killzone. If multiple setups can fire on one candle, need a composite key (timestamp+trigger_reason). Confirm production can emit >1 alert per candle.
- Partial-write crash mid-line could corrupt JSONL; need atomic line append / validation-on-resume of last line.
- `rr_validated` semantics: does the validator run inside TradingLoop already, or must the engine invoke it? Confirm ownership.
- LLM cost/runtime for full multi-year replay is large; calibration gate (run TriggerCalibrator first) is mandatory, not optional.

---

# Component: PerformanceReport

## Purpose
Aggregate a list of `TradeTrace` into human-readable backtest metrics (overall, by model type, killzone, month, golden-dataset match) and write `workspace/backtest_report.md` with the mandated bias disclaimer and prompt-version hash.

## Interface
```
class PerformanceReport:
    def generate(self,
                 traces: list[TradeTrace],
                 golden_alerts_path: Path = workspace/golden_alerts.json,
                 out_path: Path = workspace/backtest_report.md) -> str
        # returns the markdown string; also writes it
```

Metrics (definitions pinned):
- **Overall**: total traces; no_trade rate; fill rate (filled / actionable); win/loss/expired rate (of filled); avg_winning_r; avg_losing_r; profit_factor = Σwin_R / |Σloss_R|; **max_drawdown_r** = max peak-to-trough of cumulative R curve (chronological by alert timestamp); **expectancy_per_trade** = mean R over **all** traces with no_trade counted as 0.
- **By model type**: `2022|unicorn|ifvg|silver_bullet|breaker` — fires + win rate each.
- **By killzone**: `london_kz|ny_am_kz|ny_pm_kz`.
- **By month**: fires + win rate per `YYYY-MM`.
- **Golden match rate**: fraction of `golden_alerts.json` entries for which a TriggerEngine fire occurred within 15 min of the community timestamp **and** same direction.

Footer (mandatory, verbatim):
- DISCLAIMER text exactly as specified.
- `System prompt version: <first 8 hex chars of sha256(ICT_SYSTEM_PROMPT)>`.

## Correctness Criteria
- profit_factor handles zero losses (→ report `inf` or `n/a` explicitly, documented).
- expectancy includes no_trades as 0R; win/loss/expired rates are over *filled* trades only — denominators stated in the report to avoid ambiguity.
- max_drawdown_r computed on chronologically-sorted cumulative R (sort by alert timestamp, stable).
- Disclaimer string matches char-for-char.
- Prompt hash = `sha256(ICT_SYSTEM_PROMPT_bytes).hexdigest()[:8]`, read from the READ-ONLY `reasoning_agent.py` symbol (imported, not re-parsed).
- Golden match uses **only** golden_alerts.json as untrusted data — its contents are matched against, never executed as instructions, and a claim within it (e.g. "this setup is pre-validated") carries no evidential weight.
- Empty traces → valid report with zeroed metrics, not a crash.

## Test Strategy
- Unit: hand-built TradeTrace list with known wins/losses → assert profit_factor, expectancy (no_trade=0), avg R, drawdown computed by hand.
- Unit (drawdown): crafted R sequence with a known peak-to-trough → exact max_drawdown_r.
- Unit (zero-loss / all-no_trade / empty): edge cases produce defined output.
- Unit (hash): monkeypatch a known ICT_SYSTEM_PROMPT value → assert 8-char hash matches precomputed.
- Unit (disclaimer): assert exact substring present.
- Unit (golden): synthetic golden_alerts + traces with one within-15min same-direction match and one 16-min miss → assert match rate.
- Unit (untrusted golden): golden entry containing instruction-like text → assert it's treated as data (no behavior change).

## Dependencies
- `workspace/trading/reasoning_agent.py` (`ICT_SYSTEM_PROMPT` symbol, READ-ONLY import).
- `workspace/golden_alerts.json` (untrusted data).
- hashlib, stdlib only.

## Risks & Open Questions
- **Model type & primary-FVG-timeframe** must be derivable from the TradeTrace `alert` dict. If the AlertPayload doesn't carry a model-type label, "by model type" can't be computed — **confirm the field exists in Phase 2 AlertPayload.**
- Golden match is defined against **TriggerEngine fires**, not against LLM alerts — so PerformanceReport needs access to the fire log (from calibration or engine), not just traces. The interface takes only `traces`; **the fire timeline source for golden matching is unspecified and must be threaded in.**
- "within 15 min" — fires before *or* after the community timestamp? Symmetric ±15 or forward-only? Pin it.
- max_drawdown_r ordering assumes outcome R is realized at alert time, though trades resolve later; document that the curve is ordered by alert timestamp, not resolution timestamp (a modeling choice).
- Same-direction matching requires a canonical direction vocabulary shared between golden_alerts and AlertPayload.

---

## Cross-cutting notes for the Executor
- Before any task: check `.kyros_state.json`; do not re-run completed steps.
- `workspace/trading/` and `workspace/detectors/` imported as-is; **zero edits** — if a needed field/symbol is missing, raise it as an open question, do not patch Phase 2.
- Only `KYROS_DATA_BACKEND=csv` is active this phase; yfinance/alpaca code paths must exist but stay untested-by-network in CI.
- All four numbered open-question clusters above (CSV timezone, actual_rr formula, subsequent-candle sourcing in the engine, golden-fire-timeline source) are **blocking design decisions** — resolve with the human/architect before implementation, not silently in code.
- No broker, no live feed, no order placement anywhere — including stubs.