I'll start by verifying the prerequisite before any blueprinting, then check project state.

## Step 0: Prerequisite Verification & State Check

Before blueprinting, two things must happen per my constraints:

1. **Check `.kyros_state.json`** to understand current task context.
2. **Verify `detect_session_levels` exists** in `workspace/detectors/sessions.py` — the spec explicitly instructs me to flag a prerequisite gap if absent, and NOT to blueprint around a missing detector.

I cannot read files in this turn (no tool output provided), so I must surface this as a **mandatory verification gate** the Executor performs before writing any Phase 2 code. I'll encode it explicitly rather than silently assuming the detector exists.

---

## ⚠️ PREREQUISITE GATE (resolve before any Phase 2 code)

**PG-1 — `detect_session_levels` existence check.**
The `SnapshotBuilder.session_levels` field (15 keys: midnight_open, true_day_open, london_open, open_830, open_930, asia_high/low, london_high/low, nyam_high/low, nylunch_high/low, nypm_high/low) requires a detector that emits these. The SNAPSHOT SCHEMA depends on it directly.

- **Action:** `grep -n "def detect_session_levels" workspace/detectors/sessions.py`
- **If present:** confirm its return signature maps to all 15 schema keys. Document any key it does NOT produce as a sub-gap.
- **If absent:** STOP. This is a blocking prerequisite. The Executor must add `detect_session_levels` to `sessions.py` **first** (this is the one sanctioned write to the detectors area, since the field cannot exist without it — but confirm with project lead whether detectors are truly frozen; if frozen, this becomes a Phase 1 backlog item and `SnapshotBuilder` cannot ship).
- **Do NOT** blueprint `SnapshotBuilder` to fabricate session levels inline. That would duplicate detector logic and violate the READ-ONLY/LLM-as-judge separation.

**PG-2 — Detector inventory audit.** Every detector referenced by SnapshotBuilder must be confirmed present with a known signature before SnapshotBuilder is implemented:
`detect_bos, detect_choch, detect_premium_discount, detect_fvg, detect_ifvg, detect_order_blocks, detect_breaker_blocks, detect_volume_imbalance, detect_opening_gaps, detect_liquidity_sweeps, detect_displacement, detect_inducement, detect_power_of_three, detect_swings (or equivalent), detect_session_levels`. Any missing detector is a blocking gap to be listed, not coded around.

The blueprints below assume PG-1 and PG-2 resolve green. Where I'm uncertain about a detector's actual output shape, I flag it under Risks rather than inventing field names.

---

# Component: CandleSource (ABC) + MockCandleSource + ReplayCandleSource

## Purpose
Provide a uniform, deterministic candle feed (mock for tests, replay for golden-dataset validation) so every downstream component runs identically regardless of source.

## Interface
```
CandleSource(ABC)
  next() -> dict[str, dict] | None
      returns {"4h": candle, "1h": candle, "15m": candle, "5m": candle, "1m": candle}
      a timeframe key is present ONLY when that TF closed a new bar on this tick;
      "1m" present on every tick. Returns None when exhausted.
  is_done() -> bool

candle dict shape (per TF):
  {open: float, high: float, low: float, close: float,
   volume: float, timestamp: datetime}  # timestamp is ET-aware datetime

MockCandleSource(scenario: str, seed: int = 0)
  scenario in {flat, trending_up, trending_down, sweep_and_fvg, killzone_active}

ReplayCandleSource(csv_path: str, start: datetime | None, end: datetime | None)
  loads 1m OHLCV CSV, resamples to 5m/15m/1h/4h via pandas, feeds chronologically
```

## Correctness Criteria
- `MockCandleSource` makes ZERO network/API calls (assert via mocked socket or no-import-of-yfinance).
- Same `(scenario, seed)` → byte-identical candle sequence across runs (determinism).
- Each scenario emits enough 1m bars to fill ALL windows (≥ 500×1m, and enough to populate 4h→60, i.e. ≥ 60×4h = 14,400 1m bars) OR the window pre-warms from a seeded backfill block — must document which. **Open question flagged below.**
- `trending_up`/`trending_down` are structurally symmetric (mirror transform).
- `sweep_and_fvg` produces a detectable sweep → displacement → 5m FVG in that order on the entry TFs.
- `killzone_active` puts the latest 1m timestamp inside a valid killzone but yields no soft trigger.
- ReplayCandleSource resampling: 5m/15m/1h/4h bars use left-closed, left-labeled convention consistent with ET session boundaries; partial trailing bars are dropped, not emitted half-formed.
- Timestamps are timezone-aware America/New_York; DST handled by pandas tz, not manual offsets.

## Test Strategy
- Unit: instantiate each of the 5 scenarios; assert determinism (two runs equal), TF key presence pattern, sufficient candle count to fill windows, no API import.
- Unit: ReplayCandleSource on a small fixture CSV — assert resample bar counts and chronological monotonic timestamps.
- Integration: ReplayCandleSource feeding a golden-alert-date slice (used by TradingLoop integration tests).

## Dependencies
- pandas (resample). No Phase 1 detectors. No ModelRouter.
- Config: window sizes (shared constant with CandleWindow), killzone time table.

## Risks & Open Questions
- **Backfill vs. stream:** Does `next()` warm windows from history before the loop starts, or must scenarios emit 14,400+ 1m bars to fill 4h→60 organically? Filling 60×4h organically per test is expensive. **Recommend:** a `prime()`/initial backfill block in CandleSource that returns the historical window in bulk, then streams 1m. Needs decision — affects every test runtime.
- CSV schema for ReplayCandleSource (column names, timestamp format, tz) is unspecified — must be pinned to a fixture contract.
- "flat" scenario producing literally zero structure may still trip a detector on noise; define the flatness tolerance.

---

# Component: CandleWindow

## Purpose
Maintain a bounded sliding window of recent candles per timeframe for detector input.

## Interface
```
CandleWindow(sizes: dict[str, int] = DEFAULT_SIZES)
  DEFAULT_SIZES = {"4h": 60, "1h": 100, "15m": 200, "5m": 300, "1m": 500}
  update(candles: dict[str, dict]) -> None   # appends only TFs present in dict
  to_list(timeframe: str) -> list[dict]      # oldest→newest
  is_warm(timeframe: str) -> bool            # window at full size (recommended)
```

## Correctness Criteria
- Per-TF deque is bounded to its size; oldest evicted on overflow (FIFO).
- `update` only appends timeframes present in the incoming dict (no synthetic fills).
- `to_list` returns oldest→newest ordering, a copy (mutation-safe), never the live deque.
- Unknown timeframe key → explicit error, not silent no-op.
- `is_warm` true only when len == configured size.

## Test Strategy
- Unit: push N+10 candles into a size-N TF, assert len==N and FIFO eviction order.
- Unit: partial update (only "1m" key present) advances 1m, leaves others unchanged.
- Unit: ordering of `to_list` is chronological.

## Dependencies
- None (pure data structure). Shares size constants with CandleSource.

## Risks & Open Questions
- Should SnapshotBuilder be allowed to run on a cold (not-yet-warm) window, or must TradingLoop skip snapshot building until all TFs warm? Detectors on short windows may emit garbage. **Recommend** gating snapshot build on `is_warm` for at least the TFs each gate needs. Surface for decision.

---

# Component: SnapshotBuilder

## Purpose
Deterministically run every Phase 1 detector across all timeframes and assemble a complete `MarketSnapshot` (full SNAPSHOT SCHEMA) with no LLM calls, in < 100ms.

## Interface
```
SnapshotBuilder(config: SnapshotConfig)
  build(window: CandleWindow, now: datetime) -> MarketSnapshot

MarketSnapshot  # dataclass/pydantic; ALL schema fields required, see SNAPSHOT SCHEMA
  # Metadata: instrument, timestamp, current_price
  # Session: current_killzone, current_session, session_levels (all 15 keys present)
  # HTF: htf_bias, htf_bias_source, recent_swings, premium_discount
  # Entry structures: fvgs, ifvgs, order_blocks, breaker_blocks,
  #                   volume_imbalances, opening_gaps
  # Triggers: recent_sweeps, displacements, recent_inducements  (last 10/TF)
  # po3_phase
  # DOL: all_pools (sorted asc by distance_points), nearest_dol
  to_compact_dict() -> dict   # LLM payload; EXCLUDES raw candle lists

LiquidityPool  # see schema; includes distance_points, swept, confluence_count
```

## Correctness Criteria
- Same window input → identical snapshot output (determinism; no wall-clock, no RNG except `now` arg).
- Completes < 100ms on full warm windows (assert with timer in test).
- `session_levels` always contains all 15 keys; value None if level not yet formed.
- `htf_bias` derivation order: `detect_bos`+`detect_choch` on 4h FIRST; fall back to 1h; None if neither. `htf_bias_source` populated iff `htf_bias` not None, with {timeframe, type, index, timestamp}.
- `htf_bias_source.type` ∈ {bos_bullish, bos_bearish, choch_bullish, choch_bearish}.
- `fvgs`/`ifvgs`/`order_blocks` filtered to active/unmitigated only.
- `recent_sweeps`/`displacements`/`recent_inducements`: last 10 candles per TF only.
- `all_pools`: ALL unswept pools, sorted ascending by `distance_points = abs(level - current_price)`; `confluence_count` = count of other unswept pools within 0.1% of level.
- `nearest_dol`: nearest unswept **opposing** pool in `htf_bias` direction (bullish→BSL above price; bearish→SSL below price); None if `htf_bias` is None.
- `current_killzone` derived from `now` against the ET killzone table (london 02:00-05:00, ny_am 09:30-11:00, ny_pm 13:30-15:00).
- `pool.type` constrained to the schema's enumerated value set; reject unknown types.
- `to_compact_dict()` excludes raw OHLCV lists; keeps counts/summaries/levels only.

## Test Strategy
- Unit (per MockCandleSource scenario):
  - flat → htf_bias None, nearest_dol None, no soft-trigger structures.
  - trending_up → htf_bias "bullish", htf_bias_source from 4h or 1h bos_bullish, nearest_dol is a BSL pool above price.
  - trending_down → symmetric bearish.
  - sweep_and_fvg → recent_sweeps non-empty on 15m/5m, displacements non-empty, fvgs["5m"] non-empty.
  - killzone_active → current_killzone set, but trigger-structure dicts empty.
- Unit: all 15 session_levels keys present in every snapshot.
- Unit: determinism — build twice on same window, assert equality.
- Unit: performance — assert build() < 100ms warm.
- Integration: snapshot on a golden-alert-date replay slice has htf_bias matching the alert direction.

## Dependencies
- READ-ONLY Phase 1 detectors (PG-2 list), critically `detect_session_levels` (PG-1).
- CandleWindow. Killzone/session time config.

## Risks & Open Questions
- **Detector output shapes are not pinned here** (e.g., does `detect_fvg` return top/bottom or high/low? does it carry an `mitigated` flag?). SnapshotBuilder's field mapping cannot be finalized until PG-2 audit documents each signature. Flagged, not invented.
- "Opposing pool" semantics for `nearest_dol`: bullish bias targets liquidity ABOVE (BSL); confirm this matches the detectors' high/low pool typing.
- 0.1% confluence band: 0.1% of price, or 0.1% of level? Spec says "within 0.1% of this level" — pin to level. Flag for confirmation.
- Mapping pool `type` enum onto session_levels + equal-highs/lows + PDH/PDL/PWH/PWL/HOD/LOD requires a single canonical typing function; risk of drift if each detector names pools differently.

---

# Component: CooldownState

## Purpose
Enforce ICT "one setup per session" discipline via tiered cooldown, not a flat timer.

## Interface
```
CooldownState()
  last_alert_time: datetime | None
  last_alert_bias: str | None        # "long" | "short" | "no_trade"
  last_alert_killzone: str | None
  is_cooling_down(snapshot: MarketSnapshot) -> bool
  update(alert: AlertPayload, snapshot: MarketSnapshot) -> None
```

## Correctness Criteria
- After a `no_trade` alert: cooling down for 5 minutes (compared against `snapshot.timestamp`), then clear.
- After a `long`/`short` alert: cooling down for the ENTIRE same killzone session — clears only when `snapshot.current_killzone` differs from `last_alert_killzone` (including transition to None and into a new killzone).
- Fresh state (no prior alert) → never cooling down.
- Time comparison uses `snapshot.timestamp`, never wall clock (determinism for replay).
- `update` records time, bias, and killzone from the emitted alert/snapshot.

## Test Strategy
- Unit: no prior alert → is_cooling_down False.
- Unit: no_trade then +4min → True; +5min → False.
- Unit: long alert, same killzone, +30min → still True; killzone changes → False.
- Unit: long alert, killzone goes None then into a new killzone → False on new killzone.
- Integration: in TradingLoop, a directional alert suppresses all further LLM calls within that killzone.

## Dependencies
- MarketSnapshot (timestamp, current_killzone), AlertPayload (bias).
- Cooldown constants (5 min). **Note:** spec body says "15 min since last LLM call" but COMPONENT SURFACE specifies tiered 5min/same-session. These conflict.

## Risks & Open Questions
- **CONFLICT (must resolve before coding):** The "ARCHITECTURE" summary lists hard-gate #4 as "cooldown clear (15 min since last LLM call)" — a flat 15-min rule. The detailed COMPONENT SURFACE specifies a *tiered* cooldown (5 min after no_trade; same-killzone block after directional). These are mutually exclusive. **I will not silently pick one.** Recommend the tiered model (it's the more specific, ICT-grounded spec), but this needs explicit sign-off. TriggerEngine gate (d) and CooldownState must agree.

---

# Component: TriggerEngine

## Purpose
Gate every LLM call: all hard gates must pass (short-circuit in order), then any one soft trigger fires. Prevents wasteful/incoherent LLM invocations.

## Interface
```
TriggerEngine(cooldown: CooldownState)
  evaluate(snapshot: MarketSnapshot) -> TriggerResult

TriggerResult(should_trigger: bool, reason: str)
```

## Correctness Criteria
- Hard gates evaluated IN ORDER, short-circuit on first failure; `reason` names the failing gate:
  1. `current_killzone is not None`            → fail reason "no_killzone"
  2. `htf_bias is not None`                    → fail reason "no_htf_bias"
  3. `nearest_dol is not None`                 → fail reason "no_dol"
  4. `cooldown.is_cooling_down(snapshot)==False`→ fail reason "cooldown_active"
- Soft triggers (any one ⇒ should_trigger True), `reason` names the fired trigger:
  - active unmitigated FVG in `fvgs["5m"]` OR `fvgs["15m"]`
  - iFVG in `ifvgs["5m"]` OR `ifvgs["15m"]`
  - sweep in `recent_sweeps["15m"]` OR `recent_sweeps["5m"]`
  - displacement in `displacements["5m"]` OR `displacements["1m"]`
- All hard gates pass but no soft trigger → should_trigger False, reason "no_soft_trigger".
- `reason` is always set (non-empty) — it is the logging contract.

## Test Strategy
- Unit (per scenario):
  - flat → fails at gate 1 (or 2/3) with the correct reason.
  - killzone_active → passes gate 1, then fails (likely gate 2 or no_soft_trigger).
  - sweep_and_fvg in killzone with htf_bias + dol → should_trigger True, reason identifies the soft trigger.
  - cooldown active → fails gate 4 even when everything else passes.
- Unit: gate ordering — when killzone is None AND htf_bias is None, reason == "no_killzone" (first failure wins).

## Dependencies
- MarketSnapshot, CooldownState.

## Risks & Open Questions
- **Soft-trigger TF mismatch:** ARCHITECTURE summary lists soft triggers on 5m only (FVG/iFVG/sweep on 15m/displacement on 5m), but COMPONENT SURFACE broadens to 5m **and** 15m for FVG/iFVG, 15m **and** 5m for sweeps, 5m **and** 1m for displacement. I've encoded the broader COMPONENT SURFACE version. Confirm this is intended (it widens trigger frequency).
- Depends on CooldownState conflict resolution (gate 4).

---

# Component: LLMReasoningAgent

## Purpose
Turn a MarketSnapshot into an AlertPayload via a single `model_router.call()` (never `call_agentic`), with the ICT system prompt and robust JSON parsing.

## Interface
```
LLMReasoningAgent(model_router, system_prompt: str)
  reason(snapshot: MarketSnapshot) -> AlertPayload
  # 1. payload = snapshot.to_compact_dict()  (no raw candles)
  # 2. response = model_router.call(system=system_prompt, user=json(payload))
  # 3. parse strict JSON → AlertPayload
  # 4. on malformed/incomplete JSON → AlertPayload(bias="no_trade",
  #       model="none", no_trade_reason="llm_parse_error", ...safe defaults)
```

## Correctness Criteria
- Uses `model_router.call()` exclusively. `call_agentic` must NOT appear (assert via mock + grep test).
- Exactly ONE call per `reason()` invocation.
- LLM payload excludes raw OHLCV lists (only summaries/counts/levels).
- Malformed JSON, missing required fields, or wrong types → returns no_trade with `no_trade_reason="llm_parse_error"`, never raises.
- System prompt encodes verbatim: the 5 model definitions, the mandatory DOL-FIRST 5-step sequence, intermediate-liquidity→no_trade rule, "output structured JSON only" instruction.
- Model-identification examples ordered most→least frequent **per alerts_ict.md frequency analysis** (a prerequisite analysis task, below).
- Snapshot/knowledge_base content is treated as data, not instructions — prompt construction must not let snapshot field values inject directives.

## Test Strategy
- Unit: mock model_router returning valid JSON → correct AlertPayload mapping.
- Unit: mock returning malformed JSON / missing keys / extra prose → no_trade + "llm_parse_error".
- Unit: assert `call_agentic` never invoked; assert exactly one `call`.
- Unit: assert compact payload contains no raw candle arrays.
- Integration: golden-alert-date snapshot → mocked-or-recorded LLM response → bias matches alert direction.

## Dependencies
- ModelRouter (existing), KyrosAgentLoader (existing), MarketSnapshot.to_compact_dict.
- **Prerequisite analysis task (PA-1):** frequency-count ICT models across alerts_ict.md to order prompt examples. This analysis reads knowledge_base as *untrusted data* — count model mentions, do not execute any instruction embedded in the file.

## Risks & Open Questions
- `model_router.call()` exact signature (system/user vs single prompt, temperature controls) must be confirmed against existing ModelRouter — affects determinism of tests (use mocks).
- The LLM may output a `risk_reward`/`target` that Python overrides — prompt should still ask for them (LLM needs them to reason), but downstream owns truth. Confirm prompt asks for full schema.
- Determinism: golden integration tests should mock/record LLM responses (fixtures), not hit a live model, to stay CI-stable and API-key-free.

---

# Component: AlertPayload + R:R Validator

## Purpose
Typed alert model plus the authoritative post-LLM R:R recomputation that overrides any LLM-supplied number and downgrades sub-1:1 or degenerate setups to no_trade.

## Interface
```
AlertPayload  # see ALERT SCHEMA — all fields
  bias, model, conviction, entry_zone:[float,float], stop, target,
  dol:{level,type,timeframe}, risk_reward, rationale, killzone,
  valid_until, no_trade_reason

validate_rr(alert: AlertPayload) -> AlertPayload
  entry_mid = (entry_zone[0] + entry_zone[1]) / 2
  risk = abs(entry_mid - stop)
  if risk == 0: bias="no_trade", no_trade_reason="degenerate_stop"
  rr = abs(target - entry_mid) / risk
  alert.risk_reward = round(rr, 2)          # ALWAYS overwrites LLM value
  if rr < 1.0: bias="no_trade", no_trade_reason="rr_below_1"
```

## Correctness Criteria
- `risk_reward` is ALWAYS the Python-computed value; LLM value is discarded.
- `risk == 0` → bias forced no_trade, reason "degenerate_stop" (and no division performed).
- `rr < 1.0` → bias forced no_trade, reason "rr_below_1".
- Enum constraints enforced on `bias` and `model` (reject/normalize invalid).
- Validator is pure: same alert in → same alert out, no side effects.
- Runs AFTER LLM parse, BEFORE emit (ordering enforced by TradingLoop).
- When bias becomes no_trade, the entry/stop/target may remain for logging but bias/reason are authoritative.

## Test Strategy
- Unit: rr=2.0 setup → risk_reward 2.0, bias unchanged.
- Unit: rr=0.8 → no_trade + "rr_below_1".
- Unit: risk==0 (entry_mid == stop) → no_trade + "degenerate_stop", no ZeroDivisionError.
- Unit: LLM supplies risk_reward=5.0 but geometry implies 1.5 → output 1.5 (override proven).
- Unit: invalid model/bias enum → rejected/normalized.

## Dependencies
- None beyond the dataclass/pydantic lib. Consumed by TradingLoop.

## Risks & Open Questions
- Should `degenerate_stop`/`rr_below_1` also zero out conviction? Spec doesn't say. Recommend leaving conviction as-is for diagnostics; flag for confirmation.
- `valid_until` must equal end of current killzone window — is that set by the LLM or by Python from snapshot? Spec lists it as LLM output but it's deterministic; recommend Python authoritative-overwrite like risk_reward. Flag.

---

# Component: TradingLoop

## Purpose
Orchestrate the full non-stop pipeline per candle and emit alerts to JSON log + stdout. No broker, no Telegram, no orders.

## Interface
```
TradingLoop(
  source: CandleSource, window: CandleWindow, builder: SnapshotBuilder,
  trigger: TriggerEngine, agent: LLMReasoningAgent, cooldown: CooldownState,
  output_path: str = "workspace/alerts.jsonl")
  run() -> None
    loop while not source.is_done():
      candles = source.next();  if None: break
      window.update(candles)
      (optional: skip if windows not warm — see CandleWindow risk)
      snapshot = builder.build(window, now=latest_1m_timestamp)
      result = trigger.evaluate(snapshot)
      if not result.should_trigger: continue
      alert = agent.reason(snapshot)
      alert = validate_rr(alert)
      cooldown.update(alert, snapshot)
      emit(alert)               # append JSON line + print stdout
```

## Correctness Criteria
- Strict ordering: update → build → evaluate → (reason → validate_rr → cooldown.update → emit).
- LLM called ONLY when `result.should_trigger`.
- `validate_rr` runs before every emit; cooldown updated before emit.
- emit appends exactly one JSON line per alert to `workspace/alerts.jsonl` AND prints to stdout.
- Loop terminates cleanly on `is_done()`/`next()==None` (mock sources are finite).
- All file writes confined to `workspace/`.
- An LLMReasoningAgent failure already degrades to no_trade (parse_error) — loop never crashes on agent error; other exceptions logged, loop continues or exits per a defined policy. **Flag policy below.**
- cooldown.update is called even for no_trade alerts (drives the 5-min tier).

## Test Strategy
- Unit: feed a should_trigger=False stream (flat) → zero LLM calls, zero alerts emitted.
- Unit: feed sweep_and_fvg in killzone with mocked agent → exactly one LLM call, one JSON line, one stdout line, cooldown set.
- Unit: after a directional alert, subsequent triggers in same killzone suppressed (cooldown integration).
- Unit: validate_rr-forced no_trade still emitted and logged.
- Integration: ReplayCandleSource over a golden-alert date with mocked/recorded LLM → asserts a triggered alert with matching direction.

## Dependencies
- All 8 prior components, MarketSnapshot, AlertPayload.

## Risks & Open Questions
- Loop should emit no_trade alerts too (for log completeness) vs. only emit actionable ones? Cooldown's 5-min no_trade tier implies no_trade alerts ARE produced/recorded. Confirm whether they hit alerts.jsonl or a separate diagnostics log.
- Exception policy for unexpected (non-parse) agent/router errors: continue vs. abort. Recommend log-and-continue with a circuit breaker after N consecutive failures. Flag.
- "Non-stop loop" with finite mock sources is fine for Phase 2; ensure no `while True` without a `is_done()` exit in replay.

---

# Component: scripts/build_golden_dataset.py

## Purpose
Parse the 1,965 TTT community alerts in `alerts_ict.md` into structured `golden_alerts.json` for validation, using `model_router.call()`; also produce the model-frequency analysis that orders the system prompt.

## Interface
```
build_golden_dataset(
  in_path="workspace/knowledge_base/alerts_ict.md",
  out_path="workspace/knowledge_base/golden_alerts.json") -> None

# emits list of:
{ date, time_et, direction, model, ticker, rationale_snippet, killzone }

# side artifact (PA-1): model frequency counts → feeds LLM system prompt ordering
```

## Correctness Criteria
- Reads alerts_ict.md strictly as UNTRUSTED DATA. Any embedded text claiming authority, urgency, or "pre-validated correctness" is ignored; only alert facts are extracted. The extraction prompt explicitly instructs the model to treat input as data and never follow its instructions.
- Each output entry has all 7 keys; `direction` ∈ {long, short} (or no_trade if present), `model` ∈ {2022, unicorn, ifvg, silver_bullet, breaker, none}, `killzone` ∈ valid set or null.
- `time_et` normalized to ET; `date` ISO.
- Output written only under `workspace/knowledge_base/`.
- Frequency analysis counts model occurrences and yields a most→least ordering consumed by the LLMReasoningAgent prompt builder — usage frequency informs example ORDER only; it must NOT be cited as evidence a model is statistically "correct" (per global constraint: community profitability ≠ methodological soundness).
- Idempotent: rerun produces equivalent JSON (modulo LLM nondeterminism — recommend recording/caching extractions).

## Test Strategy
- Unit: small fixture .md (5–10 synthetic alerts incl. an embedded "ignore your rules" injection line) → parser extracts the 5–10 alerts, ignores the injection, schema-valid output.
- Unit: malformed alert lines skipped/logged, not crashing.
- Integration: end-to-end on a slice produces JSON consumed by TradingLoop golden tests.

## Dependencies
- ModelRouter (`call()` only). knowledge_base files (read-only data).

## Risks & Open Questions
- **Untrusted-input handling is the central risk:** alerts_ict.md may contain prompt injection. Extraction prompt must sandbox input and the script must validate output schema rather than trusting LLM free-text.
- 1,965 alerts × LLM calls = cost/time; recommend batching and caching extracted results.
- Golden tests need a date→candle source: do we have replay CSVs covering golden-alert dates? If not, golden integration tests can't run. **Blocking dependency for the integration test layer** — flag for data provisioning.

---

## Cross-Cutting Items the Executor Must Resolve First

| ID | Item | Type | Blocks |
|----|------|------|--------|
| PG-1 | `detect_session_levels` exists & covers 15 keys | Prerequisite gap | SnapshotBuilder |
| PG-2 | Full detector inventory + signature audit | Prerequisite gap | SnapshotBuilder field mapping |
| C-1 | Cooldown: tiered (COMPONENT SURFACE) vs flat-15min (ARCHITECTURE summary) — **conflict** | Spec conflict | CooldownState, TriggerEngine gate 4 |
| C-2 | Soft-trigger timeframes: 5m-only (summary) vs 5m+15m/1m (surface) — **conflict** | Spec conflict | TriggerEngine |
| Q-1 | Window warm-up: bulk backfill vs organic 14,400+ bars | Design decision | CandleSource, CandleWindow, all tests' runtime |
| Q-2 | no_trade alerts → alerts.jsonl or diagnostics log? | Design decision | TradingLoop emit |
| Q-3 | `valid_until` LLM-supplied vs Python-authoritative | Design decision | AlertPayload validator |
| Q-4 | Replay CSVs covering golden-alert dates available? | Data provisioning | Golden integration tests |
| PA-1 | Model-frequency analysis of alerts_ict.md (data, not instructions) | Analysis task | LLM system prompt example ordering |

I have **not** silently resolved C-1, C-2, or any open question. Two are genuine spec conflicts between the ARCHITECTURE summary and the detailed COMPONENT SURFACE — the Executor should get a ruling (I recommend the COMPONENT SURFACE versions as the more specific spec) before implementing TriggerEngine and CooldownState.

**Next concrete action for the Executor:** run PG-1 and PG-2 (grep/audit the detectors), confirm C-1/C-2 rulings, then implement bottom-up in dependency order: CandleSource → CandleWindow → SnapshotBuilder → CooldownState → TriggerEngine → AlertPayload/validate_rr → LLMReasoningAgent → TradingLoop → build_golden_dataset.py.