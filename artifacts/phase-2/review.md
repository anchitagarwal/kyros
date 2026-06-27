# Phase 2 Evaluation — Agentic Reasoning Engine

**Round:** 3 of 3 (final allowed round)
**Evaluator verdict:** APPROVE
**Method:** Independent audit. Did not accept Executor self-report; re-ran pytest,
re-derived each invariant, and independently probed R:R override, the tiered
cooldown, the four hard gates, the alert output, and the round-2 lookahead fix.

---

## Invariant verification

### (a) Phase 1 detectors READ-ONLY — PASS
- `git diff --name-only HEAD -- workspace/detectors/` → empty.
- `git status --short workspace/detectors/` → empty (no uncommitted changes).
- `detect_session_levels` was added in the committed Phase 1 follow-up
  (51fdfb6 "Add detect_session_levels"), not by Phase 2. Detector tree is frozen
  relative to HEAD. No modification.

### (b) detect_session_levels exists + wired into SnapshotBuilder — PASS
- `workspace/detectors/sessions.py:241` `def detect_session_levels(...)`.
- Wired in `workspace/trading/snapshot.py:53` (import), `:213` and `:381` (calls).
- Returns all 15 required keys: midnight_open, true_day_open, london_open,
  open_830, open_930, asia_high/low, london_high/low, nyam_high/low,
  nylunch_high/low, nypm_high/low (verified by reading the detector body and by
  `test_session_levels_all_15_keys_present`).

### (c) call_agentic banned from trading loop — PASS
- `grep -rn call_agentic workspace/trading/ --include=*.py` → single hit at
  `reasoning_agent.py:3`, inside a docstring that states "(NEVER `call_agentic`)".
  No invocation anywhere. The agent uses `model_router.call()` exclusively
  (`reasoning_agent.py` `reason()`), signature-compatible with
  `src/kyros/core/model_router.py:193 def call(agent_config, messages, ...)`.

### (d) R:R validator — active, Python-computed, overrides LLM — PASS
- `workspace/trading/alert.py validate_rr`: recomputes entry_mid from entry_zone,
  risk = |entry_mid - stop|, rr = |target - entry_mid| / risk, and ALWAYS
  overwrites `risk_reward` with the Python value.
- Independent probe (empty API keys):
  - LLM `risk_reward=5.0`, conviction=90, bias="long", geometry rr=0.5
    → output `bias="no_trade"`, `no_trade_reason="rr_below_1"`, `risk_reward=0.5`.
  - risk==0 (entry_mid==stop) → `bias="no_trade"`,
    `no_trade_reason="degenerate_stop"`, no ZeroDivisionError.
  - Valid rr=2.0 setup → bias unchanged, risk_reward=2.0.
- Called BEFORE emit in `trading_loop.py run()`:
  `alert = agent.reason(...) ; alert = validate_rr(alert) ; ... _emit(...)`.

### (e) TriggerEngine hard gates — each independently enforceable — PASS
- `trigger.py` evaluates the four hard gates in spec order with short-circuit:
  killzone → htf_bias → nearest_dol → cooldown; then soft triggers.
- Tests `test_gate_a/b/c/d` each set the other three gates valid and disable
  exactly one, asserting `should_trigger False` with the correct reason
  ("no_killzone" / "no_htf_bias" / "no_dol" / "cooldown_active").
- `test_gate_ordering_first_failure_wins`: killzone None AND htf_bias None →
  reason "no_killzone" (first failure). Hard-before-soft ordering correct.

### (f) Tiered cooldown — NOT a flat threshold — PASS
- `cooldown.py CooldownState.is_cooling_down` uses `snapshot.timestamp`
  (deterministic, not wall clock). Independent probe:
  - Tier 1 (no_trade): +4 min → cooling True; +5 min → False.
  - Tier 2 (directional, same killzone): +30 min → still True.
  - Tier 3 (directional, different killzone): +1 min → False (allowed).
  - Fresh state → False.
- Not a flat timer; killzone-aware for directional alerts. Matches spec.

### (g) MarketSnapshot completeness — PASS
- `snapshot.py` iterates all 5 TIMEFRAMES (4h,1h,15m,5m,1m); every detector dict
  (fvgs, ifvgs, order_blocks, breaker_blocks, volume_imbalances, opening_gaps,
  recent_sweeps, displacements, recent_inducements, po3_phase, recent_swings,
  premium_discount) gets a key for every TF (empty list when no candles).
  `test_fvgs_ifvgs_order_blocks_breaker_keys_all_timeframes` asserts the key set
  equals exactly TIMEFRAMES.
- `htf_bias_source` populated iff htf_bias not None, with timeframe/type/index/
  timestamp (`_derive_htf_bias`); 4h-first then 1h fallback.
- `session_levels` always 15 keys.
- `all_pools` sorted ascending by distance_points (`_build_pools` sorts; test
  `test_all_pools_sorted_ascending_by_distance`).
- `nearest_dol` respects bias: bullish → BSL above price, bearish → SSL below
  price (`_nearest_dol`; tests `test_nearest_dol_in_correct_bias_direction`,
  `test_nearest_dol_bearish_below_price`).
- `to_compact_dict` excludes raw OHLCV (test `test_compact_dict_excludes_raw_candles`;
  agent test `test_compact_payload_has_no_raw_candles`).

### (h) ICT system prompt — PASS
Read `ICT_SYSTEM_PROMPT` in `reasoning_agent.py`. Encodes:
- DOL-first 5-step sequence: enumerate pools → direction from htf_bias → select
  target → intermediate liquidity check → OTE modifier. ✓
- Intermediate liquidity check → no_trade ("intermediate liquidity in path"). ✓
- Unicorn OB↔FVG OVERLAP condition: "OB.top >= FVG.low AND OB.bottom <= FVG.high"
  (not mere co-presence). ✓
- 2022 requires all three: sweep + displacement + FVG from that displacement;
  explicitly states a standalone FVG is NOT a 2022 setup. ✓
- Silver Bullet time-gated to all three windows: 03:00-04:00 (London),
  10:00-11:00, 14:00-15:00 ET. London window present. ✓
- OTE conviction modifier: +15 when entry overlaps OTE band. ✓
- Output JSON only, no prose/markdown/preamble. ✓

### (i) Offline test suite — PASS
- `ANTHROPIC_API_KEY="" ZAI_API_KEY="" uv run pytest tests/phase2/` → 101 passed.
- No test fails for a missing API key. No test calls `model_router` unmocked;
  `LLMReasoningAgent` tests inject a MagicMock router returning fixture JSON and
  assert `router.call_agentic.call_count == 0` and exactly one `router.call`.

### (j) Alert output — PASS
- `trading_loop.py _emit` appends one `json.dumps` line to
  `workspace/alerts.jsonl` AND prints the same line to stdout.
- Independent run (sweep_and_fvg, cooldown disabled): 84 JSONL lines emitted,
  84 stdout lines, first 3 parse as valid JSON with all AlertPayload fields plus
  timestamp/instrument/current_price. Not DB/Telegram/stdout-only.

### (k) No broker, live data, or order placement — PASS
- `grep -rni 'ibkr|ib_insync|alpaca|polygon|place_order|submit_order|create_order|market_order'
  workspace/trading/ --include=*.py` → single hit in `__init__.py:7`, a docstring
  stating "No broker, no IBKR, no live market data, no order placement." No code.

### Round-2 lookahead fix (ReplayCandleSource) — independently re-verified PASS
- `candle_source.py` `ReplayCandleSource.next()`: the 1m bar defines `now` at its
  own (just-closed) open timestamp; higher TFs emit ONLY when
  `close_dts[cur] <= now_dt` (i.e. bar_open + tf_duration elapsed).
- Independent probe on a 240-minute monotonic fixture: 0 violations — every
  emitted higher-TF bar was fully closed by `now`, and no bar's high/low exceeded
  the running 1m extremes seen up to the decision timestamp. No future price leaks.

---

## Other observations (no action required)

- `contract.md` is present and describes the round-2 fix; the MEDIUM for a missing
  contract does NOT apply.
- No hardcoded secrets in Phase 2 code. `.env` appears only in scripts' docstring
  usage examples (`uv run --env-file .env ...`), reading from the environment —
  not writing `.env`, not embedding keys.
- `scripts/build_golden_dataset.py` treats `alerts_ict.md` strictly as untrusted
  data (module docstring + parameter docs); router is dependency-injected for
  mockable tests.

## Pre-existing failure outside Phase 2 scope (not a finding)

`tests/test_agent_loader.py::test_evaluator_prompt_includes_current_round_context`
fails (asserts "round 1 of 3" appears in the evaluator prompt template in
`config/prompts.yaml`). This test was introduced in the Phase 1 commit (426b8de),
the test file was never touched by Phase 2 work, and it concerns the orchestration
prompt template — not `workspace/trading/` or `tests/phase2/`. It is a genuine
pre-existing Phase 1 infrastructure issue, out of scope for this evaluation, and
does not affect the trading pipeline. Flagging for the Phase 1 backlog, not as a
Phase 2 blocker.

---

## Review Summary

| Severity | Count | Status |
|----------|-------|--------|
| CRITICAL | 0     | —      |
| HIGH     | 0     | —      |
| MEDIUM   | 0     | —      |
| LOW      | 0     | —      |

All Phase 2 invariants (a–k) verified independently and pass. The round-2
CRITICAL lookahead-bias fix is confirmed correct by independent probe. The
phase2 test suite runs fully offline (101 passed, no API key required, no
unmocked router). No CRITICAL or HIGH issues remain.

VERDICT: APPROVE
