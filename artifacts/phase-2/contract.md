# Phase 2 Contract — Agentic Reasoning Engine

## Round 2 (Evaluator BLOCK → fixed)

### Evaluator finding addressed

**[CRITICAL] Lookahead bias in ReplayCandleSource — higher-TF bars leak future data**
- File: `workspace/trading/candle_source.py` (`ReplayCandleSource.next`)
- Root cause: resampled bars are left-labeled at their OPEN time, but `next()`
  emitted a bar as soon as `now` (the 1m clock) reached the bar's *open*
  (`bars[cur]["timestamp"] <= now_ts`). The bar already carried its fully-closed
  OHLC, so at the first minute of a period the loop received the entire
  period's future high/low/close (e.g. a 1h bar emitted at 09:30 showed the
  09:00–09:59 high/close — 29 min of future price; a 4h bar leaks ~4h).
  Every detector (BOS/ChoCH→htf_bias, FVGs, sweeps, displacement, pools/DOL)
  and the LLM saw information that had not occurred at the decision timestamp.
- Scope: ReplayCandleSource only. MockCandleSource advances each TF one
  synthetic bar per tick and was never affected.

### Fix (localized, no rewrite of passing modules)

`workspace/trading/candle_source.py` — `ReplayCandleSource`:
1. Precompute each bar's close time (`bar_open + tf_duration`) at construction
   (`_close_dts[tf]`), alongside the existing open times (`_open_dts[tf]`).
   Datetimes are stored (not re-parsed ISO strings) so the hot-loop comparison
   is robust across DST offsets.
2. In `next()`, the 1m bar (decision granularity) is emitted at its own open
   timestamp — its OHLC is known because it just closed (standard
   process-closed-bars model). It defines `now`.
3. Higher timeframes (4h/1h/15m/5m) are emitted ONLY once they have fully
   closed: `now_dt >= close_dts[tf][cur]` (i.e. `now >= bar_open + tf_duration`).
   This guarantees a bar's fully-formed OHLC is never exposed before its
   period has elapsed — no future price leaks into the snapshot, detectors,
   or LLM.

No other module was modified. The fix is confined to the data source; the
snapshot/trigger/cooldown/alert/reasoning/trading_loop modules were already
correct (evaluator-verified PASS) and were left untouched.

### Regression tests added

`tests/phase2/test_candle_source.py` — two new tests:
- `test_replay_no_lookahead_higher_tf_emits_only_after_close`: feeds a
  contiguous monotonic 1m fixture and asserts for every emitted higher-TF
  bar that (1) `now_dt >= bar_open + tf_duration` (bar closed) and (2) the
  bar's high/low never exceed the running 1m extremes seen up to `now_dt`
  (no future price visible at the decision timestamp).
- `test_replay_first_1h_bar_emitted_at_close_not_open`: the exact failure
  mode the evaluator probed — the first 1h bar (open 09:00, covering
  09:00–09:59) must emit at 10:00 (its close), not at 09:30 (its open).

### Verification

- Independent probe (`/tmp/verify_no_lookahead.py`): PASS — all higher-TF
  bars emit only after closing; no bar's high/low exceeds running 1m extremes.
  4h bar (open 08:00) emits at 12:00; 1h (open 09:00) at 10:00; 15m at +15m;
  5m at +5m.
- `uv run pytest tests/phase2/` → 101 passed (was 99; +2 new regression tests).
- `uv run pytest tests/phase2/test_golden_integration.py` → 5 passed
  (directional replay still produces long/short alerts — delaying higher-TF
  bars does not prevent htf_bias from forming within the 1500-bar window).
- Full suite: 320 passed, 1 failed. The single failure
  (`tests/test_agent_loader.py::test_evaluator_prompt_includes_current_round_context`)
  is a PRE-EXISTING Phase 1 infrastructure issue (the evaluator prompt in
  `config/prompts.yaml` lacks "round 1 of 3" context). It is outside Phase 2
  scope (`workspace/` + `tests/phase2/`), was failing in the baseline before
  this change, and is unrelated to the trading pipeline.

---

## What already exists and passes (unchanged this round)

Phase 2 trading pipeline under `workspace/trading/` (all evaluator-verified
PASS in Round 1; not modified this round):

| Module | Purpose | Status |
|--------|---------|--------|
| `candle_source.py` | CandleSource ABC + MockCandleSource + ReplayCandleSource | **fixed this round** (lookahead) |
| `candle_window.py` | Bounded sliding window per TF | PASS (unchanged) |
| `snapshot.py` | SnapshotBuilder → MarketSnapshot (all detectors, all TFs) | PASS (unchanged) |
| `cooldown.py` | Tiered cooldown (5min no_trade / same-killzone directional) | PASS (unchanged) |
| `trigger.py` | TriggerEngine hard gates + soft triggers | PASS (unchanged) |
| `alert.py` | AlertPayload + Python-authoritative validate_rr | PASS (unchanged) |
| `reasoning_agent.py` | LLMReasoningAgent (model_router.call only, ICT prompt) | PASS (unchanged) |
| `trading_loop.py` | Orchestrator: update→build→evaluate→reason→validate→emit | PASS (unchanged) |

Phase 1 detectors under `workspace/detectors/` are READ-ONLY (verified:
`git diff --name-only HEAD -- workspace/detectors/` → empty). `detect_session_levels`
exists (sessions.py) and returns all 15 schema keys.

Tests under `tests/phase2/`: 101 passing
(test_candle_source, test_candle_window, test_snapshot, test_cooldown,
test_trigger, test_alert, test_reasoning_agent, test_trading_loop,
test_golden_integration).

`scripts/build_golden_dataset.py`: golden-dataset builder (mocked-router
tests pass; treats alerts_ict.md as untrusted data).

## Files changed this round

| File | Change | Reason |
|------|--------|--------|
| `workspace/trading/candle_source.py` | Emission gate: higher-TF bars emit only after close (`now >= bar_open + tf_duration`); precomputed close times | CRITICAL lookahead-bias fix (evaluator finding) |
| `tests/phase2/test_candle_source.py` | +2 regression tests for no-lookahead invariant | Prevent regression of the fix |

No other files were modified. No Phase 1 detector was touched.
