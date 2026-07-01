# Phase 2B Evaluation — Fibonacci Levels & Liquidity-Cycle (ERL→IRL→ERL) / DOL

Round 2 of 3. Independent audit (not self-report). `uv run pytest` executed by the Evaluator.

## Scope of change reviewed
- Committed HEAD (`cbbb6c4`) — the Phase 2B feature landing.
- Uncommitted working-tree diff on `workspace/trading/snapshot.py` and
  `tests/phase2/test_fib_liquidity_cycle.py` — the round-1 repair (#8) + the
  promised scoring behavior tests. These constitute the round-2 delta the
  contract commits to and are evaluated as the state of the phase.

## Test run (offline)
`ANTHROPIC_API_KEY="" ZAI_API_KEY="" ALPACA_API_KEY="" uv run pytest`
→ **566 passed, 1 failed**.

The single failure is `tests/test_agent_loader.py::test_evaluator_prompt_includes_current_round_context`,
which hardcodes `"round 1 of 3"` in an assertion. It fails only because the
orchestrator round counter is now 2. This is an orchestration-harness fixture
outside the Phase 2B carve-out (`workspace/detectors/`, `workspace/trading/`,
`tests/test_fibonacci.py`, `tests/phase2/`); it is not caused by any Phase 2B
code and does not require a live key. Not attributable to the Executor's work,
not a Phase 2B regression. Phase 2B suites are fully green:
- `tests/test_fibonacci.py` + `tests/phase2/test_fib_liquidity_cycle.py`: 39 passed.
- `tests/phase2/test_snapshot.py` (pre-existing): 24 passed — no golden edited.

---

## Invariant verification

### a. Frozen-boundary carve-outs — PASS
`git diff HEAD~1 HEAD -- detectors/` = only NEW `fibonacci.py` + additive
import/`__all__` lines in `__init__.py`. No other detector touched.
`trading/` diff is additive: new snapshot fields, `LiquidityPool` scope/role/
clarity_score/score_breakdown, IRL pools, cycle + DOL methods, `_fib_dict`/
`_ranked_dol_dict`, compact keys, prompt, config knobs. `_nearest_dol` body is
byte-identical to the pre-2B version (signature reflowed to one line; logic
unchanged). No non-additive change to an existing default.

### b. detect_fibonacci pure & numerically EXACT — PASS
Independently constructed 100→200 both directions:
- up: equilibrium 150.0, golden_pocket [134.0,138.2], ote.primary 129.5,
  retracement_target 161.8, extensions {-0.5:250 … -2.5:450}. ✓
- down: golden_pocket [161.8,166.0], ote.primary 170.5,
  retracement_target 138.2, extensions {-0.5:50 … -2.5:-150}. ✓
Empty / single-candle / degenerate (high==low) → `[]`. ✓
No I/O, no globals, no pandas; reuses `detect_swings` read-only and mirrors the
premium_discount price(f) convention without importing its privates.

### c. ERL/IRL classification — PASS
FVG midpoint inside the dealing range → scope "internal", role "fvg_ce".
Equal highs / swings at/beyond the boundary → "external". Range-relative
refinement flips the source default when a level crosses a boundary
(verified: swings at 140/160 inside a 100–200 range refined to "internal").

### d. Cycle-state derivation — PASS
Latest `sweep_bsl`, no post-sweep reversal → last_swept_erl_side "buyside",
target_erl_side "sellside", current_leg "seek_irl". A reversal displacement/BOS
strictly after the sweep timestamp → "expand_to_erl". No sweep → None. Derived
fresh each build from `recent_sweeps`/`market_structure`/`displacements`; no
cross-snapshot state.

### e. / i. Cycle-aware DOL AUGMENTS via weighted ranking — PASS
(Invariant i supersedes e.) `_dol_target`:
- cycle None or `dol_use_cycle` False → `_nearest_dol` over the **ERL-only**
  set (repair #1). Verified: an internal pool nearer than an external one is
  correctly ignored in the fallback; `_nearest_dol` still byte-identical.
- cycle active → `_rank_dols(...)[0]`.
`_score_pool` pure (same args → identical (score, breakdown)). Direction filter
applied FIRST: a wrong-side pool (bsl below price on a buyside draw) is
structurally excluded (verified). Confluence/tf/role monotonic (verified rise).
Proximity is a low-weight tiebreak: a far Weekly/Daily equal (4h) beats a near
weak 1m swing (verified `ranked[0] is far`). `dol_target == ranked_dols[0]`;
`ranked_dols` sorted by clarity_score desc; `score_breakdown` itemized and sums
to the score. Double-build is deterministic.

### f. Snapshot integration + serializer shape — PASS
`build()` and `to_compact_dict()` both expose `fib_levels`, `liquidity_cycle`,
`dol_target`, `ranked_dols`, and scope/role-tagged pools. `_compact_dict` keeps
its shape: no raw candles, per-TF latest fib via `_fib_dict` (like
premium_discount). Snapshot built twice from the same window+config is
byte-identical (verified). `dol_target` is identically `ranked_dols[0]`.

### g. Reasoning prompt + schema — PASS
`ICT_SYSTEM_PROMPT` references `liquidity_cycle`, `dol_target`/`ranked_dols`,
golden pocket / OTE 0.705, 0.382 retracement_target and negative extensions.
The OUTPUT JSON schema keys are unchanged; `AlertPayload`/`parse_llm_json`
imports and call sites intact; `alert.py` untouched. Only prose/docstring lines
were removed from `reasoning_agent.py`.

### h. No regression / offline — PASS (with the harness-fixture note above)
No pre-existing assertion edited; all Phase 2B tests are NEW files. No test
needs a live key. `pytest` green except the round-counter harness fixture,
which is orthogonal to this phase.

### i. Weighted scoring — PASS (detailed under e./i.)

### j. Review repairs — PASS
- #1 ERL-only fallback: verified.
- #2 wrong-side filter first in `_rank_dols`: verified.
- #3 `ote["zone"]` from explicit ratios 0.62/0.79 — reordered/short `ote_grid`
  (`(0.705,)`) does NOT IndexError and yields the correct zone: verified.
- #4 role/scope reconciliation: scope is recomputed once against the HTF range;
  see LOW note below.
- #5 coincident ERL sources deduped on `(round(level,4), type)` keeping richest
  role by fixed priority: present.
- #6 opposing external in path lowers clarity_score via `clean_path` penalty;
  prompt no_trades only on an opposing EXTERNAL pool between entry and target
  (internal arrays are expected): verified in test + prompt text.
- #7 prompt reads `liquidity_cycle` FIRST, before direction selection: verified.
- #8 detectors not re-run inside `_build_pools`/`build`: the round-2 working-tree
  fix threads the RAW detector outputs (`swings_raw`/`fvgs_raw`/`obs_raw`) — the
  committed HEAD erroneously passed the serialized/truncated dicts
  (`recent_swings`/`fvgs`/`order_blocks`). The fix is correct and the call-count
  spy test pins it (`detect_swings`/`detect_fvg`/`detect_order_blocks` called 0×
  when precomputed supplied). Landed.

---

## Findings

### [LOW] Swing pool inside HTF range keeps role="swing" while scope="internal"
File: workspace/trading/snapshot.py:493 (scope refinement loop)
Issue: After range-relative refinement, a swing whose level falls inside the
dealing range is correctly re-scoped to "internal" but retains role="swing".
Under a strict reading of repair #4 ("no pool has role/scope that disagree"),
role and scope appear to disagree.
Assessment: This is the blueprint's *documented* precedence — "the range-derived
scope wins and the role is retained for weighting." role encodes the *source*
signal (a swing) as a scorer input; scope is the authoritative ERL/IRL tag. The
scorer's `cycle_align` reads `scope` (not `role`), so no wrong-side target or
mis-classification results. Behaviorally correct; not a defect.
Fix (optional): if a strict role/scope agreement is desired, demote role to a
neutral internal tag (e.g. "swing_ce") when refinement flips scope to internal,
purely for legibility.

### [INFO] Harness fixture `test_agent_loader` fails on round advance
File: tests/test_agent_loader.py:51
Issue: Assertion hardcodes "round 1 of 3"; now round 2. Outside the Phase 2B
carve-out and outside `workspace/`. Not a Phase 2B regression. No action for the
Executor within this phase.

---

## Review Summary

| Invariant | Result |
|-----------|--------|
| a. Frozen-boundary carve-outs | PASS |
| b. detect_fibonacci pure & exact | PASS |
| c. ERL/IRL classification | PASS |
| d. Cycle-state derivation (stateless) | PASS |
| e./i. Weighted cycle-aware DOL, ERL-only fallback | PASS |
| f. Snapshot + serializer shape + determinism | PASS |
| g. Prompt updated, schema/parser unchanged | PASS |
| h. No regression / offline | PASS (harness fixture note) |
| j. Review repairs #1–#8 | PASS |

No CRITICAL or HIGH findings. One LOW (documented, non-defect) and one
informational harness note. The round-1 HIGH (repair #8) and the MEDIUM
(missing scoring tests) are both resolved in the working tree.

VERDICT: APPROVE
