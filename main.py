"""
main.py — Phase 2B: Fibonacci Levels & Liquidity-Cycle (ERL→IRL→ERL) / DOL

Run this to kick off the Planner → Executor → Evaluator cycle for Phase 2B.

Usage:
    uv run --env-file .env python main.py

Resets (move LLM artifacts to artifacts/ first per CLAUDE.md, do not rm):
    mv workspace/{blueprint,contract,review}.md artifacts/
"""

from kyros.core.orchestrator import Orchestrator, EscalationRequired

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2B TASK DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

PROBLEM_STATEMENT = """
Build Phase 2B for Project Kyros — an EXTENSION of Phase 2 (the Agentic Reasoning
Engine) that closes two gaps in the ICT model: (1) Fibonacci levels and (2) the
ERL→IRL→ERL liquidity cycle that drives Draw-on-Liquidity (DOL) selection. It
enriches the deterministic MarketSnapshot and the ICT reasoning prompt only. The
offline tuning of Phase 3B is OUT OF SCOPE and untouched.

Phases 1, 2, 3A and 3B are complete and validated.

GAP 1 — FIBONACCI. ICT entries and targets are fib-driven, but the only fib code
today is the hardcoded 0.5/0.62/0.79 inside workspace/detectors/premium_discount.py.
Missing: the golden pocket (0.618-0.66), the OTE grid (0.5/0.62/0.705/0.79 with
0.705 the primary entry), the 0.382 retracement target, and negative /
standard-deviation extensions (-0.5/-1/-1.5/-2/-2.5) used as expansion / DOL
targets. A new pure detector computes these, direction-aware, using the SAME fib
convention premium_discount already uses so the numbers agree.

GAP 2 — THE ERL→IRL→ERL LIQUIDITY CYCLE & DOL. Price sweeps External Range
Liquidity (ERL: swing highs/lows, BSL/SSL, equal highs/lows, prior day/week,
session H/L) → reverses → seeks Internal Range Liquidity (IRL: FVG CEs, order-block
CEs, dealing-range equilibrium, OTE) → expands to the OPPOSITE ERL. Today
LiquidityPool carries no external/internal tag and nearest_dol is just "nearest
unswept pool in bias direction" — the snapshot cannot express WHERE in the cycle
price is, which is the heart of ICT DOL selection. Phase 2B classifies each pool as
ERL/IRL, derives the cycle state from the most-recent sweep, and selects a
cycle-aware DOL target that AUGMENTS (never replaces) nearest_dol, combined with
HTF bias.
"""

END_GOAL = """
A tested extension to the Phase 2 reasoning layer that:

1. workspace/detectors/fibonacci.py — detect_fibonacci(): pure/stateless, mirrors
   detect_premium_discount's anchor + direction + edge handling; emits golden_pocket,
   the OTE grid (primary 0.705), the 0.382 target, and negative extensions. Exported
   from detectors/__init__.py. The ONLY new/edited detector file.

2. workspace/trading/snapshot.py — LiquidityPool gains scope ("external"|"internal")
   and role; new MarketSnapshot fields fib_levels, liquidity_cycle, dol_target;
   _build_pools adds IRL sources (FVG/OB CE, equilibrium, OTE) + scope refinement;
   _derive_liquidity_cycle() and _dol_target() (augmenting, not replacing,
   _nearest_dol); _compact_dict + a new _fib_dict mapper expose the new fields to the
   LLM without changing the serializer shape.

3. workspace/trading/reasoning_agent.py — ICT_SYSTEM_PROMPT uses liquidity_cycle,
   dol_target, golden-pocket/OTE entries, and 0.382 + negative-extension targets. The
   output JSON schema and AlertPayload/parser are UNCHANGED.

4. workspace/trading/config.py — new fib_* / irl_sources / dol_use_cycle knobs and a
   ("fib_levels",1) recency cap, all hashable and byte-preserving by default. PLUS the
   weighted-DOL knobs: dol_weights / tf_weights / role_weights (tuple-of-tuples,
   tunable later by Phase 3B) and ranked_dols_to_llm.

5. WEIGHTED "CLEAREST DOL" SCORING — the DOL is NOT chosen by proximity alone (the KB
   calls the nearest pool "Low Hanging Fruit", a first-TP waypoint, not the dominant
   draw). A pure SnapshotBuilder._score_pool produces a weighted clarity score per
   candidate from signals already on the pool/snapshot (timeframe, role, cycle
   alignment, htf bias, confluence, premium/discount, a clean-path penalty, and a
   low-weight proximity tiebreak); _rank_dols filters to the correct side, scores, and
   sorts. MarketSnapshot gains ranked_dols + per-pool clarity_score/score_breakdown;
   dol_target == ranked_dols[0]; the LLM sees the full ranking and may override only
   with a named higher-order signal. This rewrites _dol_target (deleting the first-match
   _nearest_internal/_nearest_external) and folds in the 8 Phase 2B code-review repairs.
   _nearest_dol stays byte-identical as the ERL-only fallback.

Full test suite runs offline (no API key): tests/test_fibonacci.py asserts exact fib
numbers; tests/phase2/ additions cover the ERL/IRL classifier, cycle derivation, DOL
selector, the weighted scorer (each factor monotonic; the core test that a farther
Weekly/Daily REL beats a nearer weak pool so proximity does NOT dominate), snapshot
integration, and a mocked-LLM end-to-end check. The entire pre-existing Phase 1/2/3A/3B
suite stays green except snapshot/pool goldens that gain scope/role/clarity_score/
ranked_dols.
"""

CONSTRAINTS = """
HARD CONSTRAINTS — this phase UNFREEZES earlier layers with NARROW carve-outs:
- workspace/detectors/ was READ-ONLY. The ONLY permitted change is ADDING
  fibonacci.py plus its single import/__all__ line in __init__.py. Modifying any
  other detector module is a critical scope violation.
- workspace/trading/ was SEMI-FROZEN. Only the ADDITIVE extensions above are
  permitted: new snapshot fields, LiquidityPool scope/role, IRL pool sources, the
  cycle + DOL methods, the _fib_dict mapper and _compact_dict keys, the ICT prompt,
  and the new TradingConfig knobs. Existing field defaults MUST stay byte-identical
  and _nearest_dol MUST be unchanged (the new _dol_target augments it). Any other
  change is a critical scope violation.
- The single intentional non-additive surface is LiquidityPool.to_dict gaining
  scope/role keys — update only the snapshot golden test that asserts it.
- workspace/backtesting/ and workspace/tuning/ are OFF-LIMITS.
- No broker, no IBKR, no live market data, no order placement.
- All tests run OFFLINE — no API key.

FIBONACCI — pure detector, same convention as premium_discount:
- Anchor on detect_swings (most recent confirmed swing-high + swing-low),
  direction-aware: up → price(f)=range_high-f*R; down → price(f)=range_low+f*R,
  R=range_high-range_low. Negative f = expansion beyond the origin extreme = targets.
- Edge cases: empty input, no swing pair, degenerate range (high==low) → [].
- Canonical numbers (100→200 range, R=100) the tests MUST assert:
    up:   equilibrium 150.0, golden_pocket [134.0,138.2], ote.primary 129.5,
          retracement_target 161.8, extensions {-0.5:250,-1:300,-1.5:350,-2:400,-2.5:450}
    down: golden_pocket [161.8,166.0], ote.primary 170.5, retracement_target 138.2,
          extensions {-0.5:50,-1:0,-1.5:-50,-2:-100,-2.5:-150}

ERL/IRL + CYCLE + DOL:
- ERL (scope="external"): equal highs/lows, prior day/week, session H/L, and the
  most-recent major swing high/low. IRL (scope="internal"): FVG midpoint (reuse
  detect_fvg's `midpoint` = CE), OB centre, dealing-range equilibrium, OTE 0.705.
- Scope refinement against the HTF dealing range: strictly inside → internal,
  at/beyond a boundary → external.
- Cycle (stateless, single-snapshot): from the most-recent sweep set
  last_swept_erl_side, target_erl_side (opposite), current_leg ("expand_to_erl" if a
  reversal displacement/BOS followed the sweep, else "seek_irl"), next_draw,
  agrees_with_htf_bias. No sweep → None.
- DOL: cycle None or dol_use_cycle False → fall back to nearest_dol exactly; seek_irl
  → nearest unswept INTERNAL pool in the reversal direction; expand_to_erl → nearest
  unswept EXTERNAL pool on target_erl_side beyond equilibrium; bias conflict → prefer
  htf_bias and mark agrees_with_htf_bias False.

REUSE (do not reimplement):
- detect_swings (anchor) and premium_discount's price(f) convention.
- detect_fvg's `midpoint` (CE) and detect_order_blocks (OB centre) for IRL pools.

VALIDITY:
- detect_fibonacci is pure: list[dict] in, list[dict] out, no I/O, no global state.
- Determinism: same window + same config → identical snapshot output.

WEIGHTED DOL + REVIEW REPAIRS (this continuation):
- _score_pool is pure (same args → same score/breakdown). _rank_dols filters candidates
  to the correct side of price FIRST (no wrong-side target), then scores, then sorts by
  clarity_score desc. Proximity is a low-weight tiebreak ONLY — it must not dominate.
- _dol_target = _rank_dols(...)[0] when the cycle is active, else the ERL-only
  _nearest_dol (which stays byte-identical). The bespoke first-match
  _nearest_internal/_nearest_external are deleted (subsumed by scoring).
- Fold in the 8 code-review repairs: (1) ERL-only _nearest_dol fallback; (2) wrong-side
  fixed by the direction filter; (3) fibonacci ote["zone"] from explicit ratios
  min/max(price(0.62),price(0.79)), NOT grid indices (a reordered/short fib_ote_grid
  must not crash); (4) derive scope once and reconcile role/scope; (5) dedup coincident
  ERL sources on (level,type); (6) intermediate-liquidity becomes the clean_path score
  factor (opposing ERL in the path lowers the score; the prompt still no_trades on an
  opposing ERL between entry and target); (7) ICT prompt reads the cycle before
  direction selection; (8) thread precomputed detect_swings/detect_fvg/
  detect_order_blocks into _build_pools and accept an optional swings arg on
  detect_fibonacci (no detector run 2-3x per TF).
- The OUTPUT JSON schema, AlertPayload and parser stay UNCHANGED.
"""


def main() -> None:
    orch = Orchestrator()
    try:
        result = orch.run(
            problem_statement=PROBLEM_STATEMENT,
            end_goal=END_GOAL,
            constraints=CONSTRAINTS,
        )
        print(f"\nPhase 2B complete in {result.rounds_taken} round(s).")
        print(f"Total tokens : {result.total_tokens:,}")
        print(f"Blueprint    : {result.blueprint_path}")
    except EscalationRequired as e:
        print(f"\nNeeds human review: {e.review_path}")
        raise


if __name__ == "__main__":
    main()
