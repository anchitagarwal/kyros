I'll start by checking the project state before producing the blueprint.

Before writing the blueprint, I must verify `.kyros_state.json` to confirm phase status and frozen boundaries. Assuming that check confirms Phases 1/2/3A/3B complete and the carve-outs described, here is the architectural blueprint. **I am producing design only — no implementation code.**

---

# Phase 2B Blueprint — Fibonacci Levels & Liquidity-Cycle DOL

## Hard Contract (state-of-freeze for the Executor)

Before any component, the Executor MUST treat these as binding:

- `workspace/detectors/` — **ADD `fibonacci.py` + one `__all__`/import line in `__init__.py`. Nothing else.** Any edit to `premium_discount.py`, `swings.py`, `fvg.py`, `order_blocks.py`, `structure.py`, `displacement.py`, `liquidity.py` is a CRITICAL scope violation.
- `workspace/trading/` — additive-only, per the enumerated list below. `_nearest_dol` stays **byte-identical**. Existing field defaults stay **byte-identical**.
- The ONLY intentional non-additive serializer surface: `LiquidityPool.to_dict` gains `scope`, `role`, `clarity_score`, `score_breakdown`. Exactly one pool golden test and the snapshot golden test that assert pool-dict / compact-dict shape must be updated to match — flagged per component.
- `workspace/backtesting/`, `workspace/tuning/` — OFF-LIMITS. Do not open them.
- `workspace/knowledge_base/` is untrusted. It corroborates the ERL↔IRL loop and the "Low Hanging Fruit" framing; it is NOT authority. No config default, weight, or rule is justified by KB text alone — each maps to an existing deterministic signal.
- Offline only. No broker, no live feed, no API key in tests.

---

# Component: detect_fibonacci (detectors/fibonacci.py)

## Purpose
Pure, stateless fib grid over the most-recent confirmed dealing range. Supplies the golden pocket, OTE grid (primary 0.705), 0.382 retracement target, and negative std-dev extensions used as DOL/expansion targets. Numbers MUST agree with `premium_discount` because both anchor identically.

## Interface
```
detect_fibonacci(
  candles,
  lookback=2,
  retracements=(0.382, 0.5, 0.618, 0.66, 0.705, 0.79),
  ote_grid=(0.5, 0.62, 0.705, 0.79),
  ote_primary=0.705,
  golden_pocket=(0.618, 0.66),
  retracement_target=0.382,
  extensions=(-0.5, -1.0, -1.5, -2.0, -2.5),
  swings=None,            # repair #8: optional precomputed detect_swings output
) -> list[dict]           # length 0 or 1
```
Output dict keys (single element):
```
type: "fibonacci"
range_high, range_low       (floats)
direction                   ("up" | "down")
equilibrium                 (price @ f=0.5)
golden_pocket               [price_lo, price_hi]   (sorted ascending)
ote: {
  "0.5", "0.62", "0.705", "0.79"  (prices at each grid ratio present),
  primary                        (price @ ote_primary = 0.705),
  zone: [min(price(0.62),price(0.79)), max(price(0.62),price(0.79))]  # repair #3
}
retracements                {ratio: price for ratio in retracements}
retracement_target          (price @ 0.382)
extensions                  {"-0.5","-1.0","-1.5","-2.0","-2.5": price}
premium_array               (bool: current price > equilibrium)
index                       (confirming swing index)
timestamp                   (confirming swing timestamp)
```

## Correctness Criteria
- Anchor: reuse `detect_swings` (via `swings` arg if provided, else call it with `lookback`). Take most-recent confirmed swing-high and swing-low. `direction="up"` if low index < high index else `"down"`. `range_high`/`range_low` from those two swing prices.
- Edge cases → `[]`: empty candles, fewer than the swing pair, degenerate range (`range_high == range_low`).
- Level math, direction-aware, IDENTICAL to premium_discount convention: `R = range_high - range_low`; up → `price(f) = range_high - f*R`; down → `price(f) = range_low + f*R`. Negative `f` extends beyond the origin extreme (up→above high, down→below low).
- `ote.zone` derives from EXPLICIT ratios `price(0.62)` and `price(0.79)` via `min`/`max` — NOT `ote_grid[1]`/`ote_grid[-1]` (repair #3). A reordered or shortened `fib_ote_grid` must not raise `IndexError` nor produce a wrong zone.
- Carry premium_discount's flagged dealing-range ambiguity note verbatim (same comment text/behavior when high/low ordering is ambiguous).
- `premium_array` uses the latest candle close as "current price".
- Pure: no I/O, no globals, no mutation of inputs.

## Test Strategy (tests/test_fibonacci.py)
Unit, using `MockCandleSource` producing a clean 100→200 range:
- **up** (low precedes high): equilibrium `150.0`; golden_pocket `[134.0, 138.2]`; ote.primary `129.5`; ote.zone `[121.0, 138.0]`; retracement_target `161.8`; extensions `{-0.5:250, -1:300, -1.5:350, -2:400, -2.5:450}`.
- **down** (high precedes low): equilibrium `150.0`; golden_pocket `[161.8, 166.0]`; ote.primary `170.5`; ote.zone `[162.0, 179.0]`; retracement_target `138.2`; extensions `{-0.5:50, -1:0, -1.5:-50, -2:-100, -2.5:-150}`.
- Edge: empty→`[]`; single candle→`[]`; degenerate flat range→`[]`.
- Repair #3 regression: pass `ote_grid=(0.705, 0.5)` (reordered, len 2) and `ote_grid=(0.705,)` → no crash, `zone` still from explicit 0.62/0.79.
- Purity: identical input twice → identical output; input list unmutated.
- `swings` passthrough: precomputed swings arg yields same result as internal call.

## Dependencies (REUSED)
`detect_swings` (anchor). Fib price(f) convention mirrored from `detect_premium_discount` (read-only reference — do not import its private helpers; replicate the documented formula).

## Risks & Open Questions
- Ambiguous swing ordering must resolve the same way premium_discount does — Executor must read that module to match, not to edit it.
- Float formatting: canonical numbers assume exact arithmetic on the 100→200 range; keep full precision internally, 2dp only at the `_fib_dict` serializer boundary.

---

# Component: LiquidityPool scope/role + IRL sources + scope refinement (snapshot.py `_build_pools`)

## Purpose
Tag every pool as ERL/IRL with a role, add the missing IRL pool sources, and reconcile scope against the HTF dealing range so the scorer never trusts a pool whose role and scope disagree.

## Interface
`LiquidityPool` gains (all defaulted → existing constructors stay valid):
```
scope: str = "external"
role: str = ""
clarity_score: float = 0.0        # component 9
score_breakdown: dict = {}        # component 9 (default-factory empty dict)
```
`to_dict()` emits `scope`, `role`, `clarity_score`, `score_breakdown` (**flag: pool golden test updated**).

ERL sources (scope="external"):
- equal highs/lows → role `"equal"`
- prior day/week → role `"prior"`
- session H/L/open → role `"session"`
- NEW: most-recent major swing high/low from `detect_swings` → role `"swing"` (canonical ERL)

IRL sources (scope="internal", gated by `config.irl_sources`):
- FVG midpoint (reuse `detect_fvg`'s `midpoint`) → role `"fvg_ce"`
- unmitigated OB centre `(top+bottom)/2` → role `"ob_ce"`
- dealing-range equilibrium (from fib) → role `"equilibrium"`
- fib OTE 0.705 → role `"ote"`

## Correctness Criteria
- **Repair #4 — derive scope ONCE, reconcile:** when an HTF dealing range exists (fibonacci on the first non-empty TF in `htf_tf_order`), a level strictly inside `(range_low, range_high)` → `internal`; at/beyond a boundary → `external`. Otherwise the source-type default applies. Scope is computed once per pool; role is then reconciled so no pool has `role="swing"` while `scope="internal"` contradicting the range test — the range-derived scope wins and the role is retained for weighting. Document the precedence explicitly.
- **Repair #5 — dedup:** coincident pre-existing ERL sources merge on `(level, type)`, keeping the RICHEST role (fixed priority: `equal > prior > session > swing`) and counting `confluence_count` on the merged set — no double-count, no inflated candidate list.
- **Repair #8 — thread precomputed detectors:** `_build_pools` accepts already-computed `swings`, `fvgs`, `order_blocks`, and the TF's `fibonacci` result; no detector runs 2–3× per TF on the same candles.
- Existing ERL construction and existing field defaults stay byte-identical; new sources are purely additive.

## Test Strategy (tests/phase2/)
- ERL/IRL classifier unit: fixture with equal highs, prior day, session, a swing, an FVG, an OB, plus fib → assert each pool's `scope`/`role`.
- Scope refinement: pool inside HTF range → internal even if source is a swing; pool at boundary → external.
- Repair #4: role/scope reconciliation — swing inside range does not emerge as `internal`+`swing` in a state the scorer treats as external.
- Repair #5: two coincident equal-high + prior pools at same `(level,type)` merge to one, richest role, correct confluence count.
- Repair #8: assert detectors invoked once per TF (spy/count on MockCandleSource-backed detector calls).
- `irl_sources` gating: dropping `"ote"` removes ote pools only.

## Dependencies (REUSED)
`detect_swings`, `detect_fvg` (`midpoint`), `detect_order_blocks`, and this phase's `detect_fibonacci`.

## Risks & Open Questions
- Which TF supplies the equilibrium/OTE IRL pools when multiple TFs have fib? Spec: HTF dealing range = first non-empty TF in `htf_tf_order`. Keep IRL equilibrium/ote sourced from that same range for consistency with scope refinement.

---

# Component: _derive_liquidity_cycle (snapshot.py, SnapshotBuilder)

## Purpose
Express WHERE in the ERL→IRL→ERL cycle price is, from the most-recent sweep. Stateless, single-snapshot heuristic — no cross-snapshot memory.

## Interface
```
_derive_liquidity_cycle(self, recent_sweeps, market_structure,
                        displacements, htf_bias) -> dict | None
```
Returns `None` if no sweep exists, else:
```
{
  last_swept_erl_side: "buyside" | "sellside",
  last_swept_level: float,
  last_swept_timestamp,
  current_leg: "seek_irl" | "expand_to_erl",
  next_draw: "irl" | "erl",
  target_erl_side: "sellside" | "buyside",   # opposite of last_swept
  agrees_with_htf_bias: bool,
}
```

## Correctness Criteria
- Most-recent sweep across timeframes → `last_swept_erl_side` = "buyside" if BSL swept, "sellside" if SSL swept; `target_erl_side` = opposite.
- `current_leg = "expand_to_erl"` iff a reversal-direction displacement OR BOS occurred strictly AFTER the sweep timestamp; else `"seek_irl"`.
- `next_draw = "irl"` when `seek_irl`, `"erl"` when `expand_to_erl`.
- `agrees_with_htf_bias` = does `target_erl_side` match `htf_bias` direction (buyside↔bullish, sellside↔bearish).
- Documented as stateless/derived-fresh; no persistence.

## Test Strategy
- Sweep-of-BSL, no subsequent displacement → seek_irl, target sellside, next_draw irl.
- Sweep-of-SSL followed by bullish BOS → expand_to_erl, target buyside, next_draw erl.
- No sweep → None.
- `agrees_with_htf_bias` true/false cases against both bias directions.

## Dependencies (REUSED)
Liquidity sweep detection, `detect_structure` (BOS), `detect_displacement` — all read-only, outputs consumed via the existing snapshot pipeline.

## Risks & Open Questions
- "After the sweep" ordering must use timestamps consistently across TFs. Tie: a displacement at the same timestamp as the sweep is NOT "after" → seek_irl. Document.

---

# Component: Weighted DOL scoring — _score_pool / _rank_dols / _dol_target (snapshot.py)

## Purpose
Choose the DOL by a deterministic weighted CLARITY score, not proximity. The nearest pool is a first-TP waypoint ("Low Hanging Fruit"), not the dominant draw. Expose a ranked list; `dol_target` = argmax.

## Interface
```
_score_pool(self, pool, *, htf_bias, cycle, htf_fib, current_price,
            killzone, pools) -> tuple[float, dict]     # (score, breakdown)

_rank_dols(self, pools, *, htf_bias, cycle, htf_fib, current_price,
           killzone) -> list[LiquidityPool]           # sorted clarity desc

_dol_target(self, pools, cycle, htf_bias, current_price) -> LiquidityPool | None
```

## Correctness Criteria
`_score_pool` is PURE (same args → same score+breakdown). Linear sum of weighted factor terms, each mapping to an EXISTING signal; `breakdown` itemizes every factor's contribution:
- `timeframe` : `pool.timeframe → config.tf_weights` (4h ≫ 1h ≫ 15m ≫ 5m ≫ 1m)
- `role` : `pool.role → config.role_weights` (equal > prior > session > swing > fvg_ce/ob_ce/equilibrium/ote)
- `cycle_align` : large + when pool matches `cycle` (expand_to_erl → external on `target_erl_side`; seek_irl → internal in reversal direction)
- `bias_align` : + when pool side agrees with `htf_bias` (bsl&bullish / ssl&bearish)
- `confluence` : `+ w * pool.confluence_count`
- `pd_align` : + when reaching the pool moves price across `htf_fib.equilibrium` toward the opposite side; penalize if already spent
- `clean_path` : `− w * (count of OPPOSING unswept EXTERNAL pools strictly between current_price and pool.level)` — the LRLR / intermediate-liquidity guard, numeric (repair #6)
- `proximity` : `+ small w * (distance_points / R)` — LOW weight, tiebreak only, so LHF cannot dominate
- `killzone` : `+ small w` if `current_killzone` set (optional)

`_rank_dols`:
- **Repair #2 — filter to correct side FIRST:** for a buyside draw keep pools above price; for a sellside draw keep pools below price. Structurally prevents a wrong-side target.
- Score each survivor, sort by `clarity_score` DESC with a **stable distance tiebreak**.

`_dol_target` (REWRITTEN):
- `cycle is None` or `config.dol_use_cycle is False` → fall back to `_nearest_dol` over the **ERL-only** pool set (repair #1). `_nearest_dol` stays byte-identical.
- else → `_rank_dols(...)[0]` or `None`.
- Bias conflict → prefer `htf_bias` and set `cycle.agrees_with_htf_bias = False`.
- **DELETE** the bespoke first-match `_nearest_internal` / `_nearest_external` (subsumed by scoring).

## Test Strategy
- Each factor monotonic in isolation (raise one input, hold others → score rises/falls as signed).
- **Core proximity test:** a farther Weekly/Daily ERL beats a nearer weak 5m pool — proximity does NOT dominate.
- `clean_path`: inserting an opposing unswept ERL between price and a candidate lowers its score below a clean candidate.
- Side filter: a wrong-side pool is never returned by `_rank_dols`/`_dol_target`.
- Fallback: `dol_use_cycle=False` reproduces `_nearest_dol` on the ERL-only set exactly.
- Determinism: same window+config → identical ranking and breakdowns.
- `breakdown` sums to `clarity_score`.

## Dependencies (REUSED)
Pool set from `_build_pools`; `htf_fib` from `detect_fibonacci`; cycle from `_derive_liquidity_cycle`; existing `_nearest_dol`.

## Risks & Open Questions
- Weight defaults must encode KB ORDERING without being justified BY the KB — each weight ties to a deterministic signal; Phase 3B tunes them later. Ship conservative defaults where cycle_align > tf > role > pd/bias > confluence > clean_path magnitude ≫ proximity.
- Ensure `R` for proximity normalization is well-defined when no HTF fib exists → use a documented fallback (e.g. current-TF range) or set proximity term to 0.

---

# Component: MarketSnapshot wiring + serializer (snapshot.py `build`, `_compact_dict`, mappers)

## Purpose
Populate and expose the new fields without changing serializer shape beyond the flagged additive keys.

## Interface
New `MarketSnapshot` fields:
```
fib_levels: dict[str, list[dict]]   # per-TF
liquidity_cycle: dict | None
dol_target: LiquidityPool | None
ranked_dols: list[LiquidityPool]
```
`build()` order per spec: fill `fib_levels[tf] = detect_fibonacci(...)` (threading precomputed swings), then pools (IRL + scope refinement), then `liquidity_cycle`, then `ranked_dols`, then `dol_target == ranked_dols[0]`.

New mappers:
- `_fib_dict` — per-TF latest, 2dp: `direction, equilibrium, golden_pocket, ote_primary, ote_zone, retracement_target, extensions, premium_array`.
- `_ranked_dol_dict` — carries `clarity_score` + `score_breakdown`.

`_compact_dict` gains `fib_levels`, `liquidity_cycle`, `dol_target`, `ranked_dols` (top-N via `config.ranked_dols_to_llm`). **Flag: snapshot golden test updated.**

## Correctness Criteria
- `dol_target` is identically `ranked_dols[0]` (or None).
- Serializer shape otherwise UNCHANGED; new keys additive.
- Determinism preserved.

## Test Strategy
- Snapshot integration: full build over MockCandleSource emits all new fields; goldens updated for scope/role/clarity_score/ranked_dols.
- `ranked_dols_to_llm` truncation respected in compact dict.
- `_fib_dict` 2dp rounding matches canonical numbers.

## Dependencies (REUSED)
All of the above components.

## Risks & Open Questions
- Golden churn: exactly the two flagged goldens (pool-dict, compact/snapshot) change. Any other golden diff signals an unintended non-additive change — treat as a defect.

---

# Component: ICT_SYSTEM_PROMPT (reasoning_agent.py)

## Purpose
Teach the reasoning layer to read the cycle before choosing direction/DOL, use the ranked DOLs, and apply golden-pocket/OTE entries and 0.382 + negative-extension targets. **Output JSON schema / AlertPayload / parser UNCHANGED.**

## Interface (prompt-only)
- **Repair #7 — reorder:** cycle read PRECEDES direction selection (remove the "1, 2, 1.5, 3" ordering). Read `liquidity_cycle` (last swept side, current_leg, target ERL) first.
- DOL step reads `ranked_dols` (already scored/sorted), defaults to top `clarity_score`, and MAY override only with a NAMED higher-order signal (SMT, macro/killzone) it must state.
- Briefly define each `score_breakdown` factor for the model.
- Intermediate-liquidity: only an unswept opposing EXTERNAL (ERL) pool between entry and target blocks the path → no_trade; internal arrays in the path are expected, not blockers (repair #6).
- OTE modifier: +15 conviction if entry in golden pocket (0.618–0.66); +10 more at 0.705 primary.
- Entry logic: premium arrays (above equilibrium) → short toward discount/sellside ERL; discount arrays (below) → long toward buyside ERL; entries at golden pocket / OTE 0.705; targets = 0.382 partial then negative extensions toward the opposite ERL.

## Correctness Criteria
- No change to the emitted JSON keys, AlertPayload, or parser.
- Prompt references only fields actually present in the compact dict.

## Test Strategy
- Mocked-LLM end-to-end: feed a snapshot with a defined cycle + ranked_dols; assert the (mocked) reasoning path selects `dol_target` unless it names an override, and the parser still produces a valid AlertPayload unchanged in schema.
- Regression: existing reasoning_agent parser tests stay green.

## Dependencies (REUSED)
Existing reasoning_agent scaffolding and parser (unchanged).

## Risks & Open Questions
- Prompt must not imply new output fields. Keep the schema section untouched; only the reasoning/instruction sections change.

---

# Component: TradingConfig knobs (config.py)

## Purpose
Add hashable, byte-preserving fib + IRL + weighting knobs; expose via `config_hash()`; add the recency cap. Tunable by Phase 3B later (which is otherwise untouched).

## Interface
Immutable/hashable, defaults byte-preserving:
```
fib_retracements = (0.382,0.5,0.618,0.66,0.705,0.79)
fib_golden_pocket = (0.618,0.66)
fib_ote_grid = (0.5,0.62,0.705,0.79)
fib_ote_primary = 0.705
fib_retracement_target = 0.382
fib_extensions = (-0.5,-1.0,-1.5,-2.0,-2.5)
fib_anchor_lookback = 2
irl_sources = ("fvg","order_block","equilibrium","ote")
dol_use_cycle = True
ranked_dols_to_llm = 5
```
Tuple-of-tuples mirroring `_DEFAULT_RECENCY_CAPS`, each with a `*_dict()` accessor:
```
_DEFAULT_DOL_WEIGHTS  (factor → weight)
_DEFAULT_TF_WEIGHTS   (timeframe → weight)   # 4h≫1h≫15m≫5m≫1m
_DEFAULT_ROLE_WEIGHTS (role → weight)        # equal>prior>session>swing>fvg_ce/ob_ce/equilibrium/ote
```
Add `("fib_levels", 1)` to `_DEFAULT_RECENCY_CAPS`. Add every new knob (and the three weight dicts, canonicalized) to the `config_hash()` canonical dict.

## Correctness Criteria
- Frozen/hashable; `config_hash()` deterministic and stable across runs.
- Existing default hash for the pre-existing knobs is preserved except for the deliberate additions (document that `config_hash()` value changes — this is expected and any consumer asserting a frozen hash string must update it).

## Test Strategy
- All new knobs present, defaulted, hashable.
- `config_hash()` deterministic; canonical dict includes new knobs + weight dicts.
- `*_dict()` accessors return correct mappings; weight ordering encodes the specified precedence.

## Dependencies (REUSED)
Existing config machinery (`_DEFAULT_RECENCY_CAPS` pattern, `config_hash`).

## Risks & Open Questions
- `config_hash()` string changes — confirm no Phase 3B artifact pins the old hash as an immutable contract; if it does, that pin lives in `tuning/` (OFF-LIMITS) and must not be edited — instead ensure Phase 3B consumes the hash dynamically. **Open question flagged for the state file / integration owner.**

---

## Cross-Cutting Test Matrix (summary)
- `tests/test_fibonacci.py` — exact canonical numbers (up/down), edges, repair #3, purity.
- `tests/phase2/` — ERL/IRL classifier, scope refinement + repair #4, dedup repair #5, cycle derivation, side-filter repair #2, ERL-only fallback repair #1, clean_path repair #6, factor monotonicity, the proximity-does-not-dominate core test, snapshot integration, mocked-LLM end-to-end.
- Pre-existing Phase 1/2/3A/3B suite stays green EXCEPT the two flagged goldens (pool-dict, compact/snapshot) and any config-hash-string assertion.

## Global Risks
1. Scope discipline: the strongest failure mode is editing a frozen detector to "make numbers agree." Executor must replicate premium_discount's convention in `fibonacci.py`, never edit premium_discount.
2. Golden creep: only two goldens may change. Enforce via CI diff review.
3. Determinism: all new methods pure or deterministic-over-window; no wall-clock, no RNG, no cross-snapshot state.