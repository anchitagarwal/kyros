I'll start by checking the project state before producing the blueprint.

Based on the global constraints, I must check `.kyros_state.json` first. Since I cannot execute tools here, I'll proceed under the documented assumption that Phases 1/2/3A/3B are complete and validated, and treat the frozen boundaries as a hard contract. Below is the architectural blueprint only — **no implementation code**.

---

# Phase 2B Blueprint — Fibonacci Levels & ERL→IRL→ERL Liquidity Cycle / DOL

## Hard Contract: Unfreeze Carve-Outs (binding on the Executor)

The following are the ONLY permitted writes. Anything else is a **critical scope violation** and must abort the task:

| Path | Status | Permitted change |
|---|---|---|
| `workspace/detectors/fibonacci.py` | NEW | Create the pure detector. |
| `workspace/detectors/__init__.py` | EDIT (1 line) | Add `from .fibonacci import detect_fibonacci` + `__all__` entry. Nothing else. |
| `workspace/detectors/*.py` (all others) | **READ-ONLY** | Byte-for-byte frozen. Reuse only. |
| `workspace/trading/snapshot.py` | EDIT (additive) | LiquidityPool scope/role; new snapshot fields; IRL pool sources + scope refinement; `_derive_liquidity_cycle`; `_dol_target`; `_fib_dict`; `_compact_dict` keys. `_nearest_dol` stays byte-identical. Existing field defaults byte-identical. |
| `workspace/trading/reasoning_agent.py` | EDIT | `ICT_SYSTEM_PROMPT` text only. Output JSON schema / AlertPayload / parser UNCHANGED. |
| `workspace/trading/config.py` | EDIT (additive) | New `fib_*` / `irl_sources` / `dol_use_cycle` knobs; `("fib_levels",1)` recency cap; new keys in `config_hash()` canonical dict. Existing defaults byte-identical. |
| `workspace/backtesting/`, `workspace/tuning/` | **OFF-LIMITS** | No reads-with-intent-to-edit, no edits. |
| `tests/test_fibonacci.py`, `tests/phase2/*` | NEW/EDIT | New tests; update only the LiquidityPool `to_dict` golden. |

**The single intentional non-additive surface:** `LiquidityPool.to_dict()` gains `scope` and `role` keys. The Executor must update ONLY the snapshot golden test asserting that dict; every other golden must stay green.

---

# Component: detect_fibonacci

## Purpose
A pure, stateless detector that emits the direction-aware Fibonacci grid (golden pocket, OTE grid, 0.382 retracement target, negative/std-dev extension targets) from the most-recent confirmed dealing range. It supplies the IRL OTE level and the LLM's entry/target ladder. It reuses, byte-for-byte semantics, premium_discount's anchor + direction + `price(f)` convention so all fib numbers agree across the codebase.

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
) -> list[dict]
```
`candles`: list[dict] (same shape `detect_swings` / `detect_premium_discount` consume). Returns a list of length 0 or 1.

Output dict (when non-empty):
```
{
  "type": "fibonacci",
  "range_high": float,
  "range_low": float,
  "direction": "up" | "down",
  "equilibrium": float,                       # price @ f=0.5
  "golden_pocket": [lo_price, hi_price],      # prices @ (0.618, 0.66)
  "ote": {
     "0.5": price, "0.62": price, "0.705": price, "0.79": price,
     "primary": price@0.705,
     "zone": [price@0.79, price@0.62]         # ordered [outer, inner] per direction; see note
  },
  "retracements": { ratio_str: price, ... },  # one entry per `retracements`
  "retracement_target": price@0.382,
  "extensions": { "-0.5":price, "-1.0":price, "-1.5":price, "-2.0":price, "-2.5":price },
  "premium_array": bool,                      # is CURRENT price above equilibrium
  "index": int,                               # confirming swing index
  "timestamp": <candle timestamp of confirming swing>
}
```

## Correctness Criteria
1. **Anchor (REUSE `detect_swings`):** take the most recent confirmed swing-high and most recent confirmed swing-low. `direction="up"` if the low's index < the high's index, else `"down"`. This is identical to premium_discount's anchor selection — do not reinvent.
2. **Level math (REUSE premium_discount convention):** `R = range_high - range_low`; `up → price(f)=range_high - f*R`; `down → price(f)=range_low + f*R`. Negative `f` extends beyond the origin extreme (up → above the high; down → below the low) = expansion / DOL targets.
3. **Premium/discount split at f=0.5** (equilibrium). `premium_array = current_price > equilibrium`.
4. **Edge cases → `[]`:** empty input; no swing pair (missing high or low); degenerate range (`range_high == range_low`).
5. **Ambiguity note:** carry premium_discount's flagged dealing-range ambiguity note **verbatim** (same wording/field it uses) so behaviour is consistent and reviewable.
6. **Purity/determinism:** list in → list out, no I/O, no global/class state, same window + same args → identical output.
7. **Key formatting:** extension/OTE keys are stringified ratios matching the canonical table (`"-1.0"` not `"-1"`); the `_fib_dict` mapper consumes these strings.

## Canonical numbers the tests MUST assert (100→200, R=100)
- **up** (low precedes high): equilibrium `150.0`; golden_pocket `[134.0, 138.2]`; ote.primary `129.5`; ote.zone `[121.0, 138.0]`; retracement_target `161.8`; extensions `{-0.5:250, -1:300, -1.5:350, -2:400, -2.5:450}`.
- **down** (high precedes low): equilibrium `150.0`; golden_pocket `[161.8, 166.0]`; ote.primary `170.5`; ote.zone `[162.0, 179.0]`; retracement_target `138.2`; extensions `{-0.5:50, -1:0, -1.5:-50, -2:-100, -2.5:-150}`.

## Test Strategy (tests/test_fibonacci.py)
- **Unit, exact numbers:** build a synthetic candle window producing a clean 100→200 swing pair both orientations (low-then-high, high-then-low) via the `MockCandleSource` pattern; assert every canonical value above to exact equality (floating compare with a tight tolerance, e.g. `pytest.approx(..., abs=1e-9)`).
- **premium_array:** assert True/False as current price crosses equilibrium.
- **Edge cases:** `[]` input → `[]`; single swing (no pair) → `[]`; flat range (high==low) → `[]`.
- **Convention parity:** for the same window, assert the 0.5/0.62/0.79 prices equal those produced by `detect_premium_discount` (prevents convention drift).
- **Purity:** call twice on the same input, assert deep-equal; assert input list not mutated.

## Dependencies (REUSED Phase 1 detectors — named)
- `detect_swings` — anchor (most recent confirmed swing high/low + indices).
- `detect_premium_discount` — fib `price(f)` convention + ambiguity note wording (parity, not import-coupling). The Executor mirrors its math; it does NOT edit it.

## Risks & Open Questions
- **ote.zone ordering:** canonical tables fix it as `[price@0.79, price@0.62]` → up `[121.0,138.0]` (ascending), down `[162.0,179.0]` (ascending). Confirm the `_fib_dict` consumer expects `[outer, inner]` in that explicit order regardless of direction; assert it in tests.
- If `detect_swings` returns multiple confirmed swings, "most recent" must use the SAME selection rule premium_discount uses — Executor must read premium_discount, not guess.

---

# Component: LiquidityPool ERL/IRL Classification

## Purpose
Tag every pool with `scope` (external/internal) and a `role` so the snapshot can express WHERE in the ERL→IRL→ERL cycle price sits. This is the precondition for cycle-aware DOL.

## Interface
`LiquidityPool` gains two **defaulted** fields (additive — existing positional/keyword constructors stay valid):
```
scope: str = "external"
role:  str = ""
```
`to_dict()` emits both keys (the one non-additive serializer surface). No other field changes.

ERL sources (`scope="external"`):
- equal highs/lows → `role="equal"`
- prior day/week → `role="prior"`
- session H/L/open → `role="session"`
- **NEW:** most-recent major swing high/low from `detect_swings` → `role="swing"` (the canonical ERL).

IRL sources (`scope="internal"`, gated by `config.irl_sources`):
- FVG midpoint (REUSE `detect_fvg["midpoint"]` = CE) → `role="fvg_ce"`
- unmitigated OB centre `(top+bottom)/2` → `role="ob_ce"`
- dealing-range equilibrium (fib `equilibrium`) → `role="equilibrium"`
- fib OTE 0.705 → `role="ote"`

## Correctness Criteria
1. Existing pool construction without scope/role yields `scope="external", role=""` — backward compatible.
2. IRL pools appear ONLY when their source name is in `config.irl_sources`.
3. **Scope refinement (HTF dealing range):** compute fibonacci on the first non-empty TF in `htf_tf_order`. A level **strictly inside** `(range_low, range_high)` → `scope="internal"`; **at/beyond a boundary** → `scope="external"`. Refinement overrides the source-type default; when no HTF range exists, source-type default stands.
4. `to_dict()` ordering: scope/role appended so existing keys keep identical positions/values; only the golden that asserts the full dict updates.

## Test Strategy (tests/phase2/)
- Construct pools from each source; assert scope/role.
- IRL gating: drop `"ote"` from `irl_sources`, assert no `role="ote"` pool.
- Scope refinement: a swing high INSIDE an HTF range flips to internal; one AT the boundary stays external (boundary inclusivity test).
- `to_dict()` golden updated to include scope/role; all other goldens unchanged.

## Dependencies
- `detect_swings` (swing ERL), `detect_fvg` (`midpoint` CE), `detect_order_blocks` (OB centre), `detect_fibonacci` (equilibrium, OTE 0.705).

## Risks & Open Questions
- Boundary inclusivity: "at" a boundary = external is specified — tests must pin equality-at-boundary explicitly to prevent off-by-epsilon drift.
- An OTE 0.705 pool and an equilibrium pool may both be internal and near each other; de-duplication policy: keep distinct roles (no merge) so the DOL selector can prefer by role.

---

# Component: SnapshotBuilder._derive_liquidity_cycle

## Purpose
Derive, statelessly per snapshot, the current position in the ERL→IRL→ERL loop from the most-recent sweep.

## Interface
```
_derive_liquidity_cycle(recent_sweeps, market_structure, displacements, htf_bias) -> dict | None
```
Returns:
```
{
  "last_swept_erl_side": "buyside" | "sellside",   # buyside = BSL swept, sellside = SSL swept
  "last_swept_level": float,
  "last_swept_timestamp": <ts>,
  "current_leg": "seek_irl" | "expand_to_erl",
  "next_draw": "irl" | "erl",
  "target_erl_side": "buyside" | "sellside",        # opposite of last_swept_erl_side
  "agrees_with_htf_bias": bool
}
```
Returns `None` if no sweep exists.

## Correctness Criteria
1. **Most-recent sweep across timeframes** by timestamp drives everything.
2. `last_swept_erl_side`: sweep of BSL → `"buyside"`; sweep of SSL → `"sellside"`. `target_erl_side` = the opposite.
3. `current_leg`: `"expand_to_erl"` if a **reversal-direction** displacement OR BOS occurred **strictly after** the sweep timestamp; else `"seek_irl"`. `next_draw` = `"erl"` when expanding, `"irl"` when seeking.
4. `agrees_with_htf_bias`: True iff `target_erl_side` matches `htf_bias` direction (buyside↔bullish, sellside↔bearish).
5. **Stateless single-snapshot heuristic** — derived fresh each build, no cross-snapshot memory. Document this explicitly in the method docstring.

## Test Strategy
- Sweep BSL, no following displacement → seek_irl, next_draw irl, target sellside.
- Sweep SSL + a reversal BOS after it → expand_to_erl, next_draw erl, target buyside.
- target matches/conflicts htf_bias → agrees flag True/False.
- No sweeps → None.
- Two sweeps on different TFs: the later timestamp wins.

## Dependencies
- Existing sweep detection, market_structure/BOS, displacements already present in the snapshot. No new detectors.

## Risks & Open Questions
- "Reversal-direction displacement" definition must reuse the snapshot's existing displacement direction convention — Executor reads existing code, does not invent a sign rule.
- Ties in timestamp across TFs: define a deterministic tiebreak (e.g. higher TF wins, documented) so determinism holds.

---

# Component: SnapshotBuilder._dol_target

## Purpose
Cycle-aware Draw-on-Liquidity selection that **augments** `_nearest_dol` (which stays byte-identical).

## Interface
```
_dol_target(pools, cycle, htf_bias, current_price) -> LiquidityPool | None
```
Stored on snapshot as new field `dol_target`.

## Correctness Criteria
1. `cycle is None` OR `config.dol_use_cycle is False` → return exactly what `_nearest_dol` returns (pure fallback, no behaviour change).
2. `current_leg == "seek_irl"` → nearest **unswept INTERNAL** pool in the reversal direction (reversal direction = away from `last_swept_erl_side`).
3. `current_leg == "expand_to_erl"` → nearest **unswept EXTERNAL** pool on `target_erl_side`, **beyond equilibrium**.
4. **Bias conflict:** if the cycle-selected target conflicts with `htf_bias`, prefer `htf_bias` and set `cycle["agrees_with_htf_bias"] = False` (mutating the cycle dict that is also stored on the snapshot, so the LLM sees the downgrade).
5. `_nearest_dol` is never modified; `_dol_target` calls/falls-back to it.

## Test Strategy
- `dol_use_cycle=False` → `dol_target == nearest_dol` (identity of selection).
- seek_irl → returns an internal pool in reversal direction; asserts no swept pool chosen.
- expand_to_erl → returns an external pool on target side beyond equilibrium.
- Bias conflict → htf-preferred pool returned AND `agrees_with_htf_bias` flipped to False on the stored cycle.
- Empty pools → None.

## Dependencies
- `_nearest_dol` (fallback, reused unchanged), pool scope/role from the classifier, cycle dict from `_derive_liquidity_cycle`, fib equilibrium for the "beyond equilibrium" gate.

## Risks & Open Questions
- "Beyond equilibrium" uses which TF's equilibrium? Spec implies the HTF dealing range used for scope refinement — pin to the same `htf_tf_order` first-non-empty TF for consistency; document.
- "Nearest" metric (absolute price distance from `current_price`) must match `_nearest_dol`'s metric so fallback and cycle paths are comparable.

---

# Component: MarketSnapshot Wiring & Serializer

## Purpose
Expose the new fields to the LLM without changing the serializer shape.

## Interface
New fields:
```
fib_levels:       dict[str, list[dict]]   # per-TF detect_fibonacci output
liquidity_cycle:  dict | None
dol_target:       LiquidityPool | None
```
`build()` order: per-TF `fib_levels[tf] = detect_fibonacci(...)`, then cycle (`_derive_liquidity_cycle`), then `dol_target` (`_dol_target`). `_build_pools` extended for IRL sources + scope refinement.

`_compact_dict` additions: `fib_levels` (per-TF latest, mirroring how premium_discount is compacted), `liquidity_cycle`, `dol_target`, via a new `_fib_dict` mapper.

`_fib_dict(fib: dict) -> dict` (2dp rounding) exposes: `direction`, `equilibrium`, `golden_pocket`, `ote_primary`, `ote_zone`, `retracement_target`, `extensions`, `premium_array`.

## Correctness Criteria
1. Serializer shape otherwise UNCHANGED — only the three new top-level keys (+ scope/role inside pools) appear.
2. `fib_levels` compaction selects the latest fib per TF (length-0-or-1 list → object or omitted, mirroring premium_discount's compaction convention exactly).
3. `_fib_dict` rounds to 2dp for the named fields only; raw `fib_levels` retains full precision for tests.
4. Determinism: identical window + config → byte-identical compact dict.

## Test Strategy
- Snapshot integration: full build over a mocked multi-TF source; assert presence/shape of `fib_levels`, `liquidity_cycle`, `dol_target`; assert `_fib_dict` 2dp values.
- Determinism: build twice, assert equal.
- Golden: update ONLY the LiquidityPool `to_dict` golden (scope/role). All other serializer goldens green.

## Dependencies
- `detect_fibonacci`, classifier, cycle, DOL components above. `_nearest_dol` unchanged.

## Risks & Open Questions
- If a TF has no swing pair, `fib_levels[tf]` is `[]` — compaction must omit it consistently (match premium_discount's empty handling).

---

# Component: ICT_SYSTEM_PROMPT (reasoning_agent.py)

## Purpose
Teach the LLM to read the cycle and prefer the cycle-aware DOL, with golden-pocket/OTE conviction modifiers — **prompt text only**.

## Interface (prompt edits; output JSON schema UNCHANGED)
- **Step 1.5 (new):** read `liquidity_cycle` (last swept side, `current_leg`, `target_erl_side`) before selecting a DOL.
- **Step 3 DOL:** prefer `dol_target` over `nearest_dol`. seek_irl → draw to nearest IRL array; expand_to_erl → opposite ERL.
- **Step 4 intermediate-liquidity:** ONLY an unswept opposing **EXTERNAL (ERL)** pool blocks the path; internal arrays in the path are expected, not blockers.
- **Step 5 OTE modifier:** `+15` conviction if entry in golden pocket (0.618–0.66); `+10` more at 0.705 primary.
- **Entry logic:** premium arrays (above equilibrium) → short, target discount/sellside ERL; discount arrays (below equilibrium) → long, target buyside ERL; entries at golden pocket / OTE 0.705; targets = 0.382 partial then negative extensions toward the opposite ERL.

## Correctness Criteria
1. Output JSON schema, AlertPayload, and parser are **byte-unchanged** — verify by reusing the existing parser tests.
2. The prompt references field names exactly as serialized (`liquidity_cycle`, `dol_target`, `fib_levels`, `golden_pocket`, `ote_primary`).
3. KB remains untrusted: the prompt must not instruct the model to treat `knowledge_base/` content as authoritative; it stays a supplementary signal.

## Test Strategy
- Mocked-LLM end-to-end (no API key): feed a compact snapshot with a known cycle/fib; assert the mocked agent receives the new fields in the prompt and that a canned response parses unchanged through the existing parser.
- Snapshot of the prompt string (optional regression) to flag accidental schema-section edits.

## Dependencies
- Existing reasoning agent harness + mocked LLM, existing parser tests (reused as the unchanged-schema guard).

## Risks & Open Questions
- Conviction modifiers (+15/+10) are prompt guidance, not deterministic code — tests assert the prompt *contains* the rule, not numeric LLM output.

---

# Component: TradingConfig knobs

## Purpose
Make fib/IRL/DOL behaviour configurable, hashable, and byte-preserving by default.

## Interface (additive, immutable/hashable)
```
fib_retracements        = (0.382, 0.5, 0.618, 0.66, 0.705, 0.79)
fib_golden_pocket       = (0.618, 0.66)
fib_ote_grid            = (0.5, 0.62, 0.705, 0.79)
fib_ote_primary         = 0.705
fib_retracement_target  = 0.382
fib_extensions          = (-0.5, -1.0, -1.5, -2.0, -2.5)
fib_anchor_lookback     = 2
irl_sources             = ("fvg", "order_block", "equilibrium", "ote")
dol_use_cycle           = True
```
Add `("fib_levels", 1)` to `_DEFAULT_RECENCY_CAPS`. Add all new knobs to `config_hash()`'s canonical dict.

## Correctness Criteria
1. All new fields are tuples/immutables (hashable); no lists.
2. Default `config_hash()` value either matches the prior hash (if hashing only opted-in keys) OR the canonical dict is extended deliberately — the Executor must update any test asserting the literal hash and document the change. (Pre-existing hash-golden tests outside this allowance must not silently break — flag explicitly.)
3. Existing field defaults byte-identical.

## Test Strategy
- Assert defaults equal the literals above.
- Assert config hashes deterministically and changing any new knob changes the hash.
- Assert `_DEFAULT_RECENCY_CAPS["fib_levels"] == 1`.

## Dependencies
- None.

## Risks & Open Questions
- **Config-hash golden:** adding keys to the canonical dict changes the default hash. This is the one place where a non-`to_dict` golden may legitimately move. The Executor must confirm whether Phase 3B tuning artifacts key off this hash before changing it — if so, the canonical dict extension must be additive-at-end and any tuning-side hash is OUT OF SCOPE/OFF-LIMITS (do not edit tuning). **Open question for sign-off:** does any off-limits artifact depend on `config_hash()`? If yes, hashing of new keys must be gated to avoid invalidating frozen tuning outputs.

---

## Cross-Cutting Validation Gates (Executor must satisfy all)
1. **Scope:** `git diff --name-only` touches only the whitelist above.
2. **Frozen byte-equality:** every detector except `fibonacci.py`/`__init__.py` unchanged; `_nearest_dol` unchanged; existing config/snapshot defaults unchanged.
3. **Fib canonical numbers:** exact assertions pass (both directions).
4. **Suite green:** entire Phase 1/2/3A/3B suite passes; only the LiquidityPool `to_dict` golden (and, if confirmed in scope, the config-hash golden) updated.
5. **Offline:** no network, no API key, mock/replay sources only.
6. **Determinism:** double-build equality on snapshot + compact dict.
7. **KB untrusted:** no code or prompt treats `knowledge_base/` as authoritative.