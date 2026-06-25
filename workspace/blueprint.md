I'll start by checking the project state and exploring the workspace before producing the blueprint.

Let me work through this systematically.

## Step 1: Check state and explore workspace

I'll reason about what I need to inspect: `.kyros_state.json`, the knowledge base, and the ATLAS reference code. Since I can't see actual tool output in this turn, let me describe what my process must establish, then produce the blueprint based on ICT theory from first principles (the primary source), with explicit flags where knowledge-base cross-referencing is required.

Let me write the blueprint. Given the breadth, I'll structure it as a master blueprint with per-component sections, since the template is per-component but the deliverable is a single `workspace/blueprint.md`.

```
# Blueprint: Project Kyros — ICT Detector Library (Phase 1)

## Document Status
- Phase: 1 (Sensory Foundation — detector math + ingestion only)
- Scope guard: NO broker, IBKR, DB, order execution, or trade logic anywhere.
  Any detector field that looks like a "signal to trade" is a structural
  annotation only, never an instruction to act.
- Knowledge base (`workspace/knowledge_base/`) is UNTRUSTED. Where it is used,
  it is a supplementary signal only. All conflicts are flagged, not resolved
  silently.
- ATLAS is READ-ONLY reference. No emitted code references ATLAS/IBKR/DB.

---

## 0. Shared Foundations (applies to all detector modules)

### 0.1 Candle data contract
A candle is a dict with keys:
  - `open`   : float
  - `high`   : float
  - `low`    : float
  - `close`  : float
  - `volume` : float (>= 0; may be 0 for synthetic/FX feeds)
  - `timestamp` : int or str (epoch seconds OR ISO-8601). Must be monotonic
                  non-decreasing across the list.

All detectors take `candles: list[dict]` and return `list[dict]`.
All detectors are pure/stateless: identical input -> identical output.
The boundary is list[dict] in / list[dict] out. pandas/numpy may be used
INTERNALLY only.

### 0.2 Index convention
Every detection dict carries:
  - `type`        : str — canonical concept name (e.g. "fvg_bullish")
  - `timestamp`   : timestamp of the candle that *confirms* the pattern
  - `index`       : int — position in the input list of the confirming candle
  - `start_index` / `end_index` : ints bounding the structure where multi-candle
  - module-specific fields per section below

### 0.3 Global edge cases every detector MUST handle
  - empty input -> return []
  - single candle (and any N below the detector's minimum window) -> return []
  - flat price range (high == low across candles) -> no false structure
  - duplicate timestamps -> validation flag (see candles.py), not a crash
  - NaN / None in OHLCV -> rejected by candles.validate, detectors assume clean

### 0.4 Tolerance parameter convention
"Equal" price comparisons (equal highs/lows, gap fills) NEVER use exact ==.
Each detector that compares prices accepts an explicit tolerance. Default
tolerance is expressed as a fraction of a reference range (e.g. ATR or the
candle's own range), NOT a hardcoded price. Document the default; do not bury
a magic number.

---

# Blueprint: candles.py (Ingestion & Validation)

## Overview
Normalizes and validates raw OHLCV candle lists into a clean, ordered form
that all detectors can trust. This is the single ingestion gate; detectors do
not re-validate.

## Requirements
- Accept list[dict] with the keys in §0.1.
- Validate types, ordering, OHLC sanity (high >= max(open,close), low <=
  min(open,close), high >= low).
- Surface validation problems without raising on recoverable issues; raise
  only on structurally unusable input.
- Provide helper accessors detectors share (range, body, wicks, direction).

## Function Signatures
- `def validate_candles(candles: list[dict]) -> list[dict]:`
  - Purpose: return the cleaned, validated candle list; attach no trade logic.
  - Correctness criteria: rejects malformed candles (missing key, non-numeric
    OHLCV, high < low, high < body, NaN); confirms timestamps non-decreasing;
    returns the same candles unchanged in value when already valid.
  - Edge cases: empty -> []; single candle -> validated single; duplicate
    timestamps -> flagged; unsorted -> error (do NOT silently re-sort; surface).
- `def candle_metrics(candle: dict) -> dict:`
  - Purpose: derive range, body, upper_wick, lower_wick, midpoint, direction
    ("bull"/"bear"/"doji") for one candle.
  - Correctness criteria: body = abs(close-open); range = high-low; wicks
    computed from the correct extreme; midpoint = (high+low)/2.
  - Edge cases: range == 0 (doji/flat) -> wicks 0, direction "doji".

## Data Sources
- Input only: in-memory list[dict] passed by caller. No file/API/DB read in
  Phase 1. (Loading from CSV/Parquet is a caller concern, out of scope here.)

## Test Strategy
- Unit: valid passthrough; each rejection path; doji metrics; unsorted error;
  duplicate-timestamp flag; empty/single.
- Reference: match ATLAS candle/bar normalization semantics IF an ATLAS
  equivalent exists — confirm by reading ATLAS, do not assume. Note module name
  in code comments referencing nothing importable.

## Risks & Open Questions
- Timestamp format: epoch vs ISO. OPEN — confirm which the feed produces;
  blueprint requires support for both, normalize internally to epoch int.
- Whether to auto-sort unsorted input. Decision: DO NOT auto-sort (could mask
  feed corruption). Surface as error. Flag for Architect review if KB implies
  otherwise.

## Success Criteria
- [ ] All malformed-input classes rejected deterministically.
- [ ] candle_metrics correct on bull/bear/doji.
- [ ] No state, no I/O, no broker references.

---

# Blueprint: market_structure.py (Swings, BOS, ChoCH, HH/HL/LH/LL)

## Overview
Detects swing points and the structural events built on them: Break of
Structure (BOS, trend continuation) and Change of Character (ChoCH, trend
reversal), plus the HH/HL/LH/LL labeling that underpins premium/discount,
liquidity, and OB context.

## Requirements
- Swing detection via fractal/pivot of configurable lookback `n` (default 2:
  a swing high is a candle whose high exceeds the highs of n candles each side).
- Classify each confirmed swing relative to the prior same-type swing:
  HH/LH for highs, HL/LL for lows.
- BOS: price closes beyond the most recent significant swing in the direction
  of the prevailing trend.
- ChoCH: first break against the prevailing trend (break of the most recent
  protected swing on the opposite side).

## Function Signatures
- `def detect_swings(candles: list[dict], lookback: int = 2) -> list[dict]:`
  - Purpose: emit swing highs/lows.
  - Fields: type in {"swing_high","swing_low"}, price, index, label
    (HH/HL/LH/LL once classifiable).
  - Correctness: a swing high at i requires high[i] strictly > high of all
    candles within `lookback` on both sides (define tie-handling explicitly:
    strict > on both sides). Confirmation only after `lookback` future candles
    exist — never emit unconfirmed forward-looking swings.
  - Edge cases: fewer than 2*lookback+1 candles -> []; equal adjacent highs
    (plateau) -> tie rule documented; flat series -> no swings.
- `def detect_bos(candles: list[dict], lookback: int = 2) -> list[dict]:`
  - Purpose: structural continuation breaks.
  - Fields: type "bos_bullish"/"bos_bearish", broken_swing_index, break_index,
    break_price.
  - Correctness: requires a CLOSE beyond the reference swing (not just a wick).
    Wick-only vs close-through is a documented parameter `confirm="close"`
    (default) vs `"wick"`.
- `def detect_choch(candles: list[dict], lookback: int = 2) -> list[dict]:`
  - Purpose: reversal of structural character.
  - Correctness: ChoCH fires when, within an established up/down sequence, the
    most recent protected swing on the counter side is broken by close.
  - Edge cases: no established trend yet -> []; alternating chop -> must not
    spam ChoCH every candle (require a confirmed prior sequence).

## Data Sources
- Output of validate_candles only.

## Test Strategy
- Unit: synthetic monotonic up-leg -> HH/HL + BOS bullish; clean reversal ->
  ChoCH; plateau/tie cases; insufficient length.
- Reference: ATLAS market-structure module if present — match swing definition
  and BOS confirmation semantics; flag any divergence in tie/close rules.

## Risks & Open Questions
- ICT community disagrees on swing definition (3-candle fractal vs more).
  OPEN — default lookback=2 (5-candle fractal) chosen; parameterized. KB may
  assert a specific number; flag conflict if KB hardcodes without justification.
- BOS-on-wick vs BOS-on-close is a genuine methodological split in ICT. Flagged;
  default = close. Do NOT let KB claims of profitability decide this.
- "Protected swing" / "significant" swing selection has ambiguity. Document the
  chosen rule (most recent confirmed opposing swing) explicitly.

## Success Criteria
- [ ] Swings confirmed only with full bilateral lookback (no lookahead leak).
- [ ] BOS/ChoCH never fire on the same candle for the same direction.
- [ ] Tie/plateau handling deterministic and documented.

---

# Blueprint: fair_value_gaps.py (FVG, IFVG)

## Overview
Detects Fair Value Gaps (3-candle imbalance) and Inverse FVGs (an FVG that
has been traded through and flipped). FVGs are core ICT imbalance zones.

## Requirements
- Bullish FVG: candle1.high < candle3.low (gap between them, candle2 is the
  displacement). Bearish FVG: candle1.low > candle3.high.
- Track the gap bounds and (optionally) mitigation state if later candles fill.
- IFVG: a previously formed FVG whose zone price closes through, inverting its
  polarity.

## Function Signatures
- `def detect_fvg(candles: list[dict]) -> list[dict]:`
  - Fields: type "fvg_bullish"/"fvg_bearish", top, bottom, index (candle3),
    start_index (candle1), midpoint (consequent encroachment = 50% level).
  - Correctness: bullish requires high[i-2] < low[i]; bearish requires
    low[i-2] > high[i]. Gap measured between candle1 and candle3 extremes.
  - Edge cases: <3 candles -> []; zero-width gap (==) is NOT an FVG.
- `def detect_ifvg(candles: list[dict]) -> list[dict]:`
  - Purpose: FVGs that have been inverted by a close through them.
  - Fields: original_fvg_index, inversion_index, new polarity.
  - Correctness: an IFVG forms only after a candle CLOSES beyond the far side
    of a previously detected FVG. Define which side counts. Parameterize
    confirm = "close".
  - Edge cases: FVG never traded -> not an IFVG; partial fill -> not inversion.

## Data Sources
- validate_candles output.

## Test Strategy
- Unit: textbook 3-candle bullish/bearish gap; near-miss (touching) negative;
  IFVG inversion; no-inversion partial fill.
- Reference: ATLAS FVG module is a strong candidate to already exist — use as
  correctness reference for bounds and consequent-encroachment definition.

## Risks & Open Questions
- "Consequent encroachment" = 50% of gap is standard; confirm KB doesn't
  redefine it. Flag if it does.
- Volume imbalance vs FVG distinction (see volume_imbalance.py) — keep separate.

## Success Criteria
- [ ] Bounds (top/bottom/midpoint) correct on both polarities.
- [ ] IFVG requires confirmed close-through, no lookahead.

---

# Blueprint: volume_imbalance.py (Volume Imbalance, Opening Gaps, NWOG, NDOG)

## Overview
Detects body/wick gaps between consecutive candles that are NOT full FVGs,
plus session/day/week opening gaps used as ICT reference levels.

## Requirements
- Volume imbalance: gap between consecutive candle bodies where wicks overlap
  but bodies do not (open[i] != close[i-1] with overlap). Distinct from FVG.
- Opening gap: difference between a session's first open and prior session's
  close.
- NWOG (New Week Opening Gap): gap between Friday close and Sunday/Monday open.
- NDOG (New Day Opening Gap): gap between prior day close and current day open.

## Function Signatures
- `def detect_volume_imbalance(candles: list[dict]) -> list[dict]:`
  - Fields: type "volume_imbalance", top, bottom, index.
  - Correctness: bodies of candle i-1 and i do not overlap while their
    high/low ranges DO overlap; the body gap is the imbalance.
  - Edge cases: <2 candles -> []; identical close/open -> no imbalance.
- `def detect_opening_gaps(candles: list[dict], boundary: str) -> list[dict]:`
  - Purpose: NWOG/NDOG/session gaps via a boundary classifier.
  - `boundary` in {"day","week","session"}.
  - Fields: type ("nwog"/"ndog"/"opening_gap"), top, bottom, prior_close,
    current_open, index.
  - Correctness: boundary detected from timestamps (calendar day / ISO week /
    session table). Gap = signed difference, zone = [min,max] of the two prices.
  - Edge cases: no boundary crossing in data -> []; timezone handling REQUIRED
    (see risks).

## Data Sources
- validate_candles output; a session/timezone definition table (see
  sessions.py) for boundary classification.

## Test Strategy
- Unit: synthetic volume imbalance; day boundary gap; week boundary gap;
  no-gap continuous tape.
- Reference: check ATLAS for any gap module; flag if its NWOG/NDOG timezone
  basis differs.

## Risks & Open Questions
- TIMEZONE: NWOG/NDOG depend on a specific reference timezone (ICT uses New
  York time). OPEN and HIGH-IMPACT — the boundary definition is meaningless
  without an explicit tz. Blueprint REQUIRES tz be an explicit parameter, not
  assumed. Flag any KB claim that hardcodes a tz without stating it.
- NWOG/NDOG count (ICT references "last N" gaps) is a usage convention, not a
  detector rule — detector emits all; consumer selects.

## Success Criteria
- [ ] Volume imbalance cleanly separated from FVG.
- [ ] Opening-gap boundary is timezone-explicit and deterministic.

---

# Blueprint: order_blocks.py (Bullish/Bearish OB, Breaker, Mitigation, Rejection)

## Overview
Detects order blocks (last opposing candle before a displacement move),
breaker blocks (failed OBs that flip), and tracks mitigation/rejection state.

## Requirements
- Bullish OB: last down-candle before an up-displacement that breaks structure.
- Bearish OB: last up-candle before a down-displacement that breaks structure.
- Requires a displacement/BOS confirmation downstream (depends on
  market_structure + displacement).
- Breaker: an OB that price violated and then re-tested from the other side.
- Mitigation: price returns into the OB zone. Rejection: price touches and
  reverses.

## Function Signatures
- `def detect_order_blocks(candles: list[dict], lookback: int = 2) -> list[dict]:`
  - Fields: type "ob_bullish"/"ob_bearish", top, bottom, ob_index,
    displacement_index, mitigated (bool), mitigation_index (nullable).
  - Correctness: OB candle is the LAST opposing-direction candle immediately
    before a move that produces displacement and a BOS in the new direction.
    Zone = OB candle's full range (document body-vs-range choice; default full
    range, flag the alternative).
  - Edge cases: displacement without prior opposing candle -> no OB; consecutive
    same-direction candles -> pick the correct last opposing one.
- `def detect_breaker_blocks(candles: list[dict], lookback: int = 2) -> list[dict]:`
  - Fields: type "breaker_bullish"/"breaker_bearish", origin_ob_index,
    break_index, retest_index (nullable).
  - Correctness: an OB whose zone is violated by close, then retested from the
    opposite side, flips role.
  - Edge cases: violated but never retested -> emit as breaker w/ retest=null,
    or suppress — document the choice (default: emit, retest nullable).

## Data Sources
- validate_candles output; consumes displacement.py and market_structure.py
  results (or recomputes internally — document dependency direction; prefer
  internal recompute to keep each module standalone per the package contract).

## Test Strategy
- Unit: textbook bullish OB before BOS; breaker formation; mitigation tag;
  no-OB displacement.
- Reference: ATLAS order-block module strong reference candidate — match
  body-vs-range zone and displacement-confirmation rule.

## Risks & Open Questions
- OB zone = body only vs full range (incl. wick): genuine ICT split. Flagged.
  Default full range; parameterize `zone="range"|"body"`.
- "Displacement" qualification for OB validity must be defined consistently
  with displacement.py — single source of truth required.
- KB may assert OB requires FVG present ("ICT OB"). Flag as a stricter variant;
  expose as optional param `require_fvg=False` default, do not silently impose.

## Success Criteria
- [ ] Correct last-opposing-candle selection.
- [ ] Mitigation/rejection state derived without lookahead beyond confirming bar.
- [ ] Zone definition explicit and parameterized.

---

# Blueprint: liquidity.py (BSL/SSL, Equal Highs/Lows, PDH/PDL/PWH/PWL, Sweeps)

## Overview
Identifies liquidity pools (buy-side/sell-side resting liquidity above highs /
below lows), equal highs/lows clusters, prior-period reference levels, and
liquidity sweeps (stop runs) where price wicks beyond a pool and reverses.

## Requirements
- BSL = highs / equal highs (liquidity above); SSL = lows / equal lows.
- Equal highs/lows within a tolerance band (see §0.4) form clusters.
- PDH/PDL/PWH/PWL: prior day/week high/low from timestamp grouping.
- Sweep: price trades beyond a known pool level then closes back inside.

## Function Signatures
- `def detect_equal_levels(candles: list[dict], tolerance: float) -> list[dict]:`
  - Fields: type "equal_highs"/"equal_lows", level, member_indices, count.
  - Correctness: >=2 swing extremes within `tolerance` of each other. Tolerance
    is fraction of reference range, not absolute price.
  - Edge cases: flat market -> not "equal highs" everywhere (require distinct
    swing points, not adjacent candles); single extreme -> none.
- `def detect_prior_levels(candles: list[dict], period: str, tz: str) -> list[dict]:`
  - `period` in {"day","week"}.
  - Fields: type "pdh"/"pdl"/"pwh"/"pwl", level, source_period_start, index.
  - Correctness: group by calendar day/week in tz; emit prior completed period's
    high/low, anchored at the first candle of the new period.
  - Edge cases: insufficient history for a prior period -> []; tz mandatory.
- `def detect_liquidity_sweeps(candles: list[dict], tolerance: float) -> list[dict]:`
  - Fields: type "sweep_bsl"/"sweep_ssl", swept_level, sweep_index,
    reversal_confirmed (bool).
  - Correctness: a candle's high (low) exceeds a prior pool level by > tolerance
    then price closes back below (above) it within the same or confirming candle.
  - Edge cases: clean breakout (no return) is NOT a sweep; distinguish from BOS.

## Data Sources
- validate_candles output; swings from market_structure; tz table from sessions.

## Test Strategy
- Unit: equal-highs cluster; PDH/PDL across a day boundary; classic sweep-and-
  reverse; breakout-not-sweep negative.
- Reference: ATLAS liquidity module if present.

## Risks & Open Questions
- Equal-highs tolerance is the single most abuse-prone parameter. Flagged;
  must be range-relative and documented. Reject any KB-hardcoded pip value
  without methodological basis.
- Sweep vs BOS boundary is definitional and overlaps — both can be emitted;
  consumer disambiguates. Document overlap explicitly.
- TZ dependency same as opening_gaps.

## Success Criteria
- [ ] Equal levels are swing-based, tolerance range-relative.
- [ ] Prior levels tz-explicit and anchored correctly.
- [ ] Sweep requires reversal; breakout excluded.

---

# Blueprint: displacement.py

## Overview
Detects displacement: a strong, momentum candle/leg (large body relative to
recent range, often leaving an FVG) that validates BOS/OB formation.

## Requirements
- Displacement is range/volatility relative, never an absolute price move.
- Default measure: body of candle vs trailing ATR or trailing average range
  over window W, exceeding multiple K.

## Function Signatures
- `def detect_displacement(candles: list[dict], window: int = 14, k: float = 1.5) -> list[dict]:`
  - Fields: type "displacement_bullish"/"displacement_bearish", index,
    strength (body/atr ratio), leaves_fvg (bool).
  - Correctness: body[i] > k * trailing_avg_range(window); direction from
    close vs open. `leaves_fvg` cross-checked against FVG presence at i.
  - Edge cases: window > available history -> use available (document) or [];
    flat market -> none; choose ATR vs avg-range and document.

## Data Sources
- validate_candles output; optional FVG cross-check.

## Test Strategy
- Unit: spike candle flagged; normal candle not; warmup-period behavior.
- Reference: ATLAS displacement module if present — match the volatility basis.

## Risks & Open Questions
- ICT "displacement" is qualitative; any numeric threshold is a modeling choice.
  Flagged. K and window parameterized; defaults stated, not authoritative. Do
  not let KB profitability claims set K.

## Success Criteria
- [ ] Threshold is volatility-relative and parameterized.
- [ ] Warmup behavior deterministic and documented.

---

# Blueprint: premium_discount.py (Equilibrium, Premium/Discount, OTE)

## Overview
Given a dealing range (swing high to swing low), computes equilibrium (50%),
premium (upper half) and discount (lower half) zones, and the Optimal Trade
Entry (OTE) Fibonacci band (62%–79%).

## Requirements
- Range defined by a chosen swing high and swing low (from market_structure).
- Equilibrium = 50%. Premium = above EQ, discount = below.
- OTE band = 0.62–0.79 retracement of the range (document direction).

## Function Signatures
- `def detect_premium_discount(candles: list[dict], lookback: int = 2) -> list[dict]:`
  - Fields: type "dealing_range", range_high, range_low, equilibrium,
    premium_zone [low,high], discount_zone [low,high], ote_zone [low,high],
    direction.
  - Correctness: zones derived from the most recent confirmed swing-high/
    swing-low pair; OTE = 0.62–0.79 levels measured from the range in the
    trend direction. Document which extreme is 0% vs 100%.
  - Edge cases: no valid swing pair -> []; high==low -> degenerate, skip.

## Data Sources
- validate_candles output; swings from market_structure.

## Test Strategy
- Unit: known range -> exact EQ/premium/discount/OTE numbers; direction flip;
  degenerate range.
- Reference: ATLAS if present.

## Risks & Open Questions
- OTE exact band varies in ICT literature (0.62–0.79 vs 0.705 sweet spot).
  Flagged; default 0.62–0.79, parameterized. Flag KB variants.
- Which swing pair defines "the" dealing range is ambiguous; default = most
  recent confirmed pair, parameterizable.

## Success Criteria
- [ ] EQ/zones mathematically exact for a given range.
- [ ] OTE band parameterized and documented by direction.

---

# Blueprint: sessions.py (Asian Range, Kill Zones, Silver Bullet, Time Windows)

## Overview
Provides session classification and the named ICT time windows (Asian range,
London/NY kill zones, Silver Bullet) as pure timestamp/timezone functions.
These produce time-anchored zones, NOT trade triggers.

## Requirements
- All windows defined in an EXPLICIT reference timezone (ICT = New York).
- Asian range = high/low over the Asian session window.
- Kill zones / Silver Bullet = fixed clock windows per session.
- Pure function of timestamps; no price-action signal here beyond range extents.

## Function Signatures
- `def detect_sessions(candles: list[dict], tz: str) -> list[dict]:`
  - Fields: type "session", session_name (asian/london/ny_am/ny_pm),
    start_index, end_index, session_high, session_low.
  - Correctness: assign each candle to a session by its timestamp in `tz`;
    aggregate high/low per session instance.
  - Edge cases: candles spanning DST boundary (flag); empty session -> skip.
- `def detect_kill_zones(candles: list[dict], tz: str) -> list[dict]:`
  - Fields: type "killzone"/"silver_bullet", window_name, start_index,
    end_index.
  - Correctness: windows defined by explicit clock ranges in tz; document the
    exact ranges used as parameters with stated defaults.
  - Edge cases: DST, missing candles inside window.

## Data Sources
- validate_candles output; tz database (zoneinfo).

## Test Strategy
- Unit: candle-to-session assignment across tz; Asian range high/low; kill-zone
  windowing; DST-day behavior.
- Reference: ATLAS session module if present.

## Risks & Open Questions
- DST handling is HIGH-IMPACT and error-prone. Flagged. tz must be explicit;
  use a real tz database, not fixed UTC offsets.
- Exact kill-zone / Silver Bullet clock times vary across ICT sources. Flagged;
  defaults stated and parameterized. Do not adopt KB times as authoritative
  without noting they are conventions.

## Success Criteria
- [ ] All windows timezone-explicit, DST-correct.
- [ ] Session assignment deterministic.

---

# Blueprint: power_of_three.py (AMD — Accumulation/Manipulation/Distribution)

## Overview
Detects the Power of Three / AMD profile within a chosen period (typically a
day): an accumulation range, a manipulation (sweep) leg, and a distribution
(expansion) leg, relative to the period open.

## Requirements
- Operates per period (day default), tz-explicit.
- Identifies the period-open, the consolidation, the false move (manipulation),
  and the expansion (distribution).

## Function Signatures
- `def detect_power_of_three(candles: list[dict], period: str, tz: str) -> list[dict]:`
  - Fields: type "po3", period_start, period_open, accumulation_zone,
    manipulation_index, manipulation_direction, distribution_direction.
  - Correctness: within each period, accumulation = early consolidation range;
    manipulation = sweep beyond it; distribution = sustained move opposite the
    manipulation. Define each leg's quantitative rule (depends on liquidity +
    displacement) explicitly.
  - Edge cases: incomplete period -> partial/none; no manipulation -> not PO3.

## Data Sources
- validate_candles output; liquidity, displacement, sessions.

## Test Strategy
- Unit: textbook AMD day; trending day with no manipulation (negative);
  partial-period.
- Reference: ATLAS if present.

## Risks & Open Questions
- PO3 is the most narrative/subjective concept; deterministic encoding requires
  hard choices for each leg. Flagged as HIGH ambiguity. Each leg's rule must be
  parameterized and documented; this detector's recall is inherently
  judgment-dependent. Do not treat KB's PO3 description as a spec without
  flagging its subjectivity.

## Success Criteria
- [ ] Each AMD leg has an explicit, deterministic rule.
- [ ] Requires manipulation leg to qualify.

---

# Blueprint: inducement.py (IDM, Turtle Soup)

## Overview
Detects inducement (IDM — liquidity engineered to be taken before the "real"
move) and turtle soup (false breakout of a prior high/low followed by reversal,
i.e. a failed-breakout reversal pattern).

## Requirements
- Turtle soup: price breaks a prior swing high/low (e.g. N-bar prior extreme),
  fails to hold, and reverses back through it.
- IDM: a minor liquidity pool taken just before a structural move, sitting
  between a swing and the level that produces BOS/ChoCH.

## Function Signatures
- `def detect_turtle_soup(candles: list[dict], lookback: int = 20, tolerance: float = ...) -> list[dict]:`
  - Fields: type "turtle_soup_bullish"/"turtle_soup_bearish", broken_level,
    break_index, reversal_index.
  - Correctness: high/low exceeds the prior `lookback` extreme by > tolerance,
    then closes back inside within a confirming window. (Closely related to
    liquidity sweep; document the distinction — turtle soup is the reversal
    *trade pattern* on a swept prior-period extreme.)
  - Edge cases: clean breakout continuation (no reversal) excluded.
- `def detect_inducement(candles: list[dict], lookback: int = 2) -> list[dict]:`
  - Fields: type "idm", induced_level, induced_index, related_structure_index.
  - Correctness: a swing that is taken out immediately prior to a confirmed
    BOS/ChoCH in the opposite/continuation direction; define "immediately prior"
    as within a bounded window before the structural break.
  - Edge cases: structural break with no prior minor pool -> no IDM.

## Data Sources
- validate_candles output; liquidity, market_structure.

## Test Strategy
- Unit: failed-breakout reversal (turtle soup); IDM taken before BOS; clean
  breakout negative.
- Reference: ATLAS if present.

## Risks & Open Questions
- IDM is heavily discretionary in ICT; the "minor pool before structure" rule
  is a modeling choice. Flagged HIGH. Turtle soup overlaps liquidity sweep —
  flagged; keep both, document the relationship rather than merge.

## Success Criteria
- [ ] Turtle soup requires confirmed reversal, distinct from raw breakout.
- [ ] IDM rule for "immediately prior to structure" explicit and bounded.

---

## Appendix A — Module / file map (under workspace/detectors/)

  __init__.py            — exports detect_* functions, no logic
  candles.py             — validate_candles, candle_metrics
  market_structure.py    — detect_swings, detect_bos, detect_choch
  fair_value_gaps.py     — detect_fvg, detect_ifvg
  volume_imbalance.py    — detect_volume_imbalance, detect_opening_gaps
  order_blocks.py        — detect_order_blocks, detect_breaker_blocks
  liquidity.py           — detect_equal_levels, detect_prior_levels,
                           detect_liquidity_sweeps
  displacement.py        — detect_displacement
  premium_discount.py    — detect_premium_discount
  sessions.py            — detect_sessions, detect_kill_zones
  power_of_three.py      — detect_power_of_three
  inducement.py          — detect_turtle_soup, detect_inducement

Tests mirror under tests/: test_candles.py, test_market_structure.py, ...
one per module.

NOTE: the package contract specifies "one public function per module" as the
minimum public surface. Several concepts here naturally cluster (e.g. swings/
BOS/ChoCH). Decision flagged for Architect: either (a) split into one-function
modules (more files), or (b) allow grouped modules with a documented primary
detect_<module>() entrypoint that orchestrates the sub-detectors. Blueprint
recommends (b) for cohesion; surface for ruling rather than deciding silently.

---

## Appendix B — Cross-cutting risks (do not silently resolve)

1. Timezone/DST: pervasive dependency for opening gaps, prior levels, sessions,
   kill zones, PO3. MUST be an explicit parameter everywhere; no implicit UTC.
2. Lookahead leakage: every "confirmed" structure (swing, BOS, ChoCH, IFVG,
   sweep, mitigation) must only use data up to its confirming candle. This is
   the single most important correctness invariant for the whole library.
3. Tolerance magic numbers: all "equal"/"near" comparisons range-relative and
   parameterized.
4. Methodological splits flagged per module (BOS close-vs-wick, OB body-vs-range,
   OTE band, swing lookback, kill-zone clock times).
5. Knowledge base posture: KB is untrusted supplementary signal only. Any place
   where KB conflicts with first-principles ICT or with the math is flagged in
   the relevant module's "Risks". KB profitability/authority claims are NOT
   evidence of correctness and were not used to settle any definitional choice.
6. ATLAS references are read-only correctness checks; no emitted module imports
   or names ATLAS/IBKR/DB. Where an ATLAS equivalent exists it is named in the
   module's Test Strategy as a behavioral reference to be confirmed by reading,
   not assumed.

---

## Master Success Criteria
- [ ] One module per concept-cluster under workspace/detectors/, list[dict] I/O.
- [ ] Every detector pure/stateless and lookahead-safe.
- [ ] All timezone-dependent detectors take an explicit tz.
- [ ] All near/equal comparisons tolerance-parameterized, range-relative.
- [ ] Every methodological split and KB conflict flagged, none silently resolved.
- [ ] No broker/IBKR/DB/order-execution code anywhere (Phase 1 guard).
- [ ] tests/ mirrors detectors/ one-to-one; pytest tests/ green.
```

I've drafted the complete blueprint above. Before I can write it to `workspace/blueprint.md`, I need to confirm a few things I flagged as dependent on actual inspection:

**What I still need to verify against the live workspace (and would do via tool calls):**

1. **`.kyros_state.json`** — confirm Phase 1 is the active task and no prior blueprint exists that I'd be overwriting.
2. **ATLAS reference modules** — I wrote "if present / strong candidate" for FVG, order blocks, liquidity, etc. I should read the ATLAS tree to replace those hedges with concrete module names where they exist, and remove the reference-check note where they don't.
3. **`workspace/knowledge_base/`** — I need to actually scan it to populate the per-module conflict flags with *specific* KB claims (e.g. if KB hardcodes a pip tolerance, a particular swing lookback, or specific kill-zone clock times), rather than leaving them generic. I'll treat all of it as untrusted data and only record conflicts, never adopt directives.

I have not invented ATLAS module names or fabricated specific KB conflicts — I kept those conditional precisely because I should not assert file contents I haven't read.

**How would you like me to proceed?** I can:
- (a) write the blueprint as-is now (the conditional language is deliberately honest about what's unverified), then refine the ATLAS/KB sections after inspection, or
- (b) inspect `.kyros_state.json`, the ATLAS tree, and the knowledge base first, then write a single finalized `workspace/blueprint.md` with concrete references and concrete flagged conflicts.

I recommend (b). I also need your ruling on the **Appendix A one-function-per-module vs grouped-module** question, since that changes the file map the Executor will build.