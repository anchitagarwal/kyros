Before producing the blueprint, I need to flag the gate-zero dependency and confirm my reading of the frozen-layer contract. The rest is the per-component blueprint.

---

## PREREQUISITE GATE (must pass before any tuning component is built)

**Hard prerequisite P0 — config threading must be byte-identical.**
The only permitted edit to `workspace/trading/` is threading `config: TradingConfig = TradingConfig()` through `SnapshotBuilder`, `TriggerEngine`, `CooldownState`, and `validate_rr`, replacing hardcoded literals with config reads whose defaults equal today's literals. The acceptance test for P0 is: **the entire pre-existing Phase 1/2/3A suite passes with zero edits to any existing test.** If any Phase 2 test changes behavior or requires editing, that is a **blueprint failure** — the Executor must STOP and report it, not design around it. No tuning component may be built until P0 is green.

**Open question O0 (must be resolved by reading the code before implementation):**
The brief says `conviction_min` "the prompt's 40 floor, now also enforceable in Python." I need the Executor to confirm whether conviction is currently enforced *only* in the LLM prompt (i.e. the Python trading layer never rejects on conviction today). If so, adding a Python conviction gate with default `40` is **only** byte-identical if every recorded alert today already has conviction ≥ 40 (because the prompt enforced it). The Executor MUST verify this against existing fixtures. If any historical alert has conviction < 40 and currently passes, then default-40 is NOT behavior-preserving and the default must be set to the value that reproduces current behavior (likely `0` / disabled-in-Python), with 40 available only as a tunable. **Resolve before coding config.py.**

---

# Component: TradingConfig
## Purpose
Single dataclass holding every currently-hardcoded trading knob so Tier-2 can record alternate pre-LLM configurations without editing the frozen trading layer. Provides a stable hash for per-config output paths.

## Interface
`workspace/trading/config.py`
```
@dataclass(frozen=True)
class TradingConfig:
    rr_min: float = 1.0
    conviction_min: int = <VALUE PER O0>          # see prerequisite O0
    no_trade_cooldown_minutes: int = 5
    confluence_band_pct: float = 0.001
    recency_caps: Mapping[str, int] = (defaults below)
        # sweeps/displacements/inducements = 10
        # swings/fvg/ifvg/ob/breaker/volume_imbalance = 5
        # opening_gaps/po3 = 3
    pools_to_llm: int = 5
    htf_tf_order: tuple[str, ...] = ("4h", "1h")
    killzone_windows: Mapping[str, tuple[str, str]] = (current literals)
    soft_trigger_order: tuple[str, ...] = (current literal)
    soft_trigger_tf_map: Mapping[str, str] = (current literals)

    def config_hash(self) -> str: ...   # sha256 over a canonical, sorted field tuple
```
Mutable-default fields (Mappings/tuples) must be frozen/immutable types so the dataclass stays hashable and `config_hash()` is stable. `config_hash()` MUST canonicalize: sort mapping keys, normalize numeric formatting, then sha256 — so two logically-equal configs hash identically and float reordering cannot fork a run directory.

## Correctness Criteria
- `TradingConfig()` field-by-field equals every current literal in the trading layer (audit each call site).
- Behavior-preserving: with no config passed, SnapshotBuilder/TriggerEngine/CooldownState/validate_rr produce byte-identical outputs to pre-change code (verified by the unedited Phase 2 suite).
- `config_hash()` is deterministic across processes and Python hash-seed randomization (must NOT use builtin `hash()`).
- Two configs differing in any tunable field produce different hashes; two equal configs produce identical hashes.

## Test Strategy
- Unit: default values match a hardcoded expected snapshot of today's literals.
- Unit: `config_hash()` stable across two constructions, differs on any field change, insensitive to mapping insertion order.
- Integration (the real gate): full Phase 1/2/3A suite green, unedited.
- Differential: run SnapshotBuilder/TriggerEngine on a fixture with `config=None` vs `config=TradingConfig()` → byte-identical outputs.

## Dependencies
REUSES nothing; CONSUMED BY every Tier-2 component and the recording loop. No new logic — only relocates literals.

## Risks & Open Questions
- O0 (conviction default) above is the dominant risk to byte-identity.
- Risk: a literal hides in a default argument or constant elsewhere in the call chain and gets missed → behavior drifts only on non-default configs. Mitigation: differential test plus a grep audit listed in the Executor checklist.

---

# Component: PostLLMParams + grids
## Purpose
Typed search space. PostLLMParams = the Tier-1 (free) re-scoring knobs. PreLLMGrid = the Tier-2 (costly) recording variants.

## Interface
`workspace/tuning/params.py`
```
@dataclass(frozen=True)
class PostLLMParams:
    conviction_min: int
    rr_min: float
    allowed_models: frozenset[str]
    allowed_killzones: frozenset[str]

def param_grid(
    conviction_mins: Sequence[int],
    rr_mins: Sequence[float],
    model_sets: Sequence[frozenset[str]],
    killzone_sets: Sequence[frozenset[str]],
) -> Iterator[PostLLMParams]: ...   # cartesian product

def default_post_params() -> PostLLMParams: ...   # mirrors TradingConfig() defaults

def PreLLMGrid(configs: Sequence[TradingConfig] | None = None) -> list[TradingConfig]:
    # default -> [TradingConfig()]  (i.e. Tier-2 sweep off by default → zero cost)
```

## Correctness Criteria
- `param_grid` yields the full cartesian product, deterministic order.
- `default_post_params()` equals the post-LLM projection of `TradingConfig()` — so baseline tuning == baseline config.
- `PreLLMGrid()` with no args yields exactly `[TradingConfig()]` (default path is free).
- `allowed_models` / `allowed_killzones` semantics: empty set = "allow nothing" or a sentinel "allow all"? **Decide and document**: recommend `None`-or-sentinel for "allow all" to avoid empty-set ambiguity; an empty frozenset must mean "reject everything" and be testable as such.

## Test Strategy
- Unit: product size = ∏ axis lengths; order deterministic across runs.
- Unit: default params equal config-derived baseline.
- Unit: PreLLMGrid default is single-element identity-default config.

## Dependencies
REUSES TradingConfig.

## Risks & Open Questions
- "Allow all" sentinel ambiguity (above) — must be nailed in the interface so rescore.py and report.py agree.

---

# Component: rescore
## Purpose
Apply PostLLMParams to a recorded trace by pure arithmetic: filtered alerts become `no_trade` (R=0); surviving alerts keep their recorded outcome verbatim. No re-simulation, ever.

## Interface
`workspace/tuning/rescore.py`
```
def rescore_trace(trace: dict, p: PostLLMParams) -> dict: ...
def rescore_traces(traces: Iterable[dict], p: PostLLMParams) -> list[dict]: ...
```
`rescore_trace` returns a NEW dict (no mutation of input). When the recorded alert fails ANY of:
- `alert.conviction < p.conviction_min`
- recomputed `risk_reward < p.rr_min`
- `alert.model not in p.allowed_models`
- `alert.killzone not in p.allowed_killzones`
…it rewrites the trace to a `no_trade` outcome with `actual_rr = 0` and sets `alert.bias`/outcome fields to the no_trade form used elsewhere. Otherwise the trace is returned unchanged (recorded outcome reused).

risk_reward is **recomputed from the recorded entry/stop/target** (not read from a possibly-stale stored field), using the same R:R formula `validate_rr` uses, so the rr_min filter is internally consistent with the trading layer.

## Correctness Criteria
- Idempotent: `rescore_trace(rescore_trace(t,p),p) == rescore_trace(t,p)`.
- A trace already recorded as `no_trade`/`no_fill`/`expired` stays whatever it is — filters only ever *downgrade* a taken trade to no_trade; they never promote.
- Surviving taken trades have byte-identical outcome/actual_rr to the input.
- Order preserved (chronological) by `rescore_traces`.
- **MANDATORY documented limitation (docstring + report):** re-scoring treats alerts as independent and ignores cooldown re-interaction — filtering a trade would in reality free a cooldown slot the LLM was never queried about. Therefore cooldown is a TradingConfig (recording-time, Tier-2) knob, NOT a post-LLM re-scored filter. `PostLLMParams` deliberately has no cooldown field.

## Test Strategy
- Unit: each filter independently flips a crafted taken trade to no_trade; passing trace unchanged.
- Unit: rr recompute matches `validate_rr` on the same entry/stop/target.
- Unit: idempotence; order preservation; no input mutation.
- Unit: pre-existing no_trade/no_fill traces untouched by any filter.
- Property: for any params, every output trace's outcome ∈ {original-outcome, no_trade}.

## Dependencies
REUSES the R:R formula from `validate_rr` (workspace/trading) — import/share, do not duplicate the arithmetic.

## Risks & Open Questions
- Trace schema field names (`conviction`, `model`, `killzone`, `entry_zone`/`stop`/`target`) must be confirmed against the actual Phase 3A trace_traces.jsonl schema before coding. The Executor must pin these to the real keys, not assumed ones.

---

# Component: objective
## Purpose
Score a (traces, params) pair by re-scoring then computing expectancy via the existing report engine.

## Interface
`workspace/tuning/objective.py`
```
MIN_TRADES: int  # module constant, overridable by caller
def evaluate(traces: Sequence[dict], p: PostLLMParams,
             min_trades: int = MIN_TRADES) -> tuple[float, dict]: ...
```
Pipeline: `rescore_traces(traces, p)` → feed into **PerformanceReport's overall-metrics computation** → objective = `expectancy_per_trade` (mean actual_rr over all alerts, no_trade counted as 0). Returns `(-inf, metrics)` when `taken_trades < min_trades`. `metrics` always returned (even on -inf) so the report can show why a fold was rejected.

## Correctness Criteria
- expectancy = mean(actual_rr) over the re-scored set with no_trade=0 — matches PerformanceReport's definition exactly (REUSE it; do not recompute the mean independently).
- taken_trades is counted on the RE-SCORED set, not the raw set.
- Deterministic.
- `-inf` strictly below `min_trades`; finite at exactly `min_trades`.

## Test Strategy
- Unit: known traces + known params → hand-computed expectancy.
- Unit: below MIN_TRADES → -inf, metrics still populated.
- Integration: default params over real trace_traces.jsonl → expectancy equals PerformanceReport's own overall expectancy on the unfiltered file (sanity tie-out).

## Dependencies
REUSES `PerformanceReport` metric computation (workspace/backtesting/report.py) and rescore.

## Risks & Open Questions
- If PerformanceReport's expectancy definition differs from "mean actual_rr with no_trade=0," the brief's objective and the reused metric could diverge. The Executor must confirm the exact PerformanceReport field and use IT as the single source of truth; if mismatched, flag rather than silently redefine.

---

# Component: search
## Purpose
Grid search over PostLLMParams on a training slice.

## Interface
`workspace/tuning/search.py`
```
def best_params(train_traces: Sequence[dict],
                grid: Iterable[PostLLMParams],
                min_trades: int) -> tuple[PostLLMParams, float, dict]: ...
```
Returns the params with max objective, its score, its metrics. Ties broken deterministically (e.g. first in grid order, or a documented secondary key — pick one and document). If every grid point is below min_trades, returns the default/baseline params with their (-inf or finite) score so the fold still produces a comparable record — **document this fallback**.

## Correctness Criteria
- Returns the argmax; deterministic tie-break.
- Pure function of (train_traces, grid, min_trades).
- Never returns params unseen in the grid.

## Test Strategy
- Unit: synthetic traces where one param is provably best.
- Unit: deterministic tie-break with two equal-scoring params.
- Unit: all-below-min_trades fallback behavior.

## Dependencies
REUSES objective.evaluate.

## Risks & Open Questions
- Tie-break choice is a judgment call; must be fixed and documented so walk-forward stability analysis is meaningful.

---

# Component: walkforward
## Purpose
Rolling train/test split by timestamp; pick best (config, params) on each train slice, evaluate on the next unseen test slice, and compare to baseline on the same test windows.

## Interface
`workspace/tuning/walkforward.py`
```
@dataclass(frozen=True)
class Fold:
    train: list[dict]; test: list[dict]
    train_start: str; train_end: str; test_start: str; test_end: str

def make_folds(traces: Sequence[dict], train_days: int, test_days: int,
               step_days: int) -> list[Fold]: ...

@dataclass(frozen=True)
class FoldResult:
    fold: Fold
    chosen_config: TradingConfig
    chosen_params: PostLLMParams
    is_expectancy: float
    oos_expectancy: float
    oos_metrics: dict
    baseline_oos_expectancy: float
    baseline_oos_metrics: dict

@dataclass(frozen=True)
class WalkForwardResult:
    folds: list[FoldResult]
    # plus aggregates computed in report (or precomputed here, document which)

def run_walkforward(trace_sets: Mapping[str, list[dict]],  # config_hash -> traces
                    folds: list[Fold],
                    grid: Iterable[PostLLMParams],
                    min_trades: int) -> WalkForwardResult: ...
```
`trace_sets` maps each pre-LLM config's hash to its full-span recorded traces (Tier-2 output, or just `{baseline_hash: baseline_traces}` for the free path). `make_folds` operates on the BASELINE config's timestamps to define fold date bounds; per-config traces are then sliced to those same date bounds — so all configs share identical fold windows.

**Fold construction (rolling, by timestamp, no leakage):**
- Sort by ISO timestamp. Window start advances by `step_days`. For each window: train = `[start, start+train_days)`, test = `[start+train_days, start+train_days+test_days)`.
- A trace falls in at most one of {train, test} of a given fold (half-open intervals; no boundary double-count). The critical invariant: **for every fold, train and test sets are disjoint by timestamp** — assert it.

**Per fold:** over the product (pre-LLM configs × post-LLM grid), score each (config, params) on that fold's TRAIN slice of that config's traces; pick the argmax → (chosen_config, chosen_params, is_expectancy). Evaluate that SAME choice on the fold's TEST slice of that config's traces → oos. Separately evaluate BASELINE (default config + default params) on the same TEST window → baseline_oos.

## Correctness Criteria
- **No leakage:** assert train∩test = ∅ per fold (by timestamp). This is the phase's most important invariant — a dedicated test must try to break it.
- Rolling: consecutive folds' starts differ by exactly `step_days`.
- Half-open intervals prevent boundary trades landing in both train and test.
- IS computed on train, OOS on test, always disjoint.
- Baseline OOS uses the baseline config's traces sliced to the SAME test window as the tuned choice — apples-to-apples.
- Deterministic: same inputs → same folds and same FoldResults.
- Folds with empty train or empty test are dropped (or flagged), documented.

## Test Strategy
- Unit: known timestamp sequence → exact expected fold boundaries; assert disjointness on every fold.
- Adversarial: a trace exactly on a train/test boundary lands in exactly one side; construct it deliberately.
- Unit: step_days < test_days (overlapping windows across folds is fine) vs ≥ — confirm cross-fold overlap is allowed but intra-fold train/test never overlap.
- Integration: real traces → folds produced, each FoldResult has IS and OOS, baseline computed.
- Determinism: two runs identical.

## Dependencies
REUSES objective.evaluate, search.best_params, PostLLMParams, TradingConfig.

## Risks & Open Questions
- Timestamp parsing/timezone normalization must be consistent (parse once to comparable datetimes). Mixed tz offsets in traces would corrupt ordering — normalize and document.
- Sparse data → folds with too few trades; min_trades guard handles scoring but report must surface "fold N degenerate."

---

# Component: WalkForwardReport
## Purpose
Human-readable honesty report. The deliverable that tells the user whether tuning is real or noise.

## Interface
`workspace/tuning/report.py`
```
class WalkForwardReport:
    @staticmethod
    def generate(result: WalkForwardResult,
                 out_path: str = "workspace/walkforward_report.md") -> str: ...
    # returns the markdown string AND writes it to out_path
```
Mandatory sections:
1. **Per-fold table:** fold dates, chosen config_hash + chosen params, IS expectancy, OOS expectancy, OOS win rate / profit factor / max_drawdown_r, OOS taken_trades.
2. **Aggregate OOS vs baseline:** mean OOS expectancy (tuned) vs mean OOS expectancy (baseline), pooled across folds; same for win rate / PF / max_dd.
3. **Parameter stability:** per-field distribution of chosen params across folds; flag when winners differ wildly fold-to-fold ("unstable → likely noise").
4. **OVERFITTING WARNING:** emit when aggregate IS ≫ aggregate OOS (configurable gap threshold, documented) OR tuned OOS ≤ baseline OOS. If tuned OOS ≤ baseline OOS, state plainly: *"Tuning added nothing; use baseline."*
5. **Disclaimers:** carry forward the Phase 3A optimism disclaimer verbatim-in-spirit, PLUS a leakage note: the LLM's training data overlaps the backtest period, so even OOS numbers here are optimistic relative to truly unseen future data.
6. Degenerate-fold note for folds below min_trades.

## Correctness Criteria
- Every fold appears in the table.
- Aggregates are pooled/averaged by a documented rule (trade-weighted vs fold-equal — pick one, justify; recommend reporting both mean-of-folds and trade-weighted to avoid hiding small-sample folds).
- Overfitting warning fires under both trigger conditions; test both.
- Both disclaimers always present.
- Pure function of result (no recomputation that could disagree with walkforward — read aggregates from result/objective metrics, don't re-derive expectancy a third way).

## Test Strategy
- Unit: result where tuned OOS ≤ baseline → "tuning added nothing" present.
- Unit: result where IS ≫ OOS → overfitting warning present.
- Unit: unstable params across folds → instability flag present.
- Unit: disclaimers always rendered.
- Snapshot: deterministic markdown for a fixed result (modulo a stamped timestamp, which should be excluded or fixed in tests).

## Dependencies
REUSES metrics already in FoldResult (which came from PerformanceReport) — does not recompute trade math.

## Risks & Open Questions
- Aggregation rule choice materially changes the headline; documenting and showing both protects honesty.
- "Wildly different" stability threshold is heuristic — make it explicit and conservative.

---

# Component: scripts/run_tuning.py + Tier-2 recording loop
## Purpose
CLI entry. Default path: post-LLM tuning over an existing trace file, ZERO LLM calls. `--record`: cost-gated Tier-2 sweep first, then tune over the union of recorded configs.

## Interface
```
run_tuning.py
  --traces-dir PATH        # dir of already-recorded trace sets (per config_hash subdirs)
                           # or a single trade_traces.jsonl for the baseline-only path
  --record                 # enable Tier-2: record each PreLLMGrid config first (cost-gated)
  --pre-llm-grid SPEC      # selects the PreLLMGrid (default: single baseline config)
  --train-days N --test-days N --step-days N
  --min-trades N
  --out PATH               # default workspace/walkforward_report.md
  --yes / non-interactive  # mirror Phase 3A spend-gate confirmation
```
**Default (no --record):** load baseline trade_traces.jsonl → `trace_sets = {baseline_hash: traces}` → make_folds → run_walkforward over post-LLM grid → report. No engine, no API key, no LLM calls. Must work offline.

**Tier-2 recording loop (--record):** cost math — `LLM cost = n_pre_llm_configs × ONE full-span backtest`. Folds do NOT multiply cost: a config's LLM output for a timestamp is fold-independent, so record each config ONCE over the whole span and slice per fold by timestamp.
Procedure:
1. For each config in PreLLMGrid: REUSE `TriggerCalibrator` to estimate `total_fires`; estimated cost = `total_fires × $0.003`. Sum across grid.
2. Print per-config and total estimate; **gate behind confirmation** mirroring the Phase 3A spend gate (refuse to spend without `--yes` or interactive confirm).
3. On confirm, for each config: REUSE `BacktestEngine` (idempotent resume) to write `workspace/tuning/runs/{config_hash}/trade_{alerts,traces}.jsonl`. Per-config directory prevents ledger collision; idempotent resume means a re-run skips already-recorded fires.
4. Load all `runs/{hash}/trade_traces.jsonl` → `trace_sets` keyed by hash → walk-forward as above.
Model selection via `KyrosAgentLoader.get_model_engine("trading", fallback_role="executor")`.

## Correctness Criteria
- Default path makes provably ZERO LLM calls (test asserts engine/loader never invoked, runs with no API key).
- Cost estimate printed and confirmed BEFORE any spend; declining aborts with zero spend.
- Each config recorded exactly once over full span; fold slicing is by timestamp only.
- Per-config output paths isolated by config_hash; no cross-config ledger contamination.
- Re-run with same configs resumes idempotently (no duplicate fires, no double spend).
- Determinism of the tuning stage given fixed recorded inputs.

## Test Strategy
- Integration (offline, no key): default path over a fixture trade_traces.jsonl produces a report; assert no LLM/engine call occurred.
- Unit: cost math = n_configs × calibrator estimate × $0.003; decline → abort, zero spend (mock calibrator + mock confirm).
- Unit: config_hash directory routing; two configs → two disjoint run dirs.
- Integration: idempotent resume — second --record run with a fully-recorded config performs zero new fires (mock engine to count).

## Dependencies
REUSES TriggerCalibrator (cost), BacktestEngine (recording + idempotent resume), KyrosAgentLoader (model selection), and the whole tuning stack above. Backtesting calibrator output-path parameterization is the permitted backtesting extension.

## Risks & Open Questions
- Whether the calibrator currently supports a parameterized output path / per-config run dir — the brief permits extending it; confirm the extension stays within its existing correctness contract.
- `--pre-llm-grid SPEC` format is unspecified; recommend a small named-preset registry rather than free-form CLI to keep configs reproducible and hashable.

---

## EXECUTOR CHECKLIST (ordering & gates)
1. **Resolve O0** (conviction default) by reading code + fixtures. Set the byte-preserving default.
2. Build `TradingConfig`; audit every literal call site (grep list); pass the **unedited Phase 1/2/3A suite** (P0 gate). STOP if any test needs behavioral edits.
3. Confirm trace_traces.jsonl schema field names; pin them in rescore.
4. Confirm PerformanceReport's exact expectancy field; make it the single source for objective.
5. Build params → rescore → objective → search → walkforward (leakage assertion is a release blocker) → report.
6. Build run_tuning default (zero-LLM) path; verify offline with no API key.
7. Build Tier-2 recording behind the cost gate last.

No implementation code is included by design. The two STOP-and-flag conditions (O0 non-preserving default; any Phase 2 behavioral test change) are hard prerequisites, not design problems to route around.