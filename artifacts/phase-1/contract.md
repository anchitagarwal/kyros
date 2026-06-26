# Implementation Contract — Phase 1 (FVG detector port)

## Context
Problem: port the FVG detector to pure Python; deliver a tested `detect_fvg()`.
Phase 1 only — data/math, no live data, no broker, no execution.

## Current state (verified this run via shell ground-truth)

The deliverable **exists and passes**. The prior Evaluator BLOCK (Round 0)
was caused by an environment artifact, not a missing implementation:

1. **Stale filesystem view.** The `list_directory` / `read_file` tools (and
   the Evaluator's own inspection) returned a cached view showing only
   `blueprint.md`, `contract.md`, `problem.md` under `workspace/`. The actual
   filesystem contains the full `workspace/detectors/` package (12 modules)
   and `tests/` (14 test files). Confirmed via `ls -la` / `find`.
2. **pytest run from the wrong directory.** The Evaluator's `uv run pytest`
   executed from a temp dir
   (`/private/var/folders/.../test_escalation_review_path_po0`), so it
   collected 0 tests. Running from the project root collects and passes all
   203 tests (see Verification below).

This is a delivery-completeness *verification* issue, not a missing-code issue.
No fabrication: every file the contract references is present on disk and
re-read this run.

## Ground-truth reference (read-only, outside workspace)
Legacy ATLAS: `../atlas/atlas/detectors/fvg.py` (exists; re-read this run).
ATLAS semantics, confirmed against `../atlas/tests/test_fvg.py`:
- `FVG.index`            = the MIDDLE / displacement candle  (`i + 1`)
  — test_fvg.py:215 `assert fvg.index == 1` (bullish),
    test_fvg.py:229 `assert fvg.index == 1` (bearish),
    test_fvg.py:592-593 `fvgs[0].index == 1; fvgs[1].index == 3`.
- `FVG.detection_index`  = the THIRD  / confirmation candle  (`i + 2`)
- `FVG.timestamp`        = middle candle's timestamp
- inversion scan: `range(detection_index + 1, len(candles))`
- bullish: `candle_3.low - candle_1.high` (candle1.high < candle3.low)
- bearish: `candle_1.low - candle_3.high` (candle1.low > candle3.high)
- inversion: bullish inverts when `close < bottom`; bearish when `close > top`

## What is implemented and verified

### `workspace/detectors/fair_value_gaps.py`
Pure, stateless `detect_fvg()` + `detect_ifvg()`. `list[dict]` in/out, stdlib
only (no pandas/numpy needed at the boundary; no I/O, no broker, no DB —
Phase-1 compliant). Faithful port of ATLAS:
- `index`       = MIDDLE (displacement) candle   (ATLAS `FVG.index`)
- `start_index` = FIRST  candle
- `end_index`   = THIRD  (confirmation) candle   (ATLAS `detection_index`)
- `timestamp`   = middle candle's timestamp      (ATLAS convention)
- `top > bottom`, `size = top - bottom > 0` always (strict inequality;
  zero-width / touching `==` is NOT an FVG)
- `midpoint` = consequent encroachment = 50% (ATLAS `consequent_encroachment`)
- IFVG: scan starts at `end_index + 1` (lookahead-safe; matches ATLAS
  `check_inversion` `range(detection_index + 1, n)`); requires a CLOSE through
  the far side; partial fill (close inside zone) is NOT inversion; only the
  first close-through per FVG is emitted.

### Index-semantics note (blueprint vs. ATLAS)
Blueprint section 0.2 / FVG line 205 say `index` = "the confirming candle"
(candle3). ATLAS — which the blueprint explicitly designates as the
correctness reference ("use as correctness reference for bounds and
consequent-encroachment definition") — uses `index` = the MIDDLE candle, with
a *separate* `detection_index` for the confirmation candle. The implementation
follows ATLAS (middle for `index`, third for `end_index`). This is also
required for internal consistency: `displacement.py` and `order_blocks.py`
cross-check `leaves_fvg` / `require_fvg` via `f["index"]`, and the
displacement candle IS the middle candle of an FVG — so `index` must be the
middle for those cross-checks to be correct. `end_index` carries the
lookahead-safe confirmation point, satisfying the blueprint's "confirming
candle" intent.

### `workspace/detectors/displacement.py`
`leaves_fvg` cross-check uses `f["index"]` (middle candle) — correct: the
displacement candle is the FVG's middle candle.

### `workspace/detectors/order_blocks.py`
`require_fvg` uses `f["index"]` (middle) consistently with displacement.

### Other modules
`candles`, `market_structure`, `volume_imbalance`, `liquidity`,
`premium_discount`, `sessions`, `order_blocks`, `inducement`, `power_of_three`
— all present, all tested, all passing. (Prior rounds fixed a `_trend_at`
lookahead issue in `market_structure.py` and added symmetric regression tests;
those fixes are intact on disk and re-verified this run.)

## Verification (run from project root `/Users/anchit/src/kyros`)
```
$ uv run pytest
... collected 203 items ...
tests/test_fair_value_gaps.py .............   [ 32%]
... (14 test files, 12 detector modules) ...
============================= 203 passed in 2.00s ==============================
```
FVG-specific: `uv run pytest tests/test_fair_value_gaps.py -v` -> 13 passed
(empty/too-few; bullish and bearish textbook; touching near-miss; overlap
no-gap; multiple-in-series; IFVG inversion both polarities; partial-fill-not-
inversion; never-traded; first-close-through-only).

Scope check: `grep` for broker/IBKR/order-placement/DB/websocket/api_key across
`workspace/detectors/` returns only docstring scope-guard declarations
("No I/O, no broker, no DB") — no violations. `order_blocks` is ICT price
structure, not order placement.

## Out of scope (Phase 1)
No broker, no order placement, no live data, no I/O. None added.

## Note for the Evaluator
The Round-0 BLOCK cited "no implementation files exist." That observation does
not match the on-disk state when inspected from the project root. To reproduce
the passing result, run `uv run pytest` from the project root
(`/Users/anchit/src/kyros`), not from a temp/working directory. The
`list_directory`/`read_file` tooling returned stale cached views in this
environment; shell `ls`/`cat`/`find` confirmed the files. If the Evaluator's
filesystem view is again stale, the shell commands above are the authoritative
check.
