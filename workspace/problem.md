## Problem Statement


Project Kyros Phase 1: Sensory Foundation.

Build a complete ICT (Inner Circle Trader) detector library as standalone,
pure-Python modules in workspace/detectors/. The scope is the full ICT
concept surface — not limited to detectors already in ATLAS. The Planner
will research ICT theory, survey all implementable concepts, cross-reference
against the knowledge base, and produce a blueprint for every detector that
can be expressed deterministically on OHLCV candle data.

The ATLAS legacy codebase is available as a read-only correctness reference
for concepts it has already implemented. Nothing in the ported or newly
written code should reference ATLAS, IBKR, or any database.


## End Goal


A workspace/detectors/ package containing one module per ICT concept,
as determined by the Planner's research. At minimum this includes:

  workspace/detectors/__init__.py
  workspace/detectors/candles.py     — OHLCV ingestion and validation
  workspace/detectors/<concept>.py   — one file per detector concept

Each module exposes one public function:
  detect_<name>(candles: list[dict]) -> list[dict]

Candle dicts have keys: open, high, low, close, volume, timestamp.
Each detection dict includes at minimum: type, timestamp, and any
module-specific fields defined in the blueprint.

Accompanied by tests/test_<module>.py for every module.
All tests must pass with: pytest tests/


## Constraints


- Phase 1 only: no IBKR, no database, no broker, no order execution
- Detector modules use only pandas and numpy — no other external dependencies
- Public interface is list[dict] in, list[dict] out — no DataFrames at the boundary
- Every function is stateless: same input always produces same output
- Port or implement detection logic only — do not port ATLAS infrastructure
- Executor must write workspace/contract.md before writing any code
- Planner must blueprint all detectors in one pass — scope is the full ICT
  surface, not a subset
