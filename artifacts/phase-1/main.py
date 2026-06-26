"""
main.py — Kyros Phase 1 entry point.

Run from the repo root:
    uv run python main.py

PROBLEM_STATEMENT / END_GOAL / CONSTRAINTS are the problem instance —
what to build, where, and what hard limits apply. They change per task.

Agent behavior (how the Planner researches, what ICT concepts to cover,
how the Executor implements, what the Evaluator checks) lives in
config/prompts.yaml — the single source of truth for domain knowledge.
"""

import logging
import sys

from dotenv import load_dotenv

load_dotenv()

from kyros.core.orchestrator import EscalationRequired, Orchestrator, OrchestratorError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kyros")


# ── Phase 1 Problem Definition ─────────────────────────────────────────────────

PROBLEM_STATEMENT = """
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
"""

END_GOAL = """
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
"""

CONSTRAINTS = """
- Phase 1 only: no IBKR, no database, no broker, no order execution
- Detector modules use only pandas and numpy — no other external dependencies
- Public interface is list[dict] in, list[dict] out — no DataFrames at the boundary
- Every function is stateless: same input always produces same output
- Port or implement detection logic only — do not port ATLAS infrastructure
- Executor must write workspace/contract.md before writing any code
- Planner must blueprint all detectors in one pass — scope is the full ICT
  surface, not a subset
"""


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("Kyros — Phase 1: Sensory Foundation")
    log.info("─" * 48)

    orch = Orchestrator()
    log.info("Workspace : %s", orch._ws)
    log.info("Scope     : full ICT detector surface (Planner determines)")
    log.info("─" * 48)

    try:
        result = orch.run(
            problem_statement=PROBLEM_STATEMENT,
            end_goal=END_GOAL,
            constraints=CONSTRAINTS,
        )

        log.info("─" * 48)
        log.info("✓  APPROVED in %d round(s)", result.rounds_taken)
        log.info("   Blueprint : %s", result.blueprint_path)
        log.info("   Review    : %s", result.review_path)
        log.info("   Tokens    : %d total", result.total_tokens)

    except EscalationRequired as e:
        log.info("─" * 48)
        log.warning("⚠  ESCALATED — human review required")
        log.warning("   Reason : %s", e.reason)
        log.warning("   Review : %s", e.review_path)
        log.warning("   Open the review file, address findings, then re-run.")
        sys.exit(1)

    except OrchestratorError as e:
        log.info("─" * 48)
        log.error("✗  Infrastructure error: %s", e)
        log.error("   Check your API keys and network, then re-run.")
        sys.exit(2)


if __name__ == "__main__":
    main()