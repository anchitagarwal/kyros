"""
orchestrator.py — the main Planner → Executor → Evaluator loop.

─────────────────────────────────────────────────────────────────────────────
CONCEPT: The Orchestrator's single job
─────────────────────────────────────────────────────────────────────────────
The Orchestrator owns the workflow and file I/O. That's it.

It does NOT know about providers, model strings, or HTTP retry backoff
— that's ModelRouter's job. It does NOT manage prompt text or round
counters — that's KyrosAgentLoader's job.

Its one responsibility: decide what gets called next, in what order,
with what context, and when to stop.

─────────────────────────────────────────────────────────────────────────────
CONCEPT: Why file-based handoff?
─────────────────────────────────────────────────────────────────────────────
The alternative is passing full conversation history between agents:
  planner_messages → executor_messages → evaluator_messages → ...

That approach has two problems:
  1. Context windows: each agent accumulates the entire chain. By round 2
     the Evaluator is reading the Planner's first draft, the Executor's
     first contract, the first review, the second contract, the second
     review... Token cost explodes.
  2. Debuggability: when something goes wrong, you're staring at a giant
     messages[] blob. With file handoff, you just open workspace/review.md.

File-based handoff gives each agent a clean context containing only what
it needs: the artifact it's reacting to, not the full chat history.
The workspace/ directory is the source of truth. An agent's output is
always a file, never an in-memory string passed directly to the next call.

─────────────────────────────────────────────────────────────────────────────
CONCEPT: The retry loop and escalation
─────────────────────────────────────────────────────────────────────────────
The Executor → Evaluator pair runs in a while loop capped by the round
counter in KyrosAgentLoader. The loop terminates in one of three ways:

  APPROVE  → normal exit, returns OrchestratorResult(status="APPROVED")
  ESCALATE → raised immediately as EscalationRequired (either the Evaluator
             returned it explicitly, or the round cap was hit)
  Exception → a ProviderCallError became an OrchestratorError (infra failure)

WARNING is treated the same as BLOCK: the Executor must address findings.
This matches the ECC code-review model in prompts.yaml — a warning is not
an approval; it's a softer BLOCK.

─────────────────────────────────────────────────────────────────────────────
CONCEPT: where should_escalate() is checked
─────────────────────────────────────────────────────────────────────────────
The round check happens at the TOP of each loop iteration, before the
Executor runs. This means:
  - The Executor is called exactly max_evaluator_rounds times before
    escalation (not one extra time).
  - The KyrosAgentLoader's FINAL ROUND prompt injection (which tells the
    Evaluator to self-escalate) fires at the same threshold as this check —
    belt-and-suspenders.

File layout under workspace_root/workspace/:
  problem.md    — problem context snapshot (written at run() start)
  blueprint.md  — Planner output
  contract.md   — Executor pre-coding commitment (overwritten each round)
  review.md     — Evaluator findings + verdict (overwritten each round)
"""

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

from kyros.core.agent_loader import KyrosAgentLoader
from kyros.core.executor_tools import TOOL_SCHEMAS, ExecutorToolkit
from kyros.core.model_router import ModelResponse, ModelRouter, ProviderCallError, RouterError, TokenUsage


# ── Verdict parsing ───────────────────────────────────────────────────────────
# The Evaluator is instructed to end its response with a structured line:
#   VERDICT: APPROVE  (or WARNING / BLOCK / ESCALATE)
# We search the full response text with a regex so minor formatting
# variations (trailing space, mixed case) don't cause false BLOCK.

_VERDICT_RE = re.compile(r"VERDICT:\s*(APPROVE|WARNING|BLOCK|ESCALATE)", re.IGNORECASE)


# ── Result and exception types ────────────────────────────────────────────────

@dataclass
class OrchestratorResult:
    """Returned when the loop terminates with an APPROVE verdict."""
    status: str          # "APPROVED"
    blueprint_path: Path
    review_path: Path
    rounds_taken: int    # How many Evaluator rounds it took
    total_tokens: int = 0


class EscalationRequired(Exception):
    """
    Raised when the loop exits without APPROVE:
      - The Evaluator explicitly returned ESCALATE
      - The round counter hit max_evaluator_rounds

    Callers should read .review_path for the last Evaluator findings and
    route to a human operator.
    """
    def __init__(self, reason: str, review_path: Path):
        self.reason = reason
        self.review_path = review_path
        super().__init__(f"Human operator required. Reason: {reason}. Review at: {review_path}")


class OrchestratorError(Exception):
    """
    Raised for unrecoverable infrastructure failures (e.g. provider API
    completely unreachable after all retries). Distinct from EscalationRequired
    — these are not routing decisions, they're infra failures.
    """


# ── Orchestrator ──────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Runs the Planner → Executor → Evaluator pipeline for one task.

    Designed to be instantiated once and reused. Holds no per-run state
    in instance variables — all state lives in .kyros_state.json (via
    KyrosAgentLoader) and workspace/ files.

    Example:
        orch = Orchestrator()
        try:
            result = orch.run(
                problem_statement="Port the FVG detector to pure Python",
                end_goal="A tested detect_fvg() function in workspace/detectors/",
                constraints="Phase 1 only. No IBKR or live data. No execution code.",
            )
            print(f"Done in {result.rounds_taken} round(s). Blueprint at {result.blueprint_path}")
        except EscalationRequired as e:
            print(f"Needs human review: {e.review_path}")
    """

    def __init__(self, workspace_root: str = "."):
        self.root = Path(workspace_root)
        self._ws = self.root / "workspace"
        self._ws.mkdir(parents=True, exist_ok=True)
        self.loader = KyrosAgentLoader(workspace_root)
        self.router = ModelRouter()
        self.toolkit = ExecutorToolkit(self.root)

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        problem_statement: str,
        end_goal: str,
        constraints: str,
    ) -> OrchestratorResult:
        """
        Execute the full Planner → (Executor → Evaluator)* loop.

        Raises:
            EscalationRequired: Evaluator returned ESCALATE, or max rounds hit.
            OrchestratorError: An LLM provider call failed unrecoverably.
        """
        # Reset round counter for a clean run. If run() is called multiple
        # times (e.g. for different tasks), each gets a fresh round budget.
        self.loader.reset_evaluator_round()
        total_tokens = 0

        # Persist the task definition so any agent can read it from disk.
        problem_content = self._format_problem(problem_statement, end_goal, constraints)
        self._write_artifact("problem.md", problem_content)

        # ── Planner: runs once, produces blueprint.md ────────────────────────────
        # Skip if blueprint.md already exists — allows resuming interrupted runs
        # without burning another Opus call.
        if (self._ws / "blueprint.md").exists():
            log.info("Resuming: blueprint.md already exists, skipping Planner.")
        else:
            blueprint_response = self._run_planner(problem_statement, end_goal, constraints)
            total_tokens += blueprint_response.usage.total_tokens
            self._write_artifact("blueprint.md", blueprint_response.content)

        # ── Executor → Evaluator loop ──────────────────────────────────────────
        # On resume, carry forward any existing review so the Executor knows
        # what still needs fixing rather than starting blind.
        review_content: str | None = None
        if (self._ws / "review.md").exists():
            review_content = self._read_artifact("review.md")
        rounds_taken = 0

        while True:
            # Hard escalation guard: checked before every Executor call.
            # Ensures the Executor runs at most max_evaluator_rounds times.
            if self.loader.should_escalate():
                raise EscalationRequired(
                    reason=(
                        f"Reached maximum of "
                        f"{self.loader.load_state()['max_evaluator_rounds']} "
                        f"evaluator rounds without APPROVE"
                    ),
                    review_path=self._ws / "review.md",
                )

            # ── Executor ───────────────────────────────────────────────────────
            # Skip the Executor if contract.md exists, no pending review, and
            # pytest actually passes right now. Run pytest directly rather than
            # trusting last_test_status (which can be stale after manual resets).
            contract_exists = (self._ws / "contract.md").exists()

            if contract_exists and review_content is None and self._pytest_passes():
                log.info("Resuming: pytest green and contract exists — skipping Executor.")
            else:
                # Clear contract.md so the fallback always captures this round's output.
                # In production the Executor re-writes it via tools; in tests the mock
                # doesn't touch files so the fallback must run fresh each round.
                (self._ws / "contract.md").unlink(missing_ok=True)
                executor_response = self._run_executor(
                    blueprint_content=self._read_artifact("blueprint.md"),
                    review_content=review_content,
                )
                total_tokens += executor_response.usage.total_tokens
                if not (self._ws / "contract.md").exists():
                    self._write_artifact("contract.md", executor_response.content)

            # ── Evaluator ──────────────────────────────────────────────────────
            # Clear review.md before each evaluator run so the fallback always
            # captures this round's verdict rather than returning a stale one.
            (self._ws / "review.md").unlink(missing_ok=True)
            self._cooldown_sleep()
            # Also runs as a tool-use agent: reads code, runs pytest, writes review.md.
            evaluator_response = self._run_evaluator(
                blueprint_content=self._read_artifact("blueprint.md"),
                contract_content=self._read_artifact("contract.md"),
            )
            total_tokens += evaluator_response.usage.total_tokens
            if not (self._ws / "review.md").exists():
                self._write_artifact("review.md", evaluator_response.content)
            review_content = self._read_artifact("review.md")

            verdict = self._parse_verdict(review_content)
            self.loader.record_verdict(verdict)
            rounds_taken += 1

            # ── Route on verdict ───────────────────────────────────────────────
            if verdict == "APPROVE":
                return OrchestratorResult(
                    status="APPROVED",
                    blueprint_path=self._ws / "blueprint.md",
                    review_path=self._ws / "review.md",
                    rounds_taken=rounds_taken,
                    total_tokens=total_tokens,
                )

            if verdict == "ESCALATE":
                # The Evaluator decided it cannot approve even with another
                # iteration. Immediate escalation — don't consume another round.
                raise EscalationRequired(
                    reason="Evaluator returned ESCALATE verdict",
                    review_path=self._ws / "review.md",
                )

            # BLOCK or WARNING: send the Executor back with the review.
            # Incrementing the round counter here (after the Evaluator, before
            # the next Executor call) means the next get_agent_config("evaluator")
            # will inject the correct updated round number into the prompt.
            self.loader.increment_evaluator_round()

    # ── Agent call helpers ─────────────────────────────────────────────────────
    # Each helper builds the user message, makes the call, and wraps any
    # ProviderCallError in an OrchestratorError. The distinction matters:
    # ProviderCallError is an infra failure; OrchestratorError tells the
    # caller which agent step failed.

    def _run_planner(
        self, problem_statement: str, end_goal: str, constraints: str
    ) -> ModelResponse:
        config = self.loader.get_agent_config("planner")
        messages = [
            {"role": "user", "content": self._format_problem(problem_statement, end_goal, constraints)}
        ]
        try:
            return self.router.call(config, messages)
        except RouterError as exc:
            raise OrchestratorError(f"Planner call failed: {exc}") from exc

    def _run_executor(
        self, blueprint_content: str, review_content: str | None
    ) -> ModelResponse:
        config = self.loader.get_agent_config("executor")

        body = f"## Blueprint\n\n{blueprint_content}"
        if review_content:
            body += (
                "\n\n## Evaluator Review\n\n"
                "Address all CRITICAL and HIGH severity findings below "
                "before resubmitting.\n\n"
                f"{review_content}"
            )

        messages = [{"role": "user", "content": body}]
        try:
            return self.router.call_agentic(
                config, messages, TOOL_SCHEMAS, self.toolkit.dispatch
            )
        except RouterError as exc:
            raise OrchestratorError(f"Executor call failed: {exc}") from exc

    def _run_evaluator(
        self, blueprint_content: str, contract_content: str
    ) -> ModelResponse:
        config = self.loader.get_agent_config("evaluator")

        body = (
            f"## Blueprint\n\n{blueprint_content}\n\n"
            f"## Executor Contract\n\n{contract_content}\n\n"
            "Conduct your audit using your available tools. Read the implemented "
            "code, run pytest, and write your full findings to workspace/review.md. "
            "End your review with: VERDICT: <APPROVE|WARNING|BLOCK|ESCALATE>"
        )

        messages = [{"role": "user", "content": body}]
        try:
            return self.router.call_agentic(
                config, messages, TOOL_SCHEMAS, self.toolkit.dispatch
            )
        except RouterError as exc:
            raise OrchestratorError(f"Evaluator call failed: {exc}") from exc

    # ── Cooldown sleep ────────────────────────────────────────────────────────

    def _cooldown_sleep(self, seconds: int = 15) -> None:
        """Pause between Executor and Evaluator to avoid rate-limit cascades.
        Extracted to a method so tests can monkeypatch it to a no-op."""
        log.info("Pausing %ds before Evaluator to avoid rate-limit cascade...", seconds)
        time.sleep(seconds)

    # ── Pytest check ──────────────────────────────────────────────────────────

    def _pytest_passes(self) -> bool:
        """Return True if pytest exits 0 right now. Used to skip the Executor on resume."""
        import subprocess
        result = subprocess.run(
            ["uv", "run", "pytest", "--tb=no", "-q"],
            capture_output=True,
            cwd=str(self.root),
        )
        passes = result.returncode == 0
        log.info("pytest check: %s", "PASS" if passes else "FAIL")
        return passes

    # ── Verdict parsing ────────────────────────────────────────────────────────

    def _parse_verdict(self, response_text: str) -> str:
        """
        Extract the verdict token from the Evaluator's response.

        Searches anywhere in the response for "VERDICT: <token>" — tolerant of
        leading/trailing whitespace and mixed case.

        Defaults to BLOCK on no match. Conservative: we never silently approve
        a malformed response. If the Evaluator failed to include a verdict line,
        that's a finding, not a pass.
        """
        match = _VERDICT_RE.search(response_text)
        return match.group(1).upper() if match else "BLOCK"

    # ── File I/O ───────────────────────────────────────────────────────────────

    def _write_artifact(self, filename: str, content: str) -> None:
        """Write a workspace artifact. Overwrites on retry rounds."""
        (self._ws / filename).write_text(content, encoding="utf-8")

    def _read_artifact(self, filename: str) -> str:
        """Read a workspace artifact. Raises OrchestratorError if missing."""
        path = self._ws / filename
        if not path.exists():
            raise OrchestratorError(
                f"Expected workspace artifact not found: {path}. "
                "Check that the previous agent step completed successfully."
            )
        return path.read_text(encoding="utf-8")

    # ── Formatting ─────────────────────────────────────────────────────────────

    @staticmethod
    def _format_problem(problem_statement: str, end_goal: str, constraints: str) -> str:
        """Canonical format for the problem context passed to the Planner."""
        return (
            f"## Problem Statement\n\n{problem_statement}\n\n"
            f"## End Goal\n\n{end_goal}\n\n"
            f"## Constraints\n\n{constraints}"
        )