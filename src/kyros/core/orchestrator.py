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
import uuid
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
        # Fence agent writes to the sanctioned surface only: workspace/ (all
        # phase code + artifacts) and tests/ (the suite the Executor authors).
        # Reads and pytest stay repo-wide. This is a hard guard so an Executor
        # cannot rewrite infrastructure (config/prompts.yaml, main.py, src/,
        # .kyros_state.json) the way a rogue model did during Phase 2B.
        self.toolkit = ExecutorToolkit(self.root, write_roots=["workspace", "tests"])

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
        print("\n🚀 Starting Orchestrator run...\n")
        # Reset round counter for a clean run. If run() is called multiple
        # times (e.g. for different tasks), each gets a fresh round budget.
        self.loader.reset_evaluator_round()
        total_tokens = 0
        trace_id = str(uuid.uuid4())
        log.info("Langfuse trace_id: %s", trace_id)

        # Persist the task definition so any agent can read it from disk.
        problem_content = self._format_problem(problem_statement, end_goal, constraints)
        self._write_artifact("problem.md", problem_content)

        # ── Planner: runs once, produces blueprint.md ────────────────────────────
        # Skip if blueprint.md already exists — allows resuming interrupted runs
        # without burning another Opus call.
        if (self._ws / "blueprint.md").exists():
            log.info("Resuming: blueprint.md already exists, skipping Planner.")
            print("✓ Blueprint already exists, skipping Planner.")
        else:
            print("📋 Calling Planner (this may take a minute)...")
            blueprint_response = self._run_planner(problem_statement, end_goal, constraints, trace_id)
            total_tokens += blueprint_response.usage.total_tokens
            self._write_artifact("blueprint.md", blueprint_response.content)
            print(f"✓ Planner complete ({blueprint_response.usage.total_tokens:,} tokens)")

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

            print(f"\n━━━ Round {rounds_taken + 1} ━━━")

            # ── Executor ───────────────────────────────────────────────────────
            # Skip the Executor if contract.md exists, no pending review, and
            # pytest actually passes right now. Run pytest directly rather than
            # trusting last_test_status (which can be stale after manual resets).
            contract_exists = (self._ws / "contract.md").exists()

            if contract_exists and review_content is None and self._pytest_passes():
                log.info("Resuming: pytest green and contract exists — skipping Executor.")
                print("✓ Executor skipped (contract exists, tests pass)")
            else:
                # Archive contract.md to artifacts/ before the Executor overwrites it.
                # Never delete LLM-produced artifacts — move them so history is preserved.
                self._archive_artifact("contract.md")
                print("🔨 Calling Executor...")
                executor_response = self._run_executor(
                    blueprint_content=self._read_artifact("blueprint.md"),
                    review_content=review_content,
                    trace_id=trace_id,
                )
                total_tokens += executor_response.usage.total_tokens
                print(f"✓ Executor complete ({executor_response.usage.total_tokens:,} tokens)")
                if not (self._ws / "contract.md").exists():
                    self._write_artifact("contract.md", executor_response.content)

            # ── Evaluator ──────────────────────────────────────────────────────
            # Archive review.md to artifacts/ before the Evaluator overwrites it.
            self._archive_artifact("review.md")
            self._cooldown_sleep()
            # Also runs as a tool-use agent: reads code, runs pytest, writes review.md.
            print("🔍 Calling Evaluator...")
            evaluator_response = self._run_evaluator(
                blueprint_content=self._read_artifact("blueprint.md"),
                contract_content=self._read_artifact("contract.md"),
                trace_id=trace_id,
            )
            total_tokens += evaluator_response.usage.total_tokens
            print(f"✓ Evaluator complete ({evaluator_response.usage.total_tokens:,} tokens)")
            if not (self._ws / "review.md").exists():
                self._write_artifact("review.md", evaluator_response.content)
            review_content = self._read_artifact("review.md")

            verdict = self._parse_verdict(review_content)
            self.loader.record_verdict(verdict)
            rounds_taken += 1

            # ── Route on verdict ───────────────────────────────────────────────
            if verdict == "APPROVE":
                print(f"\n✅ APPROVED in {rounds_taken} round(s)!")
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
                print(f"\n⚠️  ESCALATION required — see workspace/review.md")
                raise EscalationRequired(
                    reason="Evaluator returned ESCALATE verdict",
                    review_path=self._ws / "review.md",
                )

            # BLOCK or WARNING: send the Executor back with the review.
            # Incrementing the round counter here (after the Evaluator, before
            # the next Executor call) means the next get_agent_config("evaluator")
            # will inject the correct updated round number into the prompt.
            print(f"📝 Verdict: {verdict} — addressing findings in next round...")
            self.loader.increment_evaluator_round()

    # ── Agent call helpers ─────────────────────────────────────────────────────
    # Each helper builds the user message, makes the call, and wraps any
    # ProviderCallError in an OrchestratorError. The distinction matters:
    # ProviderCallError is an infra failure; OrchestratorError tells the
    # caller which agent step failed.

    def _run_planner(
        self, problem_statement: str, end_goal: str, constraints: str, trace_id: str | None = None
    ) -> ModelResponse:
        config = self.loader.get_agent_config("planner")
        messages = [
            {"role": "user", "content": self._format_problem(problem_statement, end_goal, constraints)}
        ]
        try:
            print("  → Waiting for Planner response...")
            return self.router.call(config, messages, name="planner", trace_id=trace_id)
        except RouterError as exc:
            raise OrchestratorError(f"Planner call failed: {exc}") from exc

    def _run_executor(
        self, blueprint_content: str, review_content: str | None, trace_id: str | None = None
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
            print("  → Waiting for Executor response (with tool use)...")
            return self.router.call_agentic(
                config, messages, TOOL_SCHEMAS, self.toolkit.dispatch,
                name="executor", trace_id=trace_id,
            )
        except RouterError as exc:
            raise OrchestratorError(f"Executor call failed: {exc}") from exc

    def _run_evaluator(
        self, blueprint_content: str, contract_content: str, trace_id: str | None = None
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
            print("  → Waiting for Evaluator response (with tool use)...")
            return self.router.call_agentic(
                config, messages, TOOL_SCHEMAS, self.toolkit.dispatch,
                name="evaluator", trace_id=trace_id,
            )
        except RouterError as exc:
            raise OrchestratorError(f"Evaluator call failed: {exc}") from exc

    # ── Artifact archival ─────────────────────────────────────────────────────

    def _archive_artifact(self, filename: str) -> None:
        """Move workspace/<filename> to artifacts/ with a timestamp suffix.

        Never deletes LLM-produced artifacts — preserves them for debugging
        and audit. No-op if the file does not exist.
        """
        src = self._ws / filename
        if not src.exists():
            return
        artifacts_dir = self.root / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        stem, suffix = filename.rsplit(".", 1) if "." in filename else (filename, "")
        ts = time.strftime("%Y%m%d_%H%M%S")
        uid = str(uuid.uuid4())[:6]
        dest_name = f"{stem}_{ts}_{uid}.{suffix}" if suffix else f"{stem}_{ts}_{uid}"
        dest = artifacts_dir / dest_name
        src.rename(dest)
        log.info("Archived %s → artifacts/%s", filename, dest_name)
        print(f"  → Archived {filename} to artifacts/{dest_name}")

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