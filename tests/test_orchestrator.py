"""
test_orchestrator.py

─────────────────────────────────────────────────────────────────────────────
CONCEPT: Testing a stateful loop
─────────────────────────────────────────────────────────────────────────────
The Orchestrator has two sources of state:
  1. The filesystem (workspace/*.md files)
  2. .kyros_state.json (round counter, last review status)

Both need to be isolated between tests. We use:
  - tmp_path for the workspace and state file (pytest built-in, safe)
  - patch.object on isolated_orchestrator.router to control LLM responses

This lets us test the full loop including file I/O without mocking the
filesystem itself — real files get written to a temp directory, which is
exactly what the Orchestrator does in production.

─────────────────────────────────────────────────────────────────────────────
PATTERN: Controlling a multi-call sequence with side_effect
─────────────────────────────────────────────────────────────────────────────
The Planner uses router.call(); Executor and Evaluator use router.call_agentic().
Tests therefore patch both methods separately:
  mock_call.side_effect      = [planner_reply]
  mock_agentic.side_effect   = [executor_reply, evaluator_reply, ...]

_pytest_passes() is patched to return False in the fixture so it never
spawns a real subprocess (the tmp_path workspace has no tests).
"""

import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from kyros.core.model_router import ModelResponse, TokenUsage
from kyros.core.orchestrator import (
    EscalationRequired,
    Orchestrator,
    OrchestratorError,
    OrchestratorResult,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def isolated_orchestrator(tmp_path, monkeypatch):
    """
    Orchestrator wired to a fully isolated tmp workspace.
    State file and prompts are copied from the real repo; no cross-test bleed.

    Stubs:
      _pytest_passes  → False  (no real code in tmp; Executor skip never fires)
      _cooldown_sleep → no-op  (15s sleep would make the suite take minutes)

    max_evaluator_rounds is pinned to 3 regardless of the live state.json so
    tests are not affected by in-progress production runs changing that field.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    shutil.copy(REPO_ROOT / "config" / "prompts.yaml", config_dir / "prompts.yaml")

    state = json.loads((REPO_ROOT / ".kyros_state.json").read_text())
    state["evaluator_round"] = 0
    state["max_evaluator_rounds"] = 3
    (tmp_path / ".kyros_state.json").write_text(json.dumps(state))

    orch = Orchestrator(workspace_root=str(tmp_path))
    monkeypatch.setattr(orch, "_pytest_passes", lambda: False)
    monkeypatch.setattr(orch, "_cooldown_sleep", lambda: None)
    return orch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resp(content: str, tokens: int = 30) -> ModelResponse:
    """Minimal ModelResponse for test scripting."""
    return ModelResponse(
        content=content,
        provider="anthropic",
        model="claude-opus-4-7",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=tokens - 10, total_tokens=tokens),
    )


@contextmanager
def _mock_agents(orch, planner_resps, agentic_resps):
    """
    Patch router.call (Planner) and router.call_agentic (Executor + Evaluator).
    Yields (mock_call, mock_agentic) for tests that need to inspect call args.
    """
    with patch.object(orch.router, "call") as mc, \
         patch.object(orch.router, "call_agentic") as ma:
        mc.side_effect = planner_resps
        ma.side_effect = agentic_resps
        yield mc, ma


PS = "Port the FVG detector to pure Python"
EG = "A tested detect_fvg() function"
CN = "Phase 1 only. No live data."


# ── Happy path ────────────────────────────────────────────────────────────────

def test_run_returns_approved_on_single_round_approve(isolated_orchestrator):
    """Ideal path: Planner → Executor → Evaluator APPROVE in one round."""
    with _mock_agents(
        isolated_orchestrator,
        planner_resps=[_resp("## Blueprint\nPhase 1 scope.")],
        agentic_resps=[
            _resp("## Contract\nI will implement X."),  # executor
            _resp("Looks good.\nVERDICT: APPROVE"),     # evaluator
        ],
    ):
        result = isolated_orchestrator.run(PS, EG, CN)

    assert isinstance(result, OrchestratorResult)
    assert result.status == "APPROVED"
    assert result.rounds_taken == 1


def test_run_total_tokens_aggregated_across_all_agents(isolated_orchestrator):
    """Token counts from all three agents should be summed."""
    with _mock_agents(
        isolated_orchestrator,
        planner_resps=[_resp("Blueprint", tokens=100)],
        agentic_resps=[
            _resp("Contract", tokens=200),
            _resp("VERDICT: APPROVE", tokens=50),
        ],
    ):
        result = isolated_orchestrator.run(PS, EG, CN)

    assert result.total_tokens == 350


# ── File I/O ──────────────────────────────────────────────────────────────────

def test_blueprint_written_to_workspace(isolated_orchestrator):
    with _mock_agents(
        isolated_orchestrator,
        planner_resps=[_resp("## Blueprint\nPhase 1 scope.")],
        agentic_resps=[_resp("## Contract\nI will build it."), _resp("VERDICT: APPROVE")],
    ):
        result = isolated_orchestrator.run(PS, EG, CN)

    assert result.blueprint_path.exists()
    assert "Phase 1 scope" in result.blueprint_path.read_text()


def test_review_written_to_workspace(isolated_orchestrator):
    with _mock_agents(
        isolated_orchestrator,
        planner_resps=[_resp("Blueprint.")],
        agentic_resps=[_resp("Contract."), _resp("No issues found.\nVERDICT: APPROVE")],
    ):
        result = isolated_orchestrator.run(PS, EG, CN)

    assert result.review_path.exists()
    assert "No issues found" in result.review_path.read_text()


def test_problem_md_written_at_start(isolated_orchestrator):
    """problem.md is a snapshot of the task definition for auditability."""
    with _mock_agents(
        isolated_orchestrator,
        planner_resps=[_resp("Blueprint.")],
        agentic_resps=[_resp("Contract."), _resp("VERDICT: APPROVE")],
    ):
        isolated_orchestrator.run(PS, EG, CN)

    problem_file = isolated_orchestrator._ws / "problem.md"
    assert problem_file.exists()
    content = problem_file.read_text()
    assert PS in content
    assert EG in content
    assert CN in content


# ── Planner call shape ────────────────────────────────────────────────────────

def test_planner_receives_all_three_problem_fields(isolated_orchestrator):
    """The Planner's user message must include PS, EG, and constraints."""
    with _mock_agents(
        isolated_orchestrator,
        planner_resps=[_resp("Blueprint.")],
        agentic_resps=[_resp("Contract."), _resp("VERDICT: APPROVE")],
    ) as (mock_call, _):
        isolated_orchestrator.run(PS, EG, CN)

    planner_messages = mock_call.call_args_list[0][0][1]
    content = planner_messages[0]["content"]
    assert PS in content
    assert EG in content
    assert CN in content


# ── Executor call shape ───────────────────────────────────────────────────────

def test_executor_first_call_receives_blueprint(isolated_orchestrator):
    with _mock_agents(
        isolated_orchestrator,
        planner_resps=[_resp("The blueprint text.")],
        agentic_resps=[_resp("Contract."), _resp("VERDICT: APPROVE")],
    ) as (_, mock_agentic):
        isolated_orchestrator.run(PS, EG, CN)

    # call_agentic[0] = first Executor call; [0][1] = messages positional arg
    executor_messages = mock_agentic.call_args_list[0][0][1]
    assert "The blueprint text." in executor_messages[0]["content"]


def test_executor_first_call_has_no_review_section(isolated_orchestrator):
    """On the first attempt there is no review — the message should not mention it."""
    with _mock_agents(
        isolated_orchestrator,
        planner_resps=[_resp("Blueprint.")],
        agentic_resps=[_resp("Contract."), _resp("VERDICT: APPROVE")],
    ) as (_, mock_agentic):
        isolated_orchestrator.run(PS, EG, CN)

    executor_messages = mock_agentic.call_args_list[0][0][1]
    assert "Evaluator Review" not in executor_messages[0]["content"]


def test_executor_retry_receives_review_content(isolated_orchestrator):
    """
    On retry, the Executor's user message must include the Evaluator's review
    so it knows exactly which findings to address.
    """
    review_text = "CRITICAL: missing return type annotation.\nVERDICT: BLOCK"

    with _mock_agents(
        isolated_orchestrator,
        planner_resps=[_resp("Blueprint.")],
        agentic_resps=[
            _resp("Contract v1."),   # executor round 1  → agentic[0]
            _resp(review_text),      # evaluator round 1 → agentic[1]
            _resp("Contract v2."),   # executor round 2  → agentic[2]
            _resp("VERDICT: APPROVE"),
        ],
    ) as (_, mock_agentic):
        isolated_orchestrator.run(PS, EG, CN)

    executor_retry_messages = mock_agentic.call_args_list[2][0][1]
    retry_content = executor_retry_messages[0]["content"]
    assert "CRITICAL: missing return type annotation" in retry_content
    assert "Evaluator Review" in retry_content


# ── Evaluator call shape ──────────────────────────────────────────────────────

def test_evaluator_receives_blueprint_and_contract(isolated_orchestrator):
    with _mock_agents(
        isolated_orchestrator,
        planner_resps=[_resp("## Blueprint\nThe plan.")],
        agentic_resps=[
            _resp("## Contract\nThe commitment."),  # executor → agentic[0]
            _resp("VERDICT: APPROVE"),              # evaluator → agentic[1]
        ],
    ) as (_, mock_agentic):
        isolated_orchestrator.run(PS, EG, CN)

    evaluator_messages = mock_agentic.call_args_list[1][0][1]
    content = evaluator_messages[0]["content"]
    assert "The plan." in content
    assert "The commitment." in content


# ── Retry loop logic ──────────────────────────────────────────────────────────

def test_block_verdict_causes_executor_retry(isolated_orchestrator):
    """BLOCK → Executor must be called a second time."""
    with _mock_agents(
        isolated_orchestrator,
        planner_resps=[_resp("Blueprint.")],
        agentic_resps=[
            _resp("Contract v1."),
            _resp("Issues found.\nVERDICT: BLOCK"),
            _resp("Contract v2."),
            _resp("VERDICT: APPROVE"),
        ],
    ) as (_, mock_agentic):
        result = isolated_orchestrator.run(PS, EG, CN)

    assert result.rounds_taken == 2
    assert mock_agentic.call_count == 4  # 2× executor + 2× evaluator


def test_warning_verdict_retries_like_block(isolated_orchestrator):
    """
    WARNING is not an approval. The Executor must address findings just as
    it would for a BLOCK. No silent passes on warnings.
    """
    with _mock_agents(
        isolated_orchestrator,
        planner_resps=[_resp("Blueprint.")],
        agentic_resps=[
            _resp("Contract v1."),
            _resp("Minor issues.\nVERDICT: WARNING"),
            _resp("Contract v2."),
            _resp("VERDICT: APPROVE"),
        ],
    ) as (_, mock_agentic):
        result = isolated_orchestrator.run(PS, EG, CN)

    assert result.rounds_taken == 2
    assert mock_agentic.call_count == 4


def test_two_blocks_then_approve(isolated_orchestrator):
    with _mock_agents(
        isolated_orchestrator,
        planner_resps=[_resp("Blueprint.")],
        agentic_resps=[
            _resp("Contract v1."), _resp("VERDICT: BLOCK"),
            _resp("Contract v2."), _resp("VERDICT: BLOCK"),
            _resp("Contract v3."), _resp("VERDICT: APPROVE"),
        ],
    ):
        result = isolated_orchestrator.run(PS, EG, CN)

    assert result.rounds_taken == 3
    assert result.status == "APPROVED"


# ── Escalation paths ──────────────────────────────────────────────────────────

def test_escalate_verdict_raises_immediately(isolated_orchestrator):
    """
    An explicit ESCALATE verdict from the Evaluator should raise without
    consuming another round — the Evaluator has decided this needs a human.
    """
    with _mock_agents(
        isolated_orchestrator,
        planner_resps=[_resp("Blueprint.")],
        agentic_resps=[
            _resp("Contract."),
            _resp("Fundamental design flaw.\nVERDICT: ESCALATE"),
        ],
    ):
        with pytest.raises(EscalationRequired, match="ESCALATE verdict"):
            isolated_orchestrator.run(PS, EG, CN)


def test_escalation_after_max_rounds(isolated_orchestrator):
    """
    If BLOCK persists for max_evaluator_rounds rounds, the Orchestrator
    should raise EscalationRequired without calling the Evaluator again.
    """
    max_rounds = isolated_orchestrator.loader.load_state()["max_evaluator_rounds"]

    agentic_resps = []
    for _ in range(max_rounds):
        agentic_resps.append(_resp("Contract."))
        agentic_resps.append(_resp("Still broken.\nVERDICT: BLOCK"))

    with _mock_agents(
        isolated_orchestrator,
        planner_resps=[_resp("Blueprint.")],
        agentic_resps=agentic_resps,
    ):
        with pytest.raises(EscalationRequired, match="maximum"):
            isolated_orchestrator.run(PS, EG, CN)


def test_escalation_review_path_points_to_workspace(isolated_orchestrator):
    """EscalationRequired.review_path must be a real path in the workspace."""
    with _mock_agents(
        isolated_orchestrator,
        planner_resps=[_resp("Blueprint.")],
        agentic_resps=[_resp("Contract."), _resp("VERDICT: ESCALATE")],
    ):
        with pytest.raises(EscalationRequired) as exc_info:
            isolated_orchestrator.run(PS, EG, CN)

    assert exc_info.value.review_path == isolated_orchestrator._ws / "review.md"


def test_round_counter_reset_between_runs(isolated_orchestrator):
    """Calling run() twice should start with a fresh round counter each time."""
    # The second run skips the Planner (blueprint.md already exists), so
    # mock_call only fires once (first run only).
    with _mock_agents(
        isolated_orchestrator,
        planner_resps=[_resp("Blueprint.")],
        agentic_resps=[
            _resp("Contract."),   _resp("VERDICT: APPROVE"),  # run 1
            _resp("Contract 2."), _resp("VERDICT: APPROVE"),  # run 2
        ],
    ):
        isolated_orchestrator.run(PS, EG, CN)
        result2 = isolated_orchestrator.run(PS, EG, CN)

    assert result2.rounds_taken == 1


# ── Verdict parser ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("All good.\nVERDICT: APPROVE", "APPROVE"),
    ("Minor nits.\nVERDICT: WARNING", "WARNING"),
    ("Broken.\nVERDICT: BLOCK", "BLOCK"),
    ("Unresolvable.\nVERDICT: ESCALATE", "ESCALATE"),
    ("verdict: approve",  "APPROVE"),    # case-insensitive
    ("VERDICT:APPROVE",   "APPROVE"),    # no space
    ("VERDICT:  BLOCK",   "BLOCK"),      # extra space
])
def test_parse_verdict_extracts_correct_token(isolated_orchestrator, text, expected):
    assert isolated_orchestrator._parse_verdict(text) == expected


def test_parse_verdict_defaults_to_block_on_missing(isolated_orchestrator):
    """
    No verdict found → BLOCK. Conservative: we never silently approve a
    malformed Evaluator response.
    """
    assert isolated_orchestrator._parse_verdict("I reviewed the code and it looks ok.") == "BLOCK"


# ── Error handling ────────────────────────────────────────────────────────────

def test_planner_provider_error_raises_orchestrator_error(isolated_orchestrator):
    """A provider failure in the Planner step surfaces as OrchestratorError."""
    from kyros.core.model_router import ProviderCallError

    with patch.object(isolated_orchestrator.router, "call",
                      side_effect=ProviderCallError("Anthropic 503")):
        with pytest.raises(OrchestratorError, match="Planner call failed"):
            isolated_orchestrator.run(PS, EG, CN)
