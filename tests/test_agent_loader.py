import json
import shutil
from pathlib import Path

import pytest

from kyros.core.agent_loader import KyrosAgentLoader


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_get_agent_config_planner():
    loader = KyrosAgentLoader(workspace_root=str(REPO_ROOT))

    config = loader.get_agent_config("planner")

    assert config["role_title"] == "Lead Quantitative Architect"
    assert "GLOBAL CONSTRAINTS" in config["final_system_prompt"]
    assert config["model_engine"]["provider"] == "anthropic"


def test_get_agent_config_executor():
    loader = KyrosAgentLoader(workspace_root=str(REPO_ROOT))

    config = loader.get_agent_config("executor")

    assert config["role_title"] == "Senior Software Engineer, Data and AI"
    assert "GLOBAL CONSTRAINTS" in config["final_system_prompt"]
    assert config["model_engine"]["provider"]  # non-empty string — don't hardcode
    # Round context is evaluator-only — executor prompt must not contain it
    assert "ROUND CONTEXT" not in config["final_system_prompt"]


def test_get_agent_config_evaluator():
    loader = KyrosAgentLoader(workspace_root=str(REPO_ROOT))

    config = loader.get_agent_config("evaluator")

    assert config["role_title"] == "Independent Code Quality & Correctness Auditor"
    assert "GLOBAL CONSTRAINTS" in config["final_system_prompt"]
    assert config["model_engine"]["provider"]  # non-empty string — don't hardcode


def test_evaluator_prompt_includes_current_round_context():
    loader = KyrosAgentLoader(workspace_root=str(REPO_ROOT))

    config = loader.get_agent_config("evaluator")

    assert "ROUND CONTEXT" in config["final_system_prompt"]
    assert "round 1 of 3" in config["final_system_prompt"].lower()
    # Not yet at the cap, so the final-round escalation instruction
    # should not be present
    assert "FINAL allowed round" not in config["final_system_prompt"]


@pytest.fixture
def isolated_workspace(tmp_path):
    """Minimal workspace with its own state + prompt files, so round-tracking
    writes in these tests never touch the real repo's .kyros_state.json."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    shutil.copy(REPO_ROOT / "config" / "prompts.yaml", config_dir / "prompts.yaml")

    state = json.loads((REPO_ROOT / ".kyros_state.json").read_text())
    state["evaluator_round"] = 0
    (tmp_path / ".kyros_state.json").write_text(json.dumps(state))

    return tmp_path


def test_increment_evaluator_round(isolated_workspace):
    loader = KyrosAgentLoader(workspace_root=str(isolated_workspace))

    assert loader.increment_evaluator_round() == 1
    assert loader.increment_evaluator_round() == 2
    assert loader.load_state()["evaluator_round"] == 2


def test_reset_evaluator_round(isolated_workspace):
    loader = KyrosAgentLoader(workspace_root=str(isolated_workspace))

    loader.increment_evaluator_round()
    loader.increment_evaluator_round()
    loader.reset_evaluator_round()

    assert loader.load_state()["evaluator_round"] == 0


def test_record_verdict_approve_resets_round(isolated_workspace):
    loader = KyrosAgentLoader(workspace_root=str(isolated_workspace))

    loader.increment_evaluator_round()
    loader.increment_evaluator_round()
    updated = loader.record_verdict("APPROVE")

    assert updated["last_review_status"] == "APPROVE"
    assert updated["evaluator_round"] == 0


def test_record_verdict_block_preserves_round(isolated_workspace):
    loader = KyrosAgentLoader(workspace_root=str(isolated_workspace))

    loader.increment_evaluator_round()
    updated = loader.record_verdict("BLOCK")

    assert updated["last_review_status"] == "BLOCK"
    assert updated["evaluator_round"] == 1


def test_should_escalate_false_below_max_rounds(isolated_workspace):
    loader = KyrosAgentLoader(workspace_root=str(isolated_workspace))

    loader.increment_evaluator_round()

    assert loader.should_escalate() is False


def test_should_escalate_true_at_max_rounds(isolated_workspace):
    loader = KyrosAgentLoader(workspace_root=str(isolated_workspace))
    max_rounds = loader.load_state()["max_evaluator_rounds"]

    for _ in range(max_rounds):
        loader.increment_evaluator_round()

    assert loader.should_escalate() is True


def test_evaluator_prompt_warns_on_final_round(isolated_workspace):
    loader = KyrosAgentLoader(workspace_root=str(isolated_workspace))
    max_rounds = loader.load_state()["max_evaluator_rounds"]

    # The FINAL ROUND fires on the last real evaluator call (round index max_rounds-1),
    # not after the orchestrator has already escalated (round index max_rounds).
    for _ in range(max_rounds - 1):
        loader.increment_evaluator_round()

    config = loader.get_agent_config("evaluator")

    assert f"round {max_rounds} of {max_rounds}" in config["final_system_prompt"].lower()
    assert "FINAL allowed round" in config["final_system_prompt"]
    assert "ESCALATE" in config["final_system_prompt"]