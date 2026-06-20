from pathlib import Path

from kyros.core.agent_loader import KyrosAgentLoader


def test_get_agent_config_planner():
    root = Path(__file__).resolve().parents[1]
    loader = KyrosAgentLoader(workspace_root=str(root))

    config = loader.get_agent_config("planner")

    assert config["role_title"] == "Lead Quantitative Architect"
    assert "GLOBAL CONSTRAINTS" in config["final_system_prompt"]
    assert config["model_engine"]["provider"] == "anthropic"
