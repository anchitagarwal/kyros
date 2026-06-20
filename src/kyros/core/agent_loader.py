import json
import yaml
from pathlib import Path


class KyrosAgentLoader:
    def __init__(self, workspace_root: str = "."):
        self.root = Path(workspace_root)
        self.state_file = self.root / ".kyros_state.json"
        self.prompt_file = self.root / "config" / "prompts.yaml"
    
    def load_state(self) -> dict:
        """Reads the current project phase and infra config"""
        if not self.state_file.exists():
            raise FileNotFoundError(f"State file missing at {self.state_file}")
        with open(self.state_file, "r") as f:
            return yaml.safe_load(f)
    
    def load_prompts(self) -> dict:
        """Loads the PromptOps YAML registry"""
        if not self.prompt_file.exists():
            raise FileNotFoundError(f"Prompt registry missing at {self.prompt_file}")
        with open(self.prompt_file, "r") as f:
            return yaml.safe_load(f)
        
    def get_agent_config(self, role: str) -> dict:
        """
        Dynamically injects the correct prompt and compute model based on the current phase
        """
        state = self.load_state()
        prompts = self.load_prompts()

        current_phase = state.get("current_phase")
        if not current_phase:
            raise ValueError(f"current_phase is not defined in .kyros_state.json")
        
        # 1. isolate the specific prompt for this role and phase
        role_config = prompts.get("agents", {}).get(role, {}).get(current_phase)
        if not current_phase:
            raise ValueError("current_phase is not defined in .kyros_state.json")
        
        system_prompt = role_config.get("system_prompt", "")

        # 2. Programmatically stitch global constraints onto the prompt
        global_constraints = prompts.get("global_constraints", [])
        if global_constraints:
            system_prompt += "\n\n### GLOBAL CONSTRAINTS ###\n"
            system_prompt += "\n".join(f"- {constraint}" for constraint in global_constraints)
        
        # 3. Fetch the compute infrastructure (OpenAI/Anthropic) from state
        infrastructure = state.get("infrastructure", {}).get(role, {})

        return {
            "role_title": role_config.get("role"),
            "goal": role_config.get("goal", "Execute designated tasks."),
            "model_engine": infrastructure,
            "final_system_prompt": system_prompt
        }


# Example invocation
if __name__ == "__main__":
    loader = KyrosAgentLoader()

    # Load the Phase 1 Planner
    planner_setup = loader.get_agent_config("planner")
    print(f"Booting {planner_setup['role_title']} using {planner_setup['model_engine'].get('model', 'default')}...")
    
    # Load the Phase 1 Executor
    executor_setup = loader.get_agent_config("executor")
    print(f"Booting {executor_setup['role_title']} using {executor_setup['model_engine'].get('model', 'default')}...")