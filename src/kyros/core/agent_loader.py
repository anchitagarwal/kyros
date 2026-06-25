import json
import os
from pathlib import Path

import yaml


class KyrosAgentLoader:
    def __init__(self, workspace_root: str = "."):
        self.root = Path(workspace_root)
        self.state_file = self.root / ".kyros_state.json"
        self.prompt_file = self.root / "config" / "prompts.yaml"

    def load_state(self) -> dict:
        """Reads the current project phase, infra config, and round tracking."""
        if not self.state_file.exists():
            raise FileNotFoundError(f"State file missing at {self.state_file}")
        with open(self.state_file, "r") as f:
            return json.load(f)

    def save_state(self, state: dict) -> None:
        """Writes the project state back to .kyros_state.json."""
        tmp = self.state_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
            f.write("\n")
        os.replace(tmp, self.state_file)

    def load_prompts(self) -> dict:
        """Loads the PromptOps YAML registry"""
        if not self.prompt_file.exists():
            raise FileNotFoundError(f"Prompt registry missing at {self.prompt_file}")
        with open(self.prompt_file, "r") as f:
            return yaml.safe_load(f)

    def get_agent_config(self, role: str) -> dict:
        """
        Dynamically injects the correct prompt and compute model based on the
        current phase. For the evaluator role, also injects live round-tracking
        context (current round vs. max_evaluator_rounds) so the agent knows
        whether this is its last allowed attempt before escalation.
        """
        state = self.load_state()
        prompts = self.load_prompts()

        current_phase = state.get("current_phase")
        if not current_phase:
            raise ValueError("current_phase is not defined in .kyros_state.json")

        # 1. isolate the specific prompt for this role and phase
        role_config = prompts.get("agents", {}).get(role, {}).get(current_phase)
        if role_config is None:
            raise ValueError(
                f"No prompt registered for role='{role}' phase='{current_phase}'"
            )

        system_prompt = role_config.get("system_prompt", "")

        # 2. Programmatically stitch global constraints onto the prompt
        global_constraints = prompts.get("global_constraints", [])
        if global_constraints:
            system_prompt += "\n\n### GLOBAL CONSTRAINTS ###\n"
            system_prompt += "\n".join(f"- {constraint}" for constraint in global_constraints)

        # 3. Inject live round-tracking context for the evaluator only. This is
        #    per-invocation runtime state, not static config, so it's appended
        #    here at call time rather than baked into the YAML prompt text.
        if role == "evaluator":
            evaluator_round = state.get("evaluator_round", 0)
            max_rounds = state.get("max_evaluator_rounds", 3)
            system_prompt += (
                "\n\n### ROUND CONTEXT ###\n"
                f"This is round {evaluator_round + 1} of {max_rounds}.\n"
            )
            if evaluator_round >= max_rounds - 1:
                system_prompt += (
                    "This is the FINAL allowed round. If CRITICAL or HIGH issues "
                    "remain, you MUST return verdict ESCALATE rather than "
                    "requesting another Executor iteration.\n"
                )

        # 4. Fetch the compute infrastructure (provider/model) from state
        infrastructure = state.get("infrastructure", {}).get(role, {})

        return {
            "role_title": role_config.get("role"),
            "goal": role_config.get("goal", "Execute designated tasks."),
            "model_engine": infrastructure,
            "final_system_prompt": system_prompt,
        }

    # -------------------------------------------------------------------
    # Round-tracking helpers. Centralizing these here keeps evaluator retry
    # bookkeeping consistent rather than scattering raw state-file reads and
    # writes through orchestration code.
    # -------------------------------------------------------------------

    def increment_evaluator_round(self) -> int:
        """Advance the evaluator round counter by one and persist it.

        Call this when the Executor is sent back to address a BLOCK/WARNING
        verdict. Returns the new round number.
        """
        state = self.load_state()
        state["evaluator_round"] = state.get("evaluator_round", 0) + 1
        self.save_state(state)
        return state["evaluator_round"]

    def reset_evaluator_round(self) -> None:
        """Reset the evaluator round counter to 0 (on APPROVE or phase advance)."""
        state = self.load_state()
        state["evaluator_round"] = 0
        self.save_state(state)

    def record_verdict(self, verdict: str) -> dict:
        """Persist the Evaluator's verdict to state.

        On APPROVE, also resets the round counter, since approval ends the
        review cycle for this unit of work. Returns the updated state dict.
        """
        state = self.load_state()
        state["last_review_status"] = verdict
        if verdict == "APPROVE":
            state["evaluator_round"] = 0
        self.save_state(state)
        return state

    def should_escalate(self) -> bool:
        """True once the evaluator round counter has reached the configured max.

        Orchestration code should check this BEFORE invoking the Evaluator
        again after a BLOCK/WARNING verdict — if True, route to the human
        operator instead of spending another round.
        """
        state = self.load_state()
        return state.get("evaluator_round", 0) >= state.get("max_evaluator_rounds", 3)


# Example invocation
if __name__ == "__main__":
    loader = KyrosAgentLoader()

    # Load the Phase 1 Planner
    planner_setup = loader.get_agent_config("planner")
    print(f"Booting {planner_setup['role_title']} using {planner_setup['model_engine'].get('model', 'default')}...")

    # Load the Phase 1 Executor
    executor_setup = loader.get_agent_config("executor")
    print(f"Booting {executor_setup['role_title']} using {executor_setup['model_engine'].get('model', 'default')}...")

    # Load the Phase 1 Evaluator
    evaluator_setup = loader.get_agent_config("evaluator")
    print(f"Booting {evaluator_setup['role_title']} using {evaluator_setup['model_engine'].get('model', 'default')}...")