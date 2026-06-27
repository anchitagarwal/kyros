"""reasoning_agent.py — LLM reasoning agent that turns a snapshot into an alert.

Uses ``model_router.call()`` (NEVER ``call_agentic``) with the ICT system
prompt. The snapshot is serialized to a compact JSON payload (no raw candle
lists) and sent as the user message. The LLM's JSON response is parsed into
an ``AlertPayload``; on any parse failure, a no_trade alert with
``no_trade_reason="llm_parse_error"`` is returned (never raises).

The ICT system prompt is embedded verbatim per the module layout spec.
"""

from __future__ import annotations

import json
import re

from .alert import AlertPayload, parse_llm_json

__all__ = ["LLMReasoningAgent", "ICT_SYSTEM_PROMPT"]


# ── ICT System Prompt (verbatim per spec) ─────────────────────────────────────

ICT_SYSTEM_PROMPT = """You are an ICT (Inner Circle Trader) futures analyst for NQ (Nasdaq E-Mini).
You will receive a MarketSnapshot JSON. Your output must be ONLY a valid JSON
object matching the AlertPayload schema. No prose. No markdown. No preamble.
If you cannot produce a valid trade, output no_trade with a reason.

DOL-FIRST REASONING — follow this sequence on every analysis:
1. Read all_pools. These are all unswept liquidity levels sorted by proximity.
2. Determine direction from htf_bias.
     bullish → target BSL pools above current_price
     bearish → target SSL pools below current_price
3. Select your target pool (DOL).
     Default to nearest_dol. Prefer higher confluence_count. Prefer pools
     from higher timeframes (1h > 15m > 5m) when R:R still clears 1:1.
4. INTERMEDIATE LIQUIDITY CHECK — mandatory, no exceptions.
     If any unswept opposing pool sits between entry_mid and your selected
     target → output no_trade, no_trade_reason: "intermediate liquidity in path"
5. OTE modifier: if entry_zone overlaps the OTE band (61.8-79% retracement of
     current dealing range from premium_discount) → add 15 conviction points.

ICT MODEL IDENTIFICATION — check in this order:

unicorn (highest conviction — check first):
  All three conditions required:
    (1) BOS or ChoCH on 5m
    (2) Displacement candle creates a 5m FVG
    (3) OB overlaps FVG: OB.top >= FVG.low AND OB.bottom <= FVG.high
  If all met: model="unicorn", conviction >= 75.
  Stop: beyond displacement origin. Entry: OB/FVG overlap zone.

2022 (AMD sequence — all three required):
  (1) recent_sweeps shows BSL or SSL swept
  (2) displacements shows displacement candle after the sweep
  (3) fvgs["5m"] contains FVG formed by that same displacement
  A standalone FVG with no preceding sweep+displacement is NOT a 2022 setup.
  Entry: retracement to FVG. Stop: beyond swept level.

silver_bullet (time-gated):
  ONLY valid when timestamp is in: 03:00-04:00, 10:00-11:00, or 14:00-15:00 ET.
  No sweep required. Any displacement FVG in the window qualifies.
  Entry: retracement to FVG. Stop: opposite end of FVG.

ifvg:
  ifvgs["5m"] contains iFVG near current_price.
  Current price approaching or testing iFVG zone from the new side.

breaker:
  breaker_blocks contains an entry near current_price (these are former OBs that
  have been mitigated and flipped — they now act as opposing S/R).
  When both breaker and ifvg conditions are present at the same level,
  prefer breaker.

none → no_trade: output no_trade with no_trade_reason if no model applies
  or conviction < 40.

OUTPUT SCHEMA — return ONLY this JSON object, nothing else:
{
  "bias": "long|short|no_trade",
  "model": "2022|unicorn|ifvg|silver_bullet|breaker|none",
  "conviction": 0,
  "entry_zone": [0.0, 0.0],
  "stop": 0.0,
  "target": 0.0,
  "dol": {"level": 0.0, "type": "", "timeframe": ""},
  "risk_reward": 0.0,
  "rationale": "",
  "killzone": "",
  "valid_until": "",
  "no_trade_reason": null
}
"""


def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from an LLM text response.

    Handles three cases:
      1. The text is pure JSON.
      2. The JSON is wrapped in a ```json ... ``` fenced block.
      3. The JSON is embedded in prose (find the first {...} block).

    Returns the parsed dict, or None if no valid JSON object is found.
    """
    text = text.strip()
    # Case 1: pure JSON.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Case 2: fenced code block.
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    # Case 3: first {...} block (greedy enough to capture nested braces).
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


class LLMReasoningAgent:
    """Turn a MarketSnapshot into an AlertPayload via a single LLM call."""

    def __init__(self, model_router, system_prompt: str = ICT_SYSTEM_PROMPT,
                 agent_config: dict | None = None):
        """Args:
            model_router: a ModelRouter instance (uses .call() only).
            system_prompt: the ICT system prompt (defaults to the verbatim spec).
            agent_config: the agent_config dict for model_router.call(). If
                None, a minimal config is built from the router's defaults.
        """
        self.model_router = model_router
        self.system_prompt = system_prompt
        self.agent_config = agent_config or self._default_agent_config()

    @staticmethod
    def _default_agent_config() -> dict:
        """A minimal agent_config for model_router.call().

        The executor in .kyros_state.json uses zai/glm-5.2; we mirror that
        so a real router call would work, though tests always mock the router.
        """
        return {
            "model_engine": {"provider": "zai", "model": "glm-5.2", "temperature": 0.0},
            "final_system_prompt": "",
        }

    def reason(self, snapshot) -> AlertPayload:
        """Produce an AlertPayload from ``snapshot`` via one LLM call.

        Serializes the snapshot to compact JSON (no raw candles), calls
        model_router.call() exactly once, parses the JSON response. On any
        parse failure, returns no_trade with no_trade_reason="llm_parse_error".
        Never raises.
        """
        payload = snapshot.to_compact_dict()
        user_msg = json.dumps(payload, default=str)

        # Build the agent_config with our system prompt injected.
        config = dict(self.agent_config)
        config["final_system_prompt"] = self.system_prompt

        try:
            response = self.model_router.call(
                agent_config=config,
                messages=[{"role": "user", "content": user_msg}],
            )
            content = response.content
        except Exception:
            # Router failure → no_trade, never crash the loop.
            return AlertPayload(bias="no_trade", no_trade_reason="llm_parse_error")

        data = _extract_json(content)
        if data is None:
            return AlertPayload(bias="no_trade", no_trade_reason="llm_parse_error")

        return parse_llm_json(data)
