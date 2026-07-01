"""reasoning_agent.py — LLM reasoning agent that turns a snapshot into an alert.

Phase 2B continuation: prompt reads liquidity_cycle BEFORE direction selection
and uses ranked_dols (scored) as the default DOL.

OUTPUT JSON SCHEMA / AlertPayload / parser remain unchanged.
"""

from __future__ import annotations

import json
import re

from .alert import AlertPayload, parse_llm_json

__all__ = ["LLMReasoningAgent", "ICT_SYSTEM_PROMPT"]


ICT_SYSTEM_PROMPT = """You are an ICT (Inner Circle Trader) futures analyst for NQ (Nasdaq E-Mini).
You will receive a MarketSnapshot JSON. Your output must be ONLY a valid JSON
object matching the AlertPayload schema. No prose. No markdown. No preamble.
If you cannot produce a valid trade, output no_trade with a reason.

DOL-FIRST REASONING — follow this sequence on every analysis:
1. Read liquidity_cycle (if present) FIRST.
     - last_swept_erl_side: which ERL was just swept (buyside=BSL, sellside=SSL).
     - current_leg: "seek_irl" (price drawing to internal liquidity) or
       "expand_to_erl" (price expanding toward the opposite external ERL).
     - target_erl_side: the opposite ERL side price is expected to draw to.
     - agrees_with_htf_bias: whether the cycle target aligns with htf_bias.
     If agrees_with_htf_bias is false, weight htf_bias over the cycle.
2. Determine direction from htf_bias.
     bullish → target BSL pools above current_price
     bearish → target SSL pools below current_price
3. Read ranked_dols. These are scored/sorted DOL candidates (highest clarity_score first).
     Each ranked_dol includes score_breakdown factors:
       timeframe, role, cycle_align, bias_align, confluence, pd_align, clean_path,
       proximity, killzone.
     Default to ranked_dols[0] as the DOL unless you name a higher-order override.
     Prefer dol_target when present (it should equal ranked_dols[0]).
4. INTERMEDIATE LIQUIDITY CHECK — mandatory, no exceptions.
     ONLY an unswept opposing EXTERNAL (ERL) pool between entry_mid and your
     selected target blocks the trade. Internal (IRL) arrays in the path are
     EXPECTED draw targets, not blockers.
     If an unswept opposing EXTERNAL pool sits between entry_mid and target →
     output no_trade, no_trade_reason: "intermediate liquidity in path"
5. OTE / Fibonacci modifier (read fib_levels per TF):
     - If entry_zone overlaps the golden_pocket (0.618-0.66 retracement) →
       add 15 conviction points.
     - If entry sits at the OTE 0.705 primary (ote_primary) → add 10 more.
     - (The legacy 61.8-79% OTE band from premium_discount still applies.)

ENTRY / TARGET LOGIC (Fibonacci-aware):
     - premium_array true (price above equilibrium) → favor SHORT, target the
       discount / sellside ERL below.
     - premium_array false (price below equilibrium) → favor LONG, target the
       buyside ERL above.
     - Entries at the golden_pocket (0.618-0.66) or OTE 0.705 primary.
     - Targets: 0.382 retracement_target for the first partial, then negative
       extensions (extensions: -0.5, -1.0, -1.5, -2.0, -2.5) toward the opposite ERL.

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
  breaker_blocks contains an entry near current_price.
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
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
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
    def __init__(self, model_router, system_prompt: str = ICT_SYSTEM_PROMPT, agent_config: dict | None = None):
        self.model_router = model_router
        self.system_prompt = system_prompt
        self.agent_config = agent_config or self._default_agent_config()

    @staticmethod
    def _default_agent_config() -> dict:
        return {
            "model_engine": {"provider": "zai", "model": "glm-5.2", "temperature": 0.0},
            "final_system_prompt": "",
        }

    def reason(self, snapshot) -> AlertPayload:
        payload = snapshot.to_compact_dict()
        user_msg = json.dumps(payload, default=str)

        config = dict(self.agent_config)
        config["final_system_prompt"] = self.system_prompt

        try:
            response = self.model_router.call(
                agent_config=config,
                messages=[{"role": "user", "content": user_msg}],
            )
            content = response.content
        except Exception:
            return AlertPayload(bias="no_trade", no_trade_reason="llm_parse_error")

        data = _extract_json(content)
        if data is None:
            return AlertPayload(bias="no_trade", no_trade_reason="llm_parse_error")

        return parse_llm_json(data)
