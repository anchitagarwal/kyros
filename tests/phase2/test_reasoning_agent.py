"""test_reasoning_agent.py — LLM agent: mock router, parse handling, no call_agentic."""

from unittest.mock import MagicMock

from trading.reasoning_agent import LLMReasoningAgent, ICT_SYSTEM_PROMPT
from trading.alert import AlertPayload


def _mock_router(content: str):
    """A MagicMock router whose .call() returns a response with .content."""
    router = MagicMock()
    resp = MagicMock()
    resp.content = content
    router.call.return_value = resp
    return router


def _snap():
    """A minimal snapshot stub with to_compact_dict()."""
    class _S:
        def to_compact_dict(self):
            return {"current_price": 20000.0, "htf_bias": "bullish"}
    return _S()


def test_valid_json_response_maps_to_alert():
    router = _mock_router('{"bias": "long", "model": "2022", "conviction": 75, '
                          '"entry_zone": [100.0, 102.0], "stop": 98.0, "target": 110.0, '
                          '"dol": {"level": 110.0, "type": "bsl", "timeframe": "1h"}, '
                          '"risk_reward": 3.0, "rationale": "sweep+fvg", '
                          '"killzone": "ny_am_kz", "valid_until": "", "no_trade_reason": null}')
    agent = LLMReasoningAgent(router)
    alert = agent.reason(_snap())
    assert alert.bias == "long"
    assert alert.model == "2022"
    assert alert.entry_zone == (100.0, 102.0)


def test_malformed_json_returns_no_trade():
    router = _mock_router("This is not JSON at all.")
    agent = LLMReasoningAgent(router)
    alert = agent.reason(_snap())
    assert alert.bias == "no_trade"
    assert alert.no_trade_reason == "llm_parse_error"


def test_json_wrapped_in_prose_returns_no_trade():
    router = _mock_router("Here is my analysis: {invalid json}")
    agent = LLMReasoningAgent(router)
    alert = agent.reason(_snap())
    assert alert.bias == "no_trade"
    assert alert.no_trade_reason == "llm_parse_error"


def test_json_in_fenced_block_parsed():
    router = _mock_router('```json\n{"bias": "short", "model": "ifvg", "conviction": 60, '
                          '"entry_zone": [100.0, 101.0], "stop": 102.0, "target": 95.0, '
                          '"dol": {"level": 95.0, "type": "ssl", "timeframe": "5m"}, '
                          '"risk_reward": 0.0, "rationale": "", "killzone": "", '
                          '"valid_until": "", "no_trade_reason": null}\n```')
    agent = LLMReasoningAgent(router)
    alert = agent.reason(_snap())
    assert alert.bias == "short"
    assert alert.model == "ifvg"


def test_exactly_one_call_per_reason():
    router = _mock_router('{"bias": "no_trade", "model": "none", "conviction": 0, '
                          '"entry_zone": [0.0, 0.0], "stop": 0.0, "target": 0.0, '
                          '"dol": {"level": 0.0, "type": "", "timeframe": ""}, '
                          '"risk_reward": 0.0, "rationale": "", "killzone": "", '
                          '"valid_until": "", "no_trade_reason": "none"}')
    agent = LLMReasoningAgent(router)
    agent.reason(_snap())
    assert router.call.call_count == 1
    # call_agentic must NEVER be called.
    assert router.call_agentic.call_count == 0


def test_router_exception_returns_no_trade():
    router = MagicMock()
    router.call.side_effect = RuntimeError("API down")
    agent = LLMReasoningAgent(router)
    alert = agent.reason(_snap())
    assert alert.bias == "no_trade"
    assert alert.no_trade_reason == "llm_parse_error"


def test_system_prompt_contains_ict_models():
    """The system prompt must encode the 5 model definitions + DOL-first sequence."""
    for model in ("unicorn", "2022", "silver_bullet", "ifvg", "breaker"):
        assert model in ICT_SYSTEM_PROMPT
    assert "DOL-FIRST REASONING" in ICT_SYSTEM_PROMPT
    assert "INTERMEDIATE LIQUIDITY CHECK" in ICT_SYSTEM_PROMPT


def test_compact_payload_has_no_raw_candles():
    """The user message sent to the LLM must not contain raw candle arrays."""
    router = _mock_router('{"bias":"no_trade","model":"none","conviction":0,'
                          '"entry_zone":[0.0,0.0],"stop":0.0,"target":0.0,'
                          '"dol":{"level":0.0,"type":"","timeframe":""},'
                          '"risk_reward":0.0,"rationale":"","killzone":"",'
                          '"valid_until":"","no_trade_reason":"none"}')
    agent = LLMReasoningAgent(router)
    agent.reason(_snap())
    # Inspect the user message content passed to call().
    args, kwargs = router.call.call_args
    messages = kwargs.get("messages") or args[1]
    user_content = messages[0]["content"]
    assert "candles" not in user_content.lower()
