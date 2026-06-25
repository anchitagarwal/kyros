"""
test_model_router.py

─────────────────────────────────────────────────────────────────────────────
CONCEPT: Why mock LiteLLM instead of making real API calls?
─────────────────────────────────────────────────────────────────────────────
Unit tests should be fast, deterministic, and free. Real API calls are slow,
cost money, fail when you have no network, and return different content each
run. We're not testing whether LiteLLM works — that's LiteLLM's problem. We're
testing that ModelRouter correctly:
  - builds the right model string for each provider
  - forwards the right parameters to litellm.completion
  - parses the response into ModelResponse correctly
  - raises the right Kyros exceptions on failure

None of those require a real API call.

─────────────────────────────────────────────────────────────────────────────
PATTERN: unittest.mock.patch as a decorator
─────────────────────────────────────────────────────────────────────────────
`@patch("kyros.core.model_router.litellm.completion")` replaces the
litellm.completion function *as seen from model_router.py* with a MagicMock
for the duration of the test. The patched object is injected as the first
argument after self (mock_completion).

The key detail: patch at the import path, not the definition path. We patch
"kyros.core.model_router.litellm.completion", not "litellm.completion",
because model_router.py imported litellm — that's the name it holds.

─────────────────────────────────────────────────────────────────────────────
PATTERN: Fixture factory functions
─────────────────────────────────────────────────────────────────────────────
_make_agent_config() and _mock_litellm_response() are builder helpers —
they produce valid test inputs without repeating the dict structure in every
test. Each test only specifies what differs.
"""

from unittest.mock import MagicMock, patch

import pytest

from kyros.core.model_router import (
    ModelResponse,
    ModelRouter,
    ProviderCallError,
    TokenUsage,
    UnsupportedProviderError,
)


# ── Test helpers ──────────────────────────────────────────────────────────────

def _make_agent_config(
    provider: str,
    model: str,
    temperature: float = 0.2,
    system_prompt: str = "You are a test agent.",
) -> dict:
    """Minimal agent_config dict matching the shape from KyrosAgentLoader."""
    return {
        "role_title": "Test Agent",
        "goal": "Execute test tasks.",
        "model_engine": {
            "provider": provider,
            "model": model,
            "temperature": temperature,
        },
        "final_system_prompt": system_prompt,
    }


def _mock_litellm_response(content: str, prompt_tokens: int = 10,
                            completion_tokens: int = 20) -> MagicMock:
    """
    Build a MagicMock that mirrors the shape of a real LiteLLM completion
    response, so ModelRouter's response-parsing code runs against a realistic
    object rather than a bare MagicMock.
    """
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


# ── Model string construction ─────────────────────────────────────────────────

@patch("kyros.core.model_router.litellm.completion")
def test_anthropic_model_string(mock_completion):
    """Anthropic calls should use the 'anthropic/' prefix."""
    mock_completion.return_value = _mock_litellm_response("Blueprint drafted.")
    router = ModelRouter()

    router.call(
        _make_agent_config("anthropic", "claude-opus-4-7"),
        messages=[{"role": "user", "content": "Plan."}],
    )

    assert mock_completion.call_args.kwargs["model"] == "anthropic/claude-opus-4-7"


@patch("kyros.core.model_router.litellm.completion")
def test_openai_model_string_has_no_prefix(mock_completion):
    """OpenAI is LiteLLM's default namespace — no provider prefix needed."""
    mock_completion.return_value = _mock_litellm_response("Contract written.")
    router = ModelRouter()

    router.call(
        _make_agent_config("openai", "gpt-5.5"),
        messages=[{"role": "user", "content": "Build."}],
    )

    assert mock_completion.call_args.kwargs["model"] == "gpt-5.5"


@patch("kyros.core.model_router.litellm.completion")
def test_google_model_string(mock_completion):
    """Google Gemini calls use the 'gemini/' prefix."""
    mock_completion.return_value = _mock_litellm_response("Reviewed.")
    router = ModelRouter()

    router.call(
        _make_agent_config("google", "gemini-3.1-pro"),
        messages=[{"role": "user", "content": "Evaluate."}],
    )

    assert mock_completion.call_args.kwargs["model"] == "gemini/gemini-3.1-pro"


# ── Message construction ──────────────────────────────────────────────────────

@patch("kyros.core.model_router.litellm.completion")
def test_system_prompt_prepended_as_first_message(mock_completion):
    """
    The system prompt from agent_config should always be the first message.
    Caller-provided messages follow it. This order matters — LLMs use the
    system message to set context before reading the user turn.
    """
    mock_completion.return_value = _mock_litellm_response("OK")
    router = ModelRouter()
    config = _make_agent_config("openai", "gpt-5.5", system_prompt="You are the Planner.")

    router.call(config, messages=[{"role": "user", "content": "Draft now."}])

    sent = mock_completion.call_args.kwargs["messages"]
    assert sent[0] == {"role": "system", "content": "You are the Planner."}
    assert sent[1] == {"role": "user", "content": "Draft now."}


@patch("kyros.core.model_router.litellm.completion")
def test_caller_messages_fully_preserved(mock_completion):
    """Multi-turn conversation history should be passed through unmodified."""
    mock_completion.return_value = _mock_litellm_response("Understood.")
    router = ModelRouter()

    user_messages = [
        {"role": "user", "content": "First message."},
        {"role": "assistant", "content": "Understood."},
        {"role": "user", "content": "Follow-up."},
    ]
    router.call(_make_agent_config("openai", "gpt-5.5"), messages=user_messages)

    sent = mock_completion.call_args.kwargs["messages"]
    assert sent[1:] == user_messages  # sent[0] is the system prompt


# ── Parameter forwarding ──────────────────────────────────────────────────────

@patch("kyros.core.model_router.litellm.completion")
def test_temperature_forwarded_from_agent_config(mock_completion):
    """Temperature is set per-agent in state.json and must be forwarded exactly."""
    mock_completion.return_value = _mock_litellm_response("Done.")
    router = ModelRouter()

    router.call(
        _make_agent_config("openai", "gpt-5.5", temperature=0.0),
        messages=[{"role": "user", "content": "Build."}],
    )

    assert mock_completion.call_args.kwargs["temperature"] == 0.0


@patch("kyros.core.model_router.litellm.completion")
def test_num_retries_forwarded_from_router_config(mock_completion):
    """
    num_retries is a router-level setting, not per-agent. It controls how many
    times LiteLLM retries a failed call with exponential backoff before giving up.
    """
    mock_completion.return_value = _mock_litellm_response("Done.")
    router = ModelRouter(num_retries=3)

    router.call(_make_agent_config("anthropic", "claude-opus-4-7"),
                messages=[{"role": "user", "content": "Plan."}])

    assert mock_completion.call_args.kwargs["num_retries"] == 3


@patch("kyros.core.model_router.litellm.completion")
def test_timeout_forwarded_from_router_config(mock_completion):
    """Timeout is also router-level — a per-call guard against hung requests."""
    mock_completion.return_value = _mock_litellm_response("Done.")
    router = ModelRouter(timeout=60.0)

    router.call(_make_agent_config("anthropic", "claude-opus-4-7"),
                messages=[{"role": "user", "content": "Plan."}])

    assert mock_completion.call_args.kwargs["timeout"] == 60.0


# ── Response parsing ──────────────────────────────────────────────────────────

@patch("kyros.core.model_router.litellm.completion")
def test_returns_model_response_dataclass(mock_completion):
    """The caller should receive a ModelResponse, not a raw LiteLLM object."""
    mock_completion.return_value = _mock_litellm_response("Here is the blueprint.")
    router = ModelRouter()

    result = router.call(
        _make_agent_config("anthropic", "claude-opus-4-7"),
        messages=[{"role": "user", "content": "Plan."}],
    )

    assert isinstance(result, ModelResponse)


@patch("kyros.core.model_router.litellm.completion")
def test_response_content_extracted(mock_completion):
    mock_completion.return_value = _mock_litellm_response("Here is the blueprint.")
    router = ModelRouter()

    result = router.call(
        _make_agent_config("anthropic", "claude-opus-4-7"),
        messages=[{"role": "user", "content": "Plan."}],
    )

    assert result.content == "Here is the blueprint."


@patch("kyros.core.model_router.litellm.completion")
def test_response_provider_and_model_set(mock_completion):
    """provider and model should reflect what was in agent_config, not the LiteLLM string."""
    mock_completion.return_value = _mock_litellm_response("Done.")
    router = ModelRouter()

    result = router.call(
        _make_agent_config("google", "gemini-3.1-pro"),
        messages=[{"role": "user", "content": "Evaluate."}],
    )

    assert result.provider == "google"
    assert result.model == "gemini-3.1-pro"


@patch("kyros.core.model_router.litellm.completion")
def test_token_usage_populated(mock_completion):
    """TokenUsage fields should be extracted from the LiteLLM response."""
    mock_completion.return_value = _mock_litellm_response(
        "Blueprint.", prompt_tokens=100, completion_tokens=50
    )
    router = ModelRouter()

    result = router.call(
        _make_agent_config("anthropic", "claude-opus-4-7"),
        messages=[{"role": "user", "content": "Plan."}],
    )

    assert isinstance(result.usage, TokenUsage)
    assert result.usage.prompt_tokens == 100
    assert result.usage.completion_tokens == 50
    assert result.usage.total_tokens == 150


# ── Error handling ────────────────────────────────────────────────────────────

def test_unsupported_provider_raises_before_api_call():
    """
    An unregistered provider should raise immediately, before LiteLLM is called.
    This is a configuration error (wrong state.json) not a transient failure —
    retrying won't help, so we surface it as UnsupportedProviderError.
    """
    router = ModelRouter()
    config = _make_agent_config("mistral", "mistral-large-2")

    with pytest.raises(UnsupportedProviderError, match="mistral"):
        router.call(config, messages=[{"role": "user", "content": "Hi."}])


@patch("kyros.core.model_router.litellm.completion",
       side_effect=Exception("Connection timeout after 120s"))
def test_api_failure_wrapped_in_provider_call_error(mock_completion):
    """
    Any exception from LiteLLM (after its own retry logic is exhausted) should
    be caught and re-raised as ProviderCallError. This gives the Orchestrator
    a single, typed exception to handle — it doesn't need to know about
    LiteLLM's internal exception hierarchy.
    """
    router = ModelRouter()

    with pytest.raises(ProviderCallError, match="Connection timeout after 120s"):
        router.call(
            _make_agent_config("openai", "gpt-5.5"),
            messages=[{"role": "user", "content": "Build."}],
        )


@patch("kyros.core.model_router.litellm.completion",
       side_effect=Exception("RateLimitError: 429"))
def test_provider_call_error_preserves_original_exception(mock_completion):
    """The original LiteLLM exception should be chained via __cause__ for debuggability."""
    router = ModelRouter()

    with pytest.raises(ProviderCallError) as exc_info:
        router.call(
            _make_agent_config("anthropic", "claude-opus-4-7"),
            messages=[{"role": "user", "content": "Plan."}],
        )

    assert exc_info.value.__cause__ is not None