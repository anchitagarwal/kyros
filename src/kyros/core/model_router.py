"""
model_router.py — the seam between Kyros agent orchestration and LLM providers.

─────────────────────────────────────────────────────────────────────────────
CONCEPT: Why does this exist?
─────────────────────────────────────────────────────────────────────────────
Each of the three LLM providers has a different SDK, auth scheme, and response
shape. Without a router, your Orchestrator becomes a giant switch statement
tangled with business logic:

    if provider == "anthropic":
        import anthropic; client = anthropic.Anthropic(); ...
    elif provider == "openai":
        from openai import OpenAI; client = OpenAI(); ...
    elif provider == "google":
        import google.generativeai as genai; ...

That's brittle. When Anthropic changes a response field (it happens), you're
hunting that change across every file that touched the API directly.

The ModelRouter is the single place that knows about providers. The rest of
the codebase only sees ModelResponse.

─────────────────────────────────────────────────────────────────────────────
INDUSTRY PATTERN: The Abstraction Layer
─────────────────────────────────────────────────────────────────────────────
This is the same pattern used by LiteLLM (33K+ stars on GitHub), LangChain's
ChatModel, and the Vercel AI SDK. The key insight from those projects: what
varies per provider is the API shape, auth, and error format — not the idea
of "send messages, get a completion back". Normalizing that variation into one
interface is all the router needs to do.

We use LiteLLM under the hood (it handles 100+ providers and normalizes retry
logic, error types, and token counting). What we add on top is Kyros-specific:
  - Reading from agent_config (the output of KyrosAgentLoader.get_agent_config)
  - System prompt injection from that config
  - A clean ModelResponse dataclass instead of LiteLLM's internal objects
  - Kyros-specific error hierarchy (ProviderCallError, UnsupportedProviderError)

─────────────────────────────────────────────────────────────────────────────
CONCEPT: Why use LiteLLM instead of raw SDKs?
─────────────────────────────────────────────────────────────────────────────
Anthropic's SDK returns a `Message` object with `.content[0].text`.
OpenAI returns a `ChatCompletion` with `.choices[0].message.content`.
Google Gemini returns a `GenerateContentResponse` with `.text`.

LiteLLM normalizes all of these into one OpenAI-compatible shape. We pick
one shape and translate everything to it, rather than writing three parsers.

The other big win: LiteLLM handles retry + exponential backoff across
providers. A 429 from Anthropic and a 429 from OpenAI are both retried the
same way — you configure `num_retries` once.

─────────────────────────────────────────────────────────────────────────────
CONCEPT: Model string format
─────────────────────────────────────────────────────────────────────────────
LiteLLM uses a "provider/model" convention for non-OpenAI providers:
  anthropic/claude-opus-4-7
  gemini/gemini-3.1-pro
  gpt-5.5    ← OpenAI is the implicit default, no prefix

Our _PROVIDER_PREFIX map handles this translation from Kyros's state.json
provider names ("anthropic", "openai", "google") to LiteLLM strings.
"""

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import litellm

_log = logging.getLogger(__name__)

# Silence LiteLLM's verbose request/response logging.
# Kyros controls its own log output; we don't want LiteLLM printing to stdout.
litellm.suppress_debug_info = True
# Some models (e.g. claude-opus-4-8 extended thinking) reject temperature != 1.
# Drop unsupported params instead of hard-failing.
litellm.drop_params = True

# Enable Langfuse tracing if credentials are present in the environment.
# Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and optionally LANGFUSE_HOST.
if os.getenv("LANGFUSE_PUBLIC_KEY"):
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]
    _log.info("Langfuse observability enabled")


# ── Provider → LiteLLM prefix map ────────────────────────────────────────────
# This is the translation table from Kyros's provider names (used in state.json)
# to LiteLLM's "provider/model" string format.
# Add a new entry here when you add a new provider — nothing else changes.
_PROVIDER_PREFIX: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "",      # OpenAI is the default namespace; no prefix needed
    "zai": "zai",     # Zhipu AI GLM models (set ZAI_API_KEY)
    "google": "gemini",
}


# ── Data classes ─────────────────────────────────────────────────────────────
# Using dataclasses rather than dicts so callers get attribute access and type
# hints, not string key lookups that fail silently when mistyped.

@dataclass
class TokenUsage:
    """Token counts from the LLM response. Always present; defaults to 0."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ModelResponse:
    """
    Normalized response returned by ModelRouter.call().

    All provider-specific shapes have been collapsed into this single structure.
    The Orchestrator only imports this class — it has no direct dependency on
    Anthropic, OpenAI, or Google SDKs.
    """
    content: str           # The text content of the model's reply
    provider: str          # e.g. "anthropic"
    model: str             # e.g. "claude-opus-4-7"
    usage: TokenUsage = field(default_factory=TokenUsage)


# ── Error hierarchy ───────────────────────────────────────────────────────────
# Kyros-specific exceptions, so callers don't need to import LiteLLM exceptions
# directly. This keeps the abstraction clean — if we swap out LiteLLM later,
# only model_router.py changes.

class RouterError(Exception):
    """Base class for all ModelRouter failures."""


class ProviderCallError(RouterError):
    """Raised when the LLM API call fails after all retries are exhausted."""


class UnsupportedProviderError(RouterError):
    """Raised when agent_config specifies a provider not in _PROVIDER_PREFIX."""


# ── ModelRouter ───────────────────────────────────────────────────────────────

class ModelRouter:
    """
    Thin, Kyros-aware wrapper over LiteLLM's completion() interface.

    This is designed to be instantiated once and reused across the entire
    agent loop — it holds no per-call state.

    Example:
        router = ModelRouter()
        response = router.call(
            agent_config=loader.get_agent_config("planner"),
            messages=[{"role": "user", "content": "Draft the blueprint."}],
        )
        blueprint_text = response.content
    """

    def __init__(self, num_retries: int = 2, timeout: float = 600.0):
        """
        Args:
            num_retries: How many times LiteLLM will retry a failed call before
                         raising. Retries use exponential backoff automatically.
                         Default 2 = 3 total attempts (1 initial + 2 retries).
            timeout:     Per-call timeout in seconds. 600s accommodates extended-
                         thinking models (e.g. claude-opus-4-8) that can run long.
        """
        self.num_retries = num_retries
        self.timeout = timeout

    def _build_model_string(self, provider: str, model: str) -> str:
        """
        Translate Kyros provider+model into a LiteLLM model identifier string.

        This is the only place in the codebase that knows LiteLLM's naming
        convention. All other code uses provider + model as separate fields
        from state.json / agent_config.
        """
        if provider not in _PROVIDER_PREFIX:
            raise UnsupportedProviderError(
                f"Provider '{provider}' is not registered in ModelRouter. "
                f"Supported providers: {sorted(_PROVIDER_PREFIX.keys())}"
            )
        prefix = _PROVIDER_PREFIX[provider]
        return f"{prefix}/{model}" if prefix else model

    def call(
        self,
        agent_config: dict,
        messages: list[dict],
        name: str | None = None,
        trace_id: str | None = None,
    ) -> ModelResponse:
        """
        Execute one LLM completion for the given agent config and message list.

        The system prompt from agent_config is always prepended as the first
        message. The caller owns the conversation history (the messages list);
        the router owns the provider call.

        Args:
            agent_config: Output of KyrosAgentLoader.get_agent_config().
                          Required keys under 'model_engine': provider, model.
                          Required key: 'final_system_prompt'.
            messages:     Conversation history as a list of role/content dicts.
                          {"role": "user", "content": "..."} is the typical form.

        Returns:
            ModelResponse with .content, .provider, .model, .usage.

        Raises:
            UnsupportedProviderError: provider not in _PROVIDER_PREFIX.
            ProviderCallError: API call failed after num_retries attempts.
        """
        engine = agent_config.get("model_engine", {})
        provider = engine.get("provider", "")
        model_name = engine.get("model", "")
        temperature = engine.get("temperature", 0.2)
        system_prompt = agent_config.get("final_system_prompt", "")

        # Raises UnsupportedProviderError early, before any API call is made
        model_string = self._build_model_string(provider, model_name)

        # Prepend the system prompt. We do this here, not in agent_loader,
        # because the caller controls the conversation history. The system
        # prompt is configuration, not conversation — keeping it separate
        # makes it easy to inspect or override in tests.
        full_messages = [{"role": "system", "content": system_prompt}, *messages]

        metadata: dict = {}
        if name:
            metadata["generation_name"] = name
        if trace_id:
            metadata["trace_id"] = trace_id
            metadata["trace_name"] = f"kyros/{name or 'call'}"

        try:
            raw = litellm.completion(
                model=model_string,
                messages=full_messages,
                temperature=temperature,
                num_retries=self.num_retries,
                timeout=self.timeout,
                **({"metadata": metadata} if metadata else {}),
            )
        except Exception as exc:
            raise ProviderCallError(
                f"LLM call failed for {provider}/{model_name} after "
                f"{self.num_retries} retries: {exc}"
            ) from exc

        content = raw.choices[0].message.content or ""
        usage = raw.usage

        return ModelResponse(
            content=content,
            provider=provider,
            model=model_name,
            usage=TokenUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                total_tokens=getattr(usage, "total_tokens", 0),
            ),
        )

    def call_agentic(
        self,
        agent_config: dict,
        messages: list[dict],
        tools: list[dict],
        tool_fn: Callable[[str, dict], str],
        max_turns: int = 80,
        name: str | None = None,
        trace_id: str | None = None,
    ) -> ModelResponse:
        """
        Run a multi-turn tool-use loop for an agent.

        Each turn: call the LLM with `tools`; if the model returns tool_calls,
        execute each via `tool_fn` and append results before the next turn.
        When the model returns a plain text response (no tool_calls), the loop
        ends and that text becomes ModelResponse.content.

        Token usage is accumulated across all turns.

        Args:
            agent_config: Same format as call() — from KyrosAgentLoader.
            messages:     Initial user message(s). Mutated in place as the
                          conversation grows; pass a fresh list each time.
            tools:        Tool schemas in OpenAI function-calling format.
            tool_fn:      Callable(name, arguments_dict) → result_str.
                          ExecutorToolkit.dispatch satisfies this signature.
            max_turns:    Hard cap on LLM calls to prevent runaway loops.

        Raises:
            ProviderCallError: Any LLM call fails after all retries.
            ProviderCallError: max_turns reached without a final text response.
        """
        engine = agent_config.get("model_engine", {})
        provider = engine.get("provider", "")
        model_name = engine.get("model", "")
        temperature = engine.get("temperature", 0.2)
        system_prompt = agent_config.get("final_system_prompt", "")

        model_string = self._build_model_string(provider, model_name)

        conv = [{"role": "system", "content": system_prompt}, *messages]

        total_prompt = 0
        total_completion = 0

        for turn in range(max_turns):
            turn_metadata: dict = {}
            if name:
                turn_metadata["generation_name"] = f"{name}-turn-{turn + 1}"
            if trace_id:
                turn_metadata["trace_id"] = trace_id
                turn_metadata["trace_name"] = f"kyros/{name or 'agentic'}"

            try:
                raw = litellm.completion(
                    model=model_string,
                    messages=conv,
                    tools=tools,
                    temperature=temperature,
                    num_retries=self.num_retries,
                    timeout=self.timeout,
                    **({"metadata": turn_metadata} if turn_metadata else {}),
                )
            except Exception as exc:
                raise ProviderCallError(
                    f"LLM call failed for {provider}/{model_name} after "
                    f"{self.num_retries} retries: {exc}"
                ) from exc

            usage = raw.usage
            total_prompt += getattr(usage, "prompt_tokens", 0)
            total_completion += getattr(usage, "completion_tokens", 0)

            message = raw.choices[0].message
            tool_calls = message.tool_calls or []

            if not tool_calls:
                return ModelResponse(
                    content=message.content or "",
                    provider=provider,
                    model=model_name,
                    usage=TokenUsage(
                        prompt_tokens=total_prompt,
                        completion_tokens=total_completion,
                        total_tokens=total_prompt + total_completion,
                    ),
                )

            # Append the assistant turn with its tool calls
            conv.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })

            # Execute each tool call and append results
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    conv.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"ERROR: malformed JSON in tool arguments: {exc}",
                    })
                    continue
                result = tool_fn(tc.function.name, args)
                _log.debug("tool=%s result_len=%d", tc.function.name, len(result))
                conv.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        raise ProviderCallError(
            f"Agentic loop for {provider}/{model_name} hit max_turns={max_turns} "
            "without returning a final text response."
        )