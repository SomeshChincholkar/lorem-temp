"""
agents/common/adk_runtime.py

Shared Google ADK runtime plumbing for the three ADK agents (Monitor
:8103, Summary Generator :8104, Host Orchestrator :8083).

Two things every ADK agent here needs:

1. A model that isn't Gemini. ADK defaults to Google's own models, but
   this project's stack is AWS Bedrock (spec section 9: Nova Lite
   primary, Command R+ fallback) reached through LiteLLM. ADK's LiteLlm
   wrapper bridges the two, so the ADK agents use exactly the same
   credentials as the LangGraph ones.

2. A way to run an agent once and get structured results back. ADK is
   event-streaming by nature; run_adk_agent() collects those events into
   the final text plus every tool result, so an A2A executor can return
   the tool's exact structured output instead of re-parsing prose the
   model wrote about it.
"""

import os
from typing import Any, Optional

from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

load_dotenv()

# LiteLLM's Bedrock naming is "bedrock/<bedrock-model-id>".
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "cohere.command-r-plus-v1:0")
ADK_MODEL_STRING = os.getenv("ADK_MODEL", f"bedrock/{BEDROCK_MODEL_ID}")

_model_cache: dict[str, LiteLlm] = {}


def get_adk_model(model_string: str = ADK_MODEL_STRING) -> LiteLlm:
    """Cached LiteLlm model instance for ADK agents."""
    if model_string not in _model_cache:
        _model_cache[model_string] = LiteLlm(model=model_string)
    return _model_cache[model_string]


def _extract_parts(event) -> tuple[list[str], list[Any]]:
    """Pull text chunks and tool results out of one ADK event."""
    texts: list[str] = []
    tool_results: list[Any] = []

    content = getattr(event, "content", None)
    if content is None:
        return texts, tool_results

    for part in getattr(content, "parts", None) or []:
        text = getattr(part, "text", None)
        if text:
            texts.append(text)

        function_response = getattr(part, "function_response", None)
        if function_response is not None:
            tool_results.append(
                {
                    "name": getattr(function_response, "name", None),
                    "response": getattr(function_response, "response", None),
                }
            )

    return texts, tool_results


async def run_adk_agent(
    agent,
    message: str,
    app_name: str,
    user_id: str = "system",
    session_id: Optional[str] = None,
) -> dict:
    """
    Run an ADK agent to completion on one message.

    Returns:
        {"text": <final assistant text>,
         "tool_results": [{"name": ..., "response": ...}, ...]}

    tool_results is the load-bearing half for machine-to-machine A2A
    calls: it's what the tool actually returned, not the model's
    description of it.
    """
    session_service = InMemorySessionService()
    runner = Runner(
        app_name=app_name,
        agent=agent,
        session_service=session_service,
        auto_create_session=True,
    )

    content = genai_types.Content(
        role="user", parts=[genai_types.Part(text=message)]
    )

    all_text: list[str] = []
    final_text: list[str] = []
    tool_results: list[Any] = []

    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id or f"{app_name}-session",
            new_message=content,
        ):
            texts, results = _extract_parts(event)
            all_text.extend(texts)
            tool_results.extend(results)

            is_final = getattr(event, "is_final_response", None)
            if callable(is_final) and is_final():
                final_text.extend(texts)
    finally:
        close = getattr(runner, "close", None)
        if callable(close):
            try:
                await close()
            except Exception:
                pass  # best-effort cleanup; never mask a real error

    return {
        "text": "".join(final_text) or "".join(all_text),
        "tool_results": tool_results,
    }
