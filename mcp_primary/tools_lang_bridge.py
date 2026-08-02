"""
tools_lang_bridge.py

The Medical Lang Bridge primitive. Translates + normalizes medical
text, but does NOT call an LLM directly itself -- it issues a sampling
request back through the client's session (ctx.session.create_message),
so the *calling agent's* LLM client actually does the completion. This
is the mandated separation of concerns: the server describes what it
wants generated, the client (agent) decides which model executes it.

The other half of this handshake -- a sampling_callback that receives
the CreateMessageRequest and routes it to a real model via LiteLLM --
lives in the calling agent's process (e.g. the LangGraph Normalizer
Agent), NOT in this MCP server. Not implemented here; see the note at
the bottom of this file.
"""

import json
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import Context  # noqa: E402
from mcp.types import (  # noqa: E402
    CreateMessageResult,
    ModelHint,
    ModelPreferences,
    SamplingMessage,
    TextContent,
)


def build_model_preferences(source_language: str) -> ModelPreferences:
    """
    Hint which model family the client should prefer for this sampling
    request. English source text can use a cheaper/faster general model;
    non-English source text should prefer a stronger multilingual model.
    """
    if source_language == "en":
        return ModelPreferences(hints=[ModelHint(name="command-r-plus")])
    return ModelPreferences(hints=[ModelHint(name="nova-lite")])  # multilingual


def parse_sampling_result(result: CreateMessageResult) -> Dict:
    """
    Parse the CreateMessageResult returned by ctx.session.create_message()
    into {translated_text, confidence}. The model was instructed (in the
    sampling prompt) to return JSON, but LLMs sometimes wrap it in
    markdown code fences or add stray whitespace -- handle both.
    """
    content = result.content
    if isinstance(content, TextContent):
        raw_text = content.text
    else:
        raw_text = str(content)

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        # strip ```json ... ``` or ``` ... ``` fences
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Model didn't return clean JSON -- fail soft rather than
        # crashing the whole pipeline; downstream can flag low confidence.
        return {
            "translated_text": raw_text.strip(),
            "confidence": 0.0,
            "parse_warning": "Sampling result was not valid JSON; returning raw text.",
        }

    return {
        "translated_text": parsed.get("translated_text", ""),
        "confidence": float(parsed.get("confidence", 0.0)),
    }


async def medical_lang_bridge_tool(ctx: Context, text: str, source_language: str) -> Dict:
    """
    Translate `text` to English and normalize medical abbreviations, by
    asking the connected client's LLM to do it via MCP sampling.

    Args:
        ctx: MCP Context.
        text: raw text to translate/normalize.
        source_language: declared/detected source language code, e.g. "en", "hi".

    Returns:
        {translated_text, confidence, model_used}
    """
    prefs = build_model_preferences(source_language)

    result = await ctx.session.create_message(
        messages=[
            SamplingMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=(
                        "Translate to English and normalize medical abbreviations. "
                        "Return ONLY valid JSON, no prose, no markdown code fences, "
                        'in this exact shape: {"translated_text": <str>, "confidence": <float 0-1>}. '
                        f"Text: {text}"
                    ),
                ),
            )
        ],
        model_preferences=prefs,
        max_tokens=2000,
    )

    parsed = parse_sampling_result(result)
    return {
        "translated_text": parsed["translated_text"],
        "confidence": parsed["confidence"],
        "model_used": getattr(result, "model", None),
    }


# ---------------------------------------------------------------------
# NOT part of this server -- documented here for completeness.
#
# The calling agent (e.g. LangGraph Normalizer Agent) must implement
# the client-side half of sampling: a callback that receives the
# CreateMessageRequest this tool issues, routes it to a real LLM
# (e.g. via litellm.completion(model=hint, messages=..., ...)), and
# returns a CreateMessageResult back to this server.
#
#   def sampling_callback(request: CreateMessageRequest) -> CreateMessageResult:
#       model_hint = request.model_preferences.hints[0].name
#       response = litellm.completion(
#           model=model_hint,
#           messages=[{"role": m.role, "content": m.content.text} for m in request.messages],
#           max_tokens=request.max_tokens,
#       )
#       return CreateMessageResult(
#           role="assistant",
#           content=TextContent(type="text", text=response.choices[0].message.content),
#           model=model_hint,
#           stopReason="endTurn",
#       )
#
# Without a client that registers this callback, calling
# medical_lang_bridge_tool will hang or error -- see the test harness
# for how to stub one for testing.
# ---------------------------------------------------------------------