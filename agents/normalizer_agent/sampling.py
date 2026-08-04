"""
agents/normalizer_agent/sampling.py

The CLIENT half of the MCP Sampling primitive (spec 2.3).

When this agent calls medical_lang_bridge_tool, the tool does not run an
LLM itself -- it issues ctx.session.create_message() back down the open
MCP connection. That request lands here. This module reads the server's
ModelPreferences hints, picks a concrete Bedrock model, runs the
completion, and hands a CreateMessageResult back to the server.

That split is the whole point of Sampling: the SERVER decides *what*
needs generating and which model family suits it; the CLIENT owns LLM
credentials, routing, and cost. The server never holds an API key.

The prompt guidance comes from MCP Prompts
(abbreviation-normalization-prompt, fetched by node_fetch_prompt), so
the instruction text is server-authored too -- it is injected here via
make_sampling_callback() rather than hardcoded in this file.
"""

import os

from mcp.types import (
    INTERNAL_ERROR,
    CreateMessageRequestParams,
    CreateMessageResult,
    ErrorData,
    TextContent,
)

from agents.common.llm import get_llm
from observability import log_error, log_sampling_event, observe, set_output

# Server-side hint name -> concrete Bedrock model id. The server asks
# for a *family* ("nova-lite" for multilingual work, "command-r-plus"
# for English-only); mapping that onto a real deployed model is the
# client's call, which is exactly the separation Sampling exists for.
MODEL_MAP = {
    "nova-lite": os.getenv("BEDROCK_MODEL_ID_MULTILINGUAL", "amazon.nova-lite-v1:0"),
    "command-r-plus": os.getenv("BEDROCK_MODEL_ID", "cohere.command-r-plus-v1:0"),
}

DEFAULT_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "cohere.command-r-plus-v1:0")


def resolve_model_id(params: CreateMessageRequestParams) -> str:
    """
    Pick a Bedrock model id from the server's ModelPreferences hints.

    Hints are ordered by server preference, so take the first one we
    recognize. An unknown or absent hint is not an error -- the spec
    treats hints as advisory -- so fall back to the default model.
    """
    prefs = params.modelPreferences
    if prefs and prefs.hints:
        for hint in prefs.hints:
            if hint.name and hint.name in MODEL_MAP:
                return MODEL_MAP[hint.name]
    return DEFAULT_MODEL_ID


def flatten_messages(params: CreateMessageRequestParams) -> str:
    """
    Collapse the SamplingMessage list into a single prompt string.

    ChatBedrockConverse takes a plain string here. Only TextContent
    blocks are meaningful for translation; image/audio blocks (which the
    Lang Bridge never sends) are skipped rather than stringified into
    noise.
    """
    parts = []
    for message in params.messages:
        content = message.content
        if isinstance(content, TextContent):
            parts.append(content.text)
    return "\n\n".join(parts)


def make_sampling_callback(instruction: str | None = None, trace_id: str | None = None):
    """
    Build the sampling callback, optionally prefixing every request with
    server-authored instruction text (the abbreviation-normalization
    prompt fetched via MCP Prompts).

    trace_id threads this into the case-level LangFuse trace. Spec 7.2
    asks specifically for sampling events recording the server's model
    preferences, the model the client actually chose, and the result --
    this callback is the only place all three are visible at once.

    Returns a callable matching the SDK's SamplingFnT protocol:
        async (RequestContext, CreateMessageRequestParams)
            -> CreateMessageResult | ErrorData
    """

    async def sampling_callback(context, params: CreateMessageRequestParams):
        try:
            model_id = resolve_model_id(params)
            prompt = flatten_messages(params)

            # Order matters: server's own systemPrompt, then the prompt
            # fetched from MCP Prompts, then the actual payload. The
            # Lang Bridge's own JSON-shape instruction lives in the
            # payload and must stay closest to the text it governs.
            preamble = [p for p in (params.systemPrompt, instruction) if p]
            if preamble:
                prompt = "\n\n".join(preamble) + "\n\n" + prompt

            hints = []
            if params.modelPreferences and params.modelPreferences.hints:
                hints = [h.name for h in params.modelPreferences.hints if h.name]

            llm = get_llm(model_id=model_id)
            with observe(
                "llm.sampling",
                as_type="generation",
                trace_seed=trace_id,
                input=prompt,
                model=model_id,
                metadata={"server_hints": hints, "primitive": "sampling"},
            ) as span:
                response = await llm.ainvoke(prompt)
                text = response.content
                if isinstance(text, list):
                    # Some Bedrock models return content as a list of blocks.
                    text = "".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in text
                    )
                set_output(span, text)

            log_sampling_event(
                trace_id,
                model_preferences=hints,
                model_selected=model_id,
                result_preview=(text or "")[:500],
            )

            return CreateMessageResult(
                role="assistant",
                content=TextContent(type="text", text=text),
                model=model_id,
                stopReason="endTurn",
            )
        except Exception as exc:  # noqa: BLE001
            # Return ErrorData rather than raising: the MCP server is
            # waiting on this response, and an exception here would
            # surface as an opaque transport failure instead of a
            # readable tool error.
            log_error(
                "llm.sampling.failed",
                exc,
                trace_seed=trace_id,
                fallback_action="returned ErrorData to the MCP server",
            )
            return ErrorData(
                code=INTERNAL_ERROR,
                message=f"Sampling callback failed: {type(exc).__name__}: {exc}",
            )

    return sampling_callback


# Convenience default for callers that don't fetch a prompt first.
sampling_callback = make_sampling_callback()
