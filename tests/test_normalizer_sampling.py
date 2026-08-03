"""
tests/test_normalizer_sampling.py

Exercises the MCP Sampling round trip (spec 2.3) with no network and no
AWS credentials.

The real medical_lang_bridge_tool is mounted on a real FastMCP server
and connected to a real ClientSession over the SDK's in-memory
transport, with the Normalizer's real sampling_callback registered. Only
the Bedrock call itself is stubbed. So this covers the part that is
genuinely easy to get wrong -- the server -> client -> server handshake
-- rather than just asserting on mocks.

Run:  python -m pytest tests/test_normalizer_sampling.py -q
"""

import json

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CreateMessageRequestParams, ModelHint, ModelPreferences, SamplingMessage, TextContent

from agents.normalizer_agent import sampling as sampling_module
from agents.normalizer_agent.sampling import (
    flatten_messages,
    make_sampling_callback,
    resolve_model_id,
)
from mcp_primary.tools_lang_bridge import medical_lang_bridge_tool


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Stands in for ChatBedrockConverse."""

    def __init__(self, content, recorder, model_id, raises=None):
        self._content = content
        self._recorder = recorder
        self._model_id = model_id
        self._raises = raises

    async def ainvoke(self, prompt):
        self._recorder.append({"model_id": self._model_id, "prompt": prompt})
        if self._raises:
            raise self._raises
        return FakeResponse(self._content)


@pytest.fixture
def llm_stub(monkeypatch):
    """
    Patches get_llm inside the sampling module. Returns a recorder list
    so tests can assert which model id the callback actually routed to.
    """
    calls = []
    state = {"content": json.dumps({"translated_text": "The patient was discharged.",
                                    "confidence": 0.93}),
             "raises": None}

    def fake_get_llm(model_id=None, **kwargs):
        return FakeLLM(state["content"], calls, model_id, state["raises"])

    monkeypatch.setattr(sampling_module, "get_llm", fake_get_llm)
    return {"calls": calls, "state": state}


def build_server():
    mcp = FastMCP("test-lang-bridge")
    mcp.tool()(medical_lang_bridge_tool)
    return mcp


def unwrap(result):
    assert not result.isError, f"tool returned an error: {result.content}"
    return json.loads(result.content[0].text)


# ---------------------------------------------------------------------
# The handshake
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_sampling_round_trip_returns_translation(llm_stub):
    """
    Server-side ctx.session.create_message() must reach the client's
    callback and the result must flow back into the tool's return value.
    """
    async with create_connected_server_and_client_session(
        build_server(), sampling_callback=make_sampling_callback()
    ) as session:
        result = await session.call_tool(
            "medical_lang_bridge_tool",
            {"text": "Der Patient wurde entlassen.", "source_language": "de"},
        )

    payload = unwrap(result)
    assert payload["translated_text"] == "The patient was discharged."
    assert payload["confidence"] == 0.93
    # model_used comes back from the callback, proving the round trip
    # completed rather than the tool short-circuiting.
    assert payload["model_used"] == "amazon.nova-lite-v1:0"
    assert len(llm_stub["calls"]) == 1


@pytest.mark.asyncio
async def test_non_english_routes_to_multilingual_model(llm_stub):
    """A non-English source must land on the nova-lite hint."""
    async with create_connected_server_and_client_session(
        build_server(), sampling_callback=make_sampling_callback()
    ) as session:
        await session.call_tool(
            "medical_lang_bridge_tool",
            {"text": "मरीज़ को छुट्टी दे दी गई।", "source_language": "hi"},
        )

    assert llm_stub["calls"][0]["model_id"] == "amazon.nova-lite-v1:0"


@pytest.mark.asyncio
async def test_english_routes_to_english_model(llm_stub):
    """An English source must land on the command-r-plus hint."""
    async with create_connected_server_and_client_session(
        build_server(), sampling_callback=make_sampling_callback()
    ) as session:
        await session.call_tool(
            "medical_lang_bridge_tool",
            {"text": "The patient was discharged.", "source_language": "en"},
        )

    assert llm_stub["calls"][0]["model_id"] == "cohere.command-r-plus-v1:0"


@pytest.mark.asyncio
async def test_server_authored_prompt_is_injected(llm_stub):
    """
    The abbreviation-normalization-prompt fetched via MCP Prompts must
    reach the model, not just sit in state.
    """
    instruction = "SERVER-AUTHORED-INSTRUCTION-MARKER"
    async with create_connected_server_and_client_session(
        build_server(), sampling_callback=make_sampling_callback(instruction)
    ) as session:
        await session.call_tool(
            "medical_lang_bridge_tool",
            {"text": "Der Patient wurde entlassen.", "source_language": "de"},
        )

    prompt = llm_stub["calls"][0]["prompt"]
    assert instruction in prompt
    # The payload must still follow the instruction block, so the JSON
    # shape requirement stays closest to the text it governs.
    assert prompt.index(instruction) < prompt.index("Der Patient wurde entlassen.")


@pytest.mark.asyncio
async def test_llm_failure_returns_error_not_hang(llm_stub):
    """
    A failing callback must come back as an MCP error. If it raised
    instead, the server would be left awaiting a response that never
    arrives.
    """
    llm_stub["state"]["raises"] = RuntimeError("bedrock exploded")

    async with create_connected_server_and_client_session(
        build_server(), sampling_callback=make_sampling_callback()
    ) as session:
        result = await session.call_tool(
            "medical_lang_bridge_tool",
            {"text": "Der Patient wurde entlassen.", "source_language": "de"},
        )

    assert result.isError


@pytest.mark.asyncio
async def test_non_json_model_output_fails_soft(llm_stub):
    """
    A model that ignores the JSON instruction must degrade to
    confidence 0.0 (which trips the low-confidence guardrail downstream)
    rather than crashing the pipeline.
    """
    llm_stub["state"]["content"] = "Sure! Here is the translation: the patient went home."

    async with create_connected_server_and_client_session(
        build_server(), sampling_callback=make_sampling_callback()
    ) as session:
        result = await session.call_tool(
            "medical_lang_bridge_tool",
            {"text": "Der Patient wurde entlassen.", "source_language": "de"},
        )

    payload = unwrap(result)
    assert payload["confidence"] == 0.0
    assert payload["translated_text"]


# ---------------------------------------------------------------------
# Pure units
# ---------------------------------------------------------------------
def _params(hints=None, messages=("hello",), system=None):
    return CreateMessageRequestParams(
        messages=[
            SamplingMessage(role="user", content=TextContent(type="text", text=m))
            for m in messages
        ],
        maxTokens=100,
        systemPrompt=system,
        modelPreferences=(
            ModelPreferences(hints=[ModelHint(name=h) for h in hints]) if hints else None
        ),
    )


def test_resolve_model_id_prefers_first_known_hint():
    assert resolve_model_id(_params(["nova-lite"])) == "amazon.nova-lite-v1:0"
    assert resolve_model_id(_params(["command-r-plus"])) == "cohere.command-r-plus-v1:0"


def test_resolve_model_id_skips_unknown_hints():
    """Unknown hints are advisory, not fatal -- fall through to a known one."""
    assert resolve_model_id(_params(["some-unreleased-model", "nova-lite"])) == (
        "amazon.nova-lite-v1:0"
    )


def test_resolve_model_id_falls_back_when_no_hints():
    assert resolve_model_id(_params(None)) == "cohere.command-r-plus-v1:0"
    assert resolve_model_id(_params(["totally-unknown"])) == "cohere.command-r-plus-v1:0"


def test_flatten_messages_joins_text_blocks_in_order():
    assert flatten_messages(_params(messages=("first", "second"))) == "first\n\nsecond"
