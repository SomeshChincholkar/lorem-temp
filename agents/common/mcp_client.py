"""
common/mcp_client.py

Shared async MCP client used by every LangGraph agent to talk to the
Primary Clinical Tools Server (port 8200, streamable-http transport at
/clinicaltools). Wraps tool calls, prompt fetches, and resource reads
behind simple async functions so agent node code never touches the raw
MCP SDK directly.
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import ListRootsResult, Root

from observability import observe, set_output

PRIMARY_MCP_URL = os.getenv("PRIMARY_MCP_URL", "http://localhost:8200/clinicaltools")

# Must match the same folder your Module 3 roots.py / Watcher tool
# treats as the authorized root (the one clinical_watcher_tool /
# clinical_data_harvester_tool call ctx.list_roots() to discover).
# Override with INPUT_ROOT_DIR in .env if your folder isn't "Data/incoming".
INPUT_ROOT_DIR = Path(os.getenv("INPUT_ROOT_DIR", "Data/incoming")).resolve()


async def _list_roots_callback(context):
    """
    Answers the server's roots/list request. Path.as_uri() produces a
    correct file:// URI on both Windows (file:///C:/...) and
    Linux/Mac (file:///home/...), so this works cross-platform.
    """
    return ListRootsResult(roots=[Root(uri=INPUT_ROOT_DIR.as_uri(), name="incoming-documents")])


@asynccontextmanager
async def mcp_session(
    url: str = PRIMARY_MCP_URL,
    sampling_callback=None,
    elicitation_callback=None,
):
    """
    Open a fresh MCP ClientSession for one call (or one graph run).

    sampling_callback: only the Normalizer agent needs to pass one --
    it's the handler that the *server's* ctx.session.create_message()
    call (inside medical_lang_bridge_tool) routes into.

    elicitation_callback: only the Validator agent needs to pass one --
    the handler that the server's ctx.elicit() call (inside
    clinical_rules_engine_tool) routes into. Without it the SDK replies
    "elicitation not supported" and the Rules Engine can never ask a
    human for missing non-blocking fields.

    list_roots_callback: always registered, since both the Harvester
    and Watcher tools call ctx.list_roots() server-side (spec section
    2.1 -- Roots primitive). Without this, the client SDK auto-replies
    "roots not supported" and every tool call that resolves a root
    fails with RuntimeError: MCP tool error: ... List roots not supported.
    """
    async with streamablehttp_client(url) as (read, write, _get_session_id):
        async with ClientSession(
            read,
            write,
            sampling_callback=sampling_callback,
            elicitation_callback=elicitation_callback,
            list_roots_callback=_list_roots_callback,
        ) as session:
            await session.initialize()
            yield session


async def call_tool(
    tool_name: str,
    arguments: dict,
    url: str = PRIMARY_MCP_URL,
    sampling_callback=None,
    elicitation_callback=None,
    trace_id: str | None = None,
) -> dict:
    """
    Call an MCP tool and return its parsed JSON/dict result.

    sampling_callback: pass this when the tool you're calling issues a
    server-side ctx.session.create_message() request (the Sampling
    primitive) that needs fulfilling on the client side -- e.g. the
    Normalizer Agent's medical_lang_bridge_tool call. Leave it None
    for tools that don't use Sampling (Harvester, etc.).

    elicitation_callback: likewise for tools that call ctx.elicit()
    (the Elicitation primitive) -- i.e. clinical_rules_engine_tool,
    called by the Validator Agent.

    trace_id: threads this call into the case-level LangFuse trace. Every
    MCP tool invocation in the system goes through this function, so
    instrumenting here covers spec 7.2's per-tool-call span requirement
    in one place rather than at eight call sites.
    """
    with observe(
        f"mcp.tool.{tool_name}",
        as_type="tool",
        trace_seed=trace_id,
        input=arguments,
        metadata={"server_url": url, "tool": tool_name},
    ) as span:
        async with mcp_session(
            url,
            sampling_callback=sampling_callback,
            elicitation_callback=elicitation_callback,
        ) as session:
            result = await session.call_tool(tool_name, arguments=arguments)
            unwrapped = _unwrap_tool_result(result)
            set_output(span, unwrapped)
            return unwrapped


async def get_prompt_text(
    name: str, arguments: dict, url: str = PRIMARY_MCP_URL, trace_id: str | None = None
) -> str:
    """Fetch a rendered prompt string via MCP Prompts (get_prompt)."""
    with observe(
        f"mcp.prompt.{name}",
        as_type="span",
        trace_seed=trace_id,
        input=arguments,
        metadata={"primitive": "prompts"},
    ) as span:
        async with mcp_session(url) as session:
            result = await session.get_prompt(name, arguments=arguments)
            # GetPromptResult.messages is list[PromptMessage]; every prompt
            # in prompts.py is a single user-role text message.
            text = result.messages[0].content.text
            set_output(span, {"chars": len(text)})
            return text


async def read_resource_text(
    uri: str, url: str = PRIMARY_MCP_URL, trace_id: str | None = None
) -> str:
    """Read an MCP Resource's text content by URI."""
    with observe(
        "mcp.resource.read",
        as_type="retriever",
        trace_seed=trace_id,
        input={"uri": uri},
        metadata={"primitive": "resources"},
    ) as span:
        async with mcp_session(url) as session:
            result = await session.read_resource(uri)
            text = result.contents[0].text
            set_output(span, {"chars": len(text)})
            return text


def _unwrap_tool_result(result) -> dict:
    """
    MCP tool results come back as a CallToolResult with a list of
    content blocks. Our tools all return a JSON-serializable dict/list,
    which FastMCP wraps as a single TextContent block containing the
    JSON text.
    """
    if result.isError:
        raise RuntimeError(f"MCP tool error: {result.content}")
    block = result.content[0]
    text = getattr(block, "text", None)
    if text is None:
        raise RuntimeError(f"Unexpected MCP tool content block: {block}")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text  # a tool may legitimately return plain text