"""
agents/rag_agent/agent.py

The Agno half of the Clinical RAG Q&A Agent (spec 2.6, port 8105).

Agno-specific requirements the spec names explicitly:
  - agno.Agent with MultiMCPTools (both MCP servers at once)
  - SQLite-backed session persistence, last 3 turns as context
  - async arun() invocation

The five RAG roles themselves live in indexing.py + roles.py. This
module is the Agno agent that carries the conversation and the MCP tool
surface; answer_question() below runs the five roles in order and is
what the A2A executor streams.
"""

import os

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.aws import AwsBedrock
from agno.tools.mcp import MultiMCPTools

from agents.common.llm import get_llm

from .indexing import index_all_documents
from .roles import (
    OUT_OF_CONTEXT_ANSWER,
    build_answer_prompt,
    build_generation_input,
    has_grounding,
    rag_triad_score,
    rerank_by_keyword,
    retrieve_top_k,
    sources_for,
)

PRIMARY_MCP_URL = os.getenv("PRIMARY_MCP_URL", "http://localhost:8200/clinicaltools")
SECONDARY_MCP_URL = os.getenv("SECONDARY_MCP_URL", "http://localhost:8201/analyticstools")

SESSION_DB_FILE = os.getenv("RAG_SESSION_DB", "data/rag_sessions.db")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "cohere.command-r-plus-v1:0")

DEFAULT_TOP_K = 5


def build_mcp_tools() -> MultiMCPTools:
    """
    Connect to BOTH MCP servers at once -- the multi-server connectivity
    the spec calls for (section 4). The RAG agent is the one component
    that legitimately needs both: clinical tools for record lookups,
    analytics tools for risk scores and population benchmarks.
    """
    return MultiMCPTools(
        urls=[PRIMARY_MCP_URL, SECONDARY_MCP_URL],
        urls_transports=["streamable-http", "streamable-http"],
        # One unreachable server shouldn't take the Q&A agent down with
        # it; the tools from whichever server is up still work.
        allow_partial_failure=True,
    )


def build_rag_agent(mcp_tools: MultiMCPTools | None = None) -> Agent:
    """
    The Agno agent: Bedrock model, SQLite session persistence carrying
    the last 3 turns, and the dual-MCP tool surface.
    """
    return Agent(
        name="Clinical RAG Q&A Agent",
        model=AwsBedrock(id=BEDROCK_MODEL_ID),
        db=SqliteDb(db_file=SESSION_DB_FILE),
        add_history_to_context=True,
        num_history_runs=3,
        tools=[mcp_tools] if mcp_tools is not None else [],
        instructions=(
            "You answer hospital administrators' questions about patient "
            "discharge records, grounded strictly in retrieved context.\n"
            f"If the context does not contain the answer, reply exactly: "
            f"{OUT_OF_CONTEXT_ANSWER}\n"
            "Never infer clinical facts that are not written in the records."
        ),
        markdown=False,
    )


async def answer_question(
    question: str,
    patient_filter: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    score_triad: bool = True,
) -> dict:
    """
    Run all five RAG roles in order and return the complete result.

    Roles: Indexing (on demand) -> Retrieval -> Augmentation ->
    Generation -> Reflection.
    """
    chunks = []
    async for event in stream_answer(question, patient_filter, top_k, score_triad):
        if event["type"] == "final":
            return event["payload"]
        if event["type"] == "chunks":
            chunks = event["payload"]
    return {"answer": OUT_OF_CONTEXT_ANSWER, "sources": [], "chunks": chunks}


async def stream_answer(
    question: str,
    patient_filter: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    score_triad: bool = True,
):
    """
    Async generator driving the five roles, yielding progress events so
    the A2A layer can stream token-by-token (spec Table 10).

    Event shapes:
        {"type": "status",  "payload": <str>}
        {"type": "chunks",  "payload": [<chunk dicts>]}
        {"type": "token",   "payload": <str>}
        {"type": "final",   "payload": {answer, sources, rag_triad, ...}}
    """
    # Role 1: Indexing (no-op when the index already exists).
    yield {"type": "status", "payload": "Indexing documents..."}
    index_all_documents()

    # Role 2: Retrieval
    yield {"type": "status", "payload": "Retrieving relevant records..."}
    retrieved = retrieve_top_k(question, k=top_k, patient_filter=patient_filter)

    # Nothing relevant found -> the spec's exact refusal string. This
    # check happens BEFORE generation so the model is never given the
    # chance to confabulate from irrelevant context.
    if not retrieved or not has_grounding(retrieved):
        yield {"type": "token", "payload": OUT_OF_CONTEXT_ANSWER}
        yield {
            "type": "final",
            "payload": {
                "answer": OUT_OF_CONTEXT_ANSWER,
                "sources": [],
                "rag_triad": None,
                "grounded": False,
                "patient_filter": patient_filter,
            },
        }
        return

    # Role 3: Augmentation
    yield {"type": "status", "payload": "Re-ranking retrieved context..."}
    ranked = rerank_by_keyword(question, retrieved)
    yield {"type": "chunks", "payload": ranked}

    # Role 4: Generation -- prompt fetched via MCP Prompts, not hardcoded.
    yield {"type": "status", "payload": "Generating grounded answer..."}
    base_prompt = await build_answer_prompt(len(ranked))
    generation_input = build_generation_input(question, ranked, base_prompt)

    llm = get_llm()
    parts: list[str] = []
    try:
        async for chunk in llm.astream(generation_input):
            text = getattr(chunk, "content", None)
            if isinstance(text, list):
                text = "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in text
                )
            if text:
                parts.append(text)
                yield {"type": "token", "payload": text}
    except Exception:
        response = await llm.ainvoke(generation_input)
        text = response.content
        if isinstance(text, list):
            text = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in text
            )
        parts.append(text or "")
        yield {"type": "token", "payload": text or ""}

    answer = "".join(parts).strip() or OUT_OF_CONTEXT_ANSWER

    # Role 5: Reflection -- RAG Triad
    triad = None
    if score_triad:
        yield {"type": "status", "payload": "Scoring answer quality (RAG Triad)..."}
        triad = await rag_triad_score(question, answer, ranked)

    yield {
        "type": "final",
        "payload": {
            "answer": answer,
            "sources": sources_for(ranked),
            "rag_triad": triad,
            "grounded": True,
            "patient_filter": patient_filter,
        },
    }
