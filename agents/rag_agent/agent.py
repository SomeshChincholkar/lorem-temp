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

from agents.common.llm import DEFAULT_MODEL_ID, get_llm
from guardrails import REJECT, GuardrailManager
from observability import (
    log_guardrail_events,
    observe,
    record_generation,
    set_output,
)

from .indexing import PatientNotIndexedError, index_patient_documents
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
            "You answer questions about ONE patient's discharge records, "
            "grounded strictly in the retrieved context.\n"
            "Every question is scoped to a single patient whose records "
            "are the only ones you can see. Never speculate about other "
            "patients or compare across patients.\n"
            f"If the context does not contain the answer, reply exactly: "
            f"{OUT_OF_CONTEXT_ANSWER}\n"
            "Never infer clinical facts that are not written in the records."
        ),
        markdown=False,
    )


MISSING_PATIENT_ANSWER = (
    "Select a patient before asking a question — this assistant answers "
    "from one patient's records at a time."
)


async def answer_question(
    question: str,
    patient_id: str,
    top_k: int = DEFAULT_TOP_K,
    score_triad: bool = True,
) -> dict:
    """
    Run all five RAG roles in order and return the complete result.

    Roles: Indexing (on demand) -> Retrieval -> Augmentation ->
    Generation -> Reflection.

    patient_id is required: retrieval reads that patient's own index.
    """
    chunks = []
    async for event in stream_answer(question, patient_id, top_k, score_triad):
        if event["type"] == "final":
            return event["payload"]
        if event["type"] == "chunks":
            chunks = event["payload"]
    return {"answer": OUT_OF_CONTEXT_ANSWER, "sources": [], "chunks": chunks}


async def stream_answer(
    question: str,
    patient_id: str,
    top_k: int = DEFAULT_TOP_K,
    score_triad: bool = True,
    trace_id: str | None = None,
):
    """
    Async generator driving the five roles, yielding progress events so
    the A2A layer can stream token-by-token (spec Table 10).

    patient_id is mandatory. Each patient has their own FAISS index, so
    every question is answered from exactly one patient's records and
    cross-patient leakage is structurally impossible rather than
    filtered-out after the fact.

    Two RAI guardrails wrap the pipeline (spec Table 12):
      - PromptInjectionGuard on the way in. This is the only place in the
        system where free user text reaches an LLM, so it is the only
        place injection is a live risk.
      - HallucinationChecker on the way out, gating on the Reflection
        agent's faithfulness score.

    Event shapes:
        {"type": "status",  "payload": <str>}
        {"type": "chunks",  "payload": [<chunk dicts>]}
        {"type": "token",   "payload": <str>}
        {"type": "final",   "payload": {answer, sources, rag_triad, ...}}
    """
    guardrails = GuardrailManager(trace_id=trace_id)

    # A question with no patient cannot be answered at all -- there is no
    # index to search. Reported distinctly from the out-of-context
    # refusal, because "you didn't pick a patient" and "their records
    # don't say" are different things and the user can act on the first.
    if not patient_id:
        yield {"type": "token", "payload": MISSING_PATIENT_ANSWER}
        yield {
            "type": "final",
            "payload": {
                "answer": MISSING_PATIENT_ANSWER,
                "sources": [],
                "rag_triad": None,
                "grounded": False,
                "patient_id": None,
                "error": "patient_id is required",
                "guardrail_events": guardrails.summary()["events"],
            },
        }
        return

    # Guardrail: prompt injection, before anything touches a model.
    injection = guardrails.check_query(question)
    if injection["action"] == REJECT:
        yield {"type": "token", "payload": OUT_OF_CONTEXT_ANSWER}
        yield {
            "type": "final",
            "payload": {
                "answer": OUT_OF_CONTEXT_ANSWER,
                "sources": [],
                "rag_triad": None,
                "grounded": False,
                "patient_id": patient_id,
                "injection_detected": True,
                "injection_matches": injection["matches"],
                "guardrail_events": guardrails.summary()["events"],
            },
        }
        log_guardrail_events(guardrails, trace_seed=trace_id)
        return

    # A sanitized query has had role markers stripped; everything
    # downstream must use it, not the original.
    question = injection["sanitized_query"] or question

    # Role 1: Indexing -- only this patient's index, built on demand.
    yield {"type": "status", "payload": f"Preparing index for {patient_id}..."}
    try:
        index_patient_documents(patient_id)
    except PatientNotIndexedError as exc:
        message = f"No records are indexed for patient {patient_id}."
        yield {"type": "token", "payload": message}
        yield {
            "type": "final",
            "payload": {
                "answer": message,
                "sources": [],
                "rag_triad": None,
                "grounded": False,
                "patient_id": patient_id,
                "error": str(exc),
                "guardrail_events": guardrails.summary()["events"],
            },
        }
        return

    # Role 2: Retrieval -- searches ONLY this patient's index.
    yield {"type": "status", "payload": "Retrieving relevant records..."}
    with observe(
        "rag.retrieval",
        as_type="retriever",
        trace_seed=trace_id,
        input={"question": question, "top_k": top_k, "patient_id": patient_id},
    ) as span:
        retrieved = retrieve_top_k(question, patient_id, k=top_k)
        set_output(span, [{"source_doc": c["source_doc"], "score": c["score"]} for c in retrieved])

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
                "patient_id": patient_id,
                "injection_detected": injection["is_injection"],
                "guardrail_events": guardrails.summary()["events"],
            },
        }
        log_guardrail_events(guardrails, trace_seed=trace_id)
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

    # Recorded after the stream completes -- see record_generation's
    # docstring for why a span cannot wrap a yielding loop.
    record_generation(
        "llm.rag_generation",
        trace_seed=trace_id,
        input=generation_input,
        output=answer,
        model=DEFAULT_MODEL_ID,
        metadata={"chunks_used": len(ranked)},
    )

    # Role 5: Reflection -- RAG Triad
    triad = None
    if score_triad:
        yield {"type": "status", "payload": "Scoring answer quality (RAG Triad)..."}
        triad = await rag_triad_score(question, answer, ranked)

    # Guardrail: hallucination. Only meaningful when the triad actually
    # ran -- with scoring disabled there is nothing to gate on, and
    # blocking every answer in that mode would be wrong.
    hallucination = None
    if score_triad:
        # attempt=max_attempts: the tokens are already streamed to the
        # client, so regeneration is not available here. Force the
        # refusal instead of letting an ungrounded answer stand.
        hallucination = guardrails.check_answer(
            answer, triad, attempt=guardrails.hallucination.max_attempts
        )
        if hallucination["blocked"]:
            answer = hallucination["safe_answer"]
            yield {"type": "status", "payload": f"Answer blocked: {hallucination['reason']}"}

    yield {
        "type": "final",
        "payload": {
            "answer": answer,
            "sources": sources_for(ranked) if not (hallucination or {}).get("blocked") else [],
            "rag_triad": triad,
            "grounded": not (hallucination or {}).get("blocked", False),
            "patient_id": patient_id,
            "injection_detected": injection["is_injection"],
            "hallucination_blocked": (hallucination or {}).get("blocked", False),
            "faithfulness": (hallucination or {}).get("faithfulness"),
            "guardrail_events": guardrails.summary()["events"],
        },
    }

    log_guardrail_events(guardrails, trace_seed=trace_id)
