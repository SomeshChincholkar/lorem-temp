"""
tests/test_orchestrator_and_rag.py

Covers the orchestrator's pure logic (guardrail + extraction unwrapping)
and the RAG roles that don't need an embedding model loaded.

Run:  python -m pytest tests/test_orchestrator_and_rag.py -q
"""

import pytest

from agents.orchestrator import pipeline as pipeline_module
from agents.orchestrator.pipeline import (
    _unwrap_fields,
    guardrail_manager,
    run_discharge_pipeline,
)
from agents.rag_agent.indexing import chunk_document
from agents.rag_agent.roles import MIN_RELEVANCE_SCORE, has_grounding, rerank_by_keyword


# ---------------------------------------------------------------------
# Guardrail (spec Table 12: HITL Escalation)
# ---------------------------------------------------------------------
def test_high_risk_requires_hitl():
    assert guardrail_manager("high", False)["requires_hitl"] is True


def test_blocked_requires_hitl_even_at_low_risk():
    assert guardrail_manager("low", True)["requires_hitl"] is True


def test_low_risk_unblocked_can_auto_approve():
    assert guardrail_manager("low", False)["requires_hitl"] is False


def test_missing_signals_do_not_auto_approve_a_high_case():
    """Risk level casing must not be a way around the guardrail."""
    assert guardrail_manager("HIGH", None)["requires_hitl"] is True


# ---------------------------------------------------------------------
# Extraction payload unwrapping
# ---------------------------------------------------------------------
def test_unwraps_single_document_shape():
    extracted = {"doc_type": "bill", "fields": {"patient_id": "P1019", "total_amount": 10}}
    assert _unwrap_fields(extracted) == {"patient_id": "P1019", "total_amount": 10}


def test_unwraps_multi_document_shape():
    extracted = {"documents": [{"doc_type": "bill", "fields": {"patient_id": "P1019"}}]}
    assert _unwrap_fields(extracted) == {"patient_id": "P1019"}


def test_passes_through_already_flat_fields():
    assert _unwrap_fields({"patient_id": "P1019"}) == {"patient_id": "P1019"}


def test_handles_none_and_junk():
    assert _unwrap_fields(None) == {}
    assert _unwrap_fields("not a dict") == {}


# ---------------------------------------------------------------------
# Full pipeline sequencing, with every A2A call stubbed
# ---------------------------------------------------------------------
def ok(artifact):
    return {"ok": True, "state": "completed", "artifacts": [artifact], "error": None}


def down(error="connection refused"):
    return {"ok": False, "state": "unreachable", "artifacts": [], "error": error}


@pytest.fixture
def stub_a2a(monkeypatch):
    """
    Replaces the A2A client with a recorder. Returns the call log plus a
    mutable response map keyed by agent name.
    """
    calls = []
    responses = {
        "monitor": ok(
            {
                "documents": [
                    {"patient_id": "P1016", "doc_type": "doctor_reports"},
                    {"patient_id": "P1016", "doc_type": "bills"},
                    {"patient_id": "P9999", "doc_type": "bills"},
                ]
            }
        ),
        "extractor": ok(
            {"extracted_fields": {"fields": {"patient_id": "P1016"}}, "language": "de"}
        ),
        "normalizer": ok({"confidence": 0.55}),
        "validator": ok(
            {
                "final_status": "blocked",
                "risk_level": "high",
                "recommendation": "Urgent Attention — Block release",
                "discharge_blocked": True,
                "ehr_findings": [],
                "json_path": "Data/reports/P1016_report.json",
            }
        ),
    }

    async def fake_send_message(agent, data, timeout=None):
        calls.append({"agent": agent, "data": data})
        return responses[agent]

    monkeypatch.setattr(pipeline_module, "send_message", fake_send_message)
    return {"calls": calls, "responses": responses}


@pytest.mark.asyncio
async def test_pipeline_runs_agents_in_order(stub_a2a):
    result = await run_discharge_pipeline("P1016")

    agents_called = [c["agent"] for c in stub_a2a["calls"]]
    assert agents_called[0] == "monitor"
    assert agents_called[-1] == "validator"
    assert result["risk_level"] == "high"
    assert result["requires_hitl"] is True


@pytest.mark.asyncio
async def test_pipeline_only_extracts_document_types_that_exist(stub_a2a):
    """
    The Monitor said this patient has a discharge report and a bill but
    no lab report, so no lab extraction should be attempted.
    """
    await run_discharge_pipeline("P1016")

    extracted_types = [
        c["data"]["doc_type"] for c in stub_a2a["calls"] if c["agent"] == "extractor"
    ]
    assert sorted(extracted_types) == ["bills", "doctor_reports"]


@pytest.mark.asyncio
async def test_pipeline_normalizes_only_non_english_documents(stub_a2a):
    """German documents go to the Normalizer; English ones must not."""
    await run_discharge_pipeline("P1016")
    assert any(c["agent"] == "normalizer" for c in stub_a2a["calls"])

    stub_a2a["calls"].clear()
    stub_a2a["responses"]["extractor"] = ok(
        {"extracted_fields": {"fields": {"patient_id": "P1019"}}, "language": "en"}
    )
    await run_discharge_pipeline("P1019")
    assert not any(c["agent"] == "normalizer" for c in stub_a2a["calls"])


@pytest.mark.asyncio
async def test_weakest_translation_confidence_is_reported(stub_a2a):
    """
    Two documents, one poorly translated: the case must carry the worst
    confidence, not an average that hides it.
    """
    confidences = iter([0.95, 0.40])

    async def fake_send_message(agent, data, timeout=None):
        stub_a2a["calls"].append({"agent": agent, "data": data})
        if agent == "normalizer":
            return ok({"confidence": next(confidences)})
        return stub_a2a["responses"][agent]

    import agents.orchestrator.pipeline as p

    p.send_message = fake_send_message
    result = await run_discharge_pipeline("P1016")

    assert result["translation_confidence"] == 0.40


@pytest.mark.asyncio
async def test_unreachable_agent_is_reported_not_swallowed(stub_a2a):
    """
    A dead Extractor must surface as a failed step, and the run must not
    end up claiming the patient is fine.
    """
    stub_a2a["responses"]["extractor"] = down()

    result = await run_discharge_pipeline("P1016")

    failed = [s for s in result["steps"] if not s["ok"]]
    assert any(s["step"] == "extract" for s in failed)


@pytest.mark.asyncio
async def test_failed_validation_does_not_auto_approve(stub_a2a):
    """The worst possible bug would be treating "no verdict" as approval."""
    stub_a2a["responses"]["validator"] = down()

    result = await run_discharge_pipeline("P1016")

    assert result["ok"] is False
    assert result["final_status"] is None
    assert result["risk_level"] is None


# ---------------------------------------------------------------------
# RAG: chunking
# ---------------------------------------------------------------------
def test_short_document_is_one_chunk():
    assert chunk_document("short text") == ["short text"]


def test_empty_document_yields_no_chunks():
    assert chunk_document("") == []
    assert chunk_document("   ") == []


def test_long_document_is_chunked_with_overlap():
    text = "A" * 1200
    chunks = chunk_document(text, size=500, overlap=50)

    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    # Overlap means the chunks together cover more than the raw length.
    assert sum(len(c) for c in chunks) > len(text)


def test_chunking_terminates_on_exact_multiple():
    """A size/overlap boundary must not spin out extra empty chunks."""
    chunks = chunk_document("B" * 500, size=500, overlap=50)
    assert chunks == ["B" * 500]


# ---------------------------------------------------------------------
# RAG: grounding gate
# ---------------------------------------------------------------------
def test_irrelevant_chunks_are_not_grounding():
    """
    This gate is what produces the spec's exact refusal string instead
    of an answer confabulated from the least-bad chunk in the corpus.
    """
    assert has_grounding([{"score": MIN_RELEVANCE_SCORE - 0.01}]) is False


def test_relevant_chunk_is_grounding():
    assert has_grounding([{"score": MIN_RELEVANCE_SCORE + 0.01}]) is True


def test_no_chunks_is_not_grounding():
    assert has_grounding([]) is False


# ---------------------------------------------------------------------
# RAG: keyword re-ranking (Augmentation role)
# ---------------------------------------------------------------------
def test_literal_keyword_match_outranks_higher_vector_score():
    """
    Drug names are exactly where pure embedding search is weakest. A
    chunk that literally says "Amoxicillin" must beat a semantically
    nearby one that doesn't.
    """
    chunks = [
        {"chunk": "The patient received antibiotics during the stay.", "score": 0.90,
         "source_doc": "a.txt"},
        {"chunk": "Discharge medication: Amoxicillin 500 mg three times daily.",
         "score": 0.70, "source_doc": "b.txt"},
    ]

    ranked = rerank_by_keyword("Was Amoxicillin prescribed?", chunks)
    assert ranked[0]["source_doc"] == "b.txt"


def test_rerank_keeps_every_chunk():
    chunks = [
        {"chunk": "one", "score": 0.5, "source_doc": "a"},
        {"chunk": "two", "score": 0.4, "source_doc": "b"},
    ]
    assert len(rerank_by_keyword("anything at all", chunks)) == 2


def test_rerank_with_only_stopwords_leaves_order_alone():
    chunks = [
        {"chunk": "first", "score": 0.9, "source_doc": "a"},
        {"chunk": "second", "score": 0.1, "source_doc": "b"},
    ]
    ranked = rerank_by_keyword("what is the", chunks)
    assert [c["source_doc"] for c in ranked] == ["a", "b"]
