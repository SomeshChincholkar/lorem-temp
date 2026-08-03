"""
tests/test_normalizer_nodes.py

Covers node_normalize_abbrev -- the deterministic abbreviation pass that
runs on top of the LLM's translation, driven by
resource://medical-abbreviations (i.e. by rules.yaml).

This is the part of normalization that must be reproducible for audit,
so the edge cases (word boundaries, case sensitivity) are pinned here.

Run:  python -m pytest tests/test_normalizer_nodes.py -q
"""

import json

import pytest

from agents.normalizer_agent import nodes as nodes_module
from agents.normalizer_agent.nodes import node_normalize_abbrev

ABBREVIATIONS = {
    "HTN": "Hypertension",
    "BID": "twice daily",
    "PO": "by mouth",
    "MI": "Myocardial Infarction",
    "Temp": "Temperature",
}


@pytest.fixture
def abbrev_resource(monkeypatch):
    """Serves a fixed abbreviation map in place of the MCP resource."""

    async def fake_read_resource_text(uri):
        assert uri == "resource://medical-abbreviations"
        return json.dumps(ABBREVIATIONS)

    monkeypatch.setattr(nodes_module, "read_resource_text", fake_read_resource_text)


@pytest.mark.asyncio
async def test_expands_abbreviations_and_reports_them(abbrev_resource):
    state = {
        "translated_text": "Patient has HTN. Take Metformin BID PO.",
        "confidence": 0.95,
    }
    result = await node_normalize_abbrev(state)

    assert result["normalized_text"] == (
        "Patient has Hypertension. Take Metformin twice daily by mouth."
    )
    expanded = {e["abbreviation"]: e["count"] for e in result["expanded_abbreviations"]}
    assert expanded == {"HTN": 1, "BID": 1, "PO": 1}


@pytest.mark.asyncio
async def test_respects_word_boundaries(abbrev_resource):
    """
    'PO' must not rewrite the middle of 'POST' or 'HYPO'. Without \\b this
    silently corrupts clinical text.
    """
    state = {"translated_text": "POST-op HYPOglycemia noted.", "confidence": 0.9}
    result = await node_normalize_abbrev(state)

    assert result["normalized_text"] == "POST-op HYPOglycemia noted."
    assert result["expanded_abbreviations"] == []


@pytest.mark.asyncio
async def test_is_case_sensitive(abbrev_resource):
    """
    Abbreviations are uppercase by convention. Matching case-insensitively
    would turn Spanish 'mi' into 'Myocardial Infarction' and the prose
    word 'temp' into 'Temperature'.
    """
    state = {"translated_text": "mi paciente; the temp was normal.", "confidence": 0.9}
    result = await node_normalize_abbrev(state)

    assert result["normalized_text"] == "mi paciente; the temp was normal."
    assert result["expanded_abbreviations"] == []


@pytest.mark.asyncio
async def test_counts_repeated_occurrences(abbrev_resource):
    state = {"translated_text": "HTN history; HTN controlled.", "confidence": 0.9}
    result = await node_normalize_abbrev(state)

    assert result["normalized_text"] == "Hypertension history; Hypertension controlled."
    assert result["expanded_abbreviations"] == [
        {"abbreviation": "HTN", "expansion": "Hypertension", "count": 2}
    ]


@pytest.mark.asyncio
async def test_low_confidence_flag_uses_rules_yaml_threshold(abbrev_resource):
    """
    rules.yaml sets quality_thresholds.translation_confidence_min: 0.70,
    and risk_scoring_matrix weights low_translation_confidence at 3.
    """
    below = await node_normalize_abbrev({"translated_text": "ok", "confidence": 0.55})
    assert below["low_confidence"] is True

    above = await node_normalize_abbrev({"translated_text": "ok", "confidence": 0.85})
    assert above["low_confidence"] is False


@pytest.mark.asyncio
async def test_empty_translation_is_low_confidence(abbrev_resource):
    """No text means the translation failed -- must not read as healthy."""
    result = await node_normalize_abbrev({"translated_text": "", "confidence": 0.99})

    assert result["normalized_text"] == ""
    assert result["low_confidence"] is True


@pytest.mark.asyncio
async def test_unreachable_resource_passes_text_through(monkeypatch):
    """
    If the MCP resource is down, the LLM's translation is still worth
    returning -- degrade, don't fail the run.
    """

    async def boom(uri):
        raise RuntimeError("resource server unreachable")

    monkeypatch.setattr(nodes_module, "read_resource_text", boom)

    result = await node_normalize_abbrev(
        {"translated_text": "Patient has HTN.", "confidence": 0.9}
    )

    assert result["normalized_text"] == "Patient has HTN."
    assert result["expanded_abbreviations"] == []
