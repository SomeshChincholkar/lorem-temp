"""
tests/test_elicitation.py

Covers the cross-process Elicitation rendezvous and the Validator's
elicitation callback.

This is the part of the system with the least obvious correctness: the
MCP server raises ctx.elicit() in one process, the reviewer answers in
another, and the three outcomes the spec mandates (accept / decline /
cancel) all have to survive that hop -- plus the case nobody plans for,
where no reviewer is watching at all.

Run:  python -m pytest tests/test_elicitation.py -q
"""

import asyncio

import pytest
from mcp.types import ElicitRequestFormParams

from agents.common import elicitation_store
from agents.validator_agent.elicitation import make_elicitation_callback

SCHEMA = {
    "type": "object",
    "properties": {"ward": {"type": "string"}, "bed_no": {"type": "string"}},
    "required": ["ward", "bed_no"],
}


@pytest.fixture(autouse=True)
def temp_store(tmp_path, monkeypatch):
    """Point the store at a temp dir so tests never touch Data/."""
    monkeypatch.setattr(elicitation_store, "ELICITATION_DIR", tmp_path / "elicitations")


def make_params(message="Missing fields", schema=None):
    return ElicitRequestFormParams(
        mode="form", message=message, requestedSchema=schema or SCHEMA
    )


# ---------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------
def test_created_request_is_pending_and_listed():
    request_id = elicitation_store.create_request(
        message="Missing ward", schema=SCHEMA, patient_id="P1019"
    )

    record = elicitation_store.get_request(request_id)
    assert record["status"] == elicitation_store.PENDING
    assert record["patient_id"] == "P1019"
    assert record["schema"] == SCHEMA

    pending = elicitation_store.list_pending()
    assert [r["request_id"] for r in pending] == [request_id]


def test_responding_removes_it_from_pending():
    request_id = elicitation_store.create_request(message="m", schema=SCHEMA)

    assert elicitation_store.respond(request_id, elicitation_store.ACCEPTED, {"ward": "3B"})
    assert elicitation_store.list_pending() == []
    assert elicitation_store.get_request(request_id)["data"] == {"ward": "3B"}


def test_second_response_is_rejected():
    """A reviewer must not be able to overwrite an answered request."""
    request_id = elicitation_store.create_request(message="m", schema=SCHEMA)

    assert elicitation_store.respond(request_id, elicitation_store.DECLINED)
    assert not elicitation_store.respond(request_id, elicitation_store.ACCEPTED, {"ward": "x"})
    assert elicitation_store.get_request(request_id)["status"] == elicitation_store.DECLINED


def test_invalid_action_is_rejected():
    request_id = elicitation_store.create_request(message="m", schema=SCHEMA)
    with pytest.raises(ValueError):
        elicitation_store.respond(request_id, "approved")


def test_data_is_dropped_on_decline():
    """Declining must not smuggle field values through."""
    request_id = elicitation_store.create_request(message="m", schema=SCHEMA)
    elicitation_store.respond(request_id, elicitation_store.DECLINED, {"ward": "3B"})
    assert elicitation_store.get_request(request_id)["data"] is None


@pytest.mark.asyncio
async def test_await_response_returns_once_answered():
    request_id = elicitation_store.create_request(message="m", schema=SCHEMA)

    async def answer_shortly():
        await asyncio.sleep(0.1)
        elicitation_store.respond(request_id, elicitation_store.ACCEPTED, {"ward": "3B"})

    asyncio.create_task(answer_shortly())
    record = await elicitation_store.await_response(request_id, timeout_seconds=5)

    assert record["status"] == elicitation_store.ACCEPTED
    assert record["data"] == {"ward": "3B"}


@pytest.mark.asyncio
async def test_timeout_declines_rather_than_hanging():
    """
    Nobody watching the dashboard must not wedge the Validator. Timing
    out declines (unresolved -> flag for HITL) rather than cancelling
    (abort and escalate), which would overstate what happened.
    """
    request_id = elicitation_store.create_request(message="m", schema=SCHEMA)
    record = await elicitation_store.await_response(request_id, timeout_seconds=0)

    assert record["status"] == elicitation_store.DECLINED
    assert record["timed_out"] is True


# ---------------------------------------------------------------------
# The callback -- all three spec-mandated outcomes
# ---------------------------------------------------------------------
async def _run_callback_with(action, data=None, timeout=5):
    callback = make_elicitation_callback(patient_id="P1019", timeout_seconds=timeout)

    async def answer_when_parked():
        for _ in range(100):
            pending = elicitation_store.list_pending()
            if pending:
                elicitation_store.respond(pending[0]["request_id"], action, data)
                return
            await asyncio.sleep(0.05)

    asyncio.create_task(answer_when_parked())
    return await callback(None, make_params())


@pytest.mark.asyncio
async def test_callback_accept_returns_reviewer_values():
    result = await _run_callback_with(
        elicitation_store.ACCEPTED, {"ward": "3B", "bed_no": "14"}
    )
    assert result.action == "accept"
    assert result.content == {"ward": "3B", "bed_no": "14"}


@pytest.mark.asyncio
async def test_callback_decline_carries_no_content():
    result = await _run_callback_with(elicitation_store.DECLINED)
    assert result.action == "decline"
    assert not result.content


@pytest.mark.asyncio
async def test_callback_cancel_maps_to_cancel():
    result = await _run_callback_with(elicitation_store.CANCELLED)
    assert result.action == "cancel"


@pytest.mark.asyncio
async def test_callback_declines_on_timeout():
    callback = make_elicitation_callback(patient_id="P1019", timeout_seconds=0)
    result = await callback(None, make_params())
    assert result.action == "decline"


@pytest.mark.asyncio
async def test_callback_carries_patient_context_to_the_dashboard():
    """
    The MCP request only carries a message and a schema. Without the
    agent attaching patient context, a reviewer would be asked to fill
    in fields with no idea which case they belong to.
    """
    callback = make_elicitation_callback(
        patient_id="P1016", doc_type="discharge_report", trace_id="trace-xyz",
        timeout_seconds=0,
    )
    await callback(None, make_params())

    # timeout_seconds=0 declines immediately, so read the answered record.
    records = list((elicitation_store.ELICITATION_DIR).glob("*.json"))
    assert len(records) == 1

    import json

    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert record["patient_id"] == "P1016"
    assert record["doc_type"] == "discharge_report"
    assert record["trace_id"] == "trace-xyz"
