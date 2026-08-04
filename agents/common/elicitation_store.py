"""
agents/common/elicitation_store.py

Cross-process rendezvous for MCP Elicitation.

The problem this solves: ctx.elicit() is raised by the Rules Engine tool
inside the MCP server, and the SDK routes it to the elicitation_callback
of whichever client session made the tool call -- which lives in the
Clinical Validation Agent process (:8101). But the human reviewer is in
the Streamlit HITL dashboard (:8501), a completely separate process.
The callback therefore cannot render the form itself; it has to park the
request somewhere the dashboard can see it, and block until a reviewer
answers.

This module is that somewhere: one JSON file per request under
Data/elicitations/. Deliberately file-backed rather than Redis/DB --
the request rate is one-per-HITL-review, and a dependency-free store
keeps the whole system runnable with `python -m ...` and nothing else.

Lifecycle of a request file:
    pending  -> written by the Validator's elicitation callback
    accepted | declined | cancelled -> written by the dashboard
    (the callback polls for the transition, then returns ElicitResult)

Writes are atomic (write temp + os.replace) so the dashboard can never
read a half-written file.
"""

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

ELICITATION_DIR = Path(
    os.getenv("ELICITATION_DIR", "Data/elicitations")
).resolve()

# How long the Validator agent waits for a human before giving up.
# On timeout the callback declines, which marks the fields unresolved
# and flags the case for HITL -- the safe outcome. Set to 0 for
# unattended runs (CI, orchestrator smoke tests) so nothing blocks.
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("ELICITATION_TIMEOUT_SECONDS", "120"))

POLL_INTERVAL_SECONDS = 0.5

PENDING = "pending"
ACCEPTED = "accepted"
DECLINED = "declined"
CANCELLED = "cancelled"

TERMINAL_STATUSES = {ACCEPTED, DECLINED, CANCELLED}


def _path_for(request_id: str) -> Path:
    return ELICITATION_DIR / f"{request_id}.json"


def _write_atomic(path: Path, payload: dict) -> None:
    """
    Write via a temp file + os.replace so a concurrent reader never sees
    a partially written request.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def create_request(
    message: str,
    schema: dict,
    patient_id: Optional[str] = None,
    doc_type: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> str:
    """
    Park a new elicitation request for the dashboard to pick up.

    Args:
        message: the server's prompt to the reviewer.
        schema: the JSON Schema the server sent (drives the dynamic form).
        patient_id/doc_type/trace_id: context so the dashboard can show
            the reviewer what they're being asked about.

    Returns:
        request_id
    """
    request_id = uuid.uuid4().hex
    _write_atomic(
        _path_for(request_id),
        {
            "request_id": request_id,
            "status": PENDING,
            "message": message,
            "schema": schema,
            "patient_id": patient_id,
            "doc_type": doc_type,
            "trace_id": trace_id,
            "created_at": time.time(),
            "data": None,
        },
    )
    return request_id


def get_request(request_id: str) -> Optional[dict]:
    path = _path_for(request_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # A reader that lost a race with a writer -- treat as "not ready
        # yet" rather than an error; the caller is polling anyway.
        return None


def list_pending() -> list[dict]:
    """Every unanswered request, oldest first. Used by dashboard page 3."""
    if not ELICITATION_DIR.exists():
        return []
    requests = []
    for path in ELICITATION_DIR.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if record.get("status") == PENDING:
            requests.append(record)
    return sorted(requests, key=lambda r: r.get("created_at", 0))


def respond(request_id: str, action: str, data: Optional[dict] = None) -> bool:
    """
    Record a reviewer's answer. Called by the Streamlit dashboard.

    Args:
        action: "accepted" | "declined" | "cancelled"
        data: field values, only meaningful when action == "accepted"

    Returns:
        True if the request existed and was still pending.
    """
    if action not in TERMINAL_STATUSES:
        raise ValueError(f"Invalid action '{action}'. Must be one of {sorted(TERMINAL_STATUSES)}")

    record = get_request(request_id)
    if record is None or record.get("status") != PENDING:
        return False

    record["status"] = action
    record["data"] = data if action == ACCEPTED else None
    record["responded_at"] = time.time()
    _write_atomic(_path_for(request_id), record)
    return True


async def await_response(
    request_id: str, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
) -> dict:
    """
    Poll until the request reaches a terminal status or the timeout
    expires.

    On timeout the request is marked declined (not cancelled): declining
    means "unresolved, flag for HITL", which is the correct outcome for
    "nobody was watching the dashboard". Cancelling would abort and
    escalate, which overstates what actually happened.

    Returns the final record.
    """
    deadline = time.monotonic() + max(0.0, timeout_seconds)

    while True:
        record = get_request(request_id)
        if record and record.get("status") in TERMINAL_STATUSES:
            return record

        if time.monotonic() >= deadline:
            respond(request_id, DECLINED)
            timed_out = get_request(request_id) or {
                "request_id": request_id,
                "status": DECLINED,
                "data": None,
            }
            timed_out["timed_out"] = True
            return timed_out

        await asyncio.sleep(POLL_INTERVAL_SECONDS)
