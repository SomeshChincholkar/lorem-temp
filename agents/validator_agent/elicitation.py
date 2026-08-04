"""
agents/validator_agent/elicitation.py

The CLIENT half of the MCP Elicitation primitive (spec 2.4.1).

When clinical_rules_engine_tool finds non-blocking missing fields it
calls ctx.elicit(). The SDK routes that request to the elicitation
callback registered on the session that made the tool call -- i.e. into
THIS agent's process, not into Streamlit.

Since the human reviewer is in a different process, this callback does
not render anything. It parks the request in the shared elicitation
store, blocks until a reviewer answers via dashboard page 3 (or the
timeout expires), then translates that answer back into an ElicitResult.

The three outcomes the spec requires are all handled:
    accept  -> reviewer's values flow back, tool continues
    decline -> tool marks the fields unresolved and flags for HITL
    cancel  -> tool aborts and escalates
A timeout maps to decline (see elicitation_store.await_response).
"""

from mcp.types import (
    INVALID_PARAMS,
    ElicitRequestFormParams,
    ElicitResult,
    ErrorData,
)

from agents.common import elicitation_store
from observability import log_elicitation_event

# Store status -> MCP ElicitResult action.
STATUS_TO_ACTION = {
    elicitation_store.ACCEPTED: "accept",
    elicitation_store.DECLINED: "decline",
    elicitation_store.CANCELLED: "cancel",
}


def make_elicitation_callback(
    patient_id: str | None = None,
    doc_type: str | None = None,
    trace_id: str | None = None,
    timeout_seconds: float | None = None,
):
    """
    Build the elicitation callback for one graph run.

    patient_id/doc_type/trace_id are closed over so the parked request
    carries enough context for the dashboard to tell the reviewer which
    case they are being asked about -- the MCP request itself only
    carries a message string and a schema.
    """

    async def elicitation_callback(context, params):
        # URL-mode elicitation asks the client to send the user to a web
        # page. This system's reviewer flow is the dashboard's own form,
        # so decline rather than pretend to support it.
        if not isinstance(params, ElicitRequestFormParams):
            return ElicitResult(action="decline")

        schema = params.requestedSchema
        if not isinstance(schema, dict):
            return ErrorData(
                code=INVALID_PARAMS,
                message="Elicitation requestedSchema was not a JSON Schema object.",
            )

        request_id = elicitation_store.create_request(
            message=params.message,
            schema=schema,
            patient_id=patient_id,
            doc_type=doc_type,
            trace_id=trace_id,
        )

        record = await elicitation_store.await_response(
            request_id,
            timeout_seconds=(
                elicitation_store.DEFAULT_TIMEOUT_SECONDS
                if timeout_seconds is None
                else timeout_seconds
            ),
        )

        action = STATUS_TO_ACTION.get(record.get("status"), "decline")

        # Spec 7.2: schema sent, reviewer response, action taken.
        log_elicitation_event(
            trace_id,
            schema=schema,
            action=action,
            reviewer_response=record.get("data"),
            timed_out=bool(record.get("timed_out")),
        )

        # content is only meaningful on accept; the SDK validates it
        # against requestedSchema, so send nothing on decline/cancel.
        if action == "accept":
            return ElicitResult(action="accept", content=record.get("data") or {})
        return ElicitResult(action=action)

    return elicitation_callback
