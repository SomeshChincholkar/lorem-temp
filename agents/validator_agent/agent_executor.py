"""
agents/validator_agent/agent_executor.py

Bridges A2A to the Clinical Validation Agent's LangGraph graph.

Expected inbound message shape (a single DataPart):
    {"patient_id": "P1019",
     "extracted_discharge": {...},          # Table 3 discharge fields
     "extracted_bill": {...},               # Table 3 bill fields
     "extracted_lab": {...},                # Table 3 lab fields
     "translation_confidence": 0.93,        # optional, from the Normalizer
     "trace_id": "<optional>"}

All three extracted_* dicts are persisted into the audit report, which
is the only thing downstream reads -- the Summary Generator and the
dashboard's medication tables build from them.
"""

from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Part, TaskState
from a2a.utils import get_data_parts, new_agent_text_message, new_task

from agents.common.a2a_server import traced_agent

from .graph import validator_app


class ValidatorAgentExecutor(AgentExecutor):
    """AgentExecutor implementation for the Clinical Validation Agent."""

    @traced_agent("validator")
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.update_status(
            TaskState.working,
            message=new_agent_text_message(
                "Validating completeness and cross-checking against the EHR..."
            ),
        )

        data_parts = get_data_parts(context.message.parts)
        if not data_parts:
            await updater.failed(
                message=new_agent_text_message(
                    "Expected a DataPart with patient_id and extracted document fields."
                )
            )
            return

        payload = data_parts[0]

        if "patient_id" not in payload:
            await updater.failed(
                message=new_agent_text_message("Missing required field: 'patient_id'")
            )
            return

        initial_state = {
            "patient_id": payload["patient_id"],
            "extracted_discharge": payload.get("extracted_discharge") or {},
            "extracted_bill": payload.get("extracted_bill") or {},
            "extracted_lab": payload.get("extracted_lab") or {},
            "translation_confidence": payload.get("translation_confidence"),
            "trace_id": payload.get("trace_id") or str(uuid4()),
        }

        thread_id = f"{initial_state['patient_id']}:validate:{initial_state['trace_id']}"

        final_state = await validator_app.ainvoke(
            initial_state, config={"configurable": {"thread_id": thread_id}}
        )

        report = final_state.get("report") or {}
        artifact_data = {
            "patient_id": final_state["patient_id"],
            "final_status": final_state.get("final_status"),
            "completeness_result": final_state.get("completeness_result", {}),
            "completeness_gaps": final_state.get("completeness_gaps", {}),
            "ehr_findings": final_state.get("ehr_findings", []),
            "risk_level": report.get("risk_level"),
            "recommendation": report.get("recommendation"),
            "discharge_blocked": report.get("discharge_blocked"),
            "json_path": report.get("json_path"),
            "html_path": report.get("html_path"),
            "trace_id": initial_state["trace_id"],
        }

        await updater.add_artifact(
            [Part(root=DataPart(data=artifact_data))],
            name="validation_report",
        )

        if final_state.get("error"):
            await updater.failed(message=new_agent_text_message(final_state["error"]))
        else:
            await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported for the Validation Agent")
