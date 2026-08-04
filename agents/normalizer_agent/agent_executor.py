"""
agents/normalizer_agent/agent_executor.py

Bridges the A2A protocol (a2a-sdk) to the Clinical Normalizer Agent's
LangGraph graph, mirroring the Extractor Agent's executor.

Expected inbound message shape (a single DataPart):
    {"patient_id": "P1015",
     "doc_type": "doctor_reports",     # optional if raw_text is given
     "raw_text": "...",                 # optional if doc_type is given
     "trace_id": "<optional>"}

source_language is deliberately NOT accepted from the caller -- it is
always detected from the text, same contract as the Extractor Agent.
"""

from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Part, TaskState
from a2a.utils import get_data_parts, new_agent_text_message, new_task

from agents.common.a2a_server import traced_agent

from .graph import normalizer_app


class NormalizerAgentExecutor(AgentExecutor):
    """AgentExecutor implementation for the Clinical Normalizer Agent."""

    @traced_agent("normalizer")
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.update_status(
            TaskState.working,
            message=new_agent_text_message("Translating and normalizing clinical text..."),
        )

        data_parts = get_data_parts(context.message.parts)
        if not data_parts:
            await updater.failed(
                message=new_agent_text_message(
                    "Expected a DataPart with patient_id and either doc_type or raw_text."
                )
            )
            return

        payload = data_parts[0]

        if "patient_id" not in payload:
            await updater.failed(
                message=new_agent_text_message("Missing required field: 'patient_id'")
            )
            return

        if not payload.get("doc_type") and not payload.get("raw_text"):
            await updater.failed(
                message=new_agent_text_message(
                    "Supply either 'doc_type' (to harvest the document) or "
                    "'raw_text' (to normalize text directly)."
                )
            )
            return

        initial_state = {
            "patient_id": payload["patient_id"],
            "trace_id": payload.get("trace_id") or str(uuid4()),
        }
        if payload.get("doc_type"):
            initial_state["doc_type"] = payload["doc_type"]
        if payload.get("raw_text"):
            initial_state["raw_text"] = payload["raw_text"]

        thread_id = (
            f"{initial_state['patient_id']}:{payload.get('doc_type', 'inline')}:"
            f"{initial_state['trace_id']}"
        )

        final_state = await normalizer_app.ainvoke(
            initial_state, config={"configurable": {"thread_id": thread_id}}
        )

        artifact_data = {
            "patient_id": final_state["patient_id"],
            "doc_type": final_state.get("doc_type"),
            "source_language": final_state.get("source_language"),
            "translated_text": final_state.get("translated_text", ""),
            "normalized_text": final_state.get("normalized_text", ""),
            "confidence": final_state.get("confidence", 0.0),
            "low_confidence": final_state.get("low_confidence", True),
            "expanded_abbreviations": final_state.get("expanded_abbreviations", []),
            "model_used": final_state.get("model_used"),
            "trace_id": initial_state["trace_id"],
        }

        await updater.add_artifact(
            [Part(root=DataPart(data=artifact_data))],
            name="normalized_text",
        )

        if final_state.get("error"):
            await updater.failed(message=new_agent_text_message(final_state["error"]))
        else:
            await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported for the Normalizer Agent")
