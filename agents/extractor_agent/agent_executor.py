"""
agents/extractor_agent/agent_executor.py

Bridges the A2A protocol (a2a-sdk) to the Clinical Extractor Agent's
LangGraph graph. This is the piece the spec means by "Tools +
Resources + Prompts" agent wired onto A2A -- NOT a hand-rolled FastAPI
route, but a real a2a.server.agent_execution.AgentExecutor plugged
into a2a-sdk's DefaultRequestHandler/A2AStarletteApplication (see
server.py).

Expected inbound message shape (sent as a single DataPart, since this
task is structured, not free text):
    {"patient_id": "P1001", "doc_type": "doctor_reports",
     "language": "en", "trace_id": "<optional>"}
"""

from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Part, TaskState
from a2a.utils import get_data_parts, new_agent_text_message, new_task

from agents.common.a2a_server import traced_agent

from .graph import extractor_app


class ExtractorAgentExecutor(AgentExecutor):
    """AgentExecutor implementation for the Clinical Extractor Agent."""

    @traced_agent("extractor")
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.update_status(
            TaskState.working,
            message=new_agent_text_message("Harvesting document and extracting fields..."),
        )

        data_parts = get_data_parts(context.message.parts)
        if not data_parts:
            await updater.failed(
                message=new_agent_text_message(
                    "Expected a DataPart with patient_id/doc_type/language."
                )
            )
            return

        payload = data_parts[0]

        try:
            initial_state = {
                "patient_id": payload["patient_id"],
                "doc_type": payload["doc_type"],
                "trace_id": payload.get("trace_id") or str(uuid4()),
            }
        except KeyError as e:
            await updater.failed(
                message=new_agent_text_message(f"Missing required field: {e}")
            )
            return

        thread_id = (
            f"{initial_state['patient_id']}:{initial_state['doc_type']}:"
            f"{initial_state['trace_id']}"
        )

        final_state = await extractor_app.ainvoke(
            initial_state, config={"configurable": {"thread_id": thread_id}}
        )

        artifact_data = {
            "patient_id": final_state["patient_id"],
            "doc_type": final_state["doc_type"],
            "language": final_state.get("language"),
            "extracted_fields": final_state.get("extracted_fields", {}),
            "trace_id": initial_state["trace_id"],
        }

        await updater.add_artifact(
            [Part(root=DataPart(data=artifact_data))],
            name="extracted_fields",
        )

        if final_state.get("error"):
            await updater.failed(message=new_agent_text_message(final_state["error"]))
        else:
            await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported for the Extractor Agent")