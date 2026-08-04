"""
agents/monitor_agent/agent_executor.py

Bridges A2A to the Discharge Monitor Agent (Google ADK).

Expected inbound message shape (a single DataPart):
    {"trigger": true,
     "subfolder": "lab_reports",   # optional
     "use_llm": false,             # optional, see below
     "trace_id": "<optional>"}

On use_llm: by default this executor calls the scan tool directly rather
than routing through the ADK Runner. Listing files is a deterministic
operation, and putting a language model between "what's on disk" and the
Orchestrator's work queue would add latency, cost and a hallucination
surface for no benefit -- the Orchestrator consumes this list
programmatically.

Passing use_llm=true runs the same request through the ADK LlmAgent
instead, which is the path the Gradio/conversational UI uses. Either
way the structured document list returned to the caller comes from the
tool itself, never from model prose.
"""

from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Part, TaskState
from a2a.utils import get_data_parts, new_agent_text_message, new_task

from agents.common.adk_runtime import run_adk_agent

from .agent import monitor_agent, scan_for_new_documents


class MonitorAgentExecutor(AgentExecutor):
    """AgentExecutor implementation for the Discharge Monitor Agent."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.update_status(
            TaskState.working,
            message=new_agent_text_message("Scanning the authorized input root..."),
        )

        data_parts = get_data_parts(context.message.parts)
        payload = data_parts[0] if data_parts else {}

        subfolder = payload.get("subfolder", "") or ""
        trace_id = payload.get("trace_id") or str(uuid4())

        try:
            if payload.get("use_llm"):
                question = (
                    f"What new discharge documents are in the {subfolder} folder?"
                    if subfolder
                    else "What new discharge documents have arrived?"
                )
                run = await run_adk_agent(
                    monitor_agent,
                    question,
                    app_name="discharge_monitor",
                    session_id=f"monitor-{trace_id}",
                )
                # Prefer the tool's own structured output over the
                # model's summary of it.
                scan = next(
                    (
                        r["response"]
                        for r in run["tool_results"]
                        if r.get("name") == "scan_for_new_documents" and r.get("response")
                    ),
                    None,
                )
                if scan is None:
                    scan = await scan_for_new_documents(subfolder)
                summary = run["text"]
            else:
                scan = await scan_for_new_documents(subfolder)
                summary = f"Found {scan['count']} new document(s)."
        except Exception as exc:  # noqa: BLE001
            await updater.failed(
                message=new_agent_text_message(f"Watcher scan failed: {exc}")
            )
            return

        artifact_data = {
            "documents": scan.get("documents", []),
            "count": scan.get("count", 0),
            "patient_ids": scan.get("patient_ids", []),
            "summary": summary,
            "trace_id": trace_id,
        }

        await updater.add_artifact(
            [Part(root=DataPart(data=artifact_data))],
            name="new_documents",
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported for the Discharge Monitor Agent")
