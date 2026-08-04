"""
agents/summary_agent/agent_executor.py

Bridges A2A to the Discharge Summary Generator, STREAMING (spec Table 10).

Expected inbound message shape (a single DataPart):
    {"patient_id": "P1019",
     "audience": "patient",     # or "clinician"; optional
     "trace_id": "<optional>"}

Streaming shape: one artifact per section, emitted as soon as that
section finishes, plus a status update announcing each section before it
starts. A client calling message/stream sees the summary build up
patient -> meds -> labs -> bill -> instructions; a client calling
message/send still gets every artifact in the final task, so
non-streaming callers are not broken.
"""

from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Part, TaskState, TextPart
from a2a.utils import get_data_parts, new_agent_text_message, new_task

from .sections import SECTION_ORDER, build_base_prompt, load_report, stream_section


class SummaryAgentExecutor(AgentExecutor):
    """AgentExecutor implementation for the Discharge Summary Generator."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        data_parts = get_data_parts(context.message.parts)
        if not data_parts or "patient_id" not in data_parts[0]:
            await updater.failed(
                message=new_agent_text_message("Expected a DataPart with 'patient_id'.")
            )
            return

        payload = data_parts[0]
        patient_id = payload["patient_id"]
        audience = payload.get("audience", "patient")
        trace_id = payload.get("trace_id") or str(uuid4())

        await updater.update_status(
            TaskState.working,
            message=new_agent_text_message(f"Loading validated report for {patient_id}..."),
        )

        try:
            report = load_report(patient_id)
            risk_level = report.get("risk_level", "low")
            base_prompt = await build_base_prompt(risk_level, audience)
        except Exception as exc:  # noqa: BLE001
            await updater.failed(
                message=new_agent_text_message(f"Could not start summary: {exc}")
            )
            return

        sections: dict[str, str] = {}

        for index, section in enumerate(SECTION_ORDER):
            await updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"Writing section {index + 1}/{len(SECTION_ORDER)}: {section}"
                ),
            )

            try:
                chunks = [
                    chunk async for chunk in stream_section(section, report, base_prompt)
                ]
            except Exception as exc:  # noqa: BLE001
                # One failed section shouldn't cost the patient the whole
                # summary -- record it and keep going.
                chunks = [f"(This section could not be generated: {exc})"]

            text = "".join(chunks).strip()
            sections[section] = text

            # Emit this section the moment it's ready -- this is the
            # progressive delivery the spec asks for.
            await updater.add_artifact(
                [Part(root=TextPart(text=text))],
                name=f"section-{section}",
                metadata={"section": section, "order": index, "trace_id": trace_id},
                last_chunk=True,
            )

        await updater.add_artifact(
            [
                Part(
                    root=DataPart(
                        data={
                            "patient_id": patient_id,
                            "risk_level": risk_level,
                            "audience": audience,
                            "sections": sections,
                            "trace_id": trace_id,
                        }
                    )
                )
            ],
            name="discharge_summary",
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported for the Discharge Summary Generator")
