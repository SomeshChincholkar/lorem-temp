"""
agents/rag_agent/agent_executor.py

Bridges A2A to the Clinical RAG Q&A Agent, STREAMING (spec Table 10:
"token-by-token answer streaming in HITL Q&A page").

Expected inbound message shape (a single DataPart):
    {"question": "What medications was this patient discharged on?",
     "patient_id": "P1019",       # REQUIRED
     "top_k": 5,                   # optional
     "trace_id": "<optional>"}

patient_id is mandatory: each patient has their own FAISS index, so a
question with no patient has no index to search. Requests without one are
rejected here rather than silently answered from the wrong records.
`patient_filter` is accepted as a deprecated alias.

Streaming shape: each generated token batch goes out as its own text
artifact chunk appended to one artifact, so a streaming client paints
the answer as it forms. The final DataPart artifact carries the complete
answer plus sources and RAG Triad scores, so non-streaming callers get
everything in one place.
"""

from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Part, TaskState, TextPart
from a2a.utils import get_data_parts, new_agent_text_message, new_task

from agents.common.a2a_server import traced_agent

from .agent import stream_answer

ANSWER_ARTIFACT_ID = "rag-answer"


class RagAgentExecutor(AgentExecutor):
    """AgentExecutor implementation for the Clinical RAG Q&A Agent."""

    @traced_agent("rag_qa")
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        data_parts = get_data_parts(context.message.parts)
        if not data_parts or not data_parts[0].get("question"):
            await updater.failed(
                message=new_agent_text_message("Expected a DataPart with 'question'.")
            )
            return

        payload = data_parts[0]
        question = payload["question"]
        # patient_filter is the pre-per-patient-index name, kept working
        # so existing callers don't break silently.
        patient_id = payload.get("patient_id") or payload.get("patient_filter")
        top_k = int(payload.get("top_k") or 5)
        trace_id = payload.get("trace_id") or str(uuid4())

        if not patient_id:
            await updater.failed(
                message=new_agent_text_message(
                    "'patient_id' is required — this agent answers from one "
                    "patient's index at a time."
                )
            )
            return

        first_chunk = True
        final_payload = None

        try:
            async for event in stream_answer(
                question, patient_id, top_k=top_k, trace_id=trace_id
            ):
                kind = event["type"]

                if kind == "status":
                    await updater.update_status(
                        TaskState.working,
                        message=new_agent_text_message(event["payload"]),
                    )

                elif kind == "token":
                    await updater.add_artifact(
                        [Part(root=TextPart(text=event["payload"]))],
                        artifact_id=ANSWER_ARTIFACT_ID,
                        name="answer",
                        # append=False on the first chunk starts the
                        # artifact; True afterwards extends it, which is
                        # what makes this a token stream rather than a
                        # sequence of separate artifacts.
                        append=not first_chunk,
                        last_chunk=False,
                    )
                    first_chunk = False

                elif kind == "final":
                    final_payload = event["payload"]

        except Exception as exc:  # noqa: BLE001
            await updater.failed(
                message=new_agent_text_message(f"RAG pipeline failed: {exc}")
            )
            return

        if not first_chunk:
            # Close the streamed text artifact.
            await updater.add_artifact(
                [Part(root=TextPart(text=""))],
                artifact_id=ANSWER_ARTIFACT_ID,
                name="answer",
                append=True,
                last_chunk=True,
            )

        await updater.add_artifact(
            [Part(root=DataPart(data={**(final_payload or {}), "trace_id": trace_id}))],
            name="rag_result",
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported for the Clinical RAG Q&A Agent")
