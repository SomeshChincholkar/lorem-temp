"""
agents/orchestrator/server.py

Host Orchestrator (spec section 8) -- Google ADK agent + Gradio UI on
:8083, acting as the A2A client to every other agent.

Four tabs:
  1. Run Pipeline      -- non-streaming A2A across Monitor/Extractor/
                          Normalizer/Validator
  2. Discharge Summary -- STREAMING A2A to :8104, section by section
  3. Clinical Q&A      -- STREAMING A2A to :8105, token by token
  4. System Health     -- AgentCard discovery across every agent

Tabs 2 and 3 are what demonstrate the streaming half of Table 10; tab 1
is the non-streaming half.

Run:  python -m agents.orchestrator.server

Prerequisites: every other agent, plus both MCP servers and the Mock
EHR. Tab 4 tells you which of them are actually up.
"""

import os

import gradio as gr

from agents.common.a2a_client import (
    AGENT_URLS,
    fetch_agent_card,
    send_message_streaming,
    stream_text_from_event,
)

from .agent import format_json, format_pipeline_result
from .pipeline import run_discharge_pipeline

ORCHESTRATOR_PORT = int(os.getenv("ORCHESTRATOR_PORT", "8083"))

EXAMPLE_PATIENTS = ["P1019", "P1015", "P1016", "P1014", "P1020"]


async def ui_run_pipeline(patient_id: str):
    patient_id = (patient_id or "").strip()
    if not patient_id:
        return "Enter a patient ID first.", "{}"

    result = await run_discharge_pipeline(patient_id)
    return format_pipeline_result(result), format_json(result)


async def ui_stream_summary(patient_id: str):
    """Streams the discharge summary section by section from :8104."""
    patient_id = (patient_id or "").strip()
    if not patient_id:
        yield "Enter a patient ID first."
        return

    text = f"Requesting summary for {patient_id}...\n\n"
    yield text

    async for event in send_message_streaming("summary", {"patient_id": patient_id}):
        if event.get("type") == "error":
            yield text + f"\n\n**Streaming failed:** {event['error']}"
            return
        chunk = stream_text_from_event(event)
        if chunk:
            text += chunk
            yield text


async def ui_stream_qa(question: str, patient_id: str):
    """Streams a grounded answer token by token from :8105."""
    question = (question or "").strip()
    if not question:
        yield "Ask a question first."
        return

    # Records are indexed per patient, so this is required, not a filter.
    patient_id = (patient_id or "").strip()
    if not patient_id:
        yield "Select a patient first — answers come from one patient's index."
        return

    payload = {"question": question, "patient_id": patient_id}

    text = ""
    async for event in send_message_streaming("rag", payload):
        if event.get("type") == "error":
            yield text + f"\n\n**Streaming failed:** {event['error']}"
            return
        chunk = stream_text_from_event(event)
        if chunk:
            text += chunk
            yield text

    if not text:
        yield "No answer was returned — check that the RAG agent is running."


async def ui_health():
    rows = []
    for name, url in AGENT_URLS.items():
        card = await fetch_agent_card(name)
        if card:
            streaming = (card.get("capabilities") or {}).get("streaming")
            rows.append([name, url, "up", card.get("name"), str(bool(streaming))])
        else:
            rows.append([name, url, "DOWN", "-", "-"])
    return rows


def build_ui():
    with gr.Blocks(title="Discharge Orchestrator") as demo:
        gr.Markdown(
            "# Hospital Discharge Orchestrator\n"
            "Google ADK host agent coordinating the LangGraph, ADK and Agno "
            "agents over the A2A protocol."
        )

        with gr.Tab("Run Pipeline"):
            patient_input = gr.Textbox(
                label="Patient ID", value="P1019", placeholder="e.g. P1019"
            )
            gr.Examples(EXAMPLE_PATIENTS, inputs=patient_input)
            run_button = gr.Button("Run discharge pipeline", variant="primary")
            pipeline_summary = gr.Markdown()
            pipeline_json = gr.Code(label="Raw result", language="json")
            run_button.click(
                ui_run_pipeline,
                inputs=patient_input,
                outputs=[pipeline_summary, pipeline_json],
            )

        with gr.Tab("Discharge Summary (streaming)"):
            summary_patient = gr.Textbox(label="Patient ID", value="P1019")
            summary_button = gr.Button("Stream summary", variant="primary")
            summary_output = gr.Markdown()
            summary_button.click(
                ui_stream_summary, inputs=summary_patient, outputs=summary_output
            )

        with gr.Tab("Clinical Q&A (streaming)"):
            question_input = gr.Textbox(
                label="Question",
                value="What medications was this patient discharged on?",
                lines=2,
            )
            qa_patient = gr.Dropdown(
                EXAMPLE_PATIENTS,
                label="Patient (required)",
                value=EXAMPLE_PATIENTS[0],
                info="Each patient has their own index — answers never cross patients.",
            )
            qa_button = gr.Button("Ask", variant="primary")
            qa_output = gr.Markdown()
            qa_button.click(
                ui_stream_qa, inputs=[question_input, qa_patient], outputs=qa_output
            )

        with gr.Tab("System Health"):
            health_button = gr.Button("Check all agents")
            health_table = gr.Dataframe(
                headers=["agent", "url", "status", "card name", "streaming"],
                label="A2A agent discovery",
            )
            health_button.click(ui_health, outputs=health_table)

    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=ORCHESTRATOR_PORT)
