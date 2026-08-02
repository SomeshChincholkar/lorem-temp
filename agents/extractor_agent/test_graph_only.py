"""
agents/extractor_agent/test_graph_only.py

Smoke-test the LangGraph graph directly, bypassing the A2A/FastAPI
layer. Run this FIRST when debugging -- it isolates MCP + LLM issues
from HTTP/auth issues.

Usage (from FA5_Project/ root):
    python -m agents.extractor_agent.test_graph_only P1001 doctor_reports
"""

import asyncio
import json
import sys

from .graph import extractor_app


async def main():
    patient_id = sys.argv[1] if len(sys.argv) > 1 else "P1001"
    doc_type = sys.argv[2] if len(sys.argv) > 2 else "doctor_reports"

    initial_state = {"patient_id": patient_id, "doc_type": doc_type}
    final_state = await extractor_app.ainvoke(
        initial_state, config={"configurable": {"thread_id": "debug-run"}}
    )

    print("---- RAW TEXT (first 300 chars) ----")
    print(final_state.get("raw_text", "")[:300])

    print(f"\n---- DETECTED LANGUAGE: {final_state.get('language')} ----")

    print("\n---- EXTRACTED FIELDS ----")
    print(json.dumps(final_state.get("extracted_fields", {}), indent=2))

    if final_state.get("error"):
        print("\n---- ERROR ----")
        print(final_state["error"])


if __name__ == "__main__":
    asyncio.run(main())