"""
agents/summary_agent/server.py

A2A server for the Discharge Summary Generator (port 8104, STREAMING).

Run:  python -m agents.summary_agent.server

Prerequisites:
  - Primary MCP Clinical Tools Server :8200 (summary-generation-prompt)
  - A validation report at Data/reports/{patient_id}_report.json,
    written by the Validation Agent
  - Live Bedrock credentials in .env
"""

import uvicorn

from agents.common.a2a_server import build_a2a_app

from .agent_card import AGENT_CARD
from .agent_executor import SummaryAgentExecutor

app = build_a2a_app(AGENT_CARD, SummaryAgentExecutor())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8104)
