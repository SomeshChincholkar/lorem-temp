"""
agents/validator_agent/server.py

A2A server for the Clinical Validation Agent (port 8101, non-streaming).

Run:  python -m agents.validator_agent.server

Prerequisites:
  - Mock EHR                          :8050  (Table 4 rules query it)
  - Primary MCP Clinical Tools Server :8200
  - Streamlit HITL dashboard          :8501  (only needed when a case
    actually elicits; without it, elicitation requests park in
    Data/elicitations/ and time out into "declined" after
    ELICITATION_TIMEOUT_SECONDS)
"""

import uvicorn

from agents.common.a2a_server import build_a2a_app

from .agent_card import AGENT_CARD
from .agent_executor import ValidatorAgentExecutor

app = build_a2a_app(AGENT_CARD, ValidatorAgentExecutor())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8101)
