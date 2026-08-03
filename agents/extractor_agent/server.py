"""
agents/extractor_agent/server.py

A2A server for the Clinical Extractor Agent (port 8100, non-streaming),
built on a2a-sdk:

  AgentCard + ExtractorAgentExecutor
    -> DefaultRequestHandler (a2a-sdk: task lifecycle, event routing)
    -> A2AStarletteApplication (a2a-sdk: JSON-RPC endpoints +
       GET /.well-known/agent.json)
    -> Starlette app, run with uvicorn

The shared-secret middleware lives in agents/common/a2a_server.py, since
every agent needs the identical check.

Run:  python -m agents.extractor_agent.server
"""

import uvicorn

from agents.common.a2a_server import build_a2a_app

from .agent_card import AGENT_CARD
from .agent_executor import ExtractorAgentExecutor

app = build_a2a_app(AGENT_CARD, ExtractorAgentExecutor())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)
