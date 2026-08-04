"""
agents/normalizer_agent/server.py

A2A server for the Clinical Normalizer Agent (port 8102, non-streaming).

Run:  python -m agents.normalizer_agent.server

Prerequisites:
  - Primary MCP Clinical Tools Server :8200 (Lang Bridge tool, prompts,
    medical-abbreviations resource)
  - Live Bedrock credentials in .env -- this agent's sampling callback
    is what actually runs the translation model.
"""

import uvicorn

from agents.common.a2a_server import build_a2a_app

from .agent_card import AGENT_CARD
from .agent_executor import NormalizerAgentExecutor

app = build_a2a_app(AGENT_CARD, NormalizerAgentExecutor())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8102)
