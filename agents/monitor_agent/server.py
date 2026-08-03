"""
agents/monitor_agent/server.py

A2A server for the Discharge Monitor Agent (port 8103, non-streaming).

Run:  python -m agents.monitor_agent.server

Prerequisites:
  - Primary MCP Clinical Tools Server :8200 (Watcher tool + Roots)
"""

import uvicorn

from agents.common.a2a_server import build_a2a_app

from .agent_card import AGENT_CARD
from .agent_executor import MonitorAgentExecutor

app = build_a2a_app(AGENT_CARD, MonitorAgentExecutor())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8103)
