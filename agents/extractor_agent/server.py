"""
agents/extractor_agent/server.py

Real A2A server for the Clinical Extractor Agent (port 8100,
non-streaming), built on a2a-sdk:

  AgentCard + ExtractorAgentExecutor
    -> DefaultRequestHandler (a2a-sdk: task lifecycle, event routing)
    -> A2AStarletteApplication (a2a-sdk: JSON-RPC endpoints +
       GET /.well-known/agent.json)
    -> Starlette app, run with uvicorn

This is what "A2A server" means per the spec's Table 6/10 -- FastAPI
is NOT the A2A layer here, it's only used as the underlying ASGI
toolkit for the one thing a2a-sdk doesn't give us out of the box:
the X-Agent-Auth-Token shared-secret check (added as Starlette
middleware around the app a2a-sdk builds).

Run:  python -m agents.extractor_agent.server
"""

import os

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .agent_card import AGENT_CARD
from .agent_executor import ExtractorAgentExecutor

load_dotenv()

AGENT_SHARED_SECRET = os.getenv("AGENT_AUTH_TOKEN", "dev-secret-change-me")

# Paths a2a-sdk exposes that must stay reachable without the shared
# secret, so other agents can discover this one before authenticating.
PUBLIC_PATHS = {"/.well-known/agent.json", "/.well-known/agent-card.json"}


class AuthTokenMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        token = request.headers.get("X-Agent-Auth-Token")
        if token != AGENT_SHARED_SECRET:
            return JSONResponse(
                {"error": "invalid or missing X-Agent-Auth-Token"}, status_code=401
            )
        return await call_next(request)


def build_app():
    request_handler = DefaultRequestHandler(
        agent_executor=ExtractorAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    a2a_app = A2AStarletteApplication(agent_card=AGENT_CARD, http_handler=request_handler)
    starlette_app = a2a_app.build()
    starlette_app.add_middleware(AuthTokenMiddleware)
    return starlette_app


app = build_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)