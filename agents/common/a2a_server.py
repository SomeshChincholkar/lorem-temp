"""
agents/common/a2a_server.py

Shared A2A server scaffolding, so all six agents expose an identical
surface instead of six copies of the same middleware drifting apart.

What a2a-sdk gives us: JSON-RPC endpoints, task lifecycle, and the
AgentCard at /.well-known/agent.json.

What it doesn't: the spec's shared-secret requirement ("Authentication
via shared secret header X-Agent-Auth-Token is required on all A2A
servers"). That's added here as Starlette middleware wrapped around the
app a2a-sdk builds.
"""

import os

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

load_dotenv()

AGENT_SHARED_SECRET = os.getenv("AGENT_AUTH_TOKEN", "dev-secret-change-me")

# Discovery must stay reachable without the shared secret, so agents can
# read each other's cards before authenticating. a2a-sdk serves the card
# at both of these paths depending on version.
PUBLIC_PATHS = {"/.well-known/agent.json", "/.well-known/agent-card.json"}


class AuthTokenMiddleware(BaseHTTPMiddleware):
    """Rejects any non-discovery request without the shared secret."""

    async def dispatch(self, request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        token = request.headers.get("X-Agent-Auth-Token")
        if token != AGENT_SHARED_SECRET:
            return JSONResponse(
                {"error": "invalid or missing X-Agent-Auth-Token"}, status_code=401
            )
        return await call_next(request)


def build_a2a_app(agent_card, agent_executor):
    """
    Wire an AgentCard + AgentExecutor into an authenticated Starlette app.

    Works for both streaming and non-streaming agents -- streaming is a
    property of the AgentCard's capabilities and of what the executor
    enqueues, not of this scaffolding.
    """
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore(),
    )
    a2a_app = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
    starlette_app = a2a_app.build()
    starlette_app.add_middleware(AuthTokenMiddleware)
    return starlette_app
