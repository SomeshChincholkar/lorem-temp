"""
agents/common/a2a_client.py

A2A client used by the Host Orchestrator to drive every other agent.

Deliberately a thin JSON-RPC client over httpx rather than a wrapper
around a2a-sdk's client classes: the payloads are simple DataParts, and
this keeps the shared-secret header, the trace_id threading, and the
streaming SSE parsing all visible in one readable place.

Two call modes, matching spec Table 10:
    send_message()            -> non-streaming, await one final artifact
    send_message_streaming()  -> SSE, yield events as they arrive
"""

import json
import os
import uuid
from typing import Any, AsyncGenerator, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

AGENT_AUTH_TOKEN = os.getenv("AGENT_AUTH_TOKEN", "dev-secret-change-me")

AGENT_URLS = {
    "extractor": os.getenv("EXTRACTOR_AGENT_URL", "http://localhost:8100"),
    "validator": os.getenv("VALIDATOR_AGENT_URL", "http://localhost:8101"),
    "normalizer": os.getenv("NORMALIZER_AGENT_URL", "http://localhost:8102"),
    "monitor": os.getenv("MONITOR_AGENT_URL", "http://localhost:8103"),
    "summary": os.getenv("SUMMARY_AGENT_URL", "http://localhost:8104"),
    "rag": os.getenv("RAG_AGENT_URL", "http://localhost:8105"),
}

DEFAULT_TIMEOUT = 300.0


def _headers() -> dict:
    return {
        "Content-Type": "application/json",
        "X-Agent-Auth-Token": AGENT_AUTH_TOKEN,
    }


def _envelope(data: dict, method: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": method,
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "data", "data": data}],
                "messageId": str(uuid.uuid4()),
            }
        },
    }


def resolve_url(agent: str) -> str:
    """Accept either a registered agent name or a raw base URL."""
    if agent in AGENT_URLS:
        return AGENT_URLS[agent]
    return agent


async def fetch_agent_card(agent: str, timeout: float = 10.0) -> Optional[dict]:
    """
    Read an agent's AgentCard. Used at orchestrator startup to confirm
    who is reachable and which agents advertise streaming, rather than
    assuming the port map is live.
    """
    url = resolve_url(agent).rstrip("/") + "/.well-known/agent.json"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            if response.status_code == 200:
                return response.json()
    except Exception:
        return None
    return None


def extract_data_artifacts(result: dict) -> list[dict]:
    """Pull every DataPart payload out of a completed task."""
    artifacts = result.get("artifacts") or []
    return [
        part["data"]
        for artifact in artifacts
        for part in artifact.get("parts", [])
        if part.get("kind") == "data"
    ]


async def send_message(
    agent: str, data: dict, timeout: float = DEFAULT_TIMEOUT
) -> dict:
    """
    Non-streaming A2A call. Returns:
        {"ok": bool, "state": str, "artifacts": [...], "error": str|None}

    Never raises on a remote failure -- the orchestrator needs to report
    which step failed and carry on, not die halfway through a pipeline.
    """
    url = resolve_url(agent).rstrip("/") + "/"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url, json=_envelope(data, "message/send"), headers=_headers()
            )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "state": "unreachable", "artifacts": [], "error": str(exc)}

    if response.status_code != 200:
        return {
            "ok": False,
            "state": f"http_{response.status_code}",
            "artifacts": [],
            "error": response.text[:500],
        }

    body = response.json()
    if "error" in body:
        return {
            "ok": False,
            "state": "jsonrpc_error",
            "artifacts": [],
            "error": json.dumps(body["error"])[:500],
        }

    result = body.get("result", {})
    state = (result.get("status") or {}).get("state")
    return {
        "ok": state == "completed",
        "state": state,
        "artifacts": extract_data_artifacts(result),
        "error": None if state == "completed" else f"task state: {state}",
    }


async def send_message_streaming(
    agent: str, data: dict, timeout: float = DEFAULT_TIMEOUT
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Streaming A2A call (message/stream, SSE). Yields each decoded event
    payload as it arrives, so the caller can paint progressive output.

    Used for the two streaming agents: Summary Generator (:8104) and
    RAG Q&A (:8105).
    """
    url = resolve_url(agent).rstrip("/") + "/"
    payload = _envelope(data, "message/stream")

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", url, json=payload, headers=_headers()
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                yield {
                    "type": "error",
                    "error": f"HTTP {response.status_code}: {body[:300]!r}",
                }
                return

            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                raw = line[len("data:") :].strip()
                if not raw:
                    continue
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError:
                    continue


def stream_text_from_event(event: dict) -> str:
    """
    Pull human-readable text out of one streamed A2A event, whether it
    arrived as an artifact update or a status message.
    """
    result = event.get("result", event)

    artifact = result.get("artifact")
    if artifact:
        return "".join(
            part.get("text", "")
            for part in artifact.get("parts", [])
            if part.get("kind") == "text"
        )

    status = result.get("status") or {}
    message = status.get("message") or {}
    return "".join(
        part.get("text", "")
        for part in message.get("parts", [])
        if part.get("kind") == "text"
    )
