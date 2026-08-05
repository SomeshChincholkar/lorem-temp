"""
agents/common/push_notifications.py

A2A push notifications (spec section 9's technology stack line:
"a2a-sdk - Non-streaming + Streaming • Push Notifications").

Why they exist here: the Validation Agent can block for minutes waiting
on a human to answer an elicitation request, and a discharge case can be
retired long after the caller's HTTP request returned. Polling for that
is wasteful and racy. Push notifications let an agent call a webhook the
client registered, so the dashboard or an external system learns about a
completed or blocked case without holding a connection open.

Transport is a plain signed POST rather than anything a2a-sdk-specific,
so any HTTP endpoint can receive it. The shared secret is the same
X-Agent-Auth-Token the A2A servers require, plus an HMAC over the body so
a receiver can verify the payload was not tampered with in transit.
"""

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

AGENT_AUTH_TOKEN = os.getenv("AGENT_AUTH_TOKEN", "dev-secret-change-me")

# Where completed-case notifications go. Unset means the feature is off,
# which is the default -- an unconfigured webhook must not produce
# connection errors on every run.
PUSH_NOTIFICATION_URL = os.getenv("PUSH_NOTIFICATION_URL", "")

TIMEOUT_SECONDS = 10.0


def is_enabled() -> bool:
    return bool(PUSH_NOTIFICATION_URL)


def sign_payload(body: bytes, secret: str = AGENT_AUTH_TOKEN) -> str:
    """HMAC-SHA256 of the raw body, hex encoded."""
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def verify_signature(body: bytes, signature: str, secret: str = AGENT_AUTH_TOKEN) -> bool:
    """
    Constant-time signature check for receivers.

    compare_digest rather than == so a receiver cannot be probed for the
    secret one byte at a time via response timing.
    """
    return hmac.compare_digest(sign_payload(body, secret), signature or "")


def build_notification(
    event: str,
    patient_id: str,
    trace_id: Optional[str] = None,
    **fields,
) -> dict:
    return {
        "event": event,
        "patient_id": patient_id,
        "trace_id": trace_id,
        "sent_at": datetime.now(timezone.utc).isoformat(),
        **fields,
    }


async def send_push_notification(
    event: str,
    patient_id: str,
    trace_id: Optional[str] = None,
    url: Optional[str] = None,
    **fields,
) -> dict:
    """
    POST a notification to the configured webhook.

    Returns {"sent": bool, "status": int|None, "error": str|None}. Never
    raises: a notification is a courtesy to an external listener, and a
    dead webhook must not fail a discharge that already completed.
    """
    target = url or PUSH_NOTIFICATION_URL
    if not target:
        return {"sent": False, "status": None, "error": "no PUSH_NOTIFICATION_URL configured"}

    payload = build_notification(event, patient_id, trace_id, **fields)
    body = json.dumps(payload).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "X-Agent-Auth-Token": AGENT_AUTH_TOKEN,
        "X-Agent-Signature": sign_payload(body),
    }

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            response = await client.post(target, content=body, headers=headers)
        return {
            "sent": 200 <= response.status_code < 300,
            "status": response.status_code,
            "error": None if response.is_success else response.text[:300],
        }
    except Exception as exc:  # noqa: BLE001
        return {"sent": False, "status": None, "error": str(exc)}
