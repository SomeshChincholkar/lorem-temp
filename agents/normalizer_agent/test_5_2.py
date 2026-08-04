"""
agents/normalizer_agent/test_5_2.py

Smoke-tests the running Normalizer A2A server (port 8102) end-to-end:
  1. GET  /.well-known/agent.json        -> discovery works
  2. POST /  (no/bad token)               -> auth middleware rejects (401)
  3. POST /  (message/send, valid token)  -> full graph runs, artifact returned

Unlike the offline suite in tests/, this one exercises the REAL Bedrock
sampling path, so it needs live AWS credentials.

Prerequisites (all must already be running):
  - Primary MCP Clinical Tools Server :8200
  - This agent's server: python -m agents.normalizer_agent.server   (:8102)
  - Working AWS creds in .env (the sampling callback runs the model)

Usage:
    python -m agents.normalizer_agent.test_5_2                    # P1015, Hindi
    python -m agents.normalizer_agent.test_5_2 P1016              # German
    python -m agents.normalizer_agent.test_5_2 P1014 doctor_reports
"""

import json
import os
import sys
import uuid

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("NORMALIZER_AGENT_URL", "http://localhost:8102")
AGENT_SHARED_SECRET = os.getenv("AGENT_AUTH_TOKEN", "dev-secret-change-me")


def jsonrpc_send_message(payload_data: dict, token: str | None = None):
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "data", "data": payload_data}],
                "messageId": str(uuid.uuid4()),
            }
        },
    }
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["X-Agent-Auth-Token"] = token

    # Generous timeout: this round trip includes a real LLM call.
    return httpx.post(BASE_URL + "/", json=payload, headers=headers, timeout=180.0)


def main():
    patient_id = sys.argv[1] if len(sys.argv) > 1 else "P1015"
    doc_type = sys.argv[2] if len(sys.argv) > 2 else "doctor_reports"

    print(f"Target: {BASE_URL}\n")

    # 1. Discovery -- no auth required
    print("---- 1. GET /.well-known/agent.json (no auth) ----")
    r = httpx.get(BASE_URL + "/.well-known/agent.json", timeout=10.0)
    print(f"status: {r.status_code}")
    if r.status_code == 200:
        card = r.json()
        print(f"name: {card.get('name')}  url: {card.get('url')}")
        print(f"streaming: {card.get('capabilities', {}).get('streaming')}  (expected False)")
    else:
        print(r.text)
        print("\nDiscovery failed -- is the server actually running on this port?")
        return

    # 2. Auth rejection
    print("\n---- 2. POST / with NO token (expect 401) ----")
    r = jsonrpc_send_message({"patient_id": patient_id, "doc_type": doc_type}, token=None)
    print(f"status: {r.status_code}  (expected 401)")

    print("\n---- 2b. POST / with WRONG token (expect 401) ----")
    r = jsonrpc_send_message(
        {"patient_id": patient_id, "doc_type": doc_type}, token="not-the-real-secret"
    )
    print(f"status: {r.status_code}  (expected 401)")

    # 3. Bad payload -- neither doc_type nor raw_text
    print("\n---- 3. POST / with neither doc_type nor raw_text (expect task 'failed') ----")
    r = jsonrpc_send_message({"patient_id": patient_id}, token=AGENT_SHARED_SECRET)
    state = r.json().get("result", {}).get("status", {}).get("state")
    print(f"status: {r.status_code}  task state: {state}  (expected 'failed')")

    # 4. Real invocation -- the full Sampling round trip
    print(f"\n---- 4. POST / valid, patient_id={patient_id}, doc_type={doc_type} ----")
    r = jsonrpc_send_message(
        {"patient_id": patient_id, "doc_type": doc_type}, token=AGENT_SHARED_SECRET
    )
    print(f"status: {r.status_code}  (expected 200)")
    body = r.json()

    if "error" in body:
        print(json.dumps(body, indent=2)[:2000])
        print("\nJSON-RPC returned an error object -- see 'error' above.")
        return

    result = body.get("result", {})
    print(f"task state: {result.get('status', {}).get('state')}  (expected 'completed')")

    artifacts = result.get("artifacts", [])
    if not artifacts:
        print("\nNo artifacts returned -- check the server console for a traceback.")
        return

    data_parts = [
        p["data"] for a in artifacts for p in a.get("parts", []) if p.get("kind") == "data"
    ]
    for data in data_parts:
        print("\n--- normalized artifact ---")
        print(f"source_language : {data.get('source_language')}")
        print(f"model_used      : {data.get('model_used')}")
        print(f"confidence      : {data.get('confidence')}")
        print(f"low_confidence  : {data.get('low_confidence')}")
        print(f"abbreviations   : {data.get('expanded_abbreviations')}")
        print("\ntranslated_text (first 600 chars):")
        print((data.get("translated_text") or "")[:600])
        print("\nnormalized_text (first 600 chars):")
        print((data.get("normalized_text") or "")[:600])

    print("\nChecks to eyeball:")
    print("  - source_language matches the document (P1015=hi, P1016=de, P1014=es)")
    print("  - model_used is the nova-lite id for non-English source")
    print("  - normalized_text is English with abbreviations expanded")


if __name__ == "__main__":
    main()
