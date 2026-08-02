"""
agents/extractor_agent/test_5_1.py

Smoke-tests the running A2A server (port 8100) end-to-end:
  1. GET  /.well-known/agent.json        -> discovery works
  2. POST /  (no/bad token)               -> auth middleware rejects (401)
  3. POST /  (message/send, valid token)  -> full pipeline runs, artifact returned

Prerequisites (all must already be running):
  - Mock EHR                 :8050
  - Primary MCP Clinical Tools Server :8200
  - This agent's server: python -m agents.extractor_agent.server   (:8100)

Usage:
    python -m agents.extractor_agent.test_server
    python -m agents.extractor_agent.test_server P1019 bill
"""

import json
import os
import sys
import uuid

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("EXTRACTOR_AGENT_URL", "http://localhost:8100")
AGENT_SHARED_SECRET = os.getenv("AGENT_AUTH_TOKEN", "dev-secret-change-me")


def jsonrpc_send_message(patient_id: str, doc_type: str, language: str = "en", token: str | None = None):
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [
                    {
                        "kind": "data",
                        "data": {
                            "patient_id": patient_id,
                            "doc_type": doc_type,
                            "language": language,
                        },
                    }
                ],
                "messageId": str(uuid.uuid4()),
            }
        },
    }
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["X-Agent-Auth-Token"] = token

    return httpx.post(BASE_URL + "/", json=payload, headers=headers, timeout=60.0)


def main():
    patient_id = sys.argv[1] if len(sys.argv) > 1 else "P1001"
    doc_type = sys.argv[2] if len(sys.argv) > 2 else "doctor_reports"
    language = sys.argv[3] if len(sys.argv) > 3 else "en"

    print(f"Target: {BASE_URL}\n")

    # 1. Discovery -- should work with no auth at all
    print("---- 1. GET /.well-known/agent.json (no auth) ----")
    r = httpx.get(BASE_URL + "/.well-known/agent.json", timeout=10.0)
    print(f"status: {r.status_code}")
    if r.status_code == 200:
        card = r.json()
        print(f"name: {card.get('name')}  url: {card.get('url')}")
    else:
        print(r.text)
        print("\nDiscovery failed -- is the server actually running on this port?")
        return

    # 2. Auth rejection -- no token
    print("\n---- 2. POST / with NO token (expect 401) ----")
    r = jsonrpc_send_message(patient_id, doc_type, language, token=None)
    print(f"status: {r.status_code}  (expected 401)")

    # 2b. Auth rejection -- wrong token
    print("\n---- 2b. POST / with WRONG token (expect 401) ----")
    r = jsonrpc_send_message(patient_id, doc_type, language, token="not-the-real-secret")
    print(f"status: {r.status_code}  (expected 401)")

    # 3. Real invocation
    print(f"\n---- 3. POST / with valid token, patient_id={patient_id}, doc_type={doc_type}, language={language} ----")
    r = jsonrpc_send_message(patient_id, doc_type, language, token=AGENT_SHARED_SECRET)
    print(f"status: {r.status_code}  (expected 200)")
    body = r.json()
    print(json.dumps(body, indent=2)[:2000])

    if "error" in body:
        print("\nJSON-RPC returned an error object -- see 'error' above.")
        return

    result = body.get("result", {})
    state = result.get("status", {}).get("state")
    print(f"\ntask state: {state}  (expected 'completed')")

    artifacts = result.get("artifacts", [])
    if artifacts:
        data_parts = [
            p["data"] for a in artifacts for p in a.get("parts", []) if p.get("kind") == "data"
        ]
        print("\nextracted artifact data:")
        print(json.dumps(data_parts, indent=2))
    else:
        print("\nNo artifacts returned -- check the server's console log for the traceback.")


if __name__ == "__main__":
    main()