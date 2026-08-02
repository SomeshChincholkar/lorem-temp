"""
TEMPORARY test harness for step 3.6.

Wires clinical_rules_engine_tool (3.6), plus lang bridge (3.5),
harvester (3.4), prompts (3.3), resources (3.2), and watcher (3.1).
Delete this file once server.py (step 3.9) is built.

Run from the PROJECT ROOT:
    python mcp_primary/_test_step_3_6.py

Starts a streamable-http server at:
    http://127.0.0.1:8200/clinicaltools

TESTING IN MCP INSPECTOR (3 scenarios to try):

Scenario A -- COMPLETE (nothing missing):
    doc_type = "bill"
    extracted_fields = {
        "patient_id": "P1019", "hospital_name": "City Hospital",
        "billing_date": "2026-07-20", "line_items": [{"desc": "Room", "amt": 4500}],
        "total_amount": 4500, "payment_status": "PAID"
    }
    Expect: {"status": "complete", "fields": {...}}

Scenario B -- BLOCKED (a blocking field missing, e.g. total_amount):
    doc_type = "bill"
    extracted_fields = {
        "patient_id": "P1019", "hospital_name": "City Hospital",
        "billing_date": "2026-07-20", "line_items": [{"desc": "Room", "amt": 4500}],
        "payment_status": "PAID"
    }
    Expect: {"status": "blocked", "unresolved_fields": ["total_amount"]}
    (No elicitation prompt appears -- blocking fields skip straight to blocked.)

Scenario C -- ELICITATION (only a non-blocking field missing, e.g. hospital_name):
    doc_type = "bill"
    extracted_fields = {
        "patient_id": "P1019",
        "billing_date": "2026-07-20", "line_items": [{"desc": "Room", "amt": 4500}],
        "total_amount": 4500, "payment_status": "PAID"
    }
    Click "Run Tool" -- it will hang.
    Switch to the "Elicitation" tab in Inspector. You should see a
    pending request asking for "hospital_name". Fill in a value (e.g.
    "City Hospital") and Accept.
    Expect: {"status": "resolved", "fields": {..., "hospital_name": "City Hospital"}}

    Try again but click Decline instead -> expect status "unresolved".
    Try again but click Cancel instead -> expect status "blocked".
"""

from mcp.server.fastmcp import FastMCP

from prompts import register_prompts
from resources import register_resources
from tools_harvester import clinical_data_harvester_tool
from tools_lang_bridge import medical_lang_bridge_tool
from tools_rules_engine import clinical_rules_engine_tool
from tools_watcher import clinical_watcher_tool

mcp = FastMCP("primary-clinical-tools-TEST", port=8200, streamable_http_path="/clinicaltools")

register_resources(mcp)
register_prompts(mcp)
mcp.tool()(clinical_watcher_tool)
mcp.tool()(clinical_data_harvester_tool)
mcp.tool()(medical_lang_bridge_tool)
mcp.tool()(clinical_rules_engine_tool)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")