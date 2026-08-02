"""
TEMPORARY test harness for step 3.7.

Wires ehr_validation_tool (3.7), plus rules engine (3.6), lang bridge
(3.5), harvester (3.4), prompts (3.3), resources (3.2), and watcher
(3.1). Delete this file once server.py (step 3.9) is built.

THIS STEP NEEDS TWO SERVERS RUNNING AT ONCE:

Terminal 1 -- Mock EHR (yours, already built):
    uvicorn mock_ehr.main:app --port 8050

Terminal 2 -- this test harness, from the PROJECT ROOT:
    python mcp_primary/_test_step_3_7.py

Starts the clinical tools server at:
    http://127.0.0.1:8200/clinicaltools

TESTING IN MCP INSPECTOR:
    1. Connect as usual (Streamable HTTP, http://127.0.0.1:8200/clinicaltools)
    2. Tools -> ehr_validation_tool
    3. Pick a patient_id that actually exists in mock_ehr/data.py
       (check PATIENTS dict there for valid IDs -- P1019 used
       elsewhere in this project's test files is only a placeholder
       and may NOT exist in your mock EHR data).
    4. extracted_discharge (JSON) -- example shape:
       {
         "patient_id": "<some id from data.py>",
         "discharge_diagnosis": "Hypertension",
         "discharge_approved": true,
         "discharge_approved_by": "Dr. Rao",
         "follow_up_appointments": "Cardiology in 7 days",
         "discharge_instructions": "Continue medications, monitor BP",
         "medications": [
           {"medicine_name": "Amlodipine", "strength": "5mg", "frequency": "OD", "route": "oral"}
         ]
       }
    5. extracted_bill (JSON) -- example shape:
       {
         "patient_id": "<same id>",
         "total_amount": 4500,
         "payment_status": "PAID"
       }
    6. Run the tool. Expect a JSON array of 7 rule results
       ({rule_id, severity, triggered, evidence, action}).

To deliberately trigger specific rules for testing:
    - allergy_contradiction_check: set a medicine_name that matches
      one of that patient's allergies in mock_ehr/data.py
    - discharge_approval_check: set "discharge_approved": false
    - bill_settlement_check: set "payment_status": "PENDING"
    - follow_up_missing_check: omit "follow_up_appointments" for a
      patient whose care_plan.followup_required is true
"""

from mcp.server.fastmcp import FastMCP

from prompts import register_prompts
from resources import register_resources
from tools_ehr_validator import ehr_validation_tool
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
mcp.tool()(ehr_validation_tool)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")