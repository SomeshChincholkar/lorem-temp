"""
TEMPORARY test harness for step 3.8.

Wires clinical_insight_reporter_tool (3.8), plus everything from
3.1-3.7. Delete this file once server.py (step 3.9) is built.

Run from the PROJECT ROOT:
    python mcp_primary/_test_step_3_8.py

Starts a streamable-http server at:
    http://127.0.0.1:8200/clinicaltools

TESTING IN MCP INSPECTOR:
    Tools -> clinical_insight_reporter_tool

    Patient Id: P1001

    All Inputs (JSON) -- using the clean P1001 scenario from step 3.7:
    {
      "extracted_discharge": {
        "patient_id": "P1001",
        "patient_name": "John Smith",
        "discharge_diagnosis": "Acute Myocardial Infarction",
        "discharge_approved": true,
        "discharge_approved_by": "Dr. Rao",
        "follow_up_appointments": "Cardiology in 7 days",
        "discharge_instructions": "Continue Aspirin, Atorvastatin, Metoprolol. Troponin-I remains elevated, continue ACS management. LDL Cholesterol elevated, continue high-intensity statin therapy.",
        "medications": [
          {"medicine_name": "Aspirin", "strength": "81 mg", "frequency": "QD", "route": "oral"},
          {"medicine_name": "Atorvastatin", "strength": "40 mg", "frequency": "HS", "route": "oral"},
          {"medicine_name": "Metoprolol", "strength": "50 mg", "frequency": "BID", "route": "oral"}
        ]
      },
      "extracted_bill": {
        "patient_id": "P1001",
        "total_amount": 4500,
        "payment_status": "PAID"
      },
      "completeness_gaps": {"missing_blocking": [], "missing_nonblocking": []},
      "ehr_findings": [
        {"rule_id": "med_omission_check", "severity": "Warning", "triggered": false, "evidence": "OK", "action": "OK"},
        {"rule_id": "allergy_contradiction_check", "severity": "Critical", "triggered": false, "evidence": "OK", "action": "OK"},
        {"rule_id": "diagnosis_mismatch_check", "severity": "Warning", "triggered": false, "evidence": "OK", "action": "OK"},
        {"rule_id": "follow_up_missing_check", "severity": "Critical", "triggered": false, "evidence": "OK", "action": "OK"},
        {"rule_id": "lab_follow_up_mismatch_check", "severity": "Warning", "triggered": false, "evidence": "OK", "action": "OK"},
        {"rule_id": "discharge_approval_check", "severity": "Critical", "triggered": false, "evidence": "OK", "action": "OK"},
        {"rule_id": "bill_settlement_check", "severity": "Critical", "triggered": false, "evidence": "OK", "action": "OK"}
      ],
      "translation_confidence": 1.0
    }

    Expect: risk_score=0, risk_level="low", discharge_blocked=false,
    and json_path/html_path pointing at data/reports/P1001_report.*

    TIP: in a real pipeline, ehr_findings would be the exact array
    already returned by ehr_validation_tool in step 3.7 -- you can
    literally copy that tool's output here instead of retyping it.

    Then check your project folder at data/reports/ for the two
    generated files, and open the .html one in a browser to see the
    rendered template.

    To test a "blocked" report, flip discharge_approval_check's
    "triggered" to true and re-run.
"""

from mcp.server.fastmcp import FastMCP

from prompts import register_prompts
from resources import register_resources
from tools_ehr_validator import ehr_validation_tool
from tools_harvester import clinical_data_harvester_tool
from tools_lang_bridge import medical_lang_bridge_tool
from tools_reporter import clinical_insight_reporter_tool
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
mcp.tool()(clinical_insight_reporter_tool)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")