"""
TEMPORARY test harness for step 3.4.

Wires clinical_data_harvester_tool (3.4), plus prompts (3.3),
resources (3.2), and the Watcher tool (3.1). Delete this file once
server.py (step 3.9) is built.

Run from the PROJECT ROOT:
    python mcp_primary/_test_step_3_4.py

Starts a streamable-http server at:
    http://127.0.0.1:8200/clinicaltools

In MCP Inspector:
    1. Transport Type = "Streamable HTTP", URL = http://127.0.0.1:8200/clinicaltools, Connect
    2. Under "Roots", add file:///absolute/path/to/Data/incoming
    3. Go to "Tools" -> clinical_data_harvester_tool
       Try: patient_id="P1019", doc_type="bill"        -> reads bills/P1019_bill.json
            patient_id="P1019", doc_type="doctor_report" -> reads doctor_reports/P1019_JohnDoe.txt
            patient_id="P1019", doc_type="lab_report"    -> reads lab_reports/P1019_labs.json
    4. Also re-check resource://discharge-report/P1019 and
       resource://lab-report/P1019 (from step 3.2) -- they now go
       through the real harvester extractor instead of the old stub.
"""

from mcp.server.fastmcp import FastMCP

from prompts import register_prompts
from resources import register_resources
from tools_harvester import clinical_data_harvester_tool
from tools_watcher import clinical_watcher_tool

mcp = FastMCP("primary-clinical-tools-TEST", port=8200, streamable_http_path="/clinicaltools")

register_resources(mcp)
register_prompts(mcp)
mcp.tool()(clinical_watcher_tool)
mcp.tool()(clinical_data_harvester_tool)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")