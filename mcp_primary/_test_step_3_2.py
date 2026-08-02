"""
TEMPORARY test harness for step 3.2 only.

Wires all 6 resources from resources.py, plus the Watcher tool from
step 3.1 (kept so you can still sanity-check the server as a whole).
Delete this file once server.py (step 3.9) is built.

Run with:
    python mcp_primary/_test_step_3_2.py

This starts a streamable-http server at:
    http://127.0.0.1:8200/clinicaltools

Then, in the MCP Inspector:
    1. Transport Type = "Streamable HTTP"
    2. URL = http://127.0.0.1:8200/clinicaltools
    3. Connect
    4. Under "Roots", add file:///absolute/path/to/Data/incoming
       (only needed for the discharge-report / lab-report resources)
    5. Go to the "Resources" tab and read each of:
       - resource://clinical-rules/completeness
       - resource://clinical-rules/cross-validation
       - resource://medical-abbreviations
       - resource://report-template/html
       - resource://discharge-report/P1019
       - resource://lab-report/P1019
"""

from mcp.server.fastmcp import FastMCP

from resources import register_resources
from tools_watcher import clinical_watcher_tool

mcp = FastMCP("primary-clinical-tools-TEST", port=8200, streamable_http_path="/clinicaltools")

register_resources(mcp)
mcp.tool()(clinical_watcher_tool)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")