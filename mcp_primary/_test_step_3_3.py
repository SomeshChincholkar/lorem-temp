"""
TEMPORARY test harness for step 3.3.

Wires all 5 prompts from prompts.py, plus resources (3.2) and the
Watcher tool (3.1), so you can sanity-check the whole server so far.
Delete this file once server.py (step 3.9) is built.

IMPORTANT: run this from the PROJECT ROOT (not from inside
mcp_primary/), since common/rules_loader.py resolves configs/rules.yaml
via a relative path against the current working directory:

    cd "C:\\Users\\somes\\Desktop\\FA5_Project"
    python mcp_primary/_test_step_3_3.py

This starts a streamable-http server at:
    http://127.0.0.1:8200/clinicaltools

Then, in the MCP Inspector:
    1. Transport Type = "Streamable HTTP"
    2. URL = http://127.0.0.1:8200/clinicaltools
    3. Connect
    4. Go to the "Prompts" tab and try each of:
       - discharge-extraction-prompt   (params: language="en", doc_types=["discharge_report"])
       - ehr-cross-validation-prompt   (params: patient_id="P1019")
       - abbreviation-normalization-prompt (params: source_language="en")
       - summary-generation-prompt     (params: risk_level="medium", audience="patient")
       - rag-answer-prompt             (params: context_length=2000)
    5. (Still works) Roots + Resources + clinical_watcher_tool from
       steps 3.1/3.2, if you want to re-verify those too.
"""

from mcp.server.fastmcp import FastMCP

from prompts import register_prompts
from resources import register_resources
from tools_watcher import clinical_watcher_tool

mcp = FastMCP("primary-clinical-tools-TEST", port=8200, streamable_http_path="/clinicaltools")

register_resources(mcp)
register_prompts(mcp)
mcp.tool()(clinical_watcher_tool)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")