"""
TEMPORARY test harness for step 3.5.

Wires medical_lang_bridge_tool (3.5), plus the harvester (3.4),
prompts (3.3), resources (3.2), and watcher (3.1). Delete this file
once server.py (step 3.9) is built.

Run from the PROJECT ROOT:
    python mcp_primary/_test_step_3_5.py

Starts a streamable-http server at:
    http://127.0.0.1:8200/clinicaltools

TESTING SAMPLING IN MCP INSPECTOR:
This tool doesn't call an LLM itself -- it asks the CONNECTED CLIENT
to run a completion via ctx.session.create_message(). The Inspector
acts as that client and has a "Sampling" tab specifically for this:

    1. Transport Type = "Streamable HTTP", URL = http://127.0.0.1:8200/clinicaltools, Connect
    2. Go to "Tools" -> medical_lang_bridge_tool
       Enter: text="Patient has HTN and DM, on BID dosing", source_language="en"
       Click "Run Tool" -- it will appear to hang.
    3. Switch to the "Sampling" tab. You should see a PENDING sampling
       request (the message this tool constructed, asking for JSON
       translation/normalization output).
    4. Manually type a response that mimics what an LLM would return, e.g.:
           {"translated_text": "Patient has Hypertension and Diabetes Mellitus, on twice a day dosing", "confidence": 0.95}
       and approve/send it back.
    5. Go back to "Tools" -- the pending call should now resolve with:
           {"translated_text": "...", "confidence": 0.95, "model_used": null}

This confirms the request/response plumbing works end-to-end. The
REAL model completion (via LiteLLM) only happens once the actual
LangGraph Normalizer Agent -- with its own sampling_callback -- is the
one connecting as client, which is a later part of the overall system
(outside this MCP server).
"""

from mcp.server.fastmcp import FastMCP

from prompts import register_prompts
from resources import register_resources
from tools_harvester import clinical_data_harvester_tool
from tools_lang_bridge import medical_lang_bridge_tool
from tools_watcher import clinical_watcher_tool

mcp = FastMCP("primary-clinical-tools-TEST", port=8200, streamable_http_path="/clinicaltools")

register_resources(mcp)
register_prompts(mcp)
mcp.tool()(clinical_watcher_tool)
mcp.tool()(clinical_data_harvester_tool)
mcp.tool()(medical_lang_bridge_tool)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")