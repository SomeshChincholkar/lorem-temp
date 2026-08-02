"""
server.py

The Primary MCP Clinical Tools Server. Single entry point wiring
together everything built across steps 3.1-3.8:

  Resources : rules, abbreviations, HTML template, per-patient docs
  Prompts   : discharge-extraction, ehr-cross-validation,
              abbreviation-normalization, summary-generation, rag-answer
  Tools     : Watcher, Harvester, Lang Bridge (sampling), Rules Engine
              (elicitation), EHR Validator, Insight Reporter

Run from the PROJECT ROOT:
    python mcp_primary/server.py

Serves streamable-http at:
    http://127.0.0.1:8200/clinicaltools

The Discharge Monitor Agent (ADK) and other agents connect to this
URL, register a root pointing at Data/incoming/, and call these tools.

NOTE: the Mock EHR must be running separately for ehr_validation_tool
to work:
    uvicorn mock_ehr.main:app --port 8050
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

mcp = FastMCP("primary-clinical-tools", port=8200, streamable_http_path="/clinicaltools")

# Resources + prompts (declarative fetches, not tool calls)
register_resources(mcp)
register_prompts(mcp)

# Tools (the 6 primitives)
mcp.tool()(clinical_watcher_tool)
mcp.tool()(clinical_data_harvester_tool)
mcp.tool()(medical_lang_bridge_tool)
mcp.tool()(clinical_rules_engine_tool)
mcp.tool()(ehr_validation_tool)
mcp.tool()(clinical_insight_reporter_tool)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")