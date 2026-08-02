"""
TEMPORARY test harness for step 3.1 only.

Wires just clinical_watcher_tool onto a FastMCP app so you can validate
it in isolation with the MCP Inspector before the rest of the server
(resources, prompts, other tools) exists.

Delete this file once server.py (step 3.9) is built -- it's a
throwaway harness, not part of the final architecture.

Run with:
    python mcp_primary/_test_step_3_1.py

This starts a streamable-http server at:
    http://127.0.0.1:8200/clinicaltools

Then, in the MCP Inspector (run separately with `mcp dev` or
`npx @modelcontextprotocol/inspector`):
    1. Set Transport Type = "Streamable HTTP"
    2. Set URL = http://127.0.0.1:8200/clinicaltools
    3. Connect
    4. Under "Roots", add a root pointing at your Data/incoming folder,
       e.g. file:///absolute/path/to/Data/incoming
    5. Call the "clinical_watcher_tool" tool (subfolder param optional)
"""

from mcp.server.fastmcp import FastMCP
from tools_watcher import clinical_watcher_tool

mcp = FastMCP("primary-clinical-tools-TEST", port=8200, streamable_http_path="/clinicaltools")
mcp.tool()(clinical_watcher_tool)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")