"""
mcp_secondary/server.py

The Secondary MCP Analytics Server. Lightweight, tools-only -- no
resources, prompts, or sampling. Just 3 standalone tools that agents
can use even without the Primary server mounted (calculate_risk_score
only needs common/rules_loader; benchmarks is a pure static lookup;
the heatmap tool reads Primary-generated reports off shared disk, but
doesn't call the Primary server itself).

Run from the PROJECT ROOT:
    python mcp_secondary/server.py

Serves streamable-http at:
    http://127.0.0.1:8201/analyticstools
"""

from mcp.server.fastmcp import FastMCP

from tools_benchmarks import get_population_benchmarks_tool
from tools_heatmap import generate_risk_heatmap_tool
from tools_risk_score import calculate_risk_score_tool

mcp = FastMCP("secondary-analytics-tools", port=8201, streamable_http_path="/analyticstools")

mcp.tool()(calculate_risk_score_tool)
mcp.tool()(get_population_benchmarks_tool)
mcp.tool()(generate_risk_heatmap_tool)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")