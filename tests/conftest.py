"""
tests/conftest.py

Path setup shared by the whole suite.

The project root is needed so `agents`, `guardrails`, `common` and
`observability` resolve. The two MCP server directories are needed *as
well* because their modules import their siblings flat (`from roots
import ...`, `from tools_benchmarks import ...`) rather than as packages.
That works in production because each server is launched as
`python mcp_primary/server.py`, which puts its own directory on sys.path
first -- but pytest runs from the project root, where it would not.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

for path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "mcp_primary",
    PROJECT_ROOT / "mcp_secondary",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
