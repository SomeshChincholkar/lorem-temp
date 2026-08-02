"""
resources.py

MCP Resources for the Primary Clinical Tools Server. Serves rules,
templates, and raw per-patient document text declaratively, so agents
fetch them via read_resource(uri) instead of ad hoc tool calls.

Registration is centralized here via register_resources(mcp), called
from server.py (step 3.9), mirroring how tools are registered via
mcp.tool()(func).

NOTE on discharge/lab report extraction: multi-format extraction
(.txt/.json/.pdf/.png/.png.ocr.txt) is delegated to
tools_harvester.extract_text_any_format / find_file_for_patient
(step 3.4). This module used to have a temporary local stub handling
only .txt/.json before that was built -- now removed in favor of the
real one.
"""

import json
import sys
from pathlib import Path

# Make the project root importable regardless of how this script/module
# is invoked, so `from common.rules_loader import ...` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import Context, FastMCP  # noqa: E402

from common.rules_loader import load_rules  # noqa: E402
from roots import resolve_authorized_root, safe_join  # noqa: E402
from tools_harvester import extract_text_any_format, find_file_for_patient  # noqa: E402

TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "discharge_summary.html"


# ---------------------------------------------------------------------
# Resource functions
# ---------------------------------------------------------------------
def get_completeness_rules() -> str:
    """resource://clinical-rules/completeness"""
    rules = load_rules()
    return json.dumps({
        "mandatory_clinical_fields": rules.get("mandatory_clinical_fields", []),
        "mandatory_prescription_fields": rules.get("mandatory_prescription_fields", []),
    }, indent=2)


def get_cross_validation_rules() -> str:
    """resource://clinical-rules/cross-validation"""
    rules = load_rules()
    return json.dumps({
        "clinical_validation_policies": rules.get("clinical_validation_policies", []),
    }, indent=2)


async def get_discharge_report(patient_id: str, ctx: Context) -> str:
    """resource://discharge-report/{patient_id}"""
    root = await resolve_authorized_root(ctx)
    folder = safe_join(root, "doctor_reports")
    path = find_file_for_patient(folder, patient_id)
    return extract_text_any_format(path)


async def get_lab_report(patient_id: str, ctx: Context) -> str:
    """resource://lab-report/{patient_id}"""
    root = await resolve_authorized_root(ctx)
    folder = safe_join(root, "lab_reports")
    path = find_file_for_patient(folder, patient_id)
    return extract_text_any_format(path)


def get_html_template() -> str:
    """resource://report-template/html"""
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE_PATH}")
    return TEMPLATE_PATH.read_text(encoding="utf-8")


def get_abbreviation_dict() -> str:
    """resource://medical-abbreviations"""
    rules = load_rules()
    return json.dumps(rules.get("normalization_standards", {}).get("abbreviation_map", {}), indent=2)


# ---------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------
def register_resources(mcp: FastMCP) -> None:
    """Wire all resource functions onto the given FastMCP app."""
    mcp.resource("resource://clinical-rules/completeness")(get_completeness_rules)
    mcp.resource("resource://clinical-rules/cross-validation")(get_cross_validation_rules)
    mcp.resource("resource://discharge-report/{patient_id}")(get_discharge_report)
    mcp.resource("resource://lab-report/{patient_id}")(get_lab_report)
    mcp.resource("resource://report-template/html")(get_html_template)
    mcp.resource("resource://medical-abbreviations")(get_abbreviation_dict)