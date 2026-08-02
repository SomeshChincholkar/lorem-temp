"""
tools_rules_engine.py

The Clinical Rules Engine primitive. Runs completeness checks from
Table 3 (completeness validation fields by document type):

  - If a BLOCKING field is missing -> skip elicitation entirely, go
    straight to "blocked" status for HITL escalation.
  - If only NON-blocking fields are missing -> pause execution and ask
    a human via ctx.elicit() to fill the gaps.

Reuses DOC_TYPE_FIELD_SCHEMAS from prompts.py as the single source of
truth for Table 3's required/blocking field lists per doc type, so
this file and the extraction prompt never drift out of sync.
"""

import sys
from pathlib import Path
from typing import Dict, List, Type

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.elicitation import (  # noqa: E402
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)
from mcp.server.fastmcp import Context  # noqa: E402
from pydantic import BaseModel, Field, create_model  # noqa: E402

from prompts import DOC_TYPE_FIELD_SCHEMAS  # noqa: E402


# ---------------------------------------------------------------------
# Completeness check (Table 3)
# ---------------------------------------------------------------------
def check_completeness(doc_type: str, extracted_fields: dict) -> Dict[str, List[str]]:
    """
    Compare extracted_fields against Table 3's required field list for
    this doc_type, and split any gaps into blocking vs non-blocking.

    Args:
        doc_type: "discharge_report" | "lab_report" | "bill" | "prescription"
        extracted_fields: dict of field_name -> value (from the Harvester
            / extraction prompt output)

    Returns:
        {"missing_blocking": [...], "missing_nonblocking": [...]}
    """
    schema = DOC_TYPE_FIELD_SCHEMAS.get(doc_type)
    if not schema:
        raise ValueError(
            f"No Table 3 field schema for doc_type='{doc_type}'. "
            f"Must be one of {list(DOC_TYPE_FIELD_SCHEMAS)}"
        )

    required = schema["required"]
    blocking = set(schema["blocking"])

    missing = [f for f in required if not extracted_fields.get(f)]

    return {
        "missing_blocking": [f for f in missing if f in blocking],
        "missing_nonblocking": [f for f in missing if f not in blocking],
    }


# ---------------------------------------------------------------------
# Dynamic elicitation schema builder
# ---------------------------------------------------------------------
def build_elicitation_schema(missing_fields: List[str]) -> Type[BaseModel]:
    """
    Build a Pydantic model on the fly with one required string field per
    missing (non-blocking) field, so the human reviewer gets a proper
    typed form. Per MCP spec, elicitation schemas must only use
    primitive types (str, int, float, bool) -- every field here is str,
    which the reviewer can normalize downstream if needed.
    """
    field_definitions = {
        field_name: (str, Field(description=f"Please provide a value for '{field_name}'"))
        for field_name in missing_fields
    }
    return create_model("MissingFieldsElicitation", **field_definitions)


# ---------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------
async def clinical_rules_engine_tool(ctx: Context, doc_type: str, extracted_fields: dict) -> Dict:
    """
    Run Table 3 completeness checks and resolve gaps.

    Args:
        ctx: MCP Context (used for ctx.elicit()).
        doc_type: "discharge_report" | "lab_report" | "bill" | "prescription"
        extracted_fields: dict of field_name -> value

    Returns:
        {status: "complete"|"resolved"|"unresolved"|"blocked",
         fields: dict,
         unresolved_fields: list[str]}   (only present when not "complete")
    """
    gaps = check_completeness(doc_type, extracted_fields)

    # Blocking fields missing -> straight to HITL escalation, no elicitation.
    if gaps["missing_blocking"]:
        return {
            "status": "blocked",
            "fields": extracted_fields,
            "unresolved_fields": gaps["missing_blocking"],
        }

    # Nothing missing at all.
    if not gaps["missing_nonblocking"]:
        return {"status": "complete", "fields": extracted_fields}

    # Only non-blocking gaps -> ask a human to fill them in.
    schema = build_elicitation_schema(gaps["missing_nonblocking"])
    result = await ctx.elicit(
        message=f"Missing fields for {doc_type}: {gaps['missing_nonblocking']}",
        schema=schema,
    )

    match result:
        case AcceptedElicitation(data=data):
            extracted_fields.update(data.model_dump())
            return {"status": "resolved", "fields": extracted_fields}
        case DeclinedElicitation():
            return {
                "status": "unresolved",
                "fields": extracted_fields,
                "unresolved_fields": gaps["missing_nonblocking"],
            }
        case CancelledElicitation():
            return {
                "status": "blocked",
                "fields": extracted_fields,
                "unresolved_fields": gaps["missing_nonblocking"],
            }
        case _:
            # Defensive fallback -- shouldn't happen, but don't silently
            # swallow an unrecognized elicitation outcome.
            return {
                "status": "unresolved",
                "fields": extracted_fields,
                "unresolved_fields": gaps["missing_nonblocking"],
            }


# ---------------------------------------------------------------------
# NOT part of this server -- documented here for completeness.
#
# The dashboard side of elicitation lives in the Streamlit HITL
# Dashboard (a separate app, page 3 of the overall system):
#
#   def elicitation_callback(schema, message):
#       """Renders st.form dynamically from schema fields, returns
#       reviewer input as an ElicitResult back to this server."""
#       ...
#
# Without a connected client that implements an elicitation handler,
# ctx.elicit() will raise "Method not found" / "elicitation not
# supported" -- see the test harness for how MCP Inspector handles
# this (it has a built-in Elicitation response UI).
# ---------------------------------------------------------------------