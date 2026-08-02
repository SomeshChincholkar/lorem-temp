"""
tools_reporter.py

The Clinical Insight Reporter primitive. Combines completeness gaps
(3.6), EHR cross-validation findings (3.7), and translation confidence
(3.5) into a single risk score, then writes both a JSON and an HTML
report per patient to data/reports/.

The exact report schema wasn't pinned down by a spec table (unlike
Table 3/4), so build_json_report()'s output shape is documented inline
below -- easy to adjust if you have a specific schema to match.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import Context  # noqa: E402

from common.rules_loader import get_risk_tier, get_rules_sha256, get_weight, load_rules  # noqa: E402
from resources import get_html_template  # noqa: E402

REPORTS_DIR = Path("data/reports")

# Maps our Table 4 rule_ids to the ACTUAL weight keys present in
# rules.yaml's risk_scoring_matrix.weights. See mcp_secondary/
# tools_risk_score.py's identical constant for the full explanation --
# duplicated here on purpose since the two servers are meant to stay
# independent of each other, both only depending on common/rules_loader.
RULE_ID_TO_WEIGHT_KEY = {
    "med_omission_check": "medication_omission",
    "allergy_contradiction_check": "allergy_contradiction",
    "diagnosis_mismatch_check": "diagnosis_mismatch",
    "follow_up_missing_check": "followup_missing",
    "lab_follow_up_mismatch_check": "abnormal_lab_unresolved",
    "discharge_approval_check": None,  # no weight in rules.yaml -- hard block instead
    "bill_settlement_check": "bill_unpaid_with_discharge_ok",
}


# ---------------------------------------------------------------------
# Risk scoring
# ---------------------------------------------------------------------
def compute_risk_score(gaps: Dict[str, List[str]], ehr_findings: List[dict], translation_confidence: float):
    """
    Args:
        gaps: {"missing_blocking": [...], "missing_nonblocking": [...]}
            from tools_rules_engine.check_completeness()
        ehr_findings: list[{rule_id, severity, triggered, evidence, action}]
            from tools_ehr_validator.ehr_validation_tool()
        translation_confidence: float 0-1, from tools_lang_bridge
            (pass 1.0 if the document never went through translation)

    Returns:
        (score: int, forced_high: bool)

    forced_high=True overrides the tier to "high" regardless of the
    numeric score, whenever: a Table 3 blocking field is missing, a
    Critical Table 4 rule is triggered (action="Block discharge"), or
    translation_confidence is below quality_thresholds.translation_confidence_min.
    """
    rules = load_rules()
    score = 0
    forced_high = False

    missing_blocking = gaps.get("missing_blocking", [])
    if missing_blocking:
        forced_high = True
    for field in missing_blocking:
        if field == "address":
            score += get_weight("missing_address")
        elif field == "gender":
            score += get_weight("missing_gender")
        else:
            score += get_weight("missing_mandatory_field")

    for finding in ehr_findings:
        if not finding.get("triggered"):
            continue
        weight_key = RULE_ID_TO_WEIGHT_KEY.get(finding["rule_id"])
        if weight_key:
            score += get_weight(weight_key)
        if finding.get("severity") == "Critical":
            forced_high = True

    confidence_min = rules.get("quality_thresholds", {}).get("translation_confidence_min", 0.70)
    if translation_confidence is not None and translation_confidence < confidence_min:
        score += get_weight("low_translation_confidence")
        forced_high = True

    return score, forced_high


# ---------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------
def build_json_report(patient_id: str, all_inputs: dict) -> dict:
    """
    Expected shape of all_inputs:
    {
        "extracted_discharge": dict,   # Table 3 discharge_report fields
        "extracted_bill": dict,        # Table 3 bill fields
        "completeness_gaps": {"missing_blocking": [...], "missing_nonblocking": [...]},
        "ehr_findings": [{rule_id, severity, triggered, evidence, action}, ...],
        "translation_confidence": float | None,
    }

    Returns a dict with this shape:
    {
        "patient_id": str,
        "patient_name": str | None,
        "generated_at": str (ISO 8601 UTC),
        "risk_score": int,
        "risk_level": "low" | "medium" | "high",
        "recommendation": str,
        "discharge_blocked": bool,
        "completeness": {"missing_blocking": [...], "missing_nonblocking": [...]},
        "ehr_findings": [...],
        "translation_confidence": float | None,
        "rules_version": str (sha256 of rules.yaml),
    }
    """
    extracted_discharge = all_inputs.get("extracted_discharge", {})
    completeness_gaps = all_inputs.get("completeness_gaps", {"missing_blocking": [], "missing_nonblocking": []})
    ehr_findings = all_inputs.get("ehr_findings", [])
    translation_confidence = all_inputs.get("translation_confidence")

    score, forced_high = compute_risk_score(completeness_gaps, ehr_findings, translation_confidence)
    tier = "high" if forced_high else get_risk_tier(score)

    recommendations = load_rules().get("reporting", {}).get("recommendations", {})
    recommendation = recommendations.get(tier, "")

    return {
        "patient_id": patient_id,
        "patient_name": extracted_discharge.get("patient_name"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "risk_score": score,
        "risk_level": tier,
        "recommendation": recommendation,
        "discharge_blocked": forced_high,
        "completeness": completeness_gaps,
        "ehr_findings": ehr_findings,
        "translation_confidence": translation_confidence,
        "rules_version": get_rules_sha256(),
        "discharge_diagnosis": extracted_discharge.get("discharge_diagnosis"),
        "follow_up_instructions": extracted_discharge.get("discharge_instructions")
        or extracted_discharge.get("follow_up_appointments"),
    }


def render_html_report(json_report: dict) -> str:
    """Fill templates/discharge_summary.html placeholders from a JSON report."""
    template = get_html_template()

    findings_rows = "\n".join(
        "<tr><td>{rule_id}</td><td>{severity}</td><td>{triggered}</td><td>{evidence}</td></tr>".format(
            rule_id=f["rule_id"],
            severity=f["severity"],
            triggered="Yes" if f.get("triggered") else "No",
            evidence=f.get("evidence", ""),
        )
        for f in json_report.get("ehr_findings", [])
    ) or "<tr><td colspan=\"4\">No cross-validation findings.</td></tr>"

    replacements = {
        "{{patient_id}}": str(json_report.get("patient_id", "")),
        "{{patient_name}}": str(json_report.get("patient_name") or "Unknown"),
        "{{discharge_diagnosis}}": str(json_report.get("discharge_diagnosis") or "Not specified"),
        "{{risk_level}}": str(json_report.get("risk_level", "")).upper(),
        "{{risk_level_lower}}": str(json_report.get("risk_level", "")).lower(),
        "{{recommendation}}": str(json_report.get("recommendation", "")),
        "{{findings_rows}}": findings_rows,
        "{{follow_up_instructions}}": str(json_report.get("follow_up_instructions") or "None documented"),
    }

    html = template
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)
    return html


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------
async def clinical_insight_reporter_tool(ctx: Context, patient_id: str, all_inputs: dict) -> Dict:
    """
    Build and persist the final JSON + HTML report for a patient.

    Args:
        ctx: MCP Context (used only for optional progress logging).
        patient_id: e.g. "P1019"
        all_inputs: see build_json_report()'s docstring for expected shape.

    Returns:
        {json_path, html_path, risk_level, recommendation, discharge_blocked}
    """
    if ctx is not None:
        try:
            await ctx.info(f"Building report for patient {patient_id}...")
        except Exception:
            pass  # logging is best-effort, never fail the tool over it

    report = build_json_report(patient_id, all_inputs)
    html = render_html_report(report)

    json_path = REPORTS_DIR / f"{patient_id}_report.json"
    html_path = REPORTS_DIR / f"{patient_id}_report.html"

    _write_file(json_path, json.dumps(report, indent=2))
    _write_file(html_path, html)

    return {
        "json_path": str(json_path),
        "html_path": str(html_path),
        "risk_level": report["risk_level"],
        "recommendation": report["recommendation"],
        "discharge_blocked": report["discharge_blocked"],
    }