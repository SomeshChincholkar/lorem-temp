"""
mcp_secondary/tools_risk_score.py

calculate_risk_score_tool. Deliberately reimplements the same scoring
logic as mcp_primary/tools_reporter.py's compute_risk_score(), rather
than importing from mcp_primary, so the Secondary Analytics Server
stays standalone -- agents that only have this server mounted (no
primary server) can still get a risk score. Both sides only depend on
the shared common/rules_loader.py, which is the actual source of
truth for weights/tiers.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import Context  # noqa: E402

from common.rules_loader import get_risk_tier, get_weight, load_rules  # noqa: E402

# Maps our Table 4 rule_ids to the ACTUAL weight keys present in
# rules.yaml's risk_scoring_matrix.weights. Names don't match 1:1
# (e.g. "med_omission_check" -> "medication_omission", not
# "med_omission"), and one rule has NO corresponding weight at all:
#
#   discharge_approval_check -> None
#
# rules.yaml has no weight for a missing discharge approval. Since
# Table 4 marks this rule Critical / "Block discharge", and
# business_rules.discharge_ok_field_required is true in rules.yaml,
# we treat it as an automatic hard block rather than silently scoring
# it as 0 -- see the `forced_high` logic in compute_risk_score below.
RULE_ID_TO_WEIGHT_KEY = {
    "med_omission_check": "medication_omission",
    "allergy_contradiction_check": "allergy_contradiction",
    "diagnosis_mismatch_check": "diagnosis_mismatch",
    "follow_up_missing_check": "followup_missing",
    "lab_follow_up_mismatch_check": "abnormal_lab_unresolved",
    "discharge_approval_check": None,  # no weight in rules.yaml -- hard block instead
    "bill_settlement_check": "bill_unpaid_with_discharge_ok",
}


def compute_risk_score(
    completeness_gaps: Dict[str, List[str]],
    ehr_findings: List[dict],
    translation_confidence: Optional[float],
):
    """
    Same formula as mcp_primary/tools_reporter.compute_risk_score().

    Returns:
        (score: int, forced_high: bool)

    forced_high is True whenever the case must be treated as "high"
    risk regardless of the numeric score, per rules.yaml semantics:
      - a Table 3 blocking field is missing (auto-generation is
        blocked outright, per section 2.4.1)
      - a Critical-severity Table 4 rule is triggered (action =
        "Block discharge")
      - translation_confidence falls below
        quality_thresholds.translation_confidence_min (rules.yaml
        explicitly lists "translation_confidence_below_threshold" as
        a hitl_hard_guardrails entry)
    """
    rules = load_rules()
    score = 0
    forced_high = False

    missing_blocking = completeness_gaps.get("missing_blocking", [])
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


async def calculate_risk_score_tool(
    ctx: Context,
    completeness_gaps: dict,
    ehr_findings: list,
    translation_confidence: Optional[float] = None,
) -> Dict:
    """
    Args:
        ctx: MCP Context (unused -- pure computation, no I/O).
        completeness_gaps: {"missing_blocking": [...], "missing_nonblocking": [...]}
        ehr_findings: list[{rule_id, severity, triggered, evidence, action}]
        translation_confidence: float 0-1, or None if not applicable.

    Returns:
        {score: int, tier: "low" | "medium" | "high", forced_high: bool}

    forced_high=True means "high" was applied as a hard override
    regardless of the numeric score -- see compute_risk_score()'s
    docstring for exactly which conditions trigger this.
    """
    score, forced_high = compute_risk_score(completeness_gaps, ehr_findings, translation_confidence)
    tier = "high" if forced_high else get_risk_tier(score)
    return {"score": score, "tier": tier, "forced_high": forced_high}