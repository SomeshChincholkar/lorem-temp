"""
common/rules_loader.py
=======================
Single source of truth for reading configs/rules.yaml. Imported directly
(not over the network) by MCP server tools — it's a library, not a service.

Used by: Clinical Data Completeness Agent, EHR Validation Agent,
Reporting Agent. Every audit report stamps get_rules_sha256() as
`rules_version` for compliance reproducibility.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

DEFAULT_RULES_PATH = "configs/rules.yaml"

# path -> parsed rules dict
_cache: dict[str, dict[str, Any]] = {}
# path -> sha256 hex digest
_sha_cache: dict[str, str] = {}


def load_rules(path: str = DEFAULT_RULES_PATH) -> dict[str, Any]:
    """Parse rules.yaml once per path and cache the result."""
    if path not in _cache:
        with open(path, encoding="utf-8") as f:
            _cache[path] = yaml.safe_load(f)
    return _cache[path]


def get_rules_sha256(path: str = DEFAULT_RULES_PATH) -> str:
    """Hex digest of the raw file bytes — stamped as `rules_version` on
    every audit report for compliance reproducibility."""
    if path not in _sha_cache:
        data = Path(path).read_bytes()
        _sha_cache[path] = hashlib.sha256(data).hexdigest()
    return _sha_cache[path]


def get_mandatory_fields(doc_type: str, path: str = DEFAULT_RULES_PATH) -> list[str]:
    """doc_type: 'clinical' | 'prescription'."""
    rules = load_rules(path)
    key_map = {
        "clinical": "mandatory_clinical_fields",
        "prescription": "mandatory_prescription_fields",
    }
    if doc_type not in key_map:
        raise ValueError(f"unknown doc_type: {doc_type!r} (expected 'clinical' or 'prescription')")
    return rules[key_map[doc_type]]


def get_weight(risk_key: str, path: str = DEFAULT_RULES_PATH) -> int:
    """e.g. get_weight('allergy_contradiction') -> 8"""
    rules = load_rules(path)
    weights = rules["risk_scoring_matrix"]["weights"]
    if risk_key not in weights:
        raise KeyError(f"unknown risk_key: {risk_key!r}")
    return weights[risk_key]


def get_risk_tier(score: int, path: str = DEFAULT_RULES_PATH) -> str:
    """score -> 'low' | 'medium' | 'high'."""
    rules = load_rules(path)
    t = rules["risk_scoring_matrix"]["thresholds"]
    if score <= t["low_max"]:
        return "low"
    if score <= t["medium_max"]:
        return "medium"
    return "high"


def is_hard_guardrail(rule_id: str, path: str = DEFAULT_RULES_PATH) -> bool:
    """Membership check against risk_scoring_matrix.hitl_hard_guardrails."""
    rules = load_rules(path)
    return rule_id in rules["risk_scoring_matrix"]["hitl_hard_guardrails"]


def expand_abbreviation(token: str, path: str = DEFAULT_RULES_PATH) -> str:
    """e.g. expand_abbreviation('HTN') -> 'Hypertension'.
    Falls back to the original token, unchanged, if not found."""
    rules = load_rules(path)
    abbrev_map = rules["normalization_standards"]["abbreviation_map"]
    return abbrev_map.get(token, token)


def get_icd10(diagnosis_text: str, path: str = DEFAULT_RULES_PATH) -> str | None:
    """Normalized diagnosis string -> ICD-10 code, or None if not found."""
    rules = load_rules(path)
    icd10_map = rules["normalization_standards"]["icd10_map"]
    return icd10_map.get(diagnosis_text)