"""
guardrails/

Responsible AI guardrails (spec section 7.1, Table 12).

| Guardrail            | Module            | Trigger                              |
|----------------------|-------------------|--------------------------------------|
| PII/PHI Redaction    | pii.py            | text containing direct identifiers    |
| Hallucination Check  | hallucination.py  | RAG answer faithfulness below floor   |
| Prompt Injection     | injection.py      | user query matches injection patterns |
| Toxicity Filter      | toxicity.py       | generated clinical instruction text   |
| HITL Escalation      | manager.py        | risk_level=high or discharge blocked  |

Import GuardrailManager for anything that runs more than one check --
it composes all five and keeps one event log for the audit trail.
"""

from .hallucination import BLOCKED_ANSWER, HallucinationChecker
from .injection import ALLOW, REJECT, SANITIZE, PromptInjectionGuard
from .manager import GuardrailManager, guardrail_manager
from .pii import PIIRedactor
from .toxicity import ToxicityFilter

__all__ = [
    "GuardrailManager",
    "guardrail_manager",
    "PIIRedactor",
    "PromptInjectionGuard",
    "ToxicityFilter",
    "HallucinationChecker",
    "BLOCKED_ANSWER",
    "ALLOW",
    "REJECT",
    "SANITIZE",
]
