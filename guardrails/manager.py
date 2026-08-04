"""
guardrails/manager.py

GuardrailManager (spec Table 12, row 5) plus the single entry point the
rest of the system imports.

Row 5's own rule is the HITL escalation: risk_level=High or
discharge_blocked=True means mandatory human review, no auto-approve --
regardless of what any individual agent concluded.

The manager also composes the other four guardrails so call sites depend
on one object rather than four, and so every intervention lands in one
event log. That log is what LangFuse will consume as guardrail spans
(spec 7.2) once observability is wired.
"""

from datetime import datetime, timezone
from typing import Iterable

from .hallucination import HallucinationChecker
from .injection import PromptInjectionGuard
from .pii import PIIRedactor
from .toxicity import ToxicityFilter


def guardrail_manager(risk_level: str | None, discharge_blocked: bool | None) -> dict:
    """
    Spec Table 12's HITL Escalation rule.

    Kept as a module-level function because the Orchestrator's pipeline
    imports it directly and it needs no state.
    """
    requires_hitl = bool(discharge_blocked) or str(risk_level or "").lower() == "high"
    return {"requires_hitl": requires_hitl}


class GuardrailManager:
    """
    Composes all five guardrails and records every intervention.

    Each check_* method returns the guardrail's own result dict and
    appends an event to self.events, so a caller can act on the result
    and ship the audit trail without doing bookkeeping at every site.
    """

    def __init__(self, known_names: Iterable[str] | None = None, trace_id: str | None = None):
        self.pii = PIIRedactor(known_names=known_names)
        self.injection = PromptInjectionGuard()
        self.toxicity = ToxicityFilter()
        self.hallucination = HallucinationChecker()
        self.trace_id = trace_id
        self.events: list[dict] = []

    # -----------------------------------------------------------------
    def _record(self, guardrail: str, triggered: bool, detail: dict) -> None:
        self.events.append(
            {
                "guardrail": guardrail,
                "triggered": triggered,
                "trace_id": self.trace_id,
                "at": datetime.now(timezone.utc).isoformat(),
                **detail,
            }
        )

    # -----------------------------------------------------------------
    def redact(self, text: str) -> str:
        """Mask identifiers before logging or an external call."""
        result = self.pii.inspect(text)
        self._record("PIIRedactor", result["count"] > 0,
                     {"found": result["found"], "count": result["count"]})
        return result["redacted"]

    def redact_payload(self, payload: dict) -> dict:
        redacted = self.pii.redact_dict(payload)
        self._record("PIIRedactor", redacted != payload, {"scope": "payload"})
        return redacted

    def check_query(self, query: str) -> dict:
        """Screen a user query before it reaches an LLM."""
        result = self.injection.check(query)
        self._record("PromptInjectionGuard", result["is_injection"],
                     {"action": result["action"], "matches": result["matches"]})
        return result

    def check_output(self, text: str) -> dict:
        """Screen generated text before a patient reads it."""
        result = self.toxicity.check(text)
        self._record("ToxicityFilter", result["verdict"] != "safe",
                     {"verdict": result["verdict"], "categories": result["categories"]})
        return result

    def check_answer(self, answer: str, triad: dict | None, attempt: int = 0) -> dict:
        """Screen a RAG answer for grounding."""
        result = self.hallucination.check(answer, triad, attempt=attempt)
        self._record("HallucinationChecker", result["blocked"],
                     {"faithfulness": result["faithfulness"],
                      "threshold": result["threshold"],
                      "reason": result["reason"]})
        return result

    def check_escalation(self, risk_level: str | None, discharge_blocked: bool | None) -> dict:
        """Table 12 row 5 -- the mandatory-human-review rule."""
        result = guardrail_manager(risk_level, discharge_blocked)
        self._record("GuardrailManager", result["requires_hitl"],
                     {"risk_level": risk_level, "discharge_blocked": discharge_blocked})
        return result

    # -----------------------------------------------------------------
    def triggered_events(self) -> list[dict]:
        return [e for e in self.events if e["triggered"]]

    def summary(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "checks_run": len(self.events),
            "interventions": len(self.triggered_events()),
            "events": self.events,
        }
