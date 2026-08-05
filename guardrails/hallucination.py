"""
guardrails/hallucination.py

Hallucination check (spec Table 12, row 2).

Trigger: RAG-generated response with faithfulness < 0.7.
Action:  block the response; request regeneration.

The faithfulness score itself comes from the Reflection agent's RAG Triad
(agents/rag_agent/roles.py). This module is the policy layer on top: it
reads the threshold from rules.yaml, decides block vs. allow, and caps how
many times a regeneration may be attempted.

Two decisions worth stating, because both are the strict reading:

  - An UNSCORED answer is treated as unverified, not as passing. If the
    judge call failed we do not know the answer is grounded, and "we
    couldn't check" must never be quietly equivalent to "it's fine".
  - Exhausting the retry budget yields the out-of-context refusal rather
    than the best of several ungrounded attempts. A confidently wrong
    answer about a patient's medication is worse than no answer.
"""

from common.rules_loader import load_rules

DEFAULT_FAITHFULNESS_MIN = 0.7
MAX_REGENERATION_ATTEMPTS = 2

BLOCKED_ANSWER = (
    "I don't know — this information is not available in the patient records."
)


def faithfulness_threshold() -> float:
    """
    rules.yaml stores this as rag_groundedness_min. Spec Table 12 quotes
    0.7 for the guardrail; rules.yaml ships 0.75, which is stricter.
    Config wins -- the whole point of rules.yaml is that thresholds are
    tunable without a code change.
    """
    try:
        thresholds = load_rules().get("quality_thresholds", {})
        return float(thresholds.get("rag_groundedness_min", DEFAULT_FAITHFULNESS_MIN))
    except Exception:
        return DEFAULT_FAITHFULNESS_MIN


class HallucinationChecker:
    """Blocks RAG answers that the Reflection agent could not ground."""

    name = "HallucinationChecker"

    def __init__(self, threshold: float | None = None,
                 max_attempts: int = MAX_REGENERATION_ATTEMPTS):
        self.threshold = faithfulness_threshold() if threshold is None else threshold
        self.max_attempts = max_attempts

    def check(self, answer: str, triad: dict | None, attempt: int = 0) -> dict:
        """
        Args:
            answer: the generated answer.
            triad:  RAG Triad scores, or None if scoring did not run.
            attempt: 0-based regeneration attempt already made.

        Returns:
            {"blocked": bool, "should_regenerate": bool,
             "faithfulness": float|None, "threshold": float,
             "reason": str|None, "safe_answer": str}
        """
        # The refusal string is already the safe outcome -- scoring it
        # would only ever produce a spurious block.
        if answer and answer.strip() == BLOCKED_ANSWER:
            return self._result(False, False, None, None, answer)

        faithfulness = (triad or {}).get("faithfulness")

        if faithfulness is None:
            return self._result(
                blocked=True,
                should_regenerate=attempt < self.max_attempts,
                faithfulness=None,
                reason="faithfulness could not be scored; answer is unverified",
                answer=answer,
            )

        if float(faithfulness) < self.threshold:
            return self._result(
                blocked=True,
                should_regenerate=attempt < self.max_attempts,
                faithfulness=float(faithfulness),
                reason=(
                    f"faithfulness {float(faithfulness):.2f} is below the "
                    f"{self.threshold:.2f} grounding threshold"
                ),
                answer=answer,
            )

        return self._result(False, False, float(faithfulness), None, answer)

    def _result(self, blocked, should_regenerate, faithfulness, reason, answer):
        return {
            "blocked": blocked,
            "should_regenerate": should_regenerate,
            "faithfulness": faithfulness,
            "threshold": self.threshold,
            "reason": reason,
            # Once regeneration is exhausted, refuse rather than ship an
            # answer we could not ground.
            "safe_answer": BLOCKED_ANSWER if blocked and not should_regenerate else answer,
        }
