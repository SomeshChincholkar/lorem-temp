"""
guardrails/toxicity.py

Toxicity filter (spec Table 12, row 4).

Trigger: LLM output destined for clinical instructions.
Action:  filter before including it in a summary.

The realistic failure mode in a discharge summary is not profanity --
it is a model producing text that alarms, blames, or gives dangerous
self-care advice to a patient who is reading it alone at home. So this
filter checks three categories:

  toxic     -- abusive or demeaning language about the patient
  alarming  -- fatalistic phrasing that would frighten rather than inform
  unsafe    -- advice that contradicts the discharge instructions, such as
               telling a patient to stop taking a prescribed medication

Keyword/pattern based, so it is deterministic and auditable. A model-based
classifier would catch more, but would also make "why was this blocked?"
unanswerable, which matters more in a clinical audit trail.
"""

import re

SAFE = "safe"
FLAGGED = "flagged"
BLOCKED = "blocked"

TOXIC_PATTERNS = [
    (r"\b(stupid|idiot|idiotic|moron|worthless|pathetic)\b", "abusive language"),
    (r"\bpatient\s+(is\s+)?(non-?compliant\s+and\s+)?(lazy|difficult|annoying)\b",
     "demeaning characterization"),
    (r"\b(drug\s?seeker|malinger(er|ing)|frequent\s+flyer)\b", "stigmatizing label"),
]

ALARMING_PATTERNS = [
    (r"\byou\s+(will|are\s+going\s+to)\s+die\b", "fatalistic phrasing"),
    (r"\b(there\s+is\s+)?(no|nothing\s+more\s+we\s+can\s+do|hopeless)\b.*\b(hope|treatment|cure)\b",
     "hopeless phrasing"),
    (r"\bterminal\b.*\byou\s+have\s+\w+\s+(weeks?|months?)\s+(left|to\s+live)\b",
     "prognosis without clinical context"),
]

# The highest-consequence category: text that would undo the discharge plan.
UNSAFE_PATTERNS = [
    (r"\b(stop|discontinue|quit)\s+(taking\s+)?(all\s+)?(your\s+)?(medications?|medicines?|pills?)\b",
     "instructs stopping medication"),
    (r"\bdo\s?n[o']?t\s+(take|follow|attend)\s+(your\s+)?"
     r"(medications?|medicines?|prescription|follow-?up|appointment)",
     "contradicts discharge plan"),
    (r"\b(ignore|skip)\s+(your\s+)?(doctor|physician|follow-?up|appointment)s?\b",
     "discourages follow-up"),
    (r"\bdouble\s+(the\s+)?dose\b|\btake\s+(extra|more)\s+than\s+prescribed\b",
     "unsafe dosing advice"),
    (r"\bno\s+need\s+to\s+(see|contact|call)\s+(a\s+)?(doctor|physician|hospital)\b",
     "discourages seeking care"),
]

CATEGORIES = (
    ("unsafe", UNSAFE_PATTERNS, BLOCKED),
    ("toxic", TOXIC_PATTERNS, BLOCKED),
    ("alarming", ALARMING_PATTERNS, FLAGGED),
)

_COMPILED = [
    (category, [(re.compile(p, re.IGNORECASE), label) for p, label in patterns], verdict)
    for category, patterns, verdict in CATEGORIES
]

REPLACEMENT = (
    "[This passage was withheld by a safety check. Please follow the "
    "instructions on your printed discharge paperwork and contact your "
    "care team if anything is unclear.]"
)


class ToxicityFilter:
    """Screens generated text before it is shown to a patient."""

    name = "ToxicityFilter"

    def check(self, text: str) -> dict:
        """
        Returns:
            {"verdict": "safe"|"flagged"|"blocked",
             "categories": [...], "matches": [label, ...],
             "filtered_text": str}

        "blocked" replaces the text outright rather than editing around
        the match: a sentence recommending a patient stop their
        medication cannot be made safe by deleting one clause.
        """
        if not text or not text.strip():
            return {
                "verdict": SAFE,
                "categories": [],
                "matches": [],
                "filtered_text": text or "",
            }

        categories: list[str] = []
        matches: list[str] = []
        verdict = SAFE

        for category, patterns, category_verdict in _COMPILED:
            for pattern, label in patterns:
                if pattern.search(text):
                    if category not in categories:
                        categories.append(category)
                    matches.append(label)
                    if category_verdict == BLOCKED:
                        verdict = BLOCKED
                    elif verdict != BLOCKED:
                        verdict = FLAGGED

        return {
            "verdict": verdict,
            "categories": categories,
            "matches": sorted(set(matches)),
            "filtered_text": REPLACEMENT if verdict == BLOCKED else text,
        }

    def filter(self, text: str) -> str:
        """Convenience wrapper returning only the safe-to-show text."""
        return self.check(text)["filtered_text"]
