"""
guardrails/injection.py

Prompt injection guard (spec Table 12, row 3).

Trigger: user query matches injection patterns.
Action:  sanitize or reject; log alert.

Where it runs: the RAG Q&A path, because that is the only place in this
system where free user text reaches an LLM. Everything else operates on
clinical documents, not on prompts a user composed.

Two tiers, by consequence:

  REJECT   -- the query is trying to override instructions or exfiltrate
              the system prompt. There is no legitimate clinical reading
              of it, so it never reaches a model.
  SANITIZE -- the query contains role markers or delimiters that could
              confuse prompt assembly, but a real question is plausibly
              in there. Strip the markers and continue.

Deliberately conservative on REJECT: a hospital administrator wrongly
blocked from asking about a patient is a real cost, so only patterns
with no benign clinical meaning are rejected outright.
"""

import re

REJECT = "reject"
SANITIZE = "sanitize"
ALLOW = "allow"

# No legitimate clinical question looks like any of these.
REJECT_PATTERNS = [
    (r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+instructions?",
     "instruction override"),
    (r"disregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+",
     "instruction override"),
    (r"forget\s+(everything|all)\s+(you|above|before)", "instruction override"),
    (r"(reveal|show|print|repeat|output)\s+(me\s+)?(your|the)\s+"
     r"(system\s+)?(prompt|instructions?|rules?)", "system prompt exfiltration"),
    (r"what\s+(are|were)\s+your\s+(original\s+)?(system\s+)?instructions?",
     "system prompt exfiltration"),
    (r"you\s+are\s+now\s+(a|an|no longer)", "persona override"),
    (r"pretend\s+(to\s+be|you\s+are)", "persona override"),
    (r"act\s+as\s+(if\s+you\s+are\s+)?(a|an)\s+\w+\s+(without|with\s+no)\s+"
     r"(restrictions?|filters?|rules?)", "persona override"),
    (r"\bDAN\b|\bdeveloper\s+mode\b|\bjailbreak\b", "known jailbreak"),
    (r"(bypass|disable|turn\s+off)\s+(the\s+)?(safety|guardrails?|filters?)",
     "guardrail bypass"),
]

# Structural markers that could break out of the prompt's context block.
SANITIZE_PATTERNS = [
    (r"<\s*/?\s*(system|assistant|user|instructions?)\s*>", "role tag"),
    (r"^\s*(system|assistant)\s*:", "role prefix"),
    (r"\[/?INST\]|<\|im_(start|end)\|>|<\|endoftext\|>", "chat template token"),
    (r"```+\s*(system|instructions?)", "fenced instruction block"),
]

_REJECT = [(re.compile(p, re.IGNORECASE), label) for p, label in REJECT_PATTERNS]
_SANITIZE = [(re.compile(p, re.IGNORECASE | re.MULTILINE), label)
             for p, label in SANITIZE_PATTERNS]


class PromptInjectionGuard:
    """Screens free-text user queries before they reach an LLM."""

    name = "PromptInjectionGuard"

    def check(self, query: str) -> dict:
        """
        Returns:
            {"action": "allow"|"sanitize"|"reject",
             "is_injection": bool,
             "sanitized_query": str,
             "matches": [label, ...]}

        Callers must use sanitized_query, not the original, whenever
        action is "sanitize".
        """
        if not query or not query.strip():
            return {
                "action": ALLOW,
                "is_injection": False,
                "sanitized_query": query or "",
                "matches": [],
            }

        rejected = [label for pattern, label in _REJECT if pattern.search(query)]
        if rejected:
            return {
                "action": REJECT,
                "is_injection": True,
                "sanitized_query": "",
                "matches": sorted(set(rejected)),
            }

        sanitized = query
        matched: list[str] = []
        for pattern, label in _SANITIZE:
            sanitized, hits = pattern.subn(" ", sanitized)
            if hits:
                matched.append(label)

        sanitized = re.sub(r"\s+", " ", sanitized).strip()

        if matched:
            return {
                "action": SANITIZE,
                "is_injection": True,
                "sanitized_query": sanitized,
                "matches": sorted(set(matched)),
            }

        return {
            "action": ALLOW,
            "is_injection": False,
            "sanitized_query": query,
            "matches": [],
        }
