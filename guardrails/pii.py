"""
guardrails/pii.py

PII/PHI redaction (spec Table 12, row 1).

Trigger: any text containing patient name, phone, Aadhaar, PAN.
Action:  mask before logging or sending to an external API.

Regex-based rather than an NER model, deliberately. Patient names come
from the extracted record, so they are known strings to be masked
exactly -- guessing at them with a model would be both slower and less
reliable. The identifier formats (Aadhaar, PAN, phone) are strictly
specified, which is precisely what regex is good at.

Redaction is one-way and is applied to copies. It must never run on the
text the clinical agents reason over -- masking a medication name or a
patient ID mid-pipeline would corrupt validation. Call it at the log and
external-call boundary only.
"""

import re
from typing import Iterable

MASK = "[REDACTED]"

# Aadhaar: 12 digits, conventionally spaced 4-4-4. Excludes leading 0/1
# per UIDAI, which also stops it swallowing ordinary 12-digit numbers.
AADHAAR_RE = re.compile(r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b")

# PAN: five letters, four digits, one letter.
PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")

# Indian mobile (+91) and US-style numbers, the two formats in the corpus.
PHONE_RE = re.compile(
    r"(\+?\d{1,3}[\s-]?)?"
    r"(\(?\d{3}\)?[\s.-]?)"
    r"\d{3}[\s.-]?\d{4}\b"
)

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")

# Medical record numbers, e.g. "MRN: 88213".
MRN_RE = re.compile(r"\bMRN[:\s#]*([A-Z0-9-]{4,})\b", re.IGNORECASE)

PATTERNS = (
    ("aadhaar", AADHAAR_RE),
    ("pan", PAN_RE),
    ("email", EMAIL_RE),
    ("mrn", MRN_RE),
    ("phone", PHONE_RE),
)


class PIIRedactor:
    """Masks direct identifiers in free text before it leaves the system."""

    name = "PIIRedactor"

    def __init__(self, known_names: Iterable[str] | None = None):
        # Names cannot be pattern-matched, so they are supplied from the
        # extracted record. Longest first, so "Thomas Wright" is masked
        # as a unit before "Thomas" would match inside it.
        self.known_names = sorted(
            {n.strip() for n in (known_names or []) if n and len(n.strip()) > 2},
            key=len,
            reverse=True,
        )

    def redact(self, text: str) -> str:
        return self.inspect(text)["redacted"]

    def inspect(self, text: str) -> dict:
        """
        Redact and report what was found.

        Returns {"redacted": str, "found": [kind, ...], "count": int} so a
        caller can log that redaction happened without logging what was
        redacted.
        """
        if not text:
            return {"redacted": text or "", "found": [], "count": 0}

        redacted = text
        found: list[str] = []
        count = 0

        for full_name in self.known_names:
            pattern = re.compile(rf"\b{re.escape(full_name)}\b", re.IGNORECASE)
            redacted, hits = pattern.subn(MASK, redacted)
            if hits:
                found.append("patient_name")
                count += hits

            # Also mask the surname alone -- clinical notes routinely
            # switch to "Mr Wright" after the first full mention.
            parts = full_name.split()
            if len(parts) > 1 and len(parts[-1]) > 3:
                surname = re.compile(rf"\b{re.escape(parts[-1])}\b", re.IGNORECASE)
                redacted, hits = surname.subn(MASK, redacted)
                if hits:
                    if "patient_name" not in found:
                        found.append("patient_name")
                    count += hits

        for kind, pattern in PATTERNS:
            redacted, hits = pattern.subn(MASK, redacted)
            if hits:
                found.append(kind)
                count += hits

        return {"redacted": redacted, "found": found, "count": count}

    def redact_dict(self, payload: dict, skip_keys: Iterable[str] = ()) -> dict:
        """
        Redact every string value in a dict, recursively.

        skip_keys preserves fields that are structural rather than
        identifying -- patient_id above all. It is the join key across
        the EHR, the reports and the vector store; masking it would make
        a log entry impossible to correlate, which defeats the point of
        logging it.
        """
        skip = set(skip_keys) | {"patient_id", "doc_type", "trace_id", "rule_id"}

        def walk(value, key=None):
            if key in skip:
                return value
            if isinstance(value, str):
                return self.redact(value)
            if isinstance(value, dict):
                return {k: walk(v, k) for k, v in value.items()}
            if isinstance(value, list):
                return [walk(v) for v in value]
            return value

        return walk(payload)
