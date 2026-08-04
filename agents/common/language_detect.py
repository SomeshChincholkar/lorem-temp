"""
agents/common/language_detect.py

Detects the language of harvested raw document text. Shared by the
Clinical Extractor Agent (to tag the source language of an extraction)
and the Clinical Normalizer Agent (to pick the sampling model hint and
decide whether translation is needed at all).

Two tiers, so this can never come back empty/None:

  1. langdetect (fast, offline, no model download) -- tried first for
     any text with enough signal.
  2. LLM fallback (Bedrock, via common.llm) -- used when langdetect
     can't decide (text too short/ambiguous) or raises.
  3. Hardcoded "en" -- absolute last resort if both of the above fail
     (e.g. empty text, or the LLM call itself errors out). This
     function is guaranteed to always return a 2-letter code and never
     raise.
"""

import re

from langdetect import DetectorFactory, LangDetectException, detect

from agents.common.llm import get_llm

DetectorFactory.seed = 0  # deterministic across runs

# Below this length, langdetect is unreliable enough that we skip
# straight to the LLM fallback instead of trusting it.
MIN_CHARS_FOR_LANGDETECT = 20

_LANG_CODE_RE = re.compile(r"^[a-z]{2}$")
HARD_FALLBACK = "en"


async def detect_language(text: str) -> str:
    """
    Always returns a 2-letter ISO 639-1 code. Never raises.
    """
    text = (text or "").strip()
    if not text:
        return HARD_FALLBACK

    if len(text) >= MIN_CHARS_FOR_LANGDETECT:
        try:
            return detect(text)
        except LangDetectException:
            pass  # fall through to the LLM tier

    llm_result = await _detect_language_via_llm(text)
    return llm_result or HARD_FALLBACK


async def _detect_language_via_llm(text: str) -> str | None:
    """
    Asks the Bedrock LLM to identify the language as a fallback for
    text langdetect couldn't handle. Returns None (never raises) if
    the call fails or the model doesn't return a clean 2-letter code.
    """
    prompt = (
        "Identify the primary language of the following text.\n"
        "Respond with ONLY the two-letter ISO 639-1 language code "
        "(e.g. en, hi, es, de, fr, nl) and nothing else -- no words, "
        "no punctuation, no explanation.\n\n"
        f"TEXT:\n{text[:1000]}"
    )
    try:
        llm = get_llm()
        response = await llm.ainvoke(prompt)
        code = re.sub(r"[^a-z]", "", response.content.strip().lower())[:2]
        if _LANG_CODE_RE.match(code):
            return code
    except Exception:
        pass
    return None