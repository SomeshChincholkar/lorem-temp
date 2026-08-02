"""
agents/extractor_agent/test_language_detect.py

Sanity check for language_detect.py.

Tier-1 cases (langdetect, long enough text) need no external
dependencies beyond langdetect itself -- run these first.

Tier-2 case (LLM fallback, short/ambiguous text) needs your Bedrock
credentials in .env to actually be live -- it's separated out so a
missing/expired AWS credential doesn't make you think langdetect is
broken.

Usage:
    python -m agents.extractor_agent.test_language_detect
"""

import asyncio

from .language_detect import detect_language

# Long enough that langdetect alone should nail these -- no LLM call
# should happen for any of these.
TIER1_SAMPLES = [
    ("English discharge note", "Patient was admitted with chest pain and discharged in stable condition after treatment.", "en"),
    ("Hindi sample", "मरीज़ को सीने में दर्द के साथ भर्ती किया गया था और उपचार के बाद स्थिर स्थिति में छुट्टी दे दी गई।", "hi"),
    ("Spanish sample", "El paciente fue ingresado con dolor en el pecho y dado de alta en condición estable después del tratamiento.", "es"),
    ("German sample", "Der Patient wurde mit Brustschmerzen aufgenommen und nach der Behandlung in stabilem Zustand entlassen.", "de"),
    ("Empty text -- hard fallback, no LLM call", "", "en"),
]

# Deliberately short/ambiguous -- forces the LLM fallback tier.
# Requires working Bedrock credentials.
TIER2_SAMPLES = [
    ("Short Hindi fragment (forces LLM fallback)", "नमस्ते डॉक्टर", "hi"),
    ("Short Spanish fragment (forces LLM fallback)", "Hola doctor", "es"),
]


async def run_cases(samples):
    all_ok = True
    for label, text, expected in samples:
        detected = await detect_language(text)
        ok = detected == expected
        all_ok = all_ok and ok
        status = "OK  " if ok else "MISS"
        print(f"[{status}] {label}: expected={expected!r} detected={detected!r}")
    return all_ok


async def main():
    print("---- Tier 1: langdetect (no AWS needed) ----")
    tier1_ok = await run_cases(TIER1_SAMPLES)

    print("\n---- Tier 2: LLM fallback (needs live Bedrock creds) ----")
    tier2_ok = await run_cases(TIER2_SAMPLES)

    print("\nTier 1 all passed" if tier1_ok else "\nTier 1 has mismatches -- check langdetect install")
    print("Tier 2 all passed" if tier2_ok else "Tier 2 has mismatches -- check AWS creds/region in .env, "
                                                "or the model may genuinely be wrong on very short fragments")


if __name__ == "__main__":
    asyncio.run(main())