"""
common/llm.py

Single factory for the Bedrock LLM client used across every agent, so
model_id / region / .env loading isn't duplicated in each agent.
"""

import json
import os
import re

from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse

load_dotenv()  # reads FA5_Project/.env (AWS creds, region, model id, etc.)

DEFAULT_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "cohere.command-r-plus-v1:0")
DEFAULT_REGION = os.getenv("AWS_REGION", "us-east-1")

_llm_cache: dict[str, ChatBedrockConverse] = {}


def get_llm(model_id: str = DEFAULT_MODEL_ID, region_name: str = DEFAULT_REGION) -> ChatBedrockConverse:
    """Return a cached ChatBedrockConverse client for this model/region pair."""
    key = f"{model_id}::{region_name}"
    if key not in _llm_cache:
        _llm_cache[key] = ChatBedrockConverse(model_id=model_id, region_name=region_name)
    return _llm_cache[key]


def safe_json_parse(text: str):
    """
    LLMs sometimes wrap JSON in ```json fences or add stray prose even
    when told not to. Strip fences, try a direct parse, then fall back
    to extracting the first top-level {...} or [...] block.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()

    attempts = [cleaned]
    # Most common LLM JSON slip: a trailing comma before a closing } or ].
    no_trailing_commas = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    if no_trailing_commas != cleaned:
        attempts.append(no_trailing_commas)

    for attempt in attempts:
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue

    match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
    if match:
        block = match.group(1)
        block_fixed = re.sub(r",(\s*[}\]])", r"\1", block)
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            return json.loads(block_fixed)

    raise ValueError(f"Could not parse JSON from LLM output: {text[:200]}...")