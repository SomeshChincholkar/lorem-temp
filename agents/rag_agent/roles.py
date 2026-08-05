"""
agents/rag_agent/roles.py

RAG roles 2-5 of the five the spec requires (Table 5). Role 1, Indexing,
lives in indexing.py.

  2. Retrieval    -- question -> embedding -> top-k FAISS chunks
  3. Augmentation -- re-rank those chunks by keyword overlap
  4. Generation   -- grounded answer, prompt fetched via MCP Prompts
  5. Reflection   -- RAG Triad scores (faithfulness / answer relevance /
                     context relevance)

The out-of-context refusal string is fixed by the spec and must be
returned verbatim, so it lives here as a constant rather than being left
to the model's phrasing.
"""

import json
import re

import numpy as np

from agents.common.llm import get_llm, safe_json_parse
from agents.common.mcp_client import get_prompt_text

from .indexing import PatientNotIndexedError, embed_texts, load_patient_index

OUT_OF_CONTEXT_ANSWER = (
    "I don't know — this information is not available in the patient records."
)

# Below this cosine similarity a chunk is treated as unrelated, so a
# question about something absent from the records returns the refusal
# instead of the least-bad chunk in the corpus.
MIN_RELEVANCE_SCORE = 0.25

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "which", "who",
    "for", "of", "to", "in", "on", "and", "or", "did", "does", "do", "has",
    "have", "had", "this", "that", "his", "her", "their", "patient",
}


# ---------------------------------------------------------------------
# Role 2: Retrieval Agent
# ---------------------------------------------------------------------
def retrieve_top_k(question: str, patient_id: str, k: int = 5) -> list[dict]:
    """
    Embed the question and pull the k nearest chunks from ONE patient's
    index.

    patient_id is required and positional, not an optional filter. Each
    patient has their own FAISS index, so scoping is structural: there is
    no code path by which another patient's chunk can reach this result
    list. That also makes the top-k exact -- no over-fetch heuristic that
    could silently return fewer results than requested.

    Raises PatientNotIndexedError for an unknown patient. That must be a
    clear error rather than an empty list, which downstream would read as
    "nothing in their records" and produce a confidently wrong refusal.
    """
    if not patient_id:
        raise PatientNotIndexedError(
            "retrieve_top_k requires a patient_id -- questions must name a patient."
        )

    index, metadata = load_patient_index(patient_id)

    fetch_k = min(k, len(metadata))
    if fetch_k <= 0:
        return []

    query_vector = embed_texts([question])
    scores, indices = index.search(query_vector, fetch_k)

    results = []
    for score, position in zip(scores[0], indices[0]):
        if position < 0:
            continue
        results.append({**metadata[position], "score": float(score)})

    return results


# ---------------------------------------------------------------------
# Role 3: Augmentation Agent
# ---------------------------------------------------------------------
def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 2}


def rerank_by_keyword(question: str, chunks: list[dict]) -> list[dict]:
    """
    Blend vector similarity with literal keyword overlap.

    Pure embedding search is weak on exactly the tokens that matter most
    in clinical records -- drug names, ICD-10 codes, patient IDs. A
    chunk that literally contains "Amoxicillin" should outrank one that
    is merely semantically nearby.
    """
    question_keywords = _keywords(question)
    if not question_keywords:
        return chunks

    reranked = []
    for chunk in chunks:
        overlap = len(question_keywords & _keywords(chunk["chunk"])) / len(question_keywords)
        reranked.append({**chunk, "keyword_overlap": overlap,
                         "combined_score": 0.7 * chunk["score"] + 0.3 * overlap})

    return sorted(reranked, key=lambda c: c["combined_score"], reverse=True)


# ---------------------------------------------------------------------
# Role 4: Generation Agent
# ---------------------------------------------------------------------
async def build_answer_prompt(context_length: int) -> str:
    """
    Fetch rag-answer-prompt via MCP Prompts.

    The spec is explicit that this must be fetched, not hardcoded
    (section 2.6), so a failure here is not silently swallowed.
    """
    return await get_prompt_text("rag-answer-prompt", {"context_length": str(context_length)})


def has_grounding(chunks: list[dict]) -> bool:
    """Whether any retrieved chunk is relevant enough to answer from."""
    return any(c.get("score", 0.0) >= MIN_RELEVANCE_SCORE for c in chunks)


def build_generation_input(question: str, chunks: list[dict], base_prompt: str) -> str:
    context = "\n---\n".join(
        f"[source: {c['source_doc']}]\n{c['chunk']}" for c in chunks
    )
    return (
        f"{base_prompt}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"If the context does not contain the answer, reply with exactly: "
        f"{OUT_OF_CONTEXT_ANSWER}"
    )


# ---------------------------------------------------------------------
# Role 5: Reflection Agent -- RAG Triad
# ---------------------------------------------------------------------
TRIAD_PROMPT = """You are scoring a retrieval-augmented answer. Return ONLY JSON.

QUESTION:
{question}

CONTEXT GIVEN TO THE MODEL:
{context}

ANSWER PRODUCED:
{answer}

Score each dimension from 0.0 to 1.0:
- faithfulness: does every claim in the answer trace back to the context?
- answer_relevance: does the answer actually address the question?
- context_relevance: was the retrieved context relevant to the question?

Return exactly: {{"faithfulness": <float>, "answer_relevance": <float>, "context_relevance": <float>}}
"""


async def rag_triad_score(question: str, answer: str, chunks: list[dict]) -> dict:
    """
    LLM-as-judge scoring across the three RAG Triad dimensions.

    Never raises: a scoring failure must not invalidate an answer that
    was already produced. On failure the scores come back as None, which
    reads as "unscored" rather than as a passing grade.
    """
    context = "\n---\n".join(c["chunk"] for c in chunks)
    prompt = TRIAD_PROMPT.format(question=question, context=context, answer=answer)

    try:
        response = await get_llm().ainvoke(prompt)
        content = response.content
        if isinstance(content, list):
            content = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )
        scores = safe_json_parse(content)
        return {
            "faithfulness": float(scores.get("faithfulness", 0.0)),
            "answer_relevance": float(scores.get("answer_relevance", 0.0)),
            "context_relevance": float(scores.get("context_relevance", 0.0)),
        }
    except Exception:
        return {
            "faithfulness": None,
            "answer_relevance": None,
            "context_relevance": None,
            "error": "RAG triad scoring failed",
        }


def sources_for(chunks: list[dict]) -> list[dict]:
    """De-duplicated source list for the dashboard's source panel."""
    seen = {}
    for chunk in chunks:
        key = chunk["source_doc"]
        if key not in seen:
            seen[key] = {
                "source_doc": key,
                "doc_type": chunk.get("doc_type"),
                "patient_id": chunk.get("patient_id"),
                "score": round(chunk.get("score", 0.0), 4),
            }
    return list(seen.values())


def as_json(value) -> str:
    return json.dumps(value, indent=2, default=str)


def normalize_scores(vectors) -> np.ndarray:
    """Kept for callers that need raw vector normalization."""
    return np.asarray(vectors, dtype="float32")
