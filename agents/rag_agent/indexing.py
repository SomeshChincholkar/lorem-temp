"""
agents/rag_agent/indexing.py

RAG role 1 of 5 -- the Indexing Agent (spec Table 5).

Parses every patient document and validation report, chunks them, embeds
with sentence-transformers/all-MiniLM-L6-v2, and writes a FAISS index to
data/vector_db/.

The embedding model is loaded lazily. Importing this module must stay
cheap, because the A2A server imports it at startup and pulling ~90MB of
model weights on import would make the server look hung.
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

VECTOR_DB_DIR = Path(os.getenv("VECTOR_DB_DIR", "data/vector_db"))
INDEX_PATH = VECTOR_DB_DIR / "clinical.index"
METADATA_PATH = VECTOR_DB_DIR / "chunks.json"

INCOMING_DIR = Path(os.getenv("INPUT_ROOT_DIR", "Data/incoming"))
REPORTS_DIR = Path("Data/reports")

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

_model = None


def get_embedding_model():
    """Load the sentence-transformers model once, on first use."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed and L2-normalize, so a FAISS inner-product index gives cosine
    similarity directly.
    """
    model = get_embedding_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def chunk_document(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding-window split. Overlap keeps facts from being cut in half."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(text):
            break
    return chunks


def _collect_source_documents() -> list[dict]:
    """
    Gather every indexable document: the raw patient paperwork plus the
    validation reports (so questions about risk level and findings are
    answerable too).
    """
    # Imported here rather than at module scope: the harvester pulls in
    # pdfplumber/pytesseract, which are heavy and only needed when an
    # index is actually being built.
    #
    # Both paths are needed: the project root so "mcp_primary.*"
    # resolves, and mcp_primary itself because the harvester imports its
    # siblings flat ("from roots import ...") for the MCP server's own
    # run-from-that-directory layout.
    import sys

    project_root = Path(__file__).resolve().parents[2]
    for path in (project_root, project_root / "mcp_primary"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from mcp_primary.tools_harvester import extract_text_any_format

    documents = []

    for folder in ("doctor_reports", "lab_reports", "bills"):
        directory = INCOMING_DIR / folder
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            try:
                text = extract_text_any_format(path)
            except Exception:
                # A single unreadable file must not abort the whole
                # index build.
                continue
            documents.append(
                {
                    "text": text,
                    "source_doc": path.name,
                    "doc_type": folder,
                    "patient_id": path.name.split("_")[0],
                }
            )

    if REPORTS_DIR.exists():
        for path in sorted(REPORTS_DIR.glob("*_report.json")):
            try:
                text = json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2)
            except Exception:
                continue
            documents.append(
                {
                    "text": text,
                    "source_doc": path.name,
                    "doc_type": "validation_report",
                    "patient_id": path.name.split("_")[0],
                }
            )

    return documents


def index_all_documents(force: bool = False) -> dict:
    """
    Build (or rebuild) the FAISS index.

    Returns {chunks, documents, index_path} so the caller can report
    what was indexed.
    """
    import faiss

    if INDEX_PATH.exists() and not force:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        return {
            "chunks": len(metadata),
            "documents": len({m["source_doc"] for m in metadata}),
            "index_path": str(INDEX_PATH),
            "rebuilt": False,
        }

    documents = _collect_source_documents()

    records = []
    for document in documents:
        for position, chunk in enumerate(chunk_document(document["text"])):
            records.append(
                {
                    "chunk": chunk,
                    "source_doc": document["source_doc"],
                    "doc_type": document["doc_type"],
                    "patient_id": document["patient_id"],
                    "position": position,
                }
            )

    if not records:
        raise RuntimeError(
            f"No indexable documents found under {INCOMING_DIR} or {REPORTS_DIR}."
        )

    vectors = embed_texts([r["chunk"] for r in records])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    METADATA_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")

    return {
        "chunks": len(records),
        "documents": len(documents),
        "index_path": str(INDEX_PATH),
        "rebuilt": True,
    }


_index_cache: dict = {"index": None, "metadata": None}


def load_index(patient_filter: Optional[str] = None):
    """
    Load the FAISS index and chunk metadata, building it first if it
    doesn't exist yet. Cached so repeated questions don't re-read from
    disk.
    """
    import faiss

    if _index_cache["index"] is None:
        if not INDEX_PATH.exists():
            index_all_documents()
        _index_cache["index"] = faiss.read_index(str(INDEX_PATH))
        _index_cache["metadata"] = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    return _index_cache["index"], _index_cache["metadata"]


def reset_index_cache() -> None:
    _index_cache["index"] = None
    _index_cache["metadata"] = None
