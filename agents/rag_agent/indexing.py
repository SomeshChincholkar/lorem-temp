"""
agents/rag_agent/indexing.py

RAG role 1 of 5 -- the Indexing Agent (spec Table 5).

**One FAISS index per patient.** Layout:

    data/vector_db/
      P1014/
        index.faiss
        chunks.json
      P1015/
        index.faiss
        chunks.json
      ...

Why per-patient rather than one shared index: FAISS has no metadata
filtering, so a shared index can only scope a query to one patient by
over-fetching and discarding everyone else's chunks afterwards. That is
approximate -- if the requested patient's chunks all rank below the
over-fetch window, the query silently returns fewer results than asked
for, or none. Searching a patient-scoped index instead makes the top-k
exact, and makes it structurally impossible for one patient's records to
appear in another patient's answer.

The trade-off, stated plainly: cross-patient questions ("which patient
has an allergy contradiction?") are no longer answerable in one query.
Every question must name a patient. That is enforced in roles.py and in
the RAG agent, not left to the caller to remember.

The embedding model is loaded lazily. Importing this module must stay
cheap, because the A2A server imports it at startup and pulling ~90MB of
model weights on import would make the server look hung.
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np

VECTOR_DB_DIR = Path(os.getenv("VECTOR_DB_DIR", "data/vector_db"))

INCOMING_DIR = Path(os.getenv("INPUT_ROOT_DIR", "Data/incoming"))
REPORTS_DIR = Path("Data/reports")

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

_model = None


class PatientNotIndexedError(LookupError):
    """Raised when a patient has no documents to build an index from."""


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
def patient_dir(patient_id: str) -> Path:
    return VECTOR_DB_DIR / patient_id


def index_path(patient_id: str) -> Path:
    return patient_dir(patient_id) / "index.faiss"


def metadata_path(patient_id: str) -> Path:
    return patient_dir(patient_id) / "chunks.json"


def is_indexed(patient_id: str) -> bool:
    return index_path(patient_id).exists() and metadata_path(patient_id).exists()


def list_indexed_patients() -> list[str]:
    """Every patient with a built index on disk."""
    if not VECTOR_DB_DIR.exists():
        return []
    return sorted(
        p.name for p in VECTOR_DB_DIR.iterdir() if p.is_dir() and is_indexed(p.name)
    )


# ---------------------------------------------------------------------
# Embeddings + chunking
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Source collection
# ---------------------------------------------------------------------
def _extract_text_any_format():
    """
    Import the harvester lazily.

    It pulls in pdfplumber/pytesseract, which are heavy and only needed
    when an index is actually being built. Both sys.path entries are
    required: the project root so "mcp_primary.*" resolves, and
    mcp_primary itself because the harvester imports its siblings flat
    ("from roots import ...") for the MCP server's run-from-that-directory
    layout.
    """
    import sys

    project_root = Path(__file__).resolve().parents[2]
    for path in (project_root, project_root / "mcp_primary"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from mcp_primary.tools_harvester import extract_text_any_format

    return extract_text_any_format


def collect_documents_by_patient() -> dict[str, list[dict]]:
    """
    Gather every indexable document, grouped by patient.

    Includes the raw paperwork plus the validation reports, so questions
    about risk level and findings are answerable from the same index.
    """
    extract_text_any_format = _extract_text_any_format()
    by_patient: dict[str, list[dict]] = {}

    def add(patient_id: str, document: dict) -> None:
        by_patient.setdefault(patient_id, []).append(document)

    for folder in ("doctor_reports", "lab_reports", "bills"):
        directory = INCOMING_DIR / folder
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file() or "_" not in path.name:
                continue
            try:
                text = extract_text_any_format(path)
            except Exception:
                # A single unreadable file must not abort a patient's
                # whole index.
                continue
            patient_id = path.name.split("_")[0]
            add(patient_id, {
                "text": text,
                "source_doc": path.name,
                "doc_type": folder,
                "patient_id": patient_id,
            })

    if REPORTS_DIR.exists():
        for path in sorted(REPORTS_DIR.glob("*_report.json")):
            try:
                text = json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2)
            except Exception:
                continue
            patient_id = path.name.split("_")[0]
            add(patient_id, {
                "text": text,
                "source_doc": path.name,
                "doc_type": "validation_report",
                "patient_id": patient_id,
            })

    return by_patient


# ---------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------
def _build_index_for(patient_id: str, documents: list[dict]) -> dict:
    import faiss

    records = []
    for document in documents:
        for position, chunk in enumerate(chunk_document(document["text"])):
            records.append({
                "chunk": chunk,
                "source_doc": document["source_doc"],
                "doc_type": document["doc_type"],
                "patient_id": patient_id,
                "position": position,
            })

    if not records:
        raise PatientNotIndexedError(f"No indexable content for patient '{patient_id}'.")

    vectors = embed_texts([r["chunk"] for r in records])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    patient_dir(patient_id).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path(patient_id)))
    metadata_path(patient_id).write_text(json.dumps(records, indent=2), encoding="utf-8")

    return {
        "patient_id": patient_id,
        "chunks": len(records),
        "documents": len(documents),
        "index_path": str(index_path(patient_id)),
    }


def index_patient_documents(patient_id: str, force: bool = False) -> dict:
    """
    Build (or rebuild) one patient's index.

    Returns {patient_id, chunks, documents, index_path, rebuilt}.
    """
    if is_indexed(patient_id) and not force:
        records = json.loads(metadata_path(patient_id).read_text(encoding="utf-8"))
        return {
            "patient_id": patient_id,
            "chunks": len(records),
            "documents": len({r["source_doc"] for r in records}),
            "index_path": str(index_path(patient_id)),
            "rebuilt": False,
        }

    documents = collect_documents_by_patient().get(patient_id)
    if not documents:
        raise PatientNotIndexedError(
            f"No documents found for patient '{patient_id}' under {INCOMING_DIR}."
        )

    result = _build_index_for(patient_id, documents)
    reset_index_cache(patient_id)
    return {**result, "rebuilt": True}


def index_all_documents(force: bool = False) -> dict:
    """
    Build every patient's index.

    Returns a per-patient breakdown plus totals. One patient failing does
    not abort the rest -- a corrupt file for P1020 should not stop P1019
    being queryable.
    """
    by_patient = collect_documents_by_patient()
    if not by_patient:
        raise RuntimeError(
            f"No indexable documents found under {INCOMING_DIR} or {REPORTS_DIR}."
        )

    patients: dict[str, dict] = {}
    errors: dict[str, str] = {}

    for patient_id, documents in sorted(by_patient.items()):
        try:
            if is_indexed(patient_id) and not force:
                records = json.loads(metadata_path(patient_id).read_text(encoding="utf-8"))
                patients[patient_id] = {
                    "patient_id": patient_id,
                    "chunks": len(records),
                    "documents": len({r["source_doc"] for r in records}),
                    "index_path": str(index_path(patient_id)),
                    "rebuilt": False,
                }
            else:
                patients[patient_id] = {**_build_index_for(patient_id, documents),
                                        "rebuilt": True}
        except Exception as exc:  # noqa: BLE001
            errors[patient_id] = f"{type(exc).__name__}: {exc}"

    reset_index_cache()

    return {
        "patients": patients,
        "errors": errors,
        "patient_count": len(patients),
        "chunks": sum(p["chunks"] for p in patients.values()),
        "documents": sum(p["documents"] for p in patients.values()),
    }


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------
_index_cache: dict[str, tuple] = {}


def load_patient_index(patient_id: str):
    """
    Load one patient's FAISS index and chunk metadata, building it first
    if it does not exist yet.

    Raises PatientNotIndexedError when the patient has no documents --
    an unknown patient must be a clear error, not an empty result set
    that reads like "nothing found in their records".
    """
    import faiss

    if not patient_id:
        raise PatientNotIndexedError("A patient_id is required to load an index.")

    if patient_id in _index_cache:
        return _index_cache[patient_id]

    if not is_indexed(patient_id):
        index_patient_documents(patient_id)

    index = faiss.read_index(str(index_path(patient_id)))
    metadata = json.loads(metadata_path(patient_id).read_text(encoding="utf-8"))
    _index_cache[patient_id] = (index, metadata)
    return _index_cache[patient_id]


def reset_index_cache(patient_id: str | None = None) -> None:
    """Drop cached indexes so the next load re-reads from disk."""
    if patient_id is None:
        _index_cache.clear()
    else:
        _index_cache.pop(patient_id, None)


def delete_patient_index(patient_id: str) -> bool:
    """Remove one patient's index from disk. Returns True if it existed."""
    directory = patient_dir(patient_id)
    if not directory.exists():
        return False
    shutil.rmtree(directory)
    reset_index_cache(patient_id)
    return True
