"""
tests/test_per_patient_index.py

Covers the per-patient FAISS index layout and the mandatory patient scope.

The property under test is a safety property, not a performance one: it
must be structurally impossible for one patient's chunks to appear in
another patient's answer. So these tests build real FAISS indexes over
real (tiny) corpora and assert on what comes back -- only the embedding
model is faked, so the suite stays fast and offline.

Run:  python -m pytest tests/test_per_patient_index.py -q
"""

import hashlib

import numpy as np
import pytest

from agents.rag_agent import indexing as indexing_module
from agents.rag_agent import roles as roles_module
from agents.rag_agent.indexing import (
    PatientNotIndexedError,
    delete_patient_index,
    index_all_documents,
    index_patient_documents,
    is_indexed,
    list_indexed_patients,
    load_patient_index,
    reset_index_cache,
)
from agents.rag_agent.roles import retrieve_top_k

DIMENSIONS = 32


def fake_embed(texts: list[str]) -> np.ndarray:
    """
    Deterministic stand-in for sentence-transformers.

    Hashes each text into a fixed vector and L2-normalizes, matching the
    real embed_texts contract (normalized, so inner product == cosine).
    Identical text always embeds identically, which is what lets the
    retrieval assertions below be exact.
    """
    vectors = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = np.frombuffer((digest * 4)[: DIMENSIONS * 4], dtype="uint8")[:DIMENSIONS]
        vectors.append(raw.astype("float32"))
    matrix = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


CORPUS = {
    "P1019": [
        {
            "text": "Thomas Wright discharged on Metformin 500 mg twice daily and Lisinopril.",
            "source_doc": "P1019_thomas_wright.txt",
            "doc_type": "doctor_reports",
            "patient_id": "P1019",
        }
    ],
    "P1016": [
        {
            "text": "Lukas Mueller has a documented Penicillin allergy but was prescribed Amoxicillin.",
            "source_doc": "P1016_lukas_mueller.json",
            "doc_type": "doctor_reports",
            "patient_id": "P1016",
        }
    ],
    "P1015": [
        {
            "text": "Ananya Sharma treated for typhoid with Azithromycin 500 mg once daily.",
            "source_doc": "P1015_ananya_sharma.txt",
            "doc_type": "doctor_reports",
            "patient_id": "P1015",
        }
    ],
}


@pytest.fixture
def indexed_corpus(tmp_path, monkeypatch):
    """Builds real per-patient FAISS indexes in a temp dir."""
    monkeypatch.setattr(indexing_module, "VECTOR_DB_DIR", tmp_path / "vector_db")
    monkeypatch.setattr(indexing_module, "embed_texts", fake_embed)
    monkeypatch.setattr(roles_module, "embed_texts", fake_embed)
    monkeypatch.setattr(
        indexing_module, "collect_documents_by_patient", lambda: dict(CORPUS)
    )
    reset_index_cache()
    result = index_all_documents(force=True)
    yield result
    reset_index_cache()


# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------
def test_one_index_directory_per_patient(indexed_corpus, tmp_path):
    for patient_id in CORPUS:
        directory = tmp_path / "vector_db" / patient_id
        assert (directory / "index.faiss").exists()
        assert (directory / "chunks.json").exists()


def test_all_patients_are_indexed(indexed_corpus):
    assert sorted(list_indexed_patients()) == sorted(CORPUS)
    assert indexed_corpus["patient_count"] == 3


def test_each_index_holds_only_its_own_patient(indexed_corpus):
    """The core safety property."""
    for patient_id in CORPUS:
        _, metadata = load_patient_index(patient_id)
        assert {r["patient_id"] for r in metadata} == {patient_id}


# ---------------------------------------------------------------------
# Retrieval scope
# ---------------------------------------------------------------------
def test_retrieval_never_crosses_patients(indexed_corpus):
    """
    The allergy question is the case that used to surface P1016 from a
    shared index. Asked about P1019 it must now return only P1019.
    """
    question = "Which medication conflicts with the penicillin allergy?"

    for patient_id in CORPUS:
        results = retrieve_top_k(question, patient_id, k=5)
        assert results, f"no chunks returned for {patient_id}"
        assert {r["patient_id"] for r in results} == {patient_id}


def test_retrieval_returns_exactly_k_when_available(indexed_corpus):
    """
    No over-fetch heuristic any more: k is exact, capped only by how many
    chunks the patient actually has.
    """
    _, metadata = load_patient_index("P1019")
    available = len(metadata)

    results = retrieve_top_k("medications", "P1019", k=available)
    assert len(results) == available


def test_retrieval_caps_at_available_chunks(indexed_corpus):
    results = retrieve_top_k("medications", "P1019", k=999)
    _, metadata = load_patient_index("P1019")
    assert len(results) == len(metadata)


def test_results_carry_scores_and_provenance(indexed_corpus):
    result = retrieve_top_k("medications", "P1019", k=1)[0]
    assert "score" in result
    assert result["source_doc"] == "P1019_thomas_wright.txt"
    assert result["patient_id"] == "P1019"


# ---------------------------------------------------------------------
# Mandatory patient scope
# ---------------------------------------------------------------------
def test_missing_patient_id_raises(indexed_corpus):
    """
    Must raise, not return []. An empty list downstream reads as
    "nothing in their records" and yields a confident refusal for a
    question that was never actually searched.
    """
    with pytest.raises(PatientNotIndexedError):
        retrieve_top_k("anything", "", k=3)

    with pytest.raises(PatientNotIndexedError):
        retrieve_top_k("anything", None, k=3)


def test_unknown_patient_raises(indexed_corpus):
    with pytest.raises(PatientNotIndexedError):
        retrieve_top_k("anything", "P9999", k=3)


def test_indexing_an_unknown_patient_raises(indexed_corpus):
    with pytest.raises(PatientNotIndexedError):
        index_patient_documents("P9999")


# ---------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------
def test_reindex_is_skipped_without_force(indexed_corpus):
    result = index_patient_documents("P1019")
    assert result["rebuilt"] is False


def test_force_rebuilds(indexed_corpus):
    result = index_patient_documents("P1019", force=True)
    assert result["rebuilt"] is True
    assert result["chunks"] > 0


def test_one_patient_rebuild_leaves_others_alone(indexed_corpus):
    before = load_patient_index("P1016")[1]
    index_patient_documents("P1019", force=True)
    after = load_patient_index("P1016")[1]

    assert before == after


def test_delete_removes_only_that_patient(indexed_corpus):
    assert delete_patient_index("P1019") is True

    assert is_indexed("P1019") is False
    assert is_indexed("P1016") is True
    assert sorted(list_indexed_patients()) == ["P1015", "P1016"]


def test_delete_is_safe_on_a_missing_patient(indexed_corpus):
    assert delete_patient_index("P9999") is False


def test_index_rebuilds_on_demand_after_deletion(indexed_corpus):
    delete_patient_index("P1019")
    assert is_indexed("P1019") is False

    # load_patient_index builds it back rather than failing.
    _, metadata = load_patient_index("P1019")
    assert {r["patient_id"] for r in metadata} == {"P1019"}


def test_one_bad_patient_does_not_abort_the_rest(tmp_path, monkeypatch):
    """A patient with no indexable content must not stop the others."""
    corpus = {**CORPUS, "P0000": []}

    monkeypatch.setattr(indexing_module, "VECTOR_DB_DIR", tmp_path / "vector_db")
    monkeypatch.setattr(indexing_module, "embed_texts", fake_embed)
    monkeypatch.setattr(indexing_module, "collect_documents_by_patient", lambda: corpus)
    reset_index_cache()

    result = index_all_documents(force=True)

    assert set(result["patients"]) == set(CORPUS)
    assert "P0000" in result["errors"]
