"""
agents/rag_agent/build_index.py

Builds the per-patient FAISS indexes ahead of time, so the first Q&A
request doesn't pay for both the model download and the embedding pass.

Usage:
    python -m agents.rag_agent.build_index                # all patients
    python -m agents.rag_agent.build_index --force        # rebuild all
    python -m agents.rag_agent.build_index P1019          # one patient
    python -m agents.rag_agent.build_index P1019 --force
    python -m agents.rag_agent.build_index --list         # show what exists
"""

import sys

from .indexing import (
    PatientNotIndexedError,
    index_all_documents,
    index_patient_documents,
    list_indexed_patients,
    reset_index_cache,
)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv

    if "--list" in sys.argv:
        indexed = list_indexed_patients()
        if not indexed:
            print("No patient indexes built yet.")
            return
        print(f"{len(indexed)} patient index(es):")
        for patient_id in indexed:
            print(f"  {patient_id}")
        return

    reset_index_cache()

    if args:
        patient_id = args[0]
        print(f"Building index for {patient_id}...")
        try:
            result = index_patient_documents(patient_id, force=force)
        except PatientNotIndexedError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        print(f"  documents : {result['documents']}")
        print(f"  chunks    : {result['chunks']}")
        print(f"  path      : {result['index_path']}")
        print("  rebuilt" if result["rebuilt"] else "  already present (use --force)")
        return

    print("Building per-patient indexes (first run downloads the embedding model)...")
    result = index_all_documents(force=force)

    for patient_id, info in sorted(result["patients"].items()):
        state = "rebuilt" if info["rebuilt"] else "cached "
        print(
            f"  {patient_id}  {state}  "
            f"{info['documents']} doc(s), {info['chunks']} chunk(s)"
        )

    print()
    print(f"patients indexed : {result['patient_count']}")
    print(f"documents        : {result['documents']}")
    print(f"chunks           : {result['chunks']}")

    if result["errors"]:
        print()
        print("Errors (these patients were skipped):")
        for patient_id, error in result["errors"].items():
            print(f"  {patient_id}: {error}")


if __name__ == "__main__":
    main()
