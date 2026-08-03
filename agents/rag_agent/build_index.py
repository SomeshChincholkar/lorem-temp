"""
agents/rag_agent/build_index.py

Builds the FAISS index ahead of time, so the first Q&A request doesn't
pay for both the model download and the full corpus embedding.

Usage:
    python -m agents.rag_agent.build_index
    python -m agents.rag_agent.build_index --force    # rebuild from scratch
"""

import sys

from .indexing import index_all_documents, reset_index_cache


def main():
    force = "--force" in sys.argv
    reset_index_cache()

    print("Building FAISS index (first run downloads the embedding model)...")
    result = index_all_documents(force=force)

    print(f"documents indexed : {result['documents']}")
    print(f"chunks written    : {result['chunks']}")
    print(f"index path        : {result['index_path']}")
    print("rebuilt" if result["rebuilt"] else "already present (use --force to rebuild)")


if __name__ == "__main__":
    main()
