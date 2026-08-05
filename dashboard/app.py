"""
dashboard/app.py

Streamlit HITL Dashboard (spec section 8, port 8501).

Five pages:
    1. Document Viewer     -- inspect incoming paperwork, trigger the pipeline
    2. Validation Report   -- completeness, cross-validation, risk verdict
    3. HITL Corrections    -- elicitation form, medication edits, approval
    4. RAG Q&A             -- streaming grounded Q&A with injection screening
    5. Discharge Summary   -- streamed patient summary, JSON/HTML/PDF export

Run from the PROJECT ROOT so the `agents`, `guardrails` and `common`
packages resolve:

    python -m streamlit run dashboard/app.py --server.port 8501

Which services each page needs:
    page 1 trigger  -> Monitor/Extractor/Normalizer/Validator + MCP + EHR
    page 3 re-run   -> Validation Agent :8101
    page 4          -> RAG Q&A Agent :8105
    page 5          -> Summary Generator :8104

Pages 2 and 5's report views read Data/reports/ directly, so they work
with every agent stopped -- useful for reviewing a completed case.
"""

import sys
from pathlib import Path

import streamlit as st

# Streamlit executes this file directly, so the project root is not on
# sys.path the way it is under `python -m`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Discharge HITL Dashboard",
    page_icon="🏥",
    layout="wide",
)

PAGES_DIR = Path(__file__).resolve().parent / "pages"

pages = [
    st.Page(PAGES_DIR / "page_1_document_viewer.py", title="Document Viewer", icon="📄"),
    st.Page(PAGES_DIR / "page_2_validation_report.py", title="Validation Report", icon="✅"),
    st.Page(PAGES_DIR / "page_3_hitl_corrections.py", title="HITL Corrections", icon="✏️"),
    st.Page(PAGES_DIR / "page_4_rag_qa.py", title="RAG Q&A", icon="💬"),
    st.Page(PAGES_DIR / "page_5_discharge_summary.py", title="Discharge Summary", icon="📋"),
]

st.sidebar.title("Discharge Review")
st.sidebar.caption("Human-in-the-loop dashboard")

navigation = st.navigation(pages)

# Pending elicitations block the Validation Agent, so surface the count
# on every page rather than only on the page that resolves them.
try:
    from agents.common import elicitation_store

    pending_count = len(elicitation_store.list_pending())
    if pending_count:
        st.sidebar.error(
            f"{pending_count} elicitation request(s) waiting — "
            "the Validation Agent is blocked until they are answered."
        )
except Exception:
    pass

navigation.run()
