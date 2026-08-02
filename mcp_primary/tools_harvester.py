"""
tools_harvester.py

The Clinical Data Harvester primitive. Extracts plain text from any of
the document formats that show up in Data/incoming/: .txt, .json, .pdf,
.png (via OCR), and .png.ocr.txt (an already-OCR'd sidecar file).

extract_text_any_format() is also the function resources.py (step 3.2)
should eventually import instead of its temporary local stub -- see
the note at the bottom of this file.
"""

import json
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import Context  # noqa: E402

from roots import resolve_authorized_root, safe_join  # noqa: E402

DOC_TYPE_TO_FOLDER = {
    "bill": "bills",
    "doctor_report": "doctor_reports",
    "lab_report": "lab_reports",
}


# ---------------------------------------------------------------------
# Format-specific readers
# ---------------------------------------------------------------------
def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> str:
    """Pretty-print JSON so the LLM gets clean, readable structure."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return json.dumps(data, indent=2)


def _read_pdf(path: Path) -> str:
    """Extract text from every page of a PDF via pdfplumber."""
    import pdfplumber

    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts).strip()


def _read_png(path: Path) -> str:
    """OCR a scanned/photographed document image."""
    import pytesseract
    from PIL import Image

    # pytesseract is just a Python wrapper -- it needs the actual
    # Tesseract OCR engine binary installed separately on the machine.
    # On Windows it's usually NOT on PATH by default, so point at it
    # explicitly. Adjust this path if you installed it elsewhere, or
    # comment this line out once tesseract.exe is confirmed on PATH.
    if sys.platform == "win32":
        default_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if Path(default_win_path).exists():
            pytesseract.pytesseract.tesseract_cmd = default_win_path

    image = Image.open(path)
    return pytesseract.image_to_string(image).strip()


def _read_png_ocr_txt(path: Path) -> str:
    """Already-OCR'd sidecar text file -- just read it directly."""
    return _read_txt(path)


# ---------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------
def extract_text_any_format(path: Path) -> str:
    """
    Dispatch to the right reader based on filename/extension.

    Checked in this order because ".png.ocr.txt" ends in ".txt" but
    must be treated as a pre-OCR'd sidecar, not a generic text file
    (doesn't change behavior here since both just read the file, but
    keeps the format explicitly recognized/labeled for callers that
    branch on it).
    """
    name = path.name.lower()

    if name.endswith(".png.ocr.txt"):
        return _read_png_ocr_txt(path)

    ext = path.suffix.lower()
    if ext == ".txt":
        return _read_txt(path)
    if ext == ".json":
        return _read_json(path)
    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".png":
        return _read_png(path)

    raise ValueError(f"Unsupported format: {ext} (file: {path.name})")


def find_file_for_patient(folder: Path, patient_id: str) -> Path:
    """
    Glob {patient_id}_* inside folder. If multiple files exist for the
    same patient, prefer structured/pre-processed formats over ones
    that still need OCR: .json > .txt > .png.ocr.txt > .pdf > .png
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    matches = [f for f in folder.iterdir() if f.is_file() and f.name.startswith(f"{patient_id}_")]
    if not matches:
        raise FileNotFoundError(f"No file found for patient '{patient_id}' in {folder}")

    def priority(p: Path) -> int:
        name = p.name.lower()
        if name.endswith(".json"):
            return 0
        if name.endswith(".png.ocr.txt"):
            return 2
        if name.endswith(".txt"):
            return 1
        if name.endswith(".pdf"):
            return 3
        if name.endswith(".png"):
            return 4
        return 5

    return sorted(matches, key=priority)[0]


# ---------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------
async def clinical_data_harvester_tool(ctx: Context, patient_id: str, doc_type: str) -> Dict:
    """
    Extract raw text for a given patient's document.

    Args:
        ctx: MCP Context (used to resolve the authorized root).
        patient_id: e.g. "P1019"
        doc_type: "bill" | "doctor_report" | "lab_report"

    Returns:
        {patient_id, doc_type, raw_text, format}
    """
    if doc_type not in DOC_TYPE_TO_FOLDER:
        raise ValueError(
            f"Invalid doc_type '{doc_type}'. Must be one of {list(DOC_TYPE_TO_FOLDER)}"
        )

    root = await resolve_authorized_root(ctx)
    folder = safe_join(root, DOC_TYPE_TO_FOLDER[doc_type])
    path = find_file_for_patient(folder, patient_id)
    text = extract_text_any_format(path)

    return {
        "patient_id": patient_id,
        "doc_type": doc_type,
        "raw_text": text,
        "format": path.suffix.lower() if not path.name.lower().endswith(".png.ocr.txt") else ".png.ocr.txt",
    }


# ---------------------------------------------------------------------
# NOTE for step 3.2 cleanup:
# resources.py currently has a temporary _extract_text_minimal() that
# only handles .txt/.json. Once this file is in place, update
# resources.py to import and use extract_text_any_format from here
# instead, and delete the local stub + find_file_for_patient duplicate.
# Not done automatically in this step since resources.py has its own
# async ctx-based file resolution; see next message for the exact diff.
# ---------------------------------------------------------------------