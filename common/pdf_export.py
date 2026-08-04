"""
common/pdf_export.py

PDF rendering for discharge summaries and audit reports (spec 2.5, which
asks for JSON + HTML/PDF, and Table 13 page 5's export buttons).

Uses fpdf2: pure Python, no system binaries. WeasyPrint would give
better HTML fidelity but needs GTK on Windows, which is a poor trade for
a document that is mostly headings and paragraphs.

fpdf2's core fonts are Latin-1 only. Discharge summaries are generated
in English, but patient names are not necessarily Latin-1 (Lucía
Fernández, Lukas Müller) and a raw UnicodeEncodeError mid-export would
lose the whole document -- so text is transliterated on the way in.
"""

import unicodedata
from datetime import datetime, timezone
from pathlib import Path

REPORTS_DIR = Path("Data/reports")

# Characters fpdf2's core fonts cannot encode, with sensible stand-ins.
_REPLACEMENTS = {
    "—": "-", "–": "-", "“": '"', "”": '"', "‘": "'", "’": "'",
    "…": "...", "•": "-", "₹": "INR ", "€": "EUR ", "£": "GBP ",
}


def latin1_safe(text: str) -> str:
    """
    Make text encodable by fpdf2's core fonts.

    Accented Latin characters are decomposed and stripped to their base
    form (Lucía -> Lucia) rather than dropped, so a patient's name stays
    recognisable. Anything still unencodable becomes "?" rather than
    raising.
    """
    if not text:
        return ""

    for source, target in _REPLACEMENTS.items():
        text = text.replace(source, target)

    try:
        text.encode("latin-1")
        return text
    except UnicodeEncodeError:
        decomposed = unicodedata.normalize("NFKD", text)
        stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
        return stripped.encode("latin-1", errors="replace").decode("latin-1")


def _cell(pdf, height: float, text: str) -> None:
    """
    Write a full-width paragraph.

    fpdf2's multi_cell defaults to new_x=RIGHT, which leaves the cursor at
    the cell's right edge. The next w=0 cell then has zero usable width
    and raises "Not enough horizontal space to render a single character",
    so every write here explicitly returns the cursor to the left margin.
    """
    from fpdf.enums import XPos, YPos

    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(
        0, height, latin1_safe(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )


def _new_pdf(title: str, subtitle: str = ""):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    _cell(pdf, 10, title)

    if subtitle:
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(90, 90, 90)
        _cell(pdf, 6, subtitle)
        pdf.set_text_color(0, 0, 0)

    pdf.ln(2)
    return pdf


def _heading(pdf, text: str) -> None:
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    _cell(pdf, 7, text)
    pdf.set_font("Helvetica", "", 10)


def _body(pdf, text: str) -> None:
    pdf.set_font("Helvetica", "", 10)
    _cell(pdf, 5, text)


SECTION_TITLES = {
    "patient": "Your hospital stay",
    "meds": "Your medicines",
    "labs": "Your test results",
    "bill": "Your bill",
    "instructions": "What to do next",
}


def summary_to_pdf_bytes(patient_id: str, summary: dict) -> bytes:
    """
    Render a patient-friendly discharge summary to PDF bytes.

    Args:
        summary: {"sections": {name: text}, "risk_level": ..., ...}
    """
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pdf = _new_pdf(
        f"Discharge Summary - {patient_id}",
        f"Generated {generated} | Prepared for: {summary.get('audience', 'patient')}",
    )

    sections = summary.get("sections") or {}
    for key, title in SECTION_TITLES.items():
        text = (sections.get(key) or "").strip()
        if not text:
            continue
        _heading(pdf, title)
        _body(pdf, text)

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    _cell(
        pdf, 4,
        "This summary was generated from your validated hospital record. "
        "If anything here differs from the instructions your care team "
        "gave you, follow your care team and contact them with questions.",
    )

    return bytes(pdf.output())


def report_to_pdf_bytes(report: dict) -> bytes:
    """Render the clinician-facing audit report to PDF bytes."""
    patient_id = report.get("patient_id", "unknown")
    pdf = _new_pdf(
        f"Discharge Audit Report - {patient_id}",
        f"Generated {report.get('generated_at', '')}",
    )

    _heading(pdf, "Verdict")
    _body(
        pdf,
        f"Risk level: {str(report.get('risk_level', '')).upper()}\n"
        f"Risk score: {report.get('risk_score')}\n"
        f"Discharge blocked: {'YES' if report.get('discharge_blocked') else 'No'}\n"
        f"Recommendation: {report.get('recommendation', '')}",
    )

    completeness = report.get("completeness") or {}
    _heading(pdf, "Completeness")
    blocking = completeness.get("missing_blocking") or []
    nonblocking = completeness.get("missing_nonblocking") or []
    _body(
        pdf,
        f"Missing blocking fields: {', '.join(blocking) if blocking else 'none'}\n"
        f"Missing non-blocking fields: {', '.join(nonblocking) if nonblocking else 'none'}",
    )

    _heading(pdf, "Cross-validation findings")
    findings = report.get("ehr_findings") or []
    if not findings:
        _body(pdf, "No findings recorded.")
    for finding in findings:
        marker = "TRIGGERED" if finding.get("triggered") else "ok"
        _body(
            pdf,
            f"[{marker}] {finding.get('rule_id')} ({finding.get('severity')}) "
            f"-> {finding.get('action')}\n    {finding.get('evidence', '')}",
        )

    confidence = report.get("translation_confidence")
    _heading(pdf, "Audit trail")
    _body(
        pdf,
        f"Translation confidence: {confidence if confidence is not None else 'n/a'}\n"
        f"Rules version (SHA-256): {report.get('rules_version', '')}",
    )

    return bytes(pdf.output())


def write_report_pdf(report: dict, output_dir: Path = REPORTS_DIR) -> Path:
    """Persist the audit report PDF next to its JSON and HTML siblings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{report.get('patient_id', 'unknown')}_report.pdf"
    path.write_bytes(report_to_pdf_bytes(report))
    return path
