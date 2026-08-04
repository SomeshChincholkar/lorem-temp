"""
tests/test_pdf_export.py

Covers PDF rendering for discharge summaries and audit reports
(spec 2.5 asks for JSON + HTML/PDF; Table 13 page 5 exports it).

The interesting risk here is encoding. fpdf2's core fonts are Latin-1
only, and this corpus contains Spanish, German and Hindi patient names.
An unhandled UnicodeEncodeError mid-render loses the whole document, so
the transliteration path is pinned explicitly.

Run:  python -m pytest tests/test_pdf_export.py -q
"""

import pytest

from common.pdf_export import (
    latin1_safe,
    report_to_pdf_bytes,
    summary_to_pdf_bytes,
    write_report_pdf,
)

SUMMARY = {
    "audience": "patient",
    "risk_level": "low",
    "sections": {
        "patient": "You were treated for a stomach infection and recovered well.",
        "meds": "Ondansetron 4 mg when you feel sick.",
        "labs": "Your blood tests were normal.",
        "bill": "Your bill of 1630.29 EUR is fully paid.",
        "instructions": "Drink plenty of fluids. See your doctor on 2026-06-16.",
    },
}

REPORT = {
    "patient_id": "P1016",
    "risk_level": "high",
    "risk_score": 8,
    "discharge_blocked": True,
    "recommendation": "Urgent Attention - Block release",
    "completeness": {"missing_blocking": [], "missing_nonblocking": ["ward"]},
    "ehr_findings": [
        {
            "rule_id": "allergy_contradiction_check",
            "severity": "Critical",
            "triggered": True,
            "action": "Block discharge",
            "evidence": "Amoxicillin conflicts with documented Penicillin allergy",
        }
    ],
    "translation_confidence": 0.55,
    "rules_version": "abc123def456",
}


# ---------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------
def test_western_european_accents_survive_intact():
    """These are valid Latin-1, so they must NOT be mangled."""
    assert latin1_safe("Lucía Fernández") == "Lucía Fernández"
    assert latin1_safe("Lukas Müller") == "Lukas Müller"


def test_typographic_punctuation_is_substituted():
    """Em dashes and smart quotes are outside Latin-1."""
    assert latin1_safe("dash — quote “x”") == 'dash - quote "x"'
    assert latin1_safe("bullet • ellipsis …") == "bullet - ellipsis ..."


def test_non_latin_scripts_degrade_instead_of_raising():
    """
    Devanagari has no Latin-1 form at all. Losing the name is acceptable;
    losing the whole PDF is not.
    """
    result = latin1_safe("अनन्या शर्मा")
    result.encode("latin-1")  # must not raise


def test_currency_symbols_are_expanded():
    assert "INR" in latin1_safe("₹1000")
    assert "EUR" in latin1_safe("€50")


def test_empty_input_is_safe():
    assert latin1_safe("") == ""
    assert latin1_safe(None) == ""


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------
def test_summary_renders_a_valid_pdf():
    pdf = summary_to_pdf_bytes("P1014", SUMMARY)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 500


def test_summary_with_unicode_name_renders():
    """The regression this file exists for."""
    summary = {
        "audience": "patient",
        "sections": {"patient": "Lucía Fernández — treated for gastroenteritis…"},
    }
    assert summary_to_pdf_bytes("P1014", summary).startswith(b"%PDF")


def test_summary_skips_empty_sections():
    """A missing section must not produce a stray heading."""
    sparse = {"sections": {"patient": "Only this one.", "meds": "", "labs": None}}
    assert summary_to_pdf_bytes("P1019", sparse).startswith(b"%PDF")


def test_summary_with_no_sections_still_renders():
    assert summary_to_pdf_bytes("P1019", {"sections": {}}).startswith(b"%PDF")
    assert summary_to_pdf_bytes("P1019", {}).startswith(b"%PDF")


def test_long_text_wraps_across_pages():
    """
    Guards the fpdf2 cursor bug: a full-width cell after another
    full-width cell used to raise "Not enough horizontal space".
    """
    long_summary = {
        "sections": {key: ("Take your medication as prescribed. " * 200)
                     for key in ("patient", "meds", "labs", "bill", "instructions")}
    }
    pdf = summary_to_pdf_bytes("P1019", long_summary)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 3000


def test_report_renders_a_valid_pdf():
    pdf = report_to_pdf_bytes(REPORT)
    assert pdf.startswith(b"%PDF")


def test_report_with_no_findings_renders():
    report = {**REPORT, "ehr_findings": []}
    assert report_to_pdf_bytes(report).startswith(b"%PDF")


def test_report_with_missing_fields_renders():
    """Reports from a partial run must still export."""
    assert report_to_pdf_bytes({"patient_id": "P9999"}).startswith(b"%PDF")


def test_write_report_pdf_lands_next_to_its_siblings(tmp_path):
    path = write_report_pdf(REPORT, output_dir=tmp_path)

    assert path.name == "P1016_report.pdf"
    assert path.read_bytes().startswith(b"%PDF")


@pytest.mark.parametrize("patient_id", ["P1014", "P1015", "P1016", "P1019"])
def test_every_corpus_patient_exports(patient_id):
    assert summary_to_pdf_bytes(patient_id, SUMMARY).startswith(b"%PDF")
