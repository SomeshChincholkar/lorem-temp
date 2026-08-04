"""
agents/orchestrator/pipeline.py

The discharge pipeline the Host Orchestrator drives (spec section 8).

Per-patient sequence:

    Monitor(8103)    -> what documents exist
    Extractor(8100)  -> structured fields, once per document type
    Normalizer(8102) -> only when a document isn't already English
    Validator(8101)  -> completeness + EHR cross-validation + report
    guardrail check  -> does this need a human before release?

One trace_id is minted at the top and threaded through every A2A call,
which is what lets LangFuse stitch the whole case into a single trace
later (spec 7.2).
"""

from uuid import uuid4

from agents.common import push_notifications
from agents.common.a2a_client import send_message

# Spec Table 12 row 5 (HITL Escalation). Re-exported here because this
# module is where the pipeline applies it; the implementation lives with
# the other four guardrails.
from guardrails import GuardrailManager, guardrail_manager  # noqa: F401
from observability import flush, log_guardrail_events, observe, set_output, trace_url

DOC_TYPES = ("doctor_reports", "lab_reports", "bills")

# Extractor doc_type -> the Validator payload key it feeds.
# lab_reports was previously extracted and then dropped on the floor,
# which left the audit report (and so the summary's "your test results"
# section) with no lab data at all.
DOC_TYPE_TO_VALIDATOR_KEY = {
    "doctor_reports": "extracted_discharge",
    "lab_reports": "extracted_lab",
    "bills": "extracted_bill",
}


def _unwrap_fields(extracted: dict | None) -> dict:
    """
    The extraction prompt returns {"doc_type":..., "fields": {...}} for a
    single document, or a {"documents": [...]} wrapper for several. The
    Validator wants the flat field dict, so normalize both shapes here
    rather than making every downstream caller handle it.
    """
    if not isinstance(extracted, dict):
        return {}
    if "fields" in extracted and isinstance(extracted["fields"], dict):
        return extracted["fields"]
    if "documents" in extracted and extracted["documents"]:
        first = extracted["documents"][0]
        if isinstance(first, dict) and isinstance(first.get("fields"), dict):
            return first["fields"]
    return extracted


async def run_discharge_pipeline(
    patient_id: str,
    trace_id: str | None = None,
    normalize_non_english: bool = True,
) -> dict:
    """
    Run the whole pipeline for one patient.

    Returns a step-by-step record rather than just the verdict, so the
    Gradio UI and the dashboard can show where a run stopped and why.
    """
    trace_id = trace_id or str(uuid4())

    # Root span for the case. Every downstream agent, tool call, LLM
    # generation and guardrail event attaches to this same trace via the
    # trace_id threaded through each A2A message's payload -- which is
    # what spec 7.2 means by "end-to-end trace ID per discharge case".
    with observe(
        "pipeline.discharge",
        as_type="chain",
        trace_seed=trace_id,
        input={"patient_id": patient_id},
        metadata={"trace_id": trace_id},
    ) as root_span:
        result = await _run_pipeline_steps(patient_id, trace_id, normalize_non_english)
        set_output(root_span, {
            "final_status": result.get("final_status"),
            "risk_level": result.get("risk_level"),
            "requires_hitl": result.get("requires_hitl"),
        })

    result["trace_url"] = trace_url(trace_id)
    flush()
    return result


async def _run_pipeline_steps(
    patient_id: str, trace_id: str, normalize_non_english: bool
) -> dict:
    """The pipeline body. Split out so the root span stays readable."""
    steps: list[dict] = []
    extracted: dict[str, dict] = {}
    languages: dict[str, str] = {}
    confidences: list[float] = []

    def record(name: str, result: dict, detail: str | None = None):
        steps.append(
            {
                "step": name,
                "ok": result.get("ok", False),
                "state": result.get("state"),
                "error": result.get("error"),
                "detail": detail,
            }
        )

    # --- 1. Monitor: what paperwork exists for this patient? ---------
    monitor = await send_message("monitor", {"trigger": True, "trace_id": trace_id})
    record("monitor", monitor)

    available = set(DOC_TYPES)
    if monitor["ok"] and monitor["artifacts"]:
        documents = monitor["artifacts"][0].get("documents", [])
        for_patient = {
            d.get("doc_type") for d in documents if d.get("patient_id") == patient_id
        }
        # An empty watcher result usually means "already processed", not
        # "nothing exists" -- so fall back to trying all three types
        # rather than silently skipping the patient.
        if for_patient:
            available = for_patient

    # --- 2. Extractor: one call per document type -------------------
    for doc_type in DOC_TYPES:
        if doc_type not in available:
            continue

        result = await send_message(
            "extractor",
            {"patient_id": patient_id, "doc_type": doc_type, "trace_id": trace_id},
        )
        record("extract", result, detail=doc_type)

        if result["ok"] and result["artifacts"]:
            artifact = result["artifacts"][0]
            extracted[doc_type] = _unwrap_fields(artifact.get("extracted_fields"))
            languages[doc_type] = artifact.get("language") or "en"

    # --- 3. Normalizer: only for documents that aren't English -------
    if normalize_non_english:
        for doc_type, language in languages.items():
            if language == "en":
                continue

            result = await send_message(
                "normalizer",
                {"patient_id": patient_id, "doc_type": doc_type, "trace_id": trace_id},
            )
            record("normalize", result, detail=f"{doc_type} ({language})")

            if result["ok"] and result["artifacts"]:
                confidence = result["artifacts"][0].get("confidence")
                if isinstance(confidence, (int, float)):
                    confidences.append(float(confidence))

    # The report should reflect the weakest translation in the case, not
    # an average that could hide one bad document.
    translation_confidence = min(confidences) if confidences else None

    # --- 4. Validator: completeness + EHR rules + audit report -------
    validator_payload = {
        "patient_id": patient_id,
        "translation_confidence": translation_confidence,
        "trace_id": trace_id,
    }
    for doc_type, key in DOC_TYPE_TO_VALIDATOR_KEY.items():
        validator_payload[key] = extracted.get(doc_type, {})

    validation = await send_message("validator", validator_payload)
    record("validate", validation)

    report = validation["artifacts"][0] if (validation["ok"] and validation["artifacts"]) else {}

    # --- 5. Retire the paperwork ------------------------------------
    # Only on a completed validation. Marking documents processed after a
    # failed run would hide them from the next scan, silently dropping
    # the case.
    if validation["ok"]:
        try:
            from agents.common.mcp_client import call_tool

            await call_tool(
                "mark_documents_processed_tool",
                {"patient_id": patient_id},
                trace_id=trace_id,
            )
        except Exception as exc:  # noqa: BLE001
            # Non-fatal: worst case the Watcher re-reports these files,
            # which is exactly the behaviour before this call existed.
            record("mark_processed", {"ok": False, "state": "error", "error": str(exc)})

    # --- 6. Guardrail: can this release without a human? ------------
    guardrails = GuardrailManager(trace_id=trace_id)
    guardrail = guardrails.check_escalation(
        report.get("risk_level"), report.get("discharge_blocked")
    )
    log_guardrail_events(guardrails, trace_seed=trace_id)

    # --- 7. Push notification (spec §9) -----------------------------
    # Fires only when configured. The cases worth waking someone for are
    # exactly the ones a human must act on, so a clean auto-approve is
    # notified as informational and a blocked case as requiring review.
    if push_notifications.is_enabled():
        push_result = await push_notifications.send_push_notification(
            "discharge.reviewed",
            patient_id,
            trace_id=trace_id,
            final_status=report.get("final_status"),
            risk_level=report.get("risk_level"),
            requires_hitl=guardrail["requires_hitl"],
            discharge_blocked=report.get("discharge_blocked"),
            report_path=report.get("json_path"),
        )
        record(
            "push_notification",
            {"ok": push_result["sent"], "state": push_result["status"],
             "error": push_result["error"]},
        )

    return {
        "patient_id": patient_id,
        "trace_id": trace_id,
        "steps": steps,
        "languages": languages,
        "translation_confidence": translation_confidence,
        "final_status": report.get("final_status"),
        "risk_level": report.get("risk_level"),
        "recommendation": report.get("recommendation"),
        "discharge_blocked": report.get("discharge_blocked"),
        "requires_hitl": guardrail["requires_hitl"],
        "ehr_findings": report.get("ehr_findings", []),
        "json_path": report.get("json_path"),
        "html_path": report.get("html_path"),
        "ok": validation["ok"],
    }
