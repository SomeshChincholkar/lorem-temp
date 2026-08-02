# Build Guide — Agentic AI Discharge Summary System
### Server-by-server, function-by-function implementation plan

This guide turns the capstone spec into an ordered build plan. Every server has:
**what it is**, **why it's built at that point**, **every function it needs** (name,
input, output, what it calls), and **pseudocode**. Read top to bottom — each layer
is a dependency for the next one.

---

## 0. Build Order (why this order)

```
1. Mock EHR System (FastAPI :8050)         <- pure data, no dependencies
2. Shared config/rules loader              <- wraps rules.yaml, used everywhere
3. Primary MCP Clinical Tools Server :8200 <- wraps EHR + rules + filesystem
4. Secondary MCP Analytics Server :8201    <- wraps EHR + risk math
5. LangGraph agents (8100, 8101, 8102)     <- call MCP servers as clients
6. Google ADK agents (8103, 8104)          <- call MCP + LangGraph agents (via A2A)
7. Agno RAG agent (8105)                   <- needs FAISS index of docs
8. Host Orchestrator (8083)                <- A2A client to everything above
9. Streamlit HITL Dashboard (8501)         <- calls Orchestrator + agents
10. RAI Guardrails + LangFuse              <- wired in as decorators/middleware
    across every agent/tool from step 3 onward
```

Reason: nothing above step 1 can be tested without patient data. Nothing above
step 3 can be tested without the rules/EHR being queryable. Agents are useless
without the MCP servers they call. The dashboard is useless without agents.
Build bottom-up, test each layer with a script/CLI before wiring the next.

---

## 1. Mock EHR System — FastAPI, port 8050

**Purpose:** Turn `mock_ehr/data.py` (static dicts: `PATIENTS`, `ALLERGIES`,
`MED_ORDERS`, `LABS`, `CARE_PLANS`, `GUIDELINES`) into a REST API so every other
service treats "the EHR" as a network dependency, not an import. This is what
`EHR Validation Tool` and the Analytics server call.

### File layout
```
mock_ehr/
  data.py        <- already have this (uploaded)
  main.py        <- FastAPI app
  schemas.py     <- Pydantic response models
```

### Functions

**`schemas.py`**
- `Patient(BaseModel)`, `Allergy(BaseModel)`, `MedOrder(BaseModel)`,
  `LabResult(BaseModel)`, `CarePlan(BaseModel)`, `Guideline(BaseModel)` — mirror
  the dict shapes in `data.py` field-for-field. Pure typing, no logic.

**`main.py`**

| Function | Route | Input | Output | Calls |
|---|---|---|---|---|
| `get_patient` | `GET /patients/{patient_id}` | `patient_id: str` | `Patient` or 404 | `PATIENTS.get()` |
| `get_allergies` | `GET /patients/{patient_id}/allergies` | `patient_id: str` | `list[str]` | `ALLERGIES.get()` |
| `get_med_orders` | `GET /patients/{patient_id}/med-orders` | `patient_id: str` | `list[MedOrder]` | `MED_ORDERS.get()` |
| `get_labs` | `GET /patients/{patient_id}/labs` | `patient_id: str` | `list[LabResult]` | `LABS.get()` |
| `get_care_plan` | `GET /patients/{patient_id}/care-plan` | `patient_id: str` | `CarePlan` or 404 | `CARE_PLANS.get()` |
| `get_guideline` | `GET /guidelines/{icd10_code}` | `icd10_code: str` | `Guideline` or 404 | `GUIDELINES.get()` |
| `get_full_ehr_bundle` | `GET /patients/{patient_id}/bundle` | `patient_id: str` | combined JSON of all 5 above | calls all functions above internally |

`get_full_ehr_bundle` is the one the EHR Validation Tool will actually use most
— one round trip instead of five.

```python
# pseudocode
@app.get("/patients/{patient_id}/bundle")
def get_full_ehr_bundle(patient_id: str):
    if patient_id not in PATIENTS:
        raise HTTPException(404, "patient not found")
    return {
        "patient": PATIENTS[patient_id],
        "allergies": ALLERGIES.get(patient_id, []),
        "med_orders": MED_ORDERS.get(patient_id, []),
        "labs": LABS.get(patient_id, []),
        "care_plan": CARE_PLANS.get(patient_id, {}),
        "guidelines": [
            GUIDELINES[code] for code in PATIENTS[patient_id]["primary_dx"]
            if code in GUIDELINES
        ],
    }
```

Run with `uvicorn mock_ehr.main:app --port 8050`. Test with `curl` before
touching MCP.

---

## 2. Shared Config / Rules Loader

**Purpose:** `rules.yaml` is loaded by three different agents (Completeness,
EHR Validation, Reporting) and must be hashed for `rules_version` stamping.
Build this once as a tiny library, not copy-pasted per agent.

**File:** `common/rules_loader.py`

| Function | Input | Output | Calls |
|---|---|---|---|
| `load_rules(path="configs/rules.yaml")` | file path | parsed dict (cached) | `yaml.safe_load` |
| `get_rules_sha256(path=...)` | file path | hex digest string | `hashlib.sha256` |
| `get_mandatory_fields(doc_type: str)` | `"clinical" \| "prescription"` | `list[str]` | reads `mandatory_clinical_fields` / `mandatory_prescription_fields` |
| `get_weight(risk_key: str)` | e.g. `"allergy_contradiction"` | `int` weight | reads `risk_scoring_matrix.weights` |
| `get_risk_tier(score: int)` | `int` | `"low" \| "medium" \| "high"` | reads `risk_scoring_matrix.thresholds` |
| `is_hard_guardrail(rule_id: str)` | rule id string | `bool` | membership check in `hitl_hard_guardrails` |
| `expand_abbreviation(token: str)` | e.g. `"HTN"` | `"Hypertension"` or original token | reads `normalization_standards.abbreviation_map` |
| `get_icd10(diagnosis_text: str)` | normalized diagnosis string | ICD-10 code or `None` | reads `normalization_standards.icd10_map` |

```python
# pseudocode
_cache = {}

def load_rules(path="configs/rules.yaml"):
    if path not in _cache:
        with open(path) as f:
            _cache[path] = yaml.safe_load(f)
    return _cache[path]

def get_risk_tier(score: int) -> str:
    rules = load_rules()
    t = rules["risk_scoring_matrix"]["thresholds"]
    if score <= t["low_max"]:
        return "low"
    if score <= t["medium_max"]:
        return "medium"
    return "high"
```

This module is imported directly (not over the network) by MCP server tools —
it's a library, not a service.

---

## 3. Primary MCP Clinical Tools Server — port 8200, path `/clinicaltools`

**Purpose:** One `FastMCP` server exposing all 6 primitives. Build it as a
single `server.py` with clearly separated sections. This is the biggest piece
of the system — build and test each primitive block independently with the
MCP Inspector before wiring agents to it.

**File layout**
```
mcp_primary/
  server.py          <- FastMCP app, wires everything below
  roots.py
  resources.py
  prompts.py
  tools_watcher.py
  tools_harvester.py
  tools_lang_bridge.py
  tools_rules_engine.py
  tools_ehr_validator.py
  tools_reporter.py
```

### 3.1 Roots — `roots.py` + `tools_watcher.py`

**Purpose:** Discharge Monitor Agent (ADK) tells the server which folder is
authorized (`Data/incoming/`). The Watcher Tool must call `ctx.list_roots()`
instead of accepting a raw path parameter — this is the mandatory pattern.

| Function | Input | Output | Calls |
|---|---|---|---|
| `resolve_authorized_root(ctx)` | MCP `Context` | validated `pathlib.Path` | `ctx.list_roots()` |
| `safe_join(root: Path, relative: str)` | root path, relative sub-path | absolute `Path` inside root | `Path.relative_to()` for traversal check |
| `clinical_watcher_tool(ctx, subfolder: str = "")` (the `@mcp.tool()`) | `ctx`, optional subfolder (`"bills"`, `"doctor_reports"`, `"lab_reports"`) | `list[dict]` of `{patient_id, filename, doc_type, path}` for new/unprocessed files | `resolve_authorized_root`, `safe_join`, filesystem `scandir` |

```python
# pseudocode
def resolve_authorized_root(ctx):
    roots = ctx.list_roots()               # returns list[Root(uri=...)]
    if not roots:
        raise PermissionError("no root registered")
    root_path = Path(roots[0].uri.replace("file://", ""))
    if not root_path.exists():
        raise FileNotFoundError(root_path)
    return root_path

def safe_join(root, relative):
    candidate = (root / relative).resolve()
    candidate.relative_to(root)             # raises ValueError if outside root
    return candidate

@mcp.tool()
def clinical_watcher_tool(ctx: Context, subfolder: str = ""):
    root = resolve_authorized_root(ctx)
    scan_dir = safe_join(root, subfolder) if subfolder else root
    found = []
    for sub in ["bills", "doctor_reports", "lab_reports"]:
        folder = safe_join(root, sub)
        for f in folder.iterdir():
            if f.is_file() and not already_processed(f):
                pid = f.name.split("_")[0]        # P1019_bill.json -> P1019
                found.append({
                    "patient_id": pid,
                    "filename": f.name,
                    "doc_type": sub,
                    "path": str(f),
                })
    return found
```

`already_processed` = check a small state file/DB (e.g. `data/processed.json`)
so re-scans don't re-emit the same file. Filename parsing rule, based on your
folder spec:
- `bills/`: `{patient_id}_bill.{ext}`
- `doctor_reports/`: `{patient_id}_{PatientName}.{ext}` (ext can be
  `.txt/.json/.pdf/.png/.png.ocr.txt`)
- `lab_reports/`: `{patient_id}_labs.{ext}`

`patient_id` is always the first `_`-delimited token — extract with
`filename.split("_")[0]`.

### 3.2 Resources — `resources.py`

**Purpose:** Serve rules and raw document text as MCP `Resources` so agents
fetch config/content declaratively instead of via ad hoc tool calls.

| Resource URI | Function | Input | Output | Calls |
|---|---|---|---|---|
| `resource://clinical-rules/completeness` | `get_completeness_rules()` | none | text (YAML/JSON dump of `mandatory_*_fields`) | `rules_loader.load_rules()` |
| `resource://clinical-rules/cross-validation` | `get_cross_validation_rules()` | none | text dump of Table 4 rule set (`clinical_validation_policies` + rule IDs) | `rules_loader.load_rules()` |
| `resource://discharge-report/{patient_id}` | `get_discharge_report(patient_id)` | `patient_id` from URI template | raw extracted text of that patient's doctor report | reads cached extraction output (produced by Harvester Tool) or raw file via `safe_join` |
| `resource://lab-report/{patient_id}` | `get_lab_report(patient_id)` | `patient_id` | raw lab report text | same as above, `lab_reports/` folder |
| `resource://report-template/html` | `get_html_template()` | none | HTML template string with `{{placeholders}}` | reads static file `templates/discharge_summary.html` |
| `resource://medical-abbreviations` | `get_abbreviation_dict()` | none | JSON dump of `normalization_standards.abbreviation_map` | `rules_loader.load_rules()` |

```python
# pseudocode
@mcp.resource("resource://clinical-rules/completeness")
def get_completeness_rules() -> str:
    rules = load_rules()
    return json.dumps({
        "mandatory_clinical_fields": rules["mandatory_clinical_fields"],
        "mandatory_prescription_fields": rules["mandatory_prescription_fields"],
    })

@mcp.resource("resource://discharge-report/{patient_id}")
def get_discharge_report(patient_id: str) -> str:
    path = find_doctor_report_file(patient_id)   # glob Data/incoming/doctor_reports/{patient_id}_*
    return extract_text_any_format(path)          # reuse Harvester's extractor
```

### 3.3 Prompts — `prompts.py`

**Purpose:** Centralize LLM prompt templates so agents fetch them via
`get_prompt(name, **params)` instead of hardcoding strings (a grading
requirement).

| Prompt name | Params | Used by | Function | Output |
|---|---|---|---|---|
| `discharge-extraction-prompt` | `language, doc_types` | Clinical Extractor Agent | `discharge_extraction_prompt(language, doc_types)` | prompt string instructing the LLM to extract structured JSON fields per Table 3 |
| `ehr-cross-validation-prompt` | `patient_id` | Clinical Validation Agent | `ehr_cross_validation_prompt(patient_id)` | prompt string embedding the EHR bundle + Table 4 rules, asking for a verdict per rule |
| `abbreviation-normalization-prompt` | `source_language` | Clinical Normalizer Agent | `abbreviation_normalization_prompt(source_language)` | prompt instructing translation to English + abbreviation expansion |
| `summary-generation-prompt` | `risk_level, audience` | Summary Generator Agent | `summary_generation_prompt(risk_level, audience)` | prompt for patient-friendly vs clinician-friendly tone |
| `rag-answer-prompt` | `context_length` | RAG Generation Agent | `rag_answer_prompt(context_length)` | prompt instructing grounded QA with the "I don't know" fallback rule |

```python
# pseudocode
@mcp.prompt()
def ehr_cross_validation_prompt(patient_id: str) -> str:
    return f"""
You are validating discharge data for patient {patient_id} against the EHR.
Apply these rules in order: med_omission_check, allergy_contradiction_check,
diagnosis_mismatch_check, follow_up_missing_check, discharge_approval_check,
bill_settlement_check.
For each rule return {{rule_id, triggered: bool, evidence, severity}}.
Return ONLY JSON, no prose.
"""
```

### 3.4 Tool: Clinical Data Harvester — `tools_harvester.py`

**Purpose:** Extract text/structured data from any of the file types you'll
see (`.txt`, `.json`, `.pdf`, `.png`, `.png.ocr.txt`).

| Function | Input | Output | Calls |
|---|---|---|---|
| `extract_text_any_format(path: Path)` | file path | plain text string | dispatches to helpers below |
| `_read_txt(path)` | path | text | `path.read_text()` |
| `_read_json(path)` | path | text (pretty-printed JSON so the LLM can parse it) | `json.load` + `json.dumps` |
| `_read_pdf(path)` | path | text | `pdfplumber`/`pypdf` page extraction |
| `_read_png(path)` | path | text | OCR via `pytesseract.image_to_string` |
| `_read_png_ocr_txt(path)` | path (already OCR'd sidecar) | text | `path.read_text()` (OCR already done, just read it) |
| `clinical_data_harvester_tool(ctx, patient_id, doc_type)` (`@mcp.tool()`) | `patient_id`, `doc_type` (`bill`/`doctor_report`/`lab_report`) | `{patient_id, doc_type, raw_text, format}` | `resolve_authorized_root`, `extract_text_any_format` |

```python
# pseudocode
def extract_text_any_format(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".png.ocr.txt"):
        return _read_txt(path)
    ext = path.suffix.lower()
    if ext == ".txt":
        return _read_txt(path)
    if ext == ".json":
        return _read_json(path)
    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".png":
        return _read_png(path)
    raise ValueError(f"unsupported format: {ext}")

@mcp.tool()
def clinical_data_harvester_tool(ctx: Context, patient_id: str, doc_type: str):
    root = resolve_authorized_root(ctx)
    folder = {"bill": "bills", "doctor_report": "doctor_reports",
              "lab_report": "lab_reports"}[doc_type]
    path = find_file_for_patient(safe_join(root, folder), patient_id)
    text = extract_text_any_format(path)
    return {"patient_id": patient_id, "doc_type": doc_type,
            "raw_text": text, "format": path.suffix}
```

`find_file_for_patient` = `glob(f"{patient_id}_*")` inside the given folder,
prefer `.json` > `.txt` > `.png.ocr.txt` > `.pdf` > `.png` if multiple exist
for the same patient (structured > pre-OCR'd > needs-OCR).

### 3.5 Tool: Medical Lang Bridge — `tools_lang_bridge.py` (Sampling)

**Purpose:** Translate + normalize, but the *tool itself does not call an LLM
directly* — it issues a **sampling request back to the calling agent's LLM
client**. This is the mandated separation of concerns.

| Function | Input | Output | Calls |
|---|---|---|---|
| `medical_lang_bridge_tool(ctx, text, source_language)` (`@mcp.tool()`) | raw text, detected/declared source language | `{translated_text, confidence, model_used}` | `ctx.session.create_message()` |
| `build_model_preferences(source_language)` | language code | `ModelPreferences` object with hints | none (pure logic) |
| `parse_sampling_result(result)` | `CreateMessageResult` | `{translated_text, confidence}` | none (pure parsing) |

```python
# pseudocode
def build_model_preferences(source_language: str) -> ModelPreferences:
    if source_language == "en":
        return ModelPreferences(hints=[{"name": "command-r-plus"}])
    return ModelPreferences(hints=[{"name": "nova-lite"}])   # multilingual

@mcp.tool()
def medical_lang_bridge_tool(ctx: Context, text: str, source_language: str):
    prefs = build_model_preferences(source_language)
    result = ctx.session.create_message(
        messages=[{"role": "user", "content":
            f"Translate to English and normalize medical abbreviations. "
            f"Return JSON {{translated_text, confidence(0-1)}}. Text: {text}"}],
        model_preferences=prefs,
        max_tokens=2000,
    )
    return parse_sampling_result(result)
```

The **calling agent** (LangGraph Normalizer) must implement the other half:

| Function | Location | Input | Output | Calls |
|---|---|---|---|---|
| `sampling_callback(request)` | Normalizer Agent process | `CreateMessageRequest` (model hints + messages) | `CreateMessageResult` | routes to LiteLLM (`litellm.completion(model=hint, ...)`) |

### 3.6 Tool: Clinical Rules Engine — `tools_rules_engine.py` (Elicitation)

**Purpose:** Run completeness checks from Table 3; for missing **non-blocking**
fields, pause and ask a human via `ctx.elicit()`. For missing **blocking**
fields, skip elicitation and flag straight to HITL escalation.

| Function | Input | Output | Calls |
|---|---|---|---|
| `check_completeness(doc_type, extracted_fields)` | `doc_type`, dict of extracted fields | `{missing_blocking: list[str], missing_nonblocking: list[str]}` | `rules_loader.get_mandatory_fields`, Table 3 blocking-field lists (hardcode per doc type as a constant dict, since Table 3 isn't 1:1 with rules.yaml's flat list) |
| `build_elicitation_schema(missing_fields)` | list of field names | Pydantic model class built dynamically (or JSON schema) | `pydantic.create_model` |
| `clinical_rules_engine_tool(ctx, doc_type, extracted_fields)` (`@mcp.tool()`) | `doc_type`, extracted fields dict | `{status: "complete"\|"resolved"\|"unresolved"\|"blocked", fields, unresolved_fields}` | `check_completeness`, `build_elicitation_schema`, `ctx.elicit()` |

```python
# pseudocode
BLOCKING_FIELDS = {
  "discharge_report": ["patient_id","patient_name","discharge_diagnosis",
                        "discharge_approved","medications"],
  "lab_report": ["patient_id","tests"],
  "bill": ["patient_id","total_amount","payment_status"],
  "prescription": ["medicine_name","strength","frequency","route"],
}

def check_completeness(doc_type, extracted_fields):
    required = get_mandatory_fields(doc_type)   # full list from rules.yaml
    missing = [f for f in required if not extracted_fields.get(f)]
    blocking = set(BLOCKING_FIELDS[doc_type])
    return {
        "missing_blocking": [f for f in missing if f in blocking],
        "missing_nonblocking": [f for f in missing if f not in blocking],
    }

@mcp.tool()
def clinical_rules_engine_tool(ctx: Context, doc_type: str, extracted_fields: dict):
    gaps = check_completeness(doc_type, extracted_fields)
    if gaps["missing_blocking"]:
        return {"status": "blocked", "fields": extracted_fields,
                "unresolved_fields": gaps["missing_blocking"]}
    if not gaps["missing_nonblocking"]:
        return {"status": "complete", "fields": extracted_fields}

    schema = build_elicitation_schema(gaps["missing_nonblocking"])
    result = ctx.elicit(
        message=f"Missing fields for {doc_type}: {gaps['missing_nonblocking']}",
        schema=schema,
    )
    if result.action == "accept":
        extracted_fields.update(result.data)
        return {"status": "resolved", "fields": extracted_fields}
    if result.action == "decline":
        return {"status": "unresolved", "fields": extracted_fields,
                 "unresolved_fields": gaps["missing_nonblocking"]}
    # cancel
    return {"status": "blocked", "fields": extracted_fields,
             "unresolved_fields": gaps["missing_nonblocking"]}
```

The **dashboard side** of elicitation:

| Function | Location | Input | Output | Calls |
|---|---|---|---|---|
| `elicitation_callback(schema, message)` | Streamlit HITL Dashboard (page 3) | schema + prompt message from server | `ElicitResult(action, data)` | renders `st.form` dynamically from schema fields, returns reviewer input |

### 3.7 Tool: EHR Validation Tool — `tools_ehr_validator.py`

**Purpose:** Implement Table 4's six cross-validation rules against the Mock
EHR (via HTTP to :8050) and `rules.yaml`.

| Function | Input | Output | Calls |
|---|---|---|---|
| `fetch_ehr_bundle(patient_id)` | `patient_id` | JSON bundle | `httpx.get("http://localhost:8050/patients/{id}/bundle")` |
| `check_med_omission_check(discharge_meds, ehr_bundle)` | discharge med list, bundle | `{triggered, evidence}` | pure set-diff logic vs `ehr_bundle["med_orders"]` |
| `check_allergy_contradiction_check(discharge_meds, ehr_bundle)` | discharge meds, bundle | `{triggered, evidence}` | check any med name overlaps `ehr_bundle["allergies"]` (string match on generic name/class) |
| `check_diagnosis_mismatch_check(discharge_dx, ehr_bundle)` | discharge diagnosis text/ICD, bundle | `{triggered, evidence}` | compare against `ehr_bundle["patient"]["primary_dx"]` |
| `check_follow_up_missing_check(discharge_followup, ehr_bundle)` | discharge followup field, bundle | `{triggered, evidence}` | compare against `ehr_bundle["care_plan"]["followup_required"]` |
| `check_lab_follow_up_mismatch_check(discharge_text, ehr_bundle)` | discharge instructions text, bundle | `{triggered, evidence}` | for each `lab in ehr_bundle["labs"] if lab["abnormal"]`, check discharge text mentions it |
| `check_discharge_approval_check(extracted_fields)` | extracted fields | `{triggered, evidence}` | check `discharge_approved` == True and `discharge_approved_by` present |
| `check_bill_settlement_check(bill_fields)` | extracted bill fields | `{triggered, evidence}` | check `payment_status` == "PAID" or guarantee-letter flag present |
| `ehr_validation_tool(ctx, patient_id, extracted_discharge, extracted_bill)` (`@mcp.tool()`) | patient id + both extracted docs | `list[{rule_id, severity, triggered, evidence, action}]` | `fetch_ehr_bundle` + all `check_*` functions above |

```python
# pseudocode
RULES = [
    ("med_omission_check", "Warning", check_med_omission_check),
    ("allergy_contradiction_check", "Critical", check_allergy_contradiction_check),
    ("diagnosis_mismatch_check", "Warning", check_diagnosis_mismatch_check),
    ("follow_up_missing_check", "Critical", check_follow_up_missing_check),
    ("lab_follow_up_mismatch_check", "Warning", check_lab_follow_up_mismatch_check),
    ("discharge_approval_check", "Critical", check_discharge_approval_check),
    ("bill_settlement_check", "Critical", check_bill_settlement_check),
]

@mcp.tool()
def ehr_validation_tool(ctx, patient_id, extracted_discharge, extracted_bill):
    bundle = fetch_ehr_bundle(patient_id)
    results = []
    for rule_id, severity, fn in RULES:
        outcome = fn(extracted_discharge, bundle) if "bill" not in rule_id \
                  else fn(extracted_bill)
        results.append({
            "rule_id": rule_id, "severity": severity,
            "triggered": outcome["triggered"], "evidence": outcome["evidence"],
            "action": "Block discharge" if severity == "Critical" and outcome["triggered"]
                      else ("Flag for review" if outcome["triggered"] else "OK"),
        })
    return results
```

### 3.8 Tool: Clinical Insight Reporter — `tools_reporter.py`

**Purpose:** Combine completeness + cross-validation + risk score into the
final report, in JSON and HTML.

| Function | Input | Output | Calls |
|---|---|---|---|
| `compute_risk_score(completeness_gaps, ehr_findings, translation_confidence)` | gap lists, rule results, confidence float | `int` total score | `rules_loader.get_weight()` per triggered condition, sums `risk_scoring_matrix.weights` |
| `build_json_report(patient_id, all_inputs)` | patient id + all upstream results | dict matching report schema (fields listed in §2.5 of spec) | `compute_risk_score`, `rules_loader.get_risk_tier`, `rules_loader.get_rules_sha256` |
| `render_html_report(json_report)` | json report dict | HTML string | `get_html_template()` resource + string templating (Jinja2) |
| `clinical_insight_reporter_tool(ctx, patient_id, all_inputs)` (`@mcp.tool()`) | patient id, all upstream agent outputs | `{json_path, html_path, risk_level, recommendation, discharge_blocked}` | `build_json_report`, `render_html_report`, file write to `data/reports/` |

```python
# pseudocode
def compute_risk_score(gaps, ehr_findings, translation_confidence):
    score = 0
    score += len(gaps["missing_blocking"]) * get_weight("missing_mandatory_field")
    for finding in ehr_findings:
        if finding["triggered"]:
            score += get_weight(finding["rule_id"].replace("_check",""))
    if translation_confidence < 0.70:
        score += get_weight("low_translation_confidence")
    return score

@mcp.tool()
def clinical_insight_reporter_tool(ctx, patient_id, all_inputs):
    score = compute_risk_score(**all_inputs["scoring_inputs"])
    tier = get_risk_tier(score)
    blocked = any(f["severity"] == "Critical" and f["triggered"]
                  for f in all_inputs["ehr_findings"])
    report = build_json_report(patient_id, {**all_inputs, "risk_score": score,
                                             "risk_level": tier, "blocked": blocked})
    html = render_html_report(report)
    json_path = f"data/reports/{patient_id}_report.json"
    html_path = f"data/reports/{patient_id}_report.html"
    write_file(json_path, json.dumps(report, indent=2))
    write_file(html_path, html)
    return {"json_path": json_path, "html_path": html_path,
            "risk_level": tier, "recommendation": RULES_REC[tier],
            "discharge_blocked": blocked}
```

`RULES_REC` = `rules.yaml -> reporting.recommendations` (low/medium/high text).

### 3.9 `server.py` — wiring

```python
# pseudocode
mcp = FastMCP("primary-clinical-tools", port=8200, path="/clinicaltools")
register_resources(mcp)     # from resources.py
register_prompts(mcp)       # from prompts.py
mcp.tool()(clinical_watcher_tool)
mcp.tool()(clinical_data_harvester_tool)
mcp.tool()(medical_lang_bridge_tool)
mcp.tool()(clinical_rules_engine_tool)
mcp.tool()(ehr_validation_tool)
mcp.tool()(clinical_insight_reporter_tool)
mcp.run(transport="streamable-http")
```

---

## 4. Secondary MCP Analytics Server — port 8201, path `/analyticstools`

**Purpose:** Lightweight, tools-only server. No resources/prompts/sampling
here — just three tools.

| Function | Input | Output | Calls |
|---|---|---|---|
| `calculate_risk_score_tool(ctx, completeness_gaps, ehr_findings, translation_confidence)` | same shape as primary reporter's scoring inputs | `{score, tier}` | reuses `common/rules_loader` (shared library — same logic as §3.8, exposed here as a standalone callable tool for agents that only have the secondary server mounted) |
| `get_population_benchmarks_tool(ctx, service_line, icd10_code)` | service line string, ICD-10 code | `{readmission_rate_30d, avg_los_days, benchmark_source}` | static/mock lookup table `BENCHMARKS[service_line][icd10_code]` you seed yourself (no real dataset required — document it as simulated) |
| `generate_risk_heatmap_tool(ctx, patient_ids: list[str])` | list of patient ids | `{heatmap_path}` (PNG or base64) or structured `{patient_id: risk_score}` matrix | calls Primary server's reporter results (read already-generated JSON reports from `data/reports/`), then `matplotlib`/`plotly` to render |

```python
# pseudocode
@mcp.tool()
def get_population_benchmarks_tool(ctx, service_line: str, icd10_code: str):
    return BENCHMARKS.get(service_line, {}).get(icd10_code,
        {"readmission_rate_30d": None, "avg_los_days": None,
         "benchmark_source": "no data"})

@mcp.tool()
def generate_risk_heatmap_tool(ctx, patient_ids: list[str]):
    scores = {}
    for pid in patient_ids:
        report = json.load(open(f"data/reports/{pid}_report.json"))
        scores[pid] = report["risk_score"]
    fig = plot_heatmap(scores)
    path = f"data/reports/heatmap_{uuid4().hex[:6]}.png"
    fig.savefig(path)
    return {"heatmap_path": path, "scores": scores}
```

---

## 5. LangGraph Agents

Each agent is a `StateGraph` with `MemorySaver` checkpointing, wrapped as an
A2A server. All three connect to the Primary MCP server (8200) as clients
(via `mcp-use` or the MCP Python SDK client) — do not re-implement tool logic
inside the agent; the agent *calls* the MCP tools.

### 5.1 Clinical Extractor Agent — port 8100

**Graph state:** `{patient_id, doc_type, raw_text, extracted_fields, trace_id}`

| Node function | Input (state) | Output (state delta) | Calls |
|---|---|---|---|
| `node_harvest(state)` | `patient_id, doc_type` | `raw_text` | MCP tool `clinical_data_harvester_tool` |
| `node_build_prompt(state)` | `raw_text, language, doc_type` | `prompt` | MCP prompt `discharge-extraction-prompt` via `get_prompt()` |
| `node_extract(state)` | `prompt` | `extracted_fields` (dict) | LLM call (LiteLLM) with the fetched prompt, parses JSON response |
| `node_emit(state)` | `extracted_fields` | final state | returns to A2A caller |

```python
# pseudocode
graph = StateGraph(ExtractorState)
graph.add_node("harvest", node_harvest)
graph.add_node("build_prompt", node_build_prompt)
graph.add_node("extract", node_extract)
graph.add_edge("harvest", "build_prompt")
graph.add_edge("build_prompt", "extract")
graph.set_entry_point("harvest")
app = graph.compile(checkpointer=MemorySaver())

def node_extract(state):
    resp = litellm.completion(model="bedrock/nova-lite",
                               messages=[{"role":"user","content": state["prompt"]}])
    state["extracted_fields"] = safe_json_parse(resp.choices[0].message.content)
    return state
```

**A2A wrapper (`agent_card.py` + `server.py`):**
| Function | Input | Output | Calls |
|---|---|---|---|
| `get_agent_card()` | none | `AgentCard` JSON (served at `/.well-known/agent.json`) | static dict |
| `handle_a2a_message(task)` | A2A `Task` with `{patient_id, doc_type}` | A2A `Artifact` with `extracted_fields` | `app.invoke(...)` (the compiled LangGraph) |

### 5.2 Clinical Normalizer Agent — port 8102

**Graph state:** `{patient_id, raw_text, source_language, translated_text,
confidence, normalized_text}`

| Node function | Input | Output | Calls |
|---|---|---|---|
| `node_detect_language(state)` | `raw_text` | `source_language` | small LLM call or `langdetect` library |
| `node_translate(state)` | `raw_text, source_language` | `translated_text, confidence` | MCP tool `medical_lang_bridge_tool` (which itself round-trips via **this agent's own** `sampling_callback`) |
| `node_normalize_abbrev(state)` | `translated_text` | `normalized_text` | MCP resource `medical-abbreviations` + local substitution, or MCP prompt `abbreviation-normalization-prompt` + LLM |
| `sampling_callback(request)` | `CreateMessageRequest` from server | `CreateMessageResult` | `litellm.completion(model=request.model_preferences.hints[0].name, ...)` |

```python
# pseudocode
def sampling_callback(request):
    model = request.model_preferences.hints[0]["name"]
    litellm_model = MODEL_MAP[model]     # "nova-lite" -> "bedrock/amazon.nova-lite"
    resp = litellm.completion(model=litellm_model, messages=request.messages,
                               max_tokens=request.max_tokens)
    return CreateMessageResult(role="assistant",
                                content=resp.choices[0].message.content,
                                model=litellm_model)

def node_translate(state):
    result = mcp_client.call_tool("medical_lang_bridge_tool",
        {"text": state["raw_text"], "source_language": state["source_language"]})
    state["translated_text"] = result["translated_text"]
    state["confidence"] = result["confidence"]
    return state
```

Register `sampling_callback` when opening the MCP client session — it's the
handler the *server's* `ctx.session.create_message()` call routes into.

### 5.3 Clinical Validation Agent — port 8101

**Graph state:** `{patient_id, extracted_discharge, extracted_bill,
completeness_result, ehr_findings, final_status}`

| Node function | Input | Output | Calls |
|---|---|---|---|
| `node_completeness_check(state)` | `extracted_discharge` (doc_type inferred) | `completeness_result` (`status`, possibly triggers elicitation) | MCP tool `clinical_rules_engine_tool` |
| `node_ehr_cross_validate(state)` | `patient_id, extracted_discharge, extracted_bill` | `ehr_findings` | MCP prompt `ehr-cross-validation-prompt` (for LLM-assisted reasoning over ambiguous cases) + MCP tool `ehr_validation_tool` (for deterministic rule checks) |
| `node_decide(state)` | `completeness_result, ehr_findings` | `final_status` (`"auto_approve"\|"hitl"\|"blocked"`) | pure logic using `rules_loader.get_risk_tier` after calling Reporter (or delegates scoring to Reporter tool directly) |

```python
# pseudocode
def node_decide(state):
    if state["completeness_result"]["status"] == "blocked":
        state["final_status"] = "blocked"
    elif any(f["severity"]=="Critical" and f["triggered"] for f in state["ehr_findings"]):
        state["final_status"] = "blocked"
    else:
        state["final_status"] = "hitl" if any(f["triggered"] for f in state["ehr_findings"]) \
                                 else "auto_approve"
    return state
```

This agent's A2A response feeds the Reporter tool (called either from inside
this agent, or by the Orchestrator right after — pick one; simplest is to
call `clinical_insight_reporter_tool` as this graph's final node).

---

## 6. Google ADK Agents

### 6.1 Discharge Monitor Agent — port 8103

Thin ADK agent. Its only job: call the Watcher tool and hand results to the
Orchestrator.

| Function | Input | Output | Calls |
|---|---|---|---|
| `poll_for_new_documents()` | none (runs on a timer/loop or on-demand A2A call) | `list[{patient_id, filename, doc_type, path}]` | MCP tool `clinical_watcher_tool` |
| `handle_a2a_message(task)` | A2A task (trigger) | A2A artifact: new-document list | `poll_for_new_documents` |

```python
# pseudocode
async def handle_a2a_message(task):
    async with mcp_client_session(roots=["file:///Data/incoming"]) as ctx:
        new_files = await ctx.call_tool("clinical_watcher_tool", {})
    return Artifact(data=new_files)
```

### 6.2 Discharge Summary Generator Agent — port 8104 (STREAMING)

**Purpose:** Produce the patient-friendly summary, streamed section by
section: patient -> meds -> labs -> bill -> instructions.

| Function | Input | Output | Calls |
|---|---|---|---|
| `build_summary_prompt(patient_id, risk_level, audience="patient")` | ids + risk level | prompt string | MCP prompt `summary-generation-prompt` |
| `stream_section(section_name, data, prompt_ctx)` | section name, relevant data slice, shared prompt context | async generator yielding text chunks | LLM streaming call (`litellm.completion(..., stream=True)`) |
| `handle_a2a_streaming_message(task)` (the A2A `streaming=True` entry point) | task with `patient_id` | async stream of `TaskArtifactUpdateEvent`s, one per section | `build_summary_prompt`, then `stream_section` for each of the 5 sections in order |

```python
# pseudocode
async def handle_a2a_streaming_message(task):
    patient_id = task.input["patient_id"]
    report = load_json(f"data/reports/{patient_id}_report.json")
    prompt_ctx = build_summary_prompt(patient_id, report["risk_level"])
    for section in ["patient", "meds", "labs", "bill", "instructions"]:
        async for chunk in stream_section(section, report[section], prompt_ctx):
            yield TaskArtifactUpdateEvent(section=section, delta=chunk)
```

---

## 7. Agno RAG Q&A Agent — port 8105 (STREAMING)

**Purpose:** Five Agno roles, one A2A-exposed agent using `MultiMCPTools` to
reach both MCP servers, SQLite session memory (last 3 turns).

### 7.1 Indexing Agent

| Function | Input | Output | Calls |
|---|---|---|---|
| `index_all_documents()` | none (scans `data/reports/` + extracted doc texts) | writes/updates FAISS index at `data/vector_db/` | chunker + `sentence-transformers/all-MiniLM-L6-v2` embeddings + `faiss.IndexFlatL2` (or `faiss.write_index`) |
| `chunk_document(text, size=500, overlap=50)` | doc text | `list[str]` chunks | plain sliding-window split |

### 7.2 Retrieval Agent

| Function | Input | Output | Calls |
|---|---|---|---|
| `embed_query(question)` | question string | vector | same embedding model as indexing |
| `retrieve_top_k(question, k=5)` | question, k | `list[{chunk, score, source_doc}]` | `embed_query` + `faiss_index.search()` |

### 7.3 Augmentation Agent

| Function | Input | Output | Calls |
|---|---|---|---|
| `rerank_by_keyword(question, chunks)` | question, retrieved chunks | reordered `list[chunks]` | simple TF/keyword-overlap scoring (BM25-style) on top of vector score |

### 7.4 Generation Agent

| Function | Input | Output | Calls |
|---|---|---|---|
| `generate_answer(question, ranked_chunks, context_length)` | question, chunks, context window budget | `{answer, sources}` | MCP prompt `rag-answer-prompt` (fetched via `get_prompt`, NOT hardcoded) + LLM call |

```python
# pseudocode
async def generate_answer(question, ranked_chunks, context_length):
    prompt_template = await mcp_client.get_prompt("rag-answer-prompt",
                                                    {"context_length": context_length})
    context = "\n---\n".join(c["chunk"] for c in ranked_chunks[:context_length])
    full_prompt = prompt_template.format(context=context, question=question)
    resp = await agno_agent.arun(full_prompt)
    if not resp or "not available in the patient records" in resp.lower():
        return {"answer": "I don't know — this information is not available "
                           "in the patient records.", "sources": []}
    return {"answer": resp, "sources": [c["source_doc"] for c in ranked_chunks[:context_length]]}
```

### 7.5 Reflection Agent (RAG Triad)

| Function | Input | Output | Calls |
|---|---|---|---|
| `score_faithfulness(answer, context_chunks)` | answer, chunks used | float 0-1 | LLM-as-judge prompt: "does every claim trace to context?" |
| `score_answer_relevance(answer, question)` | answer, question | float 0-1 | LLM-as-judge prompt |
| `score_context_relevance(context_chunks, question)` | chunks, question | float 0-1 | LLM-as-judge prompt |
| `rag_triad_score(answer, question, context_chunks)` | all of the above inputs | `{faithfulness, answer_relevance, context_relevance}` | calls the three scorers above; if `faithfulness < 0.7` this is what the Hallucination Check guardrail (§10) blocks on |

### 7.6 A2A streaming wrapper

| Function | Input | Output | Calls |
|---|---|---|---|
| `handle_a2a_streaming_message(task)` | task with `{question, patient_filter}` | async token stream | `retrieve_top_k` -> `rerank_by_keyword` -> `generate_answer` (streamed) -> (async, non-blocking) `rag_triad_score` logged to LangFuse |

Agno specifics to implement literally as named in the spec: `agno.Agent(...,
tools=MultiMCPTools(servers=[primary_url, secondary_url]), db=SqliteDb(...),
add_history_to_messages=True, num_history_runs=3)`, invoked with `await
agent.arun(prompt, stream=True)`.

---

## 8. Host Orchestrator — Google ADK, port 8083, Gradio UI

**Purpose:** Single entry point that sequences the whole pipeline per patient
and exposes a Gradio chat/monitor UI. It's an **A2A client** to every agent
above, not itself exposing clinical tools.

| Function | Input | Output | Calls |
|---|---|---|---|
| `run_discharge_pipeline(patient_id)` | `patient_id` | final report paths + status | sequentially calls (via A2A client `send_message`) Monitor(8103) -> Extractor(8100) x3 doc types -> Normalizer(8102) if non-English -> Validator(8101) -> (Validator internally triggers Reporter tool) |
| `a2a_call(agent_url, task_payload, streaming=False)` | target agent base URL, payload | `Artifact` or async stream | `a2a_sdk.send_message()` / `send_message_streaming()` with `X-Agent-Auth-Token` header |
| `stream_summary(patient_id)` | `patient_id` | forwards streamed sections to Gradio UI | `a2a_call(SUMMARY_AGENT_URL, ..., streaming=True)` |
| `gradio_ui()` | none | Gradio `Blocks` app | wires the above as button callbacks |

```python
# pseudocode
async def run_discharge_pipeline(patient_id):
    new_docs = await a2a_call(MONITOR_URL, {"trigger": True})
    extracted = {}
    for doc in new_docs:
        if doc["patient_id"] != patient_id:
            continue
        result = await a2a_call(EXTRACTOR_URL, {"patient_id": patient_id,
                                                  "doc_type": doc["doc_type"]})
        if detect_non_english(result["extracted_fields"]):
            result = await a2a_call(NORMALIZER_URL, {"patient_id": patient_id,
                                                       "raw_text": result["raw_text"]})
        extracted[doc["doc_type"]] = result

    validation = await a2a_call(VALIDATOR_URL, {
        "patient_id": patient_id,
        "extracted_discharge": extracted.get("doctor_reports"),
        "extracted_bill": extracted.get("bills"),
    })
    return validation   # contains report paths, risk_level, recommendation
```

Every `a2a_call` must send `X-Agent-Auth-Token: <shared_secret>` and read the
target's `/.well-known/agent.json` once at startup to confirm capabilities
(`get_agent_card` client-side counterpart: `fetch_agent_card(url)`).

---

## 9. Streamlit HITL Dashboard — port 8501, 5 pages

Each page is a function; shared state lives in `st.session_state`.

| Page | Function | Input | Output | Calls |
|---|---|---|---|---|
| 1. Document Viewer | `page_document_viewer()` | patient selection from `st.selectbox` | renders tabs, triggers processing | `a2a_call(ORCHESTRATOR_URL, {"patient_id": ...})` (i.e. hits the Orchestrator, not agents directly) |
| 2. Validation Report | `page_validation_report()` | `patient_id` from session state | renders completeness score, issues table, risk badge | reads `data/reports/{patient_id}_report.json` (written by Reporter tool) directly, or via a small "reports" MCP resource/tool if you want it live |
| 3. HITL Corrections | `page_hitl_corrections()` | reviewer edits via `st.data_editor` + dynamic elicitation form | writes corrections back, triggers re-run | `elicitation_callback` (registered with the MCP client used by the Validator agent's A2A call), then `a2a_call(VALIDATOR_URL, ...)` again |
| 4. RAG Q&A | `page_rag_qa()` | free-text question + patient filter | streamed answer + sources + RAG triad scores | `a2a_call(RAG_AGENT_URL, {"question":..., "patient_filter":...}, streaming=True)` |
| 5. Discharge Summary | `page_discharge_summary()` | `patient_id` (auto-approved only) | rendered summary + export buttons | `a2a_call(SUMMARY_AGENT_URL, ..., streaming=True)`, export via `render_html_report`/PDF conversion reused from §3.8 |

```python
# pseudocode (page 3 elicitation form — the trickiest one)
def render_elicitation_form(schema: dict):
    with st.form("elicit_form"):
        values = {}
        for field_name, field_type in schema["properties"].items():
            values[field_name] = st.text_input(field_name)
        accepted = st.form_submit_button("Submit")
        declined = st.form_submit_button("Decline")
        cancelled = st.form_submit_button("Cancel")
    if accepted:
        return ElicitResult(action="accept", data=values)
    if declined:
        return ElicitResult(action="decline")
    if cancelled:
        return ElicitResult(action="cancel")
    return None
```

---

## 10. RAI Guardrails — cross-cutting module, `guardrails/`

Built once, imported as a decorator/middleware into every MCP tool and every
agent's LLM call site.

| Function | Trigger | Input | Output | Calls |
|---|---|---|---|---|
| `redact_pii(text)` | before any log/API call containing patient text | raw text | text with names/phone/Aadhaar/PAN masked | regex + NER (`presidio` or custom regex set) |
| `check_hallucination(answer, context)` | RAG-generated response | answer, context chunks | `{faithfulness, blocked: bool}` | reuses `score_faithfulness` from §7.5; if `<0.7` -> `blocked=True`, caller must regenerate |
| `check_prompt_injection(user_query)` | any user-supplied query | query string | `{is_injection: bool, sanitized_query}` | pattern match against known injection phrasings + optional LLM classifier |
| `filter_toxicity(text)` | before including any text in a clinical instruction/summary | text | filtered text or rejection | moderation model call or keyword filter |
| `guardrail_manager(risk_level, discharge_blocked)` | after Reporter tool runs | risk level, blocked flag | `{requires_hitl: bool}` — `True` whenever `risk_level=="high"` or `discharge_blocked` | pure logic, called by Orchestrator before allowing auto-approve path |

```python
# pseudocode
def guardrail_manager(risk_level, discharge_blocked):
    requires_hitl = (risk_level == "high") or discharge_blocked
    return {"requires_hitl": requires_hitl}
```

Wrap tool entry points like:
```python
@mcp.tool()
def clinical_data_harvester_tool(ctx, patient_id, doc_type):
    result = _harvester_impl(ctx, patient_id, doc_type)
    result["raw_text"] = redact_pii(result["raw_text"])  # before returning/logging
    return result
```

---

## 11. LangFuse Observability — cross-cutting

| Function | Where | Input | Output | Calls |
|---|---|---|---|---|
| `start_trace(patient_id)` | Orchestrator, pipeline entry | `patient_id` | `trace_id` (passed in every downstream A2A message's metadata) | `langfuse_client.trace(...)` |
| `span_agent_call(trace_id, agent_name, fn, *args)` | wraps every A2A call | trace id, agent name, the call itself | agent result (passthrough) + emits a span with latency/payloads | `langfuse_client.span(...)` context manager around `fn(*args)` |
| `span_tool_call(trace_id, tool_name, fn, *args)` | wraps every `@mcp.tool()` invocation | trace id, tool name, call | tool result (passthrough) + span with params/result/duration | same pattern, inside MCP server |
| `log_llm_generation(trace_id, model, prompt, response, tokens, cost)` | every LiteLLM call site | generation metadata | none (fire-and-forget log) | `langfuse_client.generation(...)` |
| `log_sampling_event(trace_id, model_prefs, model_selected, result)` | inside `sampling_callback` | sampling metadata | none | `langfuse_client.event(...)` |
| `log_elicitation_event(trace_id, schema, reviewer_response, action)` | inside Rules Engine tool / dashboard | elicitation metadata | none | `langfuse_client.event(...)` |
| `log_guardrail_event(trace_id, guardrail_name, result, blocked)` | inside each guardrail function | guardrail outcome | none | `langfuse_client.event(...)` |

`trace_id` is generated once by the Orchestrator (`start_trace`) and threaded
through every A2A message's `metadata.trace_id` field so every span in every
service lands under one LangFuse trace.

---

## 12. End-to-end call graph (single patient, happy path)

```
Streamlit(page1) -> Orchestrator.run_discharge_pipeline(patient_id)
  -> A2A -> Monitor(8103) -> MCP:clinical_watcher_tool -> [files]
  -> A2A -> Extractor(8100)
       -> MCP:clinical_data_harvester_tool -> raw_text
       -> MCP prompt:discharge-extraction-prompt -> LLM -> extracted_fields
  -> (if non-English) A2A -> Normalizer(8102)
       -> MCP:medical_lang_bridge_tool -> ctx.session.create_message()
          -> Normalizer.sampling_callback() -> LiteLLM -> translated_text
       -> MCP resource:medical-abbreviations -> normalized_text
  -> A2A -> Validator(8101)
       -> MCP:clinical_rules_engine_tool -> (maybe) ctx.elicit()
            -> Dashboard.elicitation_callback() [pauses here, HITL page 3]
       -> MCP:ehr_validation_tool -> HTTP -> Mock EHR :8050 -> findings
       -> MCP:clinical_insight_reporter_tool
            -> Analytics(8201):calculate_risk_score_tool (optional cross-check)
            -> writes data/reports/{id}_report.(json|html)
  -> guardrail_manager(risk_level, blocked) -> requires_hitl?
  -> Streamlit(page2) shows report; if auto-approve -> page5 -> Summary(8104, streaming)
  -> Streamlit(page4) any time -> RAG(8105, streaming) -> Indexing/Retrieval/
     Augmentation/Generation/Reflection agents -> answer + RAG triad scores
```

Build and smoke-test each arrow above independently (curl/CLI script) before
connecting the next one — that's the fastest path to a working system given
how many moving parts this spec has.
