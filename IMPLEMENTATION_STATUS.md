# Implementation Status

Living record of what is built, what is verified, what is left, and where
this implementation knowingly departs from the capstone spec.

Companion to `BUILD_GUIDE_FA5.md` (the plan). This file is the ground truth
for *what actually exists*.

Last updated: 2026-08-04 · **219 tests passing** · **every spec section implemented**

---

## 1. Status at a glance

| Spec § | Component | Port | Status |
|---|---|---|---|
| — | Mock EHR System (FastAPI) | 8050 | **Done** |
| — | Shared rules loader (`common/rules_loader.py`) | — | **Done** |
| 4 | Primary MCP Clinical Tools Server | 8200 | **Done** — all 6 primitives |
| 4 | Secondary MCP Analytics Server | 8201 | **Done** — 3 tools |
| 2.2 | Clinical Extractor Agent (LangGraph) | 8100 | **Done** |
| 2.4 | Clinical Validation Agent (LangGraph) | 8101 | **Done** |
| 2.3 | Clinical Normalizer Agent (LangGraph) | 8102 | **Done** |
| 2.1 | Discharge Monitor Agent (Google ADK) | 8103 | **Done** |
| 2.5 | Discharge Summary Generator (ADK, streaming) | 8104 | **Done** |
| 2.6 | Clinical RAG Q&A Agent (Agno, streaming) | 8105 | **Done** |
| 8 | Host Orchestrator (ADK + Gradio) | 8083 | **Done** |
| 8 | Streamlit HITL Dashboard (5 pages) | 8501 | **Done** |
| 7.1 | RAI Guardrails (5 modules) | — | **Done** |
| 2.5 | PDF report / summary export | — | **Done** |
| 7.2 | LangFuse Observability | — | **Done** |
| 9 | A2A Push Notifications | — | **Done** |

**Every spec section is implemented.** What is left is verification
against live infrastructure, not construction — see §5.

---

## 2. What is implemented

### 2.1 Data layer

**Mock EHR (`mock_ehr/`)** — FastAPI on :8050 over the static dicts in
`data.py` (24 patients). Per-resource endpoints plus
`GET /patients/{id}/bundle`, the one-round-trip call the EHR Validation
tool actually uses.

**Test corpus (`Data/incoming/`)** — deliberately built to trip specific
rules, matching the scenario comments already in `mock_ehr/data.py`:

| Patient | Language | Format | Designed to trigger |
|---|---|---|---|
| P1019 Thomas Wright | EN | `.txt` + `.json` | Nothing — clean auto-approve baseline |
| P1015 Ananya Sharma | HI | `.txt` | Low translation confidence only |
| P1016 Lukas Müller | DE | `.json` | `allergy_contradiction_check` (Penicillin allergy + Amoxicillin) → **Critical, blocks** |
| P1014 Lucía Fernández | ES | `.docx` | `med_omission_check` ×2 (Loperamide + Hyoscine dropped) → Warning |
| P1020 Diego Morales | ES | `.pdf` | PDF extraction path |
| P1023 Grace Bennett | EN | `.png` | OCR path (needs Tesseract installed) |

Each of P1014/P1015/P1016 has a matching in-language bill, all marked PAID
so the intended rule is the *only* trigger in that case.

### 2.2 Primary MCP Server (:8200 `/clinicaltools`)

Six tools, six resources, five prompts — all six MCP primitives:

| Primitive | Where | Client half |
|---|---|---|
| **Tools** | `mcp.tool()` × 6 | `agents/common/mcp_client.py` |
| **Resources** | `resources.py` × 6 URIs | `read_resource_text()` |
| **Prompts** | `prompts.py` × 5 | `get_prompt_text()` |
| **Sampling** | `tools_lang_bridge.py` | `agents/normalizer_agent/sampling.py` |
| **Elicitation** | `tools_rules_engine.py` | `agents/validator_agent/elicitation.py` |
| **Roots** | `roots.py` + `tools_watcher.py` | `_list_roots_callback` in `mcp_client.py` |

`tools_harvester.py` reads `.txt`, `.json`, `.pdf`, `.docx`, `.png` (OCR)
and `.png.ocr.txt`. The `.docx` reader extracts **tables as well as
paragraphs** — python-docx excludes table text from `document.paragraphs`,
and the discharge reports keep demographics and prescriptions in tables, so
paragraph-only extraction would silently drop every mandatory field.

### 2.3 Agents

Every agent follows the same shape: `state.py` → `nodes.py` → `graph.py` →
`agent_executor.py` → `agent_card.py` → `server.py`, with the shared-secret
middleware from `agents/common/a2a_server.py`.

**Clinical Extractor (:8100)** — harvest → detect language → fetch prompt →
extract. Deliberately does *not* translate; that is the Normalizer's job.

**Clinical Normalizer (:8102)** — harvest → detect language → fetch prompt →
translate → normalize abbreviations.
The Sampling round trip is the interesting part: the Lang Bridge tool issues
`ctx.session.create_message()`, which the SDK routes back into this agent's
`sampling_callback`, which reads the server's model hints (`nova-lite` for
multilingual, `command-r-plus` for English), maps them to concrete Bedrock
model IDs, runs the completion, and returns a `CreateMessageResult`.
A deterministic abbreviation pass then runs over the translated text using
`resource://medical-abbreviations`, so expansions are reproducible from
`rules.yaml` rather than dependent on the model's mood.

**Clinical Validation (:8101)** — completeness → EHR cross-validation →
build report → decide.
`decide_final_status()` collapses three signals into `auto_approve` / `hitl`
/ `blocked`, with blocking always winning over advisory.
Cross-validation runs even when completeness blocks, so a reviewer sees
every problem at once instead of one per re-run.

**Discharge Monitor (:8103, ADK)** — calls the Watcher tool, which discovers
its own authorized folder via `ctx.list_roots()`. No path is ever passed as
a tool parameter.

**Summary Generator (:8104, ADK, STREAMING)** — five sections in spec order
(patient → meds → labs → bill → instructions), each emitted as its own A2A
artifact the moment it finishes. Sourced strictly from the validated audit
report. A failed section is recorded inline rather than costing the whole
summary.

**RAG Q&A (:8105, Agno, STREAMING)** — all five spec roles:

| Role | Where |
|---|---|
| Indexing | `indexing.py` — chunk, embed (all-MiniLM-L6-v2), FAISS |
| Retrieval | `roles.retrieve_top_k` |
| Augmentation | `roles.rerank_by_keyword` |
| Generation | `roles.build_answer_prompt` (via MCP Prompts) |
| Reflection | `roles.rag_triad_score` |

Agno specifics per spec: `agno.Agent` + `MultiMCPTools` across **both** MCP
servers, `SqliteDb` session persistence with `num_history_runs=3`, async
invocation.

**One FAISS index per patient**, and `patient_id` is mandatory on every
question:

```
data/vector_db/
  P1014/  index.faiss  chunks.json
  P1015/  index.faiss  chunks.json
  ...
```

FAISS has no metadata filtering, so a shared index could only scope a
query by over-fetching and discarding other patients' chunks afterwards —
approximate, and it silently returns fewer results than asked for when the
requested patient ranks below the over-fetch window. A patient-scoped
index makes the top-k exact and makes cross-patient leakage structurally
impossible rather than filtered-out after the fact.

The trade-off, accepted deliberately: **cross-patient questions are no
longer answerable in one query.** "Which patient has an allergy
contradiction?" now has to be asked per patient. Enforced in three places
so it can't be forgotten — `retrieve_top_k` raises without a patient, the
agent returns a distinct "select a patient" message, and the A2A executor
fails the task. The dashboard's page-4 example queries were rewritten
without patient IDs, since the sidebar selection supplies the scope.

**Host Orchestrator (:8083, ADK + Gradio)** — A2A client to everything.
Four tabs: Run Pipeline (non-streaming), Discharge Summary (streaming),
Clinical Q&A (streaming), System Health (AgentCard discovery).
One `trace_id` is minted at the top and threaded through every A2A call —
the hook LangFuse will use later.

### 2.4 RAI Guardrails (`guardrails/`)

All five of spec Table 12, pattern-based rather than model-based so that
"why was this blocked?" always has an answer — which matters more than
recall in a clinical audit trail.

| Guardrail | Module | Runs where |
|---|---|---|
| **PII/PHI Redaction** | `pii.py` | Log and external-call boundary |
| **Prompt Injection** | `injection.py` | RAG query, before any model sees it |
| **Toxicity Filter** | `toxicity.py` | Each summary section, before emit |
| **Hallucination Check** | `hallucination.py` | RAG answer, gated on RAG Triad faithfulness |
| **HITL Escalation** | `manager.py` | Orchestrator, after the Reporter runs |

`GuardrailManager` composes all five and keeps one event log — the
structure LangFuse will consume as guardrail spans.

Decisions worth knowing:
- **Injection has two tiers.** REJECT for patterns with no legitimate
  clinical reading; SANITIZE for role markers where a real question may
  still be in there. Deliberately conservative on REJECT — an
  administrator wrongly blocked from asking about a patient is a real cost.
- **Toxicity blocks wholesale, not by clause.** A sentence telling a
  patient to stop their medication cannot be made safe by deleting part
  of it.
- **An unscored answer is unverified, not passing.** If the RAG Triad
  judge failed, `HallucinationChecker` blocks. "We couldn't check" must
  never be silently equivalent to "it's fine".
- **`patient_id` is never redacted.** It is the join key across the EHR,
  reports and the vector store; masking it would make a log entry
  impossible to correlate, defeating the point of logging it.

### 2.5 Streamlit HITL Dashboard (`dashboard/`, :8501)

Five pages via `st.navigation`, with the patient selection shared across
them in session state.

| Page | What it does |
|---|---|
| 1. Document Viewer | Tabbed Discharge/Lab/Bill, language badge, structured preview, pipeline trigger |
| 2. Validation Report | Completeness bar, colour-coded findings table, risk badge, blocked indicator |
| 3. HITL Corrections | **Dynamic elicitation form**, editable medication table, risk override, approval, re-run |
| 4. RAG Q&A | Streaming answer, live injection indicator, source panel, RAG Triad metrics |
| 5. Discharge Summary | Streamed summary, plain-English prescriptions, JSON/HTML/PDF export |

Page 3 closes the Elicitation loop. The form is built from the JSON
Schema the *server* sent, not from a hardcoded field list — that is what
makes it a real Elicitation client rather than a form that happens to
collect similar fields. All three outcomes (accept / decline / cancel)
are reachable.

Page 5 refuses to generate a patient-facing summary for a blocked
discharge. Table 13 scopes the page to auto-approved cases, and producing
a reassuring document for a case no clinician has cleared is precisely
what the HITL guardrail exists to prevent.

Pages 2 and 5 read `Data/reports/` directly, so a completed case can be
reviewed with every agent stopped.

### 2.6 LangFuse Observability (`observability/`, spec §7.2)

Every requirement in §7.2 has a wired hook:

| Requirement | Where |
|---|---|
| End-to-end trace per discharge case | `pipeline.discharge` root span; `trace_id_for()` maps the uuid onto an OTel trace id |
| Per-agent spans | `@traced_agent(...)` on all six executors |
| Per-tool-call spans | `mcp_client.call_tool()` — one place, all eight tools |
| LLM generation events | `observe(as_type="generation")` + `record_generation()` |
| Sampling events | `log_sampling_event()` in the Normalizer's callback |
| Elicitation events | `log_elicitation_event()` in the Validator's callback |
| Guardrail intervention spans | `log_guardrail_events()` flushes the manager's log |
| Error spans | `log_error()` in every fallback branch |

Three decisions worth knowing:

- **Tracing is never load-bearing.** Every helper degrades to a no-op if
  LangFuse is unconfigured, unreachable, or throwing. No caller depends
  on a return value. An observability bug must not be able to block a
  discharge — there are tests for both the unconfigured and the
  actively-broken client.
- **PII is masked inside the SDK.** The client is built with
  `mask=` wired to `PIIRedactor`, so identifiers are stripped before
  anything is exported. A call site that forgot to redact cannot leak
  through it. `patient_id` is deliberately preserved so traces stay
  correlatable.
- **Streaming generations are recorded, not wrapped.** Holding an OTel
  context across `yield` in an async generator lets it leak into whatever
  the consumer does between tokens — or never close if the consumer
  abandons the stream. `record_generation()` logs the completed
  generation instead, trading latency capture for not corrupting the
  surrounding trace.

### 2.7 Cross-cutting pieces built along the way

| Module | Why it exists |
|---|---|
| `agents/common/a2a_server.py` | Shared-secret middleware + `traced_agent` decorator |
| `agents/common/a2a_client.py` | JSON-RPC + SSE A2A client, both call modes |
| `agents/common/adk_runtime.py` | Routes ADK agents to Bedrock via LiteLLM instead of Gemini |
| `agents/common/elicitation_store.py` | Cross-process HITL rendezvous (see §4) |
| `agents/common/language_detect.py` | langdetect → LLM → `"en"`, never raises |
| `agents/common/push_notifications.py` | Signed webhook POSTs (HMAC-SHA256) for completed cases |
| `common/pdf_export.py` | PDF for summaries + audit reports (fpdf2, no system deps) |
| `tests/conftest.py` | Puts both MCP server dirs on `sys.path` (they use flat sibling imports) |

---

## 3. Test coverage

`python -m pytest tests/ -q` — **219 tests, no network, no AWS credentials.**

| File | Covers |
|---|---|
| `test_rules_loader.py` | rules.yaml loading, weights, risk tiers |
| `test_normalizer_sampling.py` | **Real** MCP Sampling round trip over the SDK's in-memory transport |
| `test_normalizer_nodes.py` | Abbreviation expansion edge cases |
| `test_validator_decision.py` | Verdict precedence — every rule pinned separately |
| `test_elicitation.py` | Store lifecycle + all three elicitation outcomes + timeout |
| `test_orchestrator_and_rag.py` | Pipeline sequencing, guardrail, chunking, grounding gate, re-ranking |
| `test_guardrails.py` | All five guardrails, including false-positive cases |
| `test_pdf_export.py` | PDF rendering, Latin-1/Unicode encoding, page wrapping |
| `test_observability.py` | No-op when disabled, survival when the client is broken, trace identity, PII masking |
| `test_push_and_watcher_state.py` | HMAC signing/verification, processed-file ledger |
| `test_imports.py` | Every runnable entry point imports and builds |
| `test_per_patient_index.py` | Index isolation, exact top-k, mandatory patient scope, lifecycle |

`test_imports.py` exists because of a real miss: `record_generation` was
used by `agents/rag_agent/agent.py` but never exported from
`observability/__init__`, and the whole suite stayed green because no
test imported that module — the RAG tests import `rag_agent.roles`, not
`rag_agent.agent`. It only surfaced when starting the server by hand. A
suite that never imports what you actually run cannot tell you what you
actually run is broken.

The guardrail tests deliberately cover **both** failure directions:
letting something through that should be caught, *and* blocking
something legitimate. The second is easy to forget and, in a hospital
tool, is a real cost — hence explicit cases like "Ignore the abnormal
potassium — was anything else flagged?" which contains "ignore" but is a
genuine clinical question and must not be rejected.

The sampling test is worth calling out: it mounts the **real**
`medical_lang_bridge_tool` on a **real** FastMCP server and connects a
**real** `ClientSession` with the **real** callback. Only the Bedrock call
is stubbed. It is a genuine protocol test, not an assertion about mocks.

**Also verified against reality:**
- Per-patient FAISS indexes build over the real corpus — 6 patients,
  12 documents, 78 chunks
- Index isolation holds: every index contains only its own patient's
  chunks, and the penicillin-allergy question scoped to P1019 returns
  only P1019 (a shared index used to surface P1016's German document)
- Retrieval is cross-lingual within a patient: an English question about
  the penicillin allergy correctly surfaces P1016's *German* document
- The grounding gate works: "What is the capital of France?" scores 0.178,
  below the 0.25 threshold, so it returns the spec's exact refusal string
- All 8 test documents parse through the real harvester
- The Streamlit dashboard boots and serves HTTP 200 with a clean log
- PDF export produces valid `%PDF` output for Spanish, German and Hindi
  patient names

**Not yet verified:** any live end-to-end run. No `.env` with AWS
credentials exists, so no agent has been started against real Bedrock. Live
smoke tests are written and ready (`test_5_1.py`, `test_5_2.py`).

---

## 4. Design decisions and deviations

Recorded so they can be defended rather than mistaken for oversights.

**Elicitation rendezvous is file-backed.** `ctx.elicit()` fires inside the
Validator's process; the reviewer sits in Streamlit. The callback parks a
JSON request under `Data/elicitations/`, polls for an answer, then returns
an `ElicitResult`. Atomic writes (temp + `os.replace`) so the dashboard
cannot read a half-written file. Chosen over Redis because the request rate
is one-per-review and a dependency-free store keeps the system runnable with
nothing but `python -m`.

**A timeout declines rather than cancels.** Declining means "unresolved,
flag for HITL" — which is what actually happened when nobody was watching.
Cancelling would overstate it as an abort-and-escalate. Set
`ELICITATION_TIMEOUT_SECONDS=0` for unattended runs.

**ADK agents run on Bedrock, not Gemini.** ADK defaults to Google's models,
for which this project has no credentials. `adk_runtime.py` bridges through
ADK's LiteLlm wrapper — which is also the LLM gateway the spec names in §9.

**The Monitor's A2A path does not call an LLM.** Listing files is
deterministic; putting a model between "what is on disk" and the
Orchestrator's work queue adds latency, cost and a hallucination surface for
no benefit. The ADK `LlmAgent` is fully wired and drives the conversational
path (`use_llm: true`), but even then the structured document list comes
from the tool's own output, never from model prose.

**RAG refuses before generating, not after.** `has_grounding()` gates on
retrieval score, so an out-of-context question returns the spec's exact
refusal string without the model ever seeing irrelevant context it could
confabulate from.

**"No patient selected" is reported separately from "not in the records".**
Both end the query, but the first is something the user can fix and the
second is a fact about the record. Collapsing them into the one refusal
string would tell a reviewer their patient has no data when they simply
hadn't chosen one.

**Normalizer touches Resources.** Spec Table 6 lists it as
"Tools + Sampling + Prompts". It also reads
`resource://medical-abbreviations` for the deterministic expansion pass.
Strictly additive.

**`.docx` substituted for a handwritten image.** P1014 was specced as a
scanned handwritten Spanish report. A `.docx` covers the missing document
format from §9's stack table and is honestly reproducible; generating a
convincing handwritten scan was not.

**Guardrails are pattern-based, not model-based.** A classifier would
catch more paraphrases, but would make "why was this blocked?"
unanswerable. In a clinical audit trail an explainable filter beats a
slightly more sensitive one.

**The hallucination threshold comes from `rules.yaml`, not Table 12.**
Table 12 quotes 0.7; `rules.yaml` ships `rag_groundedness_min: 0.75`,
which is stricter. Config wins — the entire point of `rules.yaml` is that
thresholds move without a code change.

**PDF uses fpdf2, not WeasyPrint.** WeasyPrint gives better HTML fidelity
but needs GTK on Windows. For a document that is headings and paragraphs,
a pure-Python renderer with no system dependency is the better trade.
Consequence: fpdf2's core fonts are Latin-1 only, so `latin1_safe()`
transliterates on the way in — Western European accents survive intact,
non-Latin scripts degrade rather than raising mid-render.

---

## 5. What remains

Construction is finished. Everything below is verification against live
infrastructure, or a judgement call left open deliberately.

### 5.1 Not yet verified live

| Gap | Why it matters |
|---|---|
| **No live end-to-end run** | Every result so far is offline. Nothing has touched real Bedrock, and no agent has been started against another agent over A2A. This is the highest-value next step |
| **LangFuse never exercised against a real project** | The instrumentation is proven to no-op safely and to survive a broken client, but no span has actually landed in a LangFuse UI. Needs real keys |
| **Push notifications never delivered** | Signing/verification is tested; no webhook has received one. Point `PUSH_NOTIFICATION_URL` at any request-bin to confirm |
| **Tesseract not installed** | P1023's `.png` OCR path is the one document format never exercised |

### 5.2 Open judgement calls

| Item | Position |
|---|---|
| **Dual-MCP for most agents** | §4 says agents connect to both servers; only the RAG agent does. The others have no use for the analytics tools. Either wire a second client for show, or defend it — this doc takes the second position |
| **Secondary server underused** | `calculate_risk_score` duplicates the Reporter's local scoring. Wiring it as a cross-check would demonstrate multi-server calls in the validation path |
| **Monitor's A2A path skips the LLM** | Deliberate (see §4). Flag if graders want ADK in the machine-to-machine path too — `use_llm: true` already does this |

---

## 6. Environment health

**This project shares a global Python install with a yolov8/CV
environment, and that install is conflicted.** Everything currently imports
and all tests pass, but pip reports pre-existing incompatibilities across
`roboflow`, `mediapipe`, `paddleocr`, `tensorflow`, `inference-sdk` and —
most relevant here — `langchain-core 1.5.3` against `langgraph 0.2.59`,
which wants `<0.4.0`.

**Recommended before further work:** a dedicated virtualenv.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

`.env` does not exist yet. Copy `.env.example` and fill in AWS credentials
plus `AGENT_AUTH_TOKEN`.

---

## 7. Suggested order for the remainder

1. **Create a venv and install** (see §6) — before anything else.
2. **Fill in `.env`** with AWS credentials and an `AGENT_AUTH_TOKEN`.
3. **A live end-to-end run.** Start the services in the order in
   `imp_command.txt`, then run `P1019` (clean), `P1016` (allergy
   contradiction → blocked) and `P1015` (Hindi → translation) from the
   dashboard. This is the step most likely to surface an integration
   problem that offline testing cannot.
4. **Add LangFuse keys** and confirm one case produces a single trace
   containing agent, tool, generation, sampling and guardrail spans.
5. **Decide the open judgement calls** in §5.2.

See `imp_command.txt` for start order and test commands.
