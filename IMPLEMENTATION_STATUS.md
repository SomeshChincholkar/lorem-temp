# Implementation Status

Living record of what is built, what is verified, what is left, and where
this implementation knowingly departs from the capstone spec.

Companion to `BUILD_GUIDE_FA5.md` (the plan). This file is the ground truth
for *what actually exists*.

Last updated: 2026-08-03 · **75 tests passing**

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
| 8 | Streamlit HITL Dashboard (5 pages) | 8501 | **Not started** |
| 7.1 | RAI Guardrails (4 modules) | — | **Partial** — 1 of 5 |
| 7.2 | LangFuse Observability | — | **Not started** |
| 5 | A2A Push Notifications | — | **Not started** |

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

**Host Orchestrator (:8083, ADK + Gradio)** — A2A client to everything.
Four tabs: Run Pipeline (non-streaming), Discharge Summary (streaming),
Clinical Q&A (streaming), System Health (AgentCard discovery).
One `trace_id` is minted at the top and threaded through every A2A call —
the hook LangFuse will use later.

### 2.4 Cross-cutting pieces built along the way

| Module | Why it exists |
|---|---|
| `agents/common/a2a_server.py` | Shared-secret middleware — was about to be copy-pasted 6× |
| `agents/common/a2a_client.py` | JSON-RPC + SSE A2A client, both call modes |
| `agents/common/adk_runtime.py` | Routes ADK agents to Bedrock via LiteLLM instead of Gemini |
| `agents/common/elicitation_store.py` | Cross-process HITL rendezvous (see §4) |
| `agents/common/language_detect.py` | langdetect → LLM → `"en"`, never raises |

---

## 3. Test coverage

`python -m pytest tests/ -q` — **75 tests, no network, no AWS credentials.**

| File | Tests | Covers |
|---|---|---|
| `test_rules_loader.py` | 10 | rules.yaml loading, weights, risk tiers |
| `test_normalizer_sampling.py` | 10 | **Real** MCP Sampling round trip over the SDK's in-memory transport |
| `test_normalizer_nodes.py` | 7 | Abbreviation expansion edge cases |
| `test_validator_decision.py` | 12 | Verdict precedence — every rule pinned separately |
| `test_elicitation.py` | 12 | Store lifecycle + all three elicitation outcomes + timeout |
| `test_orchestrator_and_rag.py` | 24 | Pipeline sequencing, guardrail, chunking, grounding gate, re-ranking |

The sampling test is worth calling out: it mounts the **real**
`medical_lang_bridge_tool` on a **real** FastMCP server and connects a
**real** `ClientSession` with the **real** callback. Only the Bedrock call
is stubbed. It is a genuine protocol test, not an assertion about mocks.

**Also verified against reality:**
- FAISS index builds over the real corpus — 12 documents, 78 chunks
- Retrieval is cross-lingual: an English question about the penicillin
  allergy correctly surfaces the *German* P1016 document
- The grounding gate works: "What is the capital of France?" scores 0.178,
  below the 0.25 threshold, so it returns the spec's exact refusal string
- All 8 test documents parse through the real harvester

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

**Normalizer touches Resources.** Spec Table 6 lists it as
"Tools + Sampling + Prompts". It also reads
`resource://medical-abbreviations` for the deterministic expansion pass.
Strictly additive.

**`.docx` substituted for a handwritten image.** P1014 was specced as a
scanned handwritten Spanish report. A `.docx` covers the missing document
format from §9's stack table and is honestly reproducible; generating a
convincing handwritten scan was not.

---

## 5. What remains

### 5.1 Streamlit HITL Dashboard — :8501, spec §8 · *not started*

Five pages. The backend each one needs already exists.

| Page | Needs | Backend ready? |
|---|---|---|
| 1. Document Viewer | Patient selector, tabbed docs, language badge, process trigger | Yes — `run_discharge_pipeline()` |
| 2. Validation Report | Completeness score, findings table, risk badge, blocked indicator | Yes — `Data/reports/{id}_report.json` |
| 3. HITL Corrections | `st.data_editor` med table, **dynamic elicitation form**, approval decision | Yes — `elicitation_store.list_pending()` / `.respond()` |
| 4. RAG Q&A | Streaming answer, sources panel, RAG Triad metrics | Yes — stream `:8105` |
| 5. Discharge Summary | Streamed summary, export JSON/HTML/PDF | Partly — PDF export missing |

Page 3 is the one that closes the Elicitation loop: read pending requests,
render a form from each request's JSON Schema, write back accept/decline/
cancel. The store API is designed for exactly this and is already tested.

### 5.2 RAI Guardrails — spec §7.1 · *1 of 5 done*

| Guardrail | Status |
|---|---|
| `GuardrailManager` (HITL escalation) | **Done** — `pipeline.guardrail_manager()` |
| `PIIRedactor` | Not started |
| `HallucinationChecker` | Half done — `rag_triad_score()` produces faithfulness; the <0.7 block-and-regenerate path is missing |
| `PromptInjectionGuard` | Not started |
| `ToxicityFilter` | Not started |

Intended shape: a `guardrails/` package imported as decorators at each MCP
tool entry point and LLM call site.

### 5.3 LangFuse Observability — spec §7.2 · *not started*

`trace_id` already threads through every A2A call, so the hard part is done.
Still needed: per-agent spans, per-tool-call spans, LLM generation events
with token counts, sampling events, elicitation events, guardrail spans, and
error spans.

### 5.4 Smaller gaps

| Gap | Impact | Notes |
|---|---|---|
| **`mark_processed()` never called** | Watcher re-reports the same files forever | Defined in `tools_watcher.py:49`, zero callers. Needs a call after successful extraction |
| **PDF report export** | §2.5 asks for JSON + HTML/**PDF**; only JSON + HTML exist | Add WeasyPrint/wkhtmltopdf over the existing HTML |
| **A2A Push Notifications** | Spec §5 lists them | All cards declare `push_notifications=False` |
| **Dual-MCP for most agents** | §4 says agents connect to both servers; only the RAG agent does | Others only need the primary; worth a defensible note or a second client |
| **Secondary server underused** | `calculate_risk_score` duplicates the Reporter's local scoring | Wire it in as a cross-check, or document as intentional |
| **Tesseract not installed** | P1023's `.png` OCR path untested | Install the Tesseract binary |
| **No live end-to-end run** | Nothing has run against real Bedrock | Needs `.env` credentials |

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

1. **Streamlit dashboard** — highest visible value, and it closes the
   Elicitation loop that is currently only half-demonstrable.
2. **RAI guardrails** — self-contained, testable offline, four small modules.
3. **LangFuse** — last, because instrumenting spans touches every call site
   and is easiest once those signatures have stopped moving.
4. **Small gaps** — `mark_processed`, PDF export, push notifications.

See `imp_command.txt` for start order and test commands.
