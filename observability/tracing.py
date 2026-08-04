"""
observability/tracing.py

LangFuse instrumentation (spec section 7.2).

Design constraint that shapes everything here: **tracing must never be
load-bearing.** If LangFuse is unconfigured, unreachable, or throws, the
clinical pipeline has to carry on exactly as if it were not there. So
every helper degrades to a no-op rather than raising, and no return value
from this module is required by any caller.

That is why the context managers below always yield (sometimes None) and
why every LangFuse call sits inside a try/except. An observability bug
must not block a discharge.

Trace identity: the Orchestrator mints a uuid4 `trace_id` and threads it
through every A2A message's metadata. LangFuse v4 is OpenTelemetry-based
and needs a 32-hex-char trace id, so `trace_id_for()` maps our uuid onto
one deterministically. Same seed in any process -> same LangFuse trace,
which is what stitches six separate services into one view.

PII: the client is constructed with a `mask` function wired to the
PIIRedactor, so identifiers are stripped inside the SDK before anything
is exported. Redaction at the boundary, not at each call site.
"""

import os
from contextlib import contextmanager
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

_client = None
_client_initialised = False


def is_enabled() -> bool:
    """Tracing is on only when both keys are present."""
    return bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)


def _mask(data: Any) -> Any:
    """
    Redact PII inside the SDK, before any payload leaves the process.

    LangFuse calls this on every input/output it exports, which makes it
    the single correct place for redaction -- a call site that forgot to
    redact cannot leak through it.
    """
    try:
        from guardrails import PIIRedactor

        redactor = PIIRedactor()
        if isinstance(data, str):
            return redactor.redact(data)
        if isinstance(data, dict):
            return redactor.redact_dict(data)
        if isinstance(data, list):
            return [_mask(item) for item in data]
    except Exception:
        pass
    return data


def get_client():
    """Cached LangFuse client, or None when tracing is disabled."""
    global _client, _client_initialised

    if _client_initialised:
        return _client

    _client_initialised = True
    if not is_enabled():
        _client = None
        return None

    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
            mask=_mask,
        )
    except Exception:
        # Bad keys, wrong host, SDK missing -- run untraced rather than
        # taking the pipeline down.
        _client = None

    return _client


def trace_id_for(seed: Optional[str]) -> Optional[str]:
    """
    Map our uuid4 trace_id onto a LangFuse/OTel trace id, deterministically.

    Every service that sees the same seed produces the same trace id, so
    spans emitted by six different processes land under one trace.
    """
    if not seed:
        return None
    try:
        from langfuse import Langfuse

        return Langfuse.create_trace_id(seed=seed)
    except Exception:
        return None


def _trace_context(seed: Optional[str]):
    """TraceContext attaching a span to the case-level trace."""
    trace_id = trace_id_for(seed)
    if not trace_id:
        return None
    try:
        from langfuse.types import TraceContext

        return TraceContext(trace_id=trace_id)
    except Exception:
        return None


def trace_url(seed: Optional[str]) -> Optional[str]:
    """Deep link for the dashboard's audit trail panel."""
    client = get_client()
    trace_id = trace_id_for(seed)
    if not client or not trace_id:
        return None
    try:
        return client.get_trace_url(trace_id=trace_id)
    except Exception:
        return None


# ---------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------
@contextmanager
def observe(
    name: str,
    as_type: str = "span",
    trace_seed: Optional[str] = None,
    input: Any = None,
    metadata: Optional[dict] = None,
    **span_kwargs,
):
    """
    Wrap a unit of work in a LangFuse span.

    Yields the span handle, or None when tracing is off. Callers use it
    like:

        with observe("tool:harvester", as_type="tool", trace_seed=tid) as span:
            result = do_work()
            set_output(span, result)

    as_type maps onto LangFuse's observation types -- "tool", "agent",
    "generation", "guardrail", "retriever" -- so the UI groups them
    meaningfully rather than showing an undifferentiated span list.

    Exceptions are recorded on the span and then re-raised: the caller's
    error handling is not this module's to change.
    """
    client = get_client()
    if client is None:
        yield None
        return

    try:
        context = _trace_context(trace_seed)
        manager = client.start_as_current_observation(
            name=name,
            as_type=as_type,
            input=input,
            metadata=metadata,
            trace_context=context,
            **span_kwargs,
        )
    except Exception:
        yield None
        return

    try:
        with manager as span:
            try:
                yield span
            except Exception as exc:
                # Error span (spec 7.2: exception type + fallback action).
                try:
                    span.update(
                        level="ERROR",
                        status_message=f"{type(exc).__name__}: {exc}",
                    )
                except Exception:
                    pass
                raise
    except Exception:
        raise


def set_output(span, output: Any) -> None:
    """Attach a result to a span. Safe when span is None."""
    if span is None:
        return
    try:
        span.update(output=output)
    except Exception:
        pass


def record_generation(
    name: str,
    trace_seed: Optional[str] = None,
    input: Any = None,
    output: Any = None,
    model: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    """
    Record a completed LLM generation in one call.

    Exists because `observe()` cannot safely wrap a streaming loop: the
    token loops here live inside async generators, and holding an
    OpenTelemetry context across `yield` points lets it leak into
    whatever the consumer does between tokens -- or never close at all
    if the consumer abandons the stream.

    So streaming call sites collect their tokens first and record the
    generation afterwards. Latency is not captured, which is the
    deliberate trade for not corrupting the surrounding trace.
    """
    client = get_client()
    if client is None:
        return
    try:
        observation = client.start_observation(
            name=name,
            as_type="generation",
            input=input,
            output=output,
            model=model,
            metadata=metadata,
            trace_context=_trace_context(trace_seed),
        )
        observation.end()
    except Exception:
        pass


# ---------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------
def log_event(
    name: str,
    trace_seed: Optional[str] = None,
    input: Any = None,
    output: Any = None,
    metadata: Optional[dict] = None,
    level: str = "DEFAULT",
) -> None:
    """
    Fire-and-forget event. Used for the point-in-time things spec 7.2
    calls out: sampling, elicitation, guardrail interventions, errors.
    """
    client = get_client()
    if client is None:
        return
    try:
        client.create_event(
            name=name,
            input=input,
            output=output,
            metadata=metadata,
            level=level,
            trace_context=_trace_context(trace_seed),
        )
    except Exception:
        pass


def log_sampling_event(
    trace_seed: Optional[str],
    model_preferences: Any,
    model_selected: Optional[str],
    result_preview: Optional[str] = None,
) -> None:
    """Spec 7.2: server model preferences, client model selected, result."""
    log_event(
        "mcp.sampling",
        trace_seed=trace_seed,
        input={"model_preferences": model_preferences},
        output={"model_selected": model_selected, "result": result_preview},
        metadata={"primitive": "sampling"},
    )


def log_elicitation_event(
    trace_seed: Optional[str],
    schema: Any,
    action: str,
    reviewer_response: Any = None,
    timed_out: bool = False,
) -> None:
    """Spec 7.2: schema sent, reviewer response, action taken."""
    log_event(
        "mcp.elicitation",
        trace_seed=trace_seed,
        input={"schema": schema},
        output={"action": action, "response": reviewer_response},
        metadata={"primitive": "elicitation", "timed_out": timed_out},
        level="WARNING" if timed_out else "DEFAULT",
    )


def log_guardrail_events(manager, trace_seed: Optional[str] = None) -> None:
    """
    Flush a GuardrailManager's event log as LangFuse guardrail spans.

    Called once after a guarded operation rather than per check, so the
    guardrail's own hot path stays free of network calls.
    """
    if manager is None:
        return
    try:
        events = manager.summary().get("events", [])
    except Exception:
        return

    for event in events:
        log_event(
            f"guardrail.{event.get('guardrail', 'unknown')}",
            trace_seed=trace_seed or event.get("trace_id"),
            output={"triggered": event.get("triggered")},
            metadata=event,
            level="WARNING" if event.get("triggered") else "DEFAULT",
        )


def log_error(
    name: str,
    exc: BaseException,
    trace_seed: Optional[str] = None,
    fallback_action: Optional[str] = None,
) -> None:
    """Spec 7.2 error spans: exception type, message, fallback taken."""
    log_event(
        name,
        trace_seed=trace_seed,
        output={
            "exception_type": type(exc).__name__,
            "message": str(exc)[:2000],
            "fallback_action": fallback_action,
        },
        metadata={"kind": "error"},
        level="ERROR",
    )


def flush() -> None:
    """
    Force-send buffered spans.

    Needed because the agents are short-lived per request and the SDK
    batches in the background; without this, spans from a request that
    finishes quickly can be lost.
    """
    client = get_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        pass


def reset_client_cache() -> None:
    """Test hook -- forces re-reading configuration."""
    global _client, _client_initialised
    _client = None
    _client_initialised = False
