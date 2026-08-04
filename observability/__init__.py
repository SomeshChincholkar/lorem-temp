"""
observability/

LangFuse tracing (spec section 7.2).

Everything here degrades to a no-op when LANGFUSE_PUBLIC_KEY /
LANGFUSE_SECRET_KEY are unset, so the system runs identically with and
without observability configured.

Coverage against spec 7.2:

| Requirement                        | Helper                     |
|------------------------------------|----------------------------|
| End-to-end trace per discharge case | trace_id_for(trace_id)     |
| Per-agent spans                     | observe(as_type="agent")   |
| Per-tool-call spans                 | observe(as_type="tool")    |
| LLM generation events               | observe(as_type="generation") |
|                                     | record_generation() when streaming |
| Sampling events                     | log_sampling_event()       |
| Elicitation events                  | log_elicitation_event()    |
| Guardrail intervention spans        | log_guardrail_events()     |
| Error spans                         | log_error()                |
"""

from .tracing import (
    flush,
    get_client,
    is_enabled,
    log_elicitation_event,
    log_error,
    log_event,
    log_guardrail_events,
    log_sampling_event,
    observe,
    record_generation,
    reset_client_cache,
    set_output,
    trace_id_for,
    trace_url,
)

__all__ = [
    "observe",
    "record_generation",
    "set_output",
    "log_event",
    "log_sampling_event",
    "log_elicitation_event",
    "log_guardrail_events",
    "log_error",
    "trace_id_for",
    "trace_url",
    "is_enabled",
    "get_client",
    "flush",
    "reset_client_cache",
]
