"""
tests/test_imports.py

Imports every runnable entry point.

This file exists because of a real miss: `record_generation` was used by
agents/rag_agent/agent.py but never exported from observability/__init__,
and the whole suite stayed green because no test imported that module --
the RAG tests import `rag_agent.roles`, not `rag_agent.agent`. The break
only showed up when starting the server by hand.

A unit suite that never imports the thing you actually run will not tell
you the thing you actually run is broken. These tests are cheap and cover
that gap.

Run:  python -m pytest tests/test_imports.py -q
"""

import importlib

import pytest

AGENT_SERVERS = [
    "agents.extractor_agent.server",
    "agents.normalizer_agent.server",
    "agents.validator_agent.server",
    "agents.monitor_agent.server",
    "agents.summary_agent.server",
    "agents.rag_agent.server",
]

AGENT_MODULES = [
    "agents.extractor_agent.agent_executor",
    "agents.normalizer_agent.agent_executor",
    "agents.normalizer_agent.sampling",
    "agents.validator_agent.agent_executor",
    "agents.validator_agent.elicitation",
    "agents.monitor_agent.agent",
    "agents.monitor_agent.agent_executor",
    "agents.summary_agent.agent",
    "agents.summary_agent.sections",
    "agents.summary_agent.agent_executor",
    "agents.rag_agent.agent",          # the module the miss above was in
    "agents.rag_agent.roles",
    "agents.rag_agent.indexing",
    "agents.rag_agent.build_index",
    "agents.rag_agent.agent_executor",
    "agents.orchestrator.agent",
    "agents.orchestrator.pipeline",
    "agents.common.a2a_client",
    "agents.common.a2a_server",
    "agents.common.adk_runtime",
    "agents.common.elicitation_store",
    "agents.common.push_notifications",
]

SUPPORT_MODULES = [
    "common.rules_loader",
    "common.pdf_export",
    "guardrails",
    "observability",
    "mock_ehr.main",
    "mcp_primary.server",
    "mcp_primary.tools_watcher",
    "mcp_primary.tools_harvester",
    "mcp_primary.tools_reporter",
    "mcp_secondary.server",
]


@pytest.mark.parametrize("module", AGENT_SERVERS)
def test_agent_server_imports_and_builds_an_app(module):
    """Importing a server module builds its Starlette app at module scope."""
    imported = importlib.import_module(module)
    assert imported.app is not None


@pytest.mark.parametrize("module", AGENT_MODULES + SUPPORT_MODULES)
def test_module_imports(module):
    assert importlib.import_module(module) is not None


def test_orchestrator_ui_builds():
    """Gradio's Blocks graph is built at import time, so this catches typos."""
    from agents.orchestrator.server import demo

    assert demo is not None


@pytest.mark.parametrize(
    "name",
    [
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
        "flush",
    ],
)
def test_observability_exports_what_callers_import(name):
    """Guards exactly the class of break that motivated this file."""
    import observability

    assert hasattr(observability, name), f"observability.{name} is not exported"


@pytest.mark.parametrize(
    "name",
    [
        "GuardrailManager",
        "guardrail_manager",
        "PIIRedactor",
        "PromptInjectionGuard",
        "ToxicityFilter",
        "HallucinationChecker",
        "BLOCKED_ANSWER",
        "REJECT",
        "SANITIZE",
        "ALLOW",
    ],
)
def test_guardrails_exports_what_callers_import(name):
    import guardrails

    assert hasattr(guardrails, name), f"guardrails.{name} is not exported"
