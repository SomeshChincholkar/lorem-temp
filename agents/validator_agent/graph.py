"""
agents/validator_agent/graph.py

Builds and compiles the Clinical Validation Agent's StateGraph:
  completeness -> ehr_cross_validate -> build_report -> decide -> END

Note: the report node is named "build_report", not "report" -- LangGraph
rejects a node whose name collides with a state key, and "report" is one.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .nodes import (
    node_completeness_check,
    node_decide,
    node_ehr_cross_validate,
    node_report,
)
from .state import ValidatorState


def build_graph():
    graph = StateGraph(ValidatorState)
    graph.add_node("completeness", node_completeness_check)
    graph.add_node("ehr_cross_validate", node_ehr_cross_validate)
    graph.add_node("build_report", node_report)
    graph.add_node("decide", node_decide)

    graph.set_entry_point("completeness")
    # Cross-validation runs even when completeness blocks: a blocked
    # case still needs its full finding list on the report so the
    # reviewer sees every problem at once, not one per re-run.
    graph.add_edge("completeness", "ehr_cross_validate")
    graph.add_edge("ehr_cross_validate", "build_report")
    graph.add_edge("build_report", "decide")
    graph.add_edge("decide", END)

    return graph.compile(checkpointer=MemorySaver())


validator_app = build_graph()
