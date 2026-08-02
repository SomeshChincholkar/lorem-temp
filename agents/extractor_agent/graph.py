"""
agents/extractor_agent/graph.py

Builds and compiles the Clinical Extractor Agent's StateGraph:
  harvest -> build_prompt -> extract -> END
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .nodes import node_build_prompt, node_extract, node_harvest
from .state import ExtractorState


def build_graph():
    graph = StateGraph(ExtractorState)
    graph.add_node("harvest", node_harvest)
    graph.add_node("build_prompt", node_build_prompt)
    graph.add_node("extract", node_extract)

    graph.set_entry_point("harvest")
    graph.add_edge("harvest", "build_prompt")
    graph.add_edge("build_prompt", "extract")
    graph.add_edge("extract", END)

    return graph.compile(checkpointer=MemorySaver())


extractor_app = build_graph()