"""
agents/normalizer_agent/graph.py

Builds and compiles the Clinical Normalizer Agent's StateGraph:
  harvest -> detect_language -> fetch_prompt -> translate
          -> normalize_abbrev -> END
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .nodes import (
    node_detect_language,
    node_fetch_prompt,
    node_harvest,
    node_normalize_abbrev,
    node_translate,
)
from .state import NormalizerState


def build_graph():
    graph = StateGraph(NormalizerState)
    graph.add_node("harvest", node_harvest)
    graph.add_node("detect_language", node_detect_language)
    graph.add_node("fetch_prompt", node_fetch_prompt)
    graph.add_node("translate", node_translate)
    graph.add_node("normalize_abbrev", node_normalize_abbrev)

    graph.set_entry_point("harvest")
    graph.add_edge("harvest", "detect_language")
    graph.add_edge("detect_language", "fetch_prompt")
    graph.add_edge("fetch_prompt", "translate")
    graph.add_edge("translate", "normalize_abbrev")
    graph.add_edge("normalize_abbrev", END)

    return graph.compile(checkpointer=MemorySaver())


normalizer_app = build_graph()
