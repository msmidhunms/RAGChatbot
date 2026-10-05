"""Compile the LangGraph for the current config.

START -> condense -> transform -> retrieve [-> postprocess] [-> grade -(retry)-> transform]
      -> generate [-> self_check -(retry)-> generate] -> finalize -> END
Optional nodes are only added when enabled, so the graph mirrors the config.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from rag.graph.nodes import Components, Nodes, timed
from rag.graph.state import RAGState


def build_graph(components: Components, checkpointer: BaseCheckpointSaver | None = None) -> Any:
    cfg = components.cfg
    n = Nodes(components)
    g = StateGraph(RAGState)

    g.add_node("condense", timed("condense", n.condense))
    g.add_node("transform", timed("transform", n.transform))
    g.add_node("retrieve", timed("retrieve", n.retrieve))
    g.add_node("generate", timed("generate", n.generate))
    g.add_node("finalize", timed("finalize", n.finalize))

    g.add_edge(START, "condense")
    g.add_edge("condense", "transform")
    g.add_edge("transform", "retrieve")
    last = "retrieve"

    if components.retrieval.has_postprocess:
        g.add_node("postprocess", timed("postprocess", n.postprocess))
        g.add_edge(last, "postprocess")
        last = "postprocess"

    if cfg.generation.grade_documents:
        g.add_node("grade", timed("grade", n.grade))
        g.add_edge(last, "grade")
        g.add_conditional_edges(
            "grade", n.route_after_grade, {"transform": "transform", "generate": "generate"}
        )
    else:
        g.add_edge(last, "generate")

    if cfg.generation.self_check:
        g.add_node("self_check", timed("self_check", n.self_check))
        g.add_edge("generate", "self_check")
        g.add_conditional_edges(
            "self_check", n.route_after_check, {"generate": "generate", "finalize": "finalize"}
        )
    else:
        g.add_edge("generate", "finalize")

    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
