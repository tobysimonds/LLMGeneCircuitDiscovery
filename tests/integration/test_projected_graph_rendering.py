from __future__ import annotations

from pathlib import Path

import networkx as nx

from cathy_biology.boolean_network import build_projected_deg_graph
from cathy_biology.config import GrnConfig, SimulationConfig
from cathy_biology.render import render_circular_graph_png


def test_build_projected_deg_graph_collapses_hidden_intermediate_path() -> None:
    full_graph = nx.DiGraph()
    for node in ["A", "B", "C", "KRAS", "RAF1", "BRAF", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3", "PIK3CA", "AKT1"]:
        full_graph.add_node(node, kind="prior", basal_state=0, logic_mode="weighted_or", activation_threshold=0.5)
    full_graph.add_node("A", kind="deg", basal_state=1, logic_mode="source", activation_threshold=1.0)
    full_graph.add_node("C", kind="deg", basal_state=1, logic_mode="source", activation_threshold=1.0)
    full_graph.add_edge("A", "B", sign=1, weight=0.8, confidence=0.8, provenance=["test"])
    full_graph.add_edge("B", "C", sign=1, weight=0.75, confidence=0.75, provenance=["test"])

    projected = build_projected_deg_graph(full_graph, ["A", "C"], GrnConfig(), SimulationConfig())

    assert projected.has_edge("A", "C")
    edge = projected.edges["A", "C"]
    assert edge["collapsed_via"] == ["B"]
    assert edge["path_length"] == 2
    assert edge["sign"] == 1


def test_render_circular_graph_png_writes_png(tmp_path: Path) -> None:
    graph = nx.DiGraph()
    graph.add_node("A")
    graph.add_node("C")
    graph.add_edge("A", "C", sign=1, confidence=0.8)

    output_path = render_circular_graph_png(graph, tmp_path / "graph.png", title="Projected Graph")

    assert output_path.exists()
    assert output_path.stat().st_size > 0
