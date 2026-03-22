from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from llmgenecircuitdiscovery.utils import ensure_directory


def render_circular_graph_png(graph: nx.DiGraph, output_path: Path, *, title: str) -> Path:
    ensure_directory(output_path.parent)
    figure_size = max(14.0, min(26.0, 10.0 + graph.number_of_nodes() * 0.22))
    fig, ax = plt.subplots(figsize=(figure_size, figure_size), dpi=180)
    ax.set_title(title, fontsize=16, pad=18)
    ax.axis("off")

    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No edges available", ha="center", va="center", fontsize=14)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return output_path

    ordered_nodes = sorted(graph.nodes())
    positions = nx.circular_layout(ordered_nodes, scale=1.0)
    edge_colors = ["#198754" if data.get("sign", 1) > 0 else "#b02a37" for _, _, data in graph.edges(data=True)]
    edge_widths = [1.1 + 2.4 * float(data.get("confidence", 0.3)) for _, _, data in graph.edges(data=True)]

    nx.draw_networkx_edges(
        graph,
        pos=positions,
        ax=ax,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.72,
        arrows=False,
    )
    nx.draw_networkx_nodes(
        graph,
        pos=positions,
        ax=ax,
        node_color="#f7d38a",
        node_size=820,
        edgecolors="#3a2d0b",
        linewidths=1.1,
    )
    nx.draw_networkx_labels(
        graph,
        pos=positions,
        ax=ax,
        labels={node: node for node in ordered_nodes},
        font_size=7,
        font_weight="bold",
    )
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path
