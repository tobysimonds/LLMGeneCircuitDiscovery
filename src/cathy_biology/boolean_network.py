from __future__ import annotations

from itertools import combinations
from typing import Iterable

import networkx as nx

from cathy_biology.config import GrnConfig, SimulationConfig
from cathy_biology.models import GeneResearchResult, KnockoutHit


def boss_node_name(target_oncogene: str) -> str:
    return f"{target_oncogene}_SIGNALING"


def build_regulatory_graph(
    deg_genes: Iterable[str],
    research_results: Iterable[GeneResearchResult],
    grn_config: GrnConfig,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    tracked_targets = [grn_config.target_oncogene, *grn_config.immediate_downstream_effectors]
    boss_node = boss_node_name(grn_config.target_oncogene)

    for gene in deg_genes:
        graph.add_node(gene, kind="deg", basal_state=1)
    for target in tracked_targets:
        graph.add_node(target, kind="pathway", basal_state=0)
        graph.add_edge(target, boss_node, sign=1, confidence=1.0, provenance="aggregate")
    graph.add_node(boss_node, kind="boss", basal_state=0)

    for result in research_results:
        for interaction in result.interactions:
            if interaction.interaction_type == 0:
                continue
            if interaction.confidence_score < grn_config.confidence_threshold:
                continue
            if interaction.target not in tracked_targets:
                continue
            graph.add_edge(
                interaction.source_gene,
                interaction.target,
                sign=interaction.interaction_type,
                confidence=interaction.confidence_score,
                pmids=interaction.pmid_citations,
                evidence_summary=interaction.evidence_summary,
            )
    return graph


def simulate_boolean_network(
    graph: nx.DiGraph,
    knocked_out_genes: set[str],
    simulation_config: SimulationConfig,
) -> tuple[dict[str, int], int]:
    state = {
        node: (0 if node in knocked_out_genes else int(data.get("basal_state", 0)))
        for node, data in graph.nodes(data=True)
    }
    for node, data in graph.nodes(data=True):
        if data.get("kind") == "deg":
            state[node] = 0 if node in knocked_out_genes else 1

    for step in range(1, simulation_config.max_iterations + 1):
        new_state = state.copy()
        for node, data in graph.nodes(data=True):
            if data.get("kind") == "deg":
                continue
            incoming = list(graph.in_edges(node, data=True))
            activator_on = any(state[source] == 1 and edge["sign"] == 1 for source, _, edge in incoming)
            inhibitor_on = any(state[source] == 1 and edge["sign"] == -1 for source, _, edge in incoming)
            new_state[node] = 1 if activator_on and not inhibitor_on else 0
        if new_state == state:
            return state, step
        state = new_state
    return state, simulation_config.max_iterations


def search_knockout_combinations(
    graph: nx.DiGraph,
    grn_config: GrnConfig,
    simulation_config: SimulationConfig,
) -> list[KnockoutHit]:
    boss_node = boss_node_name(grn_config.target_oncogene)
    baseline_state, _ = simulate_boolean_network(graph, set(), simulation_config)
    if baseline_state.get(boss_node, 0) == 0:
        return []

    deg_nodes = sorted(node for node, data in graph.nodes(data=True) if data.get("kind") == "deg")
    tracked_targets = [grn_config.target_oncogene, *grn_config.immediate_downstream_effectors]
    hits: list[KnockoutHit] = []
    minimal_hit_sets: list[frozenset[str]] = []

    for knockout_size in simulation_config.knockout_sizes:
        for combo in combinations(deg_nodes, knockout_size):
            state, steps = simulate_boolean_network(graph, set(combo), simulation_config)
            if state.get(boss_node, 0) == 0:
                combo_set = frozenset(combo)
                if any(existing_hit.issubset(combo_set) for existing_hit in minimal_hit_sets):
                    continue
                minimal_hit_sets.append(combo_set)
                pathway_nodes_off = [node for node in tracked_targets if state.get(node, 0) == 0]
                confidence_score = sum(
                    edge_data.get("confidence", 0.0)
                    for source, _, edge_data in graph.edges(data=True)
                    if source in combo
                )
                score = len(pathway_nodes_off) * 10.0 + confidence_score - knockout_size
                hits.append(
                    KnockoutHit(
                        knocked_out_genes=list(combo),
                        boss_node=boss_node,
                        boss_state=0,
                        pathway_nodes_off=pathway_nodes_off,
                        convergence_steps=steps,
                        score=score,
                    )
                )
    hits.sort(key=lambda item: (len(item.knocked_out_genes), -item.score, item.knocked_out_genes))
    return hits
