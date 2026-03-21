from __future__ import annotations

from itertools import combinations
from typing import Iterable

import networkx as nx

from cathy_biology.config import GrnConfig, SimulationConfig
from cathy_biology.models import GeneResearchResult, KnockoutHit, PriorKnowledgeSummary


def boss_node_name(target_oncogene: str) -> str:
    return f"{target_oncogene}_SIGNALING"


def build_regulatory_graph(
    deg_genes: Iterable[str],
    research_results: Iterable[GeneResearchResult],
    prior_knowledge: PriorKnowledgeSummary,
    grn_config: GrnConfig,
    simulation_config: SimulationConfig,
    include_prior_edges: bool = True,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    deg_set = {gene.upper() for gene in deg_genes}
    tracked_targets = [grn_config.target_oncogene.upper(), *(node.upper() for node in grn_config.immediate_downstream_effectors)]
    boss_node = boss_node_name(grn_config.target_oncogene.upper())

    for gene in sorted(deg_set):
        graph.add_node(
            gene,
            kind="deg",
            basal_state=1,
            logic_mode="source",
            activation_threshold=1.0,
            inhibition_dominance=simulation_config.inhibition_dominance,
        )
    for node in prior_knowledge.nodes:
        if node.canonical_symbol in graph:
            continue
        graph.add_node(
            node.canonical_symbol,
            kind=node.entity_type if node.entity_type != "unknown" else "prior",
            basal_state=0,
            logic_mode="weighted_or",
            activation_threshold=simulation_config.intermediate_activation_threshold,
            inhibition_dominance=simulation_config.inhibition_dominance,
            sources=node.sources,
        )
    for target in tracked_targets:
        graph.add_node(
            target,
            kind="pathway",
            basal_state=0,
            logic_mode="weighted_or" if not simulation_config.require_multiple_support_for_pathway else "weighted_and",
            activation_threshold=simulation_config.activation_threshold,
            inhibition_dominance=simulation_config.inhibition_dominance,
        )
        graph.add_edge(target, boss_node, sign=1, weight=1.0, confidence=1.0, provenance=["aggregate"])
    graph.add_node(
        boss_node,
        kind="boss",
        basal_state=0,
        logic_mode="weighted_or",
        activation_threshold=simulation_config.activation_threshold,
        inhibition_dominance=simulation_config.inhibition_dominance,
    )

    if include_prior_edges:
        for edge in prior_knowledge.edges:
            if edge.confidence_score < max(0.3, grn_config.confidence_threshold - 0.1):
                continue
            graph.add_edge(
                edge.source_gene,
                edge.target,
                sign=edge.interaction_type,
                weight=edge.confidence_score,
                confidence=edge.confidence_score,
                provenance=edge.provenance_sources,
                evidence_scores=edge.evidence_scores.model_dump(),
                benchmark_support_score=edge.benchmark_support_score,
            )

    for result in research_results:
        for interaction in result.interactions:
            if interaction.interaction_type == 0:
                continue
            if interaction.confidence_score < grn_config.verification_confidence_threshold:
                continue
            weight = compute_edge_weight(interaction)
            _ensure_node_defaults(
                graph,
                interaction.source_gene,
                kind=interaction.source_type,
                basal_state=int(interaction.source_type == "deg"),
                activation_threshold=simulation_config.intermediate_activation_threshold,
                inhibition_dominance=simulation_config.inhibition_dominance,
            )
            _ensure_node_defaults(
                graph,
                interaction.target,
                kind=interaction.target_type,
                basal_state=0,
                activation_threshold=simulation_config.intermediate_activation_threshold,
                inhibition_dominance=simulation_config.inhibition_dominance,
            )
            graph.add_edge(
                interaction.source_gene,
                interaction.target,
                sign=interaction.interaction_type,
                weight=weight,
                confidence=interaction.confidence_score,
                provenance=interaction.provenance_sources,
                evidence_scores=interaction.evidence_scores.model_dump(),
                benchmark_support_score=interaction.benchmark_support_score,
            )
    return graph


def compute_edge_weight(interaction) -> float:
    evidence = interaction.evidence_scores
    return min(
        1.5,
        interaction.confidence_score
        + 0.25 * evidence.direct_mechanistic
        + 0.15 * evidence.pdac_specific
        + 0.1 * evidence.pancreas_relevant
        + 0.05 * evidence.review_supported
        + 0.1 * evidence.prior_supported
        + 0.1 * evidence.benchmark_supported,
    )


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
            activation_score = sum(
                edge.get("weight", edge.get("confidence", 0.0))
                for source, _, edge in incoming
                if edge.get("sign") == 1 and state.get(source, 0) == 1
            )
            inhibition_score = sum(
                edge.get("weight", edge.get("confidence", 0.0))
                for source, _, edge in incoming
                if edge.get("sign") == -1 and state.get(source, 0) == 1
            )
            activator_count = sum(
                1 for source, _, edge in incoming if edge.get("sign") == 1 and state.get(source, 0) == 1
            )
            logic_mode = data.get("logic_mode", "weighted_or")
            activation_threshold = float(
                data.get(
                    "activation_threshold",
                    simulation_config.activation_threshold if data.get("kind") == "pathway" else simulation_config.intermediate_activation_threshold,
                )
            )
            inhibition_dominance = float(data.get("inhibition_dominance", simulation_config.inhibition_dominance))
            if inhibition_score >= simulation_config.inhibition_threshold and inhibition_score >= activation_score * inhibition_dominance:
                new_state[node] = 0
                continue
            if logic_mode == "weighted_and":
                required_support = 2 if activator_count > 1 else 1
                new_state[node] = int(activator_count >= required_support and activation_score >= activation_threshold)
            else:
                new_state[node] = int(activation_score >= activation_threshold and activation_score >= inhibition_score * inhibition_dominance)
        if new_state == state:
            return state, step
        state = new_state
    return state, simulation_config.max_iterations


def prune_genes_from_graph(graph: nx.DiGraph, genes_to_remove: Iterable[str]) -> nx.DiGraph:
    pruned = graph.copy()
    pruned.remove_nodes_from([gene for gene in genes_to_remove if gene in pruned])
    return pruned


def search_knockout_combinations(
    graph: nx.DiGraph,
    grn_config: GrnConfig,
    simulation_config: SimulationConfig,
    benchmark_support: dict[str, float] | None = None,
    candidate_genes: list[str] | None = None,
) -> list[KnockoutHit]:
    boss_node = boss_node_name(grn_config.target_oncogene.upper())
    baseline_state, _ = simulate_boolean_network(graph, set(), simulation_config)
    if baseline_state.get(boss_node, 0) == 0:
        return []

    deg_nodes = candidate_genes or sorted(node for node, data in graph.nodes(data=True) if data.get("kind") == "deg")
    tracked_targets = [grn_config.target_oncogene.upper(), *(node.upper() for node in grn_config.immediate_downstream_effectors)]
    hits: list[KnockoutHit] = []
    minimal_hit_sets: list[frozenset[str]] = []
    benchmark_support = benchmark_support or {}

    for knockout_size in simulation_config.knockout_sizes:
        for combo in combinations(deg_nodes, knockout_size):
            state, steps = simulate_boolean_network(graph, set(combo), simulation_config)
            if state.get(boss_node, 0) != 0:
                continue
            combo_set = frozenset(combo)
            if any(existing_hit.issubset(combo_set) for existing_hit in minimal_hit_sets):
                continue
            minimal_hit_sets.append(combo_set)
            pathway_nodes_off = [node for node in tracked_targets if state.get(node, 0) == 0]
            edge_support_score = sum(
                edge_data.get("confidence", 0.0)
                for source, _, edge_data in graph.edges(data=True)
                if source in combo
            )
            benchmark_score = sum(benchmark_support.get(gene, 0.0) for gene in combo)
            score = len(pathway_nodes_off) * 10.0 + edge_support_score + benchmark_score - knockout_size
            hits.append(
                KnockoutHit(
                    knocked_out_genes=list(combo),
                    boss_node=boss_node,
                    boss_state=0,
                    pathway_nodes_off=pathway_nodes_off,
                    convergence_steps=steps,
                    score=score,
                    support_score=edge_support_score,
                    benchmark_score=benchmark_score,
                )
            )
    hits.sort(key=lambda item: (len(item.knocked_out_genes), -item.score, item.knocked_out_genes))
    return hits


def _ensure_node_defaults(
    graph: nx.DiGraph,
    node: str,
    *,
    kind: str,
    basal_state: int,
    activation_threshold: float,
    inhibition_dominance: float,
) -> None:
    if node not in graph:
        graph.add_node(
            node,
            kind=kind,
            basal_state=basal_state,
            logic_mode="weighted_or",
            activation_threshold=activation_threshold,
            inhibition_dominance=inhibition_dominance,
        )
        return
    data = graph.nodes[node]
    if data.get("kind") in {None, "unknown"} and kind != "unknown":
        data["kind"] = kind
    if "basal_state" not in data:
        data["basal_state"] = basal_state
    data.setdefault("logic_mode", "weighted_or")
    data.setdefault("activation_threshold", activation_threshold)
    data.setdefault("inhibition_dominance", inhibition_dominance)
