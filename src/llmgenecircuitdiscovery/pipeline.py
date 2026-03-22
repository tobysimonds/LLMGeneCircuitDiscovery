from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd

from llmgenecircuitdiscovery.boolean_network import (
    build_regulatory_graph,
    build_projected_deg_graph,
    build_projected_graph,
    prune_genes_from_graph,
    search_knockout_combinations,
)
from llmgenecircuitdiscovery.config import PipelineConfig, Settings
from llmgenecircuitdiscovery.datasets import load_dataset
from llmgenecircuitdiscovery.deg import compute_top_degs
from llmgenecircuitdiscovery.depmap import DepMapClient
from llmgenecircuitdiscovery.grn import AnthropicResearchClient, OpenAIResearchClient, PubMedHeuristicResearchClient, ResearchClient
from llmgenecircuitdiscovery.models import (
    BenchmarkReport,
    ExperimentResult,
    PipelineRunSummary,
    PriorKnowledgeSummary,
    ResearchExecutionSummary,
)
from llmgenecircuitdiscovery.priors import PriorKnowledgeBuilder
from llmgenecircuitdiscovery.render import render_circular_graph_png
from llmgenecircuitdiscovery.utils import ensure_directory, timestamped_output_dir, write_json


async def execute_pipeline(
    config: PipelineConfig,
    settings: Settings,
    output_dir: Path | None = None,
    research_client: ResearchClient | None = None,
    depmap_client: DepMapClient | None = None,
) -> PipelineRunSummary:
    run_output_dir = output_dir or timestamped_output_dir(settings.artifacts_dir, prefix="deg-network")
    ensure_directory(run_output_dir)

    adata = load_dataset(config.dataset, config.qc, settings)
    degs, processed_adata = compute_top_degs(adata, config.contrast, config.qc, config.deg)
    deg_table = pd.DataFrame([deg.model_dump() for deg in degs])
    deg_table.to_csv(run_output_dir / "top_degs.csv", index=False)
    write_json(run_output_dir / "top_degs.json", [deg.model_dump() for deg in degs])

    genes = [deg.gene.upper() for deg in degs]
    prior_builder = PriorKnowledgeBuilder(settings, run_output_dir / "prior_cache", cache_dir_alias_builder(run_output_dir / "prior_cache"))
    prior_knowledge = prior_builder.build(genes, config.grn)
    write_json(run_output_dir / "prior_knowledge.json", prior_knowledge.model_dump())

    if research_client is None:
        if config.grn.research_backend == "pubmed":
            research_client = PubMedHeuristicResearchClient(settings, run_output_dir / "pubmed_cache")
        elif config.grn.research_backend == "anthropic":
            research_client = AnthropicResearchClient(settings, run_output_dir / "anthropic_cache")
        elif settings.openai_api_key is not None:
            research_client = OpenAIResearchClient(settings, run_output_dir / "openai_cache")
        else:
            research_client = PubMedHeuristicResearchClient(settings, run_output_dir / "pubmed_cache")

    research_output = await research_client.research_genes(genes, genes, prior_knowledge, config.grn)
    write_json(run_output_dir / "discovery_interactions.json", [result.model_dump() for result in research_output.discovery_results])
    write_json(run_output_dir / "gene_interactions.json", [result.model_dump() for result in research_output.verification_results])
    analysis_results = research_output.verification_results if config.grn.enable_verification else research_output.discovery_results
    write_json(run_output_dir / "analysis_interactions.json", [result.model_dump() for result in analysis_results])
    model_counts = Counter(result.raw_model or "unknown" for result in analysis_results)
    research_execution = ResearchExecutionSummary(
        requested_backend=config.grn.research_backend,
        configured_model=config.grn.model,
        parser_model=config.grn.parser_model,
        total_genes=len(genes),
        result_model_counts=dict(sorted(model_counts.items())),
        fallback_gene_count=sum(count for model, count in model_counts.items() if model == "pubmed-heuristic"),
    )
    write_json(run_output_dir / "research_execution.json", research_execution.model_dump())

    benchmark_client = depmap_client or DepMapClient(settings, run_output_dir / "depmap_cache")
    driver_gene_set = {
        config.grn.target_oncogene.upper(),
        *(node.upper() for node in config.grn.immediate_downstream_effectors),
    }

    full_graph = build_regulatory_graph(
        genes,
        analysis_results,
        prior_knowledge,
        config.grn,
        config.simulation,
        include_prior_edges=True,
    )
    projected_graph = build_projected_graph(full_graph, genes, config.grn, config.simulation)
    prior_only_full_graph = build_regulatory_graph(
        genes,
        [],
        prior_knowledge,
        config.grn,
        config.simulation,
        include_prior_edges=True,
    )
    prior_only_deg_graph = build_projected_deg_graph(prior_only_full_graph, genes, config.grn, config.simulation)
    llm_deg_graph = build_projected_deg_graph(full_graph, genes, config.grn, config.simulation)
    write_json(run_output_dir / "regulatory_graph_full.json", nx.node_link_data(full_graph))
    write_json(run_output_dir / "regulatory_graph_projected.json", nx.node_link_data(projected_graph))
    write_json(run_output_dir / "deg_graph_prior_only.json", nx.node_link_data(prior_only_deg_graph))
    write_json(run_output_dir / "deg_graph_with_llm.json", nx.node_link_data(llm_deg_graph))
    render_circular_graph_png(
        prior_only_deg_graph,
        run_output_dir / "deg_graph_without_llm.png",
        title="Projected 50-node DEG Graph Without LLM Suggestions",
    )
    render_circular_graph_png(
        llm_deg_graph,
        run_output_dir / "deg_graph_with_llm.png",
        title="Projected 50-node DEG Graph With LLM Suggestions",
    )

    pre_simulation_genes = _benchmark_candidates_from_graph(projected_graph)
    pre_simulation_benchmark = benchmark_client.benchmark_genes(
        pre_simulation_genes,
        config.benchmark,
        stage="pre_simulation",
        prior_genes={node.canonical_symbol for node in prior_knowledge.nodes},
        driver_genes=driver_gene_set,
    )
    write_json(run_output_dir / "pre_simulation_benchmark.json", pre_simulation_benchmark.model_dump())
    support_scores = benchmark_client.support_scores(pre_simulation_benchmark)
    prunable_genes = benchmark_client.low_support_genes(pre_simulation_benchmark, config.benchmark)

    experiments = _run_experiments(
        genes=genes,
        research_results=analysis_results,
        prior_knowledge=prior_knowledge,
        config=config,
        benchmark_client=benchmark_client,
        support_scores=support_scores,
        prunable_genes=prunable_genes,
    )
    write_json(run_output_dir / "experiment_report.json", [experiment.model_dump() for experiment in experiments])
    selected = max(experiments, key=lambda experiment: experiment.score)
    selected.selected = True

    graph, _ = _graph_for_variant(
        selected.name,
        genes,
        analysis_results,
        prior_knowledge,
        config,
        prunable_genes,
    )
    write_json(run_output_dir / "regulatory_graph.json", nx.node_link_data(graph))
    write_json(run_output_dir / "knockout_hits.json", [hit.model_dump() for hit in selected.knockout_hits])
    write_json(run_output_dir / "benchmark_report.json", selected.benchmark_report.model_dump())

    summary = PipelineRunSummary(
        dataset_cells=int(processed_adata.n_obs),
        dataset_genes=int(processed_adata.n_vars),
        degs=degs,
        research_execution=research_execution,
        prior_knowledge=prior_knowledge,
        graph_nodes=graph.number_of_nodes(),
        graph_edges=graph.number_of_edges(),
        knockout_hits=selected.knockout_hits,
        benchmark_report=selected.benchmark_report,
        experiment_results=experiments,
        selected_experiment=selected.name,
        output_dir=run_output_dir,
    )
    write_json(run_output_dir / "summary.json", summary.model_dump())
    return summary


def _run_experiments(
    *,
    genes: list[str],
    research_results,
    prior_knowledge: PriorKnowledgeSummary,
    config: PipelineConfig,
    benchmark_client: DepMapClient,
    support_scores: dict[str, float],
    prunable_genes: list[str],
) -> list[ExperimentResult]:
    experiments: list[ExperimentResult] = []
    prior_gene_set = {node.canonical_symbol for node in prior_knowledge.nodes}
    driver_gene_set = {
        config.grn.target_oncogene.upper(),
        *(node.upper() for node in config.grn.immediate_downstream_effectors),
    }
    for variant in config.experiments.variants:
        graph, removed_genes = _graph_for_variant(variant, genes, research_results, prior_knowledge, config, prunable_genes)
        candidate_genes = sorted(node for node, data in graph.nodes(data=True) if data.get("kind") == "deg")
        knockout_hits = search_knockout_combinations(
            graph,
            config.grn,
            config.simulation,
            benchmark_support=support_scores,
            candidate_genes=candidate_genes,
        )
        benchmark_genes = _benchmark_candidates(genes, knockout_hits)
        benchmark_report = benchmark_client.benchmark_genes(
            benchmark_genes,
            config.benchmark,
            stage="final",
            prior_genes=prior_gene_set,
            driver_genes=driver_gene_set,
        )
        experiment = ExperimentResult(
            name=variant,
            description=_variant_description(variant),
            graph_nodes=graph.number_of_nodes(),
            graph_edges=graph.number_of_edges(),
            knockout_hits=knockout_hits,
            benchmark_report=benchmark_report,
            pruned_genes=removed_genes,
            score=_score_experiment(knockout_hits, benchmark_report),
            selected=False,
        )
        experiments.append(experiment)
    return experiments


def _graph_for_variant(
    variant: str,
    genes: list[str],
    research_results,
    prior_knowledge: PriorKnowledgeSummary,
    config: PipelineConfig,
    prunable_genes: list[str],
) -> tuple[nx.DiGraph, list[str]]:
    include_prior_edges = "priors" in variant
    full_graph = build_regulatory_graph(
        genes,
        research_results,
        prior_knowledge,
        config.grn,
        config.simulation,
        include_prior_edges=include_prior_edges,
    )
    graph = build_projected_graph(full_graph, genes, config.grn, config.simulation)
    removed_genes: list[str] = []
    if "pruned" in variant:
        removable = [gene for gene in prunable_genes if gene not in _protected_pruning_genes(config)]
        graph = prune_genes_from_graph(graph, removable)
        removed_genes = sorted(set(removable))
    return graph, removed_genes


def _protected_pruning_genes(config: PipelineConfig) -> set[str]:
    return {
        config.grn.target_oncogene.upper(),
        *(node.upper() for node in config.grn.immediate_downstream_effectors),
        *(node.upper() for node in config.grn.prior.seed_nodes),
    }


def _score_experiment(knockout_hits, benchmark_report: BenchmarkReport) -> float:
    if not knockout_hits:
        return -1_000.0
    best_hit = knockout_hits[0]
    benchmark_scores = {result.gene_symbol.upper(): result.combined_support_score for result in benchmark_report.results}
    benchmark_support = sum(benchmark_scores.get(gene.upper(), 0.0) for gene in best_hit.knocked_out_genes)
    benchmark_hits = sum(1 for result in benchmark_report.results if result.benchmark_hit)
    return benchmark_support + benchmark_hits * 2.0 - len(best_hit.knocked_out_genes)


def _variant_description(variant: str) -> str:
    descriptions = {
        "llm_verified_only": "Only verified LLM edges.",
        "llm_plus_priors": "Verified LLM edges plus curated pathway priors.",
        "llm_plus_priors_pruned": "Verified LLM edges, curated pathway priors, and pre-simulation DepMap pruning.",
    }
    return descriptions.get(variant, variant)


def _benchmark_candidates(genes: list[str], knockout_hits: list) -> list[str]:
    if knockout_hits:
        candidates = sorted({gene.upper() for hit in knockout_hits[:10] for gene in hit.knocked_out_genes})
        if candidates:
            return candidates
    return [gene.upper() for gene in genes[:20]]


def _benchmark_candidates_from_graph(graph: nx.DiGraph) -> list[str]:
    return sorted(
        node
        for node, data in graph.nodes(data=True)
        if data.get("kind") in {"deg", "intermediate", "prior"} and node.isupper()
    )


def cache_dir_alias_builder(cache_dir: Path):
    from llmgenecircuitdiscovery.aliases import GeneAliasResolver

    return GeneAliasResolver(cache_dir / "aliases")


def run_pipeline(
    config: PipelineConfig,
    settings: Settings,
    output_dir: Path | None = None,
    research_client: ResearchClient | None = None,
    depmap_client: DepMapClient | None = None,
) -> PipelineRunSummary:
    return asyncio.run(execute_pipeline(config, settings, output_dir, research_client, depmap_client))
