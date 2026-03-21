from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd

from cathy_biology.boolean_network import build_regulatory_graph, search_knockout_combinations
from cathy_biology.config import PipelineConfig, Settings
from cathy_biology.datasets import load_dataset
from cathy_biology.deg import compute_top_degs
from cathy_biology.depmap import DepMapClient
from cathy_biology.grn import AnthropicResearchClient, OpenAIResearchClient, PubMedHeuristicResearchClient, ResearchClient
from cathy_biology.models import BenchmarkReport, PipelineRunSummary, ResearchExecutionSummary
from cathy_biology.utils import ensure_directory, timestamped_output_dir, write_json


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

    genes = [deg.gene for deg in degs]
    if research_client is None:
        if config.grn.research_backend == "pubmed":
            research_client = PubMedHeuristicResearchClient(settings, run_output_dir / "pubmed_cache")
        elif config.grn.research_backend == "anthropic":
            research_client = AnthropicResearchClient(settings, run_output_dir / "anthropic_cache")
        elif settings.openai_api_key is not None:
            research_client = OpenAIResearchClient(settings, run_output_dir / "openai_cache")
        else:
            research_client = PubMedHeuristicResearchClient(settings, run_output_dir / "pubmed_cache")
    research_results = await research_client.research_genes(genes, config.grn)
    write_json(run_output_dir / "gene_interactions.json", [result.model_dump() for result in research_results])
    model_counts = Counter(result.raw_model or "unknown" for result in research_results)
    research_execution = ResearchExecutionSummary(
        requested_backend=config.grn.research_backend,
        configured_model=config.grn.model,
        parser_model=config.grn.parser_model,
        total_genes=len(genes),
        result_model_counts=dict(sorted(model_counts.items())),
        fallback_gene_count=sum(count for model, count in model_counts.items() if model == "pubmed-heuristic"),
    )
    write_json(run_output_dir / "research_execution.json", research_execution.model_dump())

    graph = build_regulatory_graph(genes, research_results, config.grn)
    graph_payload = nx.node_link_data(graph)
    write_json(run_output_dir / "regulatory_graph.json", graph_payload)

    knockout_hits = search_knockout_combinations(graph, config.grn, config.simulation)
    write_json(run_output_dir / "knockout_hits.json", [hit.model_dump() for hit in knockout_hits])

    benchmark_genes = _benchmark_candidates(genes, knockout_hits)
    benchmark_client = depmap_client or DepMapClient(settings, run_output_dir / "depmap_cache")
    benchmark_report = benchmark_client.benchmark_genes(benchmark_genes, config.benchmark)
    write_json(run_output_dir / "benchmark_report.json", benchmark_report.model_dump())

    summary = PipelineRunSummary(
        dataset_cells=int(processed_adata.n_obs),
        dataset_genes=int(processed_adata.n_vars),
        degs=degs,
        research_execution=research_execution,
        graph_nodes=graph.number_of_nodes(),
        graph_edges=graph.number_of_edges(),
        knockout_hits=knockout_hits,
        benchmark_report=benchmark_report,
        output_dir=run_output_dir,
    )
    write_json(run_output_dir / "summary.json", summary.model_dump())
    return summary


def _benchmark_candidates(genes: list[str], knockout_hits: list) -> list[str]:
    if knockout_hits:
        candidates = sorted({gene for hit in knockout_hits[:10] for gene in hit.knocked_out_genes})
        if candidates:
            return candidates
    return genes[:10]


def run_pipeline(
    config: PipelineConfig,
    settings: Settings,
    output_dir: Path | None = None,
    research_client: ResearchClient | None = None,
    depmap_client: DepMapClient | None = None,
) -> PipelineRunSummary:
    return asyncio.run(execute_pipeline(config, settings, output_dir, research_client, depmap_client))
