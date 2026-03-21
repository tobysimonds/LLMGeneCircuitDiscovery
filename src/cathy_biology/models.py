from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class DegResult(BaseModel):
    gene: str
    score: float
    log2_fold_change: float
    adjusted_pvalue: float
    ranking: int


class GeneInteraction(BaseModel):
    source_gene: str
    target: str
    interaction_type: Literal[-1, 0, 1]
    pmid_citations: list[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    evidence_summary: str = ""


class GeneResearchResult(BaseModel):
    source_gene: str
    target_oncogene: str
    context: str
    interactions: list[GeneInteraction] = Field(default_factory=list)
    no_direct_effect: bool = False
    queried_targets: list[str] = Field(default_factory=list)
    raw_model: str = ""


class KnockoutHit(BaseModel):
    knocked_out_genes: list[str]
    boss_node: str
    boss_state: int
    pathway_nodes_off: list[str]
    convergence_steps: int
    score: float


class BenchmarkGeneResult(BaseModel):
    gene_symbol: str
    depmap_column: str | None = None
    n_cell_lines: int = 0
    mean_gene_effect: float | None = None
    median_gene_effect: float | None = None
    min_gene_effect: float | None = None
    hit_rate: float = 0.0
    benchmark_hit: bool = False


class BenchmarkReport(BaseModel):
    release: str
    lineage_filter: list[str]
    primary_disease_filter: list[str]
    model_count: int
    results: list[BenchmarkGeneResult]


class PipelineRunSummary(BaseModel):
    dataset_cells: int
    dataset_genes: int
    degs: list[DegResult]
    graph_nodes: int
    graph_edges: int
    knockout_hits: list[KnockoutHit]
    benchmark_report: BenchmarkReport
    output_dir: Path
