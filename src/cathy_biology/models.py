from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

NodeKind = Literal["deg", "intermediate", "pathway", "prior", "boss", "unknown"]


class DegResult(BaseModel):
    gene: str
    score: float
    log2_fold_change: float
    adjusted_pvalue: float
    ranking: int


class EvidenceClassScores(BaseModel):
    direct_mechanistic: float = Field(default=0.0, ge=0.0, le=1.0)
    pdac_specific: float = Field(default=0.0, ge=0.0, le=1.0)
    pancreas_relevant: float = Field(default=0.0, ge=0.0, le=1.0)
    review_supported: float = Field(default=0.0, ge=0.0, le=1.0)
    prior_supported: float = Field(default=0.0, ge=0.0, le=1.0)
    benchmark_supported: float = Field(default=0.0, ge=0.0, le=1.0)


class ResolvedEntity(BaseModel):
    canonical_symbol: str
    aliases: list[str] = Field(default_factory=list)
    entity_type: NodeKind = "unknown"
    sources: list[str] = Field(default_factory=list)


class GeneInteraction(BaseModel):
    source_gene: str
    target: str
    interaction_type: Literal[-1, 0, 1]
    pmid_citations: list[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_summary: str = ""
    source_type: NodeKind = "unknown"
    target_type: NodeKind = "unknown"
    mechanistic_depth: int = Field(default=1, ge=1)
    evidence_scores: EvidenceClassScores = Field(default_factory=EvidenceClassScores)
    provenance_sources: list[str] = Field(default_factory=list)
    prior_support_sources: list[str] = Field(default_factory=list)
    benchmark_support_score: float = 0.0


class GeneResearchResult(BaseModel):
    source_gene: str
    target_oncogene: str
    context: str
    interactions: list[GeneInteraction] = Field(default_factory=list)
    discovered_entities: list[ResolvedEntity] = Field(default_factory=list)
    alias_hints: dict[str, list[str]] = Field(default_factory=dict)
    no_direct_effect: bool = False
    no_supported_edges: bool = False
    queried_targets: list[str] = Field(default_factory=list)
    raw_model: str = ""
    phase: Literal["discovery", "verification", "heuristic"] = "verification"


class ResearchOutput(BaseModel):
    discovery_results: list[GeneResearchResult] = Field(default_factory=list)
    verification_results: list[GeneResearchResult] = Field(default_factory=list)


class PriorKnowledgeSummary(BaseModel):
    node_count: int
    edge_count: int
    source_counts: dict[str, int] = Field(default_factory=dict)
    nodes: list[ResolvedEntity] = Field(default_factory=list)
    edges: list[GeneInteraction] = Field(default_factory=list)


class KnockoutHit(BaseModel):
    knocked_out_genes: list[str]
    boss_node: str
    boss_state: int
    pathway_nodes_off: list[str]
    convergence_steps: int
    score: float
    support_score: float = 0.0
    benchmark_score: float = 0.0


class BenchmarkGeneResult(BaseModel):
    gene_symbol: str
    depmap_column: str | None = None
    rnai_depmap_column: str | None = None
    n_cell_lines: int = 0
    mean_gene_effect: float | None = None
    median_gene_effect: float | None = None
    min_gene_effect: float | None = None
    hit_rate: float = 0.0
    benchmark_hit: bool = False
    rnai_mean_gene_effect: float | None = None
    rnai_median_gene_effect: float | None = None
    rnai_hit_rate: float = 0.0
    driver_alignment_score: float = 0.0
    prior_pathway_hits: list[str] = Field(default_factory=list)
    combined_support_score: float = 0.0


class BenchmarkReport(BaseModel):
    release: str
    lineage_filter: list[str]
    primary_disease_filter: list[str]
    model_count: int
    stage: Literal["pre_simulation", "final"] = "final"
    rnai_release: str | None = None
    results: list[BenchmarkGeneResult]


class ResearchExecutionSummary(BaseModel):
    requested_backend: str
    configured_model: str
    parser_model: str
    total_genes: int
    result_model_counts: dict[str, int] = Field(default_factory=dict)
    fallback_gene_count: int = 0


class ExperimentResult(BaseModel):
    name: str
    description: str = ""
    graph_nodes: int
    graph_edges: int
    knockout_hits: list[KnockoutHit] = Field(default_factory=list)
    benchmark_report: BenchmarkReport
    pruned_genes: list[str] = Field(default_factory=list)
    score: float = 0.0
    selected: bool = False


class PipelineRunSummary(BaseModel):
    dataset_cells: int
    dataset_genes: int
    degs: list[DegResult]
    research_execution: ResearchExecutionSummary
    prior_knowledge: PriorKnowledgeSummary
    graph_nodes: int
    graph_edges: int
    knockout_hits: list[KnockoutHit]
    benchmark_report: BenchmarkReport
    experiment_results: list[ExperimentResult] = Field(default_factory=list)
    selected_experiment: str = ""
    output_dir: Path
