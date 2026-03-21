from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatasetConfig(BaseModel):
    source_type: Literal["geo", "h5ad", "scanpy_builtin", "mtx_bundle"] = "geo"
    accession: str | None = "GSE242230"
    path: Path | None = None
    builtin_name: str | None = None
    annotations_filename: str | None = None
    sample_limit: int | None = None
    cache_subdir: str = "dataset"


class ContrastConfig(BaseModel):
    groupby_column: str = "cell_type_specific"
    case_labels: list[str] = Field(default_factory=lambda: ["Malignant - Classical", "Malignant - Basal"])
    control_labels: list[str] = Field(default_factory=lambda: ["Normal Epithelial"])
    filter_column: str | None = "filtered"
    exclude_filtered: bool = True


class QcConfig(BaseModel):
    min_genes: int = 200
    max_mt_fraction: float = 0.2
    max_cells: int | None = None
    random_seed: int = 17


class DegConfig(BaseModel):
    top_n: int = 50
    adjusted_pvalue_cutoff: float = 0.05
    min_log2_fold_change: float = 0.0
    method: Literal["wilcoxon"] = "wilcoxon"


class PriorConfig(BaseModel):
    enabled: bool = True
    pathway_keywords: list[str] = Field(
        default_factory=lambda: [
            "Pancreatic cancer",
            "Signaling by EGFR",
            "ERBB2 signaling",
            "FGFR signaling",
            "RAS signaling",
            "MAPK signaling",
            "PI3K AKT signaling",
            "STAT3 signaling",
            "YAP1 signaling",
        ]
    )
    kegg_pathway_ids: list[str] = Field(
        default_factory=lambda: ["hsa05212", "hsa04010", "hsa04014", "hsa04015", "hsa04012", "hsa04151"]
    )
    seed_nodes: list[str] = Field(
        default_factory=lambda: [
            "EGFR",
            "ERBB2",
            "ERBB3",
            "FGFR1",
            "FGFR2",
            "FGFR3",
            "MET",
            "SOS1",
            "SOS2",
            "GRB2",
            "SHC1",
            "SRC",
            "PTPN11",
            "STAT3",
            "YAP1",
            "TEAD1",
            "MYC",
            "JUN",
            "FOS",
            "PRKCA",
            "PRKCD",
            "RHOA",
            "CDC42",
            "PLCG1",
            "PIK3R1",
        ]
    )
    pathwaycommons_pathways_per_keyword: int = 2
    reactome_pathways_per_keyword: int = 2
    reactome_events_per_pathway: int = 25
    prior_node_limit: int = 120
    omnipath_partner_chunk_size: int = 50


class GrnConfig(BaseModel):
    research_backend: Literal["openai", "anthropic", "pubmed"] = "openai"
    context: str = "Pancreatic Ductal Adenocarcinoma"
    target_oncogene: str = "KRAS"
    model: str = "o4-mini-deep-research"
    parser_model: str = "gpt-5-mini"
    concurrency: int = 4
    confidence_threshold: float = 0.35
    verification_confidence_threshold: float = 0.45
    max_tool_calls: int = 4
    discovery_max_edges_per_gene: int = 8
    allow_deg_to_deg_edges: bool = True
    allow_intermediate_nodes: bool = True
    immediate_downstream_effectors: list[str] = Field(
        default_factory=lambda: ["RAF1", "BRAF", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3", "PIK3CA", "AKT1"]
    )
    prior: PriorConfig = Field(default_factory=PriorConfig)


class SimulationConfig(BaseModel):
    knockout_sizes: list[int] = Field(default_factory=lambda: [1, 2, 3])
    max_iterations: int = 8
    activation_threshold: float = 0.55
    inhibition_threshold: float = 0.45
    inhibition_dominance: float = 0.9
    intermediate_activation_threshold: float = 0.5
    require_multiple_support_for_pathway: bool = False


class BenchmarkConfig(BaseModel):
    release: str | None = None
    rnai_release: str | None = "DEMETER2 Data v6"
    lineage_filters: list[str] = Field(default_factory=lambda: ["Pancreas"])
    primary_disease_filters: list[str] = Field(default_factory=lambda: ["Pancreatic"])
    effect_threshold: float = -0.5
    rnai_effect_threshold: float = -0.3
    pre_simulation_prune_threshold: float = 0.0
    min_driver_alignment_score: float = 0.0


class ExperimentConfig(BaseModel):
    enabled: bool = True
    variants: list[str] = Field(
        default_factory=lambda: ["llm_verified_only", "llm_plus_priors", "llm_plus_priors_pruned"]
    )


class PipelineConfig(BaseModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    contrast: ContrastConfig = Field(default_factory=ContrastConfig)
    qc: QcConfig = Field(default_factory=QcConfig)
    deg: DegConfig = Field(default_factory=DegConfig)
    grn: GrnConfig = Field(default_factory=GrnConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    experiments: ExperimentConfig = Field(default_factory=ExperimentConfig)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    request_timeout_seconds: int = 900


def load_pipeline_config(config_path: Path | None = None) -> PipelineConfig:
    if config_path is None:
        return PipelineConfig()
    import tomllib

    with config_path.open("rb") as handle:
        raw_config = tomllib.load(handle)
    return PipelineConfig.model_validate(raw_config)
