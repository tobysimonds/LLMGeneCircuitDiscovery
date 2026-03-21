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


class GrnConfig(BaseModel):
    research_backend: Literal["openai", "anthropic", "pubmed"] = "openai"
    context: str = "Pancreatic Ductal Adenocarcinoma"
    target_oncogene: str = "KRAS"
    model: str = "o4-mini-deep-research"
    parser_model: str = "gpt-5-mini"
    concurrency: int = 4
    confidence_threshold: float = 0.35
    max_tool_calls: int = 4
    immediate_downstream_effectors: list[str] = Field(
        default_factory=lambda: ["RAF1", "BRAF", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3", "PIK3CA", "AKT1"]
    )


class SimulationConfig(BaseModel):
    knockout_sizes: list[int] = Field(default_factory=lambda: [1, 2, 3])
    max_iterations: int = 8


class BenchmarkConfig(BaseModel):
    release: str | None = None
    lineage_filters: list[str] = Field(default_factory=lambda: ["Pancreas"])
    primary_disease_filters: list[str] = Field(default_factory=lambda: ["Pancreatic"])
    effect_threshold: float = -0.5


class PipelineConfig(BaseModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    contrast: ContrastConfig = Field(default_factory=ContrastConfig)
    qc: QcConfig = Field(default_factory=QcConfig)
    deg: DegConfig = Field(default_factory=DegConfig)
    grn: GrnConfig = Field(default_factory=GrnConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)


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
