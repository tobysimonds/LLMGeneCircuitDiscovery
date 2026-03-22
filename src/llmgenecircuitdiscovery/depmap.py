from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd

from llmgenecircuitdiscovery.config import BenchmarkConfig, Settings
from llmgenecircuitdiscovery.models import BenchmarkGeneResult, BenchmarkReport
from llmgenecircuitdiscovery.utils import ensure_directory


class DepMapClient:
    CATALOG_URL = "https://depmap.org/portal/api/download/files"
    DEFAULT_DRIVER_PRIORS = {
        "KRAS",
        "TP53",
        "CDKN2A",
        "SMAD4",
        "MYC",
        "EGFR",
        "ERBB2",
        "ERBB3",
        "SRC",
        "SOS1",
        "GRB2",
        "STAT3",
        "YAP1",
        "PIK3CA",
        "AKT1",
        "RAF1",
        "BRAF",
        "MAP2K1",
        "MAP2K2",
        "MAPK1",
        "MAPK3",
        "FGFR1",
        "FGFR2",
        "MET",
    }

    def __init__(self, settings: Settings, cache_dir: Path) -> None:
        self.settings = settings
        self.cache_dir = ensure_directory(cache_dir)

    def benchmark_genes(
        self,
        genes: list[str],
        benchmark_config: BenchmarkConfig,
        *,
        stage: str = "final",
        prior_genes: set[str] | None = None,
        driver_genes: set[str] | None = None,
    ) -> BenchmarkReport:
        catalog = self._load_catalog()
        release = benchmark_config.release or self._latest_release(catalog)
        model_path = self._download_release_file(catalog, release, "Model.csv")
        effect_path = self._download_release_file(catalog, release, "CRISPRGeneEffect.csv")
        rnai_path = None
        if benchmark_config.rnai_release is not None:
            try:
                rnai_path = self._download_release_file(catalog, benchmark_config.rnai_release, "D2_combined_gene_dep_scores.csv")
            except FileNotFoundError:
                rnai_path = None

        models = pd.read_csv(model_path)
        selected_models = self._filter_models(models, benchmark_config)
        crispr_frame, crispr_columns = self._load_effect_matrix(effect_path, genes, selected_models, models)
        rnai_frame, rnai_columns = (
            self._load_effect_matrix(rnai_path, genes, selected_models, models) if rnai_path is not None else (pd.DataFrame(), {})
        )

        pathway_priors = {gene.upper() for gene in (prior_genes or set())}
        driver_priors = {gene.upper() for gene in (driver_genes or set())} | self.DEFAULT_DRIVER_PRIORS

        results: list[BenchmarkGeneResult] = []
        for gene in genes:
            crispr_column = crispr_columns.get(gene.upper())
            rnai_column = rnai_columns.get(gene.upper())
            crispr_series = (
                pd.to_numeric(crispr_frame[crispr_column], errors="coerce").dropna() if crispr_column and crispr_column in crispr_frame else pd.Series(dtype=float)
            )
            rnai_series = (
                pd.to_numeric(rnai_frame[rnai_column], errors="coerce").dropna() if rnai_column and rnai_column in rnai_frame else pd.Series(dtype=float)
            )
            crispr_mean = float(crispr_series.mean()) if not crispr_series.empty else None
            rnai_mean = float(rnai_series.mean()) if not rnai_series.empty else None
            crispr_hit_rate = float((crispr_series <= benchmark_config.effect_threshold).mean()) if not crispr_series.empty else 0.0
            rnai_hit_rate = float((rnai_series <= benchmark_config.rnai_effect_threshold).mean()) if not rnai_series.empty else 0.0
            driver_alignment = 1.0 if gene.upper() in driver_priors else 0.0
            pathway_hits = [gene.upper()] if gene.upper() in pathway_priors else []
            combined_support = max(0.0, -(crispr_mean or 0.0)) + 0.5 * max(0.0, -(rnai_mean or 0.0)) + driver_alignment
            results.append(
                BenchmarkGeneResult(
                    gene_symbol=gene,
                    depmap_column=crispr_column,
                    rnai_depmap_column=rnai_column,
                    n_cell_lines=int(max(len(crispr_series), len(rnai_series))),
                    mean_gene_effect=crispr_mean,
                    median_gene_effect=float(crispr_series.median()) if not crispr_series.empty else None,
                    min_gene_effect=float(crispr_series.min()) if not crispr_series.empty else None,
                    hit_rate=crispr_hit_rate,
                    benchmark_hit=(crispr_mean is not None and crispr_mean <= benchmark_config.effect_threshold)
                    or (rnai_mean is not None and rnai_mean <= benchmark_config.rnai_effect_threshold),
                    rnai_mean_gene_effect=rnai_mean,
                    rnai_median_gene_effect=float(rnai_series.median()) if not rnai_series.empty else None,
                    rnai_hit_rate=rnai_hit_rate,
                    driver_alignment_score=driver_alignment,
                    prior_pathway_hits=pathway_hits,
                    combined_support_score=combined_support,
                )
            )
        return BenchmarkReport(
            release=release,
            lineage_filter=benchmark_config.lineage_filters,
            primary_disease_filter=benchmark_config.primary_disease_filters,
            model_count=int(selected_models.shape[0]),
            stage="pre_simulation" if stage == "pre_simulation" else "final",
            rnai_release=benchmark_config.rnai_release,
            results=results,
        )

    def support_scores(self, report: BenchmarkReport) -> dict[str, float]:
        return {result.gene_symbol.upper(): result.combined_support_score for result in report.results}

    def low_support_genes(self, report: BenchmarkReport, benchmark_config: BenchmarkConfig) -> list[str]:
        return [
            result.gene_symbol.upper()
            for result in report.results
            if result.combined_support_score <= benchmark_config.pre_simulation_prune_threshold
            and result.driver_alignment_score <= benchmark_config.min_driver_alignment_score
        ]

    def _load_catalog(self) -> pd.DataFrame:
        catalog_path = self.cache_dir / "download_catalog.csv"
        if not catalog_path.exists():
            self._download_file(self.CATALOG_URL, catalog_path)
        return pd.read_csv(catalog_path)

    def _latest_release(self, catalog: pd.DataFrame) -> str:
        public_catalog = catalog[catalog["release"].str.startswith("DepMap Public", na=False)].copy()
        public_catalog["release_date"] = pd.to_datetime(public_catalog["release_date"])
        latest_release = public_catalog.sort_values("release_date", ascending=False).iloc[0]["release"]
        return str(latest_release)

    def _download_release_file(self, catalog: pd.DataFrame, release: str, filename: str) -> Path:
        row = catalog[(catalog["release"] == release) & (catalog["filename"] == filename)]
        if row.empty:
            raise FileNotFoundError(f"Could not resolve {filename} for DepMap release {release}.")
        destination = self.cache_dir / release.replace(" ", "_").lower() / filename
        if not destination.exists():
            self._download_file(str(row.iloc[0]["url"]), destination)
        return destination

    def _download_file(self, url: str, destination: Path) -> None:
        ensure_directory(destination.parent)
        with httpx.Client(timeout=self.settings.request_timeout_seconds, follow_redirects=True) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with destination.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)

    def _load_effect_matrix(
        self,
        matrix_path: Path | None,
        genes: list[str],
        selected_models: pd.DataFrame,
        models: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        if matrix_path is None:
            return pd.DataFrame(), {}
        header = pd.read_csv(matrix_path, nrows=0)
        index_column = header.columns[0]
        symbol_to_column = {column.split(" (", 1)[0].upper(): column for column in header.columns[1:]}
        use_columns = [index_column] + [symbol_to_column[gene.upper()] for gene in genes if gene.upper() in symbol_to_column]
        if len(use_columns) == 1:
            return pd.DataFrame(), {}
        effect_frame = pd.read_csv(matrix_path, usecols=use_columns)
        effect_frame = effect_frame.rename(columns={index_column: "model_key"})
        effect_frame = self._align_effect_frame(effect_frame, selected_models, models)
        return effect_frame, symbol_to_column

    def _align_effect_frame(
        self,
        effect_frame: pd.DataFrame,
        selected_models: pd.DataFrame,
        models: pd.DataFrame,
    ) -> pd.DataFrame:
        candidate_columns = [column for column in ["ModelID", "DepMap_ID", "StrippedCellLineName", "CCLEName", "CCLE_Name"] if column in models.columns]
        best_column = None
        best_overlap = -1
        model_keys = set(effect_frame["model_key"].astype(str))
        for column in candidate_columns:
            overlap = len(model_keys & set(selected_models[column].astype(str)))
            if overlap > best_overlap:
                best_overlap = overlap
                best_column = column
        if best_column is None or best_overlap <= 0:
            return effect_frame.iloc[0:0].copy()
        aligned = effect_frame[effect_frame["model_key"].astype(str).isin(set(selected_models[best_column].astype(str)))]
        return aligned

    def _filter_models(self, models: pd.DataFrame, benchmark_config: BenchmarkConfig) -> pd.DataFrame:
        lineage_mask = pd.Series(False, index=models.index)
        disease_mask = pd.Series(False, index=models.index)
        for lineage in benchmark_config.lineage_filters:
            lineage_mask = lineage_mask | models["OncotreeLineage"].fillna("").str.contains(lineage, case=False)
        for disease in benchmark_config.primary_disease_filters:
            disease_mask = disease_mask | models["OncotreePrimaryDisease"].fillna("").str.contains(disease, case=False)
        selected = models[lineage_mask | disease_mask].copy()
        if selected.empty:
            return models.iloc[0:0].copy()
        return selected
