from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd

from cathy_biology.config import BenchmarkConfig, Settings
from cathy_biology.models import BenchmarkGeneResult, BenchmarkReport
from cathy_biology.utils import ensure_directory


class DepMapClient:
    CATALOG_URL = "https://depmap.org/portal/api/download/files"

    def __init__(self, settings: Settings, cache_dir: Path) -> None:
        self.settings = settings
        self.cache_dir = ensure_directory(cache_dir)

    def benchmark_genes(self, genes: list[str], benchmark_config: BenchmarkConfig) -> BenchmarkReport:
        catalog = self._load_catalog()
        release = benchmark_config.release or self._latest_release(catalog)
        model_path = self._download_release_file(catalog, release, "Model.csv")
        effect_path = self._download_release_file(catalog, release, "CRISPRGeneEffect.csv")

        models = pd.read_csv(model_path)
        selected_models = self._filter_models(models, benchmark_config)

        header = pd.read_csv(effect_path, nrows=0)
        index_column = header.columns[0]
        symbol_to_column = {
            column.split(" (", 1)[0].upper(): column
            for column in header.columns[1:]
        }

        use_columns = [index_column]
        for gene in genes:
            depmap_column = symbol_to_column.get(gene.upper())
            if depmap_column is not None:
                use_columns.append(depmap_column)
        if len(use_columns) == 1:
            return BenchmarkReport(
                release=release,
                lineage_filter=benchmark_config.lineage_filters,
                primary_disease_filter=benchmark_config.primary_disease_filters,
                model_count=int(selected_models.shape[0]),
                results=[],
            )

        effect_frame = pd.read_csv(effect_path, usecols=use_columns)
        effect_frame = effect_frame.rename(columns={index_column: "ModelID"})
        effect_frame = effect_frame[effect_frame["ModelID"].isin(selected_models["ModelID"])]

        results: list[BenchmarkGeneResult] = []
        for gene in genes:
            depmap_column = symbol_to_column.get(gene.upper())
            if depmap_column is None:
                results.append(BenchmarkGeneResult(gene_symbol=gene))
                continue
            gene_series = pd.to_numeric(effect_frame[depmap_column], errors="coerce").dropna()
            if gene_series.empty:
                results.append(BenchmarkGeneResult(gene_symbol=gene, depmap_column=depmap_column))
                continue
            hit_rate = float((gene_series <= benchmark_config.effect_threshold).mean())
            results.append(
                BenchmarkGeneResult(
                    gene_symbol=gene,
                    depmap_column=depmap_column,
                    n_cell_lines=int(gene_series.shape[0]),
                    mean_gene_effect=float(gene_series.mean()),
                    median_gene_effect=float(gene_series.median()),
                    min_gene_effect=float(gene_series.min()),
                    hit_rate=hit_rate,
                    benchmark_hit=float(gene_series.mean()) <= benchmark_config.effect_threshold,
                )
            )
        return BenchmarkReport(
            release=release,
            lineage_filter=benchmark_config.lineage_filters,
            primary_disease_filter=benchmark_config.primary_disease_filters,
            model_count=int(selected_models.shape[0]),
            results=results,
        )

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
