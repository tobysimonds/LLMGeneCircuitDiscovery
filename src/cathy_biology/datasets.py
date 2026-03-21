from __future__ import annotations

import gzip
import tarfile
from pathlib import Path
from typing import Iterable

import anndata as ad
import httpx
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.io import mmread

from cathy_biology.config import DatasetConfig, QcConfig, Settings
from cathy_biology.utils import ensure_directory


def load_dataset(dataset_config: DatasetConfig, qc_config: QcConfig, settings: Settings) -> AnnData:
    cache_dir = ensure_directory(settings.data_dir / dataset_config.cache_subdir)
    if dataset_config.source_type == "geo":
        return load_geo_dataset(dataset_config, cache_dir, settings)
    if dataset_config.source_type == "h5ad":
        if dataset_config.path is None:
            raise ValueError("`dataset.path` is required for h5ad datasets.")
        return sc.read_h5ad(dataset_config.path)
    if dataset_config.source_type == "scanpy_builtin":
        return load_scanpy_builtin(dataset_config, qc_config)
    if dataset_config.source_type == "mtx_bundle":
        if dataset_config.path is None:
            raise ValueError("`dataset.path` is required for mtx_bundle datasets.")
        return load_mtx_bundle(dataset_config.path)
    raise ValueError(f"Unsupported dataset source: {dataset_config.source_type}")


def load_scanpy_builtin(dataset_config: DatasetConfig, qc_config: QcConfig) -> AnnData:
    builtin_name = dataset_config.builtin_name or "pbmc3k"
    if builtin_name != "pbmc3k":
        raise ValueError(f"Unsupported scanpy builtin dataset: {builtin_name}")
    adata = sc.datasets.pbmc3k()
    sc.pp.calculate_qc_metrics(adata, inplace=True, log1p=False)
    median_genes = float(adata.obs["n_genes_by_counts"].median())
    adata.obs["cohort"] = np.where(adata.obs["n_genes_by_counts"] >= median_genes, "case", "control")
    if qc_config.max_cells is not None and adata.n_obs > qc_config.max_cells:
        sampled = np.random.default_rng(qc_config.random_seed).choice(adata.obs_names, size=qc_config.max_cells, replace=False)
        adata = adata[sampled].copy()
    return adata


def load_geo_dataset(dataset_config: DatasetConfig, cache_dir: Path, settings: Settings) -> AnnData:
    accession = dataset_config.accession
    if accession is None:
        raise ValueError("`dataset.accession` is required for GEO datasets.")
    combined_path = cache_dir / f"{accession.lower()}_combined.h5ad"
    if combined_path.exists():
        return sc.read_h5ad(combined_path)

    suppl_dir = ensure_directory(cache_dir / "suppl")
    raw_dir = ensure_directory(cache_dir / "raw")
    annotations_name = dataset_config.annotations_filename or f"{accession}_annotations.txt.gz"
    annotations_path = download_file(_geo_url(accession, annotations_name), suppl_dir / annotations_name, settings)
    raw_tar_path = download_file(_geo_url(accession, f"{accession}_RAW.tar"), suppl_dir / f"{accession}_RAW.tar", settings)
    if not any(raw_dir.glob("*_matrix.mtx.gz")):
        with tarfile.open(raw_tar_path, "r") as archive:
            archive.extractall(raw_dir)
    adata = load_mtx_bundle(raw_dir, sample_limit=dataset_config.sample_limit)
    annotations = pd.read_csv(annotations_path, sep="\t", compression="gzip")
    if "filtered" in annotations:
        annotations["filtered"] = _coerce_bool_series(annotations["filtered"])
    annotations = annotations.set_index("cell_id")
    adata = adata[adata.obs_names.isin(annotations.index)].copy()
    annotations = annotations.drop(columns=["sample_id"], errors="ignore")
    adata.obs = adata.obs.join(annotations, how="left")
    adata.write_h5ad(combined_path, compression="gzip")
    return adata


def load_mtx_bundle(bundle_path: Path, sample_limit: int | None = None) -> AnnData:
    bundle_path = Path(bundle_path)
    sample_ids = _discover_sample_ids(bundle_path)
    if sample_limit is not None:
        sample_ids = sample_ids[:sample_limit]
    sample_adatas = [_read_sample_triplet(bundle_path, sample_id) for sample_id in sample_ids]
    if not sample_adatas:
        raise FileNotFoundError(f"No Matrix Market triplets were found in {bundle_path}.")
    combined = ad.concat(sample_adatas, join="outer", merge="same")
    combined.var_names_make_unique()
    return combined


def _discover_sample_ids(bundle_path: Path) -> list[str]:
    sample_ids: set[str] = set()
    for matrix_path in bundle_path.glob("*_matrix.mtx.gz"):
        sample_ids.add(_sample_id_from_filename(matrix_path.name))
    return sorted(sample_ids)


def _sample_id_from_filename(filename: str) -> str:
    suffixes = ("_matrix.mtx.gz", "_barcodes.tsv.gz", "_features.tsv.gz")
    prefix = filename
    for suffix in suffixes:
        if filename.endswith(suffix):
            prefix = filename[: -len(suffix)]
            break
    return prefix.split("_")[-1]


def _candidate_files(bundle_path: Path, sample_id: str, stem: str) -> Iterable[Path]:
    patterns = [f"*_{sample_id}_{stem}", f"{sample_id}_{stem}"]
    for pattern in patterns:
        yield from bundle_path.glob(pattern)


def _resolve_triplet_file(bundle_path: Path, sample_id: str, stem: str) -> Path:
    candidates = sorted(_candidate_files(bundle_path, sample_id, stem))
    if not candidates:
        raise FileNotFoundError(f"Missing {stem} file for sample {sample_id} in {bundle_path}.")
    return candidates[0]


def _read_sample_triplet(bundle_path: Path, sample_id: str) -> AnnData:
    matrix_path = _resolve_triplet_file(bundle_path, sample_id, "matrix.mtx.gz")
    barcode_path = _resolve_triplet_file(bundle_path, sample_id, "barcodes.tsv.gz")
    feature_path = _resolve_triplet_file(bundle_path, sample_id, "features.tsv.gz")

    with gzip.open(matrix_path, "rb") as handle:
        matrix = mmread(handle).tocsr().transpose().tocsr()
    barcodes = pd.read_csv(barcode_path, sep="\t", header=None, compression="gzip")[0].astype(str)
    features = pd.read_csv(feature_path, sep="\t", header=None, compression="gzip")
    gene_ids = features.iloc[:, 0].astype(str)
    gene_names = features.iloc[:, 1].astype(str) if features.shape[1] > 1 else gene_ids

    obs_names = [f"{sample_id}_{barcode}" for barcode in barcodes]
    obs = pd.DataFrame({"sample_id": sample_id}, index=obs_names)
    var = pd.DataFrame({"gene_id": gene_ids.to_list()}, index=pd.Index(gene_names.to_list(), name="gene_symbol"))
    adata = AnnData(matrix, obs=obs, var=var)
    adata.var_names_make_unique()
    return adata


def _geo_url(accession: str, filename: str) -> str:
    series_prefix = f"{accession[:-3]}nnn"
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{series_prefix}/{accession}/suppl/{filename}"


def download_file(url: str, destination: Path, settings: Settings) -> Path:
    if destination.exists():
        return destination
    ensure_directory(destination.parent)
    with httpx.Client(timeout=settings.request_timeout_seconds, follow_redirects=True) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with destination.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
    return destination


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
