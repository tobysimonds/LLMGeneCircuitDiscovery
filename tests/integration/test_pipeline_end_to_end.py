from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.io import mmwrite
from scipy.sparse import csr_matrix

from cathy_biology.config import PipelineConfig, Settings
from cathy_biology.depmap import DepMapClient
from cathy_biology.grn import MockResearchClient
from cathy_biology.models import BenchmarkGeneResult, BenchmarkReport, EvidenceClassScores, GeneInteraction
from cathy_biology.pipeline import run_pipeline


class StubDepMapClient(DepMapClient):
    def __init__(self) -> None:
        pass

    def benchmark_genes(self, genes: list[str], benchmark_config, *, stage="final", prior_genes=None, driver_genes=None):  # type: ignore[override]
        return BenchmarkReport(
            release="mock",
            lineage_filter=["Pancreas"],
            primary_disease_filter=["Pancreatic"],
            model_count=3,
            stage="pre_simulation" if stage == "pre_simulation" else "final",
            rnai_release="mock-rnai",
            results=[
                BenchmarkGeneResult(
                    gene_symbol=gene,
                    depmap_column=f"{gene} (mock)",
                    rnai_depmap_column=f"{gene} (mock-rnai)",
                    n_cell_lines=3,
                    mean_gene_effect=-0.75 if gene == "IFITM3" else -0.2,
                    median_gene_effect=-0.7 if gene == "IFITM3" else -0.1,
                    min_gene_effect=-1.1 if gene == "IFITM3" else -0.4,
                    hit_rate=1.0 if gene == "IFITM3" else 0.0,
                    benchmark_hit=gene == "IFITM3",
                    rnai_mean_gene_effect=-0.5 if gene == "IFITM3" else -0.1,
                    rnai_median_gene_effect=-0.45 if gene == "IFITM3" else -0.05,
                    rnai_hit_rate=1.0 if gene == "IFITM3" else 0.0,
                    driver_alignment_score=1.0 if gene == "IFITM3" else 0.0,
                    prior_pathway_hits=[gene] if gene == "IFITM3" else [],
                    combined_support_score=2.25 if gene == "IFITM3" else 0.3,
                )
                for gene in genes
            ],
        )


def test_pipeline_runs_end_to_end_with_h5ad(tmp_path: Path) -> None:
    genes = ["IFITM3", "KRT19", "EPCAM", "KRAS", "RAF1", "ACTB"]
    matrix = np.array(
        [
            [12, 10, 9, 0, 0, 10],
            [11, 11, 10, 0, 0, 10],
            [13, 9, 8, 0, 0, 10],
            [10, 10, 9, 0, 0, 10],
            [1, 1, 0, 0, 0, 10],
            [0, 2, 1, 0, 0, 10],
            [1, 1, 0, 0, 0, 10],
            [2, 2, 1, 0, 0, 10],
        ],
        dtype=float,
    )
    obs = pd.DataFrame(
        {"cell_type_specific": ["Malignant - Classical"] * 4 + ["Normal Epithelial"] * 4},
        index=[f"cell-{index}" for index in range(matrix.shape[0])],
    )
    adata = AnnData(X=csr_matrix(matrix), obs=obs, var=pd.DataFrame(index=genes))
    h5ad_path = tmp_path / "synthetic.h5ad"
    adata.write_h5ad(h5ad_path)

    config = PipelineConfig.model_validate(
        {
            "dataset": {"source_type": "h5ad", "path": str(h5ad_path)},
            "contrast": {
                "groupby_column": "cell_type_specific",
                "case_labels": ["Malignant - Classical"],
                "control_labels": ["Normal Epithelial"],
            },
            "qc": {"min_genes": 0, "max_mt_fraction": 1.0},
            "deg": {"top_n": 3, "adjusted_pvalue_cutoff": 1.0, "min_log2_fold_change": 0.0},
            "grn": {
                "target_oncogene": "KRAS",
                "model": "mock",
                "parser_model": "mock",
                "immediate_downstream_effectors": ["RAF1"],
                "prior": {"enabled": False},
            },
            "simulation": {"knockout_sizes": [1, 2, 3], "max_iterations": 5},
            "benchmark": {"release": "mock", "rnai_release": "mock-rnai"},
            "experiments": {"variants": ["llm_verified_only", "llm_plus_priors_pruned"]},
        }
    )
    settings = Settings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")
    research_client = MockResearchClient(
        mapping={
            "IFITM3": [
                GeneInteraction(
                    source_gene="IFITM3",
                    target="KRAS",
                    interaction_type=1,
                    confidence_score=0.8,
                    evidence_scores=EvidenceClassScores(direct_mechanistic=1.0, pdac_specific=1.0),
                )
            ],
            "KRT19": [
                GeneInteraction(
                    source_gene="KRT19",
                    target="RAF1",
                    interaction_type=1,
                    confidence_score=0.7,
                    evidence_scores=EvidenceClassScores(direct_mechanistic=0.8, pdac_specific=0.6),
                )
            ],
            "EPCAM": [
                GeneInteraction(
                    source_gene="EPCAM",
                    target="KRAS",
                    interaction_type=1,
                    confidence_score=0.6,
                    evidence_scores=EvidenceClassScores(direct_mechanistic=0.7, pdac_specific=0.6),
                )
            ],
        },
        target_oncogene="KRAS",
        context="PDAC",
    )

    summary = run_pipeline(config, settings, research_client=research_client, depmap_client=StubDepMapClient())

    assert summary.dataset_cells == 8
    assert len(summary.degs) == 3
    assert summary.research_execution.result_model_counts == {"mock": 3}
    assert summary.prior_knowledge.node_count == 0
    assert summary.selected_experiment in {"llm_verified_only", "llm_plus_priors_pruned"}
    assert summary.experiment_results
    assert summary.knockout_hits
    assert summary.benchmark_report.model_count == 3
    assert (summary.output_dir / "summary.json").exists()
    assert (summary.output_dir / "research_execution.json").exists()
    assert (summary.output_dir / "pre_simulation_benchmark.json").exists()
    assert (summary.output_dir / "experiment_report.json").exists()


def test_mtx_bundle_loader_integration(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _write_triplet(bundle_dir, "sampleA", np.array([[5, 0, 1], [1, 4, 1]], dtype=int))
    _write_triplet(bundle_dir, "sampleB", np.array([[6, 0, 1], [0, 5, 1]], dtype=int))

    annotations = pd.DataFrame(
        {
            "cell_id": ["sampleA_cell-1", "sampleA_cell-2", "sampleB_cell-1", "sampleB_cell-2"],
            "sample_id": ["sampleA", "sampleA", "sampleB", "sampleB"],
            "cell_type_specific": ["Malignant - Basal", "Malignant - Basal", "Normal Epithelial", "Normal Epithelial"],
            "filtered": [False, False, False, False],
        }
    )
    annotations_path = tmp_path / "annotations.tsv.gz"
    annotations.to_csv(annotations_path, sep="\t", index=False, compression="gzip")

    from cathy_biology.datasets import load_mtx_bundle

    adata = load_mtx_bundle(bundle_dir)
    adata.obs = adata.obs.join(annotations.set_index("cell_id"), how="left", rsuffix="_annotation")
    assert adata.n_obs == 4
    assert "sample_id" in adata.obs
    assert "geneA" in adata.var_names


def test_depmap_support_scoring_separates_driver_alignment_from_prior_membership(tmp_path: Path) -> None:
    release = "DepMap Public Test"
    rnai_release = "DEMETER2 Test"
    cache_dir = tmp_path / "depmap_cache"
    release_dir = cache_dir / release.replace(" ", "_").lower()
    rnai_dir = cache_dir / rnai_release.replace(" ", "_").lower()
    release_dir.mkdir(parents=True)
    rnai_dir.mkdir(parents=True)

    models = pd.DataFrame(
        {
            "ModelID": ["ACH-000001", "ACH-000002"],
            "OncotreeLineage": ["Pancreas", "Pancreas"],
            "OncotreePrimaryDisease": ["Pancreatic Adenocarcinoma", "Pancreatic Adenocarcinoma"],
        }
    )
    models.to_csv(release_dir / "Model.csv", index=False)
    pd.DataFrame(
        {
            "ModelID": ["ACH-000001", "ACH-000002"],
            "GENE1 (1001)": [0.2, 0.1],
            "KRAS (3845)": [-0.8, -0.7],
        }
    ).to_csv(release_dir / "CRISPRGeneEffect.csv", index=False)
    pd.DataFrame(
        {
            "ModelID": ["ACH-000001", "ACH-000002"],
            "GENE1 (1001)": [0.1, 0.05],
            "KRAS (3845)": [-0.4, -0.5],
        }
    ).to_csv(rnai_dir / "D2_combined_gene_dep_scores.csv", index=False)
    pd.DataFrame(
        [
            {"release": release, "release_date": "2026-01-01", "filename": "Model.csv", "url": "local://model"},
            {"release": release, "release_date": "2026-01-01", "filename": "CRISPRGeneEffect.csv", "url": "local://crispr"},
            {"release": rnai_release, "release_date": "2026-01-01", "filename": "D2_combined_gene_dep_scores.csv", "url": "local://rnai"},
        ]
    ).to_csv(cache_dir / "download_catalog.csv", index=False)

    client = DepMapClient(Settings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts"), cache_dir)
    config = PipelineConfig().benchmark.model_copy(update={"release": release, "rnai_release": rnai_release})
    report = client.benchmark_genes(
        ["GENE1", "KRAS"],
        config,
        prior_genes={"GENE1"},
        driver_genes={"KRAS"},
    )

    results = {result.gene_symbol: result for result in report.results}
    assert results["GENE1"].prior_pathway_hits == ["GENE1"]
    assert results["GENE1"].driver_alignment_score == 0.0
    assert results["GENE1"].combined_support_score == 0.0
    assert results["KRAS"].driver_alignment_score == 1.0
    assert results["KRAS"].combined_support_score > 1.0


def _write_triplet(bundle_dir: Path, sample_id: str, cell_by_gene: np.ndarray) -> None:
    gene_names = ["geneA", "geneB", "geneC"]
    matrix = csr_matrix(cell_by_gene).transpose()
    with gzip.open(bundle_dir / f"{sample_id}_matrix.mtx.gz", "wb") as handle:
        mmwrite(handle, matrix)
    with gzip.open(bundle_dir / f"{sample_id}_barcodes.tsv.gz", "wt", encoding="utf-8") as handle:
        for index in range(cell_by_gene.shape[0]):
            handle.write(f"cell-{index + 1}\n")
    with gzip.open(bundle_dir / f"{sample_id}_features.tsv.gz", "wt", encoding="utf-8") as handle:
        for gene_name in gene_names:
            handle.write(f"{gene_name}\t{gene_name}\n")
