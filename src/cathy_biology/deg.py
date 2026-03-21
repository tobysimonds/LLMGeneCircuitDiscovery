from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from cathy_biology.config import ContrastConfig, DegConfig, QcConfig
from cathy_biology.models import DegResult


def compute_top_degs(
    adata: AnnData,
    contrast_config: ContrastConfig,
    qc_config: QcConfig,
    deg_config: DegConfig,
) -> tuple[list[DegResult], AnnData]:
    working = adata.copy()
    if contrast_config.filter_column and contrast_config.exclude_filtered and contrast_config.filter_column in working.obs:
        filtered_mask = (
            working.obs[contrast_config.filter_column]
            .fillna(False)
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False})
            .fillna(False)
        )
        working = working[~filtered_mask].copy()

    if contrast_config.groupby_column not in working.obs:
        raise KeyError(f"Column `{contrast_config.groupby_column}` was not found in `adata.obs`.")

    group_values = working.obs[contrast_config.groupby_column].astype(str)
    case_labels = set(contrast_config.case_labels)
    control_labels = set(contrast_config.control_labels)
    mask = group_values.isin(case_labels | control_labels)
    working = working[mask].copy()
    working.obs["comparison_group"] = np.where(group_values[mask].isin(case_labels), "case", "control")

    sc.pp.filter_cells(working, min_genes=qc_config.min_genes)
    if qc_config.max_cells is not None and working.n_obs > qc_config.max_cells:
        sampled = np.random.default_rng(qc_config.random_seed).choice(working.obs_names, size=qc_config.max_cells, replace=False)
        working = working[sampled].copy()

    mt_mask = working.var_names.str.upper().str.startswith("MT-")
    if mt_mask.any():
        working.var["mt"] = np.asarray(mt_mask, dtype=bool)
        sc.pp.calculate_qc_metrics(working, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
        if qc_config.max_mt_fraction < 1.0:
            working = working[working.obs["pct_counts_mt"] <= qc_config.max_mt_fraction * 100].copy()

    sc.pp.normalize_total(working, target_sum=10_000)
    sc.pp.log1p(working)
    sc.tl.rank_genes_groups(
        working,
        groupby="comparison_group",
        groups=["case"],
        reference="control",
        method=deg_config.method,
    )

    deg_frame = sc.get.rank_genes_groups_df(working, group="case")
    deg_frame = deg_frame.rename(
        columns={
            "names": "gene",
            "scores": "score",
            "logfoldchanges": "log2_fold_change",
            "pvals_adj": "adjusted_pvalue",
        }
    )
    deg_frame = deg_frame.dropna(subset=["gene", "log2_fold_change", "adjusted_pvalue"])
    deg_frame = deg_frame[
        (deg_frame["adjusted_pvalue"] <= deg_config.adjusted_pvalue_cutoff)
        & (deg_frame["log2_fold_change"] >= deg_config.min_log2_fold_change)
    ]
    deg_frame = deg_frame.sort_values(["log2_fold_change", "score"], ascending=[False, False]).head(deg_config.top_n)

    degs = [
        DegResult(
            gene=row.gene,
            score=float(row.score),
            log2_fold_change=float(row.log2_fold_change),
            adjusted_pvalue=float(row.adjusted_pvalue),
            ranking=index + 1,
        )
        for index, row in enumerate(deg_frame.itertuples(index=False))
    ]
    return degs, working
