from __future__ import annotations

import json
from pathlib import Path

from llmgenecircuitdiscovery.site import build_results_site


def test_build_results_site_generates_static_bundle(tmp_path: Path) -> None:
    primary_run = _write_run_bundle(tmp_path / "primary-run", run_name="primary", hit_genes=["GNGT1"])
    baseline_run = _write_run_bundle(tmp_path / "baseline-run", run_name="baseline", hit_genes=["FOXD1", "NTS"])
    output_dir = tmp_path / "site"

    site_dir = build_results_site(primary_run, output_dir, baseline_run_dir=baseline_run, title="PDAC Atlas")

    assert (site_dir / "index.html").exists()
    assert (site_dir / "styles.css").exists()
    assert (site_dir / "app.js").exists()
    manifest = json.loads((site_dir / "data" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["title"] == "PDAC Atlas"
    assert manifest["runs"]["primary"]["top_hit"] == ["GNGT1"]
    assert manifest["runs"]["baseline"]["top_hit"] == ["FOXD1", "NTS"]
    assert (site_dir / "data" / "primary" / "summary.json").exists()
    assert (site_dir / "data" / "baseline" / "benchmark_report.json").exists()


def _write_run_bundle(run_dir: Path, *, run_name: str, hit_genes: list[str]) -> Path:
    run_dir.mkdir(parents=True)
    summary = {
        "dataset_cells": 10164,
        "dataset_genes": 33538,
        "degs": [{"gene": "SOX2", "log2_fold_change": 28.6, "score": 9.7, "adjusted_pvalue": 1e-8, "ranking": 1}],
        "research_execution": {
            "requested_backend": "openai",
            "configured_model": "o4-mini-deep-research",
            "parser_model": "gpt-5-mini",
            "total_genes": 50,
            "result_model_counts": {"o4-mini-deep-research": 47, "gpt-5-mini": 47},
            "fallback_gene_count": 3,
        },
        "prior_knowledge": {"node_count": 120, "edge_count": 314, "nodes": [], "edges": [], "source_counts": {"KEGG": 10}},
        "graph_nodes": 121,
        "graph_edges": 323,
        "knockout_hits": [
            {
                "knocked_out_genes": hit_genes,
                "boss_node": "KRAS_SIGNALING",
                "boss_state": 0,
                "pathway_nodes_off": ["KRAS", "RAF1"],
                "convergence_steps": 2,
                "score": 90.0,
                "support_score": 1.1,
                "benchmark_score": 0.0,
            }
        ]
        if hit_genes
        else [],
        "benchmark_report": {
            "release": "DepMap Public Test",
            "lineage_filter": ["Pancreas"],
            "primary_disease_filter": ["Pancreatic"],
            "model_count": 68,
            "stage": "final",
            "rnai_release": "DEMETER2 Test",
            "results": [
                {
                    "gene_symbol": hit_genes[0] if hit_genes else "SOX2",
                    "mean_gene_effect": 0.14,
                    "hit_rate": 0.0,
                    "benchmark_hit": False,
                    "driver_alignment_score": 0.0,
                    "prior_pathway_hits": [hit_genes[0]] if hit_genes else ["SOX2"],
                    "combined_support_score": 0.0,
                }
            ],
        },
        "experiment_results": [],
        "selected_experiment": run_name,
        "output_dir": str(run_dir),
    }
    top_degs = summary["degs"]
    gene_interactions = [
        {
            "source_gene": "SOX2",
            "target_oncogene": "KRAS",
            "context": "PDAC",
            "interactions": [
                {
                    "source_gene": "SOX2",
                    "target": "EGFR",
                    "interaction_type": 1,
                    "pmid_citations": ["12345"],
                    "confidence_score": 0.72,
                    "evidence_summary": "SOX2 activates EGFR signaling in PDAC models.",
                    "source_type": "deg",
                    "target_type": "prior",
                    "mechanistic_depth": 1,
                    "evidence_scores": {
                        "direct_mechanistic": 0.8,
                        "pdac_specific": 0.7,
                        "pancreas_relevant": 0.7,
                        "review_supported": 0.5,
                        "prior_supported": 0.0,
                        "benchmark_supported": 0.0,
                    },
                    "provenance_sources": ["openai"],
                    "prior_support_sources": [],
                    "benchmark_support_score": 0.0,
                }
            ],
            "discovered_entities": [{"canonical_symbol": "SOX2", "aliases": ["SOX2"], "entity_type": "deg", "sources": ["seed-gene"]}],
            "alias_hints": {},
            "no_direct_effect": False,
            "no_supported_edges": False,
            "queried_targets": ["KRAS", "EGFR"],
            "raw_model": "o4-mini-deep-research",
            "phase": "verification",
        }
    ]
    regulatory_graph = {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": [
            {"id": "SOX2", "kind": "deg"},
            {"id": "EGFR", "kind": "prior"},
            {"id": "KRAS", "kind": "pathway"},
        ],
        "edges": [
            {"source": "SOX2", "target": "EGFR", "sign": 1, "weight": 0.72, "provenance": ["openai"]},
            {"source": "EGFR", "target": "KRAS", "sign": 1, "weight": 0.82, "provenance": ["prior"]},
        ],
    }
    knockout_hits = summary["knockout_hits"]
    benchmark_report = summary["benchmark_report"]
    experiment_report = [
        {
            "name": run_name,
            "description": "Synthetic run",
            "graph_nodes": 121,
            "graph_edges": 323,
            "knockout_hits": knockout_hits,
            "benchmark_report": benchmark_report,
            "pruned_genes": [],
            "score": -1.0,
            "selected": True,
        }
    ]
    research_execution = summary["research_execution"]
    prior_knowledge = summary["prior_knowledge"]

    bundle = {
        "summary.json": summary,
        "top_degs.json": top_degs,
        "analysis_interactions.json": gene_interactions,
        "regulatory_graph_full.json": regulatory_graph,
        "regulatory_graph.json": regulatory_graph,
        "regulatory_graph_projected.json": regulatory_graph,
        "deg_graph_with_llm.json": regulatory_graph,
        "deg_graph_prior_only.json": regulatory_graph,
        "knockout_hits.json": knockout_hits,
        "benchmark_report.json": benchmark_report,
        "pre_simulation_benchmark.json": benchmark_report | {"stage": "pre_simulation"},
        "experiment_report.json": experiment_report,
        "research_execution.json": research_execution,
        "prior_knowledge.json": prior_knowledge,
    }
    for filename, payload in bundle.items():
        (run_dir / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return run_dir
