from __future__ import annotations

import json
from pathlib import Path

from llmgenecircuitdiscovery.site import build_results_site


def test_build_results_site_writes_interactive_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "dataset_cells": 10,
                "dataset_genes": 20,
                "graph_nodes": 3,
                "graph_edges": 2,
                "selected_experiment": "llm_plus_priors_pruned",
                "knockout_hits": [],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "top_degs.json").write_text(json.dumps([{"gene": "SOX2", "log2_fold_change": 2.1, "adjusted_pvalue": 0.001}]), encoding="utf-8")
    (run_dir / "analysis_interactions.json").write_text(
        json.dumps(
            [
                {
                    "source_gene": "SOX2",
                        "interactions": [
                            {
                                "source_gene": "SOX2",
                                "target": "CTNNB1",
                                "interaction_type": 1,
                                "confidence_score": 0.83,
                                "evidence_summary": "SOX2 increases CTNNB1 signaling.",
                                "pmid_citations": ["122"],
                                "source_refs": ["PMID:122"],
                                "provenance_sources": ["claude-sonnet-4-6"],
                                "source_type": "deg",
                                "target_type": "prior",
                                "mechanistic_depth": 1,
                                "evidence_scores": {},
                            },
                            {
                                "source_gene": "CTNNB1",
                                "target": "MYC",
                                "interaction_type": 1,
                                "confidence_score": 0.8,
                                "evidence_summary": "CTNNB1 upregulates MYC.",
                                "pmid_citations": ["123"],
                                "source_refs": ["PMID:123"],
                                "provenance_sources": ["claude-sonnet-4-6"],
                                "source_type": "prior",
                                "target_type": "prior",
                                "mechanistic_depth": 1,
                                "evidence_scores": {},
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    full_graph_payload = {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": [
            {"id": "SOX2", "kind": "deg", "basal_state": 1, "logic_mode": "source"},
            {"id": "CTNNB1", "kind": "prior", "basal_state": 0, "logic_mode": "weighted_or"},
            {"id": "MYC", "kind": "prior", "basal_state": 0, "logic_mode": "weighted_or"},
        ],
        "edges": [
            {
                "source": "SOX2",
                "target": "CTNNB1",
                "sign": 1,
                "weight": 0.8,
                "confidence": 0.8,
                "provenance": ["claude-sonnet-4-6"],
            },
            {
                "source": "CTNNB1",
                "target": "MYC",
                "sign": 1,
                "weight": 0.7,
                "confidence": 0.7,
                "provenance": ["OmniPath"],
            }
        ],
    }
    projected_payload = {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": [
            {"id": "SOX2", "kind": "deg", "basal_state": 1, "logic_mode": "source"},
            {"id": "MYC", "kind": "prior", "basal_state": 0, "logic_mode": "weighted_or"},
        ],
        "edges": [
            {
                "source": "SOX2",
                "target": "MYC",
                "sign": 1,
                "weight": 0.65,
                "confidence": 0.65,
                "provenance": ["claude-sonnet-4-6", "OmniPath"],
                "collapsed_path": ["SOX2", "CTNNB1", "MYC"],
                "collapsed_via": ["CTNNB1"],
                "path_length": 2,
            }
        ],
    }
    (run_dir / "regulatory_graph_full.json").write_text(json.dumps(full_graph_payload), encoding="utf-8")
    for filename in [
        "regulatory_graph.json",
        "regulatory_graph_projected.json",
        "deg_graph_with_llm.json",
        "deg_graph_prior_only.json",
    ]:
        (run_dir / filename).write_text(json.dumps(projected_payload), encoding="utf-8")

    (run_dir / "knockout_hits.json").write_text(
        json.dumps(
            [
                {
                    "knocked_out_genes": ["SOX2"],
                    "boss_state": 0,
                    "pathway_nodes_off": ["KRAS"],
                    "score": 99.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    benchmark_payload = {
        "release": "test",
        "rnai_release": "test",
        "lineage_filter": ["Pancreas"],
        "primary_disease_filter": ["Pancreatic"],
        "model_count": 1,
        "stage": "final",
        "results": [
            {
                "gene_symbol": "SOX2",
                "mean_gene_effect": -0.5,
                "hit_rate": 1.0,
                "benchmark_hit": True,
            }
        ],
    }
    (run_dir / "benchmark_report.json").write_text(json.dumps(benchmark_payload), encoding="utf-8")
    (run_dir / "pre_simulation_benchmark.json").write_text(json.dumps(benchmark_payload | {"stage": "pre_simulation"}), encoding="utf-8")
    (run_dir / "experiment_report.json").write_text(json.dumps([]), encoding="utf-8")
    (run_dir / "research_execution.json").write_text(
        json.dumps(
            {
                "requested_backend": "anthropic",
                "configured_model": "claude-sonnet-4-6",
                "parser_model": "claude-opus-4-6",
                "total_genes": 1,
                "result_model_counts": {"claude-sonnet-4-6": 1},
                "fallback_gene_count": 0,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "prior_knowledge.json").write_text(
        json.dumps({"node_count": 1, "edge_count": 1, "source_counts": {"OmniPath": 1}}),
        encoding="utf-8",
    )

    site_dir = build_results_site(run_dir, tmp_path / "site")

    bundle = json.loads((site_dir / "data" / "primary" / "site_bundle.json").read_text(encoding="utf-8"))
    assert bundle["graphs"]["selected"]["nodes"][0]["id"] == "SOX2"
    assert bundle["graphs"]["selected"]["edges"][0]["step_evidence"][0]["direct_evidence"][0]["source_gene"] == "SOX2"
    assert bundle["node_profiles"]["SOX2"]["deg_stats"]["gene"] == "SOX2"
