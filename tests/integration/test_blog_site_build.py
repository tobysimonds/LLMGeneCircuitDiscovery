from __future__ import annotations

import json
from pathlib import Path

from cathy_biology.blog_site import build_blog_site


MINIMAL_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360606060000000050001a5f645400000000049454e44ae426082"
)


def test_build_blog_site_writes_post_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "llm_knockout_opus").mkdir()

    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "dataset_cells": 100,
                "dataset_genes": 500,
                "graph_nodes": 4,
                "graph_edges": 3,
                "selected_experiment": "llm_plus_priors_pruned",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "top_degs.json").write_text(
        json.dumps(
            [
                {"gene": "EFS", "log2_fold_change": 2.4, "adjusted_pvalue": 0.001},
                {"gene": "SOX2", "log2_fold_change": 2.1, "adjusted_pvalue": 0.002},
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "analysis_interactions.json").write_text(
        json.dumps(
            [
                {
                    "source_gene": "EFS",
                    "interactions": [
                        {
                            "source_gene": "EFS",
                            "target": "SOX2",
                            "interaction_type": 1,
                            "confidence_score": 0.81,
                            "evidence_summary": "EFS supports SOX2 signaling.",
                            "pmid_citations": ["12345"],
                            "source_refs": ["PMID:12345"],
                            "provenance_sources": ["claude-sonnet-4-6"],
                            "source_type": "deg",
                            "target_type": "deg",
                            "mechanistic_depth": 1,
                            "evidence_scores": {},
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    full_graph = {
        "nodes": [
            {"id": "EFS", "kind": "deg", "basal_state": 1, "logic_mode": "source"},
            {"id": "SRC", "kind": "intermediate", "basal_state": 0, "logic_mode": "weighted_or"},
            {"id": "SOX2", "kind": "deg", "basal_state": 1, "logic_mode": "weighted_or"},
            {"id": "KRAS_SIGNALING", "kind": "boss", "basal_state": 1, "logic_mode": "weighted_or"},
        ],
        "edges": [
            {"source": "EFS", "target": "SRC", "sign": 1, "weight": 0.8, "confidence": 0.8, "provenance": ["claude-sonnet-4-6"]},
            {"source": "SRC", "target": "SOX2", "sign": 1, "weight": 0.7, "confidence": 0.7, "provenance": ["OmniPath"]},
            {"source": "SOX2", "target": "KRAS_SIGNALING", "sign": 1, "weight": 0.9, "confidence": 0.9, "provenance": ["prior"]},
        ],
    }
    projected_graph = {
        "nodes": [
            {"id": "EFS", "kind": "deg", "basal_state": 1, "logic_mode": "source"},
            {"id": "SOX2", "kind": "deg", "basal_state": 1, "logic_mode": "weighted_or"},
            {"id": "KRAS_SIGNALING", "kind": "boss", "basal_state": 1, "logic_mode": "weighted_or"},
        ],
        "edges": [
            {
                "source": "EFS",
                "target": "SOX2",
                "sign": 1,
                "weight": 0.72,
                "confidence": 0.72,
                "provenance": ["claude-sonnet-4-6", "OmniPath"],
                "collapsed_via": ["SRC"],
                "collapsed_path": ["EFS", "SRC", "SOX2"],
                "path_length": 2,
            },
            {"source": "SOX2", "target": "KRAS_SIGNALING", "sign": 1, "weight": 0.9, "confidence": 0.9, "provenance": ["prior"]},
        ],
    }
    (run_dir / "regulatory_graph_full.json").write_text(json.dumps(full_graph), encoding="utf-8")
    for filename in [
        "regulatory_graph.json",
        "regulatory_graph_projected.json",
        "deg_graph_with_llm.json",
        "deg_graph_prior_only.json",
    ]:
        (run_dir / filename).write_text(json.dumps(projected_graph), encoding="utf-8")

    (run_dir / "benchmark_report.json").write_text(
        json.dumps(
            {
                "model_count": 5,
                "results": [
                    {"gene_symbol": "EFS", "mean_gene_effect": -0.06, "hit_rate": 0.0, "benchmark_hit": False},
                    {"gene_symbol": "SOX2", "mean_gene_effect": -0.21, "hit_rate": 0.2, "benchmark_hit": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "pre_simulation_benchmark.json").write_text(
        json.dumps({"model_count": 5, "results": []}),
        encoding="utf-8",
    )
    (run_dir / "experiment_report.json").write_text(json.dumps([{"name": "llm_plus_priors_pruned"}]), encoding="utf-8")
    (run_dir / "research_execution.json").write_text(
        json.dumps(
            {
                "result_model_counts": {"claude-sonnet-4-6": 2},
                "fallback_gene_count": 0,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "knockout_hits.json").write_text(
        json.dumps(
            [
                {
                    "knocked_out_genes": ["EFS"],
                    "score": 9.4,
                    "pathway_nodes_off": ["KRAS_SIGNALING"],
                }
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "llm_knockout_opus" / "rankings.json").write_text(
        json.dumps(
            {
                "final_recommendation": ["EFS"],
                "candidates": [
                    {
                        "rank": 1,
                        "knocked_out_genes": ["EFS"],
                        "confidence_score": 0.55,
                        "rationale": "EFS provides the cleanest single-gene intervention point.",
                        "benchmark_assessment": "Weak external support.",
                        "toxicity_risk": "unknown",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    (run_dir / "deg_graph_with_llm.png").write_bytes(MINIMAL_PNG)
    (run_dir / "deg_graph_without_llm.png").write_bytes(MINIMAL_PNG)

    site_dir = build_blog_site(run_dir, tmp_path / "blog-site")

    assert (site_dir / "index.html").exists()
    assert (site_dir / "styles.css").exists()
    assert (site_dir / "app.js").exists()
    bundle = json.loads((site_dir / "data" / "post_bundle.json").read_text(encoding="utf-8"))
    assert bundle["meta"]["authors"] == ["Cathy Liu", "Toby Simonds"]
    assert bundle["summary"]["solver_top_hit"] == ["EFS"]
    assert bundle["summary"]["opus_top_hit"] == ["EFS"]
    assert bundle["graphs"]["selected"]["edges"][0]["step_evidence"][0]["source"] == "EFS"
    assert bundle["llm_knockout"]["candidates"][0]["knocked_out_genes"] == ["EFS"]
