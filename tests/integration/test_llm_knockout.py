from __future__ import annotations

import json
from pathlib import Path

from llmgenecircuitdiscovery.llm_knockout import LlmKnockoutRanking, build_knockout_user_prompt, load_run_context


def test_load_run_context_and_prompt(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    (run_dir / "summary.json").write_text(json.dumps({"selected_experiment": "llm_plus_priors_pruned"}), encoding="utf-8")
    (run_dir / "regulatory_graph.json").write_text(
        json.dumps(
            {
                "nodes": [{"id": "EFS"}, {"id": "KRAS_SIGNALING"}],
                "edges": [{"source": "EFS", "target": "KRAS_SIGNALING", "sign": -1}],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "benchmark_report.json").write_text(
        json.dumps({"results": [{"gene_symbol": "EFS", "mean_gene_effect": -0.1}], "model_count": 10}),
        encoding="utf-8",
    )
    (run_dir / "top_degs.json").write_text(json.dumps([{"gene": "EFS", "log2_fold_change": 4.2}]), encoding="utf-8")
    (run_dir / "analysis_interactions.json").write_text(
        json.dumps(
            [
                {
                    "source_gene": "EFS",
                    "interactions": [
                        {
                            "source_gene": "EFS",
                            "target": "KRAS_SIGNALING",
                            "interaction_type": -1,
                            "confidence_score": 0.9,
                            "evidence_summary": "EFS suppresses the boss node in the toy graph.",
                            "source_refs": ["PMID:1"],
                            "pmid_citations": ["1"],
                            "provenance_sources": ["claude-sonnet-4-6"],
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "knockout_hits.json").write_text(
        json.dumps([{"knocked_out_genes": ["EFS"], "boss_state": 0, "score": 99.0}]),
        encoding="utf-8",
    )

    context = load_run_context(run_dir)
    prompt = build_knockout_user_prompt(context)

    assert context["selected_experiment"] == "llm_plus_priors_pruned"
    assert context["solver_knockout_hits"][0]["knocked_out_genes"] == ["EFS"]
    assert "Context JSON" in prompt
    assert "\"selected_experiment\": \"llm_plus_priors_pruned\"" in prompt


def test_llm_knockout_ranking_validates_structured_payload() -> None:
    payload = {
        "model": "claude-opus-4-6",
        "run_dir": "/tmp/run",
        "target_oncogene": "KRAS",
        "graph_variant": "selected",
        "final_recommendation": ["EFS"],
        "recommendation_summary": "EFS is the strongest mechanistic single-gene knockout in the supplied graph.",
        "candidates": [
            {
                "rank": 1,
                "knocked_out_genes": ["EFS"],
                "confidence_score": 0.62,
                "rationale": "EFS sits upstream of the collapse path.",
                "pathway_nodes_expected_off": ["KRAS"],
                "supporting_edges": ["EFS -> SRC"],
                "evidence_refs": ["PMID:123"],
                "benchmark_assessment": "Weak benchmark support.",
                "toxicity_risk": "unknown",
            }
        ],
    }

    ranking = LlmKnockoutRanking.model_validate(payload)

    assert ranking.final_recommendation == ["EFS"]
    assert ranking.candidates[0].confidence_score == 0.62
