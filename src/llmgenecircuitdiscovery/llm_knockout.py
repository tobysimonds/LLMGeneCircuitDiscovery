from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

from llmgenecircuitdiscovery.config import Settings
from llmgenecircuitdiscovery.grn import _anthropic_message_to_dict, _parse_json_payload
from llmgenecircuitdiscovery.utils import ensure_directory, write_json


class LlmKnockoutCandidate(BaseModel):
    rank: int
    knocked_out_genes: list[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    pathway_nodes_expected_off: list[str] = Field(default_factory=list)
    supporting_edges: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    benchmark_assessment: str = ""
    toxicity_risk: str = ""


class LlmKnockoutRanking(BaseModel):
    model: str
    run_dir: str
    target_oncogene: str
    graph_variant: str
    final_recommendation: list[str] = Field(default_factory=list)
    recommendation_summary: str = ""
    candidates: list[LlmKnockoutCandidate] = Field(default_factory=list)


def build_knockout_system_prompt() -> str:
    return (
        "You are a senior computational oncology strategy model.\n"
        "You are given a PDAC target-discovery run with a curated graph, LLM-derived edges, projected pathway structure, and DepMap benchmark rows.\n"
        "Your task is to recommend knockout targets using only the supplied artifact context.\n\n"
        "Decision rules:\n"
        "- Optimize for shutting down KRAS-centered oncogenic signaling.\n"
        "- Prefer smaller interventions: 1-gene first, then 2-gene, then 3-gene only if clearly justified.\n"
        "- Do not recommend synthetic nodes such as KRAS_SIGNALING.\n"
        "- Prefer nodes that are actual genes/proteins in the graph.\n"
        "- Use the benchmark rows as a reality check. Weak benchmark support should lower confidence and should be stated explicitly.\n"
        "- Use the evidence-backed edges and graph structure. Do not invent literature or unstated mechanisms.\n"
        "- Be willing to disagree with the solver if the graph/benchmark context supports a better recommendation.\n"
        "- If the benchmark is weak for all candidates, still rank the best mechanistic options, but say so clearly.\n"
        "- Keep it concise: recommendation_summary <= 70 words, each rationale <= 40 words, each benchmark_assessment <= 20 words.\n"
        "- Each candidate may include at most 3 supporting_edges, at most 4 evidence_refs, and at most 6 pathway_nodes_expected_off entries.\n\n"
        "Return exactly one complete JSON object with this schema:\n"
        "{\n"
        '  "final_recommendation": ["string"],\n'
        '  "recommendation_summary": "string",\n'
        '  "candidates": [\n'
        "    {\n"
        '      "rank": 1,\n'
        '      "knocked_out_genes": ["string"],\n'
        '      "confidence_score": 0.0,\n'
        '      "rationale": "string",\n'
        '      "pathway_nodes_expected_off": ["string"],\n'
        '      "supporting_edges": ["A -> B"],\n'
        '      "evidence_refs": ["PMID:123" | "OmniPath" | "solver-hit"],\n'
        '      "benchmark_assessment": "string",\n'
        '      "toxicity_risk": "low" | "medium" | "high" | "unknown"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Return exactly 5 ranked candidates.\n"
        "Do not include any prose before or after the JSON.\n"
    )


def build_knockout_user_prompt(context: dict[str, Any]) -> str:
    return (
        "Use this PDAC run context to rank knockout candidates.\n"
        "Important: use only the supplied graph/evidence/benchmark data. Do not add outside biology.\n"
        "If benchmark support is weak, say that explicitly rather than overclaiming.\n\n"
        "Context JSON:\n"
        f"{json.dumps(context, indent=2, sort_keys=True)}\n"
    )


def load_run_context(run_dir: Path) -> dict[str, Any]:
    import json

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    graph = json.loads((run_dir / "regulatory_graph.json").read_text(encoding="utf-8"))
    benchmark = json.loads((run_dir / "benchmark_report.json").read_text(encoding="utf-8"))
    degs = json.loads((run_dir / "top_degs.json").read_text(encoding="utf-8"))
    interactions = json.loads((run_dir / "analysis_interactions.json").read_text(encoding="utf-8"))
    knockout_hits = json.loads((run_dir / "knockout_hits.json").read_text(encoding="utf-8"))

    node_ids = {node["id"] for node in graph.get("nodes", [])}
    benchmark_rows = [
        {
            "gene_symbol": row.get("gene_symbol"),
            "mean_gene_effect": row.get("mean_gene_effect"),
            "hit_rate": row.get("hit_rate"),
            "benchmark_hit": row.get("benchmark_hit"),
            "combined_support_score": row.get("combined_support_score"),
        }
        for row in benchmark.get("results", [])
        if row.get("gene_symbol") in node_ids
    ]
    solver_genes = {gene for hit in knockout_hits for gene in hit.get("knocked_out_genes", [])}
    deg_rows = [
        {
            "gene": row.get("gene"),
            "log2_fold_change": row.get("log2_fold_change"),
            "adjusted_pvalue": row.get("adjusted_pvalue"),
            "ranking": row.get("ranking"),
        }
        for row in degs
        if row.get("gene") in node_ids or row.get("gene") in solver_genes
    ]

    evidence_rows: list[dict[str, Any]] = []
    for result in interactions:
        kept_edges = [
            {
                "source_gene": edge.get("source_gene"),
                "target": edge.get("target"),
                "interaction_type": edge.get("interaction_type"),
                "confidence_score": edge.get("confidence_score"),
                "evidence_summary": edge.get("evidence_summary"),
                "source_refs": edge.get("source_refs", [])[:3],
                "pmid_citations": edge.get("pmid_citations", [])[:3],
                "provenance_sources": edge.get("provenance_sources", []),
            }
            for edge in result.get("interactions", [])
            if edge.get("source_gene") in node_ids or edge.get("target") in node_ids
        ]
        if kept_edges:
            evidence_rows.append({"source_gene": result.get("source_gene"), "interactions": kept_edges[:3]})

    return {
        "run_dir": str(run_dir),
        "target_oncogene": "KRAS",
        "selected_experiment": summary.get("selected_experiment"),
        "solver_knockout_hits": knockout_hits[:3],
        "graph_nodes": [
            {
                "id": node.get("id"),
                "kind": node.get("kind"),
            }
            for node in graph.get("nodes", [])
        ],
        "graph_edges": [
            {
                "source": edge.get("source"),
                "target": edge.get("target"),
                "sign": edge.get("sign"),
                "confidence": edge.get("confidence"),
                "provenance": edge.get("provenance", [])[:4],
                "collapsed_via": edge.get("collapsed_via", [])[:3],
            }
            for edge in graph.get("edges", [])
        ],
        "deg_rows": deg_rows,
        "benchmark_rows": benchmark_rows,
        "analysis_interactions": evidence_rows[:8],
    }


def _coerce_knockout_ranking_payload(
    payload: dict[str, Any],
    *,
    run_dir: Path,
    model_name: str,
    context: dict[str, Any],
) -> LlmKnockoutRanking:
    if "candidates" not in payload and "rank" in payload and "knocked_out_genes" in payload:
        payload = {
            "final_recommendation": payload.get("knocked_out_genes", []),
            "recommendation_summary": payload.get("rationale", ""),
            "candidates": [payload],
        }
    normalized = {
        "model": model_name,
        "run_dir": str(run_dir),
        "target_oncogene": context.get("target_oncogene", "KRAS"),
        "graph_variant": context.get("selected_experiment", ""),
        "final_recommendation": payload.get("final_recommendation", []),
        "recommendation_summary": payload.get("recommendation_summary", ""),
        "candidates": payload.get("candidates", []),
    }
    if not normalized["final_recommendation"] and normalized["candidates"]:
        normalized["final_recommendation"] = normalized["candidates"][0].get("knocked_out_genes", [])
    return LlmKnockoutRanking.model_validate(normalized)


class AnthropicKnockoutRanker:
    def __init__(self, settings: Settings, output_dir: Path) -> None:
        if settings.anthropic_api_key is None:
            raise ValueError("ANTHROPIC_API_KEY is required for Anthropic Opus knockout ranking.")
        self.settings = settings
        self.output_dir = ensure_directory(output_dir)
        self.raw_dir = ensure_directory(self.output_dir / "raw")
        self.client = AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            timeout=settings.request_timeout_seconds,
        )

    async def recommend(self, run_dir: Path, model_name: str = "claude-opus-4-6") -> LlmKnockoutRanking:
        context = load_run_context(run_dir)
        request_payload = {
            "model": model_name,
            "system": build_knockout_system_prompt(),
            "messages": [{"role": "user", "content": build_knockout_user_prompt(context)}],
            "max_tokens": 2600,
            "temperature": 0,
        }
        write_json(self.raw_dir / "request.json", request_payload)
        message = await self.client.messages.create(**request_payload)
        text_payload = "".join(block.text for block in message.content if getattr(block, "type", "") == "text").strip()
        write_json(
            self.raw_dir / "response.json",
            {
                "id": getattr(message, "id", ""),
                "model": getattr(message, "model", model_name),
                "text": text_payload,
                "raw_response": _anthropic_message_to_dict(message),
            },
        )
        parsed = _parse_json_payload(text_payload)
        write_json(self.raw_dir / "parsed.json", parsed)
        ranking = _coerce_knockout_ranking_payload(
            parsed,
            run_dir=run_dir,
            model_name=model_name,
            context=context,
        )
        write_json(self.output_dir / "rankings.json", ranking.model_dump())
        return ranking


def run_anthropic_knockout_ranking(
    run_dir: Path,
    settings: Settings,
    *,
    model_name: str = "claude-opus-4-6",
    output_subdir: str = "llm_knockout_opus",
) -> LlmKnockoutRanking:
    ranker = AnthropicKnockoutRanker(settings, run_dir / output_subdir)
    return asyncio.run(ranker.recommend(run_dir, model_name=model_name))
