from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Protocol, Sequence
from xml.etree import ElementTree

import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from cathy_biology.aliases import GeneAliasResolver
from cathy_biology.config import GrnConfig, Settings
from cathy_biology.models import (
    EvidenceClassScores,
    GeneInteraction,
    GeneResearchResult,
    PriorKnowledgeSummary,
    ResearchOutput,
    ResolvedEntity,
)
from cathy_biology.utils import ensure_directory, write_json


class ResearchClient(Protocol):
    async def research_genes(
        self,
        genes: Sequence[str],
        deg_universe: Sequence[str],
        prior_knowledge: PriorKnowledgeSummary,
        grn_config: GrnConfig,
    ) -> ResearchOutput:
        ...


def build_candidate_universe(
    deg_universe: Sequence[str],
    prior_knowledge: PriorKnowledgeSummary,
    grn_config: GrnConfig,
) -> list[str]:
    core = [grn_config.target_oncogene, *grn_config.immediate_downstream_effectors]
    ranked_prior_nodes = sorted(
        [node for node in prior_knowledge.nodes if node.canonical_symbol.upper() not in {gene.upper() for gene in deg_universe}],
        key=lambda item: (-len(item.sources), item.canonical_symbol),
    )
    prior_nodes = [node.canonical_symbol for node in ranked_prior_nodes[: grn_config.prior.prior_node_limit]]
    return sorted({*deg_universe, *core, *prior_nodes})


def build_prompt_candidate_universe(
    deg_universe: Sequence[str],
    prior_knowledge: PriorKnowledgeSummary,
    grn_config: GrnConfig,
) -> tuple[list[str], list[str]]:
    deg_nodes = sorted({gene.upper() for gene in deg_universe})
    core_nodes = [grn_config.target_oncogene.upper(), *(node.upper() for node in grn_config.immediate_downstream_effectors)]
    preferred_prior_order = [
        "EGFR",
        "ERBB2",
        "ERBB3",
        "FGFR1",
        "FGFR2",
        "FGFR3",
        "MET",
        "SOS1",
        "SOS2",
        "GRB2",
        "SHC1",
        "PTPN11",
        "SRC",
        "PRKCA",
        "PRKCD",
        "PLCG1",
        "PIK3CA",
        "AKT1",
        "RAF1",
        "BRAF",
        "MAP2K1",
        "MAP2K2",
        "MAPK1",
        "MAPK3",
        "STAT3",
        "YAP1",
        "MYC",
        "JUN",
        "FOS",
    ]
    available_prior_nodes = {
        node.canonical_symbol.upper()
        for node in prior_knowledge.nodes
        if node.canonical_symbol.upper() not in set(deg_nodes)
    }
    bridge_nodes = [node for node in preferred_prior_order if node in available_prior_nodes]
    return deg_nodes, sorted({*core_nodes, *bridge_nodes[:20]})


def build_discovery_system_prompt(
    gene: str,
    deg_universe: Sequence[str],
    prior_knowledge: PriorKnowledgeSummary,
    grn_config: GrnConfig,
) -> str:
    candidate_degs, bridge_nodes = build_prompt_candidate_universe(deg_universe, prior_knowledge, grn_config)
    return (
        "You are building a mechanistic PDAC signaling graph from top malignant-cell DEGs.\n"
        f"Seed DEG: {gene}.\n"
        "This is the discovery phase: prefer recall, but only for mechanistic gene-to-gene signaling edges.\n"
        "The final optimization graph is DEG-centered, so use intermediates only when they explain how the seed DEG reaches another DEG or KRAS-core node.\n\n"
        f"Disease context: {grn_config.context}.\n"
        f"Boss oncogene: {grn_config.target_oncogene}.\n"
        f"Core pathway nodes: {', '.join([grn_config.target_oncogene, *grn_config.immediate_downstream_effectors])}.\n"
        f"Projected DEG node set: {', '.join(candidate_degs)}.\n"
        f"Preferred bridge genes if needed: {', '.join(bridge_nodes)}.\n\n"
        "Discovery task:\n"
        f"- Find up to {grn_config.discovery_max_edges_per_gene} plausible mechanistic edges involving {gene}.\n"
        "- Return the strongest 2 to 4 candidate edges by default when evidence exists.\n"
        "- Only return more than 4 edges when the additional edges are independently supported, non-redundant, and still among the strongest findings.\n"
        "- If fewer than two supported edges truly exist, return fewer; do not fabricate edges to satisfy the target.\n"
        "- Allowed outputs include:\n"
        "  * seed DEG -> another DEG\n"
        "  * seed DEG -> intermediate signaling node\n"
        "  * intermediate signaling node -> downstream node, if needed to represent a mechanistic chain initiated by the seed DEG\n"
        "- Prefer edges that are experimentally supported in PDAC. If PDAC is unavailable, pancreas-relevant or closely related cancer evidence is acceptable but should reduce confidence.\n"
        "- Prefer receptor/adaptor/signal-transduction/TF bridge genes over vague pathway labels.\n"
        "- Use canonical HGNC gene symbols in `source_gene`, `target`, and `discovered_entities`.\n"
        "- Do not emit generic pathway names like MAPK pathway, PI3K pathway, EMT, stemness, or invasion as nodes. Emit only gene symbols.\n"
        "- If the mechanism uses an intermediate bridge, prefer one-step bridges and keep the chain compact.\n"
        "- Do not emit more than two new intermediate genes outside the preferred bridge list unless the literature makes them essential.\n"
        "- If the seed DEG only has broad pathway influence without a clear mechanistic edge, return no supported edges.\n"
        "- Keep `evidence_summary` to one sentence and under 35 words.\n"
        "- Include at most 2 PMIDs and at most 2 `source_refs` per edge.\n"
        "- `mechanistic_depth` must be 1 for direct seed-gene edges and 2 when one bridge/intermediate step is required. Never output values above 2.\n\n"
        "Output contract:\n"
        "Return exactly one JSON object. The first character must be `{` and the last character must be `}`.\n"
        "No markdown, no prose, no code fences.\n"
        "Keep the JSON compact. Discovery is for candidate generation only; do not repeat pipeline metadata.\n"
        "Use this schema:\n"
        "{\n"
        '  "interactions": [\n'
        "    {\n"
        '      "source_gene": "string",\n'
        '      "target": "string",\n'
        '      "interaction_type": -1 | 1,\n'
        '      "confidence_score": 0.0,\n'
        '      "evidence_summary": "string",\n'
        '      "pmid_citations": ["string"],\n'
        '      "source_refs": ["PMID:12345678" | "https://..."],\n'
        '      "mechanistic_depth": 1 | 2\n'
        "    }\n"
        "  ],\n"
        '  "discovered_entities": [\n'
        "    {\n"
        '      "canonical_symbol": "string",\n'
        '      "aliases": ["string"],\n'
        '      "entity_type": "deg" | "intermediate" | "prior" | "unknown"\n'
        "    }\n"
        "  ],\n"
        '  "alias_hints": {"string": ["string"]},\n'
        '  "no_supported_edges": true\n'
        "}\n"
    )


def build_discovery_user_prompt(
    gene: str,
    deg_universe: Sequence[str],
    prior_knowledge: PriorKnowledgeSummary,
    grn_config: GrnConfig,
) -> str:
    candidate_degs, bridge_nodes = build_prompt_candidate_universe(deg_universe, prior_knowledge, grn_config)
    return (
        f"Research the seed DEG {gene} in PDAC.\n"
        f"Projected DEG nodes: {', '.join(candidate_degs)}.\n"
        f"Preferred bridge genes: {', '.join(bridge_nodes)}.\n"
        "Return the strongest non-redundant mechanistic edges initiated by this seed. Return JSON only."
    )


def build_verification_system_prompt(
    gene: str,
    discovery_result: GeneResearchResult,
    grn_config: GrnConfig,
) -> str:
    return (
        "You are a senior computational oncology verification agent.\n\n"
        "Pipeline context:\n"
        "1. A discovery pass has already proposed a small mechanistic subgraph for one PDAC DEG.\n"
        "2. Your job is precision, not recall.\n"
        "3. Incorrect edges are worse than missing edges.\n"
        "4. Use web search to independently verify each candidate edge.\n\n"
        f"Disease context: {grn_config.context}.\n"
        f"Seed DEG: {gene}.\n"
        f"Boss oncogene: {grn_config.target_oncogene}.\n"
        "Verification rules:\n"
        "- Keep only edges with mechanistic literature support.\n"
        "- Evidence classes must be scored separately:\n"
        "  * direct_mechanistic\n"
        "  * pdac_specific\n"
        "  * pancreas_relevant\n"
        "  * review_supported\n"
        "- `confidence_score` should reflect the full evidence picture.\n"
        "- If the best support is only pathway membership, co-expression, prognosis, or generic signaling association, reject the edge.\n"
        "- Use canonical HGNC symbols.\n\n"
        "Output contract:\n"
        "Return exactly one JSON object with the same schema as the discovery phase, but with `phase` set to `verification`.\n"
        "Only include verified edges in `interactions`.\n"
        "Each kept edge must include `evidence_scores`, `pmid_citations`, `source_refs`, and a concise evidence summary.\n"
        "No markdown or narrative outside the JSON.\n"
        "Keep top-level metadata compact; the pipeline already knows the seed gene, context, and model.\n"
    )


def build_verification_user_prompt(discovery_result: GeneResearchResult) -> str:
    candidate_edges = [
        {
            "source_gene": edge.source_gene,
            "target": edge.target,
            "interaction_type": edge.interaction_type,
            "source_type": edge.source_type,
            "target_type": edge.target_type,
            "mechanistic_depth": edge.mechanistic_depth,
            "evidence_summary": edge.evidence_summary,
            "source_refs": edge.source_refs,
        }
        for edge in discovery_result.interactions
    ]
    return (
        "Independently verify the following candidate edges and reject unsupported ones.\n"
        f"Candidate edges: {json.dumps(candidate_edges, ensure_ascii=True)}\n"
        f"Discovered entities: {json.dumps([entity.model_dump() for entity in discovery_result.discovered_entities], ensure_ascii=True)}\n"
        "Return JSON only."
    )


def build_discovery_follow_up_user_prompt(
    gene: str,
    prior_results: Sequence[GeneResearchResult],
    grn_config: GrnConfig,
) -> str:
    prior_edges = [
        f"{edge.source_gene}->{edge.target}"
        for result in prior_results
        for edge in result.interactions
    ][:10]
    return (
        f"Continue the same PDAC literature search for {gene}.\n"
        f"Already proposed edges: {', '.join(prior_edges) if prior_edges else 'none'}.\n"
        "Generate additional non-redundant mechanistic pathways that are distinct from the edges above.\n"
        "Prefer 2 to 4 new edges if evidence exists. If there are no credible new edges, return an empty `interactions` list.\n"
        f"Never exceed {grn_config.discovery_max_edges_per_gene} total edges across all rounds.\n"
        "Return JSON only."
    )


class OpenAIResearchClient:
    def __init__(self, settings: Settings, cache_dir: Path) -> None:
        if settings.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is required for OpenAI-backed GRN extraction.")
        self.settings = settings
        self.cache_dir = ensure_directory(cache_dir)
        self.discovery_cache_dir = ensure_directory(self.cache_dir / "discovery")
        self.verification_cache_dir = ensure_directory(self.cache_dir / "verification")
        self.raw_exchange_dir = ensure_directory(self.cache_dir / "raw")
        self.alias_resolver = GeneAliasResolver(self.cache_dir / "aliases")
        self.fallback_client = PubMedHeuristicResearchClient(settings, self.cache_dir / "pubmed_fallback")
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.request_timeout_seconds,
        )
        self._openai_disabled_reason: str | None = None

    async def research_genes(
        self,
        genes: Sequence[str],
        deg_universe: Sequence[str],
        prior_knowledge: PriorKnowledgeSummary,
        grn_config: GrnConfig,
    ) -> ResearchOutput:
        semaphore = asyncio.Semaphore(grn_config.concurrency)

        async def run_gene(gene: str) -> tuple[GeneResearchResult, GeneResearchResult]:
            async with semaphore:
                return await self._research_gene(gene, deg_universe, prior_knowledge, grn_config)

        pairs = await asyncio.gather(*(run_gene(gene) for gene in genes))
        return ResearchOutput(
            discovery_results=[pair[0] for pair in pairs],
            verification_results=[pair[1] for pair in pairs],
        )

    async def _research_gene(
        self,
        gene: str,
        deg_universe: Sequence[str],
        prior_knowledge: PriorKnowledgeSummary,
        grn_config: GrnConfig,
    ) -> tuple[GeneResearchResult, GeneResearchResult]:
        discovery_cache_path = self.discovery_cache_dir / f"{gene}.json"
        verification_cache_path = self.verification_cache_dir / f"{gene}.json"
        if discovery_cache_path.exists() and verification_cache_path.exists():
            cached_discovery = GeneResearchResult.model_validate_json(discovery_cache_path.read_text(encoding="utf-8"))
            cached_verification = GeneResearchResult.model_validate_json(verification_cache_path.read_text(encoding="utf-8"))
            discovery_result = _normalize_research_result(
                cached_discovery,
                seed_gene=gene,
                deg_universe=deg_universe,
                prior_knowledge=prior_knowledge,
                grn_config=grn_config,
                model_name=cached_discovery.raw_model or grn_config.model,
                phase=cached_discovery.phase or "discovery",
                alias_resolver=self.alias_resolver,
            )
            verification_result = _normalize_research_result(
                cached_verification,
                seed_gene=gene,
                deg_universe=deg_universe,
                prior_knowledge=prior_knowledge,
                grn_config=grn_config,
                model_name=cached_verification.raw_model or (grn_config.parser_model if grn_config.enable_verification else grn_config.model),
                phase=cached_verification.phase or ("verification" if grn_config.enable_verification else "discovery"),
                alias_resolver=self.alias_resolver,
            )
            return discovery_result, verification_result

        if self._openai_disabled_reason is None:
            discovery_result = await self._call_openai(
                gene=gene,
                phase="discovery",
                model_name=grn_config.model,
                instructions=build_discovery_system_prompt(gene, deg_universe, prior_knowledge, grn_config),
                prompt=build_discovery_user_prompt(gene, deg_universe, prior_knowledge, grn_config),
                grn_config=grn_config,
            )
            if discovery_result is not None:
                discovery_result = _normalize_research_result(
                    discovery_result,
                    seed_gene=gene,
                    deg_universe=deg_universe,
                    prior_knowledge=prior_knowledge,
                    grn_config=grn_config,
                    model_name=grn_config.model,
                    phase="discovery",
                    alias_resolver=self.alias_resolver,
                )
                verification_result = discovery_result
                if grn_config.enable_verification and discovery_result.interactions:
                    verified = await self._call_openai(
                        gene=gene,
                        phase="verification",
                        model_name=grn_config.parser_model,
                        instructions=build_verification_system_prompt(gene, discovery_result, grn_config),
                        prompt=build_verification_user_prompt(discovery_result),
                        grn_config=grn_config,
                    )
                    if verified is not None:
                        verification_result = _normalize_research_result(
                            verified,
                            seed_gene=gene,
                            deg_universe=deg_universe,
                            prior_knowledge=prior_knowledge,
                            grn_config=grn_config,
                            model_name=grn_config.parser_model,
                            phase="verification",
                            alias_resolver=self.alias_resolver,
                        )
                write_json(discovery_cache_path, discovery_result.model_dump())
                write_json(verification_cache_path, verification_result.model_dump())
                return discovery_result, verification_result

        fallback = await self.fallback_client.research_genes([gene], deg_universe, prior_knowledge, grn_config)
        discovery_result = fallback.discovery_results[0]
        verification_result = fallback.verification_results[0]
        write_json(discovery_cache_path, discovery_result.model_dump())
        write_json(verification_cache_path, verification_result.model_dump())
        return discovery_result, verification_result

    async def _call_openai(
        self,
        *,
        gene: str,
        phase: str,
        model_name: str,
        instructions: str,
        prompt: str,
        grn_config: GrnConfig,
    ) -> GeneResearchResult | None:
        request_payload = {
            "model": model_name,
            "instructions": instructions,
            "input": prompt,
            "tools": [{"type": "web_search_preview"}],
            "max_output_tokens": 2_000,
        }
        self._write_exchange_artifacts(gene=gene, phase=phase, kind="request", payload=request_payload)
        try:
            response = await self.client.responses.create(**request_payload)
            text_payload = getattr(response, "output_text", "").strip()
            if not text_payload:
                text_payload = _extract_openai_text(response)
            self._write_exchange_artifacts(
                gene=gene,
                phase=phase,
                kind="response",
                payload={
                    "model": getattr(response, "model", model_name),
                    "id": getattr(response, "id", ""),
                    "text": text_payload,
                    "raw_response": _openai_response_to_dict(response),
                },
            )
            if not text_payload:
                return None
            parsed = _parse_json_payload(text_payload)
            self._write_exchange_artifacts(gene=gene, phase=phase, kind="parsed", payload=parsed)
            return _coerce_research_result(
                parsed,
                seed_gene=gene,
                target_oncogene=grn_config.target_oncogene,
                context=grn_config.context,
                phase=phase,
                model_name=model_name,
            )
        except Exception as exc:
            self._write_exchange_artifacts(
                gene=gene,
                phase=phase,
                kind="error",
                payload={"model": model_name, "error": str(exc)},
            )
            error_message = str(exc).lower()
            if any(
                marker in error_message
                for marker in [
                    "invalid_api_key",
                    "incorrect api key",
                    "insufficient_quota",
                    "billing",
                    "must be verified",
                    "model_not_found",
                    "quota",
                ]
            ):
                self._openai_disabled_reason = str(exc)
            return None

    def _write_exchange_artifacts(self, *, gene: str, phase: str, kind: str, payload: object) -> None:
        directory = ensure_directory(self.raw_exchange_dir / phase / gene)
        write_json(directory / f"{kind}.json", payload)


class AnthropicResearchClient:
    def __init__(self, settings: Settings, cache_dir: Path) -> None:
        if settings.anthropic_api_key is None:
            raise ValueError("ANTHROPIC_API_KEY is required for Anthropic-backed GRN extraction.")
        self.settings = settings
        self.cache_dir = ensure_directory(cache_dir)
        self.discovery_cache_dir = ensure_directory(self.cache_dir / "discovery")
        self.verification_cache_dir = ensure_directory(self.cache_dir / "verification")
        self.raw_exchange_dir = ensure_directory(self.cache_dir / "raw")
        self.alias_resolver = GeneAliasResolver(self.cache_dir / "aliases")
        self.fallback_client = PubMedHeuristicResearchClient(settings, self.cache_dir / "pubmed_fallback")
        self.client = AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            timeout=settings.request_timeout_seconds,
        )

    async def research_genes(
        self,
        genes: Sequence[str],
        deg_universe: Sequence[str],
        prior_knowledge: PriorKnowledgeSummary,
        grn_config: GrnConfig,
    ) -> ResearchOutput:
        semaphore = asyncio.Semaphore(grn_config.concurrency)

        async def run_gene(gene: str) -> tuple[GeneResearchResult, GeneResearchResult]:
            async with semaphore:
                return await self._research_gene(gene, deg_universe, prior_knowledge, grn_config)

        pairs = await asyncio.gather(*(run_gene(gene) for gene in genes))
        return ResearchOutput(
            discovery_results=[pair[0] for pair in pairs],
            verification_results=[pair[1] for pair in pairs],
        )

    async def _research_gene(
        self,
        gene: str,
        deg_universe: Sequence[str],
        prior_knowledge: PriorKnowledgeSummary,
        grn_config: GrnConfig,
    ) -> tuple[GeneResearchResult, GeneResearchResult]:
        discovery_cache_path = self.discovery_cache_dir / f"{gene}.json"
        verification_cache_path = self.verification_cache_dir / f"{gene}.json"
        if discovery_cache_path.exists() and verification_cache_path.exists():
            cached_discovery = GeneResearchResult.model_validate_json(discovery_cache_path.read_text(encoding="utf-8"))
            cached_verification = GeneResearchResult.model_validate_json(verification_cache_path.read_text(encoding="utf-8"))
            discovery_result = _normalize_research_result(
                cached_discovery,
                seed_gene=gene,
                deg_universe=deg_universe,
                prior_knowledge=prior_knowledge,
                grn_config=grn_config,
                model_name=cached_discovery.raw_model or grn_config.model,
                phase=cached_discovery.phase or "discovery",
                alias_resolver=self.alias_resolver,
            )
            verification_result = _normalize_research_result(
                cached_verification,
                seed_gene=gene,
                deg_universe=deg_universe,
                prior_knowledge=prior_knowledge,
                grn_config=grn_config,
                model_name=cached_verification.raw_model or (grn_config.parser_model if grn_config.enable_verification else grn_config.model),
                phase=cached_verification.phase or ("verification" if grn_config.enable_verification else "discovery"),
                alias_resolver=self.alias_resolver,
            )
            return discovery_result, verification_result

        discovery_result = await self._call_model(
            gene=gene,
            phase="discovery",
            system_prompt=build_discovery_system_prompt(gene, deg_universe, prior_knowledge, grn_config),
            user_prompt=build_discovery_user_prompt(gene, deg_universe, prior_knowledge, grn_config),
            model_name=grn_config.model,
            grn_config=grn_config,
        )
        if discovery_result is None:
            fallback = await self.fallback_client.research_genes([gene], deg_universe, prior_knowledge, grn_config)
            discovery_result = fallback.discovery_results[0]
            verification_result = fallback.verification_results[0]
            write_json(discovery_cache_path, discovery_result.model_dump())
            write_json(verification_cache_path, verification_result.model_dump())
            return discovery_result, verification_result

        discovery_result = _normalize_research_result(
            discovery_result,
            seed_gene=gene,
            deg_universe=deg_universe,
            prior_knowledge=prior_knowledge,
            grn_config=grn_config,
            model_name=grn_config.model,
            phase="discovery",
            alias_resolver=self.alias_resolver,
        )
        verification_result = discovery_result
        if grn_config.enable_verification and discovery_result.interactions:
            verified = await self._call_model(
                gene=gene,
                phase="verification",
                system_prompt=build_verification_system_prompt(gene, discovery_result, grn_config),
                user_prompt=build_verification_user_prompt(discovery_result),
                model_name=grn_config.parser_model,
                grn_config=grn_config,
            )
            if verified is not None:
                verification_result = _normalize_research_result(
                    verified,
                    seed_gene=gene,
                    deg_universe=deg_universe,
                    prior_knowledge=prior_knowledge,
                    grn_config=grn_config,
                    model_name=grn_config.parser_model,
                    phase="verification",
                    alias_resolver=self.alias_resolver,
                )

        write_json(discovery_cache_path, discovery_result.model_dump())
        write_json(verification_cache_path, verification_result.model_dump())
        return discovery_result, verification_result

    async def _call_model(
        self,
        gene: str,
        phase: str,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
        grn_config: GrnConfig,
    ) -> GeneResearchResult | None:
        messages: list[dict[str, str]] = [{"role": "user", "content": user_prompt}]
        round_results: list[GeneResearchResult] = []
        max_rounds = grn_config.discovery_rounds if phase == "discovery" else 1

        for round_index in range(1, max_rounds + 1):
            request_payload = {
                "model": model_name,
                "max_tokens": 1_200,
                "system": system_prompt,
                "messages": messages,
                "temperature": 0,
                "tools": [
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": grn_config.max_tool_calls,
                        "allowed_callers": ["direct"],
                    }
                ],
            }
            suffix = f"_round{round_index:02d}"
            self._write_exchange_artifacts(gene=gene, phase=phase, kind=f"request{suffix}", payload=request_payload)
            for attempt in range(1, 4):
                try:
                    message = await self.client.messages.create(**request_payload)
                    text_payload = _extract_anthropic_text(message.content)
                    self._write_exchange_artifacts(
                        gene=gene,
                        phase=phase,
                        kind=f"response{suffix}",
                        payload={
                            "model": getattr(message, "model", model_name),
                            "id": getattr(message, "id", ""),
                            "text": text_payload,
                            "raw_response": _anthropic_message_to_dict(message),
                        },
                    )
                    if not text_payload:
                        break
                    parsed = _parse_json_payload(text_payload)
                    self._write_exchange_artifacts(gene=gene, phase=phase, kind=f"parsed{suffix}", payload=parsed)
                    result = _coerce_research_result(
                        parsed,
                        seed_gene=gene,
                        target_oncogene=grn_config.target_oncogene,
                        context=grn_config.context,
                        phase=phase,
                        model_name=model_name,
                    )
                    prior_unique_edges = {
                        (edge.source_gene, edge.target, edge.interaction_type)
                        for previous in round_results
                        for edge in previous.interactions
                    }
                    round_results.append(result)
                    messages.append({"role": "assistant", "content": text_payload})
                    if phase != "discovery" or round_index >= max_rounds:
                        merged = _merge_research_rounds(
                            round_results,
                            seed_gene=gene,
                            target_oncogene=grn_config.target_oncogene,
                            context=grn_config.context,
                            phase=phase,
                            model_name=model_name,
                        )
                        self._write_exchange_artifacts(gene=gene, phase=phase, kind="parsed", payload=merged.model_dump())
                        return merged
                    current_unique_edges = {
                        (edge.source_gene, edge.target, edge.interaction_type)
                        for r in round_results
                        for edge in r.interactions
                    }
                    if not result.interactions or current_unique_edges == prior_unique_edges:
                        merged = _merge_research_rounds(
                            round_results,
                            seed_gene=gene,
                            target_oncogene=grn_config.target_oncogene,
                            context=grn_config.context,
                            phase=phase,
                            model_name=model_name,
                        )
                        self._write_exchange_artifacts(gene=gene, phase=phase, kind="parsed", payload=merged.model_dump())
                        return merged
                    if len(current_unique_edges) >= grn_config.discovery_max_edges_per_gene:
                        merged = _merge_research_rounds(
                            round_results,
                            seed_gene=gene,
                            target_oncogene=grn_config.target_oncogene,
                            context=grn_config.context,
                            phase=phase,
                            model_name=model_name,
                        )
                        self._write_exchange_artifacts(gene=gene, phase=phase, kind="parsed", payload=merged.model_dump())
                        return merged
                    messages.append(
                        {
                            "role": "user",
                            "content": build_discovery_follow_up_user_prompt(gene, round_results, grn_config),
                        }
                    )
                    break
                except Exception as exc:
                    error_payload = {
                        "model": model_name,
                        "attempt": attempt,
                        "error": str(exc),
                    }
                    self._write_exchange_artifacts(
                        gene=gene,
                        phase=phase,
                        kind=f"error{suffix}",
                        payload=error_payload,
                    )
                    if "rate limit" in str(exc).lower() and attempt < 3:
                        await asyncio.sleep(10 * attempt)
                        continue
                    if not round_results:
                        return None
                    merged = _merge_research_rounds(
                        round_results,
                        seed_gene=gene,
                        target_oncogene=grn_config.target_oncogene,
                        context=grn_config.context,
                        phase=phase,
                        model_name=model_name,
                    )
                    self._write_exchange_artifacts(gene=gene, phase=phase, kind="parsed", payload=merged.model_dump())
                    return merged

        if not round_results:
            return None
        merged = _merge_research_rounds(
            round_results,
            seed_gene=gene,
            target_oncogene=grn_config.target_oncogene,
            context=grn_config.context,
            phase=phase,
            model_name=model_name,
        )
        self._write_exchange_artifacts(gene=gene, phase=phase, kind="parsed", payload=merged.model_dump())
        return merged

    def _write_exchange_artifacts(self, *, gene: str, phase: str, kind: str, payload: object) -> None:
        directory = ensure_directory(self.raw_exchange_dir / phase / gene)
        write_json(directory / f"{kind}.json", payload)


class MockResearchClient:
    def __init__(self, mapping: dict[str, list[GeneInteraction]], target_oncogene: str, context: str) -> None:
        self.mapping = mapping
        self.target_oncogene = target_oncogene
        self.context = context

    async def research_genes(
        self,
        genes: Sequence[str],
        deg_universe: Sequence[str],
        prior_knowledge: PriorKnowledgeSummary,
        grn_config: GrnConfig,
    ) -> ResearchOutput:
        discovery_results: list[GeneResearchResult] = []
        verification_results: list[GeneResearchResult] = []
        for gene in genes:
            interactions = self.mapping.get(gene, [])
            result = GeneResearchResult(
                source_gene=gene,
                target_oncogene=self.target_oncogene,
                context=self.context,
                interactions=interactions,
                discovered_entities=[
                    ResolvedEntity(canonical_symbol=symbol, aliases=[symbol], entity_type="deg", sources=["mock"])
                    for symbol in {gene, *(edge.source_gene for edge in interactions), *(edge.target for edge in interactions)}
                ],
                no_direct_effect=not interactions,
                no_supported_edges=not interactions,
                queried_targets=[self.target_oncogene, *grn_config.immediate_downstream_effectors],
                raw_model="mock",
                phase="verification",
            )
            discovery_results.append(result.model_copy(update={"phase": "discovery"}))
            verification_results.append(result)
        return ResearchOutput(discovery_results=discovery_results, verification_results=verification_results)


def _extract_openai_text(response: object) -> str:
    texts: list[str] = []
    for output_item in getattr(response, "output", []):
        if not hasattr(output_item, "content"):
            continue
        for content in output_item.content:
            text = getattr(content, "text", "")
            if text:
                texts.append(text)
    return "\n".join(texts).strip()


def _extract_anthropic_text(content_blocks: Sequence[object]) -> str:
    texts: list[str] = []
    for block in content_blocks:
        if getattr(block, "type", "") == "text":
            text = getattr(block, "text", "")
            if text:
                texts.append(text)
    return "\n".join(texts).strip()


def _parse_json_payload(payload: str) -> dict:
    payload = payload.strip()
    fenced_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", payload, flags=re.DOTALL)
    if fenced_blocks:
        return json.loads(fenced_blocks[-1])
    if payload.startswith("```"):
        payload = payload.strip("`")
        payload = payload.removeprefix("json").strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        extracted = _extract_balanced_json_object(payload)
        if extracted is None:
            raise
        return json.loads(extracted)


def _extract_balanced_json_object(payload: str) -> str | None:
    start = payload.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(payload)):
            char = payload[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = payload[start : index + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break
        start = payload.find("{", start + 1)
    return None


def _coerce_research_result(
    payload: dict,
    *,
    seed_gene: str,
    target_oncogene: str,
    context: str,
    phase: str,
    model_name: str,
) -> GeneResearchResult:
    normalized = dict(payload)
    normalized = _sanitize_research_payload(normalized)
    normalized.setdefault("source_gene", seed_gene)
    normalized.setdefault("target_oncogene", target_oncogene)
    normalized.setdefault("context", context)
    normalized.setdefault("raw_model", model_name)
    normalized.setdefault("phase", phase)
    normalized.setdefault("discovered_entities", [])
    normalized.setdefault("alias_hints", {})
    normalized.setdefault("interactions", [])
    normalized.setdefault("no_supported_edges", not normalized["interactions"])
    normalized.setdefault("no_direct_effect", False)
    normalized.setdefault("queried_targets", [])
    for entity in normalized["discovered_entities"]:
        entity.setdefault("sources", [])
    return GeneResearchResult.model_validate(normalized)


def _merge_research_rounds(
    round_results: Sequence[GeneResearchResult],
    *,
    seed_gene: str,
    target_oncogene: str,
    context: str,
    phase: str,
    model_name: str,
) -> GeneResearchResult:
    merged_alias_hints: dict[str, list[str]] = {}
    merged_entities: list[ResolvedEntity] = []
    merged_interactions: list[GeneInteraction] = []
    for result in round_results:
        merged_interactions.extend(result.interactions)
        merged_entities.extend(result.discovered_entities)
        for key, values in result.alias_hints.items():
            merged_alias_hints[key] = sorted({*merged_alias_hints.get(key, []), *values})
    return GeneResearchResult(
        source_gene=seed_gene,
        target_oncogene=target_oncogene,
        context=context,
        interactions=merged_interactions,
        discovered_entities=merged_entities,
        alias_hints=merged_alias_hints,
        no_direct_effect=not merged_interactions,
        no_supported_edges=not merged_interactions,
        queried_targets=[],
        raw_model=model_name,
        phase=phase,  # type: ignore[arg-type]
    )


def _sanitize_research_payload(payload: dict) -> dict:
    sanitized = dict(payload)
    alias_hints = sanitized.get("alias_hints", {})
    if isinstance(alias_hints, dict):
        sanitized["alias_hints"] = {
            str(key): value if isinstance(value, list) else [str(value)]
            for key, value in alias_hints.items()
        }
    interactions = []
    for interaction in sanitized.get("interactions", []):
        current = dict(interaction)
        current["mechanistic_depth"] = max(1, min(int(current.get("mechanistic_depth", 1)), 2))
        current["pmid_citations"] = [str(item) for item in current.get("pmid_citations", [])]
        current["source_refs"] = [str(item) for item in current.get("source_refs", [])]
        interactions.append(current)
    sanitized["interactions"] = interactions
    entities = []
    for entity in sanitized.get("discovered_entities", []):
        current = dict(entity)
        aliases = current.get("aliases", [])
        if isinstance(aliases, str):
            current["aliases"] = [aliases]
        current.setdefault("sources", [])
        entities.append(current)
    sanitized["discovered_entities"] = entities
    return sanitized


def _openai_response_to_dict(response: object) -> dict:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "to_dict"):
        return response.to_dict()
    return {"repr": repr(response)}


def _anthropic_message_to_dict(message: object) -> dict:
    if hasattr(message, "model_dump"):
        return message.model_dump(mode="json")
    if hasattr(message, "to_dict"):
        return message.to_dict()
    return {"repr": repr(message)}


def _normalize_research_result(
    result: GeneResearchResult,
    seed_gene: str,
    deg_universe: Sequence[str],
    prior_knowledge: PriorKnowledgeSummary,
    grn_config: GrnConfig,
    model_name: str,
    phase: str,
    alias_resolver: GeneAliasResolver,
) -> GeneResearchResult:
    preferred_symbols = {
        seed_gene.upper(),
        *(gene.upper() for gene in deg_universe),
        grn_config.target_oncogene.upper(),
        *(node.upper() for node in grn_config.immediate_downstream_effectors),
        *(node.canonical_symbol.upper() for node in prior_knowledge.nodes),
    }
    alias_hints = {key.upper(): value for key, value in result.alias_hints.items()}
    entities_to_resolve = {seed_gene, *(entity.canonical_symbol for entity in result.discovered_entities)}
    for interaction in result.interactions:
        entities_to_resolve.add(interaction.source_gene)
        entities_to_resolve.add(interaction.target)
    resolved = alias_resolver.resolve_symbols(entities_to_resolve, preferred_symbols=preferred_symbols, extra_aliases=alias_hints)

    result.source_gene = resolved.get(seed_gene, ResolvedEntity(canonical_symbol=seed_gene.upper(), aliases=[seed_gene])).canonical_symbol
    result.target_oncogene = grn_config.target_oncogene.upper()
    result.context = grn_config.context
    result.queried_targets = build_candidate_universe(deg_universe, prior_knowledge, grn_config)
    result.raw_model = model_name
    result.phase = phase  # type: ignore[assignment]

    normalized_edges: dict[tuple[str, str, int], GeneInteraction] = {}
    for interaction in result.interactions:
        source = resolved.get(interaction.source_gene, ResolvedEntity(canonical_symbol=interaction.source_gene.upper())).canonical_symbol
        target = resolved.get(interaction.target, ResolvedEntity(canonical_symbol=interaction.target.upper())).canonical_symbol
        if source == target:
            continue
        normalized = interaction.model_copy(
            update={
                "source_gene": source,
                "target": target,
                "source_type": _node_type(source, deg_universe, prior_knowledge, grn_config),
                "target_type": _node_type(target, deg_universe, prior_knowledge, grn_config),
                "provenance_sources": sorted(set([*interaction.provenance_sources, model_name, phase])),
            }
        )
        key = (source, target, normalized.interaction_type)
        previous = normalized_edges.get(key)
        if previous is None or normalized.confidence_score > previous.confidence_score:
            normalized_edges[key] = normalized

    discovered_entities = {
        resolved_entity.canonical_symbol: resolved_entity
        for resolved_entity in [
            *(
                ResolvedEntity(
                    canonical_symbol=resolved.get(entity.canonical_symbol, entity).canonical_symbol,
                    aliases=sorted(set(entity.aliases + resolved.get(entity.canonical_symbol, entity).aliases)),
                    entity_type=_node_type(
                        resolved.get(entity.canonical_symbol, entity).canonical_symbol,
                        deg_universe,
                        prior_knowledge,
                        grn_config,
                    ),
                    sources=sorted(set(entity.sources + resolved.get(entity.canonical_symbol, entity).sources)),
                )
                for entity in result.discovered_entities
            ),
            ResolvedEntity(
                canonical_symbol=result.source_gene,
                aliases=[seed_gene],
                entity_type=_node_type(result.source_gene, deg_universe, prior_knowledge, grn_config),
                sources=["seed-gene"],
            ),
            *(
                ResolvedEntity(
                    canonical_symbol=edge.source_gene,
                    aliases=[edge.source_gene],
                    entity_type=edge.source_type,
                    sources=edge.provenance_sources,
                )
                for edge in normalized_edges.values()
            ),
            *(
                ResolvedEntity(
                    canonical_symbol=edge.target,
                    aliases=[edge.target],
                    entity_type=edge.target_type,
                    sources=edge.provenance_sources,
                )
                for edge in normalized_edges.values()
            ),
        ]
    }

    normalized_interactions = sorted(
        _filter_seed_anchored_edges(result.source_gene, normalized_edges.values()),
        key=lambda edge: (-edge.confidence_score, edge.source_gene, edge.target, edge.interaction_type),
    )
    result.interactions = normalized_interactions
    result.discovered_entities = sorted(discovered_entities.values(), key=lambda entity: entity.canonical_symbol)
    result.no_supported_edges = not normalized_interactions
    result.no_direct_effect = not any(
        edge.source_gene == result.source_gene
        and edge.target in {grn_config.target_oncogene.upper(), *(node.upper() for node in grn_config.immediate_downstream_effectors)}
        for edge in normalized_interactions
    )
    return result


def _node_type(
    symbol: str,
    deg_universe: Sequence[str],
    prior_knowledge: PriorKnowledgeSummary,
    grn_config: GrnConfig,
) -> str:
    symbol_upper = symbol.upper()
    if symbol_upper in {gene.upper() for gene in deg_universe}:
        return "deg"
    if symbol_upper == grn_config.target_oncogene.upper() or symbol_upper in {
        node.upper() for node in grn_config.immediate_downstream_effectors
    }:
        return "pathway"
    if symbol_upper in {node.canonical_symbol.upper() for node in prior_knowledge.nodes}:
        return "prior"
    return "intermediate"


def _filter_seed_anchored_edges(seed_gene: str, interactions: Sequence[GeneInteraction]) -> list[GeneInteraction]:
    adjacency: dict[str, set[str]] = {}
    for interaction in interactions:
        adjacency.setdefault(interaction.source_gene, set()).add(interaction.target)

    reachable = {seed_gene}
    frontier = [seed_gene]
    while frontier:
        source = frontier.pop()
        for target in adjacency.get(source, set()):
            if target in reachable:
                continue
            reachable.add(target)
            frontier.append(target)

    return [interaction for interaction in interactions if interaction.source_gene in reachable]


class PubMedHeuristicResearchClient:
    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    ACTIVATION_KEYWORDS = (
        "activate",
        "activation",
        "upregulate",
        "up-regulate",
        "promote",
        "enhance",
        "induce",
        "stimulate",
        "increase",
        "drive",
        "phosphorylate",
    )
    INHIBITION_KEYWORDS = (
        "inhibit",
        "inhibition",
        "suppress",
        "repress",
        "downregulate",
        "down-regulate",
        "attenuate",
        "block",
        "reduce",
        "decrease",
    )

    def __init__(self, settings: Settings, cache_dir: Path) -> None:
        self.settings = settings
        self.cache_dir = ensure_directory(cache_dir)
        self.discovery_cache_dir = ensure_directory(self.cache_dir / "discovery")
        self.verification_cache_dir = ensure_directory(self.cache_dir / "verification")
        self.alias_resolver = GeneAliasResolver(self.cache_dir / "aliases")

    async def research_genes(
        self,
        genes: Sequence[str],
        deg_universe: Sequence[str],
        prior_knowledge: PriorKnowledgeSummary,
        grn_config: GrnConfig,
    ) -> ResearchOutput:
        async with httpx.AsyncClient(
            timeout=self.settings.request_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "cathy-biology/0.2"},
        ) as client:
            semaphore = asyncio.Semaphore(1)

            async def run_gene(gene: str) -> tuple[GeneResearchResult, GeneResearchResult]:
                async with semaphore:
                    return await self._research_gene(client, gene, deg_universe, prior_knowledge, grn_config)

            pairs = await asyncio.gather(*(run_gene(gene) for gene in genes))
        return ResearchOutput(
            discovery_results=[pair[0] for pair in pairs],
            verification_results=[pair[1] for pair in pairs],
        )

    async def _research_gene(
        self,
        client: httpx.AsyncClient,
        gene: str,
        deg_universe: Sequence[str],
        prior_knowledge: PriorKnowledgeSummary,
        grn_config: GrnConfig,
    ) -> tuple[GeneResearchResult, GeneResearchResult]:
        discovery_cache_path = self.discovery_cache_dir / f"{gene}.json"
        verification_cache_path = self.verification_cache_dir / f"{gene}.json"
        if discovery_cache_path.exists() and verification_cache_path.exists():
            discovery_result = GeneResearchResult.model_validate_json(discovery_cache_path.read_text(encoding="utf-8"))
            verification_result = GeneResearchResult.model_validate_json(verification_cache_path.read_text(encoding="utf-8"))
            return discovery_result, verification_result

        candidate_universe = build_candidate_universe(deg_universe, prior_knowledge, grn_config)[:80]
        pmids = await self._search_pmids(client, gene, grn_config.context)
        articles = await self._fetch_articles(client, pmids) if pmids else []
        interactions = self._infer_interactions(gene, candidate_universe, articles)
        discovery_result = _normalize_research_result(
            GeneResearchResult(
                source_gene=gene,
                target_oncogene=grn_config.target_oncogene,
                context=grn_config.context,
                interactions=interactions,
                discovered_entities=[ResolvedEntity(canonical_symbol=node, aliases=[node], sources=["pubmed-heuristic"]) for node in {gene, *(edge.source_gene for edge in interactions), *(edge.target for edge in interactions)}],
                no_direct_effect=not interactions,
                no_supported_edges=not interactions,
                queried_targets=candidate_universe,
                raw_model="pubmed-heuristic",
                phase="heuristic",
            ),
            seed_gene=gene,
            deg_universe=deg_universe,
            prior_knowledge=prior_knowledge,
            grn_config=grn_config,
            model_name="pubmed-heuristic",
            phase="discovery",
            alias_resolver=self.alias_resolver,
        )
        verification_result = discovery_result.model_copy(update={"phase": "verification"})
        write_json(discovery_cache_path, discovery_result.model_dump())
        write_json(verification_cache_path, verification_result.model_dump())
        return discovery_result, verification_result

    async def _search_pmids(self, client: httpx.AsyncClient, gene: str, context: str) -> list[str]:
        params = {
            "db": "pubmed",
            "retmode": "json",
            "retmax": "12",
            "term": f'"{gene}"[Title/Abstract] AND ("{context}"[Title/Abstract] OR PDAC[Title/Abstract] OR pancreas[Title/Abstract])',
        }
        response = await self._get_with_retry(client, self.ESEARCH_URL, params=params)
        payload = response.json()
        return payload.get("esearchresult", {}).get("idlist", [])

    async def _fetch_articles(self, client: httpx.AsyncClient, pmids: list[str]) -> list[dict[str, str]]:
        params = {"db": "pubmed", "retmode": "xml", "id": ",".join(pmids)}
        response = await self._get_with_retry(client, self.EFETCH_URL, params=params)
        root = ElementTree.fromstring(response.text)
        articles: list[dict[str, str]] = []
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID", default="")
            title = " ".join(article.findtext(".//ArticleTitle", default="").split())
            abstract_parts = [" ".join(element.text.split()) for element in article.findall(".//Abstract/AbstractText") if element.text]
            articles.append({"pmid": pmid, "text": " ".join([title, *abstract_parts]).strip()})
        return articles

    def _infer_interactions(
        self,
        gene: str,
        candidate_universe: list[str],
        articles: list[dict[str, str]],
    ) -> list[GeneInteraction]:
        interactions: list[GeneInteraction] = []
        gene_lower = gene.lower()
        for target in candidate_universe:
            if target.upper() == gene.upper():
                continue
            target_lower = target.lower()
            activation_pmids: list[str] = []
            inhibition_pmids: list[str] = []
            evidence_sentences: list[str] = []
            for article in articles:
                article_lower = article["text"].lower()
                if gene_lower not in article_lower or target_lower not in article_lower:
                    continue
                sentences = re.split(r"(?<=[.!?])\s+", article["text"])
                relevant = [
                    sentence
                    for sentence in sentences
                    if gene_lower in sentence.lower() and target_lower in sentence.lower()
                ]
                if not relevant:
                    continue
                sentence_text = " ".join(relevant[:2])
                lowered = sentence_text.lower()
                evidence_sentences.extend(relevant[:2])
                if any(keyword in lowered for keyword in self.ACTIVATION_KEYWORDS):
                    activation_pmids.append(article["pmid"])
                if any(keyword in lowered for keyword in self.INHIBITION_KEYWORDS):
                    inhibition_pmids.append(article["pmid"])
            if not activation_pmids and not inhibition_pmids:
                continue
            if len(activation_pmids) == len(inhibition_pmids):
                continue
            interaction_type = 1 if len(activation_pmids) > len(inhibition_pmids) else -1
            pmids = activation_pmids if interaction_type == 1 else inhibition_pmids
            interactions.append(
                GeneInteraction(
                    source_gene=gene,
                    target=target,
                    interaction_type=interaction_type,
                    pmid_citations=sorted(set(filter(None, pmids))),
                    confidence_score=min(0.7, 0.2 + 0.12 * len(pmids)),
                    evidence_summary=" ".join(evidence_sentences[:2]),
                    provenance_sources=["pubmed-heuristic"],
                    evidence_scores=EvidenceClassScores(
                        direct_mechanistic=0.4,
                        pdac_specific=0.5,
                        pancreas_relevant=0.5,
                    ),
                )
            )
        return interactions

    async def _get_with_retry(self, client: httpx.AsyncClient, url: str, params: dict[str, str]) -> httpx.Response:
        delay_seconds = 0.5
        last_error: Exception | None = None
        for _ in range(5):
            try:
                response = await client.get(url, params=params)
                if response.status_code == 429:
                    await asyncio.sleep(delay_seconds)
                    delay_seconds *= 2
                    continue
                response.raise_for_status()
                await asyncio.sleep(0.34)
                return response
            except httpx.HTTPError as exc:
                last_error = exc
                await asyncio.sleep(delay_seconds)
                delay_seconds *= 2
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Failed to fetch {url}")
