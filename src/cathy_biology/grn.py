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
from openai.types.responses import ParsedResponse

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


def build_discovery_system_prompt(
    gene: str,
    deg_universe: Sequence[str],
    prior_knowledge: PriorKnowledgeSummary,
    grn_config: GrnConfig,
) -> str:
    candidate_universe = build_candidate_universe(deg_universe, prior_knowledge, grn_config)
    candidate_degs = ", ".join(sorted(set(deg_universe)))
    prior_nodes = ", ".join(node for node in candidate_universe if node not in set(deg_universe))
    return (
        "You are a senior computational oncology research agent building a mechanistic PDAC signaling graph.\n\n"
        "Pipeline context:\n"
        "1. Upstream code already distilled a PDAC scRNA-seq dataset to the top 50 DEGs.\n"
        f"2. The current seed DEG is {gene}.\n"
        "3. This is the discovery phase. High recall is desired, but only for mechanistic signaling edges.\n"
        "4. Your output feeds a later verification pass, curated pathway priors, and a weighted Boolean knockout simulator.\n\n"
        f"Disease context: {grn_config.context}.\n"
        f"Boss oncogene: {grn_config.target_oncogene}.\n"
        f"Core pathway nodes: {', '.join([grn_config.target_oncogene, *grn_config.immediate_downstream_effectors])}.\n"
        f"DEG universe: {candidate_degs}.\n"
        f"Curated prior/intermediate node universe: {prior_nodes}.\n\n"
        "Discovery task:\n"
        f"- Find up to {grn_config.discovery_max_edges_per_gene} plausible mechanistic edges involving {gene}.\n"
        "- Allowed outputs include:\n"
        "  * seed DEG -> another DEG\n"
        "  * seed DEG -> intermediate signaling node\n"
        "  * intermediate signaling node -> downstream node, if needed to represent a mechanistic chain initiated by the seed DEG\n"
        "- Prefer edges that are experimentally supported in PDAC. If PDAC is unavailable, pancreas-relevant or closely related cancer evidence is acceptable but should reduce confidence.\n"
        "- Prefer receptor/adaptor/signal-transduction/TF bridge nodes over vague pathway labels.\n"
        "- Use canonical HGNC gene symbols in `source_gene`, `target`, and `discovered_entities`.\n"
        "- Do not emit generic pathway names like MAPK pathway, PI3K pathway, EMT, stemness, or invasion as nodes. Emit only gene symbols.\n"
        "- Do not emit more than two new intermediate genes outside the listed candidate universe unless the literature makes them essential.\n"
        "- If the seed DEG only has broad pathway influence without a clear mechanistic edge, return no supported edges.\n\n"
        "Output contract:\n"
        "Return exactly one JSON object. The first character must be `{` and the last character must be `}`.\n"
        "No markdown, no prose, no code fences.\n"
        "Use this schema:\n"
        "{\n"
        '  "source_gene": "string",\n'
        '  "target_oncogene": "string",\n'
        '  "context": "string",\n'
        '  "interactions": [\n'
        "    {\n"
        '      "source_gene": "string",\n'
        '      "target": "string",\n'
        '      "interaction_type": -1 | 1,\n'
        '      "pmid_citations": ["string"],\n'
        '      "confidence_score": 0.0,\n'
        '      "evidence_summary": "string",\n'
        '      "source_type": "deg" | "intermediate" | "pathway" | "prior" | "unknown",\n'
        '      "target_type": "deg" | "intermediate" | "pathway" | "prior" | "unknown",\n'
        '      "mechanistic_depth": 1 | 2\n'
        "    }\n"
        "  ],\n"
        '  "discovered_entities": [\n'
        "    {\n"
        '      "canonical_symbol": "string",\n'
        '      "aliases": ["string"],\n'
        '      "entity_type": "deg" | "intermediate" | "pathway" | "prior" | "unknown",\n'
        '      "sources": ["string"]\n'
        "    }\n"
        "  ],\n"
        '  "alias_hints": {"string": ["string"]},\n'
        '  "no_direct_effect": false,\n'
        '  "no_supported_edges": true,\n'
        '  "queried_targets": ["string"],\n'
        '  "raw_model": "string",\n'
        '  "phase": "discovery"\n'
        "}\n"
    )


def build_discovery_user_prompt(
    gene: str,
    deg_universe: Sequence[str],
    prior_knowledge: PriorKnowledgeSummary,
    grn_config: GrnConfig,
) -> str:
    candidate_universe = build_candidate_universe(deg_universe, prior_knowledge, grn_config)
    return (
        f"Research the seed DEG {gene} in PDAC.\n"
        f"Candidate node universe: {', '.join(candidate_universe)}.\n"
        "Find the most plausible mechanistic outgoing edges and bridge nodes for this DEG. Return JSON only."
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
        "Each kept edge must include `evidence_scores`, `pmid_citations`, and a concise evidence summary.\n"
        "No markdown or narrative outside the JSON.\n"
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
        }
        for edge in discovery_result.interactions
    ]
    return (
        "Independently verify the following candidate edges and reject unsupported ones.\n"
        f"Candidate edges: {json.dumps(candidate_edges, ensure_ascii=True)}\n"
        f"Discovered entities: {json.dumps([entity.model_dump() for entity in discovery_result.discovered_entities], ensure_ascii=True)}\n"
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
            discovery_result = GeneResearchResult.model_validate_json(discovery_cache_path.read_text(encoding="utf-8"))
            verification_result = GeneResearchResult.model_validate_json(verification_cache_path.read_text(encoding="utf-8"))
            return discovery_result, verification_result

        if self._openai_disabled_reason is None:
            discovery_result = await self._call_openai(
                model_name=grn_config.model,
                instructions=build_discovery_system_prompt(gene, deg_universe, prior_knowledge, grn_config),
                prompt=build_discovery_user_prompt(gene, deg_universe, prior_knowledge, grn_config),
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
                if discovery_result.interactions:
                    verified = await self._call_openai(
                        model_name=grn_config.parser_model,
                        instructions=build_verification_system_prompt(gene, discovery_result, grn_config),
                        prompt=build_verification_user_prompt(discovery_result),
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

    async def _call_openai(self, model_name: str, instructions: str, prompt: str) -> GeneResearchResult | None:
        try:
            response = await self.client.responses.parse(
                model=model_name,
                instructions=instructions,
                input=prompt,
                tools=[{"type": "web_search_preview"}],
                text_format=GeneResearchResult,
                max_output_tokens=2_000,
            )
            return _extract_parsed_response(response)
        except Exception as exc:
            if "invalid_api_key" in str(exc).lower() or "incorrect api key" in str(exc).lower():
                self._openai_disabled_reason = str(exc)
            return None


class AnthropicResearchClient:
    def __init__(self, settings: Settings, cache_dir: Path) -> None:
        if settings.anthropic_api_key is None:
            raise ValueError("ANTHROPIC_API_KEY is required for Anthropic-backed GRN extraction.")
        self.settings = settings
        self.cache_dir = ensure_directory(cache_dir)
        self.discovery_cache_dir = ensure_directory(self.cache_dir / "discovery")
        self.verification_cache_dir = ensure_directory(self.cache_dir / "verification")
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
            discovery_result = GeneResearchResult.model_validate_json(discovery_cache_path.read_text(encoding="utf-8"))
            verification_result = GeneResearchResult.model_validate_json(verification_cache_path.read_text(encoding="utf-8"))
            return discovery_result, verification_result

        discovery_result = await self._call_model(
            build_discovery_system_prompt(gene, deg_universe, prior_knowledge, grn_config),
            build_discovery_user_prompt(gene, deg_universe, prior_knowledge, grn_config),
            grn_config.model,
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
        if discovery_result.interactions:
            verified = await self._call_model(
                build_verification_system_prompt(gene, discovery_result, grn_config),
                build_verification_user_prompt(discovery_result),
                grn_config.parser_model,
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

    async def _call_model(self, system_prompt: str, user_prompt: str, model_name: str) -> GeneResearchResult | None:
        try:
            message = await self.client.messages.create(
                model=model_name,
                max_tokens=2_000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0,
                tools=[
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 4,
                        "allowed_callers": ["direct"],
                    }
                ],
            )
            text_payload = _extract_anthropic_text(message.content)
            if not text_payload:
                return None
            parsed = _parse_json_payload(text_payload)
            return GeneResearchResult.model_validate(parsed)
        except Exception:
            return None


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


def _extract_parsed_response(response: ParsedResponse[GeneResearchResult]) -> GeneResearchResult | None:
    for output_item in response.output:
        if not hasattr(output_item, "content"):
            continue
        for content in output_item.content:
            parsed = getattr(content, "parsed", None)
            if parsed is not None:
                return parsed
    return None


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
    if payload.startswith("```"):
        payload = payload.strip("`")
        payload = payload.removeprefix("json").strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", payload, flags=re.DOTALL)
        if match is None:
            raise
        return json.loads(match.group(0))


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
        normalized_edges.values(),
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
