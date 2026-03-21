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

from cathy_biology.config import GrnConfig, Settings
from cathy_biology.models import GeneInteraction, GeneResearchResult
from cathy_biology.utils import ensure_directory, write_json


class ResearchClient(Protocol):
    async def research_genes(self, genes: Sequence[str], grn_config: GrnConfig) -> list[GeneResearchResult]:
        ...


def build_research_system_prompt(gene: str, grn_config: GrnConfig) -> str:
    allowed_targets = [grn_config.target_oncogene, *grn_config.immediate_downstream_effectors]
    return (
        "You are a senior computational oncology research agent building a causal gene-regulatory network for a "
        "three-stage target-discovery pipeline.\n\n"
        "Pipeline context:\n"
        "1. Upstream code has already processed a PDAC single-cell RNA-seq dataset and identified top differentially "
        f"expressed genes. The current DEG under review is {gene}.\n"
        f"2. The network is centered on the oncogene {grn_config.target_oncogene} and its immediate downstream "
        "signaling effectors.\n"
        "3. Your output will be consumed by deterministic code that builds a directed Boolean network and then brute-force "
        "tests 1-, 2-, and 3-gene knockouts.\n"
        "4. Incorrect edges are more harmful than missing edges. Be conservative.\n\n"
        f"Disease context: {grn_config.context}.\n"
        f"Allowed target nodes for direct edges: {', '.join(allowed_targets)}.\n\n"
        "Research task:\n"
        f"- Determine whether {gene} directly UP-REGULATES (+1), DOWN-REGULATES (-1), or has NO DIRECT EFFECT (0) on "
        f"{grn_config.target_oncogene} or one of the allowed target nodes in the specific context of PDAC or closely "
        "relevant pancreatic cancer signaling biology.\n"
        "- Use web search and prioritize peer-reviewed primary literature, PubMed-indexed articles, and highly credible "
        "mechanistic reviews only when they clearly summarize direct evidence.\n"
        "- Prefer direct mechanistic statements over broad associations, co-expression, prognostic correlations, or vague "
        "pathway mentions.\n"
        "- Only include an edge when the literature supports directionality from the source DEG to the target node.\n"
        "- If evidence is mixed, weak, indirect, tissue-mismatched, or only implies pathway membership, omit the edge.\n"
        "- If the DEG affects a downstream effector rather than KRAS itself, that is acceptable only if the target is one "
        "of the allowed nodes.\n\n"
        "Evidence rules:\n"
        "- Direct transcriptional activation, repression, phosphorylation-driven activation, pathway stimulation, or "
        "experimentally supported inhibition are acceptable.\n"
        "- Biomarker associations, expression signatures, survival correlations, or generic statements like 'associated "
        "with the MAPK pathway' are not sufficient by themselves.\n"
        "- PDAC evidence is best. If PDAC-specific evidence does not exist, you may use closely related pancreatic cancer "
        "mechanistic evidence only if the causal claim is still strong and you make that clear in the evidence summary.\n"
        "- Prefer PubMed IDs in `pmid_citations`. Use an empty list only if none can be confidently identified.\n\n"
        "Output contract:\n"
        "Return strict JSON only. No markdown, no prose before or after the JSON, no code fences.\n"
        "Use this exact schema:\n"
        "{\n"
        '  "source_gene": "string",\n'
        '  "target_oncogene": "string",\n'
        '  "context": "string",\n'
        '  "interactions": [\n'
        "    {\n"
        '      "source_gene": "string",\n'
        '      "target": "string",\n'
        '      "interaction_type": -1 | 0 | 1,\n'
        '      "pmid_citations": ["string"],\n'
        '      "confidence_score": 0.0,\n'
        '      "evidence_summary": "string"\n'
        "    }\n"
        "  ],\n"
        '  "no_direct_effect": true,\n'
        '  "queried_targets": ["string"],\n'
        '  "raw_model": "string"\n'
        "}\n\n"
        "Validation constraints:\n"
        "- `interactions` must be empty when `no_direct_effect` is true.\n"
        "- Every `target` must be one of the allowed target nodes.\n"
        "- `interaction_type` must be 1 for activation or -1 for inhibition. Do not emit 0-valued interactions.\n"
        "- `confidence_score` should reflect evidence quality and consistency, not optimism.\n"
        "- Set `raw_model` to the model name you used.\n"
    )


def build_research_user_prompt(gene: str, grn_config: GrnConfig) -> str:
    allowed_targets = [grn_config.target_oncogene, *grn_config.immediate_downstream_effectors]
    return (
        f"Research the DEG {gene} for the PDAC GRN.\n"
        f"Allowed targets: {', '.join(allowed_targets)}.\n"
        "Return only strict JSON following the provided schema. If you cannot support a direct edge with strong mechanistic "
        "literature evidence, return `no_direct_effect=true` and an empty `interactions` list."
    )


class OpenAIResearchClient:
    def __init__(self, settings: Settings, cache_dir: Path) -> None:
        if settings.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is required for OpenAI-backed GRN extraction.")
        self.settings = settings
        self.cache_dir = ensure_directory(cache_dir)
        self.fallback_client = PubMedHeuristicResearchClient(settings, cache_dir / "pubmed_fallback")
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.request_timeout_seconds,
        )
        self._openai_disabled_reason: str | None = None

    async def research_genes(self, genes: Sequence[str], grn_config: GrnConfig) -> list[GeneResearchResult]:
        semaphore = asyncio.Semaphore(grn_config.concurrency)

        async def run_gene(gene: str) -> GeneResearchResult:
            async with semaphore:
                return await self._research_gene(gene, grn_config)

        return await asyncio.gather(*(run_gene(gene) for gene in genes))

    async def _research_gene(self, gene: str, grn_config: GrnConfig) -> GeneResearchResult:
        cache_path = self.cache_dir / f"{gene}.json"
        if cache_path.exists():
            return GeneResearchResult.model_validate_json(cache_path.read_text(encoding="utf-8"))

        instructions = build_research_system_prompt(gene, grn_config)
        prompt = build_research_user_prompt(gene, grn_config)

        result: GeneResearchResult | None = None
        if self._openai_disabled_reason is None:
            result = await self._call_model(grn_config.model, instructions, prompt, grn_config)
            if result is None and grn_config.parser_model != grn_config.model:
                result = await self._call_model(grn_config.parser_model, instructions, prompt, grn_config)
        if result is None:
            result = await self.fallback_client.research_genes([gene], grn_config)
            result = result[0]

        result.source_gene = gene
        if not result.raw_model:
            result.raw_model = "pubmed-heuristic"
        write_json(cache_path, result.model_dump())
        return result

    async def _call_model(
        self,
        model: str,
        instructions: str,
        prompt: str,
        grn_config: GrnConfig,
    ) -> GeneResearchResult | None:
        try:
            response = await self.client.responses.parse(
                model=model,
                instructions=instructions,
                input=prompt,
                tools=[{"type": "web_search_preview"}],
                max_tool_calls=grn_config.max_tool_calls,
                text_format=GeneResearchResult,
                max_output_tokens=1_500,
            )
            parsed = _extract_parsed_response(response)
            if parsed is None:
                return None
            parsed.raw_model = model
            return parsed
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
        self.fallback_client = PubMedHeuristicResearchClient(settings, cache_dir / "pubmed_fallback")
        self.client = AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            timeout=settings.request_timeout_seconds,
        )

    async def research_genes(self, genes: Sequence[str], grn_config: GrnConfig) -> list[GeneResearchResult]:
        semaphore = asyncio.Semaphore(grn_config.concurrency)

        async def run_gene(gene: str) -> GeneResearchResult:
            async with semaphore:
                return await self._research_gene(gene, grn_config)

        return await asyncio.gather(*(run_gene(gene) for gene in genes))

    async def _research_gene(self, gene: str, grn_config: GrnConfig) -> GeneResearchResult:
        cache_path = self.cache_dir / f"{gene}.json"
        if cache_path.exists():
            return GeneResearchResult.model_validate_json(cache_path.read_text(encoding="utf-8"))

        system_prompt = build_research_system_prompt(gene, grn_config)
        user_prompt = build_research_user_prompt(gene, grn_config)

        result = await self._call_model(gene, system_prompt, user_prompt, grn_config)
        if result is None:
            fallback = await self.fallback_client.research_genes([gene], grn_config)
            result = fallback[0]
        write_json(cache_path, result.model_dump())
        return result

    async def _call_model(
        self,
        gene: str,
        system_prompt: str,
        user_prompt: str,
        grn_config: GrnConfig,
    ) -> GeneResearchResult | None:
        try:
            message = await self.client.messages.create(
                model=grn_config.model,
                max_tokens=1800,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                tools=[
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": grn_config.max_tool_calls,
                        "allowed_callers": ["direct"],
                    }
                ],
            )
            text_payload = _extract_anthropic_text(message.content)
            if not text_payload:
                return None
            parsed = _parse_json_payload(text_payload)
            result = GeneResearchResult.model_validate(parsed)
            result.source_gene = gene
            result.raw_model = grn_config.model
            return result
        except Exception:
            return None


class MockResearchClient:
    def __init__(self, mapping: dict[str, list[GeneInteraction]], target_oncogene: str, context: str) -> None:
        self.mapping = mapping
        self.target_oncogene = target_oncogene
        self.context = context

    async def research_genes(self, genes: Sequence[str], grn_config: GrnConfig) -> list[GeneResearchResult]:
        results: list[GeneResearchResult] = []
        for gene in genes:
            interactions = self.mapping.get(gene, [])
            results.append(
                GeneResearchResult(
                    source_gene=gene,
                    target_oncogene=self.target_oncogene,
                    context=self.context,
                    queried_targets=[self.target_oncogene, *grn_config.immediate_downstream_effectors],
                    interactions=interactions,
                    no_direct_effect=not interactions,
                    raw_model="mock",
                )
            )
        return results


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
        block_type = getattr(block, "type", "")
        if block_type == "text":
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


class PubMedHeuristicResearchClient:
    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    TARGET_ALIASES = {
        "KRAS": ["KRAS", "Kras"],
        "RAF1": ["RAF1", "RAF", "c-RAF"],
        "BRAF": ["BRAF", "B-Raf"],
        "MAP2K1": ["MAP2K1", "MEK1", "MEK"],
        "MAP2K2": ["MAP2K2", "MEK2", "MEK"],
        "MAPK1": ["MAPK1", "ERK2", "ERK"],
        "MAPK3": ["MAPK3", "ERK1", "ERK"],
        "PIK3CA": ["PIK3CA", "PI3K"],
        "AKT1": ["AKT1", "Akt1", "AKT", "Akt"],
    }
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
        "pathway",
        "axis",
        "signaling",
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

    async def research_genes(self, genes: Sequence[str], grn_config: GrnConfig) -> list[GeneResearchResult]:
        semaphore = asyncio.Semaphore(1)
        async with httpx.AsyncClient(
            timeout=self.settings.request_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "cathy-biology/0.1"},
        ) as client:
            async def run_gene(gene: str) -> GeneResearchResult:
                async with semaphore:
                    return await self._research_gene(client, gene, grn_config)

            return await asyncio.gather(*(run_gene(gene) for gene in genes))

    async def _research_gene(
        self,
        client: httpx.AsyncClient,
        gene: str,
        grn_config: GrnConfig,
    ) -> GeneResearchResult:
        cache_path = self.cache_dir / f"{gene}.json"
        if cache_path.exists():
            return GeneResearchResult.model_validate_json(cache_path.read_text(encoding="utf-8"))

        targets = [grn_config.target_oncogene, *grn_config.immediate_downstream_effectors]
        try:
            pmids = await self._search_pmids(client, gene, targets, grn_config.context)
            articles = await self._fetch_articles(client, pmids) if pmids else []
            interactions = self._infer_interactions(gene, targets, articles)
        except Exception:
            interactions = []
        result = GeneResearchResult(
            source_gene=gene,
            target_oncogene=grn_config.target_oncogene,
            context=grn_config.context,
            queried_targets=targets,
            interactions=interactions,
            no_direct_effect=not interactions,
            raw_model="pubmed-heuristic",
        )
        write_json(cache_path, result.model_dump())
        return result

    async def _search_pmids(
        self,
        client: httpx.AsyncClient,
        gene: str,
        targets: list[str],
        context: str,
    ) -> list[str]:
        context_query = f'"{context}"[Title/Abstract] OR PDAC[Title/Abstract] OR pancreas[Title/Abstract]'
        params = {
            "db": "pubmed",
            "retmode": "json",
            "retmax": "12",
            "term": f'"{gene}"[Title/Abstract] AND ({context_query})',
        }
        response = await self._get_with_retry(client, self.ESEARCH_URL, params=params)
        payload = response.json()
        return payload.get("esearchresult", {}).get("idlist", [])

    async def _fetch_articles(self, client: httpx.AsyncClient, pmids: list[str]) -> list[dict[str, str]]:
        params = {
            "db": "pubmed",
            "retmode": "xml",
            "id": ",".join(pmids),
        }
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
        targets: list[str],
        articles: list[dict[str, str]],
    ) -> list[GeneInteraction]:
        interactions: list[GeneInteraction] = []
        gene_lower = gene.lower()
        for target in targets:
            if target.upper() == gene.upper():
                continue
            aliases = [alias.lower() for alias in self.TARGET_ALIASES.get(target, [target])]
            target_lower = target.lower()
            activation_pmids: list[str] = []
            inhibition_pmids: list[str] = []
            evidence_sentences: list[str] = []
            for article in articles:
                article_lower = article["text"].lower()
                if gene_lower not in article_lower or not any(alias in article_lower for alias in aliases):
                    continue
                sentences = re.split(r"(?<=[.!?])\s+", article["text"])
                relevant_sentences = [
                    sentence
                    for sentence in sentences
                    if gene_lower in sentence.lower() and any(alias in sentence.lower() for alias in aliases)
                ]
                if not relevant_sentences:
                    relevant_sentences = [
                        sentence for sentence in sentences if any(alias in sentence.lower() for alias in aliases)
                    ][:2]
                if not relevant_sentences:
                    continue
                sentence_text = " ".join(relevant_sentences)
                lowered = sentence_text.lower()
                evidence_sentences.extend(relevant_sentences[:2])
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
            confidence = min(0.9, 0.25 + 0.15 * len(pmids))
            evidence_summary = " ".join(evidence_sentences[:2])
            interactions.append(
                GeneInteraction(
                    source_gene=gene,
                    target=target,
                    interaction_type=interaction_type,
                    pmid_citations=sorted(set(filter(None, pmids))),
                    confidence_score=confidence,
                    evidence_summary=evidence_summary,
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
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code == 429:
                    await asyncio.sleep(delay_seconds)
                    delay_seconds *= 2
                    continue
                raise
            except httpx.HTTPError as exc:
                last_error = exc
                await asyncio.sleep(delay_seconds)
                delay_seconds *= 2
                continue
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Failed to fetch {url}")
