from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Protocol, Sequence
from xml.etree import ElementTree

import httpx
from openai import AsyncOpenAI
from openai.types.responses import ParsedResponse

from cathy_biology.config import GrnConfig, Settings
from cathy_biology.models import GeneInteraction, GeneResearchResult
from cathy_biology.utils import ensure_directory, write_json


class ResearchClient(Protocol):
    async def research_genes(self, genes: Sequence[str], grn_config: GrnConfig) -> list[GeneResearchResult]:
        ...


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

        allowed_targets = [grn_config.target_oncogene, *grn_config.immediate_downstream_effectors]
        instructions = (
            "You are a computational biology agent. Research only peer-reviewed or primary biomedical sources. "
            "Determine whether the queried DEG directly activates (+1), inhibits (-1), or has no direct effect (0) "
            "on the target oncogene or one of the allowed immediate downstream effectors in PDAC. "
            "Return direct literature-supported interactions only. Prefer PubMed identifiers when possible."
        )
        prompt = (
            f"Gene: {gene}\n"
            f"Cancer context: {grn_config.context}\n"
            f"Target oncogene: {grn_config.target_oncogene}\n"
            f"Allowed pathway targets: {', '.join(allowed_targets)}\n"
            "Return a structured response with:\n"
            "- `source_gene`\n"
            "- `target_oncogene`\n"
            "- `context`\n"
            "- `queried_targets`\n"
            "- `no_direct_effect`\n"
            "- `interactions`: zero or more direct edges, each with source_gene, target, interaction_type, pmid_citations, confidence_score, evidence_summary.\n"
            "If there is no credible direct edge, set `no_direct_effect=true` and return an empty `interactions` list."
        )

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
