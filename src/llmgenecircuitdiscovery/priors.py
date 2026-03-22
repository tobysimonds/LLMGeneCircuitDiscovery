from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import httpx

from llmgenecircuitdiscovery.aliases import GeneAliasResolver
from llmgenecircuitdiscovery.config import GrnConfig, Settings
from llmgenecircuitdiscovery.models import EvidenceClassScores, GeneInteraction, PriorKnowledgeSummary, ResolvedEntity
from llmgenecircuitdiscovery.utils import ensure_directory, write_json


class PriorKnowledgeBuilder:
    KEGG_GET_URL = "https://rest.kegg.jp/get/{pathway_id}"
    REACTOME_SEARCH_URL = "https://reactome.org/ContentService/search/query"
    REACTOME_CONTAINED_EVENTS_URL = "https://reactome.org/ContentService/data/pathway/{st_id}/containedEvents"
    REACTOME_ENHANCED_QUERY_URL = "https://reactome.org/ContentService/data/query/enhanced/{db_id}"
    PATHWAYCOMMONS_SEARCH_URL = "https://www.pathwaycommons.org/pc2/search"
    PATHWAYCOMMONS_GET_URL = "https://www.pathwaycommons.org/pc2/get"
    OMNIPATH_INTERACTIONS_URL = "https://omnipathdb.org/interactions"

    def __init__(self, settings: Settings, cache_dir: Path, alias_resolver: GeneAliasResolver) -> None:
        self.settings = settings
        self.cache_dir = ensure_directory(cache_dir)
        self.alias_resolver = alias_resolver

    def build(self, deg_genes: list[str], grn_config: GrnConfig) -> PriorKnowledgeSummary:
        if not grn_config.prior.enabled:
            return PriorKnowledgeSummary(node_count=0, edge_count=0, source_counts={})

        node_sources: dict[str, set[str]] = defaultdict(set)
        for symbol in {grn_config.target_oncogene, *grn_config.immediate_downstream_effectors, *deg_genes, *grn_config.prior.seed_nodes}:
            node_sources[symbol.upper()].add("seed")

        kegg_nodes = self._fetch_kegg_nodes(grn_config.prior.kegg_pathway_ids)
        reactome_nodes = self._fetch_reactome_nodes(grn_config.prior.pathway_keywords, grn_config.prior.reactome_pathways_per_keyword, grn_config.prior.reactome_events_per_pathway)
        pathwaycommons_nodes = self._fetch_pathwaycommons_nodes(grn_config.prior.pathway_keywords, grn_config.prior.pathwaycommons_pathways_per_keyword)

        for symbol in kegg_nodes:
            node_sources[symbol].add("KEGG")
        for symbol in reactome_nodes:
            node_sources[symbol].add("Reactome")
        for symbol in pathwaycommons_nodes:
            node_sources[symbol].add("PathwayCommons")

        ranked_prior_nodes = self._rank_prior_nodes(node_sources, deg_genes, grn_config)
        omnipath_edges = self._fetch_omnipath_edges(ranked_prior_nodes, grn_config.prior.omnipath_partner_chunk_size)

        node_entities = [
            ResolvedEntity(canonical_symbol=symbol, aliases=[symbol], entity_type=self._node_type(symbol, set(deg_genes), grn_config), sources=sorted(sources))
            for symbol, sources in sorted(node_sources.items())
            if symbol in ranked_prior_nodes
        ]
        edge_entities = []
        for edge in omnipath_edges:
            if edge.source_gene not in ranked_prior_nodes or edge.target not in ranked_prior_nodes:
                continue
            edge.source_type = self._node_type(edge.source_gene, set(deg_genes), grn_config)
            edge.target_type = self._node_type(edge.target, set(deg_genes), grn_config)
            edge.prior_support_sources = sorted(node_sources.get(edge.source_gene, set()) | node_sources.get(edge.target, set()))
            edge_entities.append(edge)

        source_counts = Counter(source for sources in node_sources.values() for source in sources)
        summary = PriorKnowledgeSummary(
            node_count=len(node_entities),
            edge_count=len(edge_entities),
            source_counts=dict(sorted(source_counts.items())),
            nodes=node_entities,
            edges=edge_entities,
        )
        write_json(self.cache_dir / "prior_knowledge.json", summary.model_dump())
        return summary

    def _fetch_kegg_nodes(self, pathway_ids: list[str]) -> set[str]:
        cache_path = self.cache_dir / "kegg_nodes.json"
        if cache_path.exists():
            return set(json.loads(cache_path.read_text(encoding="utf-8")))
        nodes: set[str] = set()
        with httpx.Client(timeout=self.settings.request_timeout_seconds, follow_redirects=True) as client:
            for pathway_id in pathway_ids:
                response = client.get(self.KEGG_GET_URL.format(pathway_id=pathway_id))
                response.raise_for_status()
                nodes.update(_parse_kegg_gene_symbols(response.text))
        cache_path.write_text(json.dumps(sorted(nodes), indent=2), encoding="utf-8")
        return nodes

    def _fetch_reactome_nodes(self, keywords: list[str], pathways_per_keyword: int, events_per_pathway: int) -> set[str]:
        cache_path = self.cache_dir / "reactome_nodes.json"
        if cache_path.exists():
            return set(json.loads(cache_path.read_text(encoding="utf-8")))
        nodes: set[str] = set()
        with httpx.Client(timeout=self.settings.request_timeout_seconds, follow_redirects=True) as client:
            for keyword in keywords:
                response = client.get(
                    self.REACTOME_SEARCH_URL,
                    params={"query": keyword, "species": "Homo sapiens", "types": "Pathway"},
                )
                response.raise_for_status()
                payload = response.json()
                results = payload.get("results", [])
                entries = results[0].get("entries", []) if results else []
                for entry in entries[:pathways_per_keyword]:
                    st_id = entry.get("stId")
                    if not st_id:
                        continue
                    events_response = client.get(self.REACTOME_CONTAINED_EVENTS_URL.format(st_id=st_id))
                    events_response.raise_for_status()
                    contained_events = [event for event in events_response.json() if isinstance(event, dict) and event.get("className") == "Reaction"]
                    for event in contained_events[:events_per_pathway]:
                        detailed = client.get(self.REACTOME_ENHANCED_QUERY_URL.format(db_id=event["dbId"]))
                        detailed.raise_for_status()
                        nodes.update(_extract_reactome_gene_symbols(detailed.json()))
        cache_path.write_text(json.dumps(sorted(nodes), indent=2), encoding="utf-8")
        return nodes

    def _fetch_pathwaycommons_nodes(self, keywords: list[str], pathways_per_keyword: int) -> set[str]:
        cache_path = self.cache_dir / "pathwaycommons_nodes.json"
        if cache_path.exists():
            return set(json.loads(cache_path.read_text(encoding="utf-8")))
        nodes: set[str] = set()
        with httpx.Client(timeout=self.settings.request_timeout_seconds, follow_redirects=True) as client:
            for keyword in keywords:
                response = client.get(
                    self.PATHWAYCOMMONS_SEARCH_URL,
                    params={"q": keyword, "format": "json", "type": "Pathway"},
                )
                response.raise_for_status()
                payload = response.json()
                for hit in payload.get("searchHit", [])[:pathways_per_keyword]:
                    uri = hit.get("uri")
                    if not uri:
                        continue
                    graph_response = client.get(self.PATHWAYCOMMONS_GET_URL, params={"uri": uri, "format": "JSONLD"})
                    graph_response.raise_for_status()
                    nodes.update(_extract_pathwaycommons_gene_symbols(graph_response.json()))
        cache_path.write_text(json.dumps(sorted(nodes), indent=2), encoding="utf-8")
        return nodes

    def _fetch_omnipath_edges(self, candidate_nodes: list[str], chunk_size: int) -> list[GeneInteraction]:
        cache_path = self.cache_dir / "omnipath_edges.json"
        if cache_path.exists():
            return [GeneInteraction.model_validate(item) for item in json.loads(cache_path.read_text(encoding="utf-8"))]
        all_edges: dict[tuple[str, str, int], GeneInteraction] = {}
        with httpx.Client(timeout=self.settings.request_timeout_seconds, follow_redirects=True) as client:
            for chunk in _chunk(candidate_nodes, chunk_size):
                response = client.get(
                    self.OMNIPATH_INTERACTIONS_URL,
                    params={
                        "partners": ",".join(chunk),
                        "genesymbols": 1,
                        "format": "json",
                        "fields": "sources,references,curation_effort",
                    },
                )
                response.raise_for_status()
                payload = response.json()
                for item in payload:
                    source = str(item.get("source_genesymbol", "")).upper()
                    target = str(item.get("target_genesymbol", "")).upper()
                    if not source or not target:
                        continue
                    if source == target:
                        continue
                    if not item.get("is_directed", False):
                        continue
                    interaction_type = 1 if item.get("consensus_stimulation") or item.get("is_stimulation") else -1 if item.get("consensus_inhibition") or item.get("is_inhibition") else 0
                    if interaction_type == 0:
                        continue
                    references = [str(reference) for reference in _as_list(item.get("references"))]
                    sources = [str(source_name) for source_name in _as_list(item.get("sources"))]
                    confidence = min(0.95, 0.3 + 0.08 * len(sources) + 0.04 * len(references))
                    key = (source, target, interaction_type)
                    edge = GeneInteraction(
                        source_gene=source,
                        target=target,
                        interaction_type=interaction_type,
                        pmid_citations=sorted(set(filter(None, references))),
                        confidence_score=confidence,
                        evidence_summary=f"Curated prior interaction from OmniPath ({', '.join(sources[:3]) or 'OmniPath'}).",
                        evidence_scores=EvidenceClassScores(prior_supported=1.0),
                        provenance_sources=["OmniPath"],
                        mechanistic_depth=1,
                    )
                    previous = all_edges.get(key)
                    if previous is None or edge.confidence_score > previous.confidence_score:
                        all_edges[key] = edge
        serialized = [edge.model_dump() for edge in sorted(all_edges.values(), key=lambda item: (item.source_gene, item.target, item.interaction_type))]
        cache_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")
        return [GeneInteraction.model_validate(item) for item in serialized]

    def _rank_prior_nodes(self, node_sources: dict[str, set[str]], deg_genes: list[str], grn_config: GrnConfig) -> list[str]:
        deg_set = {gene.upper() for gene in deg_genes}
        pathway_set = {grn_config.target_oncogene.upper(), *(node.upper() for node in grn_config.immediate_downstream_effectors)}
        seed_set = {node.upper() for node in grn_config.prior.seed_nodes}
        ranked = sorted(
            node_sources,
            key=lambda symbol: (
                symbol not in deg_set,
                symbol not in pathway_set,
                symbol not in seed_set,
                -len(node_sources[symbol]),
                symbol,
            ),
        )
        always_keep = deg_set | pathway_set | seed_set
        selected: list[str] = []
        for symbol in ranked:
            if symbol in always_keep or len(selected) < grn_config.prior.prior_node_limit:
                selected.append(symbol)
        return sorted(set(selected))

    def _node_type(self, symbol: str, deg_set: set[str], grn_config: GrnConfig) -> str:
        symbol_upper = symbol.upper()
        if symbol_upper in {gene.upper() for gene in deg_set}:
            return "deg"
        if symbol_upper == grn_config.target_oncogene.upper() or symbol_upper in {
            node.upper() for node in grn_config.immediate_downstream_effectors
        }:
            return "pathway"
        return "prior"


def _parse_kegg_gene_symbols(payload: str) -> set[str]:
    collecting = False
    symbols: set[str] = set()
    for line in payload.splitlines():
        header = line[:12].strip()
        body = line[12:].strip()
        if header == "GENE":
            collecting = True
        elif header and header != "GENE":
            collecting = False
        if not collecting or not body:
            continue
        parts = body.split(None, 1)
        if len(parts) < 2:
            continue
        symbol = parts[1].split(";", 1)[0].strip()
        if symbol:
            symbols.add(symbol.upper())
    return symbols


def _extract_reactome_gene_symbols(payload: dict) -> set[str]:
    gene_symbols: set[str] = set()
    for key in ["input", "output", "regulatedBy"]:
        for entity in _as_list(payload.get(key)):
            gene_symbols.update(_extract_gene_candidates_from_reactome_entity(entity))
    catalyst = payload.get("catalystActivity")
    for item in _as_list(catalyst):
        if isinstance(item, dict):
            gene_symbols.update(_extract_gene_candidates_from_reactome_entity(item.get("physicalEntity")))
    return {symbol.upper() for symbol in gene_symbols if symbol}


def _extract_gene_candidates_from_reactome_entity(entity: object) -> set[str]:
    if not isinstance(entity, dict):
        return set()
    candidates: set[str] = set()
    names = [str(name) for name in _as_list(entity.get("name"))]
    display_name = str(entity.get("displayName", ""))
    for text in [display_name, *names]:
        for candidate in re.findall(r"\b[A-Z0-9-]{2,12}\b", text.upper()):
            if candidate not in {"ATP", "DNA", "RNA", "GTP", "GDP"}:
                candidates.add(candidate)
    for sub_key in ["physicalEntity", "regulator", "hasComponent", "hasMember"]:
        for sub_entity in _as_list(entity.get(sub_key)):
            candidates.update(_extract_gene_candidates_from_reactome_entity(sub_entity))
    return candidates


def _extract_pathwaycommons_gene_symbols(payload: dict) -> set[str]:
    symbols: set[str] = set()
    for item in payload.get("@graph", []):
        if item.get("@type") != "bp:ProteinReference":
            continue
        for xref in _as_list(item.get("xref")):
            if not isinstance(xref, str):
                continue
            match = re.search(r"hgnc_symbol_([^_]+)_identity", xref)
            if match:
                symbols.add(match.group(1).upper())
        if not symbols:
            for name in _as_list(item.get("name")):
                text = str(name).upper()
                if re.fullmatch(r"[A-Z0-9-]{2,12}", text):
                    symbols.add(text)
    return symbols


def _chunk(items: list[str], chunk_size: int) -> list[list[str]]:
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
