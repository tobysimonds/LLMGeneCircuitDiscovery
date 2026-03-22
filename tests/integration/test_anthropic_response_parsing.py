from __future__ import annotations

from pathlib import Path

from llmgenecircuitdiscovery.aliases import GeneAliasResolver
from llmgenecircuitdiscovery.config import GrnConfig, Settings
from llmgenecircuitdiscovery.grn import AnthropicResearchClient, _coerce_research_result, _normalize_research_result, _parse_json_payload
from llmgenecircuitdiscovery.models import GeneInteraction, GeneResearchResult, PriorKnowledgeSummary


def test_parse_json_payload_recovers_anthropic_narrative_wrapped_json() -> None:
    payload = """
    I found no direct mechanistic edge in PDAC.

    ```json
    {
      "source_gene": "SOX2",
      "target_oncogene": "KRAS",
      "context": "Pancreatic Ductal Adenocarcinoma",
      "interactions": [],
      "no_direct_effect": true,
      "queried_targets": ["KRAS", "RAF1"],
      "raw_model": "claude-sonnet-4-6"
    }
    ```
    """

    parsed = _parse_json_payload(payload)

    assert parsed["source_gene"] == "SOX2"
    assert parsed["no_direct_effect"] is True
    assert parsed["raw_model"] == "claude-sonnet-4-6"


def test_parse_json_payload_prefers_fenced_json_block() -> None:
    payload = """
    Preliminary notes before final answer.

    ```json
    {
      "interactions": [],
      "alias_hints": {},
      "no_supported_edges": true
    }
    ```
    """

    parsed = _parse_json_payload(payload)

    assert parsed["no_supported_edges"] is True
    assert parsed["interactions"] == []


def test_coerce_research_result_accepts_compact_discovery_payload() -> None:
    payload = {
        "interactions": [
            {
                "source_gene": "SOX2",
                "target": "EGFR",
                "interaction_type": 1,
                "confidence_score": 0.71,
                "evidence_summary": "SOX2 increases EGFR signaling in pancreatic cancer models.",
                "pmid_citations": ["12345678"],
                "source_refs": ["PMID:12345678"],
                "mechanistic_depth": 1,
            }
        ],
        "discovered_entities": [{"canonical_symbol": "EGFR", "aliases": ["ERBB1"], "entity_type": "intermediate"}],
        "alias_hints": {"ERBB1": ["EGFR"]},
    }

    result = _coerce_research_result(
        payload,
        seed_gene="SOX2",
        target_oncogene="KRAS",
        context="Pancreatic Ductal Adenocarcinoma",
        phase="discovery",
        model_name="claude-sonnet-4-6",
    )

    assert result.source_gene == "SOX2"
    assert result.target_oncogene == "KRAS"
    assert result.context == "Pancreatic Ductal Adenocarcinoma"
    assert result.phase == "discovery"
    assert result.raw_model == "claude-sonnet-4-6"
    assert result.interactions[0].target == "EGFR"
    assert result.interactions[0].source_refs == ["PMID:12345678"]
    assert result.discovered_entities[0].sources == []


def test_anthropic_client_writes_raw_exchange_artifacts(tmp_path: Path) -> None:
    client = AnthropicResearchClient(
        Settings(
            anthropic_api_key="test-key",
            data_dir=tmp_path / "data",
            artifacts_dir=tmp_path / "artifacts",
        ),
        tmp_path / "anthropic-cache",
    )

    client._write_exchange_artifacts(  # pyright: ignore[reportPrivateUsage]
        gene="SOX2",
        phase="discovery",
        kind="response",
        payload={"text": "{\"interactions\": []}"},
    )

    assert (tmp_path / "anthropic-cache" / "raw" / "discovery" / "SOX2" / "response.json").exists()


def test_coerce_research_result_sanitizes_alias_hints_and_mechanistic_depth() -> None:
    payload = {
        "interactions": [
            {
                "source_gene": "NTS",
                "target": "RAF1",
                "interaction_type": 1,
                "confidence_score": 0.91,
                "evidence_summary": "NTS activates RAF1 through PKC signaling.",
                "pmid_citations": [12750255],
                "source_refs": ["https://pubmed.ncbi.nlm.nih.gov/12750255/"],
                "mechanistic_depth": 3,
            }
        ],
        "discovered_entities": [{"canonical_symbol": "NTSR1", "aliases": "NTR1", "entity_type": "intermediate"}],
        "alias_hints": {"NTSR1": "neurotensin receptor 1"},
    }

    result = _coerce_research_result(
        payload,
        seed_gene="NTS",
        target_oncogene="KRAS",
        context="Pancreatic Ductal Adenocarcinoma",
        phase="discovery",
        model_name="claude-sonnet-4-6",
    )

    assert result.alias_hints["NTSR1"] == ["neurotensin receptor 1"]
    assert result.interactions[0].mechanistic_depth == 2
    assert result.interactions[0].pmid_citations == ["12750255"]
    assert result.discovered_entities[0].aliases == ["NTR1"]


def test_normalize_research_result_keeps_only_seed_anchored_edges(tmp_path: Path) -> None:
    result = GeneResearchResult(
        source_gene="EFS",
        target_oncogene="KRAS",
        context="Pancreatic Ductal Adenocarcinoma",
        interactions=[
            GeneInteraction(source_gene="EFS", target="SRC", interaction_type=1, confidence_score=0.9),
            GeneInteraction(source_gene="SRC", target="MMP2", interaction_type=1, confidence_score=0.8),
            GeneInteraction(source_gene="TP63", target="KRT6A", interaction_type=1, confidence_score=0.95),
        ],
        raw_model="claude-sonnet-4-6",
        phase="discovery",
    )

    normalized = _normalize_research_result(
        result,
        seed_gene="EFS",
        deg_universe=["EFS", "MMP2", "TP63", "KRT6A"],
        prior_knowledge=PriorKnowledgeSummary(node_count=0, edge_count=0, source_counts={}, nodes=[], edges=[]),
        grn_config=GrnConfig(),
        model_name="claude-sonnet-4-6",
        phase="discovery",
        alias_resolver=GeneAliasResolver(tmp_path / "aliases"),
    )

    kept_edges = {(edge.source_gene, edge.target) for edge in normalized.interactions}
    assert ("EFS", "SRC") in kept_edges
    assert ("SRC", "MMP2") in kept_edges
    assert ("TP63", "KRT6A") not in kept_edges
