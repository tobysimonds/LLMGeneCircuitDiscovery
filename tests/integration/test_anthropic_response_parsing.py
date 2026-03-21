from __future__ import annotations

from cathy_biology.grn import _parse_json_payload


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
