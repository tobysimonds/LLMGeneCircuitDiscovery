# Cathy Biology

`cathy-biology` is a uv-managed Python application that implements a three-stage DEG-to-network target discovery pipeline:

1. Load scRNA-seq data, quality filter it, and extract top DEGs with Scanpy.
2. Use OpenAI-, Anthropic-, or PubMed-backed discovery/verification literature search to convert DEGs into a mechanistic regulatory graph.
3. Seed the graph with curated priors, run weighted Boolean knockout search experiments, and benchmark predicted targets against DepMap pancreatic models.

## Run

```bash
uv sync
uv run run_pipeline.py --config configs/pdac_gse242230.toml
uv run run_pipeline.py --config configs/pdac_gse242230_anthropic.toml
```

To switch GRN backends at runtime:

```bash
uv run run_pipeline.py --config configs/pdac_gse242230.toml --research-backend anthropic
uv run run_pipeline.py --config configs/pdac_gse242230.toml --research-backend pubmed
```

The main outputs are written to a timestamped directory under `artifacts/`:

- `top_degs.csv`
- `gene_interactions.json`
- `discovery_interactions.json`
- `prior_knowledge.json`
- `pre_simulation_benchmark.json`
- `regulatory_graph.json`
- `knockout_hits.json`
- `benchmark_report.json`
- `experiment_report.json`
- `research_execution.json`
- `summary.json`

## Configs

- `configs/pdac_gse242230.toml`: real PDAC pipeline using GEO accession `GSE242230`.
- `configs/pdac_gse242230_anthropic.toml`: PDAC pipeline using Anthropic `claude-sonnet-4-6` for discovery and `claude-opus-4-6` for verification.
- `configs/pdac_gse242230_anthropic_opus.toml`: PDAC pipeline using Anthropic `claude-opus-4-6`.
- `configs/pbmc_smoke.toml`: fast smoke configuration used for local development.

## Testing

```bash
uv run pytest
```
