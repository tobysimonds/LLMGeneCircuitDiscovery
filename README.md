# Cathy Biology

`cathy-biology` is a uv-managed Python application that implements a three-stage DEG-to-network target discovery pipeline:

1. Load scRNA-seq data, quality filter it, and extract top DEGs with Scanpy.
2. Use OpenAI-backed literature search to convert DEGs into a directed regulatory graph.
3. Run a brute-force Boolean knockout search and benchmark predicted targets against DepMap pancreatic models.

## Run

```bash
uv sync
uv run run_pipeline.py --config configs/pdac_gse242230.toml
```

To switch GRN backends at runtime:

```bash
uv run run_pipeline.py --config configs/pdac_gse242230.toml --research-backend anthropic
uv run run_pipeline.py --config configs/pdac_gse242230.toml --research-backend pubmed
```

The main outputs are written to a timestamped directory under `artifacts/`:

- `top_degs.csv`
- `gene_interactions.json`
- `regulatory_graph.json`
- `knockout_hits.json`
- `benchmark_report.json`
- `summary.json`

## Configs

- `configs/pdac_gse242230.toml`: real PDAC pipeline using GEO accession `GSE242230`.
- `configs/pbmc_smoke.toml`: fast smoke configuration used for local development.

## Testing

```bash
uv run pytest
```
