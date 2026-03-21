from __future__ import annotations

from pathlib import Path
import sys

import typer

from cathy_biology.config import Settings, load_pipeline_config
from cathy_biology.pipeline import run_pipeline
from cathy_biology.site import build_results_site

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    config: Path = typer.Option(..., exists=True, dir_okay=False, readable=True, help="Path to a TOML pipeline config."),
    output_dir: Path | None = typer.Option(None, file_okay=False, help="Optional output directory."),
    research_backend: str | None = typer.Option(
        None,
        help="Override the GRN backend: openai, anthropic, or pubmed.",
    ),
) -> None:
    settings = Settings()
    pipeline_config = load_pipeline_config(config)
    if research_backend is not None:
        pipeline_config.grn.research_backend = research_backend
    summary = run_pipeline(pipeline_config, settings, output_dir)
    typer.echo(f"Completed run in {summary.output_dir}")
    typer.echo(f"Dataset cells: {summary.dataset_cells}, genes: {summary.dataset_genes}")
    typer.echo(f"Top DEG count: {len(summary.degs)}")
    typer.echo(
        "Research model usage: "
        + ", ".join(f"{model}={count}" for model, count in summary.research_execution.result_model_counts.items())
    )
    typer.echo(f"Research fallbacks: {summary.research_execution.fallback_gene_count}")
    typer.echo(
        f"Prior nodes/edges: {summary.prior_knowledge.node_count}/{summary.prior_knowledge.edge_count}"
    )
    typer.echo(f"Selected experiment: {summary.selected_experiment}")
    typer.echo(f"Graph nodes/edges: {summary.graph_nodes}/{summary.graph_edges}")
    typer.echo(f"Knockout hits found: {len(summary.knockout_hits)}")
    typer.echo(f"DepMap models benchmarked: {summary.benchmark_report.model_count}")


@app.command("build-site")
def build_site(
    primary_run: Path = typer.Option(..., exists=True, file_okay=False, readable=True, help="Primary artifact directory."),
    output_dir: Path = typer.Option(..., file_okay=False, help="Directory where the static site should be written."),
    baseline_run: Path | None = typer.Option(
        None,
        exists=True,
        file_okay=False,
        readable=True,
        help="Optional baseline artifact directory for comparison.",
    ),
    title: str = typer.Option("PDAC Target Discovery Atlas", help="Website title."),
) -> None:
    site_dir = build_results_site(primary_run, output_dir, baseline_run_dir=baseline_run, title=title)
    typer.echo(f"Built site in {site_dir}")
    typer.echo(f"Serve locally with: uv run python -m http.server -d {site_dir}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1].startswith("-"):
        sys.argv.insert(1, "run")
    app()
