from __future__ import annotations

from pathlib import Path

import typer

from cathy_biology.config import Settings, load_pipeline_config
from cathy_biology.pipeline import run_pipeline

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    config: Path = typer.Option(..., exists=True, dir_okay=False, readable=True, help="Path to a TOML pipeline config."),
    output_dir: Path | None = typer.Option(None, file_okay=False, help="Optional output directory."),
) -> None:
    settings = Settings()
    pipeline_config = load_pipeline_config(config)
    summary = run_pipeline(pipeline_config, settings, output_dir)
    typer.echo(f"Completed run in {summary.output_dir}")
    typer.echo(f"Dataset cells: {summary.dataset_cells}, genes: {summary.dataset_genes}")
    typer.echo(f"Top DEG count: {len(summary.degs)}")
    typer.echo(f"Graph nodes/edges: {summary.graph_nodes}/{summary.graph_edges}")
    typer.echo(f"Knockout hits found: {len(summary.knockout_hits)}")
    typer.echo(f"DepMap models benchmarked: {summary.benchmark_report.model_count}")


def main() -> None:
    app()
