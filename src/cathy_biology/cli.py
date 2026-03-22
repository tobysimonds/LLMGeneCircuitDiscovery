from __future__ import annotations

from pathlib import Path
import sys

import typer

from cathy_biology.blog_site import build_blog_site
from cathy_biology.config import Settings, load_pipeline_config
from cathy_biology.llm_knockout import run_anthropic_knockout_ranking
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
    disable_verification: bool = typer.Option(
        False,
        help="Skip the verification pass and use discovery results directly.",
    ),
) -> None:
    settings = Settings()
    pipeline_config = load_pipeline_config(config)
    if research_backend is not None:
        pipeline_config.grn.research_backend = research_backend
    if disable_verification:
        pipeline_config.grn.enable_verification = False
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


@app.command("build-blog-site")
def build_blog_site_command(
    run_dir: Path = typer.Option(..., exists=True, file_okay=False, readable=True, help="Completed artifact directory."),
    output_dir: Path = typer.Option(..., file_okay=False, help="Directory where the blog site should be written."),
    title: str = typer.Option("From Differential Expression to Virtual Knockouts", help="Blog post title."),
) -> None:
    site_dir = build_blog_site(run_dir, output_dir, title=title)
    typer.echo(f"Built blog site in {site_dir}")
    typer.echo(f"Serve locally with: uv run python -m http.server -d {site_dir}")


@app.command("rank-llm-knockouts")
def rank_llm_knockouts(
    run_dir: Path = typer.Option(..., exists=True, file_okay=False, readable=True, help="Completed artifact directory."),
    model: str = typer.Option("claude-opus-4-6", help="Anthropic model to use for graph-based knockout ranking."),
) -> None:
    settings = Settings()
    ranking = run_anthropic_knockout_ranking(run_dir, settings, model_name=model)
    typer.echo(f"LLM knockout ranking written to {run_dir / 'llm_knockout_opus' / 'rankings.json'}")
    typer.echo(f"Final recommendation: {', '.join(ranking.final_recommendation) or 'none'}")
    for candidate in ranking.candidates:
        typer.echo(
            f"#{candidate.rank}: {' + '.join(candidate.knocked_out_genes)} | conf={candidate.confidence_score:.2f} | toxicity={candidate.toxicity_risk}"
        )


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1].startswith("-"):
        sys.argv.insert(1, "run")
    app()
