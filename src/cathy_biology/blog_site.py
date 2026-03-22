from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from cathy_biology.site import _attach_graph_evidence, _build_edge_evidence_index, _build_full_edge_index, _normalize_graph
from cathy_biology.utils import ensure_directory, write_json

REQUIRED_RUN_FILES = {
    "summary": "summary.json",
    "top_degs": "top_degs.json",
    "analysis_interactions": "analysis_interactions.json",
    "regulatory_graph_full": "regulatory_graph_full.json",
    "regulatory_graph": "regulatory_graph.json",
    "regulatory_graph_projected": "regulatory_graph_projected.json",
    "deg_graph_with_llm": "deg_graph_with_llm.json",
    "deg_graph_prior_only": "deg_graph_prior_only.json",
    "benchmark_report": "benchmark_report.json",
    "experiment_report": "experiment_report.json",
    "research_execution": "research_execution.json",
    "knockout_hits": "knockout_hits.json",
    "pre_simulation_benchmark": "pre_simulation_benchmark.json",
    "llm_knockout": "llm_knockout_opus/rankings.json",
    "deg_graph_with_llm_png": "deg_graph_with_llm.png",
    "deg_graph_without_llm_png": "deg_graph_without_llm.png",
}


def build_blog_site(
    run_dir: Path,
    output_dir: Path,
    *,
    title: str = "From Differential Expression to Virtual Knockouts",
) -> Path:
    output_dir = ensure_directory(output_dir)
    data_dir = ensure_directory(output_dir / "data")

    bundle = _build_blog_bundle(run_dir, data_dir)
    write_json(data_dir / "post_bundle.json", bundle)

    html = HTML_TEMPLATE.replace("__SITE_TITLE__", escape(title))
    (output_dir / "index.html").write_text(html, encoding="utf-8")
    (output_dir / "styles.css").write_text(STYLESHEET, encoding="utf-8")
    (output_dir / "app.js").write_text(APP_SCRIPT, encoding="utf-8")
    return output_dir


def _build_blog_bundle(run_dir: Path, data_dir: Path) -> dict[str, Any]:
    run_copy_dir = ensure_directory(data_dir / "run")
    for rel_path in REQUIRED_RUN_FILES.values():
        source = run_dir / rel_path
        if not source.exists():
            raise FileNotFoundError(f"Run directory {run_dir} is missing required artifact {rel_path}.")
        target = run_copy_dir / Path(rel_path).name
        shutil.copy2(source, target)

    summary = json.loads((run_copy_dir / "summary.json").read_text(encoding="utf-8"))
    top_degs = json.loads((run_copy_dir / "top_degs.json").read_text(encoding="utf-8"))
    analysis_interactions = json.loads((run_copy_dir / "analysis_interactions.json").read_text(encoding="utf-8"))
    benchmark = json.loads((run_copy_dir / "benchmark_report.json").read_text(encoding="utf-8"))
    pre_benchmark = json.loads((run_copy_dir / "pre_simulation_benchmark.json").read_text(encoding="utf-8"))
    experiments = json.loads((run_copy_dir / "experiment_report.json").read_text(encoding="utf-8"))
    research_execution = json.loads((run_copy_dir / "research_execution.json").read_text(encoding="utf-8"))
    knockout_hits = json.loads((run_copy_dir / "knockout_hits.json").read_text(encoding="utf-8"))
    llm_knockout = json.loads((run_copy_dir / "rankings.json").read_text(encoding="utf-8"))

    full_graph = _normalize_graph(json.loads((run_copy_dir / "regulatory_graph_full.json").read_text(encoding="utf-8")))
    graphs = {
        "selected": _normalize_graph(json.loads((run_copy_dir / "regulatory_graph.json").read_text(encoding="utf-8"))),
        "projected": _normalize_graph(json.loads((run_copy_dir / "regulatory_graph_projected.json").read_text(encoding="utf-8"))),
        "deg_llm": _normalize_graph(json.loads((run_copy_dir / "deg_graph_with_llm.json").read_text(encoding="utf-8"))),
        "deg_prior": _normalize_graph(json.loads((run_copy_dir / "deg_graph_prior_only.json").read_text(encoding="utf-8"))),
    }
    edge_evidence = _build_edge_evidence_index(analysis_interactions)
    full_edge_index = _build_full_edge_index(full_graph)
    for graph in graphs.values():
        _attach_graph_evidence(graph, edge_evidence, full_edge_index)

    nonempty_results = [result for result in analysis_interactions if result.get("interactions")]
    source_counts = Counter(result.get("source_gene") for result in nonempty_results)
    edge_counts = Counter()
    for result in nonempty_results:
        edge_counts[result.get("source_gene")] = len(result.get("interactions", []))

    graph_stats = {
        "full": _graph_counts(full_graph),
        "selected": _graph_counts(graphs["selected"]),
        "projected": _graph_counts(graphs["projected"]),
        "deg_llm": _graph_counts(graphs["deg_llm"]),
        "deg_prior": _graph_counts(graphs["deg_prior"]),
    }

    solver_top = knockout_hits[0] if knockout_hits else {}
    opus_top = llm_knockout.get("candidates", [{}])[0]

    benchmark_map = {row["gene_symbol"]: row for row in benchmark.get("results", [])}
    comparison_genes = []
    for gene in [*(solver_top.get("knocked_out_genes", [])), *(llm_knockout.get("final_recommendation", [])), "SOX2", "TP63"]:
        if gene and gene not in comparison_genes:
            comparison_genes.append(gene)

    comparison_rows = []
    for gene in comparison_genes:
        row = benchmark_map.get(gene, {})
        comparison_rows.append(
            {
                "gene": gene,
                "mean_gene_effect": row.get("mean_gene_effect"),
                "hit_rate": row.get("hit_rate"),
                "benchmark_hit": row.get("benchmark_hit"),
            }
        )

    top_deg_rows = top_degs[:15]
    top_edge_rows = [
        {
            "gene": gene,
            "edge_count": count,
        }
        for gene, count in edge_counts.most_common(10)
    ]

    graph_kinds = Counter(node["kind"] for node in graphs["projected"]["nodes"])
    graph_kind_rows = [{"kind": kind, "count": count} for kind, count in sorted(graph_kinds.items())]

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dir),
            "authors": ["Cathy Liu", "Toby Simonds"],
            "title": "From Differential Expression to Virtual Knockouts",
        },
        "summary": {
            "dataset_cells": summary.get("dataset_cells"),
            "dataset_genes": summary.get("dataset_genes"),
            "selected_experiment": summary.get("selected_experiment"),
            "solver_top_hit": solver_top.get("knocked_out_genes", []),
            "opus_top_hit": llm_knockout.get("final_recommendation", []),
            "graph_nodes": summary.get("graph_nodes"),
            "graph_edges": summary.get("graph_edges"),
            "research_models": research_execution.get("result_model_counts", {}),
            "fallback_gene_count": research_execution.get("fallback_gene_count", 0),
            "nonempty_gene_count": len(nonempty_results),
            "total_interaction_edges": sum(len(result.get("interactions", [])) for result in nonempty_results),
            "depmap_model_count": benchmark.get("model_count", 0),
        },
        "charts": {
            "top_degs": top_deg_rows,
            "top_edge_sources": top_edge_rows,
            "graph_kinds": graph_kind_rows,
            "benchmark_candidates": comparison_rows,
        },
        "graphs": graphs,
        "analysis_interactions": analysis_interactions,
        "benchmark_report": benchmark,
        "pre_simulation_benchmark": pre_benchmark,
        "experiments": experiments,
        "solver_knockouts": knockout_hits,
        "llm_knockout": llm_knockout,
        "figures": {
            "deg_without_llm": "data/run/deg_graph_without_llm.png",
            "deg_with_llm": "data/run/deg_graph_with_llm.png",
        },
    }


def _graph_counts(graph: dict[str, Any]) -> dict[str, int]:
    return {"nodes": len(graph["nodes"]), "edges": len(graph["edges"])}


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>__SITE_TITLE__</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600&family=Manrope:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet" />
    <link rel="stylesheet" href="https://unpkg.com/vis-network/styles/vis-network.min.css" />
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
    <link rel="stylesheet" href="styles.css" />
  </head>
  <body>
    <div class="page-shell">
      <aside class="toc">
        <p class="toc-label">Contents</p>
        <a href="#intro">Introduction</a>
        <a href="#method">Methodology</a>
        <a href="#degs">Differential Expression</a>
        <a href="#graph">Mechanistic Graph</a>
        <a href="#ranking">Ranking Knockouts</a>
        <a href="#results">Results</a>
        <a href="#implications">Potential Implications</a>
        <a href="#conclusion">Conclusion</a>
      </aside>

      <main class="article">
        <header class="hero">
          <p class="eyebrow">Technical report</p>
          <h1>From Differential Expression to Virtual Knockouts</h1>
          <p class="dek">
            We built a PDAC target-discovery pipeline that starts with malignant-versus-normal single-cell expression,
            expands those signals into a literature-backed regulatory graph, and then ranks interventions with both an
            explicit Boolean search and a direct model-based strategy.
          </p>
          <p class="authors">By Cathy Liu and Toby Simonds</p>
          <div id="hero-metrics" class="hero-metrics"></div>
        </header>

        <section id="intro" class="section prose">
          <p>
            In this post, we describe a working prototype for a question that appears often in translational biology:
            how do we move from thousands of cancer-associated transcriptional features to a short list of intervention points
            that are specific enough to test, mechanistic enough to explain, and grounded enough to benchmark?
          </p>
          <p>
            Our pipeline focuses on pancreatic ductal adenocarcinoma. It begins with malignant-versus-normal differential expression,
            asks a language model to recover mechanistic signaling edges for the resulting genes, projects those edges into a compact
            intervention graph, and then ranks knockouts in two different ways. The first is explicit search over the graph. The second
            is a model-based reading of the graph itself: Claude Opus is asked to inspect the final network and recommend the most plausible
            knockouts directly.
          </p>
          <p>
            The result is not a validated therapeutic claim. It is a more useful outcome than that: a transparent record of what the system
            believes, where those beliefs come from, how the recommendations differ by ranking strategy, and where external benchmark data
            refuses to agree.
          </p>
        </section>

        <section class="section cards" id="method">
          <div class="section-header">
            <p class="eyebrow">Methodology</p>
            <h2>A four-stage pipeline</h2>
          </div>
          <div class="card-grid">
            <article class="method-card">
              <h3>1. Distill the expression space</h3>
              <p>We load a PDAC single-cell dataset, perform QC and normalization, and rank genes by malignant-versus-normal differential expression.</p>
            </article>
            <article class="method-card">
              <h3>2. Recover mechanistic edges</h3>
              <p>Each DEG is researched independently with Claude Sonnet 4.6. The output is constrained to signed, gene-level mechanistic edges with cited support.</p>
            </article>
            <article class="method-card">
              <h3>3. Project to an intervention graph</h3>
              <p>Intermediate biology is retained for reasoning, then collapsed into a DEG-centered control graph that is easier to simulate and interpret.</p>
            </article>
            <article class="method-card">
              <h3>4. Rank interventions two ways</h3>
              <p>We compare a brute-force Boolean search against a direct graph-reading LLM ranker using Claude Opus 4.6.</p>
            </article>
          </div>
          <div class="method-detail">
            <div class="detail-card">
              <p class="detail-label">What we added beyond the original concept note</p>
              <ul>
                <li>Parallel per-gene literature calls, with raw request and response artifacts saved for inspection.</li>
                <li>A projected 50-node DEG graph, so intermediates can support reasoning without overwhelming the intervention layer.</li>
                <li>Benchmark-aware pruning and post hoc DepMap checks, which prevent graph-only hits from being mistaken for validated targets.</li>
                <li>A second ranking head in which Claude Opus reads the graph directly and proposes knockouts as a mechanistic reviewer.</li>
              </ul>
            </div>
            <div class="detail-card">
              <p class="detail-label">Why the projection step matters</p>
              <p>
                In biological terms, intermediate nodes are often necessary to explain how a DEG reaches KRAS or MAPK signaling, but they are not always the nodes we
                want to optimize over. The projected graph separates those concerns. Claude can recover paths like <span class="mono">A → B → C</span>, while the final
                intervention surface keeps the emphasis on the tumor-altered genes themselves.
              </p>
            </div>
          </div>
        </section>

        <section id="degs" class="section">
          <div class="section-header">
            <p class="eyebrow">Stage 1</p>
            <h2>Distilling scRNA-seq into a targetable gene set</h2>
          </div>
          <div class="prose">
            <p>
              The pipeline begins with a simple but important reduction. Rather than attempting to reason over the full transcriptome,
              we first ask which genes are most strongly elevated in malignant epithelial cells relative to normal epithelial controls.
              This gives the rest of the system a cancer-weighted starting point rather than a generic pathway list.
            </p>
            <p>
              In the final run reported here, the PDAC dataset resolved to <span data-bind="dataset_cells"></span> filtered cells and
              <span data-bind="dataset_genes"></span> genes. We retained the top 50 DEGs for downstream graph construction. The chart
              below shows the leading fold changes; these genes anchor the graph and define the intervention universe.
            </p>
          </div>
          <div class="figure-card">
            <canvas id="deg-chart"></canvas>
            <p class="figure-caption">Top malignant-versus-normal DEGs ranked by log2 fold change.</p>
          </div>
        </section>

        <section id="graph" class="section">
          <div class="section-header">
            <p class="eyebrow">Stages 2 and 3</p>
            <h2>From literature-backed mechanisms to a compact graph</h2>
          </div>
          <div class="prose">
            <p>
              Literature extraction is most useful when it is allowed to be mechanistic without becoming unbounded. We therefore let Claude
              discover intermediate genes when they are necessary to explain a path, but we do not optimize directly over every intermediate node.
              Instead, we build a richer internal graph and then project it into a DEG-centered control surface that preserves the most relevant influence paths.
            </p>
            <p>
              This lets us separate explanation from optimization. The model can say <span class="mono">A → B → C</span> when that is what the
              literature supports, while the simulator can still operate on a smaller representation that is aligned with tumor-associated genes.
            </p>
          </div>
          <div class="gallery">
            <figure class="gallery-card">
              <img src="data/run/deg_graph_without_llm.png" alt="Projected DEG graph without LLM suggestions" />
              <figcaption>Projected 50-node DEG graph without LLM suggestions.</figcaption>
            </figure>
            <figure class="gallery-card">
              <img src="data/run/deg_graph_with_llm.png" alt="Projected DEG graph with LLM suggestions" />
              <figcaption>Projected 50-node DEG graph with LLM suggestions.</figcaption>
            </figure>
          </div>
          <div class="figure-grid">
            <div class="figure-card">
              <canvas id="edge-source-chart"></canvas>
              <p class="figure-caption">Which DEGs produced the most mechanistic edges in the final run.</p>
            </div>
            <div class="figure-card">
              <canvas id="graph-kind-chart"></canvas>
              <p class="figure-caption">Node composition of the projected graph used for simulation.</p>
            </div>
          </div>
          <div class="interactive-block">
            <div class="interactive-header">
              <div>
                <p class="eyebrow">Interactive view</p>
                <h3>Explore the graph and its evidence</h3>
              </div>
              <div class="controls">
                <label>
                  <span>Graph view</span>
                  <select id="graph-select"></select>
                </label>
                <label>
                  <span>Find node</span>
                  <input id="node-search" type="search" placeholder="EFS, SOX2, KRAS..." />
                </label>
              </div>
            </div>
            <div class="interactive-layout">
              <div id="network" class="network"></div>
              <aside id="inspector" class="inspector"></aside>
            </div>
          </div>
        </section>

        <section id="ranking" class="section">
          <div class="section-header">
            <p class="eyebrow">Stage 4</p>
            <h2>Two ways of choosing knockouts</h2>
          </div>
          <div class="prose">
            <p>
              We use two ranking methods because they answer slightly different questions. The solver is literal: it asks which one-, two-, or three-gene
              knockouts flip the final KRAS-signaling readout to off in the graph. The direct Opus ranker is more interpretive: it reads the final graph,
              the evidence, and the benchmark rows, then produces a ranked recommendation as if it were a mechanistic reviewer.
            </p>
            <p>
              The benefit of placing them side by side is that agreement is informative. If both methods converge on the same top intervention, the recommendation
              becomes easier to scrutinize. If they diverge, that divergence is often biologically useful in its own right.
            </p>
          </div>
          <div class="comparison-grid" id="ranking-comparison"></div>
        </section>

        <section id="results" class="section">
          <div class="section-header">
            <p class="eyebrow">Results</p>
            <h2>What the system found, and what the benchmark refused to confirm</h2>
          </div>
          <div class="prose">
            <p>
              In the final run, both ranking methods converged on <span class="mono">EFS</span> as the leading single-gene knockout. Within the graph,
              that recommendation is easy to understand: EFS connects into SRC-centered signaling and sits on the shortest path to several tracked KRAS pathway nodes.
              The solver reports a full pathway shutdown when EFS is removed, and Opus reaches the same top call after reading the graph directly.
            </p>
            <p>
              The benchmark result is the counterweight. DepMap does not support EFS strongly as a PDAC dependency. That means the system is producing a coherent
              mechanistic hypothesis, but not yet a confident translational target. This is precisely the kind of outcome that a benchmark layer should surface.
            </p>
          </div>
          <div class="figure-card">
            <canvas id="benchmark-chart"></canvas>
            <p class="figure-caption">Benchmark rows for the genes most relevant to the final solver and Opus recommendations.</p>
          </div>
          <div class="results-grid">
            <div class="result-card" id="solver-card"></div>
            <div class="result-card" id="opus-card"></div>
          </div>
        </section>

        <section id="implications" class="section prose">
          <div class="section-header">
            <p class="eyebrow">Potential implications</p>
            <h2>Why this approach is still promising</h2>
          </div>
          <p>
            The immediate result is not that EFS should now be treated as a validated therapeutic target. The more important result is that the system can connect
            differential expression, mechanistic evidence extraction, graph projection, explicit simulation, and model-based ranking in a single reproducible workflow.
          </p>
          <p>
            That matters because the pipeline is now modular. Each weak point is inspectable. The DEG layer can be improved with better state labels; the graph layer
            can be improved with stronger extraction prompts or curated priors; the benchmark layer can be expanded with normal-tissue toxicity or additional perturbation data.
            The method is therefore valuable not because it solved PDAC target discovery outright, but because it produces a tractable loop for improving it.
          </p>
          <p>
            More generally, the combination of explicit search and direct graph-reading models is useful. The solver gives a disciplined, auditable baseline. The direct
            LLM ranker offers a different kind of synthesis: one that can reason across topology, evidence quality, and benchmark weakness in a single pass. Used together,
            they make it easier to distinguish graph artifacts from biologically persuasive candidates.
          </p>
        </section>

        <section id="conclusion" class="section prose">
          <div class="section-header">
            <p class="eyebrow">Conclusion</p>
            <h2>A prototype that is honest about its limits</h2>
          </div>
          <p>
            We now have an end-to-end system that starts with single-cell data and ends with ranked knockout hypotheses, accompanied by mechanistic evidence and external benchmark checks.
            In the present PDAC run, the top answer is consistent across the solver and the direct Opus ranker, but external validation remains weak. That is not a failure of the benchmark;
            it is the benchmark doing its job.
          </p>
          <p>
            The value of the prototype is therefore not that it eliminates uncertainty. It is that it renders uncertainty legible. Every stage is inspectable, every major decision is attributable,
            and every recommendation can be interrogated in the graph itself. That makes the next iteration more scientific than the first.
          </p>
        </section>
      </main>
    </div>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script src="app.js"></script>
  </body>
</html>
"""


STYLESHEET = """
:root {
  --page: #f7f3eb;
  --panel: rgba(255, 252, 246, 0.92);
  --ink: #171717;
  --muted: #625d56;
  --line: rgba(23, 23, 23, 0.12);
  --accent: #c15b2c;
  --accent-2: #1f6f72;
  --serif: "Newsreader", serif;
  --sans: "Manrope", sans-serif;
  --mono: "IBM Plex Mono", monospace;
}

* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background:
    radial-gradient(circle at top left, rgba(193, 91, 44, 0.12), transparent 22%),
    radial-gradient(circle at top right, rgba(31, 111, 114, 0.10), transparent 20%),
    var(--page);
  color: var(--ink);
  font-family: var(--sans);
}

.page-shell {
  display: grid;
  grid-template-columns: 220px minmax(0, 1fr);
  max-width: 1500px;
  margin: 0 auto;
}

.toc {
  position: sticky;
  top: 0;
  height: 100vh;
  padding: 28px 18px 28px 24px;
  border-right: 1px solid var(--line);
  display: flex;
  flex-direction: column;
  gap: 10px;
  background: rgba(247, 243, 235, 0.85);
  backdrop-filter: blur(10px);
}

.toc-label,
.eyebrow {
  margin: 0;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--muted);
}

.toc a {
  color: var(--muted);
  text-decoration: none;
  font-size: 14px;
}

.toc a:hover { color: var(--ink); }

.article {
  padding: 0 48px 80px;
}

.hero {
  padding: 64px 0 28px;
  border-bottom: 1px solid var(--line);
}

.hero h1 {
  margin: 8px 0 18px;
  font-family: var(--serif);
  font-size: clamp(54px, 7vw, 86px);
  line-height: 0.95;
  letter-spacing: -0.03em;
  max-width: 12ch;
}

.dek {
  max-width: 56rem;
  font-size: 20px;
  line-height: 1.6;
  color: #2a2724;
}

.authors {
  margin: 16px 0 0;
  color: var(--muted);
}

.hero-metrics {
  margin-top: 28px;
  display: grid;
  gap: 14px;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
}

.metric-pill,
.method-card,
.figure-card,
.gallery-card,
.interactive-block,
.result-card {
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 26px;
  box-shadow: 0 18px 60px rgba(28, 22, 14, 0.07);
}

.metric-pill {
  padding: 16px 18px;
}

.metric-pill strong {
  display: block;
  margin-bottom: 6px;
  font-size: 13px;
  color: var(--muted);
}

.metric-pill span {
  font-size: 24px;
  font-weight: 700;
}

.section {
  padding: 42px 0 8px;
}

.section-header {
  margin-bottom: 18px;
}

.section-header h2 {
  margin: 8px 0 0;
  font-family: var(--serif);
  font-size: clamp(34px, 4.5vw, 52px);
  line-height: 1.02;
  letter-spacing: -0.02em;
  max-width: 14ch;
}

.prose {
  max-width: 760px;
}

.prose p {
  font-size: 19px;
  line-height: 1.8;
  color: #2a2724;
}

.cards .card-grid,
.figure-grid,
.results-grid,
.comparison-grid,
.interactive-layout,
.method-detail {
  display: grid;
  gap: 18px;
}

.card-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.method-card,
.figure-card,
.result-card,
.detail-card {
  padding: 22px;
}

.method-card h3,
.interactive-header h3,
.result-card h3 {
  margin: 0 0 10px;
  font-family: var(--serif);
  font-size: 28px;
}

.method-card p,
.result-card p,
.detail-card p {
  margin: 0;
  color: #2d2924;
  line-height: 1.7;
}

.method-detail {
  grid-template-columns: 1.1fr 0.9fr;
  margin-top: 18px;
}

.detail-card {
  border: 1px solid var(--line);
  background: rgba(255, 249, 239, 0.9);
  border-radius: 26px;
  box-shadow: 0 18px 60px rgba(28, 22, 14, 0.05);
}

.detail-label {
  margin-bottom: 12px !important;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--muted);
}

.detail-card ul {
  margin: 0;
  padding-left: 18px;
  color: #2d2924;
  line-height: 1.8;
}

.figure-card canvas {
  width: 100% !important;
  height: 340px !important;
}

.figure-caption,
.gallery-card figcaption {
  margin: 14px 0 0;
  font-size: 14px;
  color: var(--muted);
}

.gallery {
  display: grid;
  gap: 18px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin: 22px 0;
}

.gallery-card {
  margin: 0;
  padding: 18px;
}

.gallery-card img {
  width: 100%;
  display: block;
  border-radius: 18px;
  border: 1px solid rgba(23, 23, 23, 0.08);
}

.interactive-block {
  padding: 22px;
  margin-top: 18px;
}

.interactive-header {
  display: flex;
  justify-content: space-between;
  align-items: end;
  gap: 20px;
  margin-bottom: 16px;
}

.controls {
  display: flex;
  gap: 12px;
}

.controls label {
  display: grid;
  gap: 6px;
  font-size: 12px;
  color: var(--muted);
}

.controls select,
.controls input {
  min-width: 220px;
  border-radius: 14px;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.9);
  padding: 12px 14px;
  font: inherit;
}

.interactive-layout {
  grid-template-columns: minmax(0, 1.6fr) minmax(320px, 0.9fr);
}

.network {
  height: 680px;
  border-radius: 20px;
  border: 1px solid rgba(23, 23, 23, 0.08);
  background: linear-gradient(180deg, rgba(255,255,255,0.94), rgba(246,240,228,0.96));
}

.inspector {
  min-height: 680px;
  border-radius: 20px;
  border: 1px solid rgba(23, 23, 23, 0.08);
  background: rgba(255, 255, 255, 0.72);
  padding: 18px;
  overflow: auto;
}

.comparison-grid,
.results-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.comparison-card,
.mini-card,
.edge-card,
.evidence-card {
  border: 1px solid rgba(23, 23, 23, 0.08);
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.85);
  padding: 16px;
}

.comparison-card h3,
.mini-card strong {
  margin: 0 0 8px;
}

.mini-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
}

.mono { font-family: var(--mono); }
.muted { color: var(--muted); }

.tag-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}

.tag {
  display: inline-flex;
  align-items: center;
  padding: 5px 10px;
  border-radius: 999px;
  background: rgba(23, 23, 23, 0.06);
  font-size: 12px;
}

.edge-list,
.evidence-list {
  display: grid;
  gap: 10px;
}

a { color: var(--accent-2); }

@media (max-width: 1180px) {
  .page-shell { grid-template-columns: 1fr; }
  .toc { display: none; }
  .article { padding: 0 22px 60px; }
  .card-grid,
  .gallery,
  .figure-grid,
  .comparison-grid,
  .results-grid,
  .interactive-layout {
    grid-template-columns: 1fr;
  }
  .controls {
    flex-direction: column;
    width: 100%;
  }
  .controls select,
  .controls input {
    min-width: 0;
    width: 100%;
  }
}
"""


APP_SCRIPT = """
const bundleUrl = "data/post_bundle.json";
const GRAPH_LABELS = {
  selected: "Selected final graph",
  projected: "Projected simulation graph",
  deg_llm: "50-node DEG graph with LLM suggestions",
  deg_prior: "50-node DEG graph without LLM suggestions",
};

let bundle = null;
let currentGraphKey = "selected";
let network = null;
let nodeDataset = null;

const kindColors = {
  deg: "#c15b2c",
  pathway: "#1f6f72",
  prior: "#8a6f2a",
  intermediate: "#8a6f2a",
  boss: "#111111",
  unknown: "#6a655d",
};

async function init() {
  bundle = await fetch(bundleUrl).then((response) => response.json());
  renderHeroMetrics();
  renderCharts();
  renderRankingComparison();
  buildGraphSelect();
  renderGraph();
  document.getElementById("graph-select").addEventListener("change", (event) => {
    currentGraphKey = event.target.value;
    renderGraph();
  });
  document.getElementById("node-search").addEventListener("change", focusNodeFromSearch);
}

function renderHeroMetrics() {
  const summary = bundle.summary;
  const metrics = [
    ["Cells", number(summary.dataset_cells)],
    ["Top DEGs", "50"],
    ["Non-empty LLM genes", number(summary.nonempty_gene_count)],
    ["LLM edges", number(summary.total_interaction_edges)],
    ["Solver top hit", (summary.solver_top_hit || []).join(" + ") || "none"],
    ["Opus top hit", (summary.opus_top_hit || []).join(" + ") || "none"],
  ];
  document.getElementById("hero-metrics").innerHTML = metrics.map(([label, value]) => `
    <div class="metric-pill"><strong>${escapeHtml(label)}</strong><span>${escapeHtml(value)}</span></div>
  `).join("");
  bindText("dataset_cells", number(summary.dataset_cells));
  bindText("dataset_genes", number(summary.dataset_genes));
}

function renderCharts() {
  renderBarChart("deg-chart", bundle.charts.top_degs.map((row) => row.gene), bundle.charts.top_degs.map((row) => row.log2_fold_change), "#c15b2c", "Log2 fold change");
  renderBarChart("edge-source-chart", bundle.charts.top_edge_sources.map((row) => row.gene), bundle.charts.top_edge_sources.map((row) => row.edge_count), "#1f6f72", "Discovered edges");
  renderBarChart("graph-kind-chart", bundle.charts.graph_kinds.map((row) => row.kind), bundle.charts.graph_kinds.map((row) => row.count), "#8a6f2a", "Node count");
  renderBarChart(
    "benchmark-chart",
    bundle.charts.benchmark_candidates.map((row) => row.gene),
    bundle.charts.benchmark_candidates.map((row) => row.mean_gene_effect ?? 0),
    "#5b7c2c",
    "Mean gene effect"
  );
}

function renderBarChart(canvasId, labels, values, color, label) {
  const canvas = document.getElementById(canvasId);
  new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [{ label, data: values, backgroundColor: color, borderRadius: 8 }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#554f49" }, grid: { display: false } },
        y: { ticks: { color: "#554f49" }, grid: { color: "rgba(23,23,23,0.08)" } },
      },
    },
  });
}

function renderRankingComparison() {
  const solver = bundle.solver_knockouts[0] || {};
  const opus = bundle.llm_knockout.candidates[0] || {};
  const cards = [
    {
      title: "Boolean search",
      body: "The explicit solver exhaustively searches one-, two-, and three-gene knockouts over the selected graph and optimizes for shutting down the synthetic KRAS signaling endpoint.",
      recommendation: (solver.knocked_out_genes || []).join(" + ") || "none",
      confidence: fmt(solver.score),
      note: `Pathway nodes off: ${(solver.pathway_nodes_off || []).join(", ") || "none"}`,
    },
    {
      title: "Claude Opus graph reading",
      body: "The direct model-based ranker reads the final graph, the evidence-backed edges, and the benchmark rows, then recommends knockouts in natural language constrained to structured JSON.",
      recommendation: (opus.knocked_out_genes || []).join(" + ") || "none",
      confidence: fmt(opus.confidence_score),
      note: opus.benchmark_assessment || "No benchmark note.",
    },
  ];
  document.getElementById("ranking-comparison").innerHTML = cards.map((card) => `
    <article class="comparison-card">
      <h3>${escapeHtml(card.title)}</h3>
      <p>${escapeHtml(card.body)}</p>
      <div class="mini-grid">
        <div class="mini-card"><strong>Recommendation</strong><div class="mono">${escapeHtml(card.recommendation)}</div></div>
        <div class="mini-card"><strong>Score / confidence</strong><div>${escapeHtml(card.confidence)}</div></div>
      </div>
      <p class="muted">${escapeHtml(card.note)}</p>
    </article>
  `).join("");

  document.getElementById("solver-card").innerHTML = renderResultCard("Solver result", bundle.solver_knockouts[0], true);
  document.getElementById("opus-card").innerHTML = renderResultCard("Claude Opus recommendation", bundle.llm_knockout.candidates[0], false);
}

function renderResultCard(title, record, solver) {
  if (!record) {
    return `<h3>${escapeHtml(title)}</h3><p>No result available.</p>`;
  }
  const genes = solver ? (record.knocked_out_genes || []) : (record.knocked_out_genes || []);
  const note = solver
    ? `Pathway nodes off: ${(record.pathway_nodes_off || []).join(", ") || "none"}`
    : (record.benchmark_assessment || "");
  return `
    <h3>${escapeHtml(title)}</h3>
    <p class="mono">${escapeHtml(genes.join(" + "))}</p>
    <p>${escapeHtml(solver ? "The solver chooses the smallest knockout that collapses the boss node." : record.rationale || "")}</p>
    <p class="muted">${escapeHtml(note)}</p>
  `;
}

function buildGraphSelect() {
  const select = document.getElementById("graph-select");
  select.innerHTML = "";
  Object.entries(GRAPH_LABELS).forEach(([key, label]) => {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = label;
    select.appendChild(option);
  });
}

function renderGraph() {
  const graph = bundle.graphs[currentGraphKey];
  const positionedNodes = buildCircularNodeLayout(graph.nodes);
  const nodes = new vis.DataSet(positionedNodes.map((node) => ({
    id: node.id,
    label: node.id,
    color: kindColors[node.kind] || kindColors.unknown,
    shape: node.kind === "boss" ? "hexagon" : "dot",
    size: node.kind === "boss" ? 22 : node.kind === "pathway" ? 18 : 14,
    font: { face: "Manrope", size: 14, color: "#151515" },
    x: node.x,
    y: node.y,
  })));
  const edges = new vis.DataSet(graph.edges.map((edge, index) => ({
    id: `${edge.source}|${edge.target}|${edge.sign}|${index}`,
    from: edge.source,
    to: edge.target,
    arrows: "to",
    color: edge.sign === -1 ? "#c0392b" : "#198754",
    width: Math.max(1.5, (edge.confidence || edge.weight || 0.35) * 4),
    dashes: !edge.direct_evidence?.length && !edge.step_evidence?.length ? [6, 5] : false,
  })));
  nodeDataset = nodes;
  if (network) {
    network.destroy();
  }
  network = new vis.Network(
    document.getElementById("network"),
    { nodes, edges },
    {
      interaction: { hover: true, navigationButtons: true, keyboard: true },
      physics: { enabled: false },
      layout: { improvedLayout: false },
    }
  );
  network.on("click", (params) => handleGraphClick(params, graph));
  document.getElementById("inspector").innerHTML = `
    <div class="mini-card">
      <strong>${escapeHtml(GRAPH_LABELS[currentGraphKey])}</strong>
      <div class="muted">Click a node to inspect incoming and outgoing influences, or click an edge to see direct and collapsed evidence.</div>
    </div>`;
}

function handleGraphClick(params, graph) {
  if (params.nodes.length) {
    const nodeId = params.nodes[0];
    renderNodeInspector(nodeId, graph);
    return;
  }
  if (params.edges.length) {
    const edge = graph.edges.find((candidate, index) => `${candidate.source}|${candidate.target}|${candidate.sign}|${index}` === params.edges[0]);
    if (edge) renderEdgeInspector(edge);
  }
}

function renderNodeInspector(nodeId, graph) {
  const incoming = graph.edges.filter((edge) => edge.target === nodeId);
  const outgoing = graph.edges.filter((edge) => edge.source === nodeId);
  const interactionRows = bundle.analysis_interactions.find((row) => row.source_gene === nodeId);
  document.getElementById("inspector").innerHTML = `
    <div class="mini-card">
      <strong>${escapeHtml(nodeId)}</strong>
      <div class="muted">${incoming.length} incoming edges · ${outgoing.length} outgoing edges</div>
    </div>
    <div class="edge-list">
      ${outgoing.map(renderEdgeCard).join("") || '<div class="mini-card muted">No outgoing edges in this view.</div>'}
    </div>
    <div class="edge-list">
      ${incoming.map(renderEdgeCard).join("") || '<div class="mini-card muted">No incoming edges in this view.</div>'}
    </div>
    ${interactionRows ? `<div class="mini-card"><strong>Source gene evidence count</strong><div>${interactionRows.interactions.length}</div></div>` : ""}
  `;
}

function renderEdgeInspector(edge) {
  document.getElementById("inspector").innerHTML = renderEdgeCard(edge);
}

function renderEdgeCard(edge) {
  const directEvidence = edge.direct_evidence || [];
  const stepEvidence = edge.step_evidence || [];
  return `
    <div class="edge-card">
      <strong>${escapeHtml(edge.source)} ${edge.sign === -1 ? "−|" : "→"} ${escapeHtml(edge.target)}</strong>
      <div class="muted">confidence ${fmt(edge.confidence || edge.weight)} · provenance ${(edge.provenance || []).join(", ") || "none"}</div>
      ${edge.collapsed_via?.length ? `<div class="tag-row"><span class="tag">collapsed via ${escapeHtml(edge.collapsed_via.join(" → "))}</span></div>` : ""}
      <div class="evidence-list">
        ${directEvidence.length ? directEvidence.map(renderEvidenceCard).join("") : '<div class="mini-card muted">No direct evidence attached to the displayed edge.</div>'}
      </div>
      ${stepEvidence.length ? `
        <div class="evidence-list">
          ${stepEvidence.map((step) => `
            <div class="evidence-card">
              <strong>${escapeHtml(step.source)} → ${escapeHtml(step.target)}</strong>
              <div class="muted">confidence ${fmt(step.confidence)} · provenance ${(step.provenance || []).join(", ") || "none"}</div>
              <div class="evidence-list">
                ${(step.direct_evidence || []).length ? step.direct_evidence.map(renderEvidenceCard).join("") : '<div class="mini-card muted">No direct model citation stored for this step.</div>'}
              </div>
            </div>`).join("")}
        </div>` : ""}
    </div>
  `;
}

function renderEvidenceCard(item) {
  return `
    <div class="evidence-card">
      <strong>${escapeHtml(item.source_gene)} ${item.interaction_type === -1 ? "−|" : "→"} ${escapeHtml(item.target)}</strong>
      <div>${escapeHtml(item.evidence_summary || "")}</div>
      <div class="tag-row">
        <span class="tag">conf ${fmt(item.confidence_score)}</span>
        ${(item.provenance_sources || []).map((source) => `<span class="tag">${escapeHtml(source)}</span>`).join("")}
      </div>
      <div class="muted mono">${renderRefs(item.source_refs || [], item.pmid_citations || [])}</div>
    </div>
  `;
}

function buildCircularNodeLayout(nodes) {
  const ordered = [...nodes].sort((a, b) => a.id.localeCompare(b.id));
  const radius = Math.max(260, ordered.length * 8);
  return ordered.map((node, index) => ({
    ...node,
    x: Math.cos((Math.PI * 2 * index) / Math.max(ordered.length, 1)) * radius,
    y: Math.sin((Math.PI * 2 * index) / Math.max(ordered.length, 1)) * radius,
  }));
}

function focusNodeFromSearch() {
  const query = document.getElementById("node-search").value.trim().toUpperCase();
  if (!query || !network || !nodeDataset.get(query)) return;
  network.selectNodes([query]);
  network.focus(query, { scale: 1.2, animation: true });
  renderNodeInspector(query, bundle.graphs[currentGraphKey]);
}

function renderRefs(sourceRefs, pmids) {
  const refs = [];
  sourceRefs.forEach((ref) => {
    refs.push(ref.startsWith("http") ? `<a href="${ref}" target="_blank" rel="noreferrer">${escapeHtml(ref)}</a>` : escapeHtml(ref));
  });
  pmids.forEach((pmid) => refs.push(`<a href="https://pubmed.ncbi.nlm.nih.gov/${pmid}/" target="_blank" rel="noreferrer">PMID:${escapeHtml(pmid)}</a>`));
  return refs.join(" · ");
}

function number(value) {
  return value === null || value === undefined ? "n/a" : Number(value).toLocaleString();
}

function fmt(value) {
  if (value === null || value === undefined || value === "") return "n/a";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  return String(value);
}

function bindText(key, value) {
  document.querySelectorAll(`[data-bind="${key}"]`).forEach((node) => {
    node.textContent = value;
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

init().catch((error) => {
  console.error(error);
});
"""
