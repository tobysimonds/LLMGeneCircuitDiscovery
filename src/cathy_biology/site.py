from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from cathy_biology.utils import ensure_directory, write_json

REQUIRED_RUN_FILES = {
    "summary": "summary.json",
    "top_degs": "top_degs.json",
    "gene_interactions": "gene_interactions.json",
    "regulatory_graph": "regulatory_graph.json",
    "knockout_hits": "knockout_hits.json",
    "benchmark_report": "benchmark_report.json",
    "experiment_report": "experiment_report.json",
    "research_execution": "research_execution.json",
    "prior_knowledge": "prior_knowledge.json",
}


def build_results_site(
    primary_run_dir: Path,
    output_dir: Path,
    *,
    baseline_run_dir: Path | None = None,
    title: str = "PDAC Target Discovery Atlas",
) -> Path:
    output_dir = ensure_directory(output_dir)
    data_dir = ensure_directory(output_dir / "data")

    primary_payload = _copy_run_bundle(primary_run_dir, data_dir, slug="primary", label="Primary run")
    manifest: dict[str, Any] = {
        "title": title,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": {"primary": primary_payload},
    }
    if baseline_run_dir is not None:
        manifest["runs"]["baseline"] = _copy_run_bundle(baseline_run_dir, data_dir, slug="baseline", label="Baseline run")

    write_json(data_dir / "manifest.json", manifest)
    (output_dir / "index.html").write_text(HTML_TEMPLATE.replace("__SITE_TITLE__", escape(title)), encoding="utf-8")
    (output_dir / "styles.css").write_text(STYLESHEET, encoding="utf-8")
    (output_dir / "app.js").write_text(APP_SCRIPT, encoding="utf-8")
    return output_dir


def _copy_run_bundle(run_dir: Path, data_root: Path, *, slug: str, label: str) -> dict[str, Any]:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    target_dir = ensure_directory(data_root / slug)
    summary_source = run_dir / REQUIRED_RUN_FILES["summary"]
    if not summary_source.exists():
        raise FileNotFoundError(f"Run directory {run_dir} is missing required artifact summary.json.")
    summary = json.loads(summary_source.read_text(encoding="utf-8"))
    legacy_fallbacks = _legacy_run_fallbacks(summary, label)

    copied_files: dict[str, str] = {}
    for logical_name, filename in REQUIRED_RUN_FILES.items():
        source = run_dir / filename
        if source.exists():
            shutil.copy2(source, target_dir / filename)
        elif logical_name in legacy_fallbacks:
            write_json(target_dir / filename, legacy_fallbacks[logical_name])
        else:
            raise FileNotFoundError(f"Run directory {run_dir} is missing required artifact {filename}.")
        copied_files[logical_name] = f"data/{slug}/{filename}"

    benchmark = json.loads((run_dir / "benchmark_report.json").read_text(encoding="utf-8"))
    knockout_hits = json.loads((run_dir / "knockout_hits.json").read_text(encoding="utf-8"))
    return {
        "slug": slug,
        "label": label,
        "source_dir": str(run_dir),
        "selected_experiment": summary.get("selected_experiment"),
        "graph_nodes": summary.get("graph_nodes"),
        "graph_edges": summary.get("graph_edges"),
        "dataset_cells": summary.get("dataset_cells"),
        "dataset_genes": summary.get("dataset_genes"),
        "deg_count": len(summary.get("degs", [])),
        "top_hit": knockout_hits[0]["knocked_out_genes"] if knockout_hits else [],
        "benchmark_model_count": benchmark.get("model_count", 0),
        "benchmark_hit_count": sum(1 for result in benchmark.get("results", []) if result.get("benchmark_hit")),
        "files": copied_files,
    }


def _legacy_run_fallbacks(summary: dict[str, Any], label: str) -> dict[str, Any]:
    benchmark_report = summary.get(
        "benchmark_report",
        {
            "release": "legacy",
            "lineage_filter": [],
            "primary_disease_filter": [],
            "model_count": 0,
            "stage": "final",
            "results": [],
        },
    )
    knockout_hits = summary.get("knockout_hits", [])
    experiment_name = summary.get("selected_experiment") or label.lower().replace(" ", "-")
    return {
        "top_degs": summary.get("degs", []),
        "knockout_hits": knockout_hits,
        "benchmark_report": benchmark_report,
        "research_execution": summary.get(
            "research_execution",
            {
                "requested_backend": "legacy-artifact",
                "configured_model": "legacy-artifact",
                "parser_model": "legacy-artifact",
                "total_genes": len(summary.get("degs", [])),
                "result_model_counts": {},
                "fallback_gene_count": 0,
            },
        ),
        "prior_knowledge": summary.get(
            "prior_knowledge",
            {
                "node_count": 0,
                "edge_count": 0,
                "source_counts": {},
                "nodes": [],
                "edges": [],
            },
        ),
        "experiment_report": summary.get(
            "experiment_results",
            [
                {
                    "name": experiment_name,
                    "description": "Legacy run synthesized from summary artifacts.",
                    "graph_nodes": summary.get("graph_nodes", 0),
                    "graph_edges": summary.get("graph_edges", 0),
                    "knockout_hits": knockout_hits,
                    "benchmark_report": benchmark_report,
                    "pruned_genes": [],
                    "score": knockout_hits[0].get("score", -1_000.0) if knockout_hits else -1_000.0,
                    "selected": True,
                }
            ],
        ),
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>__SITE_TITLE__</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet" />
    <link rel="stylesheet" href="https://unpkg.com/vis-network/styles/vis-network.min.css" />
    <link rel="stylesheet" href="styles.css" />
  </head>
  <body>
    <div class="background-orb orb-a"></div>
    <div class="background-orb orb-b"></div>
    <header class="hero reveal">
      <div class="hero-copy">
        <p class="eyebrow">Interactive run atlas</p>
        <h1 id="hero-title">__SITE_TITLE__</h1>
        <p id="hero-summary" class="hero-summary">Loading run metadata...</p>
        <div id="hero-pills" class="hero-pills"></div>
      </div>
      <aside class="hero-panel">
        <p class="panel-label">Selected run</p>
        <h2 id="hero-run-name">Loading...</h2>
        <dl class="hero-stats">
          <div>
            <dt>Graph</dt>
            <dd id="hero-graph">-</dd>
          </div>
          <div>
            <dt>Knockouts</dt>
            <dd id="hero-hits">-</dd>
          </div>
          <div>
            <dt>Benchmark</dt>
            <dd id="hero-benchmark">-</dd>
          </div>
        </dl>
      </aside>
    </header>

    <main class="page-shell">
      <section class="metric-grid reveal" id="overview-metrics"></section>

      <section class="section reveal">
        <div class="section-heading">
          <div>
            <p class="eyebrow">Run comparison</p>
            <h2>Baseline versus upgraded system</h2>
          </div>
          <p class="section-copy">Compare graph density, knockout outcomes, and benchmark support between the latest run and the baseline reference.</p>
        </div>
        <div id="comparison-grid" class="comparison-grid"></div>
      </section>

      <section class="section reveal">
        <div class="section-heading">
          <div>
            <p class="eyebrow">Experiment ablations</p>
            <h2>Which architecture variants actually helped?</h2>
          </div>
          <p class="section-copy">Each card reflects the exact experiment report emitted by the pipeline, including pruning effects and benchmark outcomes.</p>
        </div>
        <div id="experiment-grid" class="experiment-grid"></div>
      </section>

      <section class="section reveal">
        <div class="section-heading">
          <div>
            <p class="eyebrow">Network explorer</p>
            <h2>Mechanistic graph</h2>
          </div>
          <p class="section-copy">Explore the selected run's directed network. Search for genes, filter node classes, and inspect edge provenance and sign.</p>
        </div>
        <div class="network-controls">
          <label class="search-shell">
            <span>Jump to gene</span>
            <input id="network-search" type="search" placeholder="EGFR, KRAS, SOX2..." />
          </label>
          <button id="network-focus" class="ghost-button" type="button">Focus</button>
          <button id="network-physics" class="ghost-button" type="button">Pause motion</button>
        </div>
        <div id="network-kind-filters" class="chip-row"></div>
        <div class="network-layout">
          <div id="network-canvas" class="network-canvas"></div>
          <aside class="network-sidepanel">
            <h3>Legend</h3>
            <ul id="network-legend" class="legend-list"></ul>
            <div class="network-callout">
              <p class="panel-label">Selected knockout</p>
              <div id="network-hit-callout">No lethal combination recorded for this run.</div>
            </div>
          </aside>
        </div>
      </section>

      <section class="section reveal split-section">
        <div class="chart-card">
          <div class="section-heading compact">
            <div>
              <p class="eyebrow">Top DEGs</p>
              <h2>Largest malignant versus normal shifts</h2>
            </div>
          </div>
          <canvas id="deg-chart" height="320"></canvas>
        </div>
        <div class="chart-card">
          <div class="section-heading compact">
            <div>
              <p class="eyebrow">Benchmark</p>
              <h2>Dependency support for nominated genes</h2>
            </div>
          </div>
          <canvas id="benchmark-chart" height="320"></canvas>
        </div>
      </section>

      <section class="section reveal split-section">
        <div class="data-card">
          <div class="section-heading compact">
            <div>
              <p class="eyebrow">Knockout outcomes</p>
              <h2>Cheat-code candidates</h2>
            </div>
          </div>
          <div id="knockout-grid" class="stack-grid"></div>
        </div>
        <div class="data-card">
          <div class="section-heading compact">
            <div>
              <p class="eyebrow">Verified literature edges</p>
              <h2>LLM-backed mechanistic evidence</h2>
            </div>
          </div>
          <div id="interaction-list" class="interaction-list"></div>
        </div>
      </section>

      <section class="section reveal">
        <div class="section-heading">
          <div>
            <p class="eyebrow">Benchmark table</p>
            <h2>DepMap and RNAi support</h2>
          </div>
          <p class="section-copy">More negative gene-effect values are better. The table reflects the selected experiment's final benchmark report.</p>
        </div>
        <div class="table-shell">
          <table>
            <thead>
              <tr>
                <th>Gene</th>
                <th>Mean CRISPR effect</th>
                <th>Hit rate</th>
                <th>Driver alignment</th>
                <th>Combined support</th>
                <th>Benchmark hit</th>
              </tr>
            </thead>
            <tbody id="benchmark-table"></tbody>
          </table>
        </div>
      </section>

      <section class="section reveal">
        <div class="section-heading">
          <div>
            <p class="eyebrow">Run diagnostics</p>
            <h2>Research execution and provenance</h2>
          </div>
          <p class="section-copy">This section shows exactly which model or fallback generated the literature graph and how much of the final network came from priors versus verified edges.</p>
        </div>
        <div id="diagnostic-grid" class="diagnostic-grid"></div>
      </section>
    </main>

    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script src="app.js"></script>
  </body>
</html>
"""


STYLESHEET = """
:root {
  --bg: #f4ecdf;
  --panel: rgba(255, 251, 245, 0.88);
  --panel-strong: #fff9f0;
  --ink: #17202a;
  --muted: #5a646f;
  --accent: #1f6f61;
  --accent-soft: rgba(31, 111, 97, 0.13);
  --warning: #a74c2d;
  --warning-soft: rgba(167, 76, 45, 0.14);
  --border: rgba(23, 32, 42, 0.08);
  --shadow: 0 18px 48px rgba(73, 54, 31, 0.12);
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background:
    radial-gradient(circle at top left, rgba(215, 145, 72, 0.22), transparent 30%),
    radial-gradient(circle at 85% 12%, rgba(31, 111, 97, 0.16), transparent 28%),
    linear-gradient(180deg, #f9f1e6 0%, #f4ecdf 35%, #efe4d4 100%);
  color: var(--ink);
  font-family: "Space Grotesk", sans-serif;
  min-height: 100vh;
}

.background-orb {
  position: fixed;
  border-radius: 999px;
  filter: blur(18px);
  opacity: 0.45;
  pointer-events: none;
  z-index: 0;
}

.orb-a {
  width: 260px;
  height: 260px;
  top: 7rem;
  right: 6rem;
  background: rgba(215, 145, 72, 0.22);
}

.orb-b {
  width: 320px;
  height: 320px;
  bottom: 8rem;
  left: -4rem;
  background: rgba(31, 111, 97, 0.12);
}

.hero,
.page-shell {
  position: relative;
  z-index: 1;
}

.hero {
  display: grid;
  grid-template-columns: minmax(0, 1.6fr) minmax(300px, 0.8fr);
  gap: 1.5rem;
  padding: 3rem clamp(1.2rem, 3vw, 3rem) 1.8rem;
}

.hero-copy,
.hero-panel,
.section,
.chart-card,
.data-card,
.metric-card,
.comparison-card,
.experiment-card {
  backdrop-filter: blur(16px);
  background: var(--panel);
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
}

.hero-copy {
  padding: 2.2rem;
  border-radius: 28px;
}

.hero-panel {
  border-radius: 24px;
  padding: 1.7rem;
  align-self: stretch;
}

.eyebrow,
.panel-label {
  text-transform: uppercase;
  letter-spacing: 0.18em;
  font-size: 0.72rem;
  color: var(--muted);
  margin: 0 0 0.8rem;
}

h1,
h2,
h3 {
  margin: 0;
  line-height: 0.95;
}

h1,
h2 {
  font-family: "Instrument Serif", serif;
  font-weight: 400;
}

h1 {
  font-size: clamp(3.2rem, 8vw, 6.4rem);
  max-width: 10ch;
}

h2 {
  font-size: clamp(2rem, 4.5vw, 3.4rem);
}

h3 {
  font-size: 1rem;
}

.hero-summary,
.section-copy,
.metric-caption,
.subtle {
  color: var(--muted);
}

.hero-summary {
  font-size: 1.08rem;
  line-height: 1.65;
  max-width: 56ch;
  margin: 1.1rem 0 0;
}

.hero-pills,
.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.65rem;
  margin-top: 1.4rem;
}

.pill,
.chip,
.ghost-button {
  border-radius: 999px;
  border: 1px solid transparent;
  font: inherit;
}

.pill {
  padding: 0.55rem 0.9rem;
  background: rgba(255, 255, 255, 0.76);
  border-color: rgba(23, 32, 42, 0.08);
  font-size: 0.9rem;
}

.pill strong {
  color: var(--accent);
}

.hero-stats {
  margin: 1.25rem 0 0;
  display: grid;
  gap: 1rem;
}

.hero-stats div {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.95rem 0;
  border-top: 1px solid rgba(23, 32, 42, 0.08);
}

.hero-stats dt {
  color: var(--muted);
}

.hero-stats dd {
  margin: 0;
  font-weight: 700;
}

.page-shell {
  padding: 0 1.2rem 3rem;
}

.section,
.chart-card,
.data-card {
  border-radius: 24px;
  padding: 1.4rem;
  margin: 1rem auto 0;
  width: min(1320px, 100%);
}

.metric-grid,
.comparison-grid,
.experiment-grid,
.diagnostic-grid,
.stack-grid {
  width: min(1320px, 100%);
  margin: 0 auto;
}

.metric-grid {
  display: grid;
  gap: 1rem;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
}

.metric-card,
.comparison-card,
.experiment-card {
  border-radius: 20px;
  padding: 1.1rem;
}

.metric-card {
  min-height: 140px;
}

.metric-value {
  font-family: "Instrument Serif", serif;
  font-size: 2.6rem;
  margin-top: 0.4rem;
}

.comparison-grid,
.experiment-grid,
.diagnostic-grid,
.stack-grid {
  display: grid;
  gap: 1rem;
}

.comparison-grid {
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
}

.experiment-grid {
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
}

.diagnostic-grid {
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
}

.comparison-card.primary {
  border: 1px solid rgba(31, 111, 97, 0.24);
}

.comparison-card.baseline {
  border: 1px solid rgba(167, 76, 45, 0.2);
}

.section-heading {
  display: flex;
  justify-content: space-between;
  gap: 1.4rem;
  align-items: end;
  margin-bottom: 1.2rem;
}

.section-heading.compact {
  margin-bottom: 0.8rem;
}

.network-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 0.8rem;
  margin-bottom: 1rem;
}

.search-shell {
  display: grid;
  gap: 0.35rem;
  min-width: min(100%, 320px);
}

.search-shell span {
  font-size: 0.82rem;
  color: var(--muted);
}

input[type="search"] {
  border-radius: 14px;
  border: 1px solid rgba(23, 32, 42, 0.12);
  padding: 0.85rem 1rem;
  font: inherit;
  background: rgba(255, 255, 255, 0.84);
}

.ghost-button {
  padding: 0.8rem 1rem;
  background: rgba(255, 255, 255, 0.7);
  border-color: rgba(23, 32, 42, 0.08);
  cursor: pointer;
}

.ghost-button.active,
.chip.active {
  background: var(--accent);
  color: white;
  border-color: var(--accent);
}

.chip {
  padding: 0.55rem 0.85rem;
  background: white;
  border: 1px solid rgba(23, 32, 42, 0.08);
  cursor: pointer;
}

.network-layout {
  display: grid;
  grid-template-columns: minmax(0, 1.4fr) minmax(240px, 0.6fr);
  gap: 1rem;
  align-items: stretch;
}

.network-canvas,
.network-sidepanel {
  border-radius: 22px;
  border: 1px solid rgba(23, 32, 42, 0.08);
  background: rgba(255, 255, 255, 0.7);
}

.network-canvas {
  min-height: 680px;
}

.network-sidepanel {
  padding: 1rem;
}

.legend-list,
.interaction-list,
.stack-grid {
  list-style: none;
  padding: 0;
  margin: 0;
}

.legend-list {
  display: grid;
  gap: 0.55rem;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

.legend-swatch {
  width: 14px;
  height: 14px;
  border-radius: 999px;
}

.network-callout {
  margin-top: 1.2rem;
  padding: 1rem;
  border-radius: 18px;
  background: linear-gradient(180deg, rgba(31, 111, 97, 0.1), rgba(31, 111, 97, 0.04));
}

.split-section {
  display: grid;
  gap: 1rem;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.data-card {
  min-height: 100%;
}

.knockout-card,
.interaction-card,
.diagnostic-card {
  padding: 1rem;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.7);
  border: 1px solid rgba(23, 32, 42, 0.08);
}

.interaction-list,
.stack-grid {
  display: grid;
  gap: 0.85rem;
}

.hit-genes {
  font-size: 1.2rem;
  font-weight: 700;
  margin: 0 0 0.4rem;
}

.mini-pill-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin-top: 0.7rem;
}

.mini-pill {
  padding: 0.35rem 0.6rem;
  border-radius: 999px;
  font-size: 0.82rem;
  background: rgba(31, 111, 97, 0.11);
}

.warning {
  background: var(--warning-soft);
  color: var(--warning);
}

.good {
  background: var(--accent-soft);
  color: var(--accent);
}

.table-shell {
  overflow-x: auto;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.66);
  border: 1px solid rgba(23, 32, 42, 0.08);
}

table {
  width: 100%;
  border-collapse: collapse;
}

th,
td {
  text-align: left;
  padding: 0.9rem 1rem;
  border-bottom: 1px solid rgba(23, 32, 42, 0.08);
  font-size: 0.94rem;
}

th {
  color: var(--muted);
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.reveal {
  animation: rise-in 620ms ease both;
}

@keyframes rise-in {
  from {
    opacity: 0;
    transform: translateY(12px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (max-width: 1080px) {
  .hero,
  .network-layout,
  .split-section {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 720px) {
  .hero {
    padding: 1.2rem 0.9rem 1rem;
  }

  .page-shell {
    padding: 0 0.9rem 2rem;
  }

  .hero-copy,
  .hero-panel,
  .section,
  .chart-card,
  .data-card {
    padding: 1rem;
  }

  .network-canvas {
    min-height: 420px;
  }

  h1 {
    font-size: 3.3rem;
  }
}
"""


APP_SCRIPT = """
const NODE_STYLES = {
  deg: { color: '#1f6f61', label: 'DEG' },
  pathway: { color: '#c26a2d', label: 'Core pathway' },
  prior: { color: '#b24f45', label: 'Curated prior' },
  intermediate: { color: '#7a5f9a', label: 'Intermediate' },
  boss: { color: '#101820', label: 'Boss node' },
  unknown: { color: '#7e8a96', label: 'Unknown' },
};

let networkInstance = null;
let networkPhysicsEnabled = true;
let networkNodes = [];
let networkEdges = [];
let activeKinds = new Set(Object.keys(NODE_STYLES));

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${path}: ${response.status}`);
  }
  return response.json();
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '-';
  }
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(Number(value));
}

function formatSigned(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '-';
  }
  const number = Number(value);
  return number > 0 ? `+${number.toFixed(2)}` : number.toFixed(2);
}

function asList(value) {
  return Array.isArray(value) ? value : [];
}

async function loadRun(runMeta) {
  const files = runMeta.files;
  const [summary, topDegs, interactions, graph, knockoutHits, benchmarkReport, experimentReport, researchExecution, priorKnowledge] = await Promise.all([
    fetchJson(files.summary),
    fetchJson(files.top_degs),
    fetchJson(files.gene_interactions),
    fetchJson(files.regulatory_graph),
    fetchJson(files.knockout_hits),
    fetchJson(files.benchmark_report),
    fetchJson(files.experiment_report),
    fetchJson(files.research_execution),
    fetchJson(files.prior_knowledge),
  ]);
  return { meta: runMeta, summary, topDegs, interactions, graph, knockoutHits, benchmarkReport, experimentReport, researchExecution, priorKnowledge };
}

function setText(id, value) {
  const element = document.getElementById(id);
  if (element) {
    element.textContent = value;
  }
}

function renderHero(manifest, primary, baseline) {
  setText('hero-title', manifest.title);
  setText('hero-run-name', primary.meta.label);
  setText(
    'hero-summary',
    `Primary run processed ${formatNumber(primary.summary.dataset_cells)} cells, distilled ${formatNumber(primary.summary.degs.length)} DEGs, and selected the ${primary.summary.selected_experiment || 'latest'} experiment for knockout search.`
  );
  setText('hero-graph', `${formatNumber(primary.summary.graph_nodes)} nodes / ${formatNumber(primary.summary.graph_edges)} edges`);
  setText('hero-hits', `${formatNumber(primary.summary.knockout_hits.length)} lethal combinations`);
  setText('hero-benchmark', `${formatNumber(primary.summary.benchmark_report.model_count)} pancreatic models`);

  const pills = [
    { label: 'Primary run', value: primary.meta.label },
    { label: 'Selected experiment', value: primary.summary.selected_experiment || 'n/a' },
    { label: 'Research backend', value: Object.keys(primary.researchExecution.result_model_counts || {}).join(', ') || 'unknown' },
  ];
  if (baseline) {
    pills.push({ label: 'Baseline', value: baseline.meta.label });
  }
  document.getElementById('hero-pills').innerHTML = pills
    .map((pill) => `<span class="pill"><strong>${pill.label}</strong> ${pill.value}</span>`)
    .join('');
}

function renderOverview(primary, baseline) {
  const metrics = [
    {
      title: 'Dataset cells',
      value: formatNumber(primary.summary.dataset_cells),
      caption: `Across ${formatNumber(primary.summary.dataset_genes)} genes after QC.`,
    },
    {
      title: 'Top DEGs',
      value: formatNumber(primary.topDegs.length),
      caption: `Selected by Scanpy differential expression.`,
    },
    {
      title: 'Verified literature edges',
      value: formatNumber(primary.interactions.reduce((count, item) => count + asList(item.interactions).length, 0)),
      caption: `Structured edges that survived verification for the selected run.`,
    },
    {
      title: 'Prior knowledge seed',
      value: formatNumber(primary.priorKnowledge.edge_count),
      caption: `Edges imported from KEGG, Reactome, OmniPath, and Pathway Commons.`,
    },
    {
      title: 'Benchmark-supported genes',
      value: formatNumber(primary.benchmarkReport.results.filter((item) => item.benchmark_hit).length),
      caption: `Genes meeting the final benchmark threshold in the selected experiment.`,
    },
    {
      title: 'Fallback genes',
      value: formatNumber(primary.researchExecution.fallback_gene_count),
      caption: baseline ? 'Useful when comparing provider-backed runs against fallback behavior.' : 'Lower is better; high values mean provider fallback dominated.',
    },
  ];
  document.getElementById('overview-metrics').innerHTML = metrics
    .map(
      (metric) => `
        <article class="metric-card">
          <p class="panel-label">${metric.title}</p>
          <div class="metric-value">${metric.value}</div>
          <p class="metric-caption">${metric.caption}</p>
        </article>
      `
    )
    .join('');
}

function renderComparison(primary, baseline) {
  const cards = [primary, baseline].filter(Boolean).map((run) => {
    const className = run === primary ? 'comparison-card primary' : 'comparison-card baseline';
    const hit = asList(run.summary.knockout_hits)[0];
    const knockoutLabel = hit ? hit.knocked_out_genes.join(' + ') : 'No lethal combination';
    const benchmarkHitCount = run.summary.benchmark_report.results.filter((item) => item.benchmark_hit).length;
    return `
      <article class="${className}">
        <p class="panel-label">${run.meta.label}</p>
        <h3>${run.summary.selected_experiment || 'n/a'}</h3>
        <div class="mini-pill-row">
          <span class="mini-pill">${formatNumber(run.summary.graph_nodes)} nodes</span>
          <span class="mini-pill">${formatNumber(run.summary.graph_edges)} edges</span>
          <span class="mini-pill">${formatNumber(run.summary.knockout_hits.length)} hits</span>
        </div>
        <p class="hero-summary">Top hit: ${knockoutLabel}.</p>
        <p class="subtle">Benchmark hits: ${formatNumber(benchmarkHitCount)} of ${formatNumber(run.summary.benchmark_report.results.length)} nominated genes.</p>
      </article>
    `;
  });
  document.getElementById('comparison-grid').innerHTML = cards.join('');
}

function renderExperiments(primary) {
  const cards = primary.experimentReport.map((experiment) => {
    const benchmarkHits = experiment.benchmark_report.results.filter((item) => item.benchmark_hit).length;
    const leadHit = experiment.knockout_hits[0]?.knocked_out_genes?.join(' + ') || 'No lethal combination';
    return `
      <article class="experiment-card">
        <p class="panel-label">${experiment.selected ? 'Selected experiment' : 'Experiment'}</p>
        <h3>${experiment.name}</h3>
        <p class="subtle">${experiment.description || 'No description recorded.'}</p>
        <div class="mini-pill-row">
          <span class="mini-pill">${formatNumber(experiment.graph_nodes)} nodes</span>
          <span class="mini-pill">${formatNumber(experiment.graph_edges)} edges</span>
          <span class="mini-pill">${formatNumber(experiment.pruned_genes.length)} pruned</span>
        </div>
        <p class="hero-summary">Lead outcome: ${leadHit}</p>
        <p class="subtle">Benchmark hits: ${benchmarkHits}. Score: ${formatSigned(experiment.score)}</p>
      </article>
    `;
  });
  document.getElementById('experiment-grid').innerHTML = cards.join('');
}

function renderNetwork(primary) {
  const graph = primary.graph;
  networkNodes = graph.nodes.map((node) => {
    const style = NODE_STYLES[node.kind] || NODE_STYLES.unknown;
    return {
      ...node,
      color: style.color,
      label: node.id,
      title: `${node.id} (${node.kind || 'unknown'})`,
    };
  });
  networkEdges = graph.edges.map((edge) => ({
    ...edge,
    from: edge.source,
    to: edge.target,
    color: edge.sign === -1 ? '#b24f45' : '#1f6f61',
  }));

  const filterHost = document.getElementById('network-kind-filters');
  filterHost.innerHTML = Object.entries(NODE_STYLES)
    .map(
      ([kind, style]) => `
        <button type="button" class="chip active" data-kind="${kind}">
          ${style.label}
        </button>
      `
    )
    .join('');
  filterHost.querySelectorAll('.chip').forEach((button) => {
    button.addEventListener('click', () => {
      const kind = button.dataset.kind;
      if (activeKinds.has(kind)) {
        activeKinds.delete(kind);
        button.classList.remove('active');
      } else {
        activeKinds.add(kind);
        button.classList.add('active');
      }
      drawNetwork(primary);
    });
  });

  document.getElementById('network-focus').addEventListener('click', () => {
    const value = document.getElementById('network-search').value.trim().toUpperCase();
    if (!value || !networkInstance) {
      return;
    }
    const match = networkNodes.find((node) => node.id.toUpperCase() === value);
    if (match) {
      networkInstance.focus(match.id, { scale: 1.3, animation: true });
      networkInstance.selectNodes([match.id]);
    }
  });

  document.getElementById('network-physics').addEventListener('click', (event) => {
    networkPhysicsEnabled = !networkPhysicsEnabled;
    event.currentTarget.classList.toggle('active', !networkPhysicsEnabled);
    event.currentTarget.textContent = networkPhysicsEnabled ? 'Pause motion' : 'Resume motion';
    drawNetwork(primary);
  });

  drawNetwork(primary);
  document.getElementById('network-legend').innerHTML = Object.values(NODE_STYLES)
    .map(
      (style) => `
        <li class="legend-item">
          <span class="legend-swatch" style="background:${style.color}"></span>
          <span>${style.label}</span>
        </li>
      `
    )
    .join('');

  const topHit = primary.knockoutHits[0];
  document.getElementById('network-hit-callout').innerHTML = topHit
    ? `
      <p class="hit-genes">${topHit.knocked_out_genes.join(' + ')}</p>
      <p class="subtle">Boss node state: ${topHit.boss_state}. Pathway nodes off: ${topHit.pathway_nodes_off.join(', ')}</p>
    `
    : `<p class="subtle">This run did not produce a lethal combination after simulation.</p>`;
}

function drawNetwork(primary) {
  const knockoutGenes = new Set((primary.knockoutHits[0]?.knocked_out_genes || []).map((gene) => gene.toUpperCase()));
  const visibleNodes = networkNodes.filter((node) => activeKinds.has(node.kind || 'unknown'));
  const visibleNodeIds = new Set(visibleNodes.map((node) => node.id));
  const degree = new Map();
  networkEdges.forEach((edge) => {
    degree.set(edge.from, (degree.get(edge.from) || 0) + 1);
    degree.set(edge.to, (degree.get(edge.to) || 0) + 1);
  });

  const nodeDataset = new vis.DataSet(
    visibleNodes.map((node) => ({
      id: node.id,
      label: node.label,
      color: node.color,
      borderWidth: knockoutGenes.has(node.id.toUpperCase()) ? 4 : 1.5,
      size: 16 + Math.min(20, (degree.get(node.id) || 0) * 1.8),
      font: { face: 'Space Grotesk', color: '#17202a', size: 18 },
      shape: 'dot',
      title: `${node.id} | ${node.kind || 'unknown'} | degree ${(degree.get(node.id) || 0)}`,
    }))
  );
  const edgeDataset = new vis.DataSet(
    networkEdges
      .filter((edge) => visibleNodeIds.has(edge.from) && visibleNodeIds.has(edge.to))
      .map((edge) => ({
        from: edge.from,
        to: edge.to,
        arrows: 'to',
        color: { color: edge.color, highlight: edge.color, hover: edge.color },
        width: Math.max(1.5, Number(edge.weight || edge.confidence || 1) * 3),
        smooth: { type: 'dynamic' },
        title: `${edge.from} ${edge.sign === -1 ? '-|' : '->'} ${edge.to} | provenance: ${(edge.provenance || []).join(', ') || 'unknown'}`,
      }))
  );
  const container = document.getElementById('network-canvas');
  networkInstance = new vis.Network(
    container,
    { nodes: nodeDataset, edges: edgeDataset },
    {
      autoResize: true,
      interaction: { hover: true, multiselect: true },
      physics: networkPhysicsEnabled
        ? {
            solver: 'forceAtlas2Based',
            forceAtlas2Based: { gravitationalConstant: -70, centralGravity: 0.008, springLength: 130 },
            stabilization: { iterations: 180 },
          }
        : false,
      edges: { selectionWidth: 2.2 },
      nodes: { shadow: true },
    }
  );
}

function renderDegChart(primary) {
  const topGenes = primary.topDegs.slice(0, 15);
  new Chart(document.getElementById('deg-chart'), {
    type: 'bar',
    data: {
      labels: topGenes.map((item) => item.gene),
      datasets: [
        {
          label: 'log2 fold change',
          data: topGenes.map((item) => item.log2_fold_change),
          backgroundColor: '#1f6f61',
          borderRadius: 12,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#5a646f' } },
        y: { ticks: { color: '#5a646f' } },
      },
    },
  });
}

function renderBenchmarkChart(primary) {
  const sorted = [...primary.benchmarkReport.results]
    .sort((left, right) => (left.mean_gene_effect ?? 999) - (right.mean_gene_effect ?? 999))
    .slice(0, 15);
  new Chart(document.getElementById('benchmark-chart'), {
    type: 'bar',
    data: {
      labels: sorted.map((item) => item.gene_symbol),
      datasets: [
        {
          label: 'mean CRISPR effect',
          data: sorted.map((item) => item.mean_gene_effect),
          backgroundColor: sorted.map((item) => (item.benchmark_hit ? '#1f6f61' : '#c26a2d')),
          borderRadius: 12,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#5a646f' } },
        y: { ticks: { color: '#5a646f' } },
      },
    },
  });
}

function renderKnockouts(primary, baseline) {
  const baselineTopHit = baseline?.summary.knockout_hits?.[0]?.knocked_out_genes?.join(' + ') || null;
  const host = document.getElementById('knockout-grid');
  if (!primary.knockoutHits.length) {
    host.innerHTML = `<div class="knockout-card"><p class="subtle">No lethal combination was found for the selected experiment.</p></div>`;
    return;
  }
  host.innerHTML = primary.knockoutHits
    .map(
      (hit, index) => `
        <article class="knockout-card">
          <p class="panel-label">Combination ${index + 1}</p>
          <p class="hit-genes">${hit.knocked_out_genes.join(' + ')}</p>
          <p class="subtle">Boss state: ${hit.boss_state}. Convergence: ${hit.convergence_steps} step(s). Pathway shutdown: ${hit.pathway_nodes_off.length} nodes.</p>
          <div class="mini-pill-row">
            <span class="mini-pill">Score ${formatSigned(hit.score)}</span>
            <span class="mini-pill">Support ${formatSigned(hit.support_score)}</span>
            <span class="mini-pill ${hit.benchmark_score > 0 ? 'good' : 'warning'}">Benchmark ${formatSigned(hit.benchmark_score)}</span>
          </div>
          ${baselineTopHit ? `<p class="subtle">Baseline top hit: ${baselineTopHit}</p>` : ''}
        </article>
      `
    )
    .join('');
}

function renderInteractions(primary) {
  const interactions = primary.interactions.flatMap((item) => asList(item.interactions)).slice(0, 20);
  const host = document.getElementById('interaction-list');
  if (!interactions.length) {
    host.innerHTML = `<div class="interaction-card"><p class="subtle">No verified literature edges were retained for this run.</p></div>`;
    return;
  }
  host.innerHTML = interactions
    .map(
      (edge) => `
        <article class="interaction-card">
          <p class="hit-genes">${edge.source_gene} ${edge.interaction_type === -1 ? '-|' : '->'} ${edge.target}</p>
          <p class="subtle">${edge.evidence_summary || 'No evidence summary provided.'}</p>
          <div class="mini-pill-row">
            <span class="mini-pill">Confidence ${formatSigned(edge.confidence_score)}</span>
            <span class="mini-pill">Depth ${edge.mechanistic_depth}</span>
            <span class="mini-pill">${(edge.pmid_citations || []).length} PMID(s)</span>
          </div>
        </article>
      `
    )
    .join('');
}

function renderBenchmarkTable(primary) {
  const rows = [...primary.benchmarkReport.results]
    .sort((left, right) => (left.mean_gene_effect ?? 999) - (right.mean_gene_effect ?? 999))
    .map(
      (item) => `
        <tr>
          <td>${item.gene_symbol}</td>
          <td>${formatSigned(item.mean_gene_effect)}</td>
          <td>${formatNumber(item.hit_rate)}</td>
          <td>${formatNumber(item.driver_alignment_score)}</td>
          <td>${formatSigned(item.combined_support_score)}</td>
          <td>${item.benchmark_hit ? '<span class="mini-pill good">Yes</span>' : '<span class="mini-pill warning">No</span>'}</td>
        </tr>
      `
    )
    .join('');
  document.getElementById('benchmark-table').innerHTML = rows;
}

function renderDiagnostics(primary) {
  const cards = [
    {
      title: 'Research model usage',
      body: Object.entries(primary.researchExecution.result_model_counts || {})
        .map(([name, count]) => `${name}: ${count}`)
        .join(', ') || 'No model usage recorded.',
    },
    {
      title: 'Fallback count',
      body: `${formatNumber(primary.researchExecution.fallback_gene_count)} genes used fallback research output.`,
    },
    {
      title: 'Prior knowledge seed',
      body: `${formatNumber(primary.priorKnowledge.node_count)} nodes and ${formatNumber(primary.priorKnowledge.edge_count)} curated edges were imported before simulation.`,
    },
    {
      title: 'Selected experiment',
      body: `${primary.summary.selected_experiment || 'n/a'} with ${formatNumber(primary.summary.graph_nodes)} nodes and ${formatNumber(primary.summary.graph_edges)} edges.`,
    },
  ];
  document.getElementById('diagnostic-grid').innerHTML = cards
    .map(
      (item) => `
        <article class="diagnostic-card">
          <p class="panel-label">${item.title}</p>
          <p class="subtle">${item.body}</p>
        </article>
      `
    )
    .join('');
}

async function main() {
  const manifest = await fetchJson('data/manifest.json');
  const primary = await loadRun(manifest.runs.primary);
  const baseline = manifest.runs.baseline ? await loadRun(manifest.runs.baseline) : null;
  renderHero(manifest, primary, baseline);
  renderOverview(primary, baseline);
  renderComparison(primary, baseline);
  renderExperiments(primary);
  renderNetwork(primary);
  renderDegChart(primary);
  renderBenchmarkChart(primary);
  renderKnockouts(primary, baseline);
  renderInteractions(primary);
  renderBenchmarkTable(primary);
  renderDiagnostics(primary);
}

main().catch((error) => {
  console.error(error);
  document.getElementById('hero-summary').textContent = `Failed to load site data: ${error.message}`;
});
"""
