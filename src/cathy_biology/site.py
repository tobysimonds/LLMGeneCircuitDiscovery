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
    "analysis_interactions": "analysis_interactions.json",
    "regulatory_graph": "regulatory_graph.json",
    "regulatory_graph_projected": "regulatory_graph_projected.json",
    "deg_graph_with_llm": "deg_graph_with_llm.json",
    "deg_graph_prior_only": "deg_graph_prior_only.json",
    "knockout_hits": "knockout_hits.json",
    "benchmark_report": "benchmark_report.json",
    "pre_simulation_benchmark": "pre_simulation_benchmark.json",
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

    manifest: dict[str, Any] = {
        "title": title,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": {
            "primary": _copy_run_bundle(primary_run_dir, data_dir, slug="primary", label="Primary run"),
        },
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
    for filename in REQUIRED_RUN_FILES.values():
        source = run_dir / filename
        if not source.exists():
            raise FileNotFoundError(f"Run directory {run_dir} is missing required artifact {filename}.")
        shutil.copy2(source, target_dir / filename)

    payload = _build_site_bundle(target_dir)
    write_json(target_dir / "site_bundle.json", payload)

    summary = payload["summary"]
    return {
        "slug": slug,
        "label": label,
        "source_dir": str(run_dir),
        "bundle": f"data/{slug}/site_bundle.json",
        "graph_nodes": summary["graph_nodes"],
        "graph_edges": summary["graph_edges"],
        "dataset_cells": summary["dataset_cells"],
        "dataset_genes": summary["dataset_genes"],
        "top_hit": summary["top_hit"],
    }


def _build_site_bundle(run_dir: Path) -> dict[str, Any]:
    summary = json.loads((run_dir / REQUIRED_RUN_FILES["summary"]).read_text(encoding="utf-8"))
    top_degs = json.loads((run_dir / REQUIRED_RUN_FILES["top_degs"]).read_text(encoding="utf-8"))
    analysis_interactions = json.loads((run_dir / REQUIRED_RUN_FILES["analysis_interactions"]).read_text(encoding="utf-8"))
    knockout_hits = json.loads((run_dir / REQUIRED_RUN_FILES["knockout_hits"]).read_text(encoding="utf-8"))
    benchmark_report = json.loads((run_dir / REQUIRED_RUN_FILES["benchmark_report"]).read_text(encoding="utf-8"))
    pre_benchmark_report = json.loads((run_dir / REQUIRED_RUN_FILES["pre_simulation_benchmark"]).read_text(encoding="utf-8"))
    experiment_report = json.loads((run_dir / REQUIRED_RUN_FILES["experiment_report"]).read_text(encoding="utf-8"))
    research_execution = json.loads((run_dir / REQUIRED_RUN_FILES["research_execution"]).read_text(encoding="utf-8"))
    prior_knowledge = json.loads((run_dir / REQUIRED_RUN_FILES["prior_knowledge"]).read_text(encoding="utf-8"))
    graphs = {
        "selected": _normalize_graph(json.loads((run_dir / REQUIRED_RUN_FILES["regulatory_graph"]).read_text(encoding="utf-8"))),
        "projected": _normalize_graph(json.loads((run_dir / REQUIRED_RUN_FILES["regulatory_graph_projected"]).read_text(encoding="utf-8"))),
        "deg_llm": _normalize_graph(json.loads((run_dir / REQUIRED_RUN_FILES["deg_graph_with_llm"]).read_text(encoding="utf-8"))),
        "deg_prior": _normalize_graph(json.loads((run_dir / REQUIRED_RUN_FILES["deg_graph_prior_only"]).read_text(encoding="utf-8"))),
    }
    edge_evidence = _build_edge_evidence_index(analysis_interactions)
    node_profiles = _build_node_profiles(graphs, analysis_interactions, top_degs, benchmark_report)

    for graph in graphs.values():
        _attach_graph_evidence(graph, edge_evidence)

    top_hit = knockout_hits[0]["knocked_out_genes"] if knockout_hits else []
    return {
        "summary": {
            "dataset_cells": summary.get("dataset_cells"),
            "dataset_genes": summary.get("dataset_genes"),
            "graph_nodes": summary.get("graph_nodes"),
            "graph_edges": summary.get("graph_edges"),
            "selected_experiment": summary.get("selected_experiment"),
            "top_hit": top_hit,
            "knockout_count": len(knockout_hits),
            "benchmark_model_count": benchmark_report.get("model_count", 0),
        },
        "graphs": graphs,
        "node_profiles": node_profiles,
        "top_degs": [
            {
                "gene": row.get("gene"),
                "log2_fold_change": row.get("log2_fold_change"),
                "adjusted_pvalue": row.get("adjusted_pvalue"),
            }
            for row in top_degs
        ],
        "knockout_hits": knockout_hits,
        "benchmark_report": benchmark_report,
        "pre_simulation_benchmark": pre_benchmark_report,
        "experiment_report": experiment_report,
        "research_execution": research_execution,
        "prior_knowledge": {
            "node_count": prior_knowledge.get("node_count", 0),
            "edge_count": prior_knowledge.get("edge_count", 0),
            "source_counts": prior_knowledge.get("source_counts", {}),
        },
    }


def _normalize_graph(raw_graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "id": node.get("id"),
                "kind": node.get("kind", "unknown"),
                "logic_mode": node.get("logic_mode", ""),
                "basal_state": node.get("basal_state", 0),
                "activation_threshold": node.get("activation_threshold"),
                "inhibition_dominance": node.get("inhibition_dominance"),
            }
            for node in raw_graph.get("nodes", [])
        ],
        "edges": [
            {
                "source": edge.get("source"),
                "target": edge.get("target"),
                "sign": edge.get("sign", 1),
                "weight": edge.get("weight"),
                "confidence": edge.get("confidence"),
                "provenance": edge.get("provenance", []),
                "collapsed_via": edge.get("collapsed_via", []),
                "collapsed_path": edge.get("collapsed_path", []),
                "path_length": edge.get("path_length"),
                "evidence_scores": edge.get("evidence_scores", {}),
                "benchmark_support_score": edge.get("benchmark_support_score", 0.0),
                "direct_evidence": [],
            }
            for edge in raw_graph.get("edges", [])
        ],
    }


def _build_edge_evidence_index(analysis_interactions: list[dict[str, Any]]) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    index: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for result in analysis_interactions:
        seed_gene = result.get("source_gene")
        for edge in result.get("interactions", []):
            key = (
                edge.get("source_gene"),
                edge.get("target"),
                int(edge.get("interaction_type", 0)),
            )
            index.setdefault(key, []).append(
                {
                    "seed_gene": seed_gene,
                    "source_gene": edge.get("source_gene"),
                    "target": edge.get("target"),
                    "interaction_type": edge.get("interaction_type"),
                    "confidence_score": edge.get("confidence_score"),
                    "evidence_summary": edge.get("evidence_summary", ""),
                    "pmid_citations": edge.get("pmid_citations", []),
                    "source_refs": edge.get("source_refs", []),
                    "provenance_sources": edge.get("provenance_sources", []),
                    "source_type": edge.get("source_type", "unknown"),
                    "target_type": edge.get("target_type", "unknown"),
                    "mechanistic_depth": edge.get("mechanistic_depth", 1),
                    "evidence_scores": edge.get("evidence_scores", {}),
                }
            )
    return index


def _attach_graph_evidence(graph: dict[str, Any], edge_evidence: dict[tuple[str, str, int], list[dict[str, Any]]]) -> None:
    for edge in graph["edges"]:
        key = (edge["source"], edge["target"], int(edge["sign"]))
        edge["direct_evidence"] = edge_evidence.get(key, [])


def _build_node_profiles(
    graphs: dict[str, dict[str, Any]],
    analysis_interactions: list[dict[str, Any]],
    top_degs: list[dict[str, Any]],
    benchmark_report: dict[str, Any],
) -> dict[str, Any]:
    deg_index = {row["gene"]: row for row in top_degs}
    benchmark_index = {row["gene_symbol"]: row for row in benchmark_report.get("results", [])}
    interaction_index: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for result in analysis_interactions:
        for edge in result.get("interactions", []):
            interaction_index.setdefault(edge["source_gene"], {"outgoing": [], "incoming": []})["outgoing"].append(edge)
            interaction_index.setdefault(edge["target"], {"outgoing": [], "incoming": []})["incoming"].append(edge)

    profiles: dict[str, Any] = {}
    all_nodes = {node["id"] for graph in graphs.values() for node in graph["nodes"]}
    for node_id in sorted(all_nodes):
        graph_presence = {}
        for graph_name, graph in graphs.items():
            incoming = [edge for edge in graph["edges"] if edge["target"] == node_id]
            outgoing = [edge for edge in graph["edges"] if edge["source"] == node_id]
            if incoming or outgoing or any(node["id"] == node_id for node in graph["nodes"]):
                graph_presence[graph_name] = {
                    "incoming_count": len(incoming),
                    "outgoing_count": len(outgoing),
                }
        profiles[node_id] = {
            "node_id": node_id,
            "deg_stats": deg_index.get(node_id),
            "benchmark": benchmark_index.get(node_id),
            "interactions": interaction_index.get(node_id, {"outgoing": [], "incoming": []}),
            "graph_presence": graph_presence,
        }
    return profiles


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>__SITE_TITLE__</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet" />
    <link rel="stylesheet" href="https://unpkg.com/vis-network/styles/vis-network.min.css" />
    <link rel="stylesheet" href="styles.css" />
  </head>
  <body>
    <div class="page-bg"></div>
    <header class="hero">
      <div>
        <p class="eyebrow">Interactive evidence explorer</p>
        <h1 id="site-title">__SITE_TITLE__</h1>
        <p id="hero-summary" class="hero-summary">Loading final run...</p>
      </div>
      <div class="hero-controls">
        <label>
          <span>Run</span>
          <select id="run-select"></select>
        </label>
        <label>
          <span>Graph view</span>
          <select id="graph-select"></select>
        </label>
        <label>
          <span>Find node</span>
          <input id="node-search" type="search" placeholder="SOX2, EFS, KRAS..." />
        </label>
      </div>
    </header>

    <main class="layout">
      <section class="panel graph-panel">
        <div class="panel-heading">
          <div>
            <p class="eyebrow">Graph explorer</p>
            <h2 id="graph-title">Graph</h2>
          </div>
          <div id="graph-stats" class="graph-stats"></div>
        </div>
        <div class="legend">
          <span><i class="swatch deg"></i> DEG</span>
          <span><i class="swatch pathway"></i> Pathway</span>
          <span><i class="swatch prior"></i> Prior / intermediate</span>
          <span><i class="swatch boss"></i> Boss node</span>
          <span><i class="swatch act"></i> Activation</span>
          <span><i class="swatch inh"></i> Inhibition</span>
        </div>
        <div id="network" class="network"></div>
      </section>

      <aside class="panel inspector-panel">
        <div class="panel-heading">
          <div>
            <p class="eyebrow">Inspector</p>
            <h2 id="inspector-title">Click a node or edge</h2>
          </div>
        </div>
        <div id="inspector-body" class="inspector-body"></div>
      </aside>

      <section class="panel data-panel">
        <div class="panel-heading">
          <div>
            <p class="eyebrow">Top findings</p>
            <h2>Knockouts, benchmarks, and DEGs</h2>
          </div>
        </div>
        <div id="summary-cards" class="summary-cards"></div>
        <div class="split">
          <div>
            <h3>Knockout suggestions</h3>
            <div id="knockout-list" class="stack"></div>
          </div>
          <div>
            <h3>Top benchmark rows</h3>
            <div id="benchmark-list" class="stack"></div>
          </div>
        </div>
        <div>
          <h3>Top DEGs</h3>
          <div id="deg-list" class="deg-list"></div>
        </div>
      </section>
    </main>

    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script src="app.js"></script>
  </body>
</html>
"""


STYLESHEET = """
:root {
  --bg: #f2efe7;
  --ink: #121212;
  --muted: #5e5a54;
  --panel: rgba(255, 252, 247, 0.94);
  --stroke: rgba(18, 18, 18, 0.12);
  --accent: #c84d28;
  --accent-2: #1a6d6d;
  --deg: #c84d28;
  --pathway: #1a6d6d;
  --prior: #8a6f2a;
  --boss: #111111;
  --act: #198754;
  --inh: #c0392b;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Space Grotesk", sans-serif;
  color: var(--ink);
  background: radial-gradient(circle at top left, #fff9ef 0%, var(--bg) 42%, #e7e1d2 100%);
}

.page-bg {
  position: fixed;
  inset: 0;
  background:
    linear-gradient(125deg, rgba(200, 77, 40, 0.10), transparent 28%),
    linear-gradient(310deg, rgba(26, 109, 109, 0.10), transparent 24%);
  pointer-events: none;
}

.hero, .layout {
  position: relative;
  z-index: 1;
}

.hero {
  padding: 32px;
  display: grid;
  gap: 24px;
  grid-template-columns: 1.4fr 1fr;
}

.eyebrow {
  margin: 0 0 8px;
  font-size: 12px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
}

#site-title {
  margin: 0 0 8px;
  font-size: clamp(34px, 5vw, 54px);
  line-height: 0.95;
}

.hero-summary {
  max-width: 70ch;
  color: var(--muted);
}

.hero-controls {
  display: grid;
  gap: 12px;
  align-content: start;
}

.hero-controls label {
  display: grid;
  gap: 6px;
  font-size: 13px;
  color: var(--muted);
}

.hero-controls select,
.hero-controls input {
  width: 100%;
  border-radius: 14px;
  border: 1px solid var(--stroke);
  background: rgba(255, 255, 255, 0.84);
  padding: 12px 14px;
  font: inherit;
  color: var(--ink);
}

.layout {
  padding: 0 32px 32px;
  display: grid;
  gap: 20px;
  grid-template-columns: minmax(0, 1.8fr) minmax(340px, 0.95fr);
  grid-template-areas:
    "graph inspector"
    "data inspector";
}

.panel {
  border: 1px solid var(--stroke);
  background: var(--panel);
  backdrop-filter: blur(16px);
  border-radius: 28px;
  box-shadow: 0 18px 60px rgba(22, 18, 10, 0.08);
  padding: 22px;
}

.graph-panel { grid-area: graph; }
.inspector-panel { grid-area: inspector; max-height: calc(100vh - 96px); overflow: auto; }
.data-panel { grid-area: data; }

.panel-heading {
  display: flex;
  justify-content: space-between;
  align-items: start;
  gap: 16px;
  margin-bottom: 16px;
}

.panel-heading h2, .panel-heading h3, .data-panel h3 {
  margin: 0;
}

.legend, .graph-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  color: var(--muted);
  font-size: 13px;
  margin-bottom: 14px;
}

.swatch {
  display: inline-block;
  width: 11px;
  height: 11px;
  border-radius: 999px;
  margin-right: 6px;
}

.swatch.deg { background: var(--deg); }
.swatch.pathway { background: var(--pathway); }
.swatch.prior { background: var(--prior); }
.swatch.boss { background: var(--boss); }
.swatch.act { background: var(--act); }
.swatch.inh { background: var(--inh); }

.network {
  height: 760px;
  border-radius: 22px;
  overflow: hidden;
  border: 1px solid rgba(18, 18, 18, 0.08);
  background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(245,240,231,0.95));
}

.inspector-body {
  display: grid;
  gap: 14px;
}

.metric-card, .evidence-card, .edge-card, .hit-card, .bench-card, .deg-chip {
  border: 1px solid rgba(18, 18, 18, 0.08);
  background: rgba(255, 255, 255, 0.82);
  border-radius: 18px;
  padding: 14px 16px;
}

.metric-card strong,
.hit-card strong,
.bench-card strong,
.edge-card strong {
  display: block;
  margin-bottom: 4px;
}

.metric-grid,
.summary-cards {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
}

.stack {
  display: grid;
  gap: 10px;
}

.split {
  display: grid;
  gap: 20px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin: 18px 0;
}

.deg-list {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.deg-chip {
  min-width: 170px;
}

.mono {
  font-family: "IBM Plex Mono", monospace;
  font-size: 12px;
}

.muted { color: var(--muted); }

.evidence-list, .edge-list {
  display: grid;
  gap: 10px;
}

.tag-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
}

.tag {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 5px 10px;
  border-radius: 999px;
  background: rgba(18, 18, 18, 0.06);
  color: var(--ink);
  font-size: 12px;
}

a {
  color: var(--accent-2);
  text-decoration: none;
}

a:hover {
  text-decoration: underline;
}

@media (max-width: 1100px) {
  .hero,
  .layout {
    grid-template-columns: 1fr;
  }
  .layout {
    grid-template-areas:
      "graph"
      "inspector"
      "data";
  }
  .inspector-panel {
    max-height: none;
  }
  .network {
    height: 560px;
  }
  .split {
    grid-template-columns: 1fr;
  }
}
"""


APP_SCRIPT = """
const manifestUrl = "data/manifest.json";
const GRAPH_LABELS = {
  selected: "Selected final graph",
  projected: "Projected simulation graph",
  deg_llm: "50-node DEG graph with LLM edges",
  deg_prior: "50-node DEG graph without LLM edges",
};

let state = {
  manifest: null,
  currentRunKey: "primary",
  currentGraphKey: "selected",
  bundle: null,
  network: null,
  nodes: null,
  edges: null,
};

const kindColors = {
  deg: "#c84d28",
  pathway: "#1a6d6d",
  prior: "#8a6f2a",
  intermediate: "#8a6f2a",
  boss: "#111111",
  unknown: "#6b675f",
};

async function init() {
  const manifest = await fetch(manifestUrl).then((response) => response.json());
  state.manifest = manifest;
  document.getElementById("site-title").textContent = manifest.title;
  buildRunSelect();
  buildGraphSelect();
  await loadRun("primary");
  document.getElementById("run-select").addEventListener("change", async (event) => {
    await loadRun(event.target.value);
  });
  document.getElementById("graph-select").addEventListener("change", (event) => {
    state.currentGraphKey = event.target.value;
    renderGraph();
  });
  document.getElementById("node-search").addEventListener("change", focusNodeFromSearch);
}

function buildRunSelect() {
  const select = document.getElementById("run-select");
  select.innerHTML = "";
  for (const [runKey, runMeta] of Object.entries(state.manifest.runs)) {
    const option = document.createElement("option");
    option.value = runKey;
    option.textContent = runMeta.label;
    select.appendChild(option);
  }
}

function buildGraphSelect() {
  const select = document.getElementById("graph-select");
  select.innerHTML = "";
  for (const [graphKey, label] of Object.entries(GRAPH_LABELS)) {
    const option = document.createElement("option");
    option.value = graphKey;
    option.textContent = label;
    select.appendChild(option);
  }
}

async function loadRun(runKey) {
  state.currentRunKey = runKey;
  const runMeta = state.manifest.runs[runKey];
  const bundle = await fetch(runMeta.bundle).then((response) => response.json());
  state.bundle = bundle;
  renderSummary(runMeta, bundle);
  renderGraph();
  renderDataPanels(bundle);
}

function renderSummary(runMeta, bundle) {
  const summary = bundle.summary;
  document.getElementById("hero-summary").textContent =
    `${runMeta.label}: ${summary.dataset_cells.toLocaleString()} cells, ${summary.dataset_genes.toLocaleString()} genes, ${summary.graph_nodes} selected nodes, ${summary.graph_edges} selected edges.`;
}

function renderGraph() {
  const graph = state.bundle.graphs[state.currentGraphKey];
  document.getElementById("graph-title").textContent = GRAPH_LABELS[state.currentGraphKey];
  document.getElementById("graph-stats").innerHTML =
    `<span>${graph.nodes.length} nodes</span><span>${graph.edges.length} edges</span>`;

  const visNodes = new vis.DataSet(
    graph.nodes.map((node) => ({
      id: node.id,
      label: node.id,
      color: kindColors[node.kind] || kindColors.unknown,
      shape: node.kind === "boss" ? "hexagon" : "dot",
      size: node.kind === "boss" ? 22 : node.kind === "pathway" ? 18 : 14,
      font: { color: "#111111", face: "Space Grotesk", size: 14 },
      borderWidth: 1,
    })),
  );
  const visEdges = new vis.DataSet(
    graph.edges.map((edge, index) => ({
      id: `${edge.source}|${edge.target}|${edge.sign}|${index}`,
      from: edge.source,
      to: edge.target,
      color: edge.sign === -1 ? "#c0392b" : "#198754",
      arrows: "to",
      width: Math.max(1.5, (edge.confidence || edge.weight || 0.35) * 4),
      dashes: edge.direct_evidence?.length ? false : [6, 5],
      smooth: { type: "dynamic" },
    })),
  );

  state.nodes = visNodes;
  state.edges = visEdges;
  const container = document.getElementById("network");
  if (state.network) {
    state.network.destroy();
  }
  state.network = new vis.Network(
    container,
    { nodes: visNodes, edges: visEdges },
    {
      interaction: { hover: true, navigationButtons: true, keyboard: true },
      physics: { enabled: false },
      layout: { improvedLayout: true },
    },
  );
  state.network.on("click", (params) => handleGraphClick(params, graph));
  renderInspectorIntro();
}

function handleGraphClick(params, graph) {
  if (params.nodes.length) {
    const nodeId = params.nodes[0];
    renderNodeInspector(nodeId, graph);
    return;
  }
  if (params.edges.length) {
    const edgeId = params.edges[0];
    const edge = graph.edges.find((candidate, index) => `${candidate.source}|${candidate.target}|${candidate.sign}|${index}` === edgeId);
    if (edge) {
      renderEdgeInspector(edge);
    }
  }
}

function focusNodeFromSearch() {
  const query = document.getElementById("node-search").value.trim().toUpperCase();
  if (!query || !state.network || !state.nodes.get(query)) {
    return;
  }
  state.network.selectNodes([query]);
  state.network.focus(query, { scale: 1.2, animation: true });
  renderNodeInspector(query, state.bundle.graphs[state.currentGraphKey]);
}

function renderInspectorIntro() {
  document.getElementById("inspector-title").textContent = "Click a node or edge";
  document.getElementById("inspector-body").innerHTML = `
    <div class="metric-card">
      <strong>What you can inspect here</strong>
      <div class="muted">Click a node to see its DEG stats, benchmark row, incoming/outgoing regulators, and evidence-supported edges. Click an edge to see its sign, provenance, collapsed path, PMIDs, and model/database support.</div>
    </div>`;
}

function renderNodeInspector(nodeId, graph) {
  const profile = state.bundle.node_profiles[nodeId] || {};
  const nodeMeta = graph.nodes.find((node) => node.id === nodeId) || { kind: "unknown" };
  const incoming = graph.edges.filter((edge) => edge.target === nodeId);
  const outgoing = graph.edges.filter((edge) => edge.source === nodeId);
  document.getElementById("inspector-title").textContent = nodeId;

  const degStats = profile.deg_stats
    ? `<div class="metric-card"><strong>DEG stats</strong><div>log2FC: ${fmt(profile.deg_stats.log2_fold_change)}</div><div>adj p: ${fmt(profile.deg_stats.adjusted_pvalue)}</div></div>`
    : "";
  const benchmark = profile.benchmark
    ? `<div class="metric-card"><strong>DepMap benchmark</strong><div>mean effect: ${fmt(profile.benchmark.mean_gene_effect)}</div><div>hit rate: ${fmt(profile.benchmark.hit_rate)}</div><div>benchmark hit: ${boolLabel(profile.benchmark.benchmark_hit)}</div></div>`
    : "";

  document.getElementById("inspector-body").innerHTML = `
    <div class="summary-cards">
      <div class="metric-card"><strong>Node type</strong><div>${nodeMeta.kind}</div></div>
      <div class="metric-card"><strong>Graph degree</strong><div>${incoming.length} incoming / ${outgoing.length} outgoing</div></div>
      <div class="metric-card"><strong>Present in</strong><div>${Object.keys(profile.graph_presence || {}).join(", ") || "current graph only"}</div></div>
      ${degStats}
      ${benchmark}
    </div>
    <div>
      <h3>Incoming influences</h3>
      <div class="edge-list">${incoming.length ? incoming.map(renderEdgeCard).join("") : '<div class="muted">No incoming edges in this graph view.</div>'}</div>
    </div>
    <div>
      <h3>Outgoing influences</h3>
      <div class="edge-list">${outgoing.length ? outgoing.map(renderEdgeCard).join("") : '<div class="muted">No outgoing edges in this graph view.</div>'}</div>
    </div>
  `;
}

function renderEdgeInspector(edge) {
  document.getElementById("inspector-title").textContent = `${edge.source} ${edge.sign === -1 ? "−|" : "→"} ${edge.target}`;
  document.getElementById("inspector-body").innerHTML = renderEdgeCard(edge);
}

function renderEdgeCard(edge) {
  const evidence = edge.direct_evidence || [];
  const evidenceHtml = evidence.length
    ? evidence.map((item) => `
        <div class="evidence-card">
          <strong>${item.source_gene} ${item.interaction_type === -1 ? "−|" : "→"} ${item.target}</strong>
          <div class="muted">${escapeHtml(item.evidence_summary || "No summary")}</div>
          <div class="tag-row">
            <span class="tag">conf ${fmt(item.confidence_score)}</span>
            <span class="tag">depth ${item.mechanistic_depth}</span>
            ${(item.provenance_sources || []).map((source) => `<span class="tag">${escapeHtml(source)}</span>`).join("")}
          </div>
          <div class="mono">${renderRefs(item.source_refs || [], item.pmid_citations || [])}</div>
        </div>`).join("")
    : `<div class="muted">No direct LLM evidence stored for this edge. This usually means it came from curated priors or a collapsed projected path.</div>`;
  const collapsed = edge.collapsed_via?.length
    ? `<div class="tag-row"><span class="tag">collapsed via ${escapeHtml(edge.collapsed_via.join(" → "))}</span></div>`
    : "";
  return `
    <div class="edge-card">
      <strong>${edge.source} ${edge.sign === -1 ? "−|" : "→"} ${edge.target}</strong>
      <div>confidence: ${fmt(edge.confidence || edge.weight)}</div>
      <div>benchmark support: ${fmt(edge.benchmark_support_score)}</div>
      <div>provenance: ${(edge.provenance || []).join(", ") || "none"}</div>
      ${collapsed}
      <div class="evidence-list">${evidenceHtml}</div>
    </div>`;
}

function renderDataPanels(bundle) {
  const summaryCards = [
    ["Selected experiment", bundle.summary.selected_experiment],
    ["Top hit", (bundle.summary.top_hit || []).join(" + ") || "none"],
    ["Knockout hits", String(bundle.knockout_hits.length)],
    ["Benchmark models", String(bundle.summary.benchmark_model_count)],
    ["Research backends", Object.entries(bundle.research_execution.result_model_counts || {}).map(([key, value]) => `${key}=${value}`).join(", ")],
    ["Prior knowledge", `${bundle.prior_knowledge.node_count} nodes / ${bundle.prior_knowledge.edge_count} edges`],
  ];
  document.getElementById("summary-cards").innerHTML = summaryCards.map(([label, value]) => `
    <div class="metric-card">
      <strong>${escapeHtml(label)}</strong>
      <div>${escapeHtml(value)}</div>
    </div>`).join("");

  document.getElementById("knockout-list").innerHTML = bundle.knockout_hits.map((hit) => `
    <div class="hit-card">
      <strong>${(hit.knocked_out_genes || []).join(" + ")}</strong>
      <div>boss state: ${hit.boss_state}</div>
      <div>score: ${fmt(hit.score)}</div>
      <div>pathway nodes off: ${(hit.pathway_nodes_off || []).join(", ")}</div>
    </div>`).join("") || '<div class="muted">No knockout hits found.</div>';

  document.getElementById("benchmark-list").innerHTML = (bundle.benchmark_report.results || []).slice(0, 12).map((row) => `
    <div class="bench-card">
      <strong>${row.gene_symbol}</strong>
      <div>mean effect: ${fmt(row.mean_gene_effect)}</div>
      <div>hit rate: ${fmt(row.hit_rate)}</div>
      <div>benchmark hit: ${boolLabel(row.benchmark_hit)}</div>
    </div>`).join("") || '<div class="muted">No benchmark rows.</div>';

  document.getElementById("deg-list").innerHTML = (bundle.top_degs || []).slice(0, 24).map((row) => `
    <div class="deg-chip">
      <strong>${row.gene}</strong>
      <div>log2FC ${fmt(row.log2_fold_change)}</div>
      <div class="muted">adj p ${fmt(row.adjusted_pvalue)}</div>
    </div>`).join("");
}

function renderRefs(sourceRefs, pmids) {
  const refs = [];
  for (const ref of sourceRefs) {
    if (ref.startsWith("http")) {
      refs.push(`<a href="${ref}" target="_blank" rel="noreferrer">${escapeHtml(ref)}</a>`);
    } else {
      refs.push(escapeHtml(ref));
    }
  }
  for (const pmid of pmids) {
    const label = `PMID:${pmid}`;
    const url = `https://pubmed.ncbi.nlm.nih.gov/${pmid}/`;
    refs.push(`<a href="${url}" target="_blank" rel="noreferrer">${label}</a>`);
  }
  return refs.join(" · ");
}

function fmt(value) {
  if (value === null || value === undefined || value === "") {
    return "n/a";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(3);
  }
  return String(value);
}

function boolLabel(value) {
  return value ? "yes" : "no";
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
  document.getElementById("hero-summary").textContent = `Failed to load site: ${error}`;
});
"""
