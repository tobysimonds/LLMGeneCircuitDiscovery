#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_RUN = ROOT / "artifacts" / "pdac-expanded-pubmed-live-20260321-v3"
BASELINE_RUN = ROOT / "artifacts" / "pdac-live-run-v6"
OUTPUT_PATH = ROOT / "artifacts" / "pdac_target_discovery_status_report_20260321.pdf"

TITLE = "Target-GPT Explorer"
SUBTITLE = "PDAC DEG-to-Network Target Discovery Status Update"
ACCENT = "#1f6f61"
ACCENT_2 = "#c26a2d"
INK = "#17202a"
MUTED = "#56606a"
LIGHT = "#f5efe6"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def figure_page():
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    return fig, ax


def add_cover_page(pdf: PdfPages, primary_summary: dict, baseline_summary: dict) -> None:
    fig, ax = figure_page()
    ax.add_patch(FancyBboxPatch((0.05, 0.06), 0.9, 0.88, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor=LIGHT, edgecolor="none"))
    fig.text(0.09, 0.89, TITLE, fontsize=26, fontweight="bold", color=ACCENT)
    fig.text(0.09, 0.85, SUBTITLE, fontsize=18, color=INK)
    fig.text(0.09, 0.80, "Prepared as a researcher-facing implementation and status report.", fontsize=11, color=MUTED)

    cards = [
        ("Dataset", "GSE242230 PDAC", "Malignant vs Normal Epithelial"),
        ("Primary run", f"{primary_summary['graph_nodes']} nodes / {primary_summary['graph_edges']} edges", f"Selected: {primary_summary['selected_experiment']}"),
        ("Baseline run", f"{baseline_summary['graph_nodes']} nodes / {baseline_summary['graph_edges']} edges", "Reference comparison"),
        ("Current outcome", "Mechanistic graph built", "Target validation still weak"),
    ]
    x_positions = [0.09, 0.53]
    y_positions = [0.62, 0.44]
    for index, (label, headline, detail) in enumerate(cards):
        x = x_positions[index % 2]
        y = y_positions[index // 2]
        ax.add_patch(FancyBboxPatch((x, y), 0.34, 0.12, boxstyle="round,pad=0.015,rounding_size=0.02", facecolor="white", edgecolor="#e0d6c7"))
        fig.text(x + 0.02, y + 0.085, label, fontsize=10, color=MUTED)
        fig.text(x + 0.02, y + 0.05, headline, fontsize=15, fontweight="bold", color=INK)
        fig.text(x + 0.02, y + 0.022, detail, fontsize=10.5, color=MUTED)

    fig.text(0.09, 0.28, "Headline status", fontsize=14, fontweight="bold", color=INK)
    summary_text = (
        "The original concept was successfully expanded into a working three-stage pipeline with richer graph construction, "
        "curated pathway priors, weighted Boolean simulation, and DepMap benchmarking. The system runs end to end on PDAC data, "
        "but the latest validated result is still exploratory because the best nominated target did not receive strong external support."
    )
    for idx, line in enumerate(textwrap.wrap(summary_text, width=92)):
        fig.text(0.09, 0.24 - idx * 0.022, line, fontsize=11.5, color=INK)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_bullets_page(pdf: PdfPages, title: str, bullets: list[str], intro: str | None = None) -> None:
    fig, ax = figure_page()
    fig.text(0.08, 0.95, title, fontsize=22, fontweight="bold", color=INK, va="top")
    y = 0.90
    if intro:
        for line in textwrap.wrap(intro, width=95):
            fig.text(0.08, y, line, fontsize=11.5, color=MUTED, va="top")
            y -= 0.023
        y -= 0.01
    for bullet in bullets:
        wrapped = textwrap.wrap(bullet, width=92) or [""]
        fig.text(0.09, y, u"\u2022", fontsize=15, color=ACCENT, va="top")
        for idx, line in enumerate(wrapped):
            fig.text(0.11, y - idx * 0.023, line, fontsize=11.2, color=INK, va="top")
        y -= 0.03 * max(1, len(wrapped)) + 0.01
        if y < 0.09:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig, ax = figure_page()
            fig.text(0.08, 0.95, f"{title} (cont.)", fontsize=20, fontweight="bold", color=INK, va="top")
            y = 0.90
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_architecture_page(pdf: PdfPages) -> None:
    fig, ax = figure_page()
    fig.text(0.08, 0.95, "Implemented Architecture", fontsize=22, fontweight="bold", color=INK, va="top")
    fig.text(0.08, 0.91, "What was added beyond the original concept note.", fontsize=11.5, color=MUTED, va="top")

    boxes = [
        (0.08, 0.72, 0.24, 0.11, "1. Data distillation", "Scanpy DEG extraction\nQC, normalization, malignant vs normal contrast"),
        (0.38, 0.72, 0.24, 0.11, "2. Discovery + verification", "Two-pass graph building\nPrompted research, alias resolution, evidence scoring"),
        (0.68, 0.72, 0.24, 0.11, "3. Prior knowledge seeding", "KEGG, Reactome, OmniPath,\nPathway Commons"),
        (0.20, 0.48, 0.24, 0.11, "4. Weighted simulation", "Boolean knockout search\nweighted edges, inhibition dominance"),
        (0.56, 0.48, 0.24, 0.11, "5. Benchmarking + pruning", "DepMap CRISPR + RNAi\npre-simulation support scoring"),
    ]
    for x, y, w, h, title, body in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", facecolor="white", edgecolor="#d7d0c3"))
        fig.text(x + 0.015, y + h - 0.028, title, fontsize=12.5, fontweight="bold", color=ACCENT)
        for idx, line in enumerate(body.split("\n")):
            fig.text(x + 0.015, y + h - 0.058 - idx * 0.023, line, fontsize=10.5, color=INK)

    arrows = [
        ((0.32, 0.775), (0.38, 0.775)),
        ((0.62, 0.775), (0.68, 0.775)),
        ((0.50, 0.71), (0.32, 0.59)),
        ((0.70, 0.71), (0.68, 0.59)),
        ((0.44, 0.535), (0.56, 0.535)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, linewidth=1.5, color=ACCENT_2))

    notes = [
        "Expanded graph schema: DEG->DEG, DEG->intermediate, and larger pathway neighborhoods instead of only a 50-gene star graph.",
        "Two-pass extraction: broad discovery followed by stricter verification.",
        "Evidence classes tracked separately: direct mechanistic, PDAC-specific, pancreas-relevant, review-supported, prior-supported, benchmark-supported.",
        "Alias normalization and gene disambiguation added before graph construction.",
        "Experiment ablations added to compare verified-only, priors-added, and priors+pruning variants.",
    ]
    y = 0.28
    for note in notes:
        for idx, line in enumerate(textwrap.wrap(note, width=95)):
            fig.text(0.08, y - idx * 0.022, ("\u2022 " if idx == 0 else "  ") + line, fontsize=11, color=INK)
        y -= 0.055

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_status_table_page(pdf: PdfPages, primary_summary: dict, baseline_summary: dict, primary_research: dict, primary_experiments: list[dict]) -> None:
    fig, ax = figure_page()
    fig.text(0.08, 0.95, "Execution Status", fontsize=22, fontweight="bold", color=INK, va="top")

    rows = [
        ("Dataset used", "GSE242230 PDAC"),
        ("Cells after QC", str(primary_summary["dataset_cells"])),
        ("Top DEGs", str(len(primary_summary["degs"]))),
        ("Research backend actually used", ", ".join(f"{k}={v}" for k, v in primary_research["result_model_counts"].items())),
        ("Fallback count", str(primary_research["fallback_gene_count"])),
        ("Baseline top hit", " + ".join(baseline_summary["knockout_hits"][0]["knocked_out_genes"])),
        ("Current top hit", " + ".join(primary_summary["knockout_hits"][0]["knocked_out_genes"]) if primary_summary["knockout_hits"] else "none"),
        ("Selected experiment", primary_summary["selected_experiment"]),
        ("Benchmark models", str(primary_summary["benchmark_report"]["model_count"])),
    ]
    y = 0.86
    for key, value in rows:
        ax.add_patch(FancyBboxPatch((0.08, y - 0.037), 0.84, 0.045, boxstyle="round,pad=0.008,rounding_size=0.01", facecolor="#fbf8f3", edgecolor="#ece3d8"))
        fig.text(0.10, y - 0.012, key, fontsize=11, color=MUTED)
        fig.text(0.45, y - 0.012, value, fontsize=11.5, color=INK)
        y -= 0.056

    fig.text(0.08, 0.30, "Experiment outcomes", fontsize=14, fontweight="bold", color=INK)
    y = 0.26
    for experiment in primary_experiments:
        lead = experiment["knockout_hits"][0]["knocked_out_genes"] if experiment["knockout_hits"] else []
        line = (
            f"{experiment['name']}: graph {experiment['graph_nodes']}/{experiment['graph_edges']}, "
            f"hits={len(experiment['knockout_hits'])}, lead={' + '.join(lead) if lead else 'none'}, "
            f"pruned={len(experiment.get('pruned_genes', []))}"
        )
        for idx, chunk in enumerate(textwrap.wrap(line, width=94)):
            fig.text(0.09, y - idx * 0.022, ("\u2022 " if idx == 0 else "  ") + chunk, fontsize=10.8, color=INK)
        y -= 0.05

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_results_page(pdf: PdfPages, verified_edges: list[dict], benchmark_results: list[dict]) -> None:
    bullets = [
        f"Verified research edges retained in the latest corrected run: {len(verified_edges)}.",
    ]
    for edge in verified_edges:
        sign = "->" if edge["interaction_type"] == 1 else "-|"
        bullets.append(
            f"{edge['source_gene']} {sign} {edge['target']} | confidence {edge.get('confidence_score', 0):.2f} | evidence: {edge.get('evidence_summary', '') or 'n/a'}"
        )
    bullets.extend(
        [
            "Most of the graph expansion came from curated pathway priors rather than a high-yield live LLM literature extraction.",
            "After correcting the pruning logic, the pruned variant no longer supported GNGT1 as a viable candidate, which was an important sanity correction.",
            "The selected unpruned variant still nominates GNGT1, but the benchmark evidence remains weak.",
            "At present, the pipeline is mechanistically richer and technically stronger, but not yet producing a validated PDAC target.",
            "",
            "Top benchmarked genes in the latest selected run:",
        ]
    )
    for result in benchmark_results[:8]:
        bullets.append(
            f"{result['gene_symbol']}: mean gene effect = {result.get('mean_gene_effect')}, hit rate = {result.get('hit_rate')}, benchmark hit = {result.get('benchmark_hit')}"
        )
    add_bullets_page(pdf, "Results So Far", bullets)


def add_blockers_page(pdf: PdfPages) -> None:
    bullets = [
        "OpenAI path was implemented and hardened, but live use of o4-mini-deep-research was blocked by model access / organization verification on the tested account.",
        "Additional OpenAI key checks later failed with account_deactivated or insufficient_quota, so provider-backed OpenAI runs could not be fully benchmarked end to end.",
        "Anthropic support was also implemented, but tested keys failed with invalid-key or low-credit / billing restrictions.",
        "Because of those provider constraints, the latest fully verified run used the PubMed heuristic fallback for all 50 genes.",
        "That means the scientific bottleneck is still the literature-research stage, not the downstream simulation or benchmarking infrastructure.",
    ]
    add_bullets_page(pdf, "Current Blockers", bullets, intro="The codebase is ahead of the provider access currently available in this environment.")


def add_next_steps_page(pdf: PdfPages) -> None:
    bullets = [
        "Run the same pipeline with a live provider-backed research model once valid OpenAI or Anthropic access is available.",
        "Log raw prompts and raw model responses per gene for future auditability and prompt iteration.",
        "Potentially restrict the interactive website graph to a smaller LLM-derived neighborhood by default, while keeping the full graph available on demand.",
        "Use the existing benchmarking loop to compare provider-backed research outputs against the current fallback baseline.",
        "If live LLM extraction remains sparse, widen the discovery strategy further toward receptor/adaptor/TF bridge discovery before final verification.",
    ]
    add_bullets_page(pdf, "Recommended Next Steps", bullets)


def add_bar_page(pdf: PdfPages, title: str, labels: list[str], values: list[float], ylabel: str, color: str) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax.barh(labels, values, color=color)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=16)
    ax.set_xlabel(ylabel)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def add_appendix_page(pdf: PdfPages) -> None:
    bullets = [
        f"Primary corrected run artifacts: {PRIMARY_RUN}",
        f"Baseline run artifacts: {BASELINE_RUN}",
        "Prompt templates: src/llmgenecircuitdiscovery/grn.py",
        "Pipeline orchestration: src/llmgenecircuitdiscovery/pipeline.py",
        "Boolean simulation: src/llmgenecircuitdiscovery/boolean_network.py",
        "DepMap benchmarking: src/llmgenecircuitdiscovery/depmap.py",
        "Results website: src/llmgenecircuitdiscovery/site.py",
    ]
    add_bullets_page(pdf, "Appendix: File Locations", bullets)


def main() -> None:
    primary_summary = load_json(PRIMARY_RUN / "summary.json")
    primary_experiments = load_json(PRIMARY_RUN / "experiment_report.json")
    primary_research = load_json(PRIMARY_RUN / "research_execution.json")
    primary_interactions = load_json(PRIMARY_RUN / "gene_interactions.json")
    baseline_summary = load_json(BASELINE_RUN / "summary.json")

    verified_edges = [edge for result in primary_interactions for edge in result.get("interactions", [])]
    degs = primary_summary["degs"][:12]
    benchmark_results = sorted(
        primary_summary["benchmark_report"]["results"],
        key=lambda item: (
            item.get("mean_gene_effect") is None,
            item.get("mean_gene_effect") if item.get("mean_gene_effect") is not None else math.inf,
        ),
    )

    original_concept_bullets = [
        "The original idea was to distill scRNA-seq features down to disease-specific DEGs, validate them through literature, and use virtual knockouts to identify intervention-ready targets.",
        "That concept has now been translated into a runnable pipeline with modular outputs, integration tests, and reproducible artifact directories.",
        "The implementation expanded the original proposal in several important ways: richer graph schema, curated pathway priors, weighted logic, early benchmarking, and experiment ablations.",
    ]

    added_techniques_bullets = [
        "Expanded graph schema beyond the original 50-gene framing: the graph now supports DEG->DEG edges, DEG->intermediate edges, and additional pathway bridge nodes.",
        "Two-pass research design: discovery for broader recall, then verification for precision.",
        "Evidence-class scoring on each edge: direct mechanistic, PDAC-specific, pancreas-relevant, review-supported, prior-supported, and benchmark-supported.",
        "Alias normalization and gene disambiguation layer to reduce symbol mismatch failures.",
        "Curated prior seeding from KEGG, Reactome, OmniPath, and Pathway Commons.",
        "Weighted Boolean logic with inhibition dominance instead of a simple unweighted ON/OFF rule.",
        "Pre-simulation DepMap / RNAi benchmarking and pruning so low-support genes can be removed before knockout search.",
        "Run-level experiment comparison so verified-only, priors-added, and priors+pruning variants can be compared directly.",
        "Lightweight website generation for result exploration.",
    ]

    with PdfPages(OUTPUT_PATH) as pdf:
        add_cover_page(pdf, primary_summary, baseline_summary)
        add_bullets_page(pdf, "Original Report to Current System", original_concept_bullets)
        add_architecture_page(pdf)
        add_bullets_page(pdf, "Additional Techniques Added", added_techniques_bullets)
        add_status_table_page(pdf, primary_summary, baseline_summary, primary_research, primary_experiments)
        add_results_page(pdf, verified_edges, benchmark_results)
        add_bar_page(
            pdf,
            "Top 12 DEGs by Log2 Fold Change",
            [item["gene"] for item in reversed(degs)],
            [item["log2_fold_change"] for item in reversed(degs)],
            "Log2 fold change",
            ACCENT,
        )
        add_bar_page(
            pdf,
            "DepMap Mean Gene Effect for Benchmarked Genes",
            [item["gene_symbol"] for item in reversed(benchmark_results[:10])],
            [item["mean_gene_effect"] if item["mean_gene_effect"] is not None else 0.0 for item in reversed(benchmark_results[:10])],
            "Mean gene effect",
            ACCENT_2,
        )
        add_blockers_page(pdf)
        add_next_steps_page(pdf)
        add_appendix_page(pdf)

    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
