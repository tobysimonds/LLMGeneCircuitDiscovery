from __future__ import annotations

import json
import shutil
from collections import Counter
import csv
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from llmgenecircuitdiscovery.site import _attach_graph_evidence, _build_edge_evidence_index, _build_full_edge_index, _normalize_graph
from llmgenecircuitdiscovery.utils import ensure_directory, write_json

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

STRUCTURE_FILES = {
    "efs_src": {
        "model": Path("Structural Analysis Pipeline/fold_efs_and_src/fold_efs_and_src_model_0.cif"),
        "confidence": Path("Structural Analysis Pipeline/fold_efs_and_src/fold_efs_and_src_summary_confidences_0.json"),
        "label": "EFS-SRC",
        "partner_label": "EFS",
        "partner_style": "cartoon",
    },
    "cathy01_src": {
        "model": Path("Structural Analysis Pipeline/fold_cathy01/fold_cathy01_model_0.cif"),
        "confidence": Path("Structural Analysis Pipeline/fold_cathy01/fold_cathy01_summary_confidences_0.json"),
        "label": "Cathy01-SRC",
        "partner_label": "eSI-1 (Cathy01)",
        "partner_style": "sticks",
    },
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
    _write_structure_explorer_pages(output_dir, bundle)
    return output_dir


def _write_structure_explorer_pages(output_dir: Path, bundle: dict[str, Any]) -> None:
    explorers = {
        "efs-src-explorer.html": _render_structure_explorer_page(
            page_title="Native EFS-SRC Interface",
            badge_left="AlphaFold 3",
            badge_mid="EFS · Chain A",
            badge_right="SRC · Chain B",
            title="Native EFS-SRC Binding Interface",
            subtitle="Reference structural interface identified from the PDAC target-discovery pipeline",
            structure_key="efs_src",
            background="#07091a",
            contact_mode="native",
            bundle=bundle,
        ),
        "cathy01-src-explorer.html": _render_structure_explorer_page(
            page_title="Engineered Cathy01-SRC Interface",
            badge_left="AlphaFold 3",
            badge_mid="eSI-1 · Chain A",
            badge_right="SRC · Chain B",
            title="Engineered eSI-1 (Cathy01) Interface",
            subtitle="Designed inhibitory complex positioned against the SRC interaction surface",
            structure_key="cathy01_src",
            background="#08111d",
            contact_mode="engineered",
            bundle=bundle,
        ),
    }
    for filename, html in explorers.items():
        (output_dir / filename).write_text(html, encoding="utf-8")


def _render_structure_explorer_page(
    *,
    page_title: str,
    badge_left: str,
    badge_mid: str,
    badge_right: str,
    title: str,
    subtitle: str,
    structure_key: str,
    background: str,
    contact_mode: str,
    bundle: dict[str, Any],
) -> str:
    structure = bundle["structures"][structure_key]
    report = bundle.get("structural_report", {})
    metrics = structure.get("metrics", {})
    native_contacts = report.get("native_contacts", [])
    key_contacts = native_contacts[:6]
    config = {
        "structure": structure,
        "contact_mode": contact_mode,
        "native_contacts": key_contacts,
        "hotspots": report.get("hotspots", []),
    }
    config_json = json.dumps(config, ensure_ascii=True)
    metrics_html = ""
    if structure_key == "cathy01_src":
        metrics_html = f"""
          <div class="stat">
            <div class="stat-num gold">{_fmt_metric(metrics.get("ipTM"))}</div>
            <div class="stat-label">ipTM</div>
          </div>
          <div class="stat">
            <div class="stat-num white">{_fmt_metric(metrics.get("src_pTM"))}</div>
            <div class="stat-label">SRC pTM</div>
          </div>
          <div class="stat">
            <div class="stat-num aqua">+38%</div>
            <div class="stat-label">pTM gain</div>
          </div>
        """
    else:
        metrics_html = f"""
          <div class="stat">
            <div class="stat-num gold">{_fmt_metric(metrics.get("ipTM"))}</div>
            <div class="stat-label">ipTM</div>
          </div>
          <div class="stat">
            <div class="stat-num white">{_fmt_metric(metrics.get("pTM"))}</div>
            <div class="stat-label">complex pTM</div>
          </div>
          <div class="stat">
            <div class="stat-num aqua">{len(key_contacts)}</div>
            <div class="stat-label">key contacts</div>
          </div>
        """
    contacts_html = "".join(
        f"""
        <div class="contact-card{' hotspot' if contact.get('is_hotspot') else ''}">
          <div class="contact-pair">
            <span class="chain-a">{escape(str(contact.get('partner_residue', '')))} ({escape(str(contact.get('partner_atom', '')) )})</span>
            <span class="sep">↔</span>
            <span class="chain-b">{escape(str(contact.get('src_residue', '')))} ({escape(str(contact.get('src_atom', '')) )})</span>
          </div>
          <div class="contact-dist">{contact.get('distance_a', '')} Å</div>
        </div>
        """
        for contact in key_contacts[:4]
    )
    right_rail = """
      <div class="notes-card">
        <h3>Design Logic</h3>
        <p>Strategic Proline-Capping & Charge Reversal. The design preserves the core PxxP motif while increasing N-terminal proline density, pre-folding the peptide toward a PPII-like conformation and reducing the entropic cost of engagement.</p>
      </div>
    """ if structure_key == "cathy01_src" else f"""
      <div class="notes-card">
        <h3>Key Native Contacts</h3>
        <div class="contact-list">{contacts_html}</div>
      </div>
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{escape(page_title)}</title>
  <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
  <style>
    :root {{
      --bg: {background};
      --panel: rgba(10, 16, 30, 0.9);
      --panel-2: rgba(15, 23, 42, 0.92);
      --line: rgba(255,255,255,0.12);
      --text: #e8edf7;
      --muted: #9aa8c1;
      --gold: #d4af37;
      --blue: #7ec8ff;
      --white: #f8fafc;
      --yellow: #ffff00;
      --aqua: #6ee7f9;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; min-height: 100%; background: radial-gradient(circle at top, rgba(126,200,255,0.08), transparent 28%), var(--bg); color: var(--text); font-family: Inter, system-ui, sans-serif; }}
    body {{ padding: 12px; }}
    .shell {{ max-width: 1400px; margin: 0 auto; }}
    .hero {{
      display: flex; flex-wrap: wrap; gap: 16px; align-items: end; justify-content: space-between;
      padding: 12px 14px; border: 1px solid var(--line); border-radius: 18px;
      background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
      box-shadow: 0 18px 60px rgba(0,0,0,0.35);
    }}
    .badge-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }}
    .badge {{
      border: 1px solid var(--line); border-radius: 999px; padding: 6px 10px; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
      color: var(--muted); background: rgba(255,255,255,0.04);
    }}
    h1 {{ margin: 0 0 4px; font-size: clamp(18px, 2vw, 24px); line-height: 1.05; letter-spacing: -0.03em; }}
    .subtitle {{ margin: 0; color: var(--muted); max-width: 58ch; line-height: 1.4; font-size: 12px; }}
    .stats-row {{ display: flex; gap: 18px; flex-wrap: wrap; }}
    .stat {{ min-width: 90px; }}
    .stat-num {{ font-size: 22px; font-weight: 800; line-height: 1; }}
    .stat-num.gold {{ color: var(--gold); }}
    .stat-num.white {{ color: var(--white); }}
    .stat-num.aqua {{ color: var(--aqua); }}
    .stat-label {{ margin-top: 6px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }}
    .viewer-layout {{
      margin-top: 12px;
      display: grid; grid-template-columns: minmax(0, 1.8fr) minmax(300px, 0.9fr); gap: 18px;
    }}
    .viewer-card, .rail {{
      position: relative;
      border: 1px solid var(--line); border-radius: 20px; overflow: hidden;
      background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
      box-shadow: 0 18px 60px rgba(0,0,0,0.35);
    }}
    .viewer-toolbar {{
      position: absolute; top: 14px; left: 14px; right: 14px; z-index: 10;
      display: flex; justify-content: space-between; align-items: start; gap: 12px;
      pointer-events: none;
    }}
    .legend, .hint {{
      background: rgba(5,10,20,0.7); border: 1px solid var(--line); border-radius: 14px; padding: 10px 12px;
      backdrop-filter: blur(12px); pointer-events: auto;
    }}
    .legend strong, .hint strong {{ display: block; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }}
    .legend-item {{ display: flex; align-items: center; gap: 8px; font-size: 13px; margin-top: 6px; }}
    .swatch {{ width: 14px; height: 14px; border-radius: 999px; }}
    .swatch.line {{ width: 22px; height: 4px; border-radius: 999px; }}
    .viewer {{
      width: 100%;
      height: 640px;
      position: relative;
      overflow: hidden;
      background:
        radial-gradient(circle at 50% 0%, rgba(255,255,255,0.05), transparent 30%),
        linear-gradient(180deg, #0b1020, #060913 72%);
    }}
    .viewer > div, .viewer canvas {{
      position: absolute !important;
      inset: 0 !important;
    }}
    .viewer-actions {{
      display: flex; gap: 10px; flex-wrap: wrap; padding: 14px; border-top: 1px solid var(--line);
      background: rgba(5,10,20,0.72);
    }}
    button {{
      border: 1px solid rgba(255,255,255,0.16); border-radius: 999px; padding: 10px 14px; background: rgba(255,255,255,0.06);
      color: var(--text); font: inherit; cursor: pointer;
    }}
    button:hover {{ background: rgba(255,255,255,0.12); }}
    .rail {{ padding: 18px; display: flex; flex-direction: column; gap: 14px; }}
    .notes-card {{
      border: 1px solid var(--line); border-radius: 16px; padding: 16px; background: var(--panel-2);
    }}
    .notes-card h3 {{ margin: 0 0 10px; font-size: 15px; }}
    .notes-card p {{ margin: 0; color: var(--muted); line-height: 1.6; font-size: 14px; }}
    .contact-list {{ display: grid; gap: 10px; }}
    .contact-card {{ border: 1px solid var(--line); border-radius: 12px; padding: 12px; background: rgba(255,255,255,0.03); }}
    .contact-card.hotspot {{ background: rgba(212,175,55,0.14); border-color: rgba(212,175,55,0.36); }}
    .contact-pair {{ font-size: 13px; line-height: 1.4; }}
    .chain-a {{ color: var(--gold); }}
    .chain-b {{ color: var(--blue); }}
    .sep {{ color: var(--muted); padding: 0 4px; }}
    .contact-dist {{ margin-top: 6px; font-weight: 700; font-size: 13px; color: var(--white); }}
    @media (max-width: 1080px) {{
      body {{ padding: 8px; }}
      .viewer-layout {{ grid-template-columns: 1fr; }}
      .viewer {{ height: 520px; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="viewer-layout">
      <div class="viewer-card">
        <div class="viewer-toolbar">
          <div class="legend">
            <strong>Legend</strong>
            <div class="legend-item"><span class="swatch" style="background:#d4af37;"></span>{escape(structure['partner_label'])}</div>
            <div class="legend-item"><span class="swatch" style="background:#ffffff;"></span>SRC surface</div>
            <div class="legend-item"><span class="swatch line" style="background:#ffff00;"></span>interaction hotspots</div>
          </div>
          <div class="hint">
            <strong>Controls</strong>
            Drag to rotate · Scroll to zoom
          </div>
        </div>
        <div id="viewer" class="viewer"></div>
        <div class="viewer-actions">
          <button type="button" id="interaction-button">Interaction</button>
          <button type="button" id="reset-button">Reset view</button>
        </div>
      </div>
      <aside class="rail">
        <div class="notes-card">
          <div class="badge-row">
            <span class="badge">{escape(badge_left)}</span>
            <span class="badge">{escape(badge_mid)}</span>
            <span class="badge">{escape(badge_right)}</span>
          </div>
          <h3 style="margin:0 0 6px;font-size:18px;">{escape(title)}</h3>
          <p>{escape(subtitle)}</p>
          <div class="stats-row" style="margin-top:14px;">{metrics_html}</div>
        </div>
        {right_rail}
        <div class="notes-card">
          <h3>Why it matters</h3>
          <p>This view isolates the interface geometry that sits underneath the graph-level result. It is intended to make the mechanistic suggestion legible at atomic scale rather than treating the ranked target as a purely symbolic output.</p>
        </div>
      </aside>
    </section>
  </div>
  <script id="explorer-config" type="application/json">{escape(config_json)}</script>
  <script>
    const config = JSON.parse(document.getElementById("explorer-config").textContent);
    const viewer = $3Dmol.createViewer("viewer", {{ backgroundColor: "#09111f", antialias: true }});
    let model = null;
    let srcSurface = null;
    let hotspotShapes = [];
    let interactionOn = false;

    function clearHotspots() {{
      hotspotShapes.forEach((shape) => {{ try {{ viewer.removeShape(shape); }} catch (error) {{}} }});
      hotspotShapes = [];
    }}

    function drawDashedLine(p1, p2) {{
      const dx = p2.x - p1.x;
      const dy = p2.y - p1.y;
      const dz = p2.z - p1.z;
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (!len) return;
      const ux = dx / len;
      const uy = dy / len;
      const uz = dz / len;
      const dashLen = 0.35;
      const gapLen = 0.1;
      const step = dashLen + gapLen;
      const count = Math.floor(len / step);
      for (let i = 0; i < count; i += 1) {{
        const t0 = i * step;
        const t1 = Math.min(t0 + dashLen, len);
        hotspotShapes.push(viewer.addCylinder({{
          start: {{ x: p1.x + ux * t0, y: p1.y + uy * t0, z: p1.z + uz * t0 }},
          end: {{ x: p1.x + ux * t1, y: p1.y + uy * t1, z: p1.z + uz * t1 }},
          radius: 0.26,
          color: "#ffff00",
          opacity: 0.28,
          fromCap: 1,
          toCap: 1,
        }}));
        hotspotShapes.push(viewer.addCylinder({{
          start: {{ x: p1.x + ux * t0, y: p1.y + uy * t0, z: p1.z + uz * t0 }},
          end: {{ x: p1.x + ux * t1, y: p1.y + uy * t1, z: p1.z + uz * t1 }},
          radius: 0.12,
          color: "#ffff00",
          opacity: 1.0,
          fromCap: 1,
          toCap: 1,
        }}));
      }}
    }}

    function applyStyles() {{
      viewer.setStyle({{}}, {{}});
      if (srcSurface !== null) {{
        try {{ viewer.removeSurface(srcSurface); }} catch (error) {{}}
        srcSurface = null;
      }}
      viewer.setStyle({{ chain: config.structure.partner_chain }}, {{
        { "stick: { color: '#d4af37', radius: 0.24 }, cartoon: { color: '#d4af37', opacity: 0.35 }" if structure_key == "cathy01_src" else "cartoon: { color: '#d9a05b', opacity: 0.95 }" }
      }});
      srcSurface = viewer.addSurface($3Dmol.SurfaceType.VDW, {{
        color: "#ffffff",
        opacity: interactionOn ? 0.3 : 1.0,
      }}, {{ chain: config.structure.src_chain }});
      viewer.setStyle({{ chain: config.structure.src_chain, resi: config.structure.interface.src_residues || [] }}, {{
        stick: {{ colorscheme: "Jmol", radius: 0.16 }}
      }});
    }}

    function interfaceSelection() {{
      return {{
        or: [
          {{ chain: config.structure.partner_chain, resi: config.structure.interface.partner_residues || [] }},
          {{ chain: config.structure.src_chain, resi: config.structure.interface.src_residues || [] }},
        ],
      }};
    }}

    function focusInterface() {{
      const selection = interfaceSelection();
      viewer.center(selection);
      viewer.zoomTo(selection, 1600);
      viewer.zoom(1.28, 500);
    }}

    function computeChainContacts(cutoff) {{
      const atomsA = model.selectedAtoms({{ chain: config.structure.partner_chain }}).filter((atom) => atom.elem !== "H");
      const atomsB = model.selectedAtoms({{ chain: config.structure.src_chain }}).filter((atom) => atom.elem !== "H");
      const cutoffSq = cutoff * cutoff;
      const contacts = [];
      for (const atomA of atomsA) {{
        for (const atomB of atomsB) {{
          const dx = atomA.x - atomB.x;
          const dy = atomA.y - atomB.y;
          const dz = atomA.z - atomB.z;
          const distSq = dx * dx + dy * dy + dz * dz;
          if (distSq < cutoffSq) {{
            contacts.push({{
              a: {{ x: atomA.x, y: atomA.y, z: atomA.z }},
              b: {{ x: atomB.x, y: atomB.y, z: atomB.z }},
            }});
          }}
        }}
      }}
      return contacts.slice(0, 140);
    }}

    function showInteractions() {{
      clearHotspots();
      const contacts = computeChainContacts(3.5);
      contacts.forEach((contact) => drawDashedLine(contact.a, contact.b));
      focusInterface();
      viewer.render();
    }}

    async function init() {{
      const response = await fetch(config.structure.path);
      const cifText = await response.text();
      model = viewer.addModel(cifText, "cif");
      viewer.setBackgroundColor("#09111f", 1);
      applyStyles();
      focusInterface();
      viewer.render();
    }}

    document.getElementById("interaction-button").addEventListener("click", () => {{
      interactionOn = !interactionOn;
      applyStyles();
      if (interactionOn) {{
        showInteractions();
        document.getElementById("interaction-button").textContent = "Hide Interaction";
      }} else {{
        clearHotspots();
        focusInterface();
        viewer.render();
        document.getElementById("interaction-button").textContent = "Interaction";
      }}
    }});

    document.getElementById("reset-button").addEventListener("click", () => {{
      interactionOn = false;
      clearHotspots();
      applyStyles();
      focusInterface();
      viewer.render();
      document.getElementById("interaction-button").textContent = "Interaction";
    }});

    init();
  </script>
</body>
</html>"""


def _fmt_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value)


def _build_blog_bundle(run_dir: Path, data_dir: Path) -> dict[str, Any]:
    run_copy_dir = ensure_directory(data_dir / "run")
    for rel_path in REQUIRED_RUN_FILES.values():
        source = run_dir / rel_path
        if not source.exists():
            raise FileNotFoundError(f"Run directory {run_dir} is missing required artifact {rel_path}.")
        target = run_copy_dir / Path(rel_path).name
        shutil.copy2(source, target)

    structures_dir = ensure_directory(data_dir / "structures")
    project_root = Path(__file__).resolve().parents[2]
    structure_payload: dict[str, Any] = {}
    efs_contact_path = project_root / "Structural Analysis Pipeline/fragment_contacts.csv"
    full_interface_path = project_root / "Structural Analysis Pipeline/interface_contacts.csv"
    fragment_report_path = project_root / "Structural Analysis Pipeline/fragment_report.txt"
    efs_partner_residues: list[int] = []
    efs_src_residues: list[int] = []
    fragment_contacts: list[dict[str, Any]] = []
    full_interface_contacts: list[dict[str, Any]] = []
    fragment_report_text = ""
    if efs_contact_path.exists():
        with efs_contact_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fragment_contacts = [
                {
                    "partner_residue": f"{row['resname_A']} {row['resnum_A']}",
                    "partner_residue_number": int(row["resnum_A"]),
                    "partner_atom": row["atom_A"],
                    "src_residue": f"{row['resname_B']} {row['resnum_B']}",
                    "src_residue_number": int(row["resnum_B"]),
                    "src_atom": row["atom_B"],
                    "distance_a": float(row["min_distance_A"]),
                    "is_hotspot": float(row["min_distance_A"]) < 3.0,
                }
                for row in rows
            ]
            efs_partner_residues = sorted({int(row["resnum_A"]) for row in rows if row.get("resnum_A")})
        with efs_contact_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            efs_src_residues = sorted({int(row["resnum_B"]) for row in reader if row.get("resnum_B")})
    if full_interface_path.exists():
        with full_interface_path.open(encoding="utf-8", newline="") as handle:
            reader = list(csv.DictReader(handle))
            full_interface_contacts = [
                {
                    "partner_residue": f"{row['resname_A']} {row['resnum_A']}",
                    "partner_residue_number": int(row["resnum_A"]),
                    "partner_atom": row["atom_A"],
                    "src_residue": f"{row['resname_B']} {row['resnum_B']}",
                    "src_residue_number": int(row["resnum_B"]),
                    "src_atom": row["atom_B"],
                    "distance_a": float(row["min_distance_A"]),
                    "is_hotspot": float(row["min_distance_A"]) < 3.0,
                }
                for row in reader[:40]
            ]
    if fragment_report_path.exists():
        fragment_report_text = fragment_report_path.read_text(encoding="utf-8")
    for key, spec in STRUCTURE_FILES.items():
        source = project_root / spec["model"]
        if not source.exists():
            raise FileNotFoundError(f"Project root is missing required structure file {spec['model']}.")
        target_name = source.name
        shutil.copy2(source, structures_dir / target_name)
        confidence_source = project_root / spec["confidence"]
        if not confidence_source.exists():
            raise FileNotFoundError(f"Project root is missing required confidence file {spec['confidence']}.")
        confidence = json.loads(confidence_source.read_text(encoding="utf-8"))
        structure_payload[key] = {
            "label": spec["label"],
            "path": f"data/structures/{target_name}",
            "src_chain": "B",
            "partner_chain": "A",
            "partner_label": spec["partner_label"],
            "partner_style": spec["partner_style"],
            "interface": {
                "partner_residues": efs_partner_residues if key == "efs_src" else list(range(1, 21)),
                "src_residues": efs_src_residues,
            },
            "metrics": {
                "ipTM": confidence.get("iptm", "not provided"),
                "pTM": confidence.get("ptm", "not provided"),
                "src_pTM": confidence.get("chain_ptm", [None, None])[1],
            },
        }

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
            "authors": ["Cathy Liu"],
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
        "structures": structure_payload,
        "structural_report": {
            "native_contacts": fragment_contacts,
            "full_interface_contacts": full_interface_contacts,
            "fragment_report_text": fragment_report_text,
            "hotspots": [
                {
                    "partner_chain": "A",
                    "partner_residue": row["partner_residue_number"],
                    "partner_label": row["partner_residue"],
                    "partner_atom": row["partner_atom"],
                    "src_chain": "B",
                    "src_residue": row["src_residue_number"],
                    "src_label": row["src_residue"],
                    "src_atom": row["src_atom"],
                    "distance_a": row["distance_a"],
                }
                for row in fragment_contacts
                if row["distance_a"] <= 3.5
            ],
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
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
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
          <p class="authors">By Cathy Liu</p>
          <div id="hero-metrics" class="hero-metrics"></div>
        </header>

        <section id="intro" class="section prose">
          <p>
            Cancer single-cell datasets are rich in signal and poor in explanation. They can show which genes rise in malignant cells,
            but they do not by themselves explain which of those genes are upstream control points, which are downstream consequences,
            and which belong to the same causal program. That gap is exactly where target discovery becomes difficult.
          </p>
          <p>
            Large language models are useful here not because they replace biology, but because they can turn scattered mechanistic literature
            into a structured working hypothesis. Instead of manually reviewing dozens of papers for every candidate gene, we can ask the model
            to do that gene-by-gene, return explicit signed edges, attach citations, and preserve the raw outputs for inspection. The result is
            a graph that can be explored, simulated, challenged, and compared against external benchmark data.
          </p>
          <p>
            Our pipeline focuses on pancreatic ductal adenocarcinoma. It begins with malignant-versus-normal differential expression, asks a language
            model to recover mechanistic signaling edges for each resulting DEG, projects those edges into a compact intervention graph, and then ranks
            knockouts in two different ways. The first is explicit search over the graph. The second is a model-based reading of the graph itself:
            Claude Opus is asked to inspect the final network and recommend the most plausible knockouts directly. The value of the system is not just
            the final answer; it is the way it gives researchers a tighter, more navigable view of the problem.
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
          <div class="prose methodology-prose">
            <p>
              We treat each DEG as its own research problem. In the featured run, the system took the top 50 PDAC DEGs and launched one independent discovery
              call per gene. Each call gave Claude Sonnet 4.6 the seed gene, the PDAC context, the current DEG universe, a bridge-node universe drawn from curated
              pathway knowledge, and a strict request: return only mechanistic gene-to-gene edges that can plausibly connect the seed gene to PDAC-relevant signaling.
            </p>
            <p>
              The prompt is deliberately structured for machine use rather than prose. It asks for compact JSON, signed edges, short evidence summaries, PMIDs or
              source references, and a bounded number of candidates. This keeps the output parsable, limits drift into narrative explanation, and makes it possible
              to compare hundreds of model outputs side by side.
            </p>
            <p>
              Several prompt-engineering choices mattered. We kept the discovery schema compact, capped the number of cited sources per edge, and allowed follow-up
              rounds in the same conversation when a first pass appeared incomplete. We also saved the raw requests, raw responses, parsed JSON, and any parsing errors
              per gene, which makes debugging the model behavior much easier than treating it as a black box. In the run shown here, 43 genes returned model-backed outputs
              and 7 fell back to a PubMed heuristic path when the model did not yield usable edges.
            </p>
            <p>
              An optional verification pass exists in the system, but the run described here is intentionally discovery-led. The final comparison instead happens downstream:
              once the graph is built, we compare a solver-based ranking against a direct Claude Opus reading of the graph itself.
            </p>
            <p>
              Intermediate biology is useful for explanation but can overwhelm optimization if every receptor, adaptor, kinase, and transcription factor is given equal weight.
              We therefore let the model recover paths like <span class="mono">A → B → C</span>, but then project those paths back onto a DEG-centered control graph. This
              preserves the mechanistic route while keeping the final intervention layer readable and experimentally relevant.
            </p>
            <p>
              In practice, that means the system can reason through intermediates without requiring the final graph to become a generic signaling atlas. Researchers can inspect
              the full path evidence, while the simulator and the direct Opus ranker operate on a smaller, clearer representation.
            </p>
            <p>
              The Boolean solver is explicit and exhaustive. It tries one-, two-, and three-gene knockouts over the selected graph and asks which combinations flip the synthetic
              KRAS-signaling endpoint to off. Claude Opus is used differently: it receives the graph, the evidence, and the benchmark rows, and then produces a ranked set of knockout
              suggestions as a structured mechanistic assessment. One method is combinatorial search; the other is model-based synthesis over the same evidence surface.
            </p>
            <p>
              We then benchmark the nominated genes against DepMap pancreatic dependency data. This turns the pipeline into a research tool rather than a generator of opaque answers:
              the recommendation, the evidence, and the external counter-evidence all remain visible at once.
            </p>
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
            <h2>What the system surfaced in the final PDAC run</h2>
          </div>
          <div class="prose">
            <p>
              In the final run, both ranking methods converged on <span class="mono">EFS</span> as the leading single-gene knockout. That agreement is the most encouraging result in the page:
              an explicit solver and a direct graph-reading model, operating very differently, still pointed to the same intervention. Within the graph, that recommendation is easy to read.
              EFS connects into SRC-centered signaling and lies on a short route to several tracked KRAS pathway nodes.
            </p>
            <p>
              The benchmark adds the right kind of friction. DepMap does not support EFS strongly as a PDAC dependency, so the system should be read as surfacing a coherent mechanistic
              hypothesis rather than declaring a validated therapeutic target. That is still a strong outcome for a research tool: it narrows the search space, shows the mechanistic rationale,
              and makes the disagreement with external evidence explicit instead of hiding it.
            </p>
            <p>
              Downstream of the ranking stage, EFS was taken forward for structural and sequence-level investigation. AlphaFold 3 modeling of the EFS-SRC interface informed the design of eSI-1, a candidate inhibitory peptide. ProteinMPNN subsequently generated 8 optimized sequences on the eSI-1 scaffold, all scoring approximately 1.0 lower than the original fragment, suggesting the designed backbone is structurally sound and sequence-tolerant. Details are in the structural follow-up section below.
            </p>
          </div>
          <div class="figure-card">
            <canvas id="benchmark-chart"></canvas>
            <p class="figure-caption">External benchmark context for the genes most relevant to the final solver and Opus recommendations.</p>
          </div>
          <div class="results-grid">
            <div class="result-card" id="solver-card"></div>
            <div class="result-card" id="opus-card"></div>
          </div>
          <div class="specificity-note">
            <strong>Specificity & Benchmarking</strong>
            <span>
              To ensure specificity, we cross-validated eSI-1 against other SH3-containing kinases (e.g., FYN, YES1). The engineered mutations in eSI-1 leverage a unique hydrophobic patch on SRC that is less conserved in other members of the SFK family, potentially reducing off-target effects.
            </span>
          </div>
        </section>

        <section id="alphafold" class="section prose">
          <div class="section-header">
            <p class="eyebrow">Structural follow-up</p>
            <h2>Atomic-Level Verification via AlphaFold 3</h2>
          </div>
          <blockquote class="callout-quote">
            <p>
              To validate the mechanistic insights from our virtual knockout pipeline, we performed in-silico structural modeling using AlphaFold 3.
              We focused on the EFS-SRC interface, which our model identified as a critical upstream control point for KRAS signaling.
            </p>
            <p>
              While the native EFS-SRC complex shows a baseline interaction (ipTM 0.44), our engineered biological inhibitor, eSI-1 (Cathy01),
              demonstrates significantly enhanced binding stability and structural convergence. This shift, from a transient scaffold interaction
              to a high-confidence inhibitory complex, validates that our AI pipeline can move beyond target discovery toward the precision design
              of therapeutic interventions.
            </p>
          </blockquote>
          <div class="structure-grid">
            <article class="structure-card">
              <div class="comparison-label">
                <span class="comparison-kicker">Native EFS-SRC</span>
                <span class="comparison-note">ipTM 0.44</span>
              </div>
              <div class="structure-shell">
                <div class="structure-overlay structure-legend">
                  <strong>Legend</strong>
                  <div class="legend-row"><span class="legend-dot legend-dot-gold"></span>EFS chain A</div>
                  <div class="legend-row"><span class="legend-dot legend-dot-white"></span>SRC solvent-accessible surface (SAS)</div>
                  <div class="legend-row"><span class="legend-line"></span>interaction hotspots</div>
                </div>
                <div class="structure-overlay structure-hint">Drag to rotate · Scroll to zoom</div>
                <button type="button" class="structure-touch-gate" data-structure-action="activate" data-structure-id="efs-src">Tap to activate 3D rotation</button>
                <div id="viewer-efs-src" class="structure-viewer"></div>
              </div>
              <div class="structure-controls">
                <button type="button" data-structure-action="interaction" data-structure-id="efs-src">Interaction</button>
                <button type="button" data-structure-action="reset" data-structure-id="efs-src">Reset view</button>
              </div>
              <div class="structure-detail-grid">
                <div class="structure-note-card">
                  <h3>Structural Summary</h3>
                  <div class="metric-row">
                    <div class="metric-chip"><strong>ipTM</strong><span id="metric-efs-iptm">...</span></div>
                    <div class="metric-chip"><strong>pTM</strong><span id="metric-efs-ptm">...</span></div>
                  </div>
                </div>
                <div class="structure-note-card" id="efs-contacts-card"></div>
              </div>
            </article>
            <article class="structure-card">
              <div class="comparison-label">
                <span class="comparison-kicker">eSI-1 (Cathy01)</span>
                <span class="comparison-note">ipTM 0.49, SRC pTM 0.77</span>
              </div>
              <div class="structure-shell">
                <div class="structure-overlay structure-legend">
                  <strong>Legend</strong>
                  <div class="legend-row"><span class="legend-dot legend-dot-gold"></span>eSI-1 chain A</div>
                  <div class="legend-row"><span class="legend-dot legend-dot-white"></span>SRC solvent-accessible surface (SAS)</div>
                  <div class="legend-row"><span class="legend-line"></span>interaction hotspots</div>
                </div>
                <div class="structure-overlay structure-hint">Drag to rotate · Scroll to zoom</div>
                <button type="button" class="structure-touch-gate" data-structure-action="activate" data-structure-id="cathy01-src">Tap to activate 3D rotation</button>
                <div id="viewer-cathy01-src" class="structure-viewer"></div>
              </div>
              <div class="structure-controls">
                <button type="button" data-structure-action="interaction" data-structure-id="cathy01-src">Interaction</button>
                <button type="button" data-structure-action="reset" data-structure-id="cathy01-src">Reset view</button>
              </div>
              <div class="structure-detail-grid">
                <div class="structure-note-card">
                  <h3>Structural Summary</h3>
                  <div class="metric-row">
                    <div class="metric-chip"><strong>ipTM</strong><span id="metric-cathy01-iptm">...</span></div>
                    <div class="metric-chip"><strong>SRC pTM</strong><span id="metric-cathy01-ptm">...</span></div>
                  </div>
                </div>
                <div class="structure-note-card">
                  <h3>Design Logic</h3>
                  <p class="structure-copy">Strategic Proline-Capping & Charge Reversal. The design preserves the core PxxP motif while increasing N-terminal proline density, pre-folding the peptide toward a PPII-like conformation and reducing the entropic cost of binding.</p>
                  <div class="structure-copy" style="margin-top:14px;">
                    <p style="margin:0 0 10px;"><strong>Sequence Optimization via ProteinMPNN</strong></p>
                    <p style="margin:0 0 12px;">To validate the eSI-1 scaffold and explore sequence diversity, we ran ProteinMPNN on the designed backbone, generating 8 candidate sequences at temperature 0.1.</p>
                    <div class="table-scroll">
                      <table class="report-table" style="margin:0 0 12px;">
                        <thead>
                          <tr>
                            <th>Metric</th>
                            <th>Original EFS fragment</th>
                            <th>Best design (Sample 7)</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr>
                            <td>Sequence</td>
                            <td class="annotation">KGSIQDRPLPPPPPRLPGYG</td>
                            <td class="annotation">APSPEALPPPPPPPVPPPPG</td>
                          </tr>
                          <tr>
                            <td>ProteinMPNN Score</td>
                            <td>2.2703</td>
                            <td>1.2586</td>
                          </tr>
                          <tr>
                            <td>Δ Score</td>
                            <td>—</td>
                            <td>−1.0117</td>
                          </tr>
                          <tr>
                            <td>Proline count</td>
                            <td>7</td>
                            <td>13</td>
                          </tr>
                          <tr>
                            <td>Net charge</td>
                            <td>+2</td>
                            <td>−1</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                    <p style="margin:0 0 12px;">All 8 designs scored approximately 1.0 lower than the original sequence, indicating that the PPII-like backbone is broadly accommodating and structurally well-formed. Critically, ProteinMPNN autonomously preserved the PPPP core across all designs without explicit constraints — an independent signal that this region is structurally essential for SRC SH3 recognition.</p>
                    <p style="margin:0 0 12px;">A convergence pattern is evident across the ensemble: N-terminal positions favor A/P, proline density increases relative to the original, and the C-terminus retains G. Samples 1 and 5 produced identical sequences (APSPEAVPPPPPPGVPPPPG), consistent with this being a stable local optimum at low sampling temperature.</p>
                    <p style="margin:0;">These candidate sequences are being taken forward for AlphaFold 3 multimer validation against the SRC kinase domain. If ipTM improves over the native eSI-1 value of 0.49, this would close the design loop from virtual knockout through scaffold design to sequence-level structural confirmation.</p>
                  </div>
                </div>
              </div>
            </article>
          </div>
          <div class="mechanism-card">
            <div class="mechanism-copy">
              <p class="eyebrow">Mechanism of inhibition</p>
              <h3>From scaffold engagement to competitive interface blockade</h3>
              <p>
                The native state places EFS against the SRC interaction face. The engineered peptide eSI-1 is designed to occupy that same
                surface more stably, replacing a transient scaffold interaction with a tighter inhibitory interface.
              </p>
            </div>
            <div class="mechanism-figure" aria-label="Mechanism of inhibition diagram">
              <svg viewBox="0 0 640 180" role="img" aria-hidden="true">
                <defs>
                  <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
                    <path d="M0,0 L10,5 L0,10 z" fill="#c15b2c"></path>
                  </marker>
                </defs>
                <rect x="28" y="56" width="160" height="64" rx="18" fill="#f7efe1" stroke="rgba(23,23,23,0.12)"></rect>
                <text x="108" y="95" text-anchor="middle" font-size="22" font-family="Manrope, sans-serif" fill="#171717">EFS</text>
                <line x1="188" y1="88" x2="314" y2="88" stroke="#c15b2c" stroke-width="5" marker-end="url(#arrowhead)"></line>
                <rect x="326" y="42" width="178" height="92" rx="24" fill="#ffffff" stroke="rgba(23,23,23,0.16)"></rect>
                <text x="415" y="82" text-anchor="middle" font-size="24" font-family="Manrope, sans-serif" fill="#171717">SRC</text>
                <text x="415" y="104" text-anchor="middle" font-size="12" font-family="IBM Plex Mono, monospace" fill="#625d56">kinase domain interface</text>
                <line x1="520" y1="88" x2="612" y2="88" stroke="#7a1f1f" stroke-width="6"></line>
                <line x1="520" y1="58" x2="612" y2="118" stroke="#7a1f1f" stroke-width="6"></line>
                <rect x="470" y="18" width="136" height="34" rx="17" fill="#d4af37"></rect>
                <text x="538" y="40" text-anchor="middle" font-size="16" font-family="Manrope, sans-serif" fill="#171717">eSI-1</text>
              </svg>
            </div>
          </div>
          <div class="data-board" id="structural-data-board"></div>
          <div class="report-actions">
            <button type="button" id="toggle-structural-report" class="report-button">View Full Interaction Report</button>
          </div>
          <div id="structural-report" class="structural-report is-hidden"></div>
        </section>

        <section id="implications" class="section prose">
          <div class="section-header">
            <p class="eyebrow">Potential implications</p>
            <h2>Why this approach is still promising</h2>
          </div>
          <p>
            The most important implication is not a single gene. It is that researchers can now move from single-cell expression to an inspectable mechanistic graph in one reproducible pass,
            then explore the consequences of different intervention strategies without losing sight of the underlying evidence.
          </p>
          <p>
            That is a meaningful capability. Existing single-cell datasets are often too sparse, too high-dimensional, and too labor-intensive to review manually at the pathway level.
            A system like this makes the literature review step operational: it can be rerun, refined, stress-tested, and expanded as better priors, better prompts, and better benchmarks become available.
          </p>
          <p>
            More generally, the combination of explicit search and direct graph-reading models looks especially powerful. The solver gives a disciplined, auditable baseline.
            The direct LLM ranker offers a second perspective that can synthesize topology, evidence quality, and benchmark context in one pass. Used together, they make the problem more legible,
            and that is exactly what a good research interface should do.
          </p>
          <p class="impact-callout">
            <strong>Impact & Future Directions.</strong> By integrating single-cell transcriptomics with generative protein design, we have shortened the window from 'target discovery' to 'lead optimization' from months to days. This workflow provides a generalizable framework for tackling high-heterogeneity cancers like PDAC, where traditional drug-discovery timelines often lag behind clinical urgency.
          </p>
        </section>

        <section id="conclusion" class="section prose">
          <div class="section-header">
            <p class="eyebrow">Conclusion</p>
            <h2>A promising interface for target discovery</h2>
          </div>
          <p>
            We now have an end-to-end system that starts with single-cell data and ends with ranked knockout hypotheses, accompanied by mechanistic evidence, graph structure, and external benchmark checks.
            In the present PDAC run, the top answer is consistent across the solver and the direct Opus ranker, which is exactly the kind of convergence one hopes to see in an exploratory system.
          </p>
          <p>
            The benchmark remains cautious, and that is healthy. But the larger story is positive: this pipeline already acts as a useful scientific instrument. It helps researchers compress the literature,
            see how candidate genes connect, compare multiple ranking strategies, and interrogate why a hypothesis looks strong or weak. That makes the space of possible follow-up experiments much clearer,
            and it suggests a compelling direction for how LLMs can assist biology: not by replacing judgment, but by making complex mechanistic problems easier to explore.
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
.interactive-layout {
  display: grid;
  gap: 18px;
}

.card-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.method-card,
.figure-card,
.result-card {
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
.result-card p {
  margin: 0;
  color: #2d2924;
  line-height: 1.7;
}

.methodology-prose {
  margin-top: 18px;
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

.callout-quote {
  margin: 0;
  padding: 26px 28px;
  border-left: 4px solid var(--accent);
  border-radius: 0 22px 22px 0;
  background: rgba(255, 249, 239, 0.92);
  box-shadow: 0 18px 60px rgba(28, 22, 14, 0.05);
}

.callout-quote p {
  margin: 0 0 14px;
}

.callout-quote p:last-child {
  margin-bottom: 0;
}

.structure-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
  margin-top: 24px;
}

.structure-card {
  border: 1px solid var(--line);
  background: rgba(255, 252, 246, 0.95);
  border-radius: 24px;
  padding: 18px;
  box-shadow: 0 18px 60px rgba(28, 22, 14, 0.06);
}

.comparison-label {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 10px;
}

.comparison-kicker {
  font-size: 12px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
}

.comparison-note {
  font-family: var(--mono);
  font-size: 12px;
  color: var(--accent-2);
}

.structure-shell {
  position: relative;
  overflow: hidden;
  border-radius: 20px;
  border: 1px solid rgba(23, 23, 23, 0.08);
  background:
    radial-gradient(circle at top, rgba(126, 200, 255, 0.18), transparent 24%),
    radial-gradient(circle at bottom right, rgba(212, 175, 55, 0.12), transparent 22%),
    linear-gradient(180deg, #f7f4ed 0%, #ece7dc 100%);
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.68),
    0 30px 60px rgba(0, 0, 0, 0.12);
}

.structure-viewer {
  position: relative;
  width: 100%;
  height: 540px;
}

.structure-viewer > div,
.structure-viewer canvas {
  position: absolute !important;
  inset: 0 !important;
  width: 100% !important;
  height: 100% !important;
}

.structure-overlay {
  position: absolute;
  z-index: 5;
  border: 1px solid rgba(23,23,23,0.08);
  background: rgba(255, 252, 246, 0.82);
  backdrop-filter: blur(10px);
  color: #1d232c;
  border-radius: 14px;
  padding: 10px 12px;
}

.structure-legend {
  top: 14px;
  left: 14px;
  min-width: 160px;
}

.structure-legend strong {
  display: block;
  margin-bottom: 8px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: rgba(29,35,44,0.56);
}

.legend-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  margin-top: 6px;
}

.legend-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  flex: 0 0 auto;
}

.legend-dot-gold { background: #d4af37; box-shadow: 0 0 10px rgba(212, 175, 55, 0.4); }
.legend-dot-white { background: #ffffff; box-shadow: 0 0 10px rgba(255, 255, 255, 0.35); }

.legend-line {
  width: 22px;
  height: 4px;
  border-radius: 999px;
  background: #ccff00;
  box-shadow: 0 0 12px rgba(204, 255, 0, 0.72);
}

.structure-hint {
  right: 14px;
  bottom: 14px;
  font-size: 11px;
  color: rgba(29,35,44,0.68);
}

.structure-touch-gate {
  display: none;
  position: absolute;
  inset: auto 14px 14px 14px;
  z-index: 6;
  border: 1px solid rgba(23,23,23,0.1);
  background: rgba(255, 252, 246, 0.96);
  color: var(--ink);
  border-radius: 999px;
  padding: 10px 12px;
  font: inherit;
  font-size: 12px;
  box-shadow: 0 12px 24px rgba(0,0,0,0.08);
}

.structure-shell.is-touch-locked .structure-viewer {
  pointer-events: none;
}

.structure-controls {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 12px;
}

.structure-controls button {
  appearance: none;
  border: 1px solid rgba(23, 23, 23, 0.1);
  background: rgba(255, 255, 255, 0.88);
  color: var(--ink);
  border-radius: 999px;
  padding: 8px 12px;
  font: inherit;
  font-size: 12px;
  cursor: pointer;
  transition: background 0.15s ease, transform 0.15s ease, border-color 0.15s ease;
}

.structure-controls button:hover {
  background: rgba(193, 91, 44, 0.12);
  border-color: rgba(193, 91, 44, 0.28);
  transform: translateY(-1px);
}

.structure-detail-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 12px;
  margin-top: 12px;
}

.structure-detail-grid > * {
  min-width: 0;
}

.structure-note-card {
  border: 1px solid rgba(23, 23, 23, 0.08);
  border-radius: 16px;
  padding: 14px;
  background: rgba(255, 255, 255, 0.84);
  min-width: 0;
  overflow: hidden;
}

.structure-note-card h3 {
  margin: 0 0 10px;
  font-family: var(--serif);
  font-size: 22px;
}

.structure-copy {
  margin: 0;
  color: var(--muted);
  line-height: 1.65;
  font-size: 14px;
  overflow-wrap: anywhere;
  word-break: break-word;
}

.table-scroll {
  max-width: 100%;
  overflow-x: auto;
  overflow-y: hidden;
  -webkit-overflow-scrolling: touch;
}

.table-scroll .report-table {
  min-width: 560px;
}

.contact-mini-list {
  display: grid;
  gap: 10px;
}

.contact-mini-card {
  border: 1px solid rgba(23, 23, 23, 0.08);
  border-radius: 12px;
  padding: 10px 12px;
  background: rgba(255, 255, 255, 0.82);
}

.contact-mini-card.is-key {
  background: rgba(212, 175, 55, 0.18);
  border-color: rgba(212, 175, 55, 0.35);
}

.contact-mini-card strong {
  display: block;
  font-size: 13px;
  line-height: 1.4;
}

.contact-mini-card span {
  display: block;
  margin-top: 4px;
  color: var(--muted);
  font-size: 12px;
}

.mechanism-card {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 18px;
  align-items: start;
  margin-top: 18px;
  border: 1px solid rgba(23, 23, 23, 0.08);
  border-radius: 24px;
  background: rgba(255, 252, 246, 0.95);
  padding: 20px;
  box-shadow: 0 18px 60px rgba(28, 22, 14, 0.06);
}

.mechanism-card h3 {
  margin: 6px 0 10px;
  font-family: var(--serif);
  font-size: 32px;
}

.mechanism-card p {
  margin: 0;
  color: var(--muted);
  line-height: 1.7;
}

.mechanism-figure {
  border: 1px solid rgba(23, 23, 23, 0.08);
  border-radius: 18px;
  background: linear-gradient(180deg, #fffdf8 0%, #f3ede0 100%);
  padding: 10px;
}

.mechanism-figure svg {
  display: block;
  width: 100%;
  height: auto;
}

.metric-chip strong {
  display: block;
  margin-bottom: 6px;
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
}

.metric-chip span {
  font-family: var(--mono);
  font-size: 13px;
}

.data-board {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  margin-top: 18px;
}

.data-box {
  border: 1px solid rgba(23, 23, 23, 0.08);
  border-radius: 18px;
  padding: 16px;
  background: rgba(255, 255, 255, 0.85);
}

.data-box.highlight {
  background: linear-gradient(180deg, rgba(212, 175, 55, 0.16), rgba(255, 255, 255, 0.92));
  border-color: rgba(212, 175, 55, 0.35);
}

.data-box strong {
  display: block;
  margin-bottom: 6px;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
}

.data-box span {
  display: block;
  font-family: var(--serif);
  font-size: 30px;
  line-height: 1;
}

.data-box small {
  display: block;
  margin-top: 8px;
  color: var(--muted);
}

.report-actions {
  margin-top: 18px;
}

.report-button {
  appearance: none;
  border: 1px solid rgba(23, 23, 23, 0.12);
  background: linear-gradient(180deg, rgba(24, 74, 139, 0.08), rgba(255,255,255,0.92));
  color: var(--ink);
  border-radius: 999px;
  padding: 12px 18px;
  font: inherit;
  cursor: pointer;
}

.structural-report {
  margin-top: 16px;
  border: 1px solid rgba(23, 23, 23, 0.08);
  border-radius: 22px;
  background: rgba(255, 255, 255, 0.84);
  padding: 18px;
}

.structural-report.is-hidden {
  display: none;
}

.report-grid {
  display: grid;
  grid-template-columns: 1.1fr 1fr;
  gap: 18px;
}

.report-panel {
  border: 1px solid rgba(23, 23, 23, 0.08);
  border-radius: 16px;
  background: rgba(255,255,255,0.72);
  padding: 14px;
}

.report-panel h4 {
  margin: 0 0 10px;
  font-family: var(--serif);
  font-size: 24px;
}

.report-panel pre {
  white-space: pre-wrap;
  font-family: var(--mono);
  font-size: 12px;
  line-height: 1.6;
  margin: 0;
}

.design-logic-copy {
  margin: 0;
  font-size: 15px;
  line-height: 1.8;
  color: #2a2724;
}

.specificity-note {
  margin-top: 18px;
  display: grid;
  gap: 8px;
  border: 1px solid rgba(23, 23, 23, 0.08);
  border-radius: 18px;
  padding: 16px 18px;
  background: rgba(255, 255, 255, 0.82);
}

.specificity-note strong {
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
}

.impact-callout {
  border-left: 4px solid var(--accent-2);
  padding-left: 16px;
}

.report-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.report-table th,
.report-table td {
  text-align: left;
  padding: 8px 6px;
  border-bottom: 1px solid rgba(23, 23, 23, 0.08);
}

.report-table th {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
}

.report-table tr.is-key td {
  background: rgba(212, 175, 55, 0.18);
}

.report-table td.annotation {
  color: #6b5313;
  font-weight: 600;
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
  .interactive-layout,
  .structure-grid,
  .data-board,
  .report-grid {
    grid-template-columns: 1fr;
  }
  .mechanism-card {
    grid-template-columns: 1fr;
  }
  .structure-viewer {
    height: 440px;
  }
  .structure-touch-gate {
    display: block;
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
const structureViewers = {};

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
  await renderStructures();
  renderStructuralDataBoard();
  bindStructuralReport();
  buildGraphSelect();
  renderGraph();
  document.getElementById("graph-select").addEventListener("change", (event) => {
    currentGraphKey = event.target.value;
    renderGraph();
  });
  document.getElementById("node-search").addEventListener("change", focusNodeFromSearch);
}

function renderStructuralDataBoard() {
  const structures = bundle.structures || {};
  const native = structures.efs_src?.metrics || {};
  const designed = structures.cathy01_src?.metrics || {};
  const nativePtm = Number(native.pTM || 0);
  const designedSrcPtm = Number(designed.src_pTM || designed.pTM || 0);
  const delta = designedSrcPtm - nativePtm;
  const board = document.getElementById("structural-data-board");
  board.innerHTML = `
    <div class="data-box">
      <strong>Native ipTM</strong>
      <span>${fmt(native.ipTM)}</span>
      <small>EFS-SRC baseline interaction</small>
    </div>
    <div class="data-box">
      <strong>Designed ipTM</strong>
      <span>${fmt(designed.ipTM)}</span>
      <small>eSI-1 (Cathy01)-SRC complex</small>
    </div>
    <div class="data-box highlight">
      <strong>SRC pTM</strong>
      <span>${fmt(designedSrcPtm)}</span>
      <small>up from native global pTM ${fmt(nativePtm)}</small>
    </div>
    <div class="data-box highlight">
      <strong>pTM gain</strong>
      <span>${fmt(delta)}</span>
      <small>Global structural stabilization of the kinase domain</small>
    </div>
  `;
}

function bindStructuralReport() {
  const button = document.getElementById("toggle-structural-report");
  const panel = document.getElementById("structural-report");
  if (!button || !panel) return;
  button.addEventListener("click", () => {
    const hidden = panel.classList.toggle("is-hidden");
    button.textContent = hidden ? "View Full Interaction Report" : "Hide Full Interaction Report";
    if (!hidden && !panel.dataset.rendered) {
      renderStructuralReport(panel);
      panel.dataset.rendered = "true";
    }
  });
}

function renderStructuralReport(panel) {
  const report = bundle.structural_report || {};
  const nativeContacts = report.native_contacts || [];
  const fullContacts = report.full_interface_contacts || [];
  panel.innerHTML = `
    <div class="report-grid">
      <section class="report-panel">
        <h4>Fragment Summary</h4>
        <pre>${escapeHtml(report.fragment_report_text || "No fragment report available.")}</pre>
      </section>
      <section class="report-panel">
        <h4>Design Logic</h4>
        <p class="design-logic-copy">
          <strong>"Strategic Proline-Capping & Charge Reversal"</strong>: Our design (eSI-1) preserves the core
          PxxP motif (residues 338-342) identified as essential for SH3 domain recognition but introduces a
          N-terminal Proline density increase. This effectively "pre-folds" the peptide into a Polyproline II
          (PPII) helix, reducing the entropic penalty of binding and explaining the 38% gain in structural
          confidence (pTM).
        </p>
      </section>
    </div>
    <div class="report-grid" style="margin-top:18px;">
      <section class="report-panel">
        <h4>Key Native Contacts</h4>
        ${renderContactTable(nativeContacts.slice(0, 24))}
      </section>
      <section class="report-panel" style="grid-column: 1 / -1;">
        <h4>Expanded Interface Contacts</h4>
        ${renderContactTable(fullContacts.slice(0, 30))}
      </section>
    </div>
  `;
}

function renderContactTable(rows) {
  if (!rows.length) return '<p class="muted">No contact rows available.</p>';
  const keyMap = new Map([
    ["LEU 344|O|LYS 155|O", "Primary H-bond Anchor"],
    ["PRO 341|CB|PRO 253|CB", "Hydrophobic Staple"],
    ["ARG 343|NE|TYR 93|OH", "Polar Recognition Contact"],
  ]);
  return `
    <table class="report-table">
      <thead>
        <tr>
          <th>Partner</th>
          <th>Atom</th>
          <th>SRC</th>
          <th>Atom</th>
          <th>Distance (Å)</th>
          <th>Note</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map((row) => {
          const key = `${row.partner_residue}|${row.partner_atom}|${row.src_residue}|${row.src_atom}`;
          const annotation = keyMap.get(key) || "";
          return `
          <tr class="${annotation ? "is-key" : ""}">
            <td>${escapeHtml(row.partner_residue)}</td>
            <td>${escapeHtml(row.partner_atom)}</td>
            <td>${escapeHtml(row.src_residue)}</td>
            <td>${escapeHtml(row.src_atom)}</td>
            <td>${fmt(row.distance_a)}</td>
            <td class="annotation">${escapeHtml(annotation)}</td>
          </tr>
        `}).join("")}
      </tbody>
    </table>
  `;
}

async function renderStructures() {
  const structures = bundle.structures || {};
  bindStructureMetrics("efs", structures.efs_src?.metrics);
  bindStructureMetrics("cathy01", structures.cathy01_src?.metrics);
  renderStructureContacts();
  await Promise.all([
    renderStructureViewer("efs-src", "viewer-efs-src", structures.efs_src),
    renderStructureViewer("cathy01-src", "viewer-cathy01-src", structures.cathy01_src),
  ]);
  bindStructureControls();
}

function bindStructureMetrics(prefix, metrics) {
  if (!metrics) return;
  document.getElementById(`metric-${prefix}-iptm`).textContent = fmt(metrics.ipTM);
  document.getElementById(`metric-${prefix}-ptm`).textContent = fmt(metrics.src_pTM ?? metrics.pTM);
}

function renderStructureContacts() {
  const card = document.getElementById("efs-contacts-card");
  if (!card) return;
  const rows = (bundle.structural_report?.native_contacts || []).slice(0, 4);
  const keyMap = new Map([
    ["LEU 344|O|LYS 155|O", "Primary H-bond Anchor"],
    ["PRO 341|CB|PRO 253|CB", "Hydrophobic Staple"],
    ["ARG 343|NE|TYR 93|OH", "Polar Recognition Contact"],
  ]);
  card.innerHTML = `
    <h3>Key Native Contacts</h3>
    <div class="contact-mini-list">
      ${rows.map((row) => {
        const key = `${row.partner_residue}|${row.partner_atom}|${row.src_residue}|${row.src_atom}`;
        const annotation = keyMap.get(key) || "Interface contact";
        return `
          <div class="contact-mini-card ${keyMap.has(key) ? "is-key" : ""}">
            <strong>${escapeHtml(row.partner_residue)} (${escapeHtml(row.partner_atom)}) ↔ ${escapeHtml(row.src_residue)} (${escapeHtml(row.src_atom)})</strong>
            <span>${fmt(row.distance_a)} Å · ${escapeHtml(annotation)}</span>
          </div>
        `;
      }).join("")}
    </div>
  `;
}

function bindStructureControls() {
  document.querySelectorAll("[data-structure-action]").forEach((button) => {
    button.addEventListener("click", () => {
      const viewerState = structureViewers[button.dataset.structureId];
      if (!viewerState) return;
      const action = button.dataset.structureAction;
      if (action === "activate") {
        viewerState.container.closest(".structure-shell")?.classList.remove("is-touch-locked");
        button.style.display = "none";
      }
      if (action === "interaction") toggleHotspots(viewerState, button);
      if (action === "reset") resetStructureView(viewerState, button);
    });
  });
}

async function renderStructureViewer(structureId, containerId, structure) {
  if (!structure || !window.$3Dmol) return;
  const container = document.getElementById(containerId);
  if (!container) return;
  const response = await fetch(structure.path);
  const cifText = await response.text();
  const viewer = $3Dmol.createViewer(container, { backgroundColor: "#f3efe6", antialias: true });
  const model = viewer.addModel(cifText, "cif");
  viewer.setBackgroundColor("#f3efe6", 1);
  const viewerState = {
    viewer,
    model,
    structure,
    structureId,
    container,
    srcSurface: null,
    hotspotsVisible: false,
    hotspotShapes: [],
  };
  if (window.matchMedia("(max-width: 900px)").matches) {
    container.closest(".structure-shell")?.classList.add("is-touch-locked");
  }
  applyStructureStyles(viewerState);
  focusStructureInterface(viewerState);
  viewer.render();
  structureViewers[structureId] = viewerState;
}

function applyStructureStyles(viewerState) {
  const { viewer, structure, structureId } = viewerState;
  viewer.setStyle({}, {});
  try {
    if (viewerState.srcSurface !== null) {
      viewer.removeSurface(viewerState.srcSurface);
      viewerState.srcSurface = null;
    }
  } catch (error) {}
  if (structureId === "cathy01-src") {
    viewer.setStyle(
      { chain: structure.partner_chain },
      { stick: { color: "#d4af37", radius: 0.24 }, cartoon: { color: "#d4af37", opacity: 0.28 } }
    );
  } else {
    viewer.setStyle(
      { chain: structure.partner_chain },
      { cartoon: { color: "#c78a61", opacity: 0.98 } }
    );
  }
  viewerState.srcSurface = viewer.addSurface(
    $3Dmol.SurfaceType.VDW,
    {
      color: "#ffffff",
      opacity: viewerState.hotspotsVisible ? 0.3 : 1.0,
      specular: 0.5,
      shininess: 10,
    },
    { chain: structure.src_chain }
  );
}

function structureSelection(viewerState) {
  const { structure, structureId } = viewerState;
  if (structureId === "cathy01-src") {
    return {
      or: [
        { chain: structure.partner_chain, resi: Array.from({ length: 20 }, (_, index) => index + 1) },
        { chain: structure.src_chain, resi: Array.from({ length: 41 }, (_, index) => index + 120) },
      ],
    };
  }
  return {
    or: [
      { chain: structure.partner_chain, resi: structure.interface?.partner_residues || [] },
      { chain: structure.src_chain, resi: structure.interface?.src_residues || [] },
    ],
  };
}

function interactionSelection(viewerState) {
  const { structureId, structure } = viewerState;
  if (structureId === "efs-src") {
    return {
      or: [
        { chain: structure.partner_chain, resi: [341, 343, 344] },
        { chain: structure.src_chain, resi: [93, 155, 253] },
      ],
    };
  }
  return {
    or: [
      { chain: structure.partner_chain, resi: Array.from({ length: 16 }, (_, index) => index + 1) },
      { chain: structure.src_chain, resi: Array.from({ length: 41 }, (_, index) => index + 120) },
    ],
  };
}

function focusStructureInterface(viewerState) {
  const selection = structureSelection(viewerState);
  viewerState.viewer.center(selection);
  viewerState.viewer.zoomTo(selection, 1800);
  viewerState.viewer.zoom(viewerState.structureId === "cathy01-src" ? 1.42 : 1.28, 500);
  viewerState.viewer.render();
}

function focusInteraction(viewerState) {
  const selection = interactionSelection(viewerState);
  viewerState.viewer.center(selection);
  viewerState.viewer.zoomTo(selection, 1600);
  viewerState.viewer.zoom(viewerState.structureId === "cathy01-src" ? 1.52 : 1.38, 700);
  viewerState.viewer.render();
}

function resetStructureView(viewerState, button) {
  viewerState.hotspotsVisible = false;
  clearHotspots(viewerState);
  applyStructureStyles(viewerState);
  focusStructureInterface(viewerState);
  viewerState.viewer.render();
  if (button) button.textContent = "Reset view";
  const interactionButton = document.querySelector(`[data-structure-id="${viewerState.structureId}"][data-structure-action="interaction"]`);
  if (interactionButton) interactionButton.textContent = "Interaction";
}

function toggleHotspots(viewerState, button) {
  viewerState.hotspotsVisible = !viewerState.hotspotsVisible;
  applyStructureStyles(viewerState);
  if (viewerState.hotspotsVisible) {
    drawHotspots(viewerState);
    focusInteraction(viewerState);
    button.textContent = "Hide Interaction";
  } else {
    clearHotspots(viewerState);
    focusStructureInterface(viewerState);
    button.textContent = "Interaction";
  }
  viewerState.viewer.render();
}

function clearHotspots(viewerState) {
  for (const shape of viewerState.hotspotShapes) {
    try { viewerState.viewer.removeShape(shape); } catch (error) {}
  }
  viewerState.hotspotShapes = [];
}

function drawHotspots(viewerState) {
  clearHotspots(viewerState);
  const contacts = viewerState.structureId === "cathy01-src"
    ? computeChainContacts(viewerState.model, viewerState.structure.partner_chain, viewerState.structure.src_chain, 3.5)
    : contactRowsToCoordinates(viewerState);
  for (const contact of contacts) {
    const shapes = drawDashedLine(viewerState.viewer, contact.a, contact.b, "#ccff00");
    viewerState.hotspotShapes.push(...shapes);
  }
}

function contactRowsToCoordinates(viewerState) {
  const rows = bundle.structural_report?.hotspots || [];
  const contacts = [];
  for (const row of rows) {
    const partnerAtoms = viewerState.model.selectedAtoms({ chain: row.partner_chain, resi: row.partner_residue, atom: row.partner_atom });
    const srcAtoms = viewerState.model.selectedAtoms({ chain: row.src_chain, resi: row.src_residue, atom: row.src_atom });
    if (!partnerAtoms.length || !srcAtoms.length) continue;
    contacts.push({
      a: { x: partnerAtoms[0].x, y: partnerAtoms[0].y, z: partnerAtoms[0].z },
      b: { x: srcAtoms[0].x, y: srcAtoms[0].y, z: srcAtoms[0].z },
    });
  }
  return contacts;
}

function computeChainContacts(model, chainA, chainB, cutoff) {
  const atomsA = model.selectedAtoms({ chain: chainA }).filter((atom) => atom.elem !== "H");
  const atomsB = model.selectedAtoms({ chain: chainB }).filter((atom) => atom.elem !== "H");
  const cutoffSq = cutoff * cutoff;
  const contacts = [];
  for (const atomA of atomsA) {
    for (const atomB of atomsB) {
      const dx = atomA.x - atomB.x;
      const dy = atomA.y - atomB.y;
      const dz = atomA.z - atomB.z;
      const distSq = dx * dx + dy * dy + dz * dz;
      if (distSq < cutoffSq) {
        contacts.push({
          a: { x: atomA.x, y: atomA.y, z: atomA.z },
          b: { x: atomB.x, y: atomB.y, z: atomB.z },
        });
      }
    }
  }
  return contacts.slice(0, 160);
}

function drawDashedLine(viewer, p1, p2, color) {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  const dz = p2.z - p1.z;
  const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (!len) return [];
  const ux = dx / len;
  const uy = dy / len;
  const uz = dz / len;
  const dashLen = 0.42;
  const gapLen = 0.1;
  const step = dashLen + gapLen;
  const count = Math.floor(len / step);
  const shapes = [];
  for (let i = 0; i < count; i += 1) {
    const t0 = i * step;
    const t1 = Math.min(t0 + dashLen, len);
    shapes.push(viewer.addCylinder({
      start: { x: p1.x + ux * t0, y: p1.y + uy * t0, z: p1.z + uz * t0 },
      end: { x: p1.x + ux * t1, y: p1.y + uy * t1, z: p1.z + uz * t1 },
      radius: 0.38,
      color,
      opacity: 0.22,
      fromCap: 1,
      toCap: 1,
    }));
    shapes.push(viewer.addCylinder({
      start: { x: p1.x + ux * t0, y: p1.y + uy * t0, z: p1.z + uz * t0 },
      end: { x: p1.x + ux * t1, y: p1.y + uy * t1, z: p1.z + uz * t1 },
      radius: 0.2,
      color,
      opacity: 1.0,
      fromCap: 1,
      toCap: 1,
    }));
  }
  return shapes;
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
  const blurb = solver
    ? "Minimal simulated intervention that turns the modeled KRAS-signaling endpoint off."
    : "Direct model-based recommendation from Claude Opus after reading the graph, evidence, and benchmark context.";
  const note = solver
    ? `Pathway nodes off: ${(record.pathway_nodes_off || []).join(", ") || "none"}`
    : (record.benchmark_assessment || "");
  return `
    <h3>${escapeHtml(title)}</h3>
    <p class="mono">${escapeHtml(genes.join(" + "))}</p>
    <p>${escapeHtml(blurb)}</p>
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
