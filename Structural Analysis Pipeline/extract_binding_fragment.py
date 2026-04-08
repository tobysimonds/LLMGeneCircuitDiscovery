"""
Extract the 20-residue EFS fragment that binds SRC most tightly,
then write it as a ProteinMPNN-ready PDB (backbone only: N, CA, C, O + CB).

Scoring: sliding-window sum of 1/d^2  over all EFS-SRC contacts,
so shorter distances contribute exponentially more to the score.

Output files
  efs_binding_fragment.pdb   — backbone PDB for ProteinMPNN
  fragment_contacts.csv      — contacts that fall inside the chosen window
  fragment_report.txt        — human-readable summary
"""

import csv, sys
from collections import defaultdict
from Bio import PDB

CIF_FILE   = "fold_efs_and_src_model_1.cif"
CSV_FILE   = "interface_contacts.csv"
FRAG_LEN   = 20
OUT_PDB    = "efs_binding_fragment.pdb"
OUT_CSV    = "fragment_contacts.csv"
OUT_REPORT = "fragment_report.txt"

BACKBONE   = {"N", "CA", "C", "O", "CB"}   # atoms kept for ProteinMPNN

# ── 1. Build per-residue contact score (1/d²) ────────────────────────
res_score   = defaultdict(float)   # resnum_A → weighted score
res_contacts= defaultdict(list)    # resnum_A → list of contact rows

with open(CSV_FILE) as f:
    for row in csv.DictReader(f):
        rnum = int(row["resnum_A"])
        d    = float(row["min_distance_A"])
        res_score[rnum]    += 1.0 / (d * d)
        res_contacts[rnum].append(row)

# ── 2. Load structure, get EFS residue list ───────────────────────────
parser    = PDB.MMCIFParser(QUIET=True)
structure = parser.get_structure("complex", CIF_FILE)
model     = structure[0]
chain_a   = model["A"]

efs_residues = sorted(
    [r for r in chain_a.get_residues() if r.id[0] == " "],
    key=lambda r: r.id[1]
)
res_nums = [r.id[1] for r in efs_residues]

# ── 3. Sliding-window search for best 20-residue segment ─────────────
best_score  = -1
best_start  = 0

for i in range(len(res_nums) - FRAG_LEN + 1):
    window = res_nums[i : i + FRAG_LEN]
    score  = sum(res_score.get(r, 0.0) for r in window)
    if score > best_score:
        best_score = score
        best_start = i

window_nums  = set(res_nums[best_start : best_start + FRAG_LEN])
window_list  = res_nums[best_start : best_start + FRAG_LEN]
frag_residues = efs_residues[best_start : best_start + FRAG_LEN]
print(f"Best window  : residues {window_list[0]}–{window_list[-1]}")
print(f"Window score : {best_score:.4f}")

# ── 4. Collect contacts that fall inside the window ───────────────────
frag_contacts = []
for rnum in window_list:
    frag_contacts.extend(res_contacts.get(rnum, []))
frag_contacts.sort(key=lambda r: float(r["min_distance_A"]))

with open(OUT_CSV, "w", newline="") as f:
    fieldnames = list(frag_contacts[0].keys()) if frag_contacts else []
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader(); w.writerows(frag_contacts)

print(f"Fragment contacts : {len(frag_contacts)}")

# ── 5. Write backbone PDB ─────────────────────────────────────────────
# Build a clean standalone structure (avoids ID collision from in-place renumber)
new_struct  = PDB.Structure.Structure("frag")
new_model   = PDB.Model.Model(0)
new_chain   = PDB.Chain.Chain("A")
new_struct.add(new_model)
new_model.add(new_chain)

for new_num, res in enumerate(frag_residues, start=1):
    new_res = PDB.Residue.Residue((" ", new_num, " "), res.resname, res.segid)
    for atom in res.get_atoms():
        if atom.get_name().strip() in BACKBONE:
            new_atom = PDB.Atom.Atom(
                atom.name, atom.coord.copy(), atom.bfactor,
                atom.occupancy, atom.altloc, atom.fullname,
                atom.serial_number, atom.element
            )
            new_res.add(new_atom)
    new_chain.add(new_res)

io = PDB.PDBIO()
io.set_structure(new_struct)
io.save(OUT_PDB)
print(f"PDB written   : {OUT_PDB}")

# ── 6. Summary report ────────────────────────────────────────────────
aa3 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}

seq   = "".join(aa3.get(r.resname, "X") for r in frag_residues)
lines = [
    "=" * 60,
    "  EFS BINDING FRAGMENT — ProteinMPNN Input Summary",
    "=" * 60,
    f"  Original residues : {window_list[0]} – {window_list[-1]} (EFS Chain A)",
    f"  Length            : {FRAG_LEN} amino acids",
    f"  Binding score     : {best_score:.4f}  (Σ 1/d²)",
    f"  Total contacts    : {len(frag_contacts)}  (with SRC, < 5.0 Å)",
    f"  Sequence (1-letter): {seq}",
    "",
    "  Residue breakdown:",
]
for rnum, res in zip(window_list, frag_residues):
    sc = res_score.get(rnum, 0.0)
    nc = len(res_contacts.get(rnum, []))
    marker = " ◀ KEY" if sc > 0.05 else ""
    lines.append(f"    {rnum:>4}  {res.resname}  score={sc:7.4f}  contacts={nc}{marker}")

lines += [
    "",
    "  Top 10 closest contacts in fragment:",
    f"  {'ResA':>5} {'AA_A':>5} {'AtomA':>5}  {'ResB':>5} {'AA_B':>5} {'AtomB':>5}  {'Dist(Å)':>8}",
    "  " + "-" * 50,
]
for c in frag_contacts[:10]:
    lines.append(
        f"  {c['resnum_A']:>5} {c['resname_A']:>5} {c['atom_A']:>5}  "
        f"{c['resnum_B']:>5} {c['resname_B']:>5} {c['atom_B']:>5}  "
        f"{float(c['min_distance_A']):>8.3f}"
    )
lines += [
    "",
    "  ProteinMPNN command (example):",
    "    python protein_mpnn_run.py \\",
    f"      --pdb_path {OUT_PDB} \\",
    "      --out_folder ./mpnn_output \\",
    "      --num_seq_per_target 8 \\",
    "      --sampling_temp 0.1",
    "=" * 60,
]

report = "\n".join(lines)
print("\n" + report)
with open(OUT_REPORT, "w") as f:
    f.write(report + "\n")
print(f"\nReport saved  : {OUT_REPORT}")
