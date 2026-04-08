"""
Find interface contacts between Chain A (EFS) and Chain B (SRC)
in an AlphaFold 3 CIF file. Reports all residue pairs with any
atom-atom distance < 5.0 Angstroms.
"""

import csv
from itertools import product
from Bio import PDB

CIF_FILE = "fold_efs_and_src_model_1.cif"
OUTPUT_CSV = "interface_contacts.csv"
DISTANCE_CUTOFF = 5.0


def get_residues(chain):
    """Return standard amino acid residues (skip HETATM/water)."""
    return [r for r in chain.get_residues() if r.id[0] == " "]


def min_atom_distance(res_a, res_b):
    """Return the minimum atom-atom distance between two residues,
    along with the names of the closest atom pair."""
    min_dist = float("inf")
    closest_pair = ("", "")
    for atom_a, atom_b in product(res_a.get_atoms(), res_b.get_atoms()):
        dist = atom_a - atom_b
        if dist < min_dist:
            min_dist = dist
            closest_pair = (atom_a.name, atom_b.name)
    return min_dist, closest_pair


def main():
    parser = PDB.MMCIFParser(QUIET=True)
    structure = parser.get_structure("complex", CIF_FILE)

    model = structure[0]

    if "A" not in model or "B" not in model:
        available = [c.id for c in model.get_chains()]
        raise ValueError(f"Expected chains A and B, found: {available}")

    chain_a = model["A"]  # EFS
    chain_b = model["B"]  # SRC

    residues_a = get_residues(chain_a)
    residues_b = get_residues(chain_b)

    print(f"Chain A (EFS): {len(residues_a)} residues")
    print(f"Chain B (SRC): {len(residues_b)} residues")
    print(f"Scanning all residue pairs (cutoff = {DISTANCE_CUTOFF} Å)...\n")

    contacts = []

    for res_a in residues_a:
        for res_b in residues_b:
            dist, (atom_a, atom_b) = min_atom_distance(res_a, res_b)
            if dist < DISTANCE_CUTOFF:
                contacts.append({
                    "chain_A": "A",
                    "resnum_A": res_a.id[1],
                    "resname_A": res_a.resname,
                    "atom_A": atom_a,
                    "chain_B": "B",
                    "resnum_B": res_b.id[1],
                    "resname_B": res_b.resname,
                    "atom_B": atom_b,
                    "min_distance_A": round(dist, 3),
                })

    # Sort by distance
    contacts.sort(key=lambda x: x["min_distance_A"])

    # Print to terminal
    header = f"{'ResA':>6} {'AA_A':>5} {'AtomA':>5}  {'ResB':>6} {'AA_B':>5} {'AtomB':>5}  {'Dist(Å)':>8}"
    print(header)
    print("-" * len(header))
    for c in contacts:
        print(
            f"{c['resnum_A']:>6} {c['resname_A']:>5} {c['atom_A']:>5}  "
            f"{c['resnum_B']:>6} {c['resname_B']:>5} {c['atom_B']:>5}  "
            f"{c['min_distance_A']:>8.3f}"
        )

    print(f"\nTotal contacts found: {len(contacts)}")

    # Save to CSV
    fieldnames = [
        "chain_A", "resnum_A", "resname_A", "atom_A",
        "chain_B", "resnum_B", "resname_B", "atom_B",
        "min_distance_A",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(contacts)

    print(f"Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
