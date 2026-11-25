#!/usr/bin/env python3
"""
crowding_detector.py

Geometric "many-body crowding" detector for compressed molecular geometries,
built to distinguish two competing explanations for the very short (<0.2 A)
nearest-neighbor contacts found by sign_alignment_check.py:

  (A) UFF atom-typer failure: UFF could not assign a sensible type/valence to
      an atom (e.g. a hypervalent S or P center), producing a geometry where
      several atoms around that center are crushed together simultaneously
      (a genuine many-body pileup, not a simple pair squeeze).
  (B) 50-step optimization cap: UFF was run for only 50 steps (per the
      generation pipeline: RDKit graph -> rough embedded geometry -> UFF
      optimize for 50 steps -> SCF x3, keep only SCF-converged structures).
      A hard step cap can simply leave an ordinary pairwise clash unrelieved,
      with no atom-typer problem at all.

Neither hypothesis can be tested with a single nearest-neighbor distance
(as sign_alignment_check.py uses) -- both a clean two-body squeeze and a
crowded many-body pileup can produce a very short "nearest neighbor"
distance. What differs is how many *other* atoms are also critically close
to the same center. This script generalizes from "1 nearest neighbor" to
"all critically-close neighbors" per atom, using a covalent-radius-RELATIVE
cutoff (not an absolute one like 0.2/0.3 A), because pathological compression
looks very different in absolute terms depending on the atom pair: e.g. in
one real example from this dataset, H-H contacts of ~0.14-0.22 A (Cov(H)=
0.31 A each) and S-H contacts of ~0.46-0.57 A (Cov(S)=1.05, Cov(H)=0.31, sum
0.5*1.36=0.68 A) are BOTH part of the same crowded cluster (an S center with
3 H's + a charge, all crushed together) -- a fixed absolute cutoff of e.g.
0.3 A would catch the H-H pairs but completely miss the S-H compression,
because 0.46-0.57 A looks "long" in absolute terms while still being ~34% of
a normal S-H bond (~1.34 A).

Definition used here, per atom i (within one molecule's geometry):
  - r_cov_i, r_cov_j: covalent radii (A) of atoms i, j (Cordero-style table).
  - critical neighbor: any j != i with r_ij < f * (r_cov_i + r_cov_j), where
    f defaults to 0.5 (i.e. the pair is compressed to under half the sum of
    covalent radii -- roughly comparable to, or tighter than, a very strong
    single bond; f is configurable via --factor).
  - n_critical(i): number of critical neighbors of atom i.

Per-molecule classification (geometry is IDENTICAL across all QM methods/
basis sets for a given molecule in this pipeline -- confirmed empirically:
the suspicious short contacts in nn_force_distance_v2.csv / sign_alignment_
check.py showed exactly the same r_ij across B3LYP/HF/WB97X-D for the same
mol_id). So crowding is a property of the GEOMETRY, computed and reported
ONCE per molecule, not once per (molecule, calc) row -- avoiding the ~9x
duplication that would otherwise occur (one row per available calc).

  - none            : no atom in the molecule has any critical neighbor.
  - clean_pair       : the critical-neighbor graph consists only of isolated
                        pairs, i.e. every atom with n_critical(i) >= 1 has
                        exactly 1 critical neighbor, and that neighbor's only
                        critical neighbor is likewise just it (a true 2-body
                        squeeze, potentially a useful strained ML sample).
  - crowded_cluster  : at least one atom has n_critical(i) >= 2, OR a pair's
                        partner has additional critical neighbors of its own
                        (a many-body pileup, more consistent with a UFF
                        typer failure than a simple unrelieved pair clash).

Per-molecule output also includes: n_atoms, elements involved in crowding,
and n_atoms/elements are exactly what's needed to start discriminating
hypothesis (A) vs (B) after the fact:
  - (A) typer failure  -> expect clustering on specific elements/valences
        (e.g. hypervalent S/P, or other elements UFF's typer handles badly),
        largely independent of overall molecule size.
  - (B) step-cap        -> expect correlation with molecule size/complexity
        (more atoms -> more clashes to resolve in the same 50 steps) rather
        than specific elements.
This script only measures the geometric fact of crowding; it does NOT decide
between (A) and (B) -- that requires cross-tabulating this output against
molecule size and element composition (and, if available, UFF logs/energies
at step 50), which is a natural follow-up once this CSV exists.

IMPORTANT framing: all structures examined here are SCF-converged (the
pipeline discards molecules where SCF did not converge in any of the 3
independent runs). So even a "crowded_cluster" structure carries a genuine
QM force on a physically extreme geometry -- this is not a numerical error,
it is a DATA CURATION / LABELING question (keep as a hard/strained training
sample vs. exclude as unrepresentative), and should be handled as a label,
not a silent drop, so the decision stays reproducible.

Usage:
    python crowding_detector.py [--dataset-root PATH] [--dataset NAME]
                                 [--factor 0.5] [--out crowding.csv]
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

BOHR_TO_ANGSTROM = 0.529177210903

DEFAULT_DATASET_ROOT = Path(".")
DEFAULT_DATASET = "dataset_2node"

# Cordero et al. (2008)-style single-bond covalent radii, Angstrom.
# Only elements expected in this dataset's periodic-table range are included;
# unknown elements fall back to a generic 0.75 A rather than crashing.
COVALENT_RADII_A = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05,
    "Cl": 1.02, "Ar": 1.06,
}
DEFAULT_COVALENT_RADIUS_A = 0.75


def covalent_radius(elem: str) -> float:
    return COVALENT_RADII_A.get(elem, DEFAULT_COVALENT_RADIUS_A)


def load_molecule_geometry(mol_dir: Path):
    """Return (mol_id, elements, coords_A) for the FIRST available calc's
    geometry in this molecule, or None if nothing usable is found.

    Geometry is assumed identical across all available calcs for a given
    molecule in this pipeline (verified empirically -- see module docstring);
    this function additionally cross-checks that assumption against a
    second available calc, if any, and reports a warning flag rather than
    silently trusting it.
    """
    metadata_file = mol_dir / "metadata.json"
    if not metadata_file.exists():
        return None
    with open(metadata_file, encoding="utf-8") as fh:
        metadata = json.load(fh)

    mol_id = metadata.get("mol_id", mol_dir.name)
    avail_calcs = metadata.get("available_calculations", {})
    rep = metadata.get("representative_structure", {})
    elements_fallback = rep.get("coordinate_elements", [])

    chosen = None
    chosen_calc_name = None
    all_calc_coords = {}

    for calc_name, calc_info in avail_calcs.items():
        npz_path = mol_dir / calc_info.get("npz_file", "")
        if not npz_path.exists():
            continue
        try:
            data = np.load(npz_path, allow_pickle=False)
        except Exception:
            continue
        if "step_0_current_cartesian_coordinates" not in data:
            continue

        if "step_0_atomic_numbers" in data:
            n_atoms = len(np.array(data["step_0_atomic_numbers"]).flatten())
        else:
            n_atoms = len(elements_fallback)
        if n_atoms < 2:
            continue

        coords_raw = np.array(data["step_0_current_cartesian_coordinates"], dtype=float)
        if coords_raw.ndim == 1:
            if coords_raw.size != n_atoms * 3:
                continue
            coords_raw = coords_raw.reshape(n_atoms, 3)
        if coords_raw.shape[0] != n_atoms or not np.all(np.isfinite(coords_raw)):
            continue

        coords_A = coords_raw * BOHR_TO_ANGSTROM
        all_calc_coords[calc_name] = coords_A

        if chosen is None:
            elements = elements_fallback if len(elements_fallback) == n_atoms else ["?"] * n_atoms
            chosen = (elements, coords_A)
            chosen_calc_name = calc_name

    if chosen is None:
        return None

    elements, coords_A = chosen

    # Cross-check: if a second calc is available, verify the geometry matches
    # (within a small numerical tolerance) -- this is a consistency check on
    # the "geometry is method-independent" assumption, not just an assertion.
    geometry_consistent = True
    n_calcs_checked = 0
    for calc_name, other_coords in all_calc_coords.items():
        if calc_name == chosen_calc_name:
            continue
        n_calcs_checked += 1
        if other_coords.shape != coords_A.shape:
            geometry_consistent = False
            continue
        if not np.allclose(other_coords, coords_A, atol=1e-4):
            geometry_consistent = False

    return {
        "mol_id": mol_id,
        "elements": elements,
        "coords_A": coords_A,
        "chosen_calc": chosen_calc_name,
        "n_calcs_available": len(all_calc_coords),
        "geometry_consistent_across_calcs": geometry_consistent if n_calcs_checked > 0 else None,
    }


def find_critical_neighbors(elements, coords_A, factor: float):
    """For each atom i, return the list of j != i with r_ij < factor *
    (r_cov_i + r_cov_j), as (j, r_ij_A, ratio) tuples, ratio = r_ij / (r_cov_i+r_cov_j).
    """
    n = len(elements)
    radii = np.array([covalent_radius(e) for e in elements])
    diff = coords_A[:, None, :] - coords_A[None, :, :]
    dmat = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    sum_radii = radii[:, None] + radii[None, :]
    cutoff = factor * sum_radii
    np.fill_diagonal(dmat, np.inf)

    critical = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dmat[i, j] < cutoff[i, j]:
                ratio = float(dmat[i, j] / sum_radii[i, j])
                critical[i].append((j, float(dmat[i, j]), ratio))
    return critical


def classify_molecule(critical):
    """Classify a molecule's crowding pattern from its per-atom critical-
    neighbor lists. Returns (label, n_atoms_involved, n_critical_pairs).
    """
    involved = [i for i, lst in enumerate(critical) if lst]
    if not involved:
        return "none", 0, 0

    n_critical_pairs = sum(len(lst) for lst in critical) // 2  # each pair counted from both sides (usually)

    is_clean_pair = True
    for i in involved:
        if len(critical[i]) != 1:
            is_clean_pair = False
            break
        j = critical[i][0][0]
        # the partner must ALSO have exactly this one critical neighbor (i)
        if len(critical[j]) != 1 or critical[j][0][0] != i:
            is_clean_pair = False
            break

    label = "clean_pair" if is_clean_pair else "crowded_cluster"
    return label, len(involved), n_critical_pairs


def process_molecule(mol_dir: Path, factor: float):
    """Return a single result dict for this molecule, or None if unusable."""
    geo = load_molecule_geometry(mol_dir)
    if geo is None:
        return None

    elements = geo["elements"]
    coords_A = geo["coords_A"]
    n_atoms = len(elements)

    critical = find_critical_neighbors(elements, coords_A, factor)
    label, n_atoms_involved, n_critical_pairs = classify_molecule(critical)

    involved_elements = sorted({elements[i] for i, lst in enumerate(critical) if lst})

    # Most-compressed single contact, for quick eyeballing / cross-checking
    # against sign_alignment_check.py's SUSPICIOUS listing.
    min_ratio = None
    min_ratio_pair = None
    for i, lst in enumerate(critical):
        for j, r_ij, ratio in lst:
            if min_ratio is None or ratio < min_ratio:
                min_ratio = ratio
                min_ratio_pair = (i, j, elements[i], elements[j], r_ij)

    return {
        "mol_id": geo["mol_id"],
        "n_atoms": n_atoms,
        "chosen_calc": geo["chosen_calc"],
        "n_calcs_available": geo["n_calcs_available"],
        "geometry_consistent_across_calcs": geo["geometry_consistent_across_calcs"],
        "label": label,
        "n_atoms_involved": n_atoms_involved,
        "n_critical_pairs": n_critical_pairs,
        "involved_elements": ",".join(involved_elements),
        "min_ratio": min_ratio,
        "min_ratio_elem_i": min_ratio_pair[2] if min_ratio_pair else "",
        "min_ratio_elem_j": min_ratio_pair[3] if min_ratio_pair else "",
        "min_ratio_r_ij_A": min_ratio_pair[4] if min_ratio_pair else "",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                    help="Directory containing the dataset subset directory")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="Dataset subset directory name (e.g. dataset_2node, dataset_3node)")
    p.add_argument("--factor", type=float, default=0.5,
                    help="Critical-neighbor cutoff as a fraction of the sum of covalent radii")
    p.add_argument("--out", type=Path, default=None,
                    help="Output CSV path (relative to --dataset-root unless absolute); "
                         "default: analysis_output/<dataset>_new/crowding_detector.csv")
    p.add_argument("--progress-every", type=int, default=200)
    args = p.parse_args()

    dataset_dir = args.dataset_root / args.dataset
    if args.out is None:
        out_path = args.dataset_root / "analysis_output" / f"{args.dataset.replace('dataset_', '')}_new" / "crowding_detector.csv"
    else:
        out_path = args.out if args.out.is_absolute() else (args.dataset_root / args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mol_dirs = sorted(dataset_dir.glob("mol_*"))
    print(f"Scanning {len(mol_dirs)} molecules in {dataset_dir} "
          f"(factor={args.factor}, i.e. critical if r_ij < {args.factor} * (r_cov_i+r_cov_j)) ...")

    fieldnames = [
        "mol_id", "n_atoms", "chosen_calc", "n_calcs_available",
        "geometry_consistent_across_calcs", "label", "n_atoms_involved",
        "n_critical_pairs", "involved_elements", "min_ratio",
        "min_ratio_elem_i", "min_ratio_elem_j", "min_ratio_r_ij_A",
    ]

    n_processed = 0
    n_unusable = 0
    label_counts = {"none": 0, "clean_pair": 0, "crowded_cluster": 0}
    inconsistent_geometry_mols = []

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, mol_dir in enumerate(mol_dirs, start=1):
            result = process_molecule(mol_dir, args.factor)
            if result is None:
                n_unusable += 1
                continue
            writer.writerow(result)
            n_processed += 1
            label_counts[result["label"]] += 1
            if result["geometry_consistent_across_calcs"] is False:
                inconsistent_geometry_mols.append(result["mol_id"])

            if idx % args.progress_every == 0:
                print(f"  processed {idx}/{len(mol_dirs)} molecules")

    print()
    print(f"DONE. molecules_scanned={len(mol_dirs)} molecules_written={n_processed} "
          f"unusable={n_unusable}")
    print(f"LABEL_COUNTS: {label_counts}  "
          f"(fraction crowded_cluster among affected: "
          f"{label_counts['crowded_cluster'] / max(1, label_counts['clean_pair'] + label_counts['crowded_cluster']):.4f})")
    if inconsistent_geometry_mols:
        print(f"WARNING: {len(inconsistent_geometry_mols)} molecules had geometry that did NOT "
              f"match across available calcs (assumption violated): {inconsistent_geometry_mols[:20]}"
              f"{' ...' if len(inconsistent_geometry_mols) > 20 else ''}")
    else:
        print("Geometry-consistency cross-check: no mismatches found among molecules with >1 available calc.")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
