#!/usr/bin/env python3
"""
nn_force_distance_v2.py

Fixed version of nn_force_distance.py, now with pair_key / is_mutual_nn columns.

The original version computed the nearest-neighbor pair axis (r_hat_ij) from
metadata.json's representative_structure coordinates, but read forces
(F_i, F_j) from each calculation's own .npz file (step_0_cartesian_forces).
QM programs commonly reorient the molecule into their own internal Cartesian
frame before running, so representative_structure and a given calc's npz
frame can differ by an arbitrary rotation -- this was confirmed directly for
mol_000001 (dataset_2node): metadata direction (Li1->Li2) normalizes to
(0.769, 0.466, 0.438), while the npz direction for the same pair is exactly
(0, 0, 1). Dotting a metadata-frame r_hat with an npz-frame force is
meaningless (wrong magnitude, and for a general polyatomic, potentially wrong
sign).

Fix: for each calculation, read geometry AND forces from the SAME .npz file
(step_0_current_cartesian_coordinates, step_0_cartesian_forces), so they are
guaranteed to be in the same frame. Coordinates in the npz are in Bohr; r_ij
is converted to Angstrom only for reporting/plotting. Since r_hat_ij is a
dimensionless unit vector, the Bohr-vs-Angstrom choice does not affect
f_rel/f_common themselves, only the reported r_ij column.

For every atom i (per molecule, per available calculation), finds the single
nearest-neighbor atom j (within that calc's own geometry), and computes:

    r_hat_ij = (coords[i] - coords[j]) / |coords[i] - coords[j]|   (points j -> i)

    f_rel    = (F_i - F_j) . r_hat_ij / 2   -- genuine two-body radial
                                                repulsion/attraction component
                                                (symmetric under i<->j swap)
    f_common = (F_i + F_j) . r_hat_ij / 2   -- "common"/many-body/external
                                                component acting on the pair
                                                as a whole (ANTI-symmetric
                                                under i<->j swap, since
                                                r_hat_ji = -r_hat_ij; large
                                                |f_common| relative to |f_rel|
                                                suggests the nearest neighbor
                                                is not the dominant interaction)

Positive f_rel means the pair is being pushed apart (repulsive), negative
means pulled together (attractive), under this sign convention -- assuming
step_0_cartesian_forces is truly a force, not a stored gradient (-F); this
cannot be settled from near-equilibrium data alone (see TODO below).

Output: one row per (molecule, calc, atom_i):

    mol_id, calc, atom_i, atom_j, elem_i, elem_j, r_ij_A, f_rel, f_common,
    pair_key, is_mutual_nn

  - pair_key: "min(i,j)_max(i,j)", identical for the atom_i->atom_j and
    atom_j->atom_i rows of the same physical contact, so pair-centric
    (deduplicated) statistics can be derived from this atom-centric CSV by
    e.g. `df.drop_duplicates("pair_key")` restricted to is_mutual_nn rows,
    or by picking one row per pair_key deterministically.
  - is_mutual_nn: True if j's own nearest neighbor is also i (mutual nearest
    neighbor pair). For a mutual pair, the CSV contains BOTH the (i,j) and
    (j,i) rows, which have identical f_rel but OPPOSITE-SIGN f_common (since
    f_common flips sign under the swap) -- naively averaging/pooling
    f_common over all rows will therefore be biased toward zero regardless
    of true many-body contamination. For a one-sided (non-mutual) nearest
    neighbor, only one row exists for that directed relationship.

TODO: currently only the single nearest-neighbor pair per atom is considered.
Could be extended to all pairs under a distance cutoff (e.g. < 3-4 Angstrom)
to check whether the near-equilibrium/repulsive-wall picture holds beyond
just the closest contact.

TODO: this only tells us about frame consistency and two-body/many-body
decomposition, not whether step_0_cartesian_forces is truly a force (F) or
a stored gradient (-F) -- near-equilibrium structures can't distinguish the
two (both ~0). Sign convention should be double-checked against a genuinely
strained (short r_ij) case.

Usage:
    python nn_force_distance_v2.py --dataset-root /path/to/scientific_data \
        --dataset dataset_2node \
        --out analysis_output/2node_new/nn_force_distance_v2.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

BOHR_TO_ANGSTROM = 0.529177210903


def nearest_neighbor_pairs(coords: np.ndarray):
    """For each atom i, return (j, distance) of its single nearest neighbor."""
    n = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dmat = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    np.fill_diagonal(dmat, np.inf)
    nn_j = np.argmin(dmat, axis=1)
    nn_d = dmat[np.arange(n), nn_j]
    return nn_j, nn_d


def process_molecule(mol_dir: Path, skip_counts: dict):
    """Return list of row-dicts for this molecule (one per calc x atom)."""
    rows = []
    metadata_file = mol_dir / "metadata.json"
    if not metadata_file.exists():
        skip_counts["no_metadata"] += 1
        return rows

    with open(metadata_file, encoding="utf-8") as fh:
        metadata = json.load(fh)

    mol_id = metadata.get("mol_id", mol_dir.name)
    avail_calcs = metadata.get("available_calculations", {})

    rep = metadata.get("representative_structure", {})
    elements_fallback = rep.get("coordinate_elements", [])

    for calc_name, calc_info in avail_calcs.items():
        npz_path = mol_dir / calc_info.get("npz_file", "")
        if not npz_path.exists():
            skip_counts["missing_npz"] += 1
            continue
        try:
            data = np.load(npz_path, allow_pickle=False)
        except Exception:
            skip_counts["npz_load_error"] += 1
            continue

        if "step_0_cartesian_forces" not in data:
            skip_counts["no_force_key"] += 1
            continue
        if "step_0_current_cartesian_coordinates" not in data:
            skip_counts["no_coord_key"] += 1
            continue

        if "step_0_atomic_numbers" in data:
            n_atoms = len(np.array(data["step_0_atomic_numbers"]).flatten())
        else:
            n_atoms = len(elements_fallback)
        if n_atoms < 2:
            skip_counts["too_few_atoms"] += 1
            continue

        coords_raw = np.array(data["step_0_current_cartesian_coordinates"], dtype=float)
        if coords_raw.ndim == 1:
            if coords_raw.size != n_atoms * 3:
                skip_counts["coord_shape_mismatch"] += 1
                continue
            coords_raw = coords_raw.reshape(n_atoms, 3)
        if coords_raw.shape[0] != n_atoms or not np.all(np.isfinite(coords_raw)):
            skip_counts["coord_invalid"] += 1
            continue

        f_raw = np.array(data["step_0_cartesian_forces"], dtype=float)
        if f_raw.ndim == 1:
            if f_raw.size != n_atoms * 3:
                skip_counts["force_shape_mismatch"] += 1
                continue
            f_raw = f_raw.reshape(n_atoms, 3)
        if f_raw.shape[0] != n_atoms or not np.all(np.isfinite(f_raw)):
            skip_counts["force_invalid"] += 1
            continue

        elements = elements_fallback if len(elements_fallback) == n_atoms else ["?"] * n_atoms

        nn_j, nn_d = nearest_neighbor_pairs(coords_raw)  # coords in Bohr

        for i in range(n_atoms):
            j = int(nn_j[i])
            r_ij_bohr = float(nn_d[i])
            if r_ij_bohr == 0.0 or not np.isfinite(r_ij_bohr):
                continue

            vec_ij = coords_raw[i] - coords_raw[j]   # points j -> i, in Bohr
            r_hat_ij = vec_ij / r_ij_bohr             # dimensionless, frame-consistent

            F_i = f_raw[i]
            F_j = f_raw[j]

            f_rel = float(np.dot(F_i - F_j, r_hat_ij) / 2.0)
            f_common = float(np.dot(F_i + F_j, r_hat_ij) / 2.0)

            r_ij_A = r_ij_bohr * BOHR_TO_ANGSTROM

            pair_key = f"{min(i, j)}_{max(i, j)}"
            is_mutual_nn = bool(int(nn_j[j]) == i)

            rows.append({
                "mol_id": mol_id,
                "calc": calc_name,
                "atom_i": i,
                "atom_j": j,
                "elem_i": elements[i] if i < len(elements) else "?",
                "elem_j": elements[j] if j < len(elements) else "?",
                "r_ij_A": f"{r_ij_A:.6f}",
                "f_rel": f"{f_rel:.6f}",
                "f_common": f"{f_common:.6f}",
                "pair_key": pair_key,
                "is_mutual_nn": is_mutual_nn,
            })

    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset-root", type=Path, required=True,
                    help="Directory containing the dataset subset directory")
    p.add_argument("--dataset", default="dataset_2node",
                    help="Dataset subset directory name")
    p.add_argument("--out", type=Path,
                    default=Path("analysis_output/2node_new/nn_force_distance_v2.csv"),
                    help="Output CSV path, relative to --dataset-root unless absolute")
    p.add_argument("--progress-every", type=int, default=200,
                    help="Print a progress line every N molecules")
    args = p.parse_args()

    dataset_dir = args.dataset_root / args.dataset
    out_path = args.out if args.out.is_absolute() else (args.dataset_root / args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mol_dirs = sorted(dataset_dir.glob("mol_*"))
    print(f"Found {len(mol_dirs)} molecule directories under {dataset_dir}")

    fieldnames = ["mol_id", "calc", "atom_i", "atom_j", "elem_i", "elem_j",
                  "r_ij_A", "f_rel", "f_common", "pair_key", "is_mutual_nn"]

    skip_counts = {
        "no_metadata": 0, "missing_npz": 0, "npz_load_error": 0,
        "no_force_key": 0, "no_coord_key": 0, "too_few_atoms": 0,
        "coord_shape_mismatch": 0, "coord_invalid": 0,
        "force_shape_mismatch": 0, "force_invalid": 0,
    }

    n_rows = 0
    n_mols_with_rows = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, mol_dir in enumerate(mol_dirs, start=1):
            rows = process_molecule(mol_dir, skip_counts)
            if rows:
                writer.writerows(rows)
                n_rows += len(rows)
                n_mols_with_rows += 1
            if idx % args.progress_every == 0:
                print(f"  processed {idx}/{len(mol_dirs)} molecules, {n_rows} rows so far")

    print(f"DONE. molecules_processed={len(mol_dirs)} molecules_with_rows={n_mols_with_rows} "
          f"total_rows={n_rows}")
    print(f"SKIP_COUNTS: {skip_counts}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
