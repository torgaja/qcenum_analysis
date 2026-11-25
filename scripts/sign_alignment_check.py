#!/usr/bin/env python3
"""
sign_alignment_check.py

Focused sign/alignment check for compressed (short-distance) nearest-neighbor
contacts, re-reading geometry+forces directly from each calculation's own
.npz file (frame-consistent, same fix as nn_force_distance_v2.py).

For every atom i whose nearest-neighbor distance r_ij is below a threshold
(0.8 A and, more strictly, 0.5 A), computes:

  r_hat_ij      = (coords[i]-coords[j]) / |coords[i]-coords[j]|   (points j->i)
  f_rel         = (F_i - F_j) . r_hat_ij / 2      (two-body radial component)
  cos_theta_i   = (F_i . r_hat_ij) / |F_i|        (alignment of i's FULL force
                                                     vector with the pair axis;
                                                     +1 = purely radial,
                                                     pointing away from j)
  cos_theta_j   = (F_j . r_hat_ij) / |F_j|        (-1 expected if j's force is
                                                     purely radial, pointing
                                                     away from i)
  uniformity    = |F_i + F_j| / |F_i - F_j|       (-> 0 for an ideal, purely
                                                     two-body pair interaction,
                                                     where F_i = -F_j exactly)
  mag_asym      = |\u2016F_i\u2016-\u2016F_j\u2016| / (\u2016F_i\u2016+\u2016F_j\u2016)   (-> 0 for a clean pair
                                                     interaction; how much the
                                                     force *magnitudes* differ)

IMPORTANT -- diatomic vs polyatomic stratification:
n_atoms here is the TOTAL atom count from the .npz (step_0_atomic_numbers),
including hydrogens -- NOT the "node" (heavy-atom) count used to name the
dataset subsets (dataset_2node/dataset_3node can still contain molecules with
many more than 2/3 total atoms once H's are counted). For any TRUE 2-atom
(n_atoms==2) molecule, translational invariance forces F_i = -F_j exactly,
and rotational symmetry confines both forces to the molecular axis (there is
no other direction available) -- so for n_atoms==2, cos_theta_i = +1,
cos_theta_j = -1, uniformity = 0 and mag_asym = 0 EXACTLY, by symmetry, with
zero physical information about "how uniform the QM repulsion is". Pooling
2-atom rows together with >=3-atom rows would artificially push the alignment
statistics toward the "perfectly radial" extreme regardless of what the
>=3-atom (the only rows that actually test anything) data show. All summaries
below are therefore reported separately for n_atoms==2 and n_atoms>=3, and
only the n_atoms>=3 numbers should be used to judge alignment/uniformity.
A separate n_atoms histogram (see print_natoms_histogram) makes the actual
composition of the short-distance sample visible, instead of assuming it.

Pure, uniform two-body repulsion along the bond axis would give f_rel > 0,
cos_theta_i ~ +1, cos_theta_j ~ -1, uniformity ~ 0. Deviation indicates
non-radial (many-body / electronic-structure) contributions to the force on
that atom, even when the *radial component* itself is repulsive.

Assumes step_0_cartesian_forces is a true force (not a stored gradient -F);
if it were a gradient, all signs (f_rel and cos_theta) would flip, but
magnitudes (|cos_theta|, uniformity, mag_asym) would be unaffected -- so
those are reported explicitly as the convention-robust diagnostics.

Cross-check: the earlier nn_force_distance_v2.csv histogram over ALL
molecules (2-node, full run) showed 187 rows in [0, 0.5) A and 680 rows in
[0.5, 0.8) A, i.e. 867 rows with r_ij_A < 0.8. Since this script re-derives
nearest neighbors independently from the .npz files (not from that CSV), the
n reported here for the 0.8 A band should come out close to 867 (exactly
equal only if run against the exact same dataset/subset) -- a mismatch would
indicate an inconsistency between the two scripts worth investigating.

Usage:
    python sign_alignment_check.py [--dataset-root PATH] [--dataset NAME]
                                    [--r-cutoff 0.8] [--r-strict 0.5]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

BOHR_TO_ANGSTROM = 0.529177210903
DEFAULT_DATASET_ROOT = Path(".")
DEFAULT_DATASET = "dataset_2node"

# Forces below this magnitude (Ha/Bohr) are treated as numerically
# meaningless direction-wise (too permissive a guard like 1e-12 would
# normalize pure numerical noise into a "direction").
MIN_FORCE_NORM = 1e-6


def nearest_neighbor_pairs(coords: np.ndarray):
    n = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dmat = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    np.fill_diagonal(dmat, np.inf)
    nn_j = np.argmin(dmat, axis=1)
    nn_d = dmat[np.arange(n), nn_j]
    return nn_j, nn_d


def process_molecule(mol_dir: Path, r_cutoff_A: float):
    """Return list of dicts for atoms whose NN distance < r_cutoff_A."""
    rows = []
    metadata_file = mol_dir / "metadata.json"
    if not metadata_file.exists():
        return rows
    with open(metadata_file, encoding="utf-8") as fh:
        metadata = json.load(fh)

    mol_id = metadata.get("mol_id", mol_dir.name)
    avail_calcs = metadata.get("available_calculations", {})
    rep = metadata.get("representative_structure", {})
    elements_fallback = rep.get("coordinate_elements", [])

    r_cutoff_bohr = r_cutoff_A / BOHR_TO_ANGSTROM

    for calc_name, calc_info in avail_calcs.items():
        npz_path = mol_dir / calc_info.get("npz_file", "")
        if not npz_path.exists():
            continue
        try:
            data = np.load(npz_path, allow_pickle=False)
        except Exception:
            continue
        if "step_0_cartesian_forces" not in data or "step_0_current_cartesian_coordinates" not in data:
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

        f_raw = np.array(data["step_0_cartesian_forces"], dtype=float)
        if f_raw.ndim == 1:
            if f_raw.size != n_atoms * 3:
                continue
            f_raw = f_raw.reshape(n_atoms, 3)
        if f_raw.shape[0] != n_atoms or not np.all(np.isfinite(f_raw)):
            continue

        elements = elements_fallback if len(elements_fallback) == n_atoms else ["?"] * n_atoms

        nn_j, nn_d = nearest_neighbor_pairs(coords_raw)

        for i in range(n_atoms):
            j = int(nn_j[i])
            r_ij_bohr = float(nn_d[i])
            if r_ij_bohr == 0.0 or not np.isfinite(r_ij_bohr) or r_ij_bohr > r_cutoff_bohr:
                continue

            vec_ij = coords_raw[i] - coords_raw[j]
            r_hat_ij = vec_ij / r_ij_bohr

            F_i = f_raw[i]
            F_j = f_raw[j]
            norm_i = float(np.linalg.norm(F_i))
            norm_j = float(np.linalg.norm(F_j))

            f_rel = float(np.dot(F_i - F_j, r_hat_ij) / 2.0)

            cos_i = float(np.dot(F_i, r_hat_ij) / norm_i) if norm_i > MIN_FORCE_NORM else float("nan")
            cos_j = float(np.dot(F_j, r_hat_ij) / norm_j) if norm_j > MIN_FORCE_NORM else float("nan")

            diff_norm = float(np.linalg.norm(F_i - F_j))
            sum_norm = float(np.linalg.norm(F_i + F_j))
            uniformity = (sum_norm / diff_norm) if diff_norm > MIN_FORCE_NORM else float("nan")

            denom = norm_i + norm_j
            mag_asym = (abs(norm_i - norm_j) / denom) if denom > MIN_FORCE_NORM else float("nan")

            rows.append({
                "mol_id": mol_id,
                "calc": calc_name,
                "atom_i": i, "atom_j": j,
                "elem_i": elements[i] if i < len(elements) else "?",
                "elem_j": elements[j] if j < len(elements) else "?",
                "n_atoms": n_atoms,
                "r_ij_A": r_ij_bohr * BOHR_TO_ANGSTROM,
                "f_rel": f_rel,
                "norm_F_i": norm_i,
                "norm_F_j": norm_j,
                "cos_theta_i": cos_i,
                "cos_theta_j": cos_j,
                "uniformity": uniformity,
                "mag_asym": mag_asym,
            })

    return rows


def nanfrac(arr, cond_fn):
    """Fraction of non-NaN entries satisfying cond_fn, plus the NaN count."""
    arr = np.asarray(arr, dtype=float)
    valid = np.isfinite(arr)
    n_nan = int(np.sum(~valid))
    if not np.any(valid):
        return float("nan"), n_nan
    return float(np.mean(cond_fn(arr[valid]))), n_nan


def print_natoms_histogram(rows, label):
    """Print a histogram of n_atoms values in rows, so the true composition
    of a short-distance sample (real diatomics vs. H-decorated 2-node
    molecules that happen to have >2 total atoms) is visible, not assumed."""
    if not rows:
        print(f"[{label}] n_atoms histogram: no rows.")
        return
    counts = Counter(r["n_atoms"] for r in rows)
    total = len(rows)
    print(f"[{label}] n_atoms histogram (n={total}):")
    for n_atoms in sorted(counts):
        c = counts[n_atoms]
        print(f"  n_atoms={n_atoms:3d}: {c:6d}  ({100.0 * c / total:.2f}%)")
    print()


def summarize(rows, label):
    if not rows:
        print(f"[{label}] no rows.")
        return
    r_ij = np.array([r["r_ij_A"] for r in rows])
    f_rel = np.array([r["f_rel"] for r in rows])
    cos_i = np.array([r["cos_theta_i"] for r in rows])
    cos_j = np.array([r["cos_theta_j"] for r in rows])
    uniformity = np.array([r["uniformity"] for r in rows])
    mag_asym = np.array([r["mag_asym"] for r in rows])
    norm_i = np.array([r["norm_F_i"] for r in rows])
    norm_j = np.array([r["norm_F_j"] for r in rows])

    print(f"[{label}] n={len(rows)}")
    print(f"  r_ij_A        : min={r_ij.min():.4f} max={r_ij.max():.4f} mean={r_ij.mean():.4f}")
    print(f"  FRACTION_F_REL_POSITIVE_(REPULSIVE): {np.mean(f_rel > 0):.4f}")

    print(f"  |F_i|, |F_j|  : median|F_i|={np.median(norm_i):.4f} median|F_j|={np.median(norm_j):.4f} "
          f"(Ha/Bohr)")

    frac_i_09, nan_i = nanfrac(cos_i, lambda a: a > 0.9)
    frac_i_05, _ = nanfrac(cos_i, lambda a: a > 0.5)
    absfrac_i_09, _ = nanfrac(cos_i, lambda a: np.abs(a) > 0.9)
    print(f"  cos_theta_i   : mean={np.nanmean(cos_i):.4f} median={np.nanmedian(cos_i):.4f} "
          f"frac(>0.9)={frac_i_09:.4f} frac(>0.5)={frac_i_05:.4f} frac(|cos|>0.9)={absfrac_i_09:.4f} "
          f"n_nan={nan_i}")

    frac_j_09, nan_j = nanfrac(cos_j, lambda a: a < -0.9)
    frac_j_05, _ = nanfrac(cos_j, lambda a: a < -0.5)
    absfrac_j_09, _ = nanfrac(cos_j, lambda a: np.abs(a) > 0.9)
    print(f"  cos_theta_j   : mean={np.nanmean(cos_j):.4f} median={np.nanmedian(cos_j):.4f} "
          f"frac(<-0.9)={frac_j_09:.4f} frac(<-0.5)={frac_j_05:.4f} frac(|cos|>0.9)={absfrac_j_09:.4f} "
          f"n_nan={nan_j}")

    frac_uni_02, nan_u = nanfrac(uniformity, lambda a: a < 0.2)
    print(f"  uniformity |F_i+F_j|/|F_i-F_j| (->0 = clean pair interaction): "
          f"median={np.nanmedian(uniformity):.4f} mean={np.nanmean(uniformity):.4f} "
          f"frac(<0.2)={frac_uni_02:.4f} n_nan={nan_u}")

    frac_asym_01, nan_a = nanfrac(mag_asym, lambda a: a < 0.1)
    print(f"  mag_asym |Fi-Fj|/(Fi+Fj) (->0 = equal magnitudes): "
          f"median={np.nanmedian(mag_asym):.4f} mean={np.nanmean(mag_asym):.4f} "
          f"frac(<0.1)={frac_asym_01:.4f} n_nan={nan_a}")
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                    help="Directory containing the dataset subset directory")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="Dataset subset directory name (e.g. dataset_2node, dataset_3node)")
    p.add_argument("--r-cutoff", type=float, default=0.8)
    p.add_argument("--r-strict", type=float, default=0.5)
    p.add_argument("--suspicious", type=float, default=0.2,
                    help="print individual rows below this distance (A)")
    args = p.parse_args()

    dataset_dir = args.dataset_root / args.dataset
    mol_dirs = sorted(dataset_dir.glob("mol_*"))
    print(f"Scanning {len(mol_dirs)} molecules in {dataset_dir} "
          f"for nearest-neighbor contacts < {args.r_cutoff} A ...")

    all_rows = []
    for mol_dir in mol_dirs:
        all_rows.extend(process_molecule(mol_dir, args.r_cutoff))

    strict_rows = [r for r in all_rows if r["r_ij_A"] < args.r_strict]
    suspicious_rows = [r for r in all_rows if r["r_ij_A"] < args.suspicious]

    # n_atoms composition of the short-distance sample -- makes visible
    # (rather than assumed) how many true diatomics vs. larger (H-decorated)
    # molecules are contributing to each distance band.
    print_natoms_histogram(all_rows, f"r_ij < {args.r_cutoff} A, dataset={args.dataset}")
    print_natoms_histogram(strict_rows, f"r_ij < {args.r_strict} A, dataset={args.dataset}")

    diatomic_all = [r for r in all_rows if r["n_atoms"] == 2]
    poly_all = [r for r in all_rows if r["n_atoms"] >= 3]
    diatomic_strict = [r for r in strict_rows if r["n_atoms"] == 2]
    poly_strict = [r for r in strict_rows if r["n_atoms"] >= 3]

    print()
    print(f"NOTE: cross-check expectation for r_ij_A < 0.8 band (from prior full-run CSV "
          f"histogram on dataset_2node): n ~= 867 (187 in [0,0.5) + 680 in [0.5,0.8)). "
          f"Observed n here: {len(all_rows)} (dataset={args.dataset}).")
    print()
    print("=== 2-ATOM MOLECULES (cos_theta/uniformity/mag_asym are symmetry-forced to their")
    print("    ideal values here -- these numbers carry NO information about alignment quality,")
    print("    only f_rel and |F| are physically meaningful for this subset) ===")
    summarize(diatomic_all, f"n_atoms==2, r_ij < {args.r_cutoff} A")
    summarize(diatomic_strict, f"n_atoms==2, r_ij < {args.r_strict} A")

    print("=== >=3-ATOM MOLECULES (the only rows that actually test alignment/uniformity) ===")
    summarize(poly_all, f"n_atoms>=3, r_ij < {args.r_cutoff} A")
    summarize(poly_strict, f"n_atoms>=3, r_ij < {args.r_strict} A")

    print("=== ALL MOLECULES POOLED (for reference only -- biased toward the 2-atom symmetry")
    print("    extremes if diatomics dominate the sample; prefer the >=3-atom numbers above) ===")
    summarize(all_rows, f"ALL, r_ij < {args.r_cutoff} A")
    summarize(strict_rows, f"ALL, r_ij < {args.r_strict} A")

    if suspicious_rows:
        print(f"SUSPICIOUS rows (r_ij_A < {args.suspicious}):")
        for r in sorted(suspicious_rows, key=lambda x: x["r_ij_A"]):
            print(f"  {r['mol_id']} {r['calc']} n_atoms={r['n_atoms']} {r['elem_i']}-{r['elem_j']} "
                  f"r_ij_A={r['r_ij_A']:.4f} f_rel={r['f_rel']:.4f} "
                  f"cos_i={r['cos_theta_i']:.3f} cos_j={r['cos_theta_j']:.3f} "
                  f"uniformity={r['uniformity']:.3f} mag_asym={r['mag_asym']:.3f}")
    else:
        print(f"No rows below {args.suspicious} A -- no obviously pathological geometries found.")


if __name__ == "__main__":
    main()
