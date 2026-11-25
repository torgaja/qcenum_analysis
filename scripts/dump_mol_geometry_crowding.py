#!/usr/bin/env python3
"""
dump_mol_geometry_crowding.py

Dumps full geometry (all atoms, elements, coordinates in Angstrom) and the
complete critical-pair list (covalent-radius-relative criterion, same as
crowding_detector.py: r_ij < factor * (r_cov_i + r_cov_j), default
factor=0.5) for a single molecule, plus a connected-components analysis of
the critical-pair graph -- to distinguish a single "knot" (all crowded atoms
mutually close, one connected blob) from a "chain" (A-B-C-D where A and D
are not directly critical to each other, just linked transitively).

Usage:
    python dump_mol_geometry_crowding.py --mol-id mol_002876
"""

import argparse
import json
from pathlib import Path

import numpy as np

BOHR_TO_ANGSTROM = 0.529177210903
DEFAULT_DATASET_ROOT = Path(".")
DEFAULT_DATASET = "dataset_2node"

COVALENT_RADII_A = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05,
    "Cl": 1.02, "Ar": 1.06,
}
DEFAULT_COVALENT_RADIUS_A = 0.75


def covalent_radius(elem):
    return COVALENT_RADII_A.get(elem, DEFAULT_COVALENT_RADIUS_A)


def load_molecule_geometry(mol_dir: Path):
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
        elements = elements_fallback if len(elements_fallback) == n_atoms else ["?"] * n_atoms
        chosen = (elements, coords_A)
        chosen_calc_name = calc_name
        break

    if chosen is None:
        return None
    elements, coords_A = chosen
    return {"mol_id": mol_id, "elements": elements, "coords_A": coords_A, "chosen_calc": chosen_calc_name,
            "n_calcs_available": len(avail_calcs)}


def find_critical_neighbors(elements, coords_A, factor):
    n = len(elements)
    radii = np.array([covalent_radius(e) for e in elements])
    diff = coords_A[:, None, :] - coords_A[None, :, :]
    dmat = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    sum_radii = radii[:, None] + radii[None, :]
    cutoff = factor * sum_radii
    np.fill_diagonal(dmat, np.inf)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if dmat[i, j] < cutoff[i, j]:
                ratio = float(dmat[i, j] / sum_radii[i, j])
                pairs.append((i, j, float(dmat[i, j]), ratio))
    return pairs, dmat


def connected_components(n, pairs):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j, _, _ in pairs:
        union(i, j)

    comps = {}
    for i in range(n):
        r = find(i)
        comps.setdefault(r, []).append(i)
    return [sorted(v) for v in comps.values() if len(v) > 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--mol-id", required=True)
    p.add_argument("--factor", type=float, default=0.5)
    args = p.parse_args()

    mol_dir = args.dataset_root / args.dataset / args.mol_id
    geo = load_molecule_geometry(mol_dir)
    if geo is None:
        print(f"COULD NOT LOAD GEOMETRY for {args.mol_id} in {mol_dir}")
        return

    elements = geo["elements"]
    coords_A = geo["coords_A"]
    n = len(elements)

    print("=====GEOMETRY_START=====")
    print(f"mol_id={geo['mol_id']}  n_atoms={n}  chosen_calc={geo['chosen_calc']}  "
          f"n_calcs_available={geo['n_calcs_available']}")
    print()
    print("Full geometry (Angstrom):")
    for i, (elem, xyz) in enumerate(zip(elements, coords_A)):
        print(f"  [{i:2d}] {elem:2s}  {xyz[0]: .6f} {xyz[1]: .6f} {xyz[2]: .6f}")

    pairs, dmat = find_critical_neighbors(elements, coords_A, args.factor)
    print()
    print(f"Critical pairs (r_ij < {args.factor} * (r_cov_i+r_cov_j)), n={len(pairs)}:")
    for i, j, r_ij, ratio in sorted(pairs, key=lambda x: x[3]):
        print(f"  ({i:2d}){elements[i]:2s} -- ({j:2d}){elements[j]:2s}  r_ij_A={r_ij:.4f}  ratio={ratio:.4f}")

    comps = connected_components(n, pairs)
    print()
    print(f"Connected components in critical-pair graph (size>1): {len(comps)}")
    for comp in comps:
        elems_in_comp = [elements[i] for i in comp]
        print(f"  component atoms={comp}  elements={elems_in_comp}  size={len(comp)}")
        # print all pairwise distances within this component (not just critical ones)
        # to see if it's a tight knot (all close) or a spread chain (only neighbors close)
        print("    full pairwise distance matrix within component (Angstrom):")
        header = "        " + "".join(f"{f'[{i}]{elements[i]}':>10s}" for i in comp)
        print(header)
        for i in comp:
            row = f"    [{i}]{elements[i]:2s}"
            for j in comp:
                row += f"{dmat[i, j]:10.4f}" if i != j else f"{'--':>10s}"
            print(row)

    # Classify: knot = every pair within the component is ALSO critical (clique);
    # chain = component is critical-connected but NOT a clique (some pairs far apart)
    print()
    for comp in comps:
        comp_set = set(comp)
        critical_pairs_in_comp = set()
        for i, j, r_ij, ratio in pairs:
            if i in comp_set and j in comp_set:
                critical_pairs_in_comp.add((i, j))
        n_possible = len(comp) * (len(comp) - 1) // 2
        n_actual = len(critical_pairs_in_comp)
        topology = "KNOT (clique - all mutually critical)" if n_actual == n_possible else \
                   "CHAIN/SPARSE (not all pairs critical - transitively linked only)"
        print(f"  Component {comp}: {n_actual}/{n_possible} possible critical pairs present -> {topology}")

    print("=====GEOMETRY_END=====")


if __name__ == "__main__":
    main()
