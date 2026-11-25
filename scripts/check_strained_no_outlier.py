#!/usr/bin/env python3
"""
check_strained_no_outlier.py

Cross-checks "strained (< threshold Ang) with NO force outlier (> 10 Ha/Bohr)"
molecules from analyze_dataset.py's overlap analysis. For each such molecule,
loads every available QM calculation's .npz file and reports the actual max
per-atom force magnitude, to distinguish between:
  - missing NPZ file (calculation never ran / wasn't saved)
  - present but empty/malformed force array (possible SCF failure)
  - present with a real (sub-threshold) force value (legitimate soft repulsion)

There is NO built-in default molecule list anymore -- you must always supply
the molecule ids explicitly, either directly or via a CSV file, so that this
never silently checks the wrong subset's molecules.

Usage:
    # Explicit ids
    python check_strained_no_outlier.py --dataset-root /path/to/datasets \\
        --dataset dataset_2node --mol-ids mol_000191 mol_000203 ...

    # From analyze_dataset.py's strained_no_force_outlier.csv
    python check_strained_no_outlier.py --dataset-root /path/to/datasets \\
        --dataset dataset_3node \\
        --from-csv analysis_output/3node_new/strained_no_force_outlier.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def min_pairwise_distance(coords):
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if n < 2:
        return float("inf"), 0, 0
    diff = coords[:, None, :] - coords[None, :, :]
    dmat = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    np.fill_diagonal(dmat, np.inf)
    flat = int(np.argmin(dmat))
    i, j = divmod(flat, n)
    return float(dmat[i, j]), i, j


def check_molecule(mol_dir: Path, elements):
    """Return list of dicts, one per available_calculation, with force diagnosis."""
    metadata_file = mol_dir / "metadata.json"
    rows = []
    if not metadata_file.exists():
        rows.append({"calc": None, "status": "NO_METADATA", "max_force": None})
        return rows

    with open(metadata_file, encoding="utf-8") as fh:
        metadata = json.load(fh)

    avail_calcs = metadata.get("available_calculations", {})
    if not avail_calcs:
        rows.append({"calc": None, "status": "NO_AVAILABLE_CALCULATIONS", "max_force": None})
        return rows

    for calc_name, calc_info in avail_calcs.items():
        npz_file = calc_info.get("npz_file", "")
        npz_path = mol_dir / npz_file

        if not npz_path.exists():
            rows.append({"calc": calc_name, "status": "MISSING_NPZ", "max_force": None})
            continue

        try:
            data = np.load(npz_path, allow_pickle=False)
        except Exception as e:
            rows.append({"calc": calc_name, "status": f"LOAD_ERROR: {e}", "max_force": None})
            continue

        if "step_0_cartesian_forces" not in data:
            keys = list(data.keys())
            rows.append({
                "calc": calc_name,
                "status": f"NO_FORCE_KEY (keys={keys})",
                "max_force": None,
            })
            continue

        qm_raw = np.array(data["step_0_cartesian_forces"], dtype=float)
        n_atoms = len(elements)
        expected = n_atoms * 3
        if qm_raw.ndim == 1:
            if qm_raw.size != expected:
                rows.append({
                    "calc": calc_name,
                    "status": f"SIZE_MISMATCH (got {qm_raw.size}, expected {expected})",
                    "max_force": None,
                })
                continue
            qm_raw = qm_raw.reshape(n_atoms, 3)

        if not np.all(np.isfinite(qm_raw)):
            rows.append({"calc": calc_name, "status": "NON_FINITE_FORCES", "max_force": None})
            continue

        mags = np.linalg.norm(qm_raw, axis=1)
        max_f = float(np.max(mags))
        max_atom = int(np.argmax(mags))
        elem = elements[max_atom] if max_atom < len(elements) else "?"
        rows.append({
            "calc": calc_name,
            "status": "OK",
            "max_force": max_f,
            "max_force_atom": f"{max_atom}:{elem}",
        })

    return rows


def find_mol_dir(mol_id: str, dataset_root: Path, dataset_name: str = None) -> Path:
    candidates = [dataset_name] if dataset_name else ["dataset_2node", "dataset_3node"]
    for subset in candidates:
        candidate = dataset_root / subset / mol_id
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"'{mol_id}' not found under {dataset_root}/{{{','.join(candidates)}}}")


def load_mol_ids_from_csv(csv_path: Path) -> list[str]:
    """Read the mol_id column from a CSV, auto-detecting the column name
    (looks for a header containing both 'mol' and 'id', case-insensitive;
    falls back to the first column if no such header is found)."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"CSV file has no header row: {csv_path}")
        id_col = None
        for cand in fieldnames:
            lc = cand.lower()
            if "mol" in lc and "id" in lc:
                id_col = cand
                break
        if id_col is None:
            id_col = fieldnames[0]
        rows = list(reader)
    return [r[id_col] for r in rows]


def main():
    default_root = Path(".")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset-root", type=Path, default=default_root,
                    help="Directory containing dataset_2node / dataset_3node")
    p.add_argument("--dataset", default=None,
                    help="Restrict search to this subset only, e.g. dataset_2node")

    id_group = p.add_mutually_exclusive_group(required=True)
    id_group.add_argument("--mol-ids", nargs="*", default=None,
                           help="Explicit mol_id list")
    id_group.add_argument("--from-csv", type=Path, default=None,
                           help="Path to a CSV file (e.g. analyze_dataset.py's "
                                "strained_no_force_outlier.csv) containing a mol_id column")

    args = p.parse_args()

    if args.mol_ids is not None:
        if len(args.mol_ids) == 0:
            p.error("--mol-ids was given but no ids followed it")
        mol_ids = args.mol_ids
    else:
        mol_ids = load_mol_ids_from_csv(args.from_csv)
        if not mol_ids:
            p.error(f"--from-csv {args.from_csv} produced zero mol_ids")

    print("=" * 78)
    print(f"  Checking {len(mol_ids)} 'strained, no force outlier' molecules")
    print("=" * 78)

    summary_counts = {"OK_small": 0, "MISSING_NPZ": 0, "OTHER_ISSUE": 0}

    for mol_id in mol_ids:
        try:
            mol_dir = find_mol_dir(mol_id, args.dataset_root, args.dataset)
        except FileNotFoundError as e:
            print(f"\n{mol_id}: ERROR - {e}")
            summary_counts["OTHER_ISSUE"] += 1
            continue

        with open(mol_dir / "metadata.json", encoding="utf-8") as fh:
            metadata = json.load(fh)
        rep = metadata.get("representative_structure", {})
        elements = rep.get("coordinate_elements", [])
        coords = rep.get("coordinates", [])
        min_dist, mi, mj = min_pairwise_distance(coords)

        print(f"\n{mol_id}  (min_dist={min_dist:.4f} Ang, atoms {mi}:{elements[mi] if mi < len(elements) else '?'}"
              f" -- {mj}:{elements[mj] if mj < len(elements) else '?'})")

        rows = check_molecule(mol_dir, elements)
        for row in rows:
            calc = row.get("calc") or "-"
            status = row["status"]
            if status == "OK":
                mf = row["max_force"]
                atom = row.get("max_force_atom", "?")
                flag = " <-- SUB-THRESHOLD BUT NONTRIVIAL" if 1.0 <= mf < 10.0 else ""
                print(f"    {calc:22s} max_force={mf:10.4f} Ha/Bohr  [atom {atom}]{flag}")
                summary_counts["OK_small"] += 1
            elif status == "MISSING_NPZ":
                print(f"    {calc:22s} MISSING_NPZ")
                summary_counts["MISSING_NPZ"] += 1
            else:
                print(f"    {calc:22s} {status}")
                summary_counts["OTHER_ISSUE"] += 1

    print()
    print("=" * 78)
    print("  Summary")
    print("=" * 78)
    print(f"  OK (force computed, sub-threshold) : {summary_counts['OK_small']}")
    print(f"  MISSING_NPZ                        : {summary_counts['MISSING_NPZ']}")
    print(f"  OTHER_ISSUE (size/key/parse errors): {summary_counts['OTHER_ISSUE']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
