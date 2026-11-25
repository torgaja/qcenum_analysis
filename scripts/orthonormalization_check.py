#!/usr/bin/env python3
"""
orthonormalization_check.py

Inspects the raw step_0_orthonormalization_matrix and step_0_overlap_matrix
arrays stored in a molecule .npz archive.
"""
import argparse
from pathlib import Path
import os
import sys

import numpy as np


def validate_mol_id(mol_id: str) -> None:
    """Reject mol_ids with path-traversal characters."""
    for forbidden in (os.sep, "/", "\\", ".."):
        if forbidden in mol_id:
            raise ValueError(
                f"mol_id '{mol_id}' contains forbidden token '{forbidden}'."
            )
    if not mol_id:
        raise ValueError("mol_id must not be empty.")


def print_in_various_formats(matrix):
    print(matrix)


def parse_args():
    default_root = Path(".")
    p = argparse.ArgumentParser(
        description="Inspect stored orthonormalization and overlap matrices in a molecule .npz archive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mol_id", help="Molecule identifier, e.g. mol_097583", default='mol_000001')
    p.add_argument("--basis", help="basis e.g. STO-3G", default='STO-3G')
    p.add_argument("--dataset-root", type=Path, default=default_root,
                   help="Root dir containing dataset_2node / dataset_3node")
    p.add_argument("--subset", type=Path, default="dataset_2node",
                   help="subset: dataset_2node / dataset_3node")
    return p.parse_args()


def main():
    args = parse_args()
    mol_id = args.mol_id
    dataset_root = args.dataset_root
    subset = args.subset
    basis = args.basis

# --- validate mol_id ---
    try:
        validate_mol_id(mol_id)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    data_dir = dataset_root / subset / mol_id
    matches = sorted(p for p in data_dir.glob("*.npz") if p.name.endswith(f"_{basis}.npz"))
    if not matches:
        print(f"ERROR: No *_{basis}.npz file found in {data_dir}")
        sys.exit(1)
    data_file_path = matches[0]
    data = np.load(data_file_path)

    orthonormalization_matrix_plain = data['step_0_orthonormalization_matrix']
    overlap_matrix_plain = data['step_0_overlap_matrix']
    print(f'orthonorm: {orthonormalization_matrix_plain}')
    print(f'overlap: {overlap_matrix_plain}')

if __name__ == '__main__':
    main()