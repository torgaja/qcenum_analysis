#!/usr/bin/env python3
"""
uff_force_validation.py

Validates stored molecular geometries using UFF force field:
  - Reports UFF energy and per-atom force magnitudes at the stored geometry
  - Compares with QM forces from dataset .npz files
  - Runs extra UFF optimization steps to diagnose convergence vs. oscillation

Usage:
    python uff_force_validation.py mol_097583
    python uff_force_validation.py mol_097583 --steps 200 --top-n 5 --verbose
    python uff_force_validation.py mol_097583 --dataset-root /path/to/datasets
"""

import argparse
import json
import os
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("ERROR: RDKit is required. Install via: conda install -c conda-forge rdkit")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HA_BOHR_LABEL = "Ha/Bohr"

_BOND_TYPE_MAP = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str, width: int = 70) -> None:
    print()
    print("-" * width)
    print(f"  {title}")
    print("-" * width)


def validate_mol_id(mol_id: str) -> None:
    """Reject mol_ids with path-traversal characters."""
    for forbidden in (os.sep, "/", "\\", ".."):
        if forbidden in mol_id:
            raise ValueError(
                f"mol_id '{mol_id}' contains forbidden token '{forbidden}'."
            )
    if not mol_id:
        raise ValueError("mol_id must not be empty.")


def find_mol_dir(mol_id: str, dataset_root: Path) -> Path:
    """Search dataset_2node and dataset_3node under dataset_root."""
    for subset in ("dataset_2node", "dataset_3node"):
        candidate = dataset_root / subset / mol_id
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"'{mol_id}' not found under {dataset_root}/dataset_2node "
        f"or {dataset_root}/dataset_3node"
    )


def min_pairwise_distance(coords: np.ndarray):
    """Return (min_dist_Ang, atom_i, atom_j) using vectorised numpy."""
    n = len(coords)
    if n < 2:
        return float("inf"), 0, 0
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]   # (N,N,3)
    dmat = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))          # (N,N)
    np.fill_diagonal(dmat, np.inf)
    flat = int(np.argmin(dmat))
    i, j = divmod(flat, n)
    return float(dmat[i, j]), i, j


def bond_order_to_rdkit(order: int) -> Chem.rdchem.BondType:
    return _BOND_TYPE_MAP.get(int(order), Chem.rdchem.BondType.SINGLE)


# ---------------------------------------------------------------------------
# Molecule construction
# ---------------------------------------------------------------------------

def build_rdkit_mol(elements, coordinates, edges, edge_bonds,
                    atom_states, verbose=False) -> Chem.RWMol:
    """
    Build RDKit RWMol from metadata fields.
    atom_states keys are str(atom_index), values are dicts with optional
    'formal_charge' and 'radical_electrons'.
    edge_bonds may be a dict keyed by "i,j" strings OR a list parallel to edges.
    """
    mol = Chem.RWMol()

    for idx, sym in enumerate(elements):
        atom = Chem.Atom(sym)
        state = atom_states.get(str(idx), {})
        if "formal_charge" in state:
            atom.SetFormalCharge(int(state["formal_charge"]))
        if "radical_electrons" in state:
            atom.SetNumRadicalElectrons(int(state["radical_electrons"]))
        mol.AddAtom(atom)

    # edge_bonds in metadata is a dict {"i,j": order} according to tex file
    if isinstance(edge_bonds, dict):
        eb_dict = {tuple(map(int, k.split(","))): v for k, v in edge_bonds.items()}
        for i, j in edges:
            key = (int(i), int(j))
            rev = (int(j), int(i))
            order = eb_dict.get(key) or eb_dict.get(rev) or 1
            if key not in eb_dict and rev not in eb_dict and verbose:
                warnings.warn(f"Bond {i}-{j} not in edge_bonds dict; defaulting to SINGLE.")
            mol.AddBond(int(i), int(j), bond_order_to_rdkit(order))
    else:
        # fallback: list parallel to edges
        for bond_idx, (i, j) in enumerate(edges):
            order = edge_bonds[bond_idx] if bond_idx < len(edge_bonds) else 1
            mol.AddBond(int(i), int(j), bond_order_to_rdkit(order))

    # -----------------------------------------------------------------------
    # The metadata edges only store heavy-heavy bonds. H atoms in
    # coordinate_elements have positions but no bond entries. UFF needs
    # connectivity for every atom, so we infer H-heavy bonds from proximity:
    # each unconnected H is bonded (SINGLE) to its nearest non-H neighbour.
    # -----------------------------------------------------------------------
    coords_np = np.array(coordinates, dtype=float)
    heavy_indices = [i for i, e in enumerate(elements) if e != "H"]

    for idx, elem in enumerate(elements):
        if elem != "H":
            continue
        if mol.GetAtomWithIdx(idx).GetDegree() > 0:
            continue  # already bonded — e.g. H explicitly in edges
        if not heavy_indices:
            if verbose:
                warnings.warn(f"H atom {idx} has no heavy neighbours to bond to.")
            continue
        dists = np.linalg.norm(coords_np[heavy_indices] - coords_np[idx], axis=1)
        nearest = heavy_indices[int(np.argmin(dists))]
        mol.AddBond(idx, nearest, Chem.rdchem.BondType.SINGLE)
        if verbose:
            print(f"    Inferred H-bond: atom {idx}(H) → atom {nearest}({elements[nearest]}), "
                  f"dist={float(np.min(dists)):.3f} Å")

    conf = Chem.Conformer(len(elements))
    for idx, (x, y, z) in enumerate(coordinates):
        conf.SetAtomPosition(idx, (float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)

    return mol


def get_coords_from_mol(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    return np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
                    dtype=float)


# ---------------------------------------------------------------------------
# UFF force computation
# ---------------------------------------------------------------------------

def compute_uff_forces(ff, n_atoms: int):
    """
    Returns (energy_kcal_mol, force_magnitudes_kcal_mol_ang).
    CalcGrad() = -force, so force = -CalcGrad().
    """
    energy = ff.CalcEnergy()
    grad = np.array(ff.CalcGrad(), dtype=float)       # (3*N,), kcal/(mol*Ang)
    forces = -grad.reshape(n_atoms, 3)                 # force = -gradient
    mags = np.linalg.norm(forces, axis=1)
    return energy, mags


def print_force_table(mags, elements, unit, top_n=None):
    order = np.argsort(mags)[::-1]
    if top_n:
        order = order[:top_n]
    print(f"  {'Atom':>5}  {'Elem':>5}  {'|Force|':>18}  ({unit})")
    print(f"  {'----':>5}  {'----':>5}  {'-------':>18}")
    for idx in order:
        print(f"  {idx:>5d}  {elements[idx]:>5s}  {mags[idx]:>18.6f}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    default_root = Path(".")
    p = argparse.ArgumentParser(
        description="Validate stored UFF geometry: UFF forces, QM comparison, convergence diagnosis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("mol_id", help="Molecule identifier, e.g. mol_097583")
    p.add_argument("--dataset-root", type=Path, default=default_root,
                   help="Root dir containing dataset_2node / dataset_3node")
    p.add_argument("--steps", type=int, default=100,
                   help="Extra UFF optimization steps for convergence diagnosis")
    p.add_argument("--top-n", type=int, default=None, dest="top_n",
                   help="Show only top-N atoms by force magnitude (default: all)")
    p.add_argument("--verbose", action="store_true",
                   help="Print full tracebacks on errors")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    mol_id = args.mol_id
    dataset_root = args.dataset_root
    extra_steps = args.steps
    top_n = args.top_n
    verbose = args.verbose

    # --- validate mol_id ---
    try:
        validate_mol_id(mol_id)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # STEP 1: Locate molecule
    # -----------------------------------------------------------------------
    section("STEP 1: Locating molecule directory")
    try:
        mol_dir = find_mol_dir(mol_id, dataset_root)
        print(f"  Found: {mol_dir}")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # STEP 2: Load metadata
    # -----------------------------------------------------------------------
    section("STEP 2: Loading metadata")
    try:
        with open(mol_dir / "metadata.json", encoding="utf-8") as fh:
            metadata = json.load(fh)
        print(f"  OK: {mol_dir / 'metadata.json'}")
    except (OSError, json.JSONDecodeError) as e:
        print(f"  ERROR: {e}")
        if verbose:
            traceback.print_exc()
        sys.exit(1)

    rep          = metadata.get("representative_structure", {})
    elements     = rep.get("coordinate_elements", [])
    coordinates  = rep.get("coordinates", [])
    total_charge = int(rep.get("total_charge", 0))
    multiplicity = int(rep.get("multiplicity", 1))
    edges        = rep.get("edges", [])
    edge_bonds   = rep.get("edge_bonds", {})   # dict or list
    atom_states  = rep.get("atom_states", {})
    smiles       = rep.get("smiles", "N/A")
    avail_calcs  = metadata.get("available_calculations", {})

    if not elements or not coordinates:
        print("  ERROR: Missing coordinate_elements or coordinates in metadata.")
        sys.exit(1)
    if len(elements) != len(coordinates):
        print(f"  ERROR: elements ({len(elements)}) / coordinates ({len(coordinates)}) length mismatch.")
        sys.exit(1)

    n_atoms = len(elements)
    print(f"  Atoms   : {n_atoms}")
    print(f"  Bonds   : {len(edges)}")
    print(f"  Charge  : {total_charge}")
    print(f"  Mult    : {multiplicity}")
    print(f"  SMILES  : {smiles}")

    # -----------------------------------------------------------------------
    # STEP 3: Build RDKit molecule
    # -----------------------------------------------------------------------
    section("STEP 3: Building RDKit molecule")
    try:
        mol = build_rdkit_mol(elements, coordinates, edges, edge_bonds,
                              atom_states, verbose=verbose)
        print(f"  RWMol: {mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds")
    except Exception as e:
        print(f"  ERROR: {e}")
        if verbose:
            traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------------------------
    # STEP 4: Sanitize
    # -----------------------------------------------------------------------
    section("STEP 4: Sanitizing")
    _PARTIAL_FLAGS = (
        Chem.SanitizeFlags.SANITIZE_FINDRADICALS
        | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
        | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
        | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
        | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
    )
    try:
        Chem.SanitizeMol(mol)
        print("  Full sanitization: OK")
    except Exception as e:
        print(f"  WARNING: Full sanitization failed ({e})")
        print("  Attempting partial sanitization (hybridization + rings, no valence check)...")
        if verbose:
            traceback.print_exc()
        try:
            Chem.SanitizeMol(mol, _PARTIAL_FLAGS)
            print("  Partial sanitization: OK (hybridization set, valence check skipped)")
        except Exception as e2:
            print(f"  WARNING: Partial sanitization also failed ({e2}) — UFF may fail")
            if verbose:
                traceback.print_exc()

    # UpdatePropertyCache with strict=False so UFF can compute implicit Hs
    # without crashing on unusual valences (e.g. hypervalent B, S in exotic rings)
    try:
        mol.UpdatePropertyCache(strict=False)
        print("  UpdatePropertyCache(strict=False): OK")
    except Exception as e:
        print(f"  WARNING: UpdatePropertyCache failed ({e})")
        if verbose:
            traceback.print_exc()

    # All atoms (including H) are explicitly present in coordinate_elements,
    # so mark every atom NoImplicit=True. This prevents UFFGetMoleculeForceField
    # from trying to compute implicit-H counts and throwing AtomValenceException
    # for atoms with unusual valences (e.g. hypervalent B, allenic B in a 3-ring).
    for atom in mol.GetAtoms():
        atom.SetNoImplicit(True)
    print("  SetNoImplicit(True) on all atoms: OK")

    # -----------------------------------------------------------------------
    # STEP 5: Setup UFF
    # -----------------------------------------------------------------------
    section("STEP 5: Setting up UFF force field")
    ff = None
    try:
        ff = AllChem.UFFGetMoleculeForceField(mol)
    except Exception as e:
        print(f"  WARNING: UFFGetMoleculeForceField raised exception: {e}")
        if verbose:
            traceback.print_exc()

    if ff is None:
        print(
            "  UFF unavailable for this molecule (unrecognized atom type or\n"
            "  unsupported valence state — e.g. exotic B hybridization, noble gas).\n"
            "  UFF force/convergence steps will be skipped; QM comparison continues."
        )
    else:
        print("  UFF initialized OK")

    # -----------------------------------------------------------------------
    # STEP 6: UFF energy + forces at stored geometry
    # -----------------------------------------------------------------------
    section("STEP 6: UFF energy and forces at stored geometry")

    coords_arr = np.array(coordinates, dtype=float)
    min_dist, mi, mj = min_pairwise_distance(coords_arr)
    dist_warn = "  *** < 0.5 Ang! ***" if min_dist < 0.5 else ""
    print(f"\n  Min pairwise distance: {min_dist:.4f} Ang "
          f"(atom {mi}:{elements[mi]} -- atom {mj}:{elements[mj]}){dist_warn}")

    energy_init = None
    force_mags_init = None
    if ff is None:
        print("  Skipped (UFF unavailable).")
    else:
        try:
            energy_init, force_mags_init = compute_uff_forces(ff, n_atoms)
        except Exception as e:
            print(f"  ERROR computing UFF forces: {e}")
            if verbose:
                traceback.print_exc()

        if force_mags_init is not None:
            max_f_init = float(np.max(force_mags_init))
            max_f_atom = int(np.argmax(force_mags_init))
            print(f"\n  E_UFF (stored geometry) = {energy_init:.6f} kcal/mol")
            print(f"  Max |UFF force|         = {max_f_init:.6f} kcal/(mol*Ang)"
                  f"  [atom {max_f_atom}: {elements[max_f_atom]}]")
            print(f"\n  Per-atom UFF forces (force = -grad, kcal/(mol*Ang)):")
            print_force_table(force_mags_init, elements, "kcal/(mol*Ang)", top_n=top_n)

    # -----------------------------------------------------------------------
    # STEP 7: QM forces comparison
    # -----------------------------------------------------------------------
    section("STEP 7: QM forces comparison")

    if not avail_calcs:
        print("  No available_calculations found in metadata.")
    else:
        for calc_name, calc_info in avail_calcs.items():
            npz_file = calc_info.get("npz_file", "")
            npz_path = mol_dir / npz_file
            print(f"\n  [{calc_name}]  {npz_path}")
            if not npz_path.exists():
                print("    File not found — skipping.")
                continue
            try:
                data = np.load(npz_path, allow_pickle=False)
            except Exception as e:
                print(f"    ERROR loading npz: {e}")
                if verbose:
                    traceback.print_exc()
                continue

            if "step_0_cartesian_forces" not in data:
                print(f"    WARNING: 'step_0_cartesian_forces' not in file. "
                      f"Keys: {list(data.keys())}")
                continue

            qm_raw = data["step_0_cartesian_forces"]
            expected = n_atoms * 3
            if qm_raw.ndim == 1 and qm_raw.size != expected:
                print(f"    ERROR: force array has {qm_raw.size} elements, "
                      f"expected {expected}. Skipping.")
                continue

            qm_forces = np.array(qm_raw, dtype=float)
            if qm_forces.ndim == 1:
                qm_forces = qm_forces.reshape(n_atoms, 3)
            qm_mags = np.linalg.norm(qm_forces, axis=1)
            print(f"    Max |QM force| = {float(np.max(qm_mags)):.6f} {HA_BOHR_LABEL}")
            print(f"    Per-atom QM forces ({HA_BOHR_LABEL}):")
            print_force_table(qm_mags, elements, HA_BOHR_LABEL, top_n=top_n)

    # -----------------------------------------------------------------------
    # STEP 8: Extra UFF optimization steps — convergence diagnosis
    # -----------------------------------------------------------------------
    section(f"STEP 8: UFF convergence diagnosis ({extra_steps} extra steps)")

    if ff is None or energy_init is None:
        print("  Skipped (UFF unavailable).")
    else:
        energy_trajectory = [energy_init]

        print(f"\n  Running {extra_steps} UFF minimization steps (1 step at a time)...")
        for step in range(extra_steps):
            ff.Minimize(maxIts=1)
            e = ff.CalcEnergy()
            energy_trajectory.append(e)

        print(f"\n  Energy trajectory (every 10 steps):")
        print(f"  {'Step':>6}  {'E (kcal/mol)':>18}  {'dE':>14}")
        print(f"  {'----':>6}  {'------------':>18}  {'--':>14}")
        for i in range(0, len(energy_trajectory), 10):
            dE = energy_trajectory[i] - energy_trajectory[max(0, i - 10)] if i > 0 else 0.0
            print(f"  {i:>6d}  {energy_trajectory[i]:>18.6f}  {dE:>14.6f}")
        last = len(energy_trajectory) - 1
        if last % 10 != 0:
            dE = energy_trajectory[last] - energy_trajectory[last - 1]
            print(f"  {last:>6d}  {energy_trajectory[last]:>18.6f}  {dE:>14.6f}")

        tail = np.array(energy_trajectory[-10:])
        tail_var = float(np.max(tail) - np.min(tail))
        converged = tail_var < 1e-4
        print(f"\n  Energy range over last 10 steps: {tail_var:.2e} kcal/mol")
        if converged:
            print("  DIAGNOSIS: Converged (energy flat) — stored geometry was near a UFF minimum.")
        else:
            print("  DIAGNOSIS: NOT converged — stored geometry was NOT at a UFF stationary point.")
            print("             The original 50-step UFF run likely stopped mid-optimization.")
            print("             Large QM forces here may partly reflect UFF geometry quality.")

    # -----------------------------------------------------------------------
    # STEP 9: Final geometry diagnostics
    # -----------------------------------------------------------------------
    section("STEP 9: Final geometry after extra steps")

    if ff is None or energy_init is None:
        print("  Skipped (UFF unavailable).")
    else:
        coords_final = get_coords_from_mol(mol)
        min_dist_f, fi, fj = min_pairwise_distance(coords_final)
        dist_warn_f = "  *** < 0.5 Ang! ***" if min_dist_f < 0.5 else ""

        energy_final, force_mags_final = compute_uff_forces(ff, n_atoms)
        max_f_final = float(np.max(force_mags_final))

        disp = np.linalg.norm(coords_final - coords_arr, axis=1)
        max_disp = float(np.max(disp))
        max_disp_atom = int(np.argmax(disp))

        print(f"\n  E_UFF (final)          = {energy_final:.6f} kcal/mol")
        print(f"  E_UFF (initial)        = {energy_init:.6f} kcal/mol")
        print(f"  Delta E                = {energy_final - energy_init:.6f} kcal/mol")
        print(f"\n  Max |UFF force| final  = {max_f_final:.6f} kcal/(mol*Ang)")
        print(f"  Max |UFF force| init   = {float(np.max(force_mags_init)):.6f} kcal/(mol*Ang)")
        print(f"\n  Min pairwise dist (final) : {min_dist_f:.4f} Ang "
              f"(atom {fi}:{elements[fi]} -- atom {fj}:{elements[fj]}){dist_warn_f}")
        print(f"  Min pairwise dist (init)  : {min_dist:.4f} Ang")
        print(f"\n  Max atom displacement      : {max_disp:.4f} Ang "
              f"(atom {max_disp_atom}: {elements[max_disp_atom]})")

    print()
    print("=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
