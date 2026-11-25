#!/usr/bin/env python3
"""
Single-pass, parallel dataset verification script for QMEnum-2-3
(dataset_2node and dataset_3node).

Performs comprehensive file integrity, metadata consistency, Q-Chem .qcout
parsing, .npz schema/dtype/finiteness checks, basis-shell and AO/MO matrix
identities, HOMO/LUMO and <S^2> diagnostics, Cordero crowding classification,
filter-overlap tabulation, raw per-calculation / per-geometry CSV.gz exports,
and 3x3 method-by-basis SCF energy histogram plotting.
"""

import os

# Must be set BEFORE importing numpy to prevent BLAS thread oversubscription
# across worker processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import concurrent.futures
import csv
import gzip
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


METHODS = ["HF", "B3LYP", "WB97X-D"]
BASIS_SETS = ["STO-3G", "6-31G*", "DEF2-SVP"]
EXPECTED_METHODS = frozenset(METHODS)
EXPECTED_BASES = frozenset(BASIS_SETS)

BOHR_TO_ANGSTROM = 0.529177210903

ATOMIC_NUMBERS = {
    "H": 1, "He": 2,
    "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
}

# Cordero et al. (2008)-style single-bond covalent radii (Angstrom),
# matching scientific_data/crowding_detector.py.
COVALENT_RADII_A = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05,
    "Cl": 1.02, "Ar": 1.06,
}
DEFAULT_COVALENT_RADIUS_A = 0.75

SHELL_FN_COUNTS = {
    0: 1,
    1: 3,
    -1: 4,
    2: 6,
    -2: 5,
    3: 10,
    -3: 7,
}

ALLOWED_SHELL_TYPES_BY_BASIS = {
    "STO-3G": frozenset({0, 1, -1}),
    "6-31G*": frozenset({0, 1, -1, 2}),
    "DEF2-SVP": frozenset({0, 1, -2}),
}

COMMON_NPZ_KEYS = frozenset({
    "num_steps",
    "step_0_alpha_mo_coefficients",
    "step_0_alpha_orbital_energies",
    "step_0_atomic_numbers",
    "step_0_cartesian_forces",
    "step_0_charge",
    "step_0_contraction_coefficients",
    "step_0_coordinates_of_each_shell",
    "step_0_core_hamiltonian_matrix",
    "step_0_current_cartesian_coordinates",
    "step_0_dipole_data",
    "step_0_hexadecapole_data",
    "step_0_highest_angular_momentum",
    "step_0_kinetic_energy_matrix",
    "step_0_largest_degree_of_contraction",
    "step_0_multiplicity",
    "step_0_multipole_matrix_001",
    "step_0_multipole_matrix_002",
    "step_0_multipole_matrix_010",
    "step_0_multipole_matrix_011",
    "step_0_multipole_matrix_020",
    "step_0_multipole_matrix_100",
    "step_0_multipole_matrix_101",
    "step_0_multipole_matrix_110",
    "step_0_multipole_matrix_200",
    "step_0_nuclear_attraction_matrix",
    "step_0_nuclear_charges",
    "step_0_number_of_alpha_electrons",
    "step_0_number_of_atoms",
    "step_0_number_of_basis_functions",
    "step_0_number_of_beta_electrons",
    "step_0_number_of_contracted_shells",
    "step_0_number_of_electrons",
    "step_0_number_of_independent_functions",
    "step_0_number_of_primitive_shells",
    "step_0_number_of_primitives_per_shell",
    "step_0_octapole_data",
    "step_0_orthonormalization_matrix",
    "step_0_overlap_matrix",
    "step_0_primitive_exponents",
    "step_0_quadrupole_data",
    "step_0_shell_to_atom_map",
    "step_0_shell_types",
    "step_0_total_scf_density",
})

SCALAR_INT32_KEYS = frozenset({
    "num_steps",
    "step_0_charge",
    "step_0_highest_angular_momentum",
    "step_0_largest_degree_of_contraction",
    "step_0_multiplicity",
    "step_0_number_of_alpha_electrons",
    "step_0_number_of_atoms",
    "step_0_number_of_basis_functions",
    "step_0_number_of_beta_electrons",
    "step_0_number_of_contracted_shells",
    "step_0_number_of_electrons",
    "step_0_number_of_independent_functions",
    "step_0_number_of_primitive_shells",
})

OPEN_SHELL_NPZ_KEYS = frozenset({
    "step_0_beta_mo_coefficients",
    "step_0_beta_orbital_energies",
    "step_0_spin_scf_density",
})
PSP_NPZ_KEY = "step_0_ps=p_contraction_coefficients"
ALL_KNOWN_NPZ_KEYS = COMMON_NPZ_KEYS | OPEN_SHELL_NPZ_KEYS | {PSP_NPZ_KEY}

TRIL_MATRIX_KEYS_COMMON = (
    "step_0_overlap_matrix",
    "step_0_core_hamiltonian_matrix",
    "step_0_kinetic_energy_matrix",
    "step_0_nuclear_attraction_matrix",
    "step_0_total_scf_density",
    "step_0_multipole_matrix_001",
    "step_0_multipole_matrix_002",
    "step_0_multipole_matrix_010",
    "step_0_multipole_matrix_011",
    "step_0_multipole_matrix_020",
    "step_0_multipole_matrix_100",
    "step_0_multipole_matrix_101",
    "step_0_multipole_matrix_110",
    "step_0_multipole_matrix_200",
)

CALC_CSV_FIELDS = [
    "mol_id",
    "calc_name",
    "method",
    "basis",
    "wf_type",
    "n_atoms",
    "charge",
    "multiplicity",
    "n_electrons",
    "n_alpha",
    "n_beta",
    "n_bas",
    "n_indep",
    "scf_energy_ha",
    "e_per_electron_ha",
    "nuc_rep_energy_ha",
    "s2",
    "s2_exact",
    "delta_s2",
    "homo_alpha_ha",
    "lumo_alpha_ha",
    "homo_beta_ha",
    "lumo_beta_ha",
    "homo_eff_ha",
    "lumo_eff_ha",
    "gap_eff_ha",
    "max_force_ha_bohr",
    "rms_force_ha_bohr",
    "n_atom_forces_le_1",
    "net_force_ha_bohr",
    "net_torque_ha_rad",
    "d_min_A",
    "min_rcov_ratio",
    "crowding_label",
    "pass_crowding_050",
    "pass_low_strain_calc",
    "tr_ps_err",
    "tr_pspin_err",
    "ca_ortho_err",
    "cb_ortho_err",
    "p_recon_err",
    "t_plus_v_err",
    "geom_meta_npz_max_diff_A",
    "qcout_normal_term",
    "qcout_lin_dep",
]

GEOM_CSV_FIELDS = [
    "mol_id",
    "geometry_hash",
    "has_converged_calc",
    "n_calcs",
    "completeness_str",
    "n_atoms",
    "n_heavy_atoms",
    "elements_str",
    "charge",
    "multiplicity",
    "num_alt_structures",
    "single_spec_latin_ok",
    "alt_specs_same_qm",
    "geom_consistent_across_calcs",
    "d_min_A",
    "min_rcov_ratio",
    "crowding_label",
    "n_atoms_involved_crowding",
    "n_critical_pairs",
    "min_force_across_calcs",
    "max_force_across_calcs",
    "pass_crowding_050",
    "pass_low_strain_all_calcs",
    "any_calc_force_gt_10",
    "any_calc_lin_dep",
]

_ALPHA_BETA_RX = re.compile(r"There are\s+(\d+)\s+alpha and\s+(\d+)\s+beta electrons")
_ROHF_TOKEN_RX = re.compile(r"\brohf\b", re.IGNORECASE)

# Per-worker lower-triangular index cache
_TRIL_CACHE = {}


def get_tril_indices(n: int):
    """Return cached np.tril_indices(n) for fast symmetric matrix unpacking."""
    iu = _TRIL_CACHE.get(n)
    if iu is None:
        iu = np.tril_indices(n)
        _TRIL_CACHE[n] = iu
    return iu


def unpack_tril(vec: np.ndarray, n: int) -> np.ndarray:
    """Unpack a 1D packed lower-triangular vector into a float64 symmetric (n, n) matrix."""
    iu = get_tril_indices(n)
    mat = np.empty((n, n), dtype=np.float64)
    mat[iu] = vec
    mat.T[iu] = vec
    return mat


def covalent_radius(elem: str) -> float:
    return COVALENT_RADII_A.get(elem, DEFAULT_COVALENT_RADIUS_A)


def pairwise_distance_matrix(coords_3d: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix for shape (N, 3) float64 array."""
    diff = coords_3d[:, None, :] - coords_3d[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def analyze_geometry_crowding(elements, coords_A: np.ndarray, factor: float = 0.50):
    """
    Compute d_min_A, min_rcov_ratio, and Cordero crowding classification
    ('none', 'clean_pair', 'crowded_cluster') matching crowding_detector.py.
    """
    n = coords_A.shape[0]
    if n < 2:
        return float("nan"), float("nan"), "none", 0, 0

    dmat = pairwise_distance_matrix(coords_A)
    radii = np.array([covalent_radius(e) for e in elements], dtype=np.float64)
    sum_radii = radii[:, None] + radii[None, :]

    dmat_nodiag = dmat.copy()
    np.fill_diagonal(dmat_nodiag, np.inf)
    d_min_A = float(np.min(dmat_nodiag))

    ratio_mat = dmat_nodiag / sum_radii
    min_rcov_ratio = float(np.min(ratio_mat))

    cutoff = factor * sum_radii
    critical = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and dmat_nodiag[i, j] < cutoff[i, j]:
                critical[i].append(j)

    involved = [i for i, lst in enumerate(critical) if lst]
    if not involved:
        return d_min_A, min_rcov_ratio, "none", 0, 0

    n_critical_pairs = sum(len(lst) for lst in critical) // 2
    is_clean_pair = True
    for i in involved:
        if len(critical[i]) != 1:
            is_clean_pair = False
            break
        partner = critical[i][0]
        if len(critical[partner]) != 1 or critical[partner][0] != i:
            is_clean_pair = False
            break

    label = "clean_pair" if is_clean_pair else "crowded_cluster"
    return d_min_A, min_rcov_ratio, label, len(involved), n_critical_pairs


def parse_qcout_full(qcout_file: Path) -> dict:
    """Single-pass parser for Q-Chem .qcout files."""
    res = {
        "read_error": None,
        "qcout_charge": None,
        "qcout_mult": None,
        "qcout_n_alpha": None,
        "qcout_n_beta": None,
        "scf_energy_ha": None,
        "nuc_rep_energy_ha": None,
        "s2": None,
        "qcout_lin_dep": False,
        "qcout_unrestricted": False,
        "qcout_rohf": False,
        "qcout_normal_term": False,
    }
    in_mol = False
    got_mol_qm = False

    try:
        with open(qcout_file, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not got_mol_qm:
                    s_strip = line.strip()
                    s_low = s_strip.lower()
                    if not in_mol:
                        if s_low == "$molecule" or s_low.startswith("$molecule "):
                            in_mol = True
                            continue
                    else:
                        if s_low.startswith("$end"):
                            in_mol = False
                        elif s_strip and not s_strip.startswith("!"):
                            parts = s_strip.split()
                            if len(parts) >= 2:
                                try:
                                    res["qcout_charge"] = int(parts[0])
                                    res["qcout_mult"] = int(parts[1])
                                    got_mol_qm = True
                                except ValueError:
                                    pass
                            in_mol = False

                if "SCF   energy in the final basis set" in line:
                    try:
                        res["scf_energy_ha"] = float(line.split("=")[-1].strip())
                    except ValueError:
                        pass
                elif "<S^2>" in line and "=" in line:
                    try:
                        res["s2"] = float(line.split("=")[-1].split()[0])
                    except (ValueError, IndexError):
                        pass
                elif res["nuc_rep_energy_ha"] is None and "Nuclear Repulsion Energy =" in line:
                    try:
                        res["nuc_rep_energy_ha"] = float(line.split("=")[-1].split()[0])
                    except (ValueError, IndexError):
                        pass
                elif res["qcout_n_alpha"] is None and "alpha and" in line and "beta electrons" in line:
                    m = _ALPHA_BETA_RX.search(line)
                    if m:
                        res["qcout_n_alpha"] = int(m.group(1))
                        res["qcout_n_beta"] = int(m.group(2))
                elif "Linear dependence detected in AO basis" in line:
                    res["qcout_lin_dep"] = True
                elif "Thank you very much for using Q-Chem" in line:
                    res["qcout_normal_term"] = True
                else:
                    low = line.lower()
                    if "unrestricted scf calculation" in low or "alpha mos, unrestricted" in low:
                        res["qcout_unrestricted"] = True
                    elif (
                        ("restricted open-shell" in low and "unrestricted" not in low)
                        or _ROHF_TOKEN_RX.search(low)
                    ):
                        res["qcout_rohf"] = True
    except Exception as exc:
        res["read_error"] = f"{type(exc).__name__}: {exc}"

    return res


def parse_qcout_energy(qcout_file):
    """Backwards-compatible helper: parse SCF energy (Ha) from a .qcout file."""
    return parse_qcout_full(Path(qcout_file))["scf_energy_ha"]


def process_molecule_worker(mol_dir_str: str) -> dict:
    """
    Top-level picklable worker function: validates one molecule directory
    (metadata.json + all .qcout and .npz files) in a single pass.
    """
    mol_dir = Path(mol_dir_str)
    mol_id = mol_dir.name
    errors = []
    warnings = []

    try:
        metadata_file = mol_dir / "metadata.json"
        if not metadata_file.exists():
            errors.append(f"{mol_id}: Missing metadata.json")
            return {
                "mol_id": mol_id,
                "errors": errors,
                "warnings": warnings,
                "geom_row": None,
                "calc_rows": [],
                "atom_force_norms": np.empty(0, dtype=np.float32),
            }

        try:
            with open(metadata_file, "r", encoding="utf-8") as fh:
                metadata = json.load(fh)
        except Exception as exc:
            errors.append(f"{mol_id}: Invalid JSON in metadata.json: {exc}")
            return {
                "mol_id": mol_id,
                "errors": errors,
                "warnings": warnings,
                "geom_row": None,
                "calc_rows": [],
                "atom_force_norms": np.empty(0, dtype=np.float32),
            }

        geom_hash = metadata.get("geometry_hash") or ""
        if not geom_hash:
            errors.append(f"{mol_id}: Missing geometry_hash in metadata.json")

        completeness_str = str(metadata.get("completeness", ""))
        rep = metadata.get("representative_structure", {})
        coord_elements = rep.get("coordinate_elements", [])
        meta_coords_raw = rep.get("coordinates", [])
        meta_charge = int(rep.get("total_charge", 0))
        meta_mult = int(rep.get("multiplicity", 1))

        meta_n_atoms = len(coord_elements) if coord_elements else len(meta_coords_raw)
        n_heavy_atoms = sum(1 for el in coord_elements if el != "H")
        elements_str = "-".join(coord_elements)

        expected_z = []
        unknown_elems = []
        for el in coord_elements:
            if el in ATOMIC_NUMBERS:
                expected_z.append(ATOMIC_NUMBERS[el])
            else:
                unknown_elems.append(el)
                expected_z.append(-1)
        if unknown_elems:
            errors.append(f"{mol_id}: Unknown element symbols in coordinate_elements: {unknown_elems}")

        meta_coords_A = None
        dmat_meta = None
        if meta_coords_raw:
            arr_mc = np.asarray(meta_coords_raw, dtype=np.float64)
            if arr_mc.shape == (meta_n_atoms, 3) and np.all(np.isfinite(arr_mc)):
                meta_coords_A = arr_mc
                dmat_meta = pairwise_distance_matrix(meta_coords_A)
            else:
                errors.append(
                    f"{mol_id}: metadata representative_structure.coordinates invalid shape "
                    f"{arr_mc.shape} or non-finite for {meta_n_atoms} atoms"
                )

        # Alternative structure checks
        num_alt = int(metadata.get("num_alternative_structures", 1))
        alt_hashes = metadata.get("alternative_molecular_hashes", [])
        if len(alt_hashes) != num_alt:
            errors.append(
                f"{mol_id}: num_alternative_structures={num_alt} "
                f"but {len(alt_hashes)} alternative_molecular_hashes listed"
            )

        alt_specs_same_qm = None
        if num_alt > 1:
            alt_details = metadata.get("alternative_structures_details", [])
            if len(alt_details) != num_alt:
                errors.append(
                    f"{mol_id}: num_alternative_structures={num_alt} "
                    f"but {len(alt_details)} entries in alternative_structures_details"
                )
            spec_qms = set()
            for idx_s, spec in enumerate(alt_details):
                states = spec.get("atom_states", {}).values()
                q_spec = sum(int(st["charge"]) for st in states)
                m_spec = sum(int(st["unpaired_electrons"]) for st in states) + 1
                spec_qms.add((q_spec, m_spec))
                if (q_spec, m_spec) != (meta_charge, meta_mult):
                    errors.append(
                        f"{mol_id}: alternative spec #{idx_s} (Q={q_spec}, 2S+1={m_spec}) "
                        f"differs from representative_structure (Q={meta_charge}, 2S+1={meta_mult})"
                    )
            alt_specs_same_qm = len(spec_qms) <= 1

        # File & orphan checks
        avail_calcs = metadata.get("available_calculations", {})
        n_calcs = len(avail_calcs)

        all_dir_files = {p.name for p in mol_dir.iterdir() if p.is_file()}
        expected_files = {"metadata.json"}
        for calc_name, calc_info in avail_calcs.items():
            qf_name = calc_info.get("qcout_file", "")
            nf_name = calc_info.get("npz_file", "")
            if qf_name:
                expected_files.add(qf_name)
            if nf_name:
                expected_files.add(nf_name)
            if not qf_name or not (mol_dir / qf_name).exists():
                errors.append(f"{mol_id}/{calc_name}: Missing qcout_file '{qf_name}'")
            if not nf_name or not (mol_dir / nf_name).exists():
                errors.append(f"{mol_id}/{calc_name}: Missing npz_file '{nf_name}'")

        orphans = all_dir_files - expected_files
        if orphans:
            warnings.append(f"{mol_id}: Orphaned files: {sorted(orphans)}")

        # Method/basis design check
        methods_list = []
        bases_list = []
        for calc_name, calc_info in avail_calcs.items():
            m_name = calc_info.get("method", "")
            b_name = calc_info.get("basis", "")
            methods_list.append(m_name)
            bases_list.append(b_name)
            if m_name not in EXPECTED_METHODS:
                errors.append(f"{mol_id}/{calc_name}: Unexpected method '{m_name}'")
            if b_name not in EXPECTED_BASES:
                errors.append(f"{mol_id}/{calc_name}: Unexpected basis '{b_name}'")
            expected_cname = f"{m_name}_{b_name}"
            if calc_name != expected_cname:
                errors.append(
                    f"{mol_id}/{calc_name}: Key name does not match '{expected_cname}'"
                )

        single_spec_latin_ok = None
        if num_alt == 1:
            single_spec_latin_ok = (
                len(methods_list) == len(set(methods_list))
                and len(bases_list) == len(set(bases_list))
                and n_calcs <= 3
            )
            if not single_spec_latin_ok:
                errors.append(
                    f"{mol_id}: Single-specification Latin-square method/basis rule violated "
                    f"(methods={methods_list}, bases={bases_list})"
                )

        # Process each calculation once
        first_npz_coords_A = None
        n_extra_coords_checked = 0
        geom_consistent_across_calcs = True
        raw_calc_entries = []
        force_norm_chunks = []

        for calc_name in sorted(avail_calcs.keys()):
            calc_info = avail_calcs[calc_name]
            method = calc_info.get("method") or calc_name.split("_", 1)[0]
            basis = calc_info.get("basis") or calc_name.split("_", 1)[-1]
            qcout_path = mol_dir / calc_info.get("qcout_file", "")
            npz_path = mol_dir / calc_info.get("npz_file", "")

            wf_type = (
                ("RHF" if meta_mult == 1 else "UHF")
                if method == "HF"
                else ("RKS" if meta_mult == 1 else "UKS")
            )

            # 1. Parse and validate .qcout
            qres = {
                "qcout_charge": None,
                "qcout_mult": None,
                "qcout_n_alpha": None,
                "qcout_n_beta": None,
                "scf_energy_ha": None,
                "nuc_rep_energy_ha": None,
                "s2": None,
                "qcout_lin_dep": False,
                "qcout_unrestricted": False,
                "qcout_rohf": False,
                "qcout_normal_term": False,
            }
            if qcout_path.is_file():
                qres = parse_qcout_full(qcout_path)
                if qres["read_error"]:
                    errors.append(f"{mol_id}/{calc_name}: Error reading .qcout: {qres['read_error']}")
                else:
                    if qres["scf_energy_ha"] is None or not math.isfinite(qres["scf_energy_ha"]):
                        errors.append(f"{mol_id}/{calc_name}: Missing or non-finite SCF energy in .qcout")
                    if not qres["qcout_normal_term"]:
                        warnings.append(f"{mol_id}/{calc_name}: Missing normal Q-Chem termination marker in .qcout")
                    if qres["qcout_charge"] is None or qres["qcout_mult"] is None:
                        errors.append(f"{mol_id}/{calc_name}: Could not parse $molecule (charge, mult) in .qcout")
                    elif (qres["qcout_charge"], qres["qcout_mult"]) != (meta_charge, meta_mult):
                        errors.append(
                            f"{mol_id}/{calc_name}: .qcout $molecule ({qres['qcout_charge']}, {qres['qcout_mult']}) "
                            f"!= metadata ({meta_charge}, {meta_mult})"
                        )
                    if meta_mult > 1 and qres["qcout_rohf"]:
                        warnings.append(f"{mol_id}/{calc_name}: ROHF marker detected in open-shell .qcout")

            scf_energy_ha = (
                float(qres["scf_energy_ha"])
                if qres["scf_energy_ha"] is not None and math.isfinite(qres["scf_energy_ha"])
                else float("nan")
            )
            nuc_rep_ha = (
                float(qres["nuc_rep_energy_ha"])
                if qres["nuc_rep_energy_ha"] is not None and math.isfinite(qres["nuc_rep_energy_ha"])
                else float("nan")
            )

            # Spin contamination
            s_val = 0.5 * (meta_mult - 1)
            s2_exact = s_val * (s_val + 1.0)
            s2_val = float("nan")
            delta_s2 = float("nan")
            missing_s2 = False
            if meta_mult > 1:
                if qres["s2"] is None or not math.isfinite(qres["s2"]):
                    if qcout_path.is_file():
                        warnings.append(f"{mol_id}/{calc_name}: Open-shell (2S+1={meta_mult}) missing <S^2> in .qcout")
                        missing_s2 = True
                else:
                    s2_val = float(qres["s2"])
                    delta_s2 = s2_val - s2_exact

            # Default calculation metrics before NPZ load
            n_atoms_c = meta_n_atoms
            charge_c = meta_charge
            mult_c = meta_mult
            n_elec_c = sum(z for z in expected_z if z > 0) - meta_charge if expected_z else 0
            n_alpha_c = (n_elec_c + (meta_mult - 1)) // 2
            n_beta_c = (n_elec_c - (meta_mult - 1)) // 2
            n_bas = 0
            n_indep = 0
            homo_alpha_ha = float("nan")
            lumo_alpha_ha = float("nan")
            homo_beta_ha = float("nan")
            lumo_beta_ha = float("nan")
            homo_eff_ha = float("nan")
            lumo_eff_ha = float("nan")
            gap_eff_ha = float("nan")
            max_force_ha_bohr = float("nan")
            rms_force_ha_bohr = float("nan")
            n_atom_forces_le_1 = 0
            net_force_ha_bohr = float("nan")
            net_torque_ha_rad = float("nan")
            tr_ps_err = float("nan")
            tr_pspin_err = float("nan")
            ca_ortho_err = float("nan")
            cb_ortho_err = float("nan")
            p_recon_err = float("nan")
            t_plus_v_err = float("nan")
            geom_meta_npz_max_diff_A = float("nan")

            # 2. Load and validate .npz
            if npz_path.is_file():
                try:
                    with np.load(npz_path, allow_pickle=False) as data:
                        present_keys = set(data.files)
                        missing_common = COMMON_NPZ_KEYS - present_keys
                        if missing_common:
                            errors.append(
                                f"{mol_id}/{calc_name}: Missing common NPZ keys: {sorted(missing_common)}"
                            )

                        unexpected_keys = present_keys - ALL_KNOWN_NPZ_KEYS
                        if unexpected_keys:
                            warnings.append(
                                f"{mol_id}/{calc_name}: Unexpected extra NPZ keys: {sorted(unexpected_keys)}"
                            )

                        # Finiteness, ndim, and dtype checks across all arrays
                        all_arrays_finite = True
                        for k_arr in data.files:
                            arr = data[k_arr]
                            if not np.all(np.isfinite(arr)):
                                errors.append(f"{mol_id}/{calc_name}: Non-finite values in NPZ key '{k_arr}'")
                                all_arrays_finite = False
                            if k_arr in SCALAR_INT32_KEYS:
                                if arr.ndim != 0 or arr.dtype != np.int32:
                                    errors.append(
                                        f"{mol_id}/{calc_name}: Scalar key '{k_arr}' has shape={arr.shape}, "
                                        f"dtype={arr.dtype} (expected 0D int32)"
                                    )
                            elif k_arr in ALL_KNOWN_NPZ_KEYS:
                                if arr.ndim != 1 or arr.dtype not in (np.int32, np.float32):
                                    errors.append(
                                        f"{mol_id}/{calc_name}: 1D key '{k_arr}' has shape={arr.shape}, "
                                        f"dtype={arr.dtype} (expected 1D int32/float32)"
                                    )

                        if "num_steps" in present_keys and int(data["num_steps"]) != 1:
                            errors.append(
                                f"{mol_id}/{calc_name}: num_steps={int(data['num_steps'])} != 1"
                            )

                        if not missing_common and all_arrays_finite:
                            n_atoms_c = int(data["step_0_number_of_atoms"])
                            charge_c = int(data["step_0_charge"])
                            mult_c = int(data["step_0_multiplicity"])
                            n_elec_c = int(data["step_0_number_of_electrons"])
                            n_alpha_c = int(data["step_0_number_of_alpha_electrons"])
                            n_beta_c = int(data["step_0_number_of_beta_electrons"])
                            n_bas = int(data["step_0_number_of_basis_functions"])
                            n_indep = int(data["step_0_number_of_independent_functions"])
                            n_shells = int(data["step_0_number_of_contracted_shells"])
                            n_prims = int(data["step_0_number_of_primitive_shells"])

                            # Conditional keys: open-shell beta + spin density
                            present_open = OPEN_SHELL_NPZ_KEYS & present_keys
                            if mult_c > 1:
                                if present_open != OPEN_SHELL_NPZ_KEYS:
                                    errors.append(
                                        f"{mol_id}/{calc_name}: Open-shell (2S+1={mult_c}) missing "
                                        f"beta/spin keys: {sorted(OPEN_SHELL_NPZ_KEYS - present_open)}"
                                    )
                            else:
                                if present_open:
                                    errors.append(
                                        f"{mol_id}/{calc_name}: Closed-shell (2S+1=1) unexpectedly has "
                                        f"open-shell keys: {sorted(present_open)}"
                                    )

                            # Metadata <-> NPZ <-> qcout global consistency
                            z_arr = data["step_0_atomic_numbers"]
                            nuc_q = data["step_0_nuclear_charges"]
                            if n_atoms_c != meta_n_atoms:
                                errors.append(
                                    f"{mol_id}/{calc_name}: NPZ n_atoms={n_atoms_c} != metadata {meta_n_atoms}"
                                )
                            if charge_c != meta_charge:
                                errors.append(
                                    f"{mol_id}/{calc_name}: NPZ charge={charge_c} != metadata {meta_charge}"
                                )
                            if mult_c != meta_mult:
                                errors.append(
                                    f"{mol_id}/{calc_name}: NPZ multiplicity={mult_c} != metadata {meta_mult}"
                                )
                            if z_arr.size != n_atoms_c:
                                errors.append(
                                    f"{mol_id}/{calc_name}: atomic_numbers length {z_arr.size} != {n_atoms_c}"
                                )
                            elif expected_z and not unknown_elems and z_arr.tolist() != expected_z:
                                errors.append(
                                    f"{mol_id}/{calc_name}: NPZ atomic_numbers {z_arr.tolist()} "
                                    f"!= metadata {expected_z}"
                                )
                            if nuc_q.size != n_atoms_c or not np.allclose(nuc_q, z_arr, atol=1e-4):
                                errors.append(
                                    f"{mol_id}/{calc_name}: nuclear_charges inconsistent with atomic_numbers"
                                )

                            z_sum_npz = int(np.sum(z_arr))
                            if n_elec_c != z_sum_npz - charge_c:
                                errors.append(
                                    f"{mol_id}/{calc_name}: n_electrons={n_elec_c} != sum(Z)-charge={z_sum_npz - charge_c}"
                                )
                            if n_alpha_c + n_beta_c != n_elec_c:
                                errors.append(
                                    f"{mol_id}/{calc_name}: n_alpha({n_alpha_c})+n_beta({n_beta_c}) != n_elec({n_elec_c})"
                                )
                            if n_alpha_c - n_beta_c != mult_c - 1:
                                errors.append(
                                    f"{mol_id}/{calc_name}: n_alpha({n_alpha_c})-n_beta({n_beta_c}) != mult-1({mult_c - 1})"
                                )
                            if (
                                qres["qcout_n_alpha"] is not None
                                and (qres["qcout_n_alpha"], qres["qcout_n_beta"]) != (n_alpha_c, n_beta_c)
                            ):
                                errors.append(
                                    f"{mol_id}/{calc_name}: .qcout (n_alpha, n_beta)="
                                    f"({qres['qcout_n_alpha']}, {qres['qcout_n_beta']}) != NPZ ({n_alpha_c}, {n_beta_c})"
                                )

                            # Basis & shell checks
                            shell_types = data["step_0_shell_types"]
                            shell_map = data["step_0_shell_to_atom_map"]
                            prims_per_shell = data["step_0_number_of_primitives_per_shell"]
                            shell_coords = data["step_0_coordinates_of_each_shell"]
                            prim_exp = data["step_0_primitive_exponents"]
                            contr_coeff = data["step_0_contraction_coefficients"]

                            st_list = shell_types.tolist()
                            has_sp_shell = -1 in st_list
                            has_psp_key = PSP_NPZ_KEY in present_keys
                            if has_sp_shell != has_psp_key:
                                errors.append(
                                    f"{mol_id}/{calc_name}: sp-shell presence ({has_sp_shell}) "
                                    f"mismatches '{PSP_NPZ_KEY}' presence ({has_psp_key})"
                                )

                            if (
                                shell_types.size != n_shells
                                or shell_map.size != n_shells
                                or prims_per_shell.size != n_shells
                                or shell_coords.size != 3 * n_shells
                            ):
                                errors.append(
                                    f"{mol_id}/{calc_name}: Per-shell array length mismatch for n_shells={n_shells}"
                                )
                            if (
                                int(np.sum(prims_per_shell)) != n_prims
                                or prim_exp.size != n_prims
                                or contr_coeff.size != n_prims
                                or (has_psp_key and data[PSP_NPZ_KEY].size != n_prims)
                            ):
                                errors.append(
                                    f"{mol_id}/{calc_name}: Per-primitive array length mismatch for n_prims={n_prims}"
                                )

                            unknown_st = [st for st in st_list if st not in SHELL_FN_COUNTS]
                            if unknown_st:
                                errors.append(
                                    f"{mol_id}/{calc_name}: Unknown shell_types codes: {unknown_st}"
                                )
                            else:
                                counted_nbas = sum(SHELL_FN_COUNTS[st] for st in st_list)
                                if counted_nbas != n_bas:
                                    errors.append(
                                        f"{mol_id}/{calc_name}: shell_types sum to {counted_nbas} != n_bas={n_bas}"
                                    )

                            if basis in ALLOWED_SHELL_TYPES_BY_BASIS:
                                bad_st = set(st_list) - ALLOWED_SHELL_TYPES_BY_BASIS[basis]
                                if bad_st:
                                    errors.append(
                                        f"{mol_id}/{calc_name}: Disallowed shell_types {sorted(bad_st)} for basis {basis}"
                                    )

                            if shell_map.size > 0:
                                sm_min = int(np.min(shell_map))
                                sm_max = int(np.max(shell_map))
                                if sm_min < 1 or sm_max > n_atoms_c:
                                    errors.append(
                                        f"{mol_id}/{calc_name}: shell_to_atom_map range [{sm_min}, {sm_max}] "
                                        f"outside [1, {n_atoms_c}]"
                                    )

                            # Coordinates & distance matrix parity
                            coords_raw = data["step_0_current_cartesian_coordinates"]
                            coords_bohr = None
                            if coords_raw.size != 3 * n_atoms_c:
                                errors.append(
                                    f"{mol_id}/{calc_name}: Coordinates length {coords_raw.size} != {3 * n_atoms_c}"
                                )
                            else:
                                coords_bohr = coords_raw.astype(np.float64).reshape(n_atoms_c, 3)
                                coords_A = coords_bohr * BOHR_TO_ANGSTROM
                                if dmat_meta is not None and n_atoms_c == meta_n_atoms:
                                    dmat_npz = pairwise_distance_matrix(coords_A)
                                    geom_meta_npz_max_diff_A = float(np.max(np.abs(dmat_npz - dmat_meta)))
                                    if geom_meta_npz_max_diff_A > 1e-2:
                                        errors.append(
                                            f"{mol_id}/{calc_name}: Pairwise distance mismatch vs metadata "
                                            f"{geom_meta_npz_max_diff_A:.6f} A (> 0.01 A)"
                                        )
                                    elif geom_meta_npz_max_diff_A > 1e-3:
                                        warnings.append(
                                            f"{mol_id}/{calc_name}: Pairwise distance drift vs metadata "
                                            f"{geom_meta_npz_max_diff_A:.6f} A (> 0.001 A)"
                                        )

                                if first_npz_coords_A is None:
                                    first_npz_coords_A = coords_A
                                else:
                                    n_extra_coords_checked += 1
                                    if (
                                        coords_A.shape != first_npz_coords_A.shape
                                        or not np.allclose(coords_A, first_npz_coords_A, atol=1e-4)
                                    ):
                                        geom_consistent_across_calcs = False
                                        warnings.append(
                                            f"{mol_id}/{calc_name}: Standard-orientation coordinates differ "
                                            f"across calculations (> 1e-4 A)"
                                        )

                            # Forces & mechanics
                            forces_raw = data["step_0_cartesian_forces"]
                            if forces_raw.size != 3 * n_atoms_c:
                                errors.append(
                                    f"{mol_id}/{calc_name}: Force array shape {forces_raw.shape} "
                                    f"!= ({3 * n_atoms_c},) for {n_atoms_c} atoms"
                                )
                            else:
                                forces = forces_raw.astype(np.float64).reshape(n_atoms_c, 3)
                                atom_fnorms = np.linalg.norm(forces, axis=1)
                                max_force_ha_bohr = float(np.max(atom_fnorms))
                                rms_force_ha_bohr = float(np.sqrt(np.mean(atom_fnorms ** 2)))
                                n_atom_forces_le_1 = int(np.sum(atom_fnorms <= 1.0))
                                force_norm_chunks.append(atom_fnorms.astype(np.float32))
                                net_force_ha_bohr = float(np.linalg.norm(np.sum(forces, axis=0)))
                                if coords_bohr is not None:
                                    r_centered = coords_bohr - np.mean(coords_bohr, axis=0)
                                    net_torque_ha_rad = float(
                                        np.linalg.norm(np.sum(np.cross(r_centered, forces), axis=0))
                                    )

                            # Matrix & MO checks
                            if n_indep <= 0 or n_indep > n_bas:
                                errors.append(
                                    f"{mol_id}/{calc_name}: Invalid n_indep={n_indep} for n_bas={n_bas}"
                                )
                            elif n_bas > 0:
                                n_tril = n_bas * (n_bas + 1) // 2
                                tril_len_ok = True
                                for mk in TRIL_MATRIX_KEYS_COMMON:
                                    if data[mk].size != n_tril:
                                        errors.append(
                                            f"{mol_id}/{calc_name}: '{mk}' size {data[mk].size} != {n_tril}"
                                        )
                                        tril_len_ok = False
                                if mult_c > 1 and "step_0_spin_scf_density" in present_keys:
                                    if data["step_0_spin_scf_density"].size != n_tril:
                                        errors.append(
                                            f"{mol_id}/{calc_name}: 'step_0_spin_scf_density' size "
                                            f"{data['step_0_spin_scf_density'].size} != {n_tril}"
                                        )
                                        tril_len_ok = False

                                if tril_len_ok:
                                    T_vec = data["step_0_kinetic_energy_matrix"].astype(np.float64)
                                    V_vec = data["step_0_nuclear_attraction_matrix"].astype(np.float64)
                                    Hc_vec = data["step_0_core_hamiltonian_matrix"].astype(np.float64)
                                    t_plus_v_err = float(np.max(np.abs((T_vec + V_vec) - Hc_vec)))
                                    if t_plus_v_err > 1e-3:
                                        errors.append(
                                            f"{mol_id}/{calc_name}: |T + V - H_core|_max = {t_plus_v_err:.3e} > 1e-3"
                                        )

                                    S_mat = unpack_tril(data["step_0_overlap_matrix"], n_bas)
                                    P_tot = unpack_tril(data["step_0_total_scf_density"], n_bas)
                                    tr_ps_err = float(abs(np.sum(P_tot * S_mat) - n_elec_c))
                                    if tr_ps_err > 1e-2:
                                        errors.append(
                                            f"{mol_id}/{calc_name}: |Tr(P_tot S) - N_e| = {tr_ps_err:.3e} > 1e-2"
                                        )

                                    if mult_c > 1 and "step_0_spin_scf_density" in present_keys:
                                        P_spin = unpack_tril(data["step_0_spin_scf_density"], n_bas)
                                        tr_pspin_err = float(
                                            abs(np.sum(P_spin * S_mat) - (n_alpha_c - n_beta_c))
                                        )
                                        if tr_pspin_err > 1e-2:
                                            errors.append(
                                                f"{mol_id}/{calc_name}: |Tr(P_spin S) - (N_a - N_b)| = "
                                                f"{tr_pspin_err:.3e} > 1e-2"
                                            )

                                    ca_raw = data["step_0_alpha_mo_coefficients"]
                                    eps_a = data["step_0_alpha_orbital_energies"]
                                    if ca_raw.size != n_indep * n_bas or eps_a.size != n_indep:
                                        errors.append(
                                            f"{mol_id}/{calc_name}: Alpha MO shapes ({ca_raw.size}, {eps_a.size}) "
                                            f"mismatch n_indep={n_indep}, n_bas={n_bas}"
                                        )
                                    else:
                                        Ca = ca_raw.astype(np.float64).reshape(n_indep, n_bas)
                                        ca_ortho_err = float(
                                            np.max(np.abs(Ca @ S_mat @ Ca.T - np.eye(n_indep)))
                                        )
                                        if ca_ortho_err > 1e-2:
                                            errors.append(
                                                f"{mol_id}/{calc_name}: |C_a S C_a^T - I|_max = "
                                                f"{ca_ortho_err:.3e} > 1e-2"
                                            )

                                        if 1 <= n_alpha_c <= n_indep:
                                            homo_alpha_ha = float(eps_a[n_alpha_c - 1])
                                        if 0 <= n_alpha_c < n_indep:
                                            lumo_alpha_ha = float(eps_a[n_alpha_c])

                                        P_alpha = (
                                            Ca[:n_alpha_c, :].T @ Ca[:n_alpha_c, :]
                                            if n_alpha_c <= n_indep
                                            else None
                                        )

                                        if mult_c == 1:
                                            homo_eff_ha = homo_alpha_ha
                                            lumo_eff_ha = lumo_alpha_ha
                                            if P_alpha is not None:
                                                p_recon_err = float(np.max(np.abs(P_tot - 2.0 * P_alpha)))
                                                if p_recon_err > 1e-2:
                                                    errors.append(
                                                        f"{mol_id}/{calc_name}: |P_tot - 2 P_a|_max = "
                                                        f"{p_recon_err:.3e} > 1e-2"
                                                    )
                                        elif (
                                            "step_0_beta_mo_coefficients" in present_keys
                                            and "step_0_beta_orbital_energies" in present_keys
                                        ):
                                            cb_raw = data["step_0_beta_mo_coefficients"]
                                            eps_b = data["step_0_beta_orbital_energies"]
                                            if cb_raw.size != n_indep * n_bas or eps_b.size != n_indep:
                                                errors.append(
                                                    f"{mol_id}/{calc_name}: Beta MO shapes ({cb_raw.size}, {eps_b.size}) "
                                                    f"mismatch n_indep={n_indep}, n_bas={n_bas}"
                                                )
                                            else:
                                                Cb = cb_raw.astype(np.float64).reshape(n_indep, n_bas)
                                                cb_ortho_err = float(
                                                    np.max(np.abs(Cb @ S_mat @ Cb.T - np.eye(n_indep)))
                                                )
                                                if cb_ortho_err > 1e-2:
                                                    errors.append(
                                                        f"{mol_id}/{calc_name}: |C_b S C_b^T - I|_max = "
                                                        f"{cb_ortho_err:.3e} > 1e-2"
                                                    )

                                                if 1 <= n_beta_c <= n_indep:
                                                    homo_beta_ha = float(eps_b[n_beta_c - 1])
                                                if 0 <= n_beta_c < n_indep:
                                                    lumo_beta_ha = float(eps_b[n_beta_c])

                                                homo_eff_ha = (
                                                    max(homo_alpha_ha, homo_beta_ha)
                                                    if math.isfinite(homo_beta_ha) and math.isfinite(homo_alpha_ha)
                                                    else homo_alpha_ha
                                                )
                                                finite_lumos = [
                                                    v for v in (lumo_alpha_ha, lumo_beta_ha) if math.isfinite(v)
                                                ]
                                                lumo_eff_ha = min(finite_lumos) if finite_lumos else float("nan")

                                                if P_alpha is not None and n_beta_c <= n_indep:
                                                    P_beta = (
                                                        Cb[:n_beta_c, :].T @ Cb[:n_beta_c, :]
                                                        if n_beta_c > 0
                                                        else 0.0
                                                    )
                                                    p_recon_err = float(
                                                        np.max(np.abs(P_tot - (P_alpha + P_beta)))
                                                    )
                                                    if p_recon_err > 1e-2:
                                                        errors.append(
                                                            f"{mol_id}/{calc_name}: |P_tot - (P_a + P_b)|_max = "
                                                            f"{p_recon_err:.3e} > 1e-2"
                                                        )

                                        if math.isfinite(homo_eff_ha) and math.isfinite(lumo_eff_ha):
                                            gap_eff_ha = lumo_eff_ha - homo_eff_ha
                except Exception as exc:
                    errors.append(f"{mol_id}/{calc_name}: Error loading/validating NPZ: {type(exc).__name__}: {exc}")

            e_per_electron_ha = (
                scf_energy_ha / n_elec_c
                if math.isfinite(scf_energy_ha) and n_elec_c > 0
                else float("nan")
            )

            raw_calc_entries.append({
                "mol_id": mol_id,
                "calc_name": calc_name,
                "method": method,
                "basis": basis,
                "wf_type": wf_type,
                "n_atoms": n_atoms_c,
                "charge": charge_c,
                "multiplicity": mult_c,
                "n_electrons": n_elec_c,
                "n_alpha": n_alpha_c,
                "n_beta": n_beta_c,
                "n_bas": n_bas,
                "n_indep": n_indep,
                "scf_energy_ha": scf_energy_ha,
                "e_per_electron_ha": e_per_electron_ha,
                "nuc_rep_energy_ha": nuc_rep_ha,
                "s2": s2_val,
                "s2_exact": s2_exact,
                "delta_s2": delta_s2,
                "homo_alpha_ha": homo_alpha_ha,
                "lumo_alpha_ha": lumo_alpha_ha,
                "homo_beta_ha": homo_beta_ha,
                "lumo_beta_ha": lumo_beta_ha,
                "homo_eff_ha": homo_eff_ha,
                "lumo_eff_ha": lumo_eff_ha,
                "gap_eff_ha": gap_eff_ha,
                "max_force_ha_bohr": max_force_ha_bohr,
                "rms_force_ha_bohr": rms_force_ha_bohr,
                "n_atom_forces_le_1": n_atom_forces_le_1,
                "net_force_ha_bohr": net_force_ha_bohr,
                "net_torque_ha_rad": net_torque_ha_rad,
                "tr_ps_err": tr_ps_err,
                "tr_pspin_err": tr_pspin_err,
                "ca_ortho_err": ca_ortho_err,
                "cb_ortho_err": cb_ortho_err,
                "p_recon_err": p_recon_err,
                "t_plus_v_err": t_plus_v_err,
                "geom_meta_npz_max_diff_A": geom_meta_npz_max_diff_A,
                "qcout_normal_term": bool(qres["qcout_normal_term"]),
                "qcout_lin_dep": bool(qres["qcout_lin_dep"]),
                "_qcout_unrestricted": bool(qres["qcout_unrestricted"]),
                "_qcout_rohf": bool(qres["qcout_rohf"]),
                "_missing_s2": missing_s2,
            })

        # Geometry-level crowding & filter-overlap evaluation
        has_converged_calc = n_calcs >= 1
        ref_coords_A = first_npz_coords_A if first_npz_coords_A is not None else meta_coords_A
        if ref_coords_A is not None and ref_coords_A.shape[0] >= 2:
            elems_for_crowd = (
                coord_elements
                if len(coord_elements) == ref_coords_A.shape[0]
                else ["?"] * ref_coords_A.shape[0]
            )
            (
                d_min_A,
                min_rcov_ratio,
                crowding_label,
                n_atoms_involved_crowding,
                n_critical_pairs,
            ) = analyze_geometry_crowding(elems_for_crowd, ref_coords_A, factor=0.50)
        else:
            d_min_A = float("nan")
            min_rcov_ratio = float("nan")
            crowding_label = "none"
            n_atoms_involved_crowding = 0
            n_critical_pairs = 0

        pass_crowding_050 = crowding_label == "none"

        finite_fmax = [
            c["max_force_ha_bohr"]
            for c in raw_calc_entries
            if math.isfinite(c["max_force_ha_bohr"])
        ]
        for c in raw_calc_entries:
            c["d_min_A"] = d_min_A
            c["min_rcov_ratio"] = min_rcov_ratio
            c["crowding_label"] = crowding_label
            c["pass_crowding_050"] = pass_crowding_050
            fmax_c = c["max_force_ha_bohr"]
            c["pass_low_strain_calc"] = (
                math.isfinite(d_min_A)
                and math.isfinite(fmax_c)
                and (d_min_A >= 0.60)
                and (fmax_c <= 1.0)
            )
            if math.isfinite(c["scf_energy_ha"]):
                if c["scf_energy_ha"] > 0.0:
                    d_str = f"{d_min_A:.4f} A" if math.isfinite(d_min_A) else "N/A"
                    warnings.append(
                        f"{mol_id}/{c['calc_name']}: Positive SCF energy {c['scf_energy_ha']:.4f} Ha "
                        f"(Q={c['charge']:+d}, 2S+1={c['multiplicity']}, d_min={d_str})"
                    )
                elif c["scf_energy_ha"] < -1500.0:
                    warnings.append(
                        f"{mol_id}/{c['calc_name']}: Very low SCF energy {c['scf_energy_ha']:.4f} Ha"
                    )

        min_force_across_calcs = min(finite_fmax) if finite_fmax else None
        max_force_across_calcs = max(finite_fmax) if finite_fmax else None
        if has_converged_calc and len(finite_fmax) == n_calcs and math.isfinite(d_min_A):
            pass_low_strain_all_calcs = (d_min_A >= 0.60) and all(f <= 1.0 for f in finite_fmax)
        elif has_converged_calc:
            pass_low_strain_all_calcs = False
        else:
            pass_low_strain_all_calcs = None

        any_calc_force_gt_10 = any(f > 10.0 for f in finite_fmax)
        any_calc_lin_dep = any(
            (0 < c["n_indep"] < c["n_bas"]) or c["qcout_lin_dep"]
            for c in raw_calc_entries
        )

        geom_row = {
            "mol_id": mol_id,
            "geometry_hash": geom_hash,
            "has_converged_calc": has_converged_calc,
            "n_calcs": n_calcs,
            "completeness_str": completeness_str,
            "n_atoms": meta_n_atoms,
            "n_heavy_atoms": n_heavy_atoms,
            "elements_str": elements_str,
            "charge": meta_charge,
            "multiplicity": meta_mult,
            "num_alt_structures": num_alt,
            "single_spec_latin_ok": single_spec_latin_ok,
            "alt_specs_same_qm": alt_specs_same_qm,
            "geom_consistent_across_calcs": (
                geom_consistent_across_calcs if n_extra_coords_checked > 0 else None
            ),
            "d_min_A": d_min_A,
            "min_rcov_ratio": min_rcov_ratio,
            "crowding_label": crowding_label,
            "n_atoms_involved_crowding": n_atoms_involved_crowding,
            "n_critical_pairs": n_critical_pairs,
            "min_force_across_calcs": min_force_across_calcs,
            "max_force_across_calcs": max_force_across_calcs,
            "pass_crowding_050": pass_crowding_050,
            "pass_low_strain_all_calcs": pass_low_strain_all_calcs,
            "any_calc_force_gt_10": any_calc_force_gt_10,
            "any_calc_lin_dep": any_calc_lin_dep,
            "_methods_present": sorted(set(methods_list)),
            "_bases_present": sorted(set(bases_list)),
        }

        pooled_forces = (
            np.concatenate(force_norm_chunks)
            if force_norm_chunks
            else np.empty(0, dtype=np.float32)
        )
        return {
            "mol_id": mol_id,
            "errors": errors,
            "warnings": warnings,
            "geom_row": geom_row,
            "calc_rows": raw_calc_entries,
            "atom_force_norms": pooled_forces,
        }
    except Exception as exc:
        errors.append(f"{mol_id}: Unhandled worker exception: {type(exc).__name__}: {exc}")
        return {
            "mol_id": mol_id,
            "errors": errors,
            "warnings": warnings,
            "geom_row": None,
            "calc_rows": [],
            "atom_force_norms": np.empty(0, dtype=np.float32),
        }


def compute_distribution_stats(values):
    """Compute summary statistics and quantiles (including p90) for finite 1D floats."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    pcts = [0.1, 1.0, 5.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.9]
    qvals = np.percentile(arr, pcts)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p0.1": float(qvals[0]),
        "p1": float(qvals[1]),
        "p5": float(qvals[2]),
        "p25": float(qvals[3]),
        "p50": float(qvals[4]),
        "p75": float(qvals[5]),
        "p90": float(qvals[6]),
        "p95": float(qvals[7]),
        "p99": float(qvals[8]),
        "p99.9": float(qvals[9]),
        "max": float(np.max(arr)),
    }


def count_exceeding(values, thresholds=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2)):
    """Count how many finite entries strictly exceed each threshold."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    out = {"total_finite": n}
    for thr in thresholds:
        cnt = int(np.sum(arr > thr))
        out[f"gt_{thr:g}"] = {
            "count": cnt,
            "pct": round(100.0 * cnt / n, 6) if n > 0 else 0.0,
        }
    return out


def build_2x2_overlap(pass_a_flags, pass_b_flags, label_a="crowding_050", label_b="low_strain"):
    """Build a 2x2 filter-overlap dictionary from equal-length boolean sequences."""
    n = len(pass_a_flags)
    both = sum(1 for a, b in zip(pass_a_flags, pass_b_flags) if a and b)
    a_only = sum(1 for a, b in zip(pass_a_flags, pass_b_flags) if a and not b)
    b_only = sum(1 for a, b in zip(pass_a_flags, pass_b_flags) if (not a) and b)
    neither = sum(1 for a, b in zip(pass_a_flags, pass_b_flags) if (not a) and (not b))
    pct = lambda k: round(100.0 * k / n, 4) if n > 0 else 0.0
    return {
        "total": n,
        "label_a": label_a,
        "label_b": label_b,
        "pass_a_total": {"count": both + a_only, "pct": pct(both + a_only)},
        "pass_b_total": {"count": both + b_only, "pct": pct(both + b_only)},
        "both_pass": {"count": both, "pct": pct(both)},
        "a_only": {"count": a_only, "pct": pct(a_only)},
        "b_only": {"count": b_only, "pct": pct(b_only)},
        "neither_pass": {"count": neither, "pct": pct(neither)},
    }


def format_csv_row(row_dict: dict, fieldnames: list) -> dict:
    """Format dictionary values cleanly for CSV export (None/NaN -> empty string)."""
    out = {}
    for key in fieldnames:
        val = row_dict.get(key)
        if val is None:
            out[key] = ""
        elif isinstance(val, bool):
            out[key] = "True" if val else "False"
        elif isinstance(val, (float, np.floating)):
            fval = float(val)
            out[key] = repr(fval) if math.isfinite(fval) else ""
        else:
            out[key] = val
    return out


def sanitize_for_json(obj):
    """Recursively convert NumPy scalars and replace non-finite floats with None."""
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        fv = float(obj)
        return fv if math.isfinite(fv) else None
    return obj


class DatasetVerifier:
    """Single-pass parallel dataset verifier and report/plot generator."""

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.dataset_name = self.dataset_path.name
        self.errors = []
        self.warnings = []

        self.geom_rows = []
        self.calc_rows = []
        self.geometry_hashes = set()
        self.duplicate_hashes = []
        self.pooled_atom_forces = np.empty(0, dtype=np.float32)

        # Energy convenience views used by plotting helpers
        self.all_energies = []
        self.energies_by_combo = defaultdict(list)
        self.energies_by_method = defaultdict(list)
        self.energies_by_basis = defaultdict(list)

        self.elapsed_seconds = 0.0
        self.summary = {}

    def run_single_pass(self, workers=4, limit=0):
        """Verify all mol_* directories in self.dataset_path in one parallel pass."""
        print(f"\n{'=' * 72}")
        print(f"SINGLE-PASS DATASET VERIFICATION: {self.dataset_name}")
        print(f"{'=' * 72}")

        if not self.dataset_path.exists():
            msg = f"Dataset path does not exist: {self.dataset_path}"
            self.errors.append(msg)
            print(f"ERROR: {msg}")
            return False

        mol_dirs = sorted(self.dataset_path.glob("mol_*"))
        if limit and limit > 0:
            mol_dirs = mol_dirs[: int(limit)]
        mol_dir_strs = [str(p) for p in mol_dirs]
        print(
            f"Scanning {len(mol_dir_strs):,} molecule directories "
            f"(workers={max(1, int(workers))})..."
        )

        t0 = time.perf_counter()
        force_chunks = []
        seen_hashes = {}

        if int(workers) <= 1:
            iterator = map(process_molecule_worker, mol_dir_strs)
            for wres in tqdm(iterator, total=len(mol_dir_strs), desc=f"Verifying {self.dataset_name}"):
                self._ingest_worker_result(wres, seen_hashes, force_chunks)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=int(workers)) as executor:
                iterator = executor.map(process_molecule_worker, mol_dir_strs, chunksize=32)
                for wres in tqdm(iterator, total=len(mol_dir_strs), desc=f"Verifying {self.dataset_name}"):
                    self._ingest_worker_result(wres, seen_hashes, force_chunks)

        self.pooled_atom_forces = (
            np.concatenate(force_chunks)
            if force_chunks
            else np.empty(0, dtype=np.float32)
        )
        self.elapsed_seconds = time.perf_counter() - t0
        self.summary = self._build_summary()

        print(
            f"\n✓ Single-pass scan finished in {self.elapsed_seconds:.1f}s "
            f"({len(self.geom_rows):,} geometries, {len(self.calc_rows):,} calculations)"
        )
        return len(self.errors) == 0

    def _ingest_worker_result(self, wres: dict, seen_hashes: dict, force_chunks: list):
        self.errors.extend(wres["errors"])
        self.warnings.extend(wres["warnings"])

        grow = wres["geom_row"]
        if grow is not None:
            ghash = grow["geometry_hash"]
            if ghash:
                if ghash in seen_hashes:
                    first_mol = seen_hashes[ghash]
                    self.duplicate_hashes.append((grow["mol_id"], first_mol, ghash))
                    self.errors.append(
                        f"{grow['mol_id']}: Duplicate geometry_hash {ghash} (already in {first_mol})"
                    )
                else:
                    seen_hashes[ghash] = grow["mol_id"]
                    self.geometry_hashes.add(ghash)
            self.geom_rows.append(grow)

        for crow in wres["calc_rows"]:
            self.calc_rows.append(crow)
            en = crow["scf_energy_ha"]
            if math.isfinite(en):
                ckey = f"{crow['method']}_{crow['basis']}"
                self.all_energies.append(en)
                self.energies_by_combo[ckey].append(en)
                self.energies_by_method[crow["method"]].append(en)
                self.energies_by_basis[crow["basis"]].append(en)

        af = wres["atom_force_norms"]
        if af.size > 0:
            force_chunks.append(af)

    def _build_summary(self) -> dict:
        n_geoms = len(self.geom_rows)
        conv_geoms = [g for g in self.geom_rows if g["has_converged_calc"]]
        n_conv_geoms = len(conv_geoms)
        n_zero_geoms = n_geoms - n_conv_geoms
        n_calcs_total = len(self.calc_rows)

        n_calcs_dist = Counter(g["n_calcs"] for g in self.geom_rows)
        alt_geoms = [g for g in self.geom_rows if g["num_alt_structures"] > 1]
        single_geoms = [g for g in self.geom_rows if g["num_alt_structures"] == 1]

        # Method / basis counts and coverage
        method_counts = Counter(c["method"] for c in self.calc_rows)
        basis_counts = Counter(c["basis"] for c in self.calc_rows)
        combo_counts = Counter(f"{c['method']}_{c['basis']}" for c in self.calc_rows)
        incomplete_method_cov = sum(
            1 for g in self.geom_rows if set(g["_methods_present"]) != EXPECTED_METHODS
        )
        incomplete_basis_cov = sum(
            1 for g in self.geom_rows if set(g["_bases_present"]) != EXPECTED_BASES
        )

        # Energy distributions & outliers
        e_per_elec_vals = [
            c["e_per_electron_ha"] for c in self.calc_rows if math.isfinite(c["e_per_electron_ha"])
        ]
        nuc_rep_vals = [
            c["nuc_rep_energy_ha"] for c in self.calc_rows if math.isfinite(c["nuc_rep_energy_ha"])
        ]
        pos_energy_cases = [
            {
                "mol_id": c["mol_id"],
                "calc_name": c["calc_name"],
                "scf_energy_ha": c["scf_energy_ha"],
                "charge": c["charge"],
                "multiplicity": c["multiplicity"],
                "n_atoms": c["n_atoms"],
                "n_electrons": c["n_electrons"],
                "d_min_A": c["d_min_A"],
                "min_rcov_ratio": c["min_rcov_ratio"],
            }
            for c in self.calc_rows
            if math.isfinite(c["scf_energy_ha"]) and c["scf_energy_ha"] > 0.0
        ]
        pos_energy_cases.sort(key=lambda x: x["scf_energy_ha"], reverse=True)

        very_low_energy_cases = [
            {
                "mol_id": c["mol_id"],
                "calc_name": c["calc_name"],
                "scf_energy_ha": c["scf_energy_ha"],
                "d_min_A": c["d_min_A"],
            }
            for c in self.calc_rows
            if math.isfinite(c["scf_energy_ha"]) and c["scf_energy_ha"] < -1500.0
        ]
        e_per_elec_outliers = [
            {
                "mol_id": c["mol_id"],
                "calc_name": c["calc_name"],
                "scf_energy_ha": c["scf_energy_ha"],
                "e_per_electron_ha": c["e_per_electron_ha"],
                "n_electrons": c["n_electrons"],
                "charge": c["charge"],
                "multiplicity": c["multiplicity"],
                "d_min_A": c["d_min_A"],
            }
            for c in self.calc_rows
            if math.isfinite(c["e_per_electron_ha"])
            and (c["e_per_electron_ha"] > 0.0 or c["e_per_electron_ha"] < -40.0)
        ]

        energy_by_method_stats = {}
        for m in METHODS:
            vals = self.energies_by_method.get(m, [])
            if vals:
                energy_by_method_stats[m] = {
                    "stats": compute_distribution_stats(vals),
                    "positive_count": sum(1 for v in vals if v > 0.0),
                }

        energy_by_basis_stats = {}
        for b in BASIS_SETS:
            vals = self.energies_by_basis.get(b, [])
            if vals:
                energy_by_basis_stats[b] = {
                    "stats": compute_distribution_stats(vals),
                    "positive_count": sum(1 for v in vals if v > 0.0),
                }

        energy_by_combo_stats = {}
        for m in METHODS:
            for b in BASIS_SETS:
                ckey = f"{m}_{b}"
                vals = self.energies_by_combo.get(ckey, [])
                if vals:
                    energy_by_combo_stats[ckey] = {
                        "stats": compute_distribution_stats(vals),
                        "positive_count": sum(1 for v in vals if v > 0.0),
                    }

        # Linear dependence (n_indep < n_bas)
        lin_dep_cases = [
            {
                "mol_id": c["mol_id"],
                "calc_name": c["calc_name"],
                "method": c["method"],
                "basis": c["basis"],
                "n_bas": c["n_bas"],
                "n_indep": c["n_indep"],
                "d_min_A": c["d_min_A"],
                "min_rcov_ratio": c["min_rcov_ratio"],
                "ca_ortho_err": c["ca_ortho_err"],
                "qcout_lin_dep": c["qcout_lin_dep"],
            }
            for c in self.calc_rows
            if 0 < c["n_indep"] < c["n_bas"]
        ]

        # Matrix & geometry consistency metrics
        tr_ps_vals = [c["tr_ps_err"] for c in self.calc_rows if math.isfinite(c["tr_ps_err"])]
        tr_pspin_vals = [c["tr_pspin_err"] for c in self.calc_rows if math.isfinite(c["tr_pspin_err"])]
        ca_ortho_vals = [c["ca_ortho_err"] for c in self.calc_rows if math.isfinite(c["ca_ortho_err"])]
        cb_ortho_vals = [c["cb_ortho_err"] for c in self.calc_rows if math.isfinite(c["cb_ortho_err"])]
        max_mo_ortho_vals = []
        for c in self.calc_rows:
            ca_e = c["ca_ortho_err"]
            cb_e = c["cb_ortho_err"]
            if math.isfinite(ca_e) and math.isfinite(cb_e):
                max_mo_ortho_vals.append(max(ca_e, cb_e))
            elif math.isfinite(ca_e):
                max_mo_ortho_vals.append(ca_e)
        p_recon_vals = [c["p_recon_err"] for c in self.calc_rows if math.isfinite(c["p_recon_err"])]
        tv_vals = [c["t_plus_v_err"] for c in self.calc_rows if math.isfinite(c["t_plus_v_err"])]
        geom_diff_vals = [
            c["geom_meta_npz_max_diff_A"]
            for c in self.calc_rows
            if math.isfinite(c["geom_meta_npz_max_diff_A"])
        ]
        net_f_vals = [c["net_force_ha_bohr"] for c in self.calc_rows if math.isfinite(c["net_force_ha_bohr"])]
        net_t_vals = [c["net_torque_ha_rad"] for c in self.calc_rows if math.isfinite(c["net_torque_ha_rad"])]

        worst_ortho = sorted(
            [
                c for c in self.calc_rows if math.isfinite(c["ca_ortho_err"])
            ],
            key=lambda c: (
                max(c["ca_ortho_err"], c["cb_ortho_err"])
                if math.isfinite(c["cb_ortho_err"])
                else c["ca_ortho_err"]
            ),
            reverse=True,
        )[:10]
        worst_ortho_summary = [
            {
                "mol_id": c["mol_id"],
                "calc_name": c["calc_name"],
                "ca_ortho_err": c["ca_ortho_err"],
                "cb_ortho_err": c["cb_ortho_err"],
                "n_bas": c["n_bas"],
                "n_indep": c["n_indep"],
                "d_min_A": c["d_min_A"],
            }
            for c in worst_ortho
        ]

        multi_calc_geoms = [
            g for g in self.geom_rows if g["geom_consistent_across_calcs"] is not None
        ]
        inconsistent_geoms = [
            g["mol_id"] for g in multi_calc_geoms if g["geom_consistent_across_calcs"] is False
        ]

        # HOMO > 0 analysis (effective HOMO and alpha HOMO)
        def summarize_homo_subset(rows):
            valid_eff = [c for c in rows if math.isfinite(c["homo_eff_ha"])]
            valid_a = [c for c in rows if math.isfinite(c["homo_alpha_ha"])]
            n_eff = len(valid_eff)
            pos_eff = sum(1 for c in valid_eff if c["homo_eff_ha"] > 0.0)
            n_a = len(valid_a)
            pos_a = sum(1 for c in valid_a if c["homo_alpha_ha"] > 0.0)
            return {
                "count": n_eff,
                "homo_eff_gt_0_count": pos_eff,
                "homo_eff_gt_0_pct": round(100.0 * pos_eff / n_eff, 4) if n_eff > 0 else 0.0,
                "homo_alpha_gt_0_count": pos_a,
                "homo_alpha_gt_0_pct": round(100.0 * pos_a / n_a, 4) if n_a > 0 else 0.0,
                "homo_eff_stats": compute_distribution_stats([c["homo_eff_ha"] for c in valid_eff]),
            }

        charges_present = sorted({c["charge"] for c in self.calc_rows})
        homo_by_charge = {
            f"{q:+d}": summarize_homo_subset([c for c in self.calc_rows if c["charge"] == q])
            for q in charges_present
        }
        anion_rows = [c for c in self.calc_rows if c["charge"] < 0]
        non_anion_rows = [c for c in self.calc_rows if c["charge"] >= 0]
        neutral_rows = [c for c in self.calc_rows if c["charge"] == 0]
        cation_rows = [c for c in self.calc_rows if c["charge"] > 0]

        homo_by_combo_all = {}
        homo_by_combo_anions = {}
        for m in METHODS:
            for b in BASIS_SETS:
                ckey = f"{m}_{b}"
                homo_by_combo_all[ckey] = summarize_homo_subset(
                    [c for c in self.calc_rows if c["method"] == m and c["basis"] == b]
                )
                homo_by_combo_anions[ckey] = summarize_homo_subset(
                    [c for c in anion_rows if c["method"] == m and c["basis"] == b]
                )

        gap_vals = [c["gap_eff_ha"] for c in self.calc_rows if math.isfinite(c["gap_eff_ha"])]

        # Spin contamination (<S^2> and delta_s2 on mult > 1)
        open_rows = [c for c in self.calc_rows if c["multiplicity"] > 1]

        def summarize_spin_subset(rows):
            valid = [c["delta_s2"] for c in rows if math.isfinite(c["delta_s2"])]
            abs_v = [abs(v) for v in valid]
            n_v = len(valid)
            out_s = {
                "total_open_calcs": len(rows),
                "parsed_s2_count": n_v,
                "missing_s2_count": sum(1 for c in rows if c["_missing_s2"]),
                "unrestricted_confirmed_count": sum(1 for c in rows if c["_qcout_unrestricted"]),
                "rohf_detected_count": sum(1 for c in rows if c["_qcout_rohf"]),
                "delta_s2_stats": compute_distribution_stats(valid),
                "abs_delta_s2_stats": compute_distribution_stats(abs_v),
            }
            for thr in (0.05, 0.1, 0.5):
                cnt = sum(1 for v in abs_v if v > thr)
                out_s[f"abs_delta_s2_gt_{thr}"] = {
                    "count": cnt,
                    "pct": round(100.0 * cnt / n_v, 4) if n_v > 0 else 0.0,
                }
            return out_s

        spin_by_wf = {
            wf: summarize_spin_subset([c for c in open_rows if c["wf_type"] == wf])
            for wf in ("UHF", "UKS")
        }
        spin_by_method = {
            m: summarize_spin_subset([c for c in open_rows if c["method"] == m])
            for m in METHODS
        }
        open_mults = sorted({c["multiplicity"] for c in open_rows})
        spin_by_mult = {
            str(mv): summarize_spin_subset([c for c in open_rows if c["multiplicity"] == mv])
            for mv in open_mults
        }

        # Pooled per-atom forces (Table 2) & calculation/geometry force outliers
        n_atom_forces = int(self.pooled_atom_forces.size)
        n_atom_forces_le_1_tot = int(np.sum(self.pooled_atom_forces <= 1.0)) if n_atom_forces > 0 else 0
        pooled_force_stats = compute_distribution_stats(self.pooled_atom_forces)

        force_outliers_calc = {}
        force_outliers_geom = {}
        for f_thr in (10.0, 50.0, 100.0, 200.0, 300.0):
            c_hits = [
                c for c in self.calc_rows
                if math.isfinite(c["max_force_ha_bohr"]) and c["max_force_ha_bohr"] > f_thr
            ]
            g_hits = [
                g for g in conv_geoms
                if g["max_force_across_calcs"] is not None and g["max_force_across_calcs"] > f_thr
            ]
            key_f = f"gt_{int(f_thr)}"
            c_dmins = [c["d_min_A"] for c in c_hits if math.isfinite(c["d_min_A"])]
            g_dmins = [g["d_min_A"] for g in g_hits if math.isfinite(g["d_min_A"])]
            force_outliers_calc[key_f] = {
                "count": len(c_hits),
                "pct": round(100.0 * len(c_hits) / n_calcs_total, 4) if n_calcs_total > 0 else 0.0,
                "median_d_min_A": float(np.median(c_dmins)) if c_dmins else None,
            }
            force_outliers_geom[key_f] = {
                "count": len(g_hits),
                "pct": round(100.0 * len(g_hits) / n_conv_geoms, 4) if n_conv_geoms > 0 else 0.0,
                "median_d_min_A": float(np.median(g_dmins)) if g_dmins else None,
            }

        # Crowding classification (Table 3: geometries with >= 1 converged calculation)
        crowd_conv_counts = Counter(g["crowding_label"] for g in conv_geoms)
        crowd_all_counts = Counter(g["crowding_label"] for g in self.geom_rows)
        crowding_table3 = {
            "evaluated_geometries_with_converged_calc": n_conv_geoms,
            "excluded_zero_calc_geometries": n_zero_geoms,
        }
        for clbl in ("none", "clean_pair", "crowded_cluster"):
            cnt_c = crowd_conv_counts.get(clbl, 0)
            cl_dmins = [
                g["d_min_A"] for g in conv_geoms
                if g["crowding_label"] == clbl and math.isfinite(g["d_min_A"])
            ]
            crowding_table3[clbl] = {
                "count": cnt_c,
                "pct": round(100.0 * cnt_c / n_conv_geoms, 4) if n_conv_geoms > 0 else 0.0,
                "median_d_min_A": float(np.median(cl_dmins)) if cl_dmins else None,
                "all_geoms_including_zero_calc_count": crowd_all_counts.get(clbl, 0),
            }

        # 2x2 Filter overlap tables
        calc_overlap_2x2 = build_2x2_overlap(
            [bool(c["pass_crowding_050"]) for c in self.calc_rows],
            [bool(c["pass_low_strain_calc"]) for c in self.calc_rows],
            label_a="pass_crowding_050 (r_ij/r_cov >= 0.50)",
            label_b="pass_low_strain_calc (d_min >= 0.60 A & F_max <= 1.0 Ha/Bohr)",
        )
        geom_overlap_2x2 = build_2x2_overlap(
            [bool(g["pass_crowding_050"]) for g in conv_geoms],
            [bool(g["pass_low_strain_all_calcs"]) for g in conv_geoms],
            label_a="pass_crowding_050 (r_ij/r_cov >= 0.50)",
            label_b="pass_low_strain_all_calcs (d_min >= 0.60 A & all F_max <= 1.0 Ha/Bohr)",
        )

        return {
            "dataset_name": self.dataset_name,
            "dataset_path": str(self.dataset_path),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "counts": {
                "total_molecules": n_geoms,
                "molecules_with_converged_calcs": n_conv_geoms,
                "molecules_with_zero_calcs": n_zero_geoms,
                "total_calculations": n_calcs_total,
                "n_calcs_per_molecule_distribution": {
                    str(k): n_calcs_dist[k] for k in sorted(n_calcs_dist.keys())
                },
                "unique_geometry_hashes": len(self.geometry_hashes),
                "duplicate_geometry_hashes": len(self.duplicate_hashes),
                "molecules_with_alternatives": len(alt_geoms),
                "max_alternatives": max((g["num_alt_structures"] for g in self.geom_rows), default=0),
                "alt_geoms_same_qm": sum(1 for g in alt_geoms if g["alt_specs_same_qm"] is True),
                "alt_geoms_diff_qm": sum(1 for g in alt_geoms if g["alt_specs_same_qm"] is False),
                "single_spec_molecules": len(single_geoms),
                "single_spec_latin_violations": sum(
                    1 for g in single_geoms if g["single_spec_latin_ok"] is False
                ),
                "errors_count": len(self.errors),
                "warnings_count": len(self.warnings),
            },
            "method_basis": {
                "by_method": {m: method_counts.get(m, 0) for m in METHODS},
                "by_basis": {b: basis_counts.get(b, 0) for b in BASIS_SETS},
                "by_combination": {
                    f"{m}_{b}": combo_counts.get(f"{m}_{b}", 0)
                    for m in METHODS
                    for b in BASIS_SETS
                },
                "incomplete_method_coverage_molecules": incomplete_method_cov,
                "incomplete_basis_coverage_molecules": incomplete_basis_cov,
            },
            "energy_validation": {
                "overall_scf_energy_ha": compute_distribution_stats(self.all_energies),
                "e_per_electron_ha": compute_distribution_stats(e_per_elec_vals),
                "nuclear_repulsion_energy_ha": compute_distribution_stats(nuc_rep_vals),
                "by_method": energy_by_method_stats,
                "by_basis": energy_by_basis_stats,
                "by_combination": energy_by_combo_stats,
                "positive_scf_energy_count": len(pos_energy_cases),
                "positive_scf_energy_cases": pos_energy_cases,
                "very_low_scf_energy_count": len(very_low_energy_cases),
                "very_low_scf_energy_cases": very_low_energy_cases,
                "e_per_electron_outliers_count": len(e_per_elec_outliers),
                "e_per_electron_outliers": e_per_elec_outliers[:100],
            },
            "npz_and_matrix_validation": {
                "qcout_missing_normal_termination_count": sum(
                    1 for c in self.calc_rows if not c["qcout_normal_term"]
                ),
                "n_indep_lt_n_bas_count": len(lin_dep_cases),
                "n_indep_lt_n_bas_cases": lin_dep_cases,
                "qcout_lin_dep_flag_count": sum(1 for c in self.calc_rows if c["qcout_lin_dep"]),
                "geometry_consistency_across_calcs": {
                    "molecules_with_ge_2_calcs": len(multi_calc_geoms),
                    "inconsistent_molecules_count": len(inconsistent_geoms),
                    "inconsistent_molecule_ids": inconsistent_geoms[:50],
                },
                "geom_meta_vs_npz_distance_diff_A": {
                    "stats": compute_distribution_stats(geom_diff_vals),
                    "thresholds": count_exceeding(geom_diff_vals, (1e-5, 1e-4, 1e-3, 1e-2)),
                },
                "tr_ps_minus_nelec_abs_err": {
                    "stats": compute_distribution_stats(tr_ps_vals),
                    "thresholds": count_exceeding(tr_ps_vals, (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)),
                },
                "tr_pspin_minus_na_nb_abs_err": {
                    "stats": compute_distribution_stats(tr_pspin_vals),
                    "thresholds": count_exceeding(tr_pspin_vals, (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)),
                },
                "ca_s_cat_ortho_max_err": {
                    "stats": compute_distribution_stats(ca_ortho_vals),
                    "thresholds": count_exceeding(ca_ortho_vals, (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)),
                },
                "cb_s_cbt_ortho_max_err": {
                    "stats": compute_distribution_stats(cb_ortho_vals),
                    "thresholds": count_exceeding(cb_ortho_vals, (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)),
                },
                "max_mo_ortho_err_combined": {
                    "stats": compute_distribution_stats(max_mo_ortho_vals),
                    "thresholds": count_exceeding(max_mo_ortho_vals, (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)),
                    "top_10_worst_cases": worst_ortho_summary,
                },
                "density_reconstruction_from_mos_max_err": {
                    "stats": compute_distribution_stats(p_recon_vals),
                    "thresholds": count_exceeding(p_recon_vals, (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)),
                },
                "t_plus_v_minus_hcore_max_err": {
                    "stats": compute_distribution_stats(tv_vals),
                    "thresholds": count_exceeding(tv_vals, (1e-6, 1e-5, 1e-4, 1e-3)),
                },
                "net_force_norm_ha_bohr": {
                    "stats": compute_distribution_stats(net_f_vals),
                    "thresholds": count_exceeding(net_f_vals, (1e-5, 1e-4, 1e-3, 1e-2)),
                },
                "net_torque_norm_ha_rad": {
                    "stats": compute_distribution_stats(net_t_vals),
                    "thresholds": count_exceeding(net_t_vals, (1e-5, 1e-4, 1e-3, 1e-2)),
                },
            },
            "homo_lumo_diagnostics": {
                "overall": summarize_homo_subset(self.calc_rows),
                "anions_Q_lt_0": summarize_homo_subset(anion_rows),
                "non_anions_Q_ge_0": summarize_homo_subset(non_anion_rows),
                "neutral_Q_eq_0": summarize_homo_subset(neutral_rows),
                "cations_Q_gt_0": summarize_homo_subset(cation_rows),
                "by_charge": homo_by_charge,
                "by_combination_all_charges": homo_by_combo_all,
                "by_combination_anions_Q_lt_0": homo_by_combo_anions,
                "homo_lumo_gap_eff_ha": {
                    "stats": compute_distribution_stats(gap_vals),
                    "negative_gap_count": sum(1 for g_v in gap_vals if g_v < 0.0),
                },
            },
            "spin_contamination": {
                "overall_open_shell": summarize_spin_subset(open_rows),
                "by_wf_type": spin_by_wf,
                "by_method": spin_by_method,
                "by_multiplicity": spin_by_mult,
            },
            "forces_crowding_and_filters": {
                "pooled_per_atom_force_magnitudes_ha_bohr": {
                    "stats": pooled_force_stats,
                    "le_1_0_ha_bohr": {
                        "count": n_atom_forces_le_1_tot,
                        "pct": (
                            round(100.0 * n_atom_forces_le_1_tot / n_atom_forces, 4)
                            if n_atom_forces > 0
                            else 0.0
                        ),
                    },
                },
                "force_outliers_by_calculation": force_outliers_calc,
                "force_outliers_by_geometry": force_outliers_geom,
                "crowding_table3_factor_050": crowding_table3,
                "filter_overlap_2x2_calculation_level": calc_overlap_2x2,
                "filter_overlap_2x2_geometry_level": geom_overlap_2x2,
            },
        }

    def save_all_outputs(self, output_dir, bins=80, make_plot=True):
        """
        Save compressed raw CSV tables, JSON summary, text reports, and 3x3 PNG histogram.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Raw per-calculation CSV.gz
        calc_csv_path = out_dir / f"calc_metrics_{self.dataset_name}.csv.gz"
        with gzip.open(calc_csv_path, "wt", encoding="utf-8", newline="") as gz_calc:
            writer = csv.DictWriter(gz_calc, fieldnames=CALC_CSV_FIELDS)
            writer.writeheader()
            for crow in self.calc_rows:
                writer.writerow(format_csv_row(crow, CALC_CSV_FIELDS))
        print(f"  ✓ Saved per-calculation raw metrics: {calc_csv_path}")

        # 2. Raw per-geometry CSV.gz
        geom_csv_path = out_dir / f"geom_metrics_{self.dataset_name}.csv.gz"
        with gzip.open(geom_csv_path, "wt", encoding="utf-8", newline="") as gz_geom:
            writer = csv.DictWriter(gz_geom, fieldnames=GEOM_CSV_FIELDS)
            writer.writeheader()
            for grow in self.geom_rows:
                writer.writerow(format_csv_row(grow, GEOM_CSV_FIELDS))
        print(f"  ✓ Saved per-geometry raw metrics:    {geom_csv_path}")

        # 3. Structured JSON summary
        json_path = out_dir / f"verification_summary_{self.dataset_name}.json"
        clean_summary = sanitize_for_json(self.summary)
        with open(json_path, "w", encoding="utf-8") as jfh:
            json.dump(clean_summary, jfh, indent=2, allow_nan=False)
        print(f"  ✓ Saved JSON verification summary:   {json_path}")

        # 4. Human-readable energy report & comprehensive verification report
        energy_txt_path = self._write_energy_report_txt(out_dir)
        report_txt_path = self._write_full_verification_report_txt(out_dir)

        # 5. 3x3 equal-width log-y histogram PNG (300 DPI)
        fig_path = None
        if make_plot and self.all_energies:
            fig_path = self._plot_energy_grid_3x3(out_dir, bins=bins)

        return {
            "calc_csv": calc_csv_path,
            "geom_csv": geom_csv_path,
            "summary_json": json_path,
            "energy_txt": energy_txt_path,
            "report_txt": report_txt_path,
            "energy_png": fig_path,
        }

    def _write_energy_report_txt(self, out_dir: Path) -> Path:
        stats_file = out_dir / f"energy_stats_{self.dataset_name}.txt"
        ev = self.summary.get("energy_validation", {})
        estats = ev.get("overall_scf_energy_ha")
        ep_stats = ev.get("e_per_electron_ha")

        with open(stats_file, "w", encoding="utf-8") as fh:
            fh.write(f"Energy Validation Report: {self.dataset_name}\n")
            fh.write("=" * 96 + "\n\n")
            if not estats:
                fh.write("No valid SCF energies parsed.\n")
                return stats_file

            fh.write(f"OVERALL SCF ENERGY (n={estats['count']:,}):\n")
            fh.write(f"  Mean ± Std : {estats['mean']:12.4f} ± {estats['std']:.4f} Ha\n")
            fh.write(f"  Min / Max  : {estats['min']:12.4f} / {estats['max']:.4f} Ha\n")
            for qk in ["p0.1", "p1", "p5", "p25", "p50", "p75", "p90", "p95", "p99", "p99.9"]:
                fh.write(f"  {qk:<10} : {estats[qk]:12.4f} Ha\n")
            fh.write(f"  E > 0 Ha   : {ev['positive_scf_energy_count']} calculations\n")
            fh.write(f"  E < -1500  : {ev['very_low_scf_energy_count']} calculations\n\n")

            if ep_stats:
                fh.write(f"ENERGY PER ELECTRON E / N_e (n={ep_stats['count']:,}):\n")
                fh.write(f"  Mean ± Std : {ep_stats['mean']:12.4f} ± {ep_stats['std']:.4f} Ha/e-\n")
                fh.write(f"  Min / Max  : {ep_stats['min']:12.4f} / {ep_stats['max']:.4f} Ha/e-\n")
                for qk in ["p0.1", "p1", "p5", "p50", "p95", "p99", "p99.9"]:
                    fh.write(f"  {qk:<10} : {ep_stats[qk]:12.4f} Ha/e-\n")
                fh.write(
                    f"  Outliers outside [-40, 0] Ha/e-: {ev['e_per_electron_outliers_count']}\n\n"
                )

            header = (
                f"{'Group':<18} {'Count':>8} {'Min':>11} {'p1':>10} {'p5':>10} "
                f"{'Median':>10} {'p95':>10} {'p99':>10} {'Max':>10} {'E>0':>6}"
            )
            for sec_title, mapping, keys in (
                ("BY METHOD", ev["by_method"], METHODS),
                ("BY BASIS SET", ev["by_basis"], BASIS_SETS),
                (
                    "BY METHOD-BASIS COMBINATION (3x3)",
                    ev["by_combination"],
                    [f"{m}_{b}" for m in METHODS for b in BASIS_SETS],
                ),
            ):
                fh.write(f"{sec_title}:\n{header}\n" + "-" * 96 + "\n")
                for k in keys:
                    if k in mapping:
                        st = mapping[k]["stats"]
                        n_pos = mapping[k]["positive_count"]
                        fh.write(
                            f"{k:<18} {st['count']:>8d} {st['min']:>11.2f} {st['p1']:>10.2f} "
                            f"{st['p5']:>10.2f} {st['p50']:>10.2f} {st['p95']:>10.2f} "
                            f"{st['p99']:>10.2f} {st['max']:>10.2f} {n_pos:>6d}\n"
                        )
                fh.write("\n")

            pos_cases = ev.get("positive_scf_energy_cases", [])
            if pos_cases:
                fh.write(f"POSITIVE SCF ENERGY CASES (n={len(pos_cases)}):\n")
                for item in pos_cases:
                    d_val = item["d_min_A"]
                    d_str = f"{d_val:.4f}" if d_val is not None and math.isfinite(d_val) else "N/A"
                    fh.write(
                        f"  {item['mol_id']}  {item['calc_name']:<18}  "
                        f"E={item['scf_energy_ha']:10.4f} Ha  Q={item['charge']:+d}  "
                        f"2S+1={item['multiplicity']}  N_e={item['n_electrons']}  d_min={d_str} A\n"
                    )

        print(f"  ✓ Saved energy statistics report:    {stats_file}")
        return stats_file

    def _write_full_verification_report_txt(self, out_dir: Path) -> Path:
        rep_file = out_dir / f"verification_report_{self.dataset_name}.txt"
        s = self.summary
        cnts = s.get("counts", {})
        mb = s.get("method_basis", {})
        nm = s.get("npz_and_matrix_validation", {})
        hl = s.get("homo_lumo_diagnostics", {})
        sp = s.get("spin_contamination", {})
        fc = s.get("forces_crowding_and_filters", {})

        with open(rep_file, "w", encoding="utf-8") as fh:
            fh.write(f"Comprehensive Technical Validation Report: {self.dataset_name}\n")
            fh.write("=" * 96 + "\n")
            fh.write(f"Dataset path    : {self.dataset_path}\n")
            fh.write(f"Elapsed time    : {self.elapsed_seconds:.2f} s\n")
            fh.write(f"Total errors    : {cnts.get('errors_count', 0)}\n")
            fh.write(f"Total warnings  : {cnts.get('warnings_count', 0)}\n\n")

            fh.write("1. DATASET & STRUCTURE INTEGRITY\n" + "-" * 96 + "\n")
            fh.write(f"  Total molecule directories           : {cnts.get('total_molecules', 0):,}\n")
            fh.write(f"  Geometries with >=1 converged calc   : {cnts.get('molecules_with_converged_calcs', 0):,}\n")
            fh.write(f"  Geometries with 0 converged calcs    : {cnts.get('molecules_with_zero_calcs', 0):,}\n")
            fh.write(f"  Total converged calculations         : {cnts.get('total_calculations', 0):,}\n")
            fh.write(f"  n_calcs distribution                 : {cnts.get('n_calcs_per_molecule_distribution', {})}\n")
            fh.write(f"  Unique geometry hashes               : {cnts.get('unique_geometry_hashes', 0):,}\n")
            fh.write(f"  Duplicate geometry hashes            : {cnts.get('duplicate_geometry_hashes', 0)}\n")
            fh.write(
                f"  Molecules with alternative specs     : {cnts.get('molecules_with_alternatives', 0):,} "
                f"(max={cnts.get('max_alternatives', 0)}, same_QM={cnts.get('alt_geoms_same_qm', 0):,}, "
                f"diff_QM={cnts.get('alt_geoms_diff_qm', 0):,})\n"
            )
            fh.write(
                f"  Single-spec Latin-square violations  : {cnts.get('single_spec_latin_violations', 0)}\n"
            )
            fh.write(f"  Method counts                        : {mb.get('by_method', {})}\n")
            fh.write(f"  Basis counts                         : {mb.get('by_basis', {})}\n\n")

            fh.write("2. NPZ SCHEMA, BASIS & MATRIX IDENTITIES\n" + "-" * 96 + "\n")
            fh.write(
                f"  Missing normal Q-Chem termination    : {nm.get('qcout_missing_normal_termination_count', 0)}\n"
            )
            fh.write(
                f"  Linearly dependent AO cases (N_i<N_b): {nm.get('n_indep_lt_n_bas_count', 0)} "
                f"(.qcout lin-dep flag: {nm.get('qcout_lin_dep_flag_count', 0)})\n"
            )
            for lcase in nm.get("n_indep_lt_n_bas_cases", [])[:25]:
                fh.write(
                    f"    - {lcase['mol_id']}/{lcase['calc_name']}: N_bas={lcase['n_bas']}, "
                    f"N_indep={lcase['n_indep']}, d_min={lcase['d_min_A']:.4f} A, "
                    f"ca_ortho_err={lcase['ca_ortho_err']:.3e}\n"
                )
            gcons = nm.get("geometry_consistency_across_calcs", {})
            fh.write(
                f"  Cross-calc coordinate consistency    : "
                f"{gcons.get('inconsistent_molecules_count', 0)} mismatches / "
                f"{gcons.get('molecules_with_ge_2_calcs', 0):,} multi-calc molecules\n"
            )

            for label_m, m_key in (
                ("|D_npz - D_meta|_max (A)", "geom_meta_vs_npz_distance_diff_A"),
                ("|Tr(P_tot S) - N_e|", "tr_ps_minus_nelec_abs_err"),
                ("|Tr(P_spin S) - (N_a - N_b)|", "tr_pspin_minus_na_nb_abs_err"),
                ("|C_a S C_a^T - I|_max", "ca_s_cat_ortho_max_err"),
                ("|C_b S C_b^T - I|_max", "cb_s_cbt_ortho_max_err"),
                ("max MO ortho err (a/b)", "max_mo_ortho_err_combined"),
                ("|P_tot - P_recon|_max", "density_reconstruction_from_mos_max_err"),
                ("|T + V - H_core|_max", "t_plus_v_minus_hcore_max_err"),
                ("||F_net|| (Ha/Bohr)", "net_force_norm_ha_bohr"),
                ("||tau_net|| (Ha/rad)", "net_torque_norm_ha_rad"),
            ):
                entry = nm.get(m_key, {})
                st = entry.get("stats")
                th = entry.get("thresholds", {})
                if st:
                    gt_1e4 = th.get("gt_0.0001", {}).get("count", 0)
                    gt_1e3 = th.get("gt_0.001", {}).get("count", 0)
                    fh.write(
                        f"  {label_m:<30}: p50={st['p50']:.3e}, p99={st['p99']:.3e}, "
                        f"max={st['max']:.3e}  (>1e-4: {gt_1e4}, >1e-3: {gt_1e3})\n"
                    )
            fh.write("\n")

            fh.write("3. HOMO > 0 DIAGNOSTICS (ANIONS VS NON-ANIONS)\n" + "-" * 96 + "\n")
            for grp_lbl, grp_key in (
                ("All calculations", "overall"),
                ("Anions (Q < 0)", "anions_Q_lt_0"),
                ("Neutral (Q = 0)", "neutral_Q_eq_0"),
                ("Cations (Q > 0)", "cations_Q_gt_0"),
            ):
                hd = hl.get(grp_key, {})
                fh.write(
                    f"  {grp_lbl:<22}: n={hd.get('count', 0):>8,}, "
                    f"HOMO_eff > 0: {hd.get('homo_eff_gt_0_count', 0):>7,} "
                    f"({hd.get('homo_eff_gt_0_pct', 0.0):6.2f}%)\n"
                )
            fh.write("  By formal charge Q:\n")
            for q_str, hd in hl.get("by_charge", {}).items():
                fh.write(
                    f"    Q={q_str:<4}: n={hd['count']:>8,}, "
                    f"HOMO_eff > 0: {hd['homo_eff_gt_0_count']:>7,} ({hd['homo_eff_gt_0_pct']:6.2f}%)\n"
                )
            fh.write("  Anions (Q < 0) by method_basis:\n")
            for ckey, hd in hl.get("by_combination_anions_Q_lt_0", {}).items():
                fh.write(
                    f"    {ckey:<18}: n={hd['count']:>7,}, "
                    f"HOMO_eff > 0: {hd['homo_eff_gt_0_count']:>6,} ({hd['homo_eff_gt_0_pct']:6.2f}%)\n"
                )
            fh.write("\n")

            fh.write("4. OPEN-SHELL SPIN CONTAMINATION (<S^2> - S(S+1))\n" + "-" * 96 + "\n")
            sp_all = sp.get("overall_open_shell", {})
            fh.write(
                f"  Total open-shell calcs: {sp_all.get('total_open_calcs', 0):,} "
                f"(parsed <S^2>: {sp_all.get('parsed_s2_count', 0):,}, "
                f"missing <S^2>: {sp_all.get('missing_s2_count', 0)}, "
                f"ROHF flags: {sp_all.get('rohf_detected_count', 0)})\n"
            )
            for lbl_s, sdict in [
                ("Overall Open-Shell", sp_all),
                ("UHF (HF)", sp.get("by_wf_type", {}).get("UHF", {})),
                ("UKS (B3LYP + WB97X-D)", sp.get("by_wf_type", {}).get("UKS", {})),
                ("Method: HF", sp.get("by_method", {}).get("HF", {})),
                ("Method: B3LYP", sp.get("by_method", {}).get("B3LYP", {})),
                ("Method: WB97X-D", sp.get("by_method", {}).get("WB97X-D", {})),
            ]:
                dst = sdict.get("delta_s2_stats")
                if dst:
                    g005 = sdict["abs_delta_s2_gt_0.05"]
                    g01 = sdict["abs_delta_s2_gt_0.1"]
                    g05 = sdict["abs_delta_s2_gt_0.5"]
                    fh.write(
                        f"  {lbl_s:<24}: n={dst['count']:>7,}, med={dst['p50']:+.5f}, "
                        f"p95={dst['p95']:+.5f}, max={dst['max']:+.5f} | "
                        f"|dS2|>0.05: {g005['count']} ({g005['pct']:.2f}%), "
                        f">0.1: {g01['count']} ({g01['pct']:.2f}%), "
                        f">0.5: {g05['count']} ({g05['pct']:.2f}%)\n"
                    )
            fh.write("\n")

            fh.write("5. POOLED FORCES (TABLE 2), CROWDING (TABLE 3) & FILTER OVERLAP\n" + "-" * 96 + "\n")
            pf = fc.get("pooled_per_atom_force_magnitudes_ha_bohr", {})
            pf_st = pf.get("stats")
            if pf_st:
                le1 = pf["le_1_0_ha_bohr"]
                fh.write(
                    f"  Pooled atom forces (N={pf_st['count']:,}): "
                    f"p50={pf_st['p50']:.4f}, p75={pf_st['p75']:.4f}, p90={pf_st['p90']:.4f}, "
                    f"p95={pf_st['p95']:.4f}, p99={pf_st['p99']:.4f}, p99.9={pf_st['p99.9']:.4f}, "
                    f"max={pf_st['max']:.4f} Ha/Bohr\n"
                )
                fh.write(
                    f"  Atom force vectors <= 1.0 Ha/Bohr: {le1['count']:,} ({le1['pct']:.2f}%)\n"
                )

            fh.write("  Force outliers (F_max > threshold):\n")
            fo_c = fc.get("force_outliers_by_calculation", {})
            fo_g = fc.get("force_outliers_by_geometry", {})
            for thr_k in ("gt_10", "gt_50", "gt_100", "gt_200", "gt_300"):
                if thr_k in fo_c:
                    c_item = fo_c[thr_k]
                    g_item = fo_g[thr_k]
                    fh.write(
                        f"    {thr_k:<8}: calcs={c_item['count']:>6,} ({c_item['pct']:.3f}%), "
                        f"geoms={g_item['count']:>5,} ({g_item['pct']:.3f}%)\n"
                    )

            ct3 = fc.get("crowding_table3_factor_050", {})
            fh.write(
                f"  Crowding Table 3 (geometries with >=1 calc, N={ct3.get('evaluated_geometries_with_converged_calc', 0):,}; "
                f"0-calc excluded={ct3.get('excluded_zero_calc_geometries', 0)}):\n"
            )
            for clbl in ("none", "clean_pair", "crowded_cluster"):
                cinfo = ct3.get(clbl, {})
                fh.write(
                    f"    {clbl:<18}: {cinfo.get('count', 0):>7,} ({cinfo.get('pct', 0.0):6.2f}%) "
                    f"[median d_min={cinfo.get('median_d_min_A')}]\n"
                )

            for ov_title, ov_key in (
                ("Calculation-level 2x2 overlap", "filter_overlap_2x2_calculation_level"),
                ("Geometry-level 2x2 overlap (>=1 calc)", "filter_overlap_2x2_geometry_level"),
            ):
                ov = fc.get(ov_key, {})
                fh.write(f"  {ov_title} (total={ov.get('total', 0):,}):\n")
                fh.write(
                    f"    pass_crowding_050 total : {ov['pass_a_total']['count']:>8,} ({ov['pass_a_total']['pct']:6.2f}%)\n"
                    f"    pass_low_strain total   : {ov['pass_b_total']['count']:>8,} ({ov['pass_b_total']['pct']:6.2f}%)\n"
                    f"    both pass               : {ov['both_pass']['count']:>8,} ({ov['both_pass']['pct']:6.2f}%)\n"
                    f"    crowding_050 only       : {ov['a_only']['count']:>8,} ({ov['a_only']['pct']:6.2f}%)\n"
                    f"    low_strain only         : {ov['b_only']['count']:>8,} ({ov['b_only']['pct']:6.2f}%)\n"
                    f"    neither pass            : {ov['neither_pass']['count']:>8,} ({ov['neither_pass']['pct']:6.2f}%)\n"
                )

            if self.errors:
                fh.write(f"\nERRORS (showing up to 100 of {len(self.errors)}):\n" + "-" * 96 + "\n")
                for idx_e, err in enumerate(self.errors[:100], 1):
                    fh.write(f"  {idx_e}. {err}\n")
            if self.warnings:
                fh.write(f"\nWARNINGS (showing up to 100 of {len(self.warnings)}):\n" + "-" * 96 + "\n")
                for idx_w, warn in enumerate(self.warnings[:100], 1):
                    fh.write(f"  {idx_w}. {warn}\n")

        print(f"  ✓ Saved full verification report:    {rep_file}")
        return rep_file

    def _plot_energy_grid_3x3(self, out_dir: Path, bins=80) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        emin = float(np.min(self.all_energies))
        emax = float(np.max(self.all_energies))
        if emax <= emin:
            emin -= 1.0
            emax += 1.0
        pad = 0.01 * (emax - emin)
        bin_edges = np.linspace(emin - pad, emax + pad, int(bins) + 1)

        fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=True)
        for r_idx, method in enumerate(METHODS):
            for c_idx, basis in enumerate(BASIS_SETS):
                ax = axes[r_idx][c_idx]
                ckey = f"{method}_{basis}"
                vals = np.asarray(self.energies_by_combo.get(ckey, []), dtype=np.float64)
                disp_method = "ωB97X-D" if method == "WB97X-D" else method
                ax.set_title(f"{disp_method} / {basis}", fontsize=11, fontweight="bold")

                if vals.size == 0:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                    continue

                ax.hist(
                    vals,
                    bins=bin_edges,
                    color="#3b6fb6",
                    edgecolor="#1d3b66",
                    linewidth=0.35,
                    alpha=0.85,
                )
                ax.set_yscale("log")
                ax.grid(True, which="major", alpha=0.3, linestyle="--")

                med_val = float(np.median(vals))
                ax.axvline(med_val, color="#d95f02", linestyle="--", linewidth=1.2)
                if emax >= 0.0:
                    ax.axvline(0.0, color="#b2182b", linestyle=":", linewidth=1.0)

                n_pos = int(np.sum(vals > 0.0))
                info_txt = (
                    f"n = {vals.size:,}\n"
                    f"min = {np.min(vals):.1f} Ha\n"
                    f"med = {med_val:.1f} Ha\n"
                    f"max = {np.max(vals):.2f} Ha"
                )
                if n_pos > 0:
                    info_txt += f"\nE>0: {n_pos}"
                ax.text(
                    0.03,
                    0.96,
                    info_txt,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8.5,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.85,
                        edgecolor="#cccccc",
                    ),
                )

                if c_idx == 0:
                    ax.set_ylabel("Calculation Count (log scale)", fontsize=10)
                if r_idx == 2:
                    ax.set_xlabel("Total SCF Energy (Ha)", fontsize=10)

        fig.suptitle(
            f"Total SCF Energy Distributions by Method and Basis — {self.dataset_name} "
            f"(N = {len(self.all_energies):,}, {int(bins)} equal-width bins)",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.965])
        fig_file = out_dir / f"energy_hist_3x3_{self.dataset_name}.png"
        fig.savefig(fig_file, dpi=300)
        plt.close(fig)
        print(f"  ✓ Saved 3x3 energy histogram (300 DPI): {fig_file}")
        return fig_file

    def generate_report(self) -> bool:
        """Print concise console summary of all verification checks."""
        s = self.summary
        cnts = s.get("counts", {})
        ev = s.get("energy_validation", {})
        nm = s.get("npz_and_matrix_validation", {})
        hl = s.get("homo_lumo_diagnostics", {})
        sp = s.get("spin_contamination", {})
        fc = s.get("forces_crowding_and_filters", {})

        print(f"\n{'=' * 72}")
        print(f"VERIFICATION SUMMARY: {self.dataset_name}")
        print(f"{'=' * 72}")
        print(
            f"  Geometries: {cnts.get('total_molecules', 0):,} "
            f"(>=1 calc: {cnts.get('molecules_with_converged_calcs', 0):,}, "
            f"0 calcs: {cnts.get('molecules_with_zero_calcs', 0):,}) | "
            f"Calculations: {cnts.get('total_calculations', 0):,}"
        )
        est = ev.get("overall_scf_energy_ha")
        if est:
            print(
                f"  SCF Energy (Ha): min={est['min']:.2f}, p50={est['p50']:.2f}, "
                f"max={est['max']:.2f} (E > 0: {ev.get('positive_scf_energy_count', 0)})"
            )
        mo_st = nm.get("max_mo_ortho_err_combined", {}).get("stats")
        ps_st = nm.get("tr_ps_minus_nelec_abs_err", {}).get("stats")
        if mo_st and ps_st:
            gt1e4 = (
                nm["max_mo_ortho_err_combined"]["thresholds"].get("gt_0.0001", {}).get("count", 0)
            )
            print(
                f"  Matrices: max |Tr(PS)-N_e|={ps_st['max']:.3e}, "
                f"max |CSC^T-I|={mo_st['max']:.3e} (>1e-4: {gt1e4}), "
                f"N_indep<N_bas: {nm.get('n_indep_lt_n_bas_count', 0)}"
            )
        an_h = hl.get("anions_Q_lt_0", {})
        print(
            f"  Anion HOMO > 0 (Q < 0): {an_h.get('homo_eff_gt_0_count', 0):,}/"
            f"{an_h.get('count', 0):,} ({an_h.get('homo_eff_gt_0_pct', 0.0):.2f}%)"
        )
        sp_o = sp.get("overall_open_shell", {})
        if sp_o.get("parsed_s2_count", 0) > 0:
            g01 = sp_o["abs_delta_s2_gt_0.1"]
            print(
                f"  Open-shell <S^2>: n={sp_o['parsed_s2_count']:,}, "
                f"|dS^2|>0.1: {g01['count']:,} ({g01['pct']:.2f}%)"
            )
        ct3 = fc.get("crowding_table3_factor_050", {})
        if ct3:
            print(
                f"  Crowding (r/r_cov<0.50): none={ct3.get('none', {}).get('count', 0):,}, "
                f"clean_pair={ct3.get('clean_pair', {}).get('count', 0):,}, "
                f"crowded_cluster={ct3.get('crowded_cluster', {}).get('count', 0):,}"
            )

        print(f"\n  Total errors  : {len(self.errors)}")
        print(f"  Total warnings: {len(self.warnings)}")

        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for i, err in enumerate(self.errors[:30], 1):
                print(f"  {i}. {err}")
            if len(self.errors) > 30:
                print(f"  ... and {len(self.errors) - 30} more")
        else:
            print("\n✅ No errors found!")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warn in enumerate(self.warnings[:20], 1):
                print(f"  {i}. {warn}")
            if len(self.warnings) > 20:
                print(f"  ... and {len(self.warnings) - 20} more")

        return len(self.errors) == 0


def plot_combined_energy_grid(verifiers, output_dir, bins=80):
    """Overlay multiple datasets on a single 3x3 method-basis log-y histogram (300 DPI)."""
    active = [v for v in verifiers if v.all_energies]
    if len(active) < 2:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emin = min(float(np.min(v.all_energies)) for v in active)
    emax = max(float(np.max(v.all_energies)) for v in active)
    pad = 0.01 * max(emax - emin, 1.0)
    bin_edges = np.linspace(emin - pad, emax + pad, int(bins) + 1)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=True)
    for r_idx, method in enumerate(METHODS):
        for c_idx, basis in enumerate(BASIS_SETS):
            ax = axes[r_idx][c_idx]
            ckey = f"{method}_{basis}"
            disp_method = "ωB97X-D" if method == "WB97X-D" else method
            ax.set_title(f"{disp_method} / {basis}", fontsize=11, fontweight="bold")

            for idx_v, ver in enumerate(active):
                vals = np.asarray(ver.energies_by_combo.get(ckey, []), dtype=np.float64)
                if vals.size == 0:
                    continue
                col = palette[idx_v % len(palette)]
                disp_ds = (
                    "2-node" if "2node" in ver.dataset_name
                    else ("3-node" if "3node" in ver.dataset_name else ver.dataset_name)
                )
                ax.hist(
                    vals,
                    bins=bin_edges,
                    histtype="stepfilled",
                    alpha=0.45,
                    edgecolor=col,
                    facecolor=col,
                    linewidth=1.1,
                    label=f"{disp_ds} (n = {vals.size:,})",
                )
            ax.set_yscale("log")
            ax.grid(True, which="major", alpha=0.3, linestyle="--")
            if emax >= 0.0:
                ax.axvline(0.0, color="#b2182b", linestyle=":", linewidth=1.0)
            ax.legend(fontsize=8, loc="upper left")
            if c_idx == 0:
                ax.set_ylabel("Calculation Count (log scale)", fontsize=10)
            if r_idx == 2:
                ax.set_xlabel("Total SCF Energy (Ha)", fontsize=10)

    ds_labels = " and ".join(
        ("2-node" if "2node" in v.dataset_name else ("3-node" if "3node" in v.dataset_name else v.dataset_name))
        for v in active
    )
    fig.suptitle(
        f"Total SCF Energy Distributions by Method and Basis ({ds_labels}, {int(bins)} equal-width bins)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig_file = out_dir / "energy_hist_3x3_combined.png"
    fig.savefig(fig_file, dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved combined 3x3 energy histogram (300 DPI): {fig_file}")
    return fig_file


def verify_cross_dataset(datasets_or_verifiers) -> bool:
    """
    Check for overlapping geometry hashes between datasets.
    Accepts either DatasetVerifier instances (zero extra disk I/O) or dataset paths.
    """
    print(f"\n{'=' * 72}")
    print("CROSS-DATASET GEOMETRY HASH CHECK")
    print(f"{'=' * 72}")

    dataset_hashes = {}
    for item in datasets_or_verifiers:
        if isinstance(item, DatasetVerifier):
            dataset_hashes[item.dataset_name] = item.geometry_hashes
            print(f"  {item.dataset_name}: {len(item.geometry_hashes):,} unique geometry hashes")
        else:
            dpath = Path(item)
            if not dpath.exists():
                print(f"  Warning: {dpath} does not exist, skipping")
                continue
            hashes = set()
            for mol_dir in tqdm(sorted(dpath.glob("mol_*")), desc=f"Reading {dpath.name}"):
                mf = mol_dir / "metadata.json"
                if mf.exists():
                    with open(mf, "r", encoding="utf-8") as fh:
                        meta = json.load(fh)
                    gh = meta.get("geometry_hash")
                    if gh:
                        hashes.add(gh)
            dataset_hashes[dpath.name] = hashes
            print(f"  {dpath.name}: {len(hashes):,} unique geometry hashes")

    ok = True
    names = list(dataset_hashes.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ds1, ds2 = names[i], names[j]
            overlap = dataset_hashes[ds1] & dataset_hashes[ds2]
            if overlap:
                ok = False
                print(f"\n❌ OVERLAP between {ds1} and {ds2}: {len(overlap)} geometries")
                print(f"   Examples: {sorted(overlap)[:5]}")
            else:
                print(f"\n✅ No geometry_hash overlap between {ds1} and {ds2}")
    return ok


def main():
    default_workers = min(8, os.cpu_count() or 4)
    parser = argparse.ArgumentParser(
        description="Single-pass parallel verification of QMEnum-2-3 datasets"
    )
    parser.add_argument("datasets", nargs="+", help="Dataset directories to verify")
    parser.add_argument(
        "--plot-dir",
        "--output-dir",
        dest="output_dir",
        default="verification_plots",
        help="Output directory for CSV.gz tables, JSON/TXT reports, and PNG plots (default: verification_plots)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Number of parallel worker processes (default: {default_workers})",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Number of equal-width bins for 3x3 energy histograms (default: 80)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating PNG histogram figures (still writes CSV.gz, JSON, and TXT reports)",
    )
    parser.add_argument(
        "--skip-cross-check",
        action="store_true",
        help="Skip cross-dataset geometry_hash overlap check",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, only process the first N sorted mol_* directories per dataset (for dry-runs)",
    )

    args = parser.parse_args()

    all_passed = True
    verifiers = []

    for dataset_path in args.datasets:
        verifier = DatasetVerifier(dataset_path)
        verifier.run_single_pass(workers=args.workers, limit=args.limit)
        verifier.save_all_outputs(
            output_dir=args.output_dir,
            bins=args.bins,
            make_plot=(not args.no_plots),
        )
        passed = verifier.generate_report()
        all_passed = all_passed and passed
        verifiers.append(verifier)

    if not args.no_plots and len(verifiers) > 1:
        plot_combined_energy_grid(verifiers, output_dir=args.output_dir, bins=args.bins)

    if not args.skip_cross_check and len(verifiers) > 1:
        cross_ok = verify_cross_dataset(verifiers)
        all_passed = all_passed and cross_ok

    if all_passed:
        print("\n✅ All verifications passed!")
        sys.exit(0)
    else:
        print("\n❌ Some verifications failed - review errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
