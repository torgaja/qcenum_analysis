# QMEnum-2-3 Validation and Analysis Scripts

Companion validation, geometry-crowding, force-decomposition, and figure-generation scripts for the **QMEnum-2-3** quantum chemistry dataset descriptor:

> **Quantum chemistry calculations from systematic enumeration of small molecule structures**  
> Gábor János Tornai, Álex Kálmán Balogh, Ádám Rák, and György Cserey

---

## Dataset Access

The complete **QMEnum-2-3** dataset (112,958 unique non-equilibrium molecular geometries and 338,616 converged Q-Chem 6.1 single-point energy/gradient calculations across all 18 elements H–Ar, charges $-2 \ldots +2$, and spin multiplicities up to 7) is archived on Zenodo under CC BY 4.0:

- **Zenodo DOI:** [https://doi.org/10.5281/zenodo.20055649](https://doi.org/10.5281/zenodo.20055649)
- Archives: `dataset_2node.tar.gz` (2,960 geometries; 8,973 calculations) and `dataset_3node.tar.gz` (109,998 geometries; 329,643 calculations).

Extract the archives into a working data directory `<DATA_ROOT>` so that `<DATA_ROOT>/dataset_2node/` and `<DATA_ROOT>/dataset_3node/` exist.

---

## Repository Structure

```text
├── LICENSE                  # MIT License (applies to all Python scripts)
├── README.md                # Repository documentation and usage guide
├── requirements.txt         # Python dependencies
├── scripts/                 # Technical validation and analysis scripts
│   ├── verify_datasets.py
│   ├── analyze_dataset.py
│   ├── crowding_detector.py
│   ├── sign_alignment_check.py
│   ├── nn_force_distance_v2.py
│   ├── ratio_vs_force_knee.py
│   ├── check_alternatives.py
│   ├── check_strained_no_outlier.py
│   ├── dump_mol_geometry_crowding.py
│   ├── uff_force_validation.py
│   ├── scan_typo_states.py
│   └── orthonormalization_check.py
└── manuscript/              # Data descriptor LaTeX source and figures (CC BY 4.0)
    ├── LICENSE
    ├── dataset_descriptor.tex
    ├── refek.bib
    └── *.png
```

---

## Installation

Python 3.10+ is recommended:

```bash
pip install -r requirements.txt
```

`scripts/uff_force_validation.py` additionally requires [RDKit](https://www.rdkit.org) (e.g., `conda install -c conda-forge rdkit` or `pip install rdkit`).

---

## Usage

### 1. Full Single-Pass Archive & Matrix Verification (`verify_datasets.py`)

Verifies file completeness, metadata consistency, Latin-square method/basis assignment, Q-Chem `.qcout` convergence/energies/$\langle S^2 \rangle$, `.npz` schema/dtypes/finiteness, core-Hamiltonian additivity, translational force invariance, electron-count normalization $\mathrm{Tr}(P S) = N_e$, MO orthonormality $C S C^{\top} = I$, HOMO/LUMO statistics, Cordero covalent-radius crowding regimes, low-strain filter retention, and generates the 3×3 SCF energy distribution figure (`energy_distributions_3x3.png`):

```bash
python3 scripts/verify_datasets.py     /path/to/dataset_2node /path/to/dataset_3node     --output-dir verification_plots --workers 8 --bins 80
```

### 2. Bond-Length, Chemical-Space, and Force Distributions (`analyze_dataset.py`)

Generates the bond-length histograms, periodic half-bond box plots, chemical-space coverage overview, 3×3 element-resolved force distributions, and force/geometry outlier tables:

```bash
python3 scripts/analyze_dataset.py     --dataset-root /path/to/datasets     --output-dir analysis_output
```

### 3. Geometric Crowding & Short-Range Repulsive Regime

- **Covalent-radius crowding classification (`crowding_detector.py`):** Classifies each unique geometry into `none`, `clean_pair`, or `crowded_cluster` using $r_{ij} / (r_{\mathrm{cov},i} + r_{\mathrm{cov},j}) < 0.50$:
  ```bash
  python3 scripts/crowding_detector.py --dataset-root /path/to/datasets --dataset dataset_3node
  ```
- **Radial force sign & alignment (`sign_alignment_check.py`):** Evaluates two-body radial force components $f_{\mathrm{rel}}$ and directional alignment at short interatomic contacts ($< 0.8$ Å and $< 0.5$ Å) in Q-Chem's standard orientation:
  ```bash
  python3 scripts/sign_alignment_check.py --dataset-root /path/to/datasets --dataset dataset_3node
  ```
- **Nearest-neighbor force–distance export (`nn_force_distance_v2.py`) & compression-ratio analysis (`ratio_vs_force_knee.py`):**
  ```bash
  python3 scripts/nn_force_distance_v2.py       --dataset-root /path/to/datasets --dataset dataset_3node       --out analysis_output/3node_new/nn_force_distance_v2.csv

  python3 scripts/ratio_vs_force_knee.py       --csv analysis_output/3node_new/nn_force_distance_v2.csv       --out-png analysis_output/3node_new/ratio_vs_force_knee.png
  ```

### 4. Targeted Inspection Utilities

- **`check_alternatives.py`** — Verifies that alternative Lewis/graph specifications sharing a 3D geometry match the $(Q, 2S+1)$ state in Q-Chem `.qcout` inputs:
  ```bash
  python3 scripts/check_alternatives.py --dataset-root /path/to/datasets --dataset dataset_3node
  ```
- **`check_strained_no_outlier.py`** — Inspects compressed geometries whose maximum atomic force remains $\le 10$ Ha/Bohr.
- **`dump_mol_geometry_crowding.py`** — Prints full atomic coordinates, critical-pair ratios, and connected-component topology (`knot` vs. `chain`) for an individual molecule (`--mol-id mol_002876`).
- **`uff_force_validation.py`** — Evaluates RDKit UFF energies/forces on a stored geometry and compares them with QM forces (`python3 scripts/uff_force_validation.py mol_000001 --dataset-root /path/to/datasets`).
- **`scan_typo_states.py`** — Read-only scan of `metadata.json` files for specific valence-state entries (`python3 scripts/scan_typo_states.py --base /path/to/datasets`).
- **`orthonormalization_check.py`** — Inspects raw `step_0_orthonormalization_matrix` and `step_0_overlap_matrix` vectors in a molecule `.npz` archive.

---

## Building the Manuscript

To compile the LaTeX manuscript in `manuscript/`:

```bash
cd manuscript
pdflatex dataset_descriptor.tex
bibtex dataset_descriptor
pdflatex dataset_descriptor.tex
pdflatex dataset_descriptor.tex
```

---

## License

- **Code (`scripts/` and root configuration):** [MIT License](LICENSE).
- **Manuscript text and figures (`manuscript/`):** [Creative Commons Attribution 4.0 International (CC BY 4.0)](manuscript/LICENSE).
