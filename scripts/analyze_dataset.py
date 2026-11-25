#!/usr/bin/env python3
"""
Dataset analysis script for QM calculation datasets.

Tasks:
1. Box plots of half bond lengths for each atom type vs all other atom types
2. Histograms of bond lengths for selected atom pairs
3. Distribution of atoms across the dataset
4. Distribution of atomic forces by element
5. Relative meeting abundance of element pairs
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')
METHODS    = ['HF', 'B3LYP', 'WB97X-D']
BASIS_SETS = ['STO-3G', '6-31G*', 'DEF2-SVP']
class DatasetAnalyzer:
    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)

        # Data storage (geometry-level, unchanged)
        self.half_bond_lengths = defaultdict(lambda: defaultdict(list))
        self.bond_lengths = defaultdict(list)
        self.atom_counts = Counter()
        self.molecules_with_element = defaultdict(set)
        self.molecules_with_pair = defaultdict(set)

        # Per calc_name QM data
        self.element_forces = defaultdict(lambda: defaultdict(list))  # [calc_name][elem] -> [forces]
        self.energies = defaultdict(list)              # [calc_name] -> list of dicts

        # Geometry-based outliers (method-independent)
        self.geometry_outliers = []                    # list of dicts, one per flagged geometry

        self.force_outlier_threshold = 10.0            # Ha/Bohr — above this is flagged
        self.force_outliers = []                       # list of dicts, one per flagged mol/calc

        self.all_elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar']
        self.atomic_numbers = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'S': 16, 'Cl': 17, 'Ar': 18
        }
        self.distance_short_threshold = 0.8  # Å — below this is suspicious
    
    def load_and_analyze(self):
        """Load all molecules and extract data."""
        print(f"Analyzing dataset: {self.dataset_path}")
        
        mol_dirs = sorted(self.dataset_path.glob('mol_*'))
        total = len(mol_dirs)
        
        for idx, mol_dir in enumerate(mol_dirs):
            if (idx + 1) % 100 == 0:
                print(f"  Processing {idx + 1}/{total}...")
            
            try:
                self._process_molecule(mol_dir)
            except Exception as e:
                print(f"  Warning: Failed to process {mol_dir.name}: {e}")
                continue
        
        print(f"✓ Processed {total} molecules")
    
    def _process_molecule(self, mol_dir: Path):
        """Process a single molecule directory."""
        mol_id = mol_dir.name

        with open(mol_dir / 'metadata.json') as f:
            metadata = json.load(f)

        rep_struct = metadata['representative_structure']
        coords = np.array(rep_struct['coordinates'])
        coord_elements = rep_struct['coordinate_elements']
        n_atoms = len(coord_elements)

        # Task 3: Count atoms
        for elem in coord_elements:
            self.atom_counts[elem] += 1

        # Tasks 5 & 6: Track element presence and pairs
        unique_elements = set(coord_elements)
        for elem in unique_elements:
            self.molecules_with_element[elem].add(mol_id)
        for elem1 in unique_elements:
            for elem2 in unique_elements:
                if elem1 <= elem2:
                    self.molecules_with_pair[tuple(sorted([elem1, elem2]))].add(mol_id)

        # Tasks 1 & 2: Bond lengths (geometry-level, unchanged)
        for bond_key in rep_struct.get('edge_bonds', {}):
            idx1, idx2 = map(int, bond_key.split(','))
            elem1, elem2 = coord_elements[idx1], coord_elements[idx2]
            bond_length = np.linalg.norm(coords[idx1] - coords[idx2])
            self.half_bond_lengths[elem1][elem2].append(bond_length / 2.0)
            self.half_bond_lengths[elem2][elem1].append(bond_length / 2.0)
            self.bond_lengths[tuple(sorted([elem1, elem2]))].append(bond_length)

        # Geometry outlier detection — done ONCE, method-independent
        short_pairs = []
        for i in range(n_atoms):
            for j in range(i):
                d = float(np.linalg.norm(coords[i] - coords[j]))
                if d < self.distance_short_threshold:
                    short_pairs.append({
                        'elem_i': coord_elements[j],
                        'elem_j': coord_elements[i],
                        'distance': d,
                    })
        if short_pairs:
            self.geometry_outliers.append({
                'mol_id': mol_id,
                'coordinate_elements': coord_elements,
                'coordinates': coords,
                'short_pairs': short_pairs,
                'calc_names': list(metadata['available_calculations'].keys()),
            })

        # Shared metadata for energy records (computed once before the loop)
        charge = rep_struct.get('charge', 0)
        multiplicity = rep_struct.get('multiplicity', 1)
        n_heavy = sum(1 for e in coord_elements if e != 'H')
        n_electrons = sum(self.atomic_numbers.get(e, 0) for e in coord_elements) - charge

        # Tasks 4 & 7: loop over ALL available calculations
        for calc_name, calc_info in metadata['available_calculations'].items():

            # Task 4: forces
            npz_file = mol_dir / calc_info['npz_file']
            if npz_file.exists():
                try:
                    forces = np.load(npz_file)['step_0_cartesian_forces'].reshape(n_atoms, 3)
                    force_mags = np.linalg.norm(forces, axis=1)
                    for elem, mag in zip(coord_elements, force_mags):
                        self.element_forces[calc_name][elem].append(float(mag))
                    max_force = float(np.max(force_mags))
                    if max_force > self.force_outlier_threshold:
                        self.force_outliers.append({
                            'mol_id':              mol_id,
                            'calc_name':           calc_name,
                            'coordinate_elements': coord_elements,
                            'force_mags':          force_mags.tolist(),
                            'max_force':           max_force,
                        })
                except Exception:
                    pass

            # Task 7: energy
            qcout_file = mol_dir / calc_info.get('qcout_file', '')
            if qcout_file.exists():
                try:
                    with open(qcout_file) as f:
                        for line in f:
                            if 'SCF   energy in the final basis set' in line:
                                self.energies[calc_name].append({
                                    'energy': float(line.split('=')[1].strip()),
                                    'n_heavy': n_heavy, 'n_atoms': n_atoms,
                                    'n_electrons': n_electrons,
                                    'charge': charge, 'multiplicity': multiplicity,
                                })
                                break
                except Exception:
                    pass
    
    def plot_task1_half_bond_boxplots(self, output_dir: Path):
        """Task 1: Box plots of half bond lengths for each element."""
        print("\nGenerating Task 1: Half bond length boxplots...")
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Get elements that actually have bonds
        elements_with_bonds = sorted([e for e in self.half_bond_lengths.keys() 
                                     if any(len(v) > 0 for v in self.half_bond_lengths[e].values())])
        
        for elem in elements_with_bonds:
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Collect data for all partner elements
            partner_data = []
            partner_labels = []
            
            for partner_elem in self.all_elements:
                if partner_elem in self.half_bond_lengths[elem]:
                    lengths = self.half_bond_lengths[elem][partner_elem]
                    if len(lengths) > 0:
                        partner_data.append(lengths)
                        partner_labels.append(partner_elem)
            
            if len(partner_data) > 0:
                bp = ax.boxplot(partner_data, labels=partner_labels, patch_artist=True)
                
                # Styling
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.7)
                
                ax.set_xlabel('Partner Element', fontsize=12)
                ax.set_ylabel('Half Bond Length (Å)', fontsize=12)
                ax.set_title(f'Half Bond Lengths: {elem} with Other Elements', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                output_file = output_dir / f'task1_half_bonds_{elem}.png'
                plt.savefig(output_file, dpi=150)
                plt.close()
                
                print(f"  ✓ Saved {output_file.name}")
    
    def plot_task2_selected_bond_histograms(self, output_dir: Path, 
                                           pairs: List[Tuple[str, str]]):
        """Task 2: Histograms of bond lengths for selected pairs."""
        print("\nGenerating Task 2: Bond length histograms...")
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, pair in enumerate(pairs):
            ax = axes[idx]
            sorted_pair = tuple(sorted(pair))
            
            if sorted_pair in self.bond_lengths:
                lengths = self.bond_lengths[sorted_pair]
                
                if len(lengths) > 0:
                    ax.hist(lengths, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
                    ax.set_xlabel('Bond Length (Å)', fontsize=10)
                    ax.set_ylabel('Count', fontsize=10)
                    ax.set_title(f'{pair[0]}-{pair[1]} Bond Lengths (n={len(lengths)})', 
                               fontsize=11, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f'No {pair[0]}-{pair[1]} bonds found', 
                           ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, f'No {pair[0]}-{pair[1]} bonds found', 
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        output_file = output_dir / 'task2_bond_length_histograms.png'
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        print(f"  ✓ Saved {output_file.name}")
    
    def plot_task3_atom_distribution(self, output_dir: Path):
        """Task 3: Distribution of atoms across dataset."""
        print("\nGenerating Task 3: Atom distribution...")
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Sort by periodic table order
        sorted_elements = [(e, self.atom_counts[e]) for e in self.all_elements if e in self.atom_counts]
        elements, counts = zip(*sorted_elements) if sorted_elements else ([], [])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(elements, counts, color='steelblue', alpha=0.7, edgecolor='black')
        
        # Color code: organic elements in one color
        organic = {'C', 'H', 'N', 'O', 'S', 'P'}
        for bar, elem in zip(bars, elements):
            if elem in organic:
                bar.set_color('forestgreen')
        
        ax.set_xlabel('Element', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Atom Distribution Across Dataset', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        output_file = output_dir / 'task3_atom_distribution.png'
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        print(f"  ✓ Saved {output_file.name}")
        
        # Also save as text
        txt_file = output_dir / 'task3_atom_distribution.txt'
        with open(txt_file, 'w') as f:
            f.write("Element\tCount\n")
            for elem, count in sorted_elements:
                f.write(f"{elem}\t{count}\n")
        print(f"  ✓ Saved {txt_file.name}")
    

    def plot_task3_chemical_space_overview(self, output_dir: Path):
        """Task 3: Combined chemical space overview.
        
        Left panel (a): Periodic table layout (Periods 1-3, H to Ar) showing the
        percentage of molecules containing each element, with log-scaled viridis colormap.
        
        Right panel (b): Charge x Multiplicity distribution (5 charges x 7 multiplicities),
        annotated heatmap with log-scaled viridis colormap.
        """
        print("\nGenerating Task 3: Chemical space overview (Periodic table + Charge x Multiplicity)...")
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        import matplotlib.colors as mcolors
        
        # 1. Gather metadata / molecule counts
        metadata_path = self.dataset_path / 'dataset_metadata.json'
        mol_list = []
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                mdata = json.load(f)
                mol_list = mdata.get('molecule_list', [])
        
        total_molecules = len(mol_list) if mol_list else getattr(self, 'total_molecules', 0)
        if total_molecules == 0:
            print("  Warning: No molecule_list found in metadata, attempting fallback...")
        
        # Count per-molecule element occurrences and (charge, mult)
        elem_mol_counts = Counter()
        qm_counts = Counter()
        
        for mol in mol_list:
            elems = set()
            if 'elements' in mol:
                elems.update(mol['elements'])
            elif 'coordinate_elements' in mol:
                elems.update(mol['coordinate_elements'])
            for el in elems:
                if el:
                    elem_mol_counts[el] += 1
            
            q = mol.get('charge', mol.get('total_charge', 0))
            m = mol.get('multiplicity', 1)
            qm_counts[(q, m)] += 1
        
        # Percentage dictionaries
        elem_pcts = {el: (c / total_molecules) * 100 for el, c in elem_mol_counts.items()} if total_molecules else {}
        
        # 2. Setup figure with 2 subplots
        fig, (ax_pt, ax_qm) = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={'width_ratios': [1.15, 1.0]})
        
        # --- Panel (a): Periodic Table Grid (3 rows x 8 cols) ---
        pt_layout = [
            (0, 0, 'H'),                                                (0, 7, 'He'),
            (1, 0, 'Li'), (1, 1, 'Be'), (1, 2, 'B'),  (1, 3, 'C'),  (1, 4, 'N'), (1, 5, 'O'), (1, 6, 'F'),  (1, 7, 'Ne'),
            (2, 0, 'Na'), (2, 1, 'Mg'), (2, 2, 'Al'), (2, 3, 'Si'), (2, 4, 'P'), (2, 5, 'S'), (2, 6, 'Cl'), (2, 7, 'Ar'),
        ]
        
        pt_grid_vals = np.full((3, 8), np.nan)
        pt_cell_labels = {}
        
        for r, c, sym in pt_layout:
            pct = elem_pcts.get(sym, 0.0)
            pt_grid_vals[r, c] = pct if pct > 0 else np.nan
            pt_cell_labels[(r, c)] = (sym, pct)
            
        valid_elem_pcts = [p for p in elem_pcts.values() if p > 0]
        min_elem_p = min(valid_elem_pcts) if valid_elem_pcts else 1.0
        norm_elem = mcolors.LogNorm(vmin=max(0.5, min_elem_p * 0.8), vmax=100.0)
        
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='#e0e0e0')
        
        im_pt = ax_pt.imshow(pt_grid_vals, cmap=cmap, norm=norm_elem, aspect='equal')
        
        # Annotate Periodic Table cells
        for r in range(3):
            for c in range(8):
                if (r, c) in pt_cell_labels:
                    sym, pct = pt_cell_labels[(r, c)]
                    val = pt_grid_vals[r, c]
                    normed = norm_elem(val) if not np.isnan(val) else 0
                    txt_color = 'black' if normed > 0.65 else 'white'
                    ax_pt.text(c, r, f"{sym}\n{pct:.1f}%", ha='center', va='center',
                               fontsize=11, fontweight='bold', color=txt_color)
                else:
                    ax_pt.text(c, r, '', ha='center', va='center')
                    
        col_names = ['1 (IA)', '2 (IIA)', '13 (IIIA)', '14 (IVA)', '15 (VA)', '16 (VIA)', '17 (VIIA)', '18 (VIIIA)']
        ax_pt.set_xticks(range(8))
        ax_pt.set_xticklabels(col_names, rotation=35, ha='right', fontsize=9)
        ax_pt.set_yticks(range(3))
        ax_pt.set_yticklabels(['Period 1', 'Period 2', 'Period 3'], fontsize=10)
        ax_pt.set_title('(a) Element occurrence (% of molecules)', fontsize=13, fontweight='bold', pad=12)
        
        cbar_pt = fig.colorbar(im_pt, ax=ax_pt, fraction=0.046, pad=0.04)
        cbar_pt.set_label('% of molecules containing element', fontsize=10)
        
        # --- Panel (b): Charge x Multiplicity Heatmap (5 charges x 7 multiplicities) ---
        charges = [2, 1, 0, -1, -2]
        mults = [1, 2, 3, 4, 5, 6, 7]
        
        qm_grid = np.zeros((len(charges), len(mults)))
        
        for i, q in enumerate(charges):
            for j, m in enumerate(mults):
                cnt = qm_counts.get((q, m), 0)
                qm_grid[i, j] = (cnt / total_molecules) * 100 if total_molecules else 0.0
                
        qm_grid_masked = np.where(qm_grid > 0, qm_grid, np.nan)
        norm_qm = mcolors.LogNorm(vmin=0.01, vmax=20.0)
        
        im_qm = ax_qm.imshow(qm_grid_masked, cmap=cmap, norm=norm_qm, aspect='auto')
        
        # Annotate QM cells
        for i, q in enumerate(charges):
            for j, m in enumerate(mults):
                val = qm_grid[i, j]
                if val == 0:
                    ax_qm.text(j, i, '–', ha='center', va='center', color='#888888', fontsize=11)
                elif val < 0.1:
                    normed = norm_qm(val)
                    txt_color = 'black' if normed > 0.65 else 'white'
                    ax_qm.text(j, i, '<0.1', ha='center', va='center', color=txt_color, fontsize=9)
                else:
                    normed = norm_qm(val)
                    txt_color = 'black' if normed > 0.65 else 'white'
                    ax_qm.text(j, i, f"{val:.1f}", ha='center', va='center', color=txt_color,
                               fontsize=10, fontweight='bold')
                               
        ax_qm.set_xticks(range(len(mults)))
        ax_qm.set_xticklabels([f"m={m}" for m in mults], fontsize=10)
        ax_qm.set_yticks(range(len(charges)))
        ax_qm.set_yticklabels([f"q={q:+d}" if q != 0 else "q=0" for q in charges], fontsize=10)
        ax_qm.set_xlabel('Spin Multiplicity ($m$)', fontsize=11)
        ax_qm.set_ylabel('Total Charge ($q$)', fontsize=11)
        ax_qm.set_title(f'(b) Charge and Multiplicity (% of molecules, $N={total_molecules:,}$)', fontsize=13, fontweight='bold', pad=12)
        
        cbar_qm = fig.colorbar(im_qm, ax=ax_qm, fraction=0.046, pad=0.04)
        cbar_qm.set_label('% of molecules in state', fontsize=10)
        
        plt.tight_layout()
        output_file = output_dir / 'task3_chemical_space_overview.png'
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"  ✓ Saved {output_file.name}")

    def plot_task4_force_distribution(self, output_dir: Path):
        """Task 4: Force distributions per method-basis combination (3x3 grid) and percentiles."""
        print("\nGenerating Task 4: Force distributions (per method-basis)...")
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        if not self.element_forces:
            print("  Warning: No force data found!")
            return

        # --- Statistics text file ---
        txt_file = output_dir / 'task4_force_statistics.txt'
        percentiles = [50, 75, 90, 95, 99, 99.9]

        # Overall aggregate forces across all calculations
        all_combined_forces = []
        for calc_name, elem_dict in self.element_forces.items():
            for forces in elem_dict.values():
                all_combined_forces.extend(forces)
        all_combined_forces = np.array(all_combined_forces)

        with open(txt_file, 'w') as f:
            f.write("Force Magnitude Statistics per Method-Basis Combination\n")
            f.write("=" * 80 + "\n\n")

            if len(all_combined_forces) > 0:
                f.write(f"OVERALL AGGREGATE (n_forces={len(all_combined_forces)}):\n")
                f.write(f"  Mean:   {np.mean(all_combined_forces):.6f} Ha/Bohr\n")
                f.write(f"  Std:    {np.std(all_combined_forces):.6f} Ha/Bohr\n")
                f.write(f"  Min:    {np.min(all_combined_forces):.6f} Ha/Bohr\n")
                f.write(f"  Max:    {np.max(all_combined_forces):.6f} Ha/Bohr\n")
                for p in percentiles:
                    f.write(f"  p{p:<5}: {np.percentile(all_combined_forces, p):.6f} Ha/Bohr\n")
                f.write("\n" + "=" * 80 + "\n\n")

            for calc_name in sorted(self.element_forces.keys()):
                all_forces = []
                for elem_forces in self.element_forces[calc_name].values():
                    all_forces.extend(elem_forces)
                all_forces = np.array(all_forces)

                f.write(f"{calc_name} (n_atoms={len(all_forces)}):\n")
                f.write(f"  Mean:   {np.mean(all_forces):.6f} Ha/Bohr\n")
                f.write(f"  Std:    {np.std(all_forces):.6f} Ha/Bohr\n")
                f.write(f"  Min:    {np.min(all_forces):.6f} Ha/Bohr\n")
                f.write(f"  Max:    {np.max(all_forces):.6f} Ha/Bohr\n")
                for p in percentiles:
                    f.write(f"  p{p:<5}: {np.percentile(all_forces, p):.6f} Ha/Bohr\n")
                f.write("\n")

                header = f"  {'Element':<6} {'Count':>8} {'Mean':>10} {'Median':>10} {'p90':>10} {'p95':>10} {'p99':>10} {'Max':>10}"
                f.write(header + "\n")
                for elem in self.all_elements:
                    if elem in self.element_forces[calc_name]:
                        forces = np.array(self.element_forces[calc_name][elem])
                        p50 = np.percentile(forces, 50)
                        p90 = np.percentile(forces, 90)
                        p95 = np.percentile(forces, 95)
                        p99 = np.percentile(forces, 99)
                        f.write(f"  {elem:<6} {len(forces):>8} {np.mean(forces):>10.4f} "
                               f"{p50:>10.4f} {p90:>10.4f} {p95:>10.4f} {p99:>10.4f} {np.max(forces):>10.4f}\n")
                f.write("\n" + "-" * 80 + "\n\n")

        print(f"  ✓ Saved {txt_file.name}")

        # --- 3x3 grid figure ---
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))

        for row_idx, method in enumerate(METHODS):
            for col_idx, basis in enumerate(BASIS_SETS):
                ax = axes[row_idx][col_idx]
                calc_name = f"{method}_{basis}"

                if calc_name not in self.element_forces:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(calc_name, fontsize=10)
                    continue

                elements_with_forces = [e for e in self.all_elements
                                        if e in self.element_forces[calc_name]]
                force_data = [self.element_forces[calc_name][e] for e in elements_with_forces]

                bp = ax.boxplot(force_data, labels=elements_with_forces, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightcoral')
                    patch.set_alpha(0.7)

                ax.set_title(calc_name, fontsize=10, fontweight='bold')
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3, axis='y')
                ax.tick_params(axis='x', rotation=45, labelsize=7)

                if col_idx == 0:
                    ax.set_ylabel('Force (Ha/Bohr)', fontsize=9)
                if row_idx == 2:
                    ax.set_xlabel('Element', fontsize=9)

        fig.suptitle('Atomic Force Distributions by Method-Basis Combination', fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_file = output_dir / 'task4_force_distribution.png'
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"  ✓ Saved {output_file.name}")

    def calculate_task5_pair_abundance(self, output_dir: Path):
        """Task 5: Relative meeting abundance of element pairs."""
        print("\nGenerating Task 5: Element pair abundance...")
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        results = []
        
        for elem_a in self.all_elements:
            if elem_a not in self.molecules_with_element:
                continue
            
            n_with_a = len(self.molecules_with_element[elem_a])
            
            for elem_b in self.all_elements:
                if elem_b not in self.molecules_with_element:
                    continue
                
                pair = tuple(sorted([elem_a, elem_b]))
                
                if pair in self.molecules_with_pair:
                    n_with_both = len(self.molecules_with_pair[pair])
                    abundance = n_with_both / n_with_a if n_with_a > 0 else 0
                    
                    results.append({
                        'elem_a': elem_a,
                        'elem_b': elem_b,
                        'n_with_a': n_with_a,
                        'n_with_both': n_with_both,
                        'abundance': abundance
                    })
        
        # Save to file
        txt_file = output_dir / 'task5_pair_abundance.txt'
        with open(txt_file, 'w') as f:
            f.write("Element_A\tElement_B\tN_with_A\tN_with_Both\tAbundance\n")
            for r in sorted(results, key=lambda x: x['abundance'], reverse=True):
                f.write(f"{r['elem_a']}\t{r['elem_b']}\t{r['n_with_a']}\t"
                       f"{r['n_with_both']}\t{r['abundance']:.6f}\n")
        
        print(f"  ✓ Saved {txt_file.name}")
        
        # Create a heatmap for visualization
        # Get elements in periodic table order
        elements_present = [e for e in self.all_elements if e in set([r['elem_a'] for r in results])]
        n = len(elements_present)
        
        if n > 0:
            abundance_matrix = np.zeros((n, n))
            
            for r in results:
                i = elements_present.index(r['elem_a'])
                j = elements_present.index(r['elem_b'])
                abundance_matrix[i, j] = r['abundance']
            
            fig, ax = plt.subplots(figsize=(12, 10))
            # Apply log10 transformation for better visualization
            log_abundance = np.log10(abundance_matrix + 0.01)
            im = ax.imshow(log_abundance, cmap='YlOrRd', aspect='auto')
            
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(elements_present)
            ax.set_yticklabels(elements_present)
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            ax.set_xlabel('Element B', fontsize=12)
            ax.set_ylabel('Element A', fontsize=12)
            ax.set_title('Pair Abundance: P(B|A) = N(A∩B) / N(A) [log scale]', fontsize=14, fontweight='bold')
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('log₁₀(Abundance + 0.01)', fontsize=12)
            
            plt.tight_layout()
            output_file = output_dir / 'task5_pair_abundance_heatmap.png'
            plt.savefig(output_file, dpi=150)
            plt.close()
            
            print(f"  ✓ Saved {output_file.name}")
    
    def calculate_task6_pair_cooccurrence(self, output_dir: Path):
        """Task 6: Raw co-occurrence counts of element pairs."""
        print("\nGenerating Task 6: Element pair co-occurrence...")
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Get elements in periodic table order that appear in molecules
        elements_present = [e for e in self.all_elements if e in self.molecules_with_element]
        n = len(elements_present)
        
        if n == 0:
            print("  Warning: No elements found!")
            return
        
        # Create symmetric co-occurrence matrix
        cooccurrence_matrix = np.zeros((n, n), dtype=int)
        
        for i, elem_a in enumerate(elements_present):
            for j, elem_b in enumerate(elements_present):
                pair = tuple(sorted([elem_a, elem_b]))
                if pair in self.molecules_with_pair:
                    count = len(self.molecules_with_pair[pair])
                    cooccurrence_matrix[i, j] = count
        
        # Save to file
        txt_file = output_dir / 'task6_pair_cooccurrence.txt'
        with open(txt_file, 'w') as f:
            f.write("Element_A\tElement_B\tN_with_Both\n")
            for i, elem_a in enumerate(elements_present):
                for j, elem_b in enumerate(elements_present):
                    count = cooccurrence_matrix[i, j]
                    if count > 0:
                        f.write(f"{elem_a}\t{elem_b}\t{count}\n")
        
        print(f"  ✓ Saved {txt_file.name}")
        
        # Create heatmap with logarithmic scale
        fig, ax = plt.subplots(figsize=(12, 10))
        # Apply log10 transformation for better visualization
        log_matrix = np.log10(cooccurrence_matrix + 1)
        im = ax.imshow(log_matrix, cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(elements_present)
        ax.set_yticklabels(elements_present)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        ax.set_xlabel('Element B', fontsize=12)
        ax.set_ylabel('Element A', fontsize=12)
        ax.set_title('Pair Co-occurrence: N(A∩B) [log scale]', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('log₁₀(Count + 1)', fontsize=12)
        
        plt.tight_layout()
        output_file = output_dir / 'task6_pair_cooccurrence_heatmap.png'
        plt.savefig(output_file, dpi=150)
        plt.close()
        
    
    def plot_task7_energy_distributions(self, output_dir: Path):
        """Task 7: Energy distributions per method-basis combination (3x3 grid)."""
        print("\nGenerating Task 7: Energy distributions (per method-basis)...")
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        if not self.energies:
            print("  Warning: No energy data found!")
            return

        charge_colors = {-2: 'darkblue', -1: 'royalblue', 0: 'forestgreen', 1: 'orange', 2: 'red'}
        mult_labels   = {1: 'Singlet', 2: 'Doublet', 3: 'Triplet', 4: 'Quartet', 5: 'Quintet',
                         6: 'Sextet', 7: 'Septet'}

        # --- 3x3 grid: energy vs atom count, colored by charge ---
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))

        for row_idx, method in enumerate(METHODS):
            for col_idx, basis in enumerate(BASIS_SETS):
                ax = axes[row_idx][col_idx]
                calc_name = f"{method}_{basis}"

                if calc_name not in self.energies or not self.energies[calc_name]:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(calc_name, fontsize=10)
                    continue

                data = self.energies[calc_name]
                for charge in sorted(set(d['charge'] for d in data)):
                    subset = [d for d in data if d['charge'] == charge]
                    ax.scatter([d['n_atoms'] for d in subset],
                               [d['energy'] for d in subset],
                               c=charge_colors.get(charge, 'gray'),
                               label=f"q={charge:+d} (n={len(subset)})",
                               alpha=0.3, s=4, edgecolors='none')

                ax.set_title(calc_name, fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=6, loc='best', markerscale=3)

                if col_idx == 0:
                    ax.set_ylabel('Energy (Ha)', fontsize=9)
                if row_idx == 2:
                    ax.set_xlabel('Atom Count', fontsize=9)

        fig.suptitle('Total Energy Distributions by Method-Basis Combination', fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_file = output_dir / 'task7_energy_distributions.png'
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"  ✓ Saved {output_file.name}")

        # --- Statistics text file ---
        txt_file = output_dir / 'task7_energy_statistics.txt'
        with open(txt_file, 'w') as f:
            f.write("Energy Statistics per Method-Basis Combination\n")
            f.write("=" * 80 + "\n\n")

            for calc_name in sorted(self.energies.keys()):
                data = self.energies[calc_name]
                if not data:
                    continue
                energies_arr = np.array([d['energy'] for d in data])

                f.write(f"{calc_name} (n={len(data)}):\n")
                f.write(f"  Mean:   {np.mean(energies_arr):12.4f} Ha\n")
                f.write(f"  Std:    {np.std(energies_arr):12.4f} Ha\n")
                f.write(f"  Min:    {np.min(energies_arr):12.4f} Ha\n")
                f.write(f"  Max:    {np.max(energies_arr):12.4f} Ha\n\n")

                f.write("  By charge:\n")
                for charge in sorted(set(d['charge'] for d in data)):
                    subset = [d for d in data if d['charge'] == charge]
                    e = np.array([d['energy'] for d in subset])
                    e_per_elec = np.array([d['energy'] / d['n_electrons']
                                           for d in subset if d['n_electrons'] > 0])
                    f.write(f"    q={charge:+d} (n={len(subset):6d}): "
                            f"E = {np.mean(e):10.4f} ± {np.std(e):8.4f} Ha")
                    if len(e_per_elec):
                        f.write(f",  E/e⁻ = {np.mean(e_per_elec):8.4f} ± {np.std(e_per_elec):6.4f} Ha/e⁻")
                    f.write("\n")

                f.write("\n  By multiplicity:\n")
                for mult in sorted(set(d['multiplicity'] for d in data)):
                    subset = [d for d in data if d['multiplicity'] == mult]
                    e = np.array([d['energy'] for d in subset])
                    label = mult_labels.get(mult, f"M={mult}")
                    f.write(f"    {label} (n={len(subset):6d}): "
                            f"E = {np.mean(e):10.4f} ± {np.std(e):8.4f} Ha\n")

                f.write("\n" + "-" * 80 + "\n\n")

        print(f"  ✓ Saved {txt_file.name}")

    def inspect_geometry_outliers(self, output_dir: Path, verbose: bool = False):
        """Summarize geometry outliers (short pairwise distances), write CSV."""
        from collections import Counter
        import csv

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        outliers = self.geometry_outliers
        n_total_geom = sum(self.atom_counts.values())  # rough proxy; actual count tracked below
        print(f"\nGeometry outliers (any pairwise distance < {self.distance_short_threshold} Å): "
              f"{len(outliers)} geometries")

        # Total affected calculations
        n_affected_calcs = sum(len(m['calc_names']) for m in outliers)
        print(f"  Affected calculations: {n_affected_calcs} "
              f"(avg {n_affected_calcs / len(outliers):.1f} per geometry)" if outliers else "")

        # --- Severity distribution by distance ---
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        print("\n  Severity distribution (min pairwise distance per geometry):")
        for t in thresholds:
            count = sum(1 for m in outliers
                        if any(p['distance'] < t for p in m['short_pairs']))
            print(f"    < {t:.1f} Å: {count:5d} geometries")

        # --- Per-geometry summary + accumulation ---
        csv_rows = []
        pair_counter = Counter()
        category_counter = Counter()

        if verbose:
            print(f"\n  Per-geometry summary:")
        for mol in outliers:
            mol_id      = mol['mol_id']
            short_pairs = mol['short_pairs']
            calc_names  = mol['calc_names']

            if verbose:
                short_summary = ", ".join(
                    f"{p['elem_i']}-{p['elem_j']}:{p['distance']:.3f}Å"
                    for p in short_pairs
                )
                print(f"    {mol_id} | calcs={len(calc_names)} | {short_summary}")

            for p in short_pairs:
                ei, ej = p['elem_i'], p['elem_j']

                if ei == 'H' and ej == 'H':
                    category = 'H-H'
                elif ei == 'H' or ej == 'H':
                    category = 'heavy-H'
                else:
                    category = 'heavy-heavy'

                pair_key = tuple(sorted([ei, ej]))
                pair_counter[pair_key] += 1
                category_counter[category] += 1

                for calc_name in calc_names:
                    csv_rows.append({
                        'mol_id':     mol_id,
                        'calc_name':  calc_name,
                        'elem_i':     ei,
                        'elem_j':     ej,
                        'distance_A': f"{p['distance']:.4f}",
                        'category':   category,
                    })

        # --- Category breakdown ---
        if category_counter:
            print("\n  Short-pair categories (geometry-level counts):")
            for cat in ['heavy-heavy', 'heavy-H', 'H-H']:
                print(f"    {cat:15s}: {category_counter.get(cat, 0)}")

        # --- Most frequent pairs ---
        if pair_counter:
            print("\n  Most frequent short-contact element pairs:")
            for pair, count in pair_counter.most_common(10):
                print(f"    {pair[0]}-{pair[1]:4s}: {count}")

        # --- CSV output ---
        if csv_rows:
            csv_file = output_dir / 'geometry_outliers.csv'
            fieldnames = ['mol_id', 'calc_name', 'elem_i', 'elem_j', 'distance_A', 'category']
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"\n  ✓ Saved {csv_file}")
        else:
            print("\n  No short pairs found — CSV not written.")





    def inspect_force_outliers(self, output_dir: Path, severe_threshold: float = 100.0):
        """Summarize force outliers (> force_outlier_threshold), with extra accounting above severe_threshold."""
        import csv

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        outliers = self.force_outliers
        print(f"\nForce outliers (> {self.force_outlier_threshold} Ha/Bohr): {len(outliers)} mol/calc pairs")

        if not outliers:
            return

        # --- Severity sub-accounting ---
        severe = [m for m in outliers if m['max_force'] > severe_threshold]
        print(f"  Severe (> {severe_threshold} Ha/Bohr): {len(severe)} mol/calc pairs")

        thresholds = [10, 50, 100, 200, 500, 1000]
        print("\n  Severity distribution (max force per mol/calc):")
        for t in thresholds:
            count = sum(1 for m in outliers if m['max_force'] > t)
            print(f"    > {t:5d} Ha/Bohr: {count:6d}")

        # --- Per calc_name breakdown ---
        from collections import Counter
        calc_counter = Counter(m['calc_name'] for m in outliers)
        severe_calc_counter = Counter(m['calc_name'] for m in severe)
        print("\n  Per method-basis (>10 / >100):")
        for calc_name in sorted(calc_counter):
            print(f"    {calc_name:<20s}: {calc_counter[calc_name]:6d} / {severe_calc_counter.get(calc_name, 0):6d}")

        # --- CSV: all outliers ---
        csv_rows = []
        for m in outliers:
            mol_id = m['mol_id']
            calc_name = m['calc_name']
            for elem, mag in zip(m['coordinate_elements'], m['force_mags']):
                if mag > self.force_outlier_threshold:
                    csv_rows.append({
                        'mol_id':     mol_id,
                        'calc_name':  calc_name,
                        'element':    elem,
                        'force_mag':  f"{mag:.4f}",
                        'severe':     'yes' if mag > severe_threshold else 'no',
                    })

        csv_file = output_dir / 'force_outliers.csv'
        fieldnames = ['mol_id', 'calc_name', 'element', 'force_mag', 'severe']
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n  ✓ Saved {csv_file}")

    def inspect_force_geometry_overlap(self, output_dir: Path, severe_threshold: float = 100.0,
                                        strained_dist_threshold: float = 0.5):
        """Cross-reference force outliers and geometry outliers by mol_id.

        Also reports strained geometries (min_dist < strained_dist_threshold) that have
        NO corresponding force outlier — potential SCF failures or missing data.

        Computes a Coulomb-normalized descriptor F*r^2 / (Z_i*Z_j) for each short pair
        in the overlap set, to assess whether repulsion is Coulomb-dominated.
        """
        import csv

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Build lookup sets / dicts
        force_mol_ids = {m['mol_id'] for m in self.force_outliers}
        geom_mol_ids  = {m['mol_id'] for m in self.geometry_outliers}
        overlap_ids   = force_mol_ids & geom_mol_ids

        print(f"\nForce/geometry outlier cross-reference:")
        print(f"  Force outlier molecules   : {len(force_mol_ids)}")
        print(f"  Geometry outlier molecules: {len(geom_mol_ids)}")
        print(f"  Overlap (both)            : {len(overlap_ids)}")

        # Build lookup dicts keyed by mol_id
        geom_by_mol = {m['mol_id']: m for m in self.geometry_outliers}
        from collections import defaultdict
        force_by_mol = defaultdict(list)
        for m in self.force_outliers:
            force_by_mol[m['mol_id']].append(m)

        # -------------------------------------------------------------------
        # Q1: Strained geometries (min_dist < threshold) with NO force outlier
        # -------------------------------------------------------------------
        strained_no_force = [
            m for m in self.geometry_outliers
            if any(p['distance'] < strained_dist_threshold for p in m['short_pairs'])
            and m['mol_id'] not in force_mol_ids
        ]
        print(f"\n  Strained (< {strained_dist_threshold} Å) with NO force outlier (>{self.force_outlier_threshold} Ha/Bohr): "
              f"{len(strained_no_force)} geometries")
        if strained_no_force:
            print(f"  (possible missing NPZ, SCF failure, or legitimate soft repulsion)")
            for m in strained_no_force:
                min_dist = min(p['distance'] for p in m['short_pairs'])
                short_summary = ", ".join(
                    f"{p['elem_i']}-{p['elem_j']}:{p['distance']:.3f}Å"
                    for p in m['short_pairs'] if p['distance'] < strained_dist_threshold
                )
                print(f"    {m['mol_id']} | min_dist={min_dist:.3f}Å | {short_summary}")

            csv_file_nf = output_dir / 'strained_no_force_outlier.csv'
            fieldnames_nf = ['mol_id', 'min_dist_A', 'short_pairs', 'calc_names']
            with open(csv_file_nf, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames_nf)
                writer.writeheader()
                for m in strained_no_force:
                    min_dist = min(p['distance'] for p in m['short_pairs'])
                    short_summary = ", ".join(
                        f"{p['elem_i']}-{p['elem_j']}:{p['distance']:.3f}Å"
                        for p in m['short_pairs'] if p['distance'] < strained_dist_threshold
                    )
                    writer.writerow({
                        'mol_id':      m['mol_id'],
                        'min_dist_A':  f"{min_dist:.4f}",
                        'short_pairs': short_summary,
                        'calc_names':  "|".join(m['calc_names']),
                    })
            print(f"  ✓ Saved {csv_file_nf}")

        # -------------------------------------------------------------------
        # Overlap summary + Coulomb-normalized descriptor
        # -------------------------------------------------------------------
        if not overlap_ids:
            print("\n  No overlap — force and geometry outliers are disjoint.")
            return

        print(f"\n  Overlap details (mol_id | max_force | min_dist | F*r²/(Zi*Zj) | short pairs):")
        csv_rows = []
        for mol_id in sorted(overlap_ids):
            geom         = geom_by_mol[mol_id]
            force_entries = force_by_mol[mol_id]
            max_force_all = max(e['max_force'] for e in force_entries)
            min_dist      = min(p['distance'] for p in geom['short_pairs'])

            # Coulomb-normalized descriptor for the closest pair
            closest = min(geom['short_pairs'], key=lambda p: p['distance'])
            zi = self.atomic_numbers.get(closest['elem_i'], 1)
            zj = self.atomic_numbers.get(closest['elem_j'], 1)
            r  = closest['distance']
            coulomb_desc = (max_force_all * r**2) / (zi * zj)  # Ha·Å²/e²

            short_summary = ", ".join(
                f"{p['elem_i']}-{p['elem_j']}:{p['distance']:.3f}Å"
                for p in geom['short_pairs']
            )
            print(f"    {mol_id} | F={max_force_all:8.1f} Ha/Bohr | r={min_dist:.3f}Å "
                  f"| F*r²/(Zi*Zj)={coulomb_desc:.4f} | {short_summary}")

            for e in force_entries:
                csv_rows.append({
                    'mol_id':          mol_id,
                    'calc_name':       e['calc_name'],
                    'max_force':       f"{e['max_force']:.4f}",
                    'severe':          'yes' if e['max_force'] > severe_threshold else 'no',
                    'min_dist_A':      f"{min_dist:.4f}",
                    'coulomb_desc':    f"{coulomb_desc:.6f}",
                    'zi':              zi,
                    'zj':              zj,
                    'short_pairs':     short_summary,
                })

        csv_file = output_dir / 'force_geometry_overlap.csv'
        fieldnames = ['mol_id', 'calc_name', 'max_force', 'severe', 'min_dist_A',
                      'coulomb_desc', 'zi', 'zj', 'short_pairs']
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n  ✓ Saved {csv_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run bond-length, chemical-space, force, energy, and outlier analyses."
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("."),
        help="Directory containing dataset_2node and dataset_3node (default: current directory)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("analysis_output"),
        help="Output directory for plots and tables (default: analysis_output)"
    )
    args = parser.parse_args()

    dataset_2node = args.dataset_root / "dataset_2node"
    dataset_3node = args.dataset_root / "dataset_3node"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Selected pairs for Task 2
    selected_pairs = [
        ('C', 'C'),
        ('C', 'N'),
        ('C', 'O'),
        ('S', 'O'),
        ('Mg', 'Cl'),
        ('Al', 'O')
    ]
    
    # Analyze 2-node dataset
    print("="*60)
    print("DATASET ANALYSIS - 2-NODE")
    print("="*60)
    
    analyzer_2node = DatasetAnalyzer(dataset_2node)
    analyzer_2node.load_and_analyze()
    
    output_2node = output_dir / "2node_new"
    output_2node.mkdir(exist_ok=True)
    
    analyzer_2node.plot_task1_half_bond_boxplots(output_2node)
    analyzer_2node.plot_task2_selected_bond_histograms(output_2node, selected_pairs)
    analyzer_2node.plot_task3_chemical_space_overview(output_2node)
    analyzer_2node.plot_task4_force_distribution(output_2node)
    analyzer_2node.calculate_task5_pair_abundance(output_2node)
    analyzer_2node.calculate_task6_pair_cooccurrence(output_2node)
    analyzer_2node.plot_task7_energy_distributions(output_2node)
    analyzer_2node.inspect_geometry_outliers(output_2node)
    analyzer_2node.inspect_force_outliers(output_2node)
    analyzer_2node.inspect_force_geometry_overlap(output_2node)

    print(f"\n✅ 2-node analysis complete! Results in: {output_2node}")
    
    # Analyze 3-node dataset
    print("\n" + "="*60)
    print("DATASET ANALYSIS - 3-NODE")
    print("="*60)
    #
    analyzer_3node = DatasetAnalyzer(dataset_3node)
    analyzer_3node.load_and_analyze()
    #
    output_3node = output_dir / "3node_new"
    output_3node.mkdir(exist_ok=True)
    #
    analyzer_3node.plot_task1_half_bond_boxplots(output_3node)
    analyzer_3node.plot_task2_selected_bond_histograms(output_3node, selected_pairs)
    analyzer_3node.plot_task3_chemical_space_overview(output_3node)
    analyzer_3node.plot_task4_force_distribution(output_3node)
    analyzer_3node.calculate_task5_pair_abundance(output_3node)
    analyzer_3node.calculate_task6_pair_cooccurrence(output_3node)
    analyzer_3node.plot_task7_energy_distributions(output_3node)
    analyzer_3node.inspect_geometry_outliers(output_3node)
    analyzer_3node.inspect_force_outliers(output_3node)
    analyzer_3node.inspect_force_geometry_overlap(output_3node)

    print(f"\n✅ 3-node analysis complete! Results in: {output_3node}")


if __name__ == "__main__":
    main()
