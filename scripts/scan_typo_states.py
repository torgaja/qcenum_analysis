#!/usr/bin/env python3
"""
Read-only scan of per-molecule metadata.json files for generator states
that originate from typos in element_data.py:

  A) Cl, valence 4, charge 0, unpaired 0    (intended: unpaired 1)
  B) N/P, valence 1, charge +1, unpaired 1  (intended: charge -1)

Secondary check: molecule-level electron parity vs. total unpaired electrons.
(Only detects an odd number of type-A atoms; type B is parity-consistent.)

Usage:
  python scan_typo_states.py [--base .] [--examples 3] [--json-out hits.json]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

Z = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
     'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16,
     'Cl': 17, 'Ar': 18}

DATASETS = ['dataset_2node', 'dataset_3node']


def iter_structures(obj, path='$'):
    """Yield every dict containing both 'elements' and 'atom_states',
    anywhere in the JSON tree (covers alternative states if stored)."""
    if isinstance(obj, dict):
        if 'elements' in obj and 'atom_states' in obj:
            yield path, obj
        for k, v in obj.items():
            yield from iter_structures(v, f'{path}.{k}')
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from iter_structures(v, f'{path}[{i}]')


def check_structure(s):
    elements = s['elements']
    states = s['atom_states']
    hits, warnings = [], []
    n_elec, n_unpaired = 0, 0
    parity_reliable = True

    if len(states) != len(elements):
        warnings.append(f'atom_states has {len(states)} entries, elements has {len(elements)}')
        parity_reliable = False

    for idx_str, st in states.items():
        idx = int(idx_str)
        el = elements[idx]
        v = st['valence']
        q = st['charge']
        r = st['unpaired_electrons']
        bo = st.get('bond_order')

        if el == 'Cl' and v == 4 and q == 0 and r == 0:
            hits.append(('Cl_v4_q0_r0', idx))
        if el in ('N', 'P') and v == 1 and q == 1 and r == 1:
            hits.append(('NP_v1_q+1_r1', idx))

        if bo is None:
            parity_reliable = False
            n_h = 0
        else:
            n_h = v - bo  # implicit hydrogens
            if n_h < 0:
                warnings.append(f'atom {idx} ({el}): valence {v} < bond_order {bo}')
                parity_reliable = False
        n_elec += Z[el] + n_h - q
        n_unpaired += r

    parity_ok = (n_elec % 2) == (n_unpaired % 2)
    return hits, warnings, parity_ok, parity_reliable, n_elec, n_unpaired


def scan_dataset(ds_dir, n_examples):
    res = {
        'files': 0, 'structures': 0, 'json_errors': [],
        'hit_dirs': defaultdict(set), 'hit_atoms': defaultdict(int),
        'examples': defaultdict(list),
        'parity_fail': [], 'parity_unreliable': 0, 'warnings': [],
    }
    for meta in sorted(ds_dir.glob('mol_*/metadata.json')):
        res['files'] += 1
        mol = meta.parent.name
        try:
            data = json.loads(meta.read_text())
        except Exception as e:
            res['json_errors'].append((mol, str(e)))
            continue

        for jpath, s in iter_structures(data):
            res['structures'] += 1
            try:
                hits, warns, p_ok, p_rel, n_e, n_u = check_structure(s)
            except Exception as e:
                res['warnings'].append((mol, jpath, f'check failed: {e}'))
                continue

            for w in warns:
                res['warnings'].append((mol, jpath, w))
            for cat, idx in hits:
                res['hit_dirs'][cat].add(mol)
                res['hit_atoms'][cat] += 1
                if len(res['examples'][cat]) < n_examples:
                    res['examples'][cat].append({
                        'mol': mol, 'json_path': jpath, 'atom': idx,
                        'smiles': s.get('smiles'),
                        'atom_state': s['atom_states'][str(idx)],
                    })
            if not p_rel:
                res['parity_unreliable'] += 1
            elif not p_ok:
                res['parity_fail'].append({
                    'mol': mol, 'json_path': jpath, 'smiles': s.get('smiles'),
                    'n_electrons': n_e, 'sum_unpaired': n_u,
                })
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='.', help='Directory containing dataset_2node and dataset_3node')
    ap.add_argument('--examples', type=int, default=3)
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    report = {}
    for ds in DATASETS:
        ds_dir = Path(args.base) / ds
        if not ds_dir.is_dir():
            print(f'[{ds}] directory not found: {ds_dir}')
            continue
        r = scan_dataset(ds_dir, args.examples)

        print(f'\n=== {ds} ===')
        print(f'metadata.json files: {r["files"]}, structures checked: {r["structures"]}')
        print(f'JSON errors: {len(r["json_errors"])}, warnings: {len(r["warnings"])}')
        for cat in ('Cl_v4_q0_r0', 'NP_v1_q+1_r1'):
            print(f'{cat}: {len(r["hit_dirs"][cat])} molecule dirs, {r["hit_atoms"][cat]} atoms')
            for ex in r['examples'][cat]:
                print(f'   e.g. {ex["mol"]} {ex["json_path"]} atom {ex["atom"]} '
                      f'{ex["smiles"]} {ex["atom_state"]}')
        print(f'parity failures: {len(r["parity_fail"])} '
              f'(unreliable/skipped: {r["parity_unreliable"]})')
        for pf in r['parity_fail'][:args.examples]:
            print(f'   e.g. {pf}')

        report[ds] = {
            'files': r['files'], 'structures': r['structures'],
            'json_errors': r['json_errors'], 'warnings': r['warnings'],
            'hit_dirs': {k: sorted(v) for k, v in r['hit_dirs'].items()},
            'hit_atoms': dict(r['hit_atoms']),
            'examples': dict(r['examples']),
            'parity_fail': r['parity_fail'],
            'parity_unreliable': r['parity_unreliable'],
        }

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f'\nFull report written to {args.json_out}')


if __name__ == '__main__':
    main()
