#!/usr/bin/env python3
"""
check_alternatives.py

For geometries with several electronic-structure specifications, compare the
(total charge, multiplicity) derived from the metadata atom_states with the
(charge, multiplicity) actually used in each .qcout ($molecule block).
Read-only.

Usage:
    python check_alternatives.py --dataset-root PATH --dataset dataset_2node
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

MOL_RX = re.compile(r'\$molecule\s*\n\s*(-?\d+)\s+(\d+)', re.IGNORECASE)


def spec_qm(spec):
    states = list(spec['atom_states'].values())
    q = sum(s['charge'] for s in states)
    m = sum(s['unpaired_electrons'] for s in states) + 1  # high-spin convention
    return (q, m)


def qcout_qm(path):
    with open(path, encoding='utf-8', errors='replace') as fh:
        head = fh.read(20000)
    m = MOL_RX.search(head)
    return (int(m.group(1)), int(m.group(2))) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', type=Path, required=True)
    ap.add_argument('--dataset', default='dataset_2node')
    ap.add_argument('--show', type=int, default=15)
    args = ap.parse_args()

    mol_dirs = sorted((args.dataset_root / args.dataset).glob('mol_*'))
    n_alt = 0
    n_multi_qm = 0
    ncalc_by_nqm = Counter()
    problems = {'qm_not_in_specs': [], 'qm_group_without_calc': [],
                'rep_qm_mismatch': [], 'qcout_unparsed': []}

    for d in mol_dirs:
        mf = d / 'metadata.json'
        if not mf.exists():
            continue
        meta = json.loads(mf.read_text(encoding='utf-8'))
        if meta.get('num_alternative_structures', 1) <= 1:
            continue
        n_alt += 1
        specs = meta['alternative_structures_details']
        spec_set = {spec_qm(s) for s in specs}
        if len(spec_set) > 1:
            n_multi_qm += 1

        rep = meta['representative_structure']
        for s in specs:
            if s['molecular_hash'] == rep['molecular_hash']:
                if spec_qm(s) != (rep['total_charge'], rep['multiplicity']):
                    problems['rep_qm_mismatch'].append(d.name)

        run_qm = {}
        for name, info in meta['available_calculations'].items():
            qm = qcout_qm(d / info['qcout_file'])
            if qm is None:
                problems['qcout_unparsed'].append((d.name, name))
                continue
            run_qm[name] = qm
            if qm not in spec_set:
                problems['qm_not_in_specs'].append((d.name, name, qm))

        for qm in spec_set - set(run_qm.values()):
            problems['qm_group_without_calc'].append((d.name, qm))
        ncalc_by_nqm[(len(spec_set), len(run_qm))] += 1

    print(f'{args.dataset}: geometries with alternatives: {n_alt}')
    print(f'  with >=2 distinct (Q, 2S+1): {n_multi_qm}')
    print(f'  with a single (Q, 2S+1): {n_alt - n_multi_qm}')
    print('  (n distinct (Q,M), n calculations) -> geometries:')
    for k in sorted(ncalc_by_nqm):
        print(f'    {k}: {ncalc_by_nqm[k]}')
    for key, rows in problems.items():
        print(f'{key}: {len(rows)}')
        for r in rows[:args.show]:
            print('   ', r)


if __name__ == '__main__':
    main()
