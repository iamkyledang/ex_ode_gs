"""
The ablation suite on one GPU (1x L4 by default), every scene with the same short schedule.

Every variant keeps the mirrored virtual camera and the single glass plane; only the motion model and the refinement
change. For each scene, in this order (each trained variant has its own model folder <out>/<scene>/<variant>/):

    full               the method with the short schedule: the reference row
    no_ode_reflection  G_R without ExODE-GS (static); Phase I and the glass plane copied from full
    no_ode_interface   G_I without ExODE-GS (static); everything retrained
    no_ode_both        G_I and G_R without ExODE-GS; Phase I and the glass plane copied from no_ode_interface
    (no_refinement)    no run of its own: full's test/ours_<N>/, the composite before the refiner

Rows of <out>/summary.csv (all from test/refined_<M>/ except no_refinement):
    full, no_ode_interface, no_ode_reflection, no_ode_both, no_refinement

    python run_ablations.py --out output/ablations
    python run_ablations.py --out output/ablations --scenes vsgd,3dfront --iterations 10000
"""

import csv
import json
import os
import subprocess
import sys
import time
from argparse import ArgumentParser

from run_exode import DEFAULT_DATA_ROOT, REPO

ORDER = ('full', 'no_ode_reflection', 'no_ode_interface', 'no_ode_both')
PHASE1_FROM = {'no_ode_reflection': 'full', 'no_ode_both': 'no_ode_interface'}  # variants that reuse a static/dynamic G_I
ROWS = ('full', 'no_ode_interface', 'no_ode_reflection', 'no_ode_both', 'no_refinement')


def rows_of(scene, variant, folder):
    path = os.path.join(folder, 'results.json')
    if not os.path.exists(path):
        return []
    picks = []
    for method, metrics in json.load(open(path)).items():
        if method.startswith('refined_'):
            picks.append((variant, method, metrics))
        elif variant == 'full' and method.startswith('ours_'):
            picks.append(('no_refinement', method, metrics))
    return [{'scene': scene, 'variant': v, 'folder': m, **{k: round(x, 4) for k, x in metrics.items()}} for v, m, metrics in picks]


def main():
    parser = ArgumentParser(description='Ablation suite (one GPU)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--scenes', default='vsgd,wayve,3dfront')
    parser.add_argument('--variants', default=','.join(ORDER))
    parser.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--iterations', type=int, default=12000, help='iterations of each phase (short schedule)')
    parser.add_argument('--refiner_iterations', type=int, default=2000)
    parser.add_argument('--history', type=int, default=3)
    parser.add_argument('--tag', default='', help='status_<tag>.json, so several processes (one per GPU) can share --out')
    parser.add_argument('--no_summary', action='store_true', help='do not write summary.csv (another process writes it)')
    parser.add_argument('--summary_only', action='store_true', help='only collect summary.csv from the finished variants')
    args = parser.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    variants = [v for v in ORDER if v in args.variants.split(',')]
    status_path = os.path.join(out, f'status_{args.tag}.json' if args.tag else 'status.json')
    status = json.load(open(status_path)) if os.path.exists(status_path) else {}
    common = ['--data_root', args.data_root, '--iterations', str(args.iterations), '--refiner_iterations', str(args.refiner_iterations), '--history', str(args.history)]

    for scene in [] if args.summary_only else args.scenes.split(','):
        os.makedirs(os.path.join(out, scene), exist_ok=True)
        for variant in variants:
            folder = os.path.join(out, scene, variant)
            cmd = [sys.executable, 'run_exode.py', '--scene', scene, '-m', folder, '--variant', variant, *common]
            if variant in PHASE1_FROM:
                source = os.path.join(out, scene, PHASE1_FROM[variant])
                if os.path.exists(os.path.join(source, '.done_plane')):
                    cmd += ['--phase1_from', source]  # otherwise Phase I is trained here
            start = time.time()
            with open(folder + '.log', 'a') as log:
                code = subprocess.call(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            status[f'{scene}/{variant}'] = {'exit': code, 'minutes': round((time.time() - start) / 60, 1)}
            with open(status_path, 'w') as f:
                json.dump(status, f, indent=1)
            print(f'{scene}/{variant}: exit {code}, {status[f"{scene}/{variant}"]["minutes"]} min', flush=True)

    if args.no_summary:
        return
    rows = [r for scene in args.scenes.split(',') for v in variants for r in rows_of(scene, v, os.path.join(out, scene, v))]
    rows.sort(key=lambda r: (r['scene'], ROWS.index(r['variant'])))
    if rows:
        keys = ['scene', 'variant', 'folder'] + sorted({k for r in rows for k in r} - {'scene', 'variant', 'folder'})
        with open(os.path.join(out, 'summary.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f'Summary: {os.path.join(out, "summary.csv")}')


if __name__ == '__main__':
    main()
