"""
Main experiment: the full method on the three benchmark scenes, spread over several GPUs (2x T4 by default).

Each GPU runs its scenes one after another with run_exode.py (resumable); the GPUs run in parallel. The default split
balances the work of a 2x T4 machine: WayveScene101 (the largest images) on GPU 0, VSGD and 3D-FRONT-T on GPU 1.

    python run_main.py --out output/main                                    # 30k + 30k iterations, 4k refiner
    python run_main.py --out output/main --iterations 20000                 # if 30k does not fit the time budget
    python run_main.py --out output/main --data_root /kaggle/input/exode-data

Writes <out>/<scene>/ (one model folder per scene), <out>/<scene>.log and, at the end, <out>/summary.csv.
"""

import csv
import json
import os
import subprocess
import sys
import time
from argparse import ArgumentParser

from run_exode import DEFAULT_DATA_ROOT, REPO

DEFAULT_SPLIT = 'wayve;vsgd,3dfront'  # GPU groups separated by ';'


def summarize(out, scenes, rows_from):
    rows = []
    for scene in scenes:
        path = os.path.join(out, scene, 'results.json')
        if not os.path.exists(path):
            continue
        results = json.load(open(path))
        for method, metrics in sorted(results.items()):
            rows.append({'scene': scene, 'method': rows_from(method), **{k: round(v, 4) for k, v in metrics.items()}})
    if rows:
        keys = ['scene', 'method'] + sorted({k for r in rows for k in r} - {'scene', 'method'})
        with open(os.path.join(out, 'summary.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f'Summary: {os.path.join(out, "summary.csv")}')


def method_name(folder):
    if folder.startswith('refined_'):
        return 'ExODE-GS + causal AR refinement (' + folder + ')'
    if folder.startswith('ours_'):
        return 'ExODE-GS, no refinement (' + folder + ')'
    return folder


def main():
    parser = ArgumentParser(description='Main experiment on several GPUs')
    parser.add_argument('--out', required=True)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--split', default=DEFAULT_SPLIT, help="scene groups per GPU, e.g. 'wayve;vsgd,3dfront'")
    parser.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--iterations', type=int, default=30000)
    parser.add_argument('--refiner_iterations', type=int, default=4000)
    parser.add_argument('--history', type=int, default=3)
    args = parser.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    gpus = args.gpus.split(',')
    groups = [g.split(',') for g in args.split.split(';')]
    if len(groups) > len(gpus):
        groups = [[s for g in groups for s in g]]  # one GPU: everything in sequence
    processes = []
    for gpu, group in zip(gpus, groups):
        commands = []
        for scene in group:
            commands.append(' '.join([sys.executable, 'run_exode.py', '--scene', scene, '-m', os.path.join(out, scene), '--data_root', args.data_root, '--iterations', str(args.iterations),
                                      '--refiner_iterations', str(args.refiner_iterations), '--history', str(args.history), f'> {os.path.join(out, scene)}.log 2>&1']))
        script = ' && '.join(commands)
        print(f'GPU {gpu}: {group}')
        processes.append(subprocess.Popen(['bash', '-c', script], cwd=REPO, env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu}))
    start = time.time()
    codes = [p.wait() for p in processes]
    print(f'Finished in {(time.time() - start) / 3600:.2f} h, exit codes {codes}')
    summarize(out, [s for g in groups for s in g], method_name)


if __name__ == '__main__':
    main()
