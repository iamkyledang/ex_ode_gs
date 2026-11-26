"""
The full pipeline for one scene, resumable:

    1 interface   train_interface.py    Phase I:  G_I + ExODE-GS against the reflection-removed images
    2 plane       extract_plane.py      the glass plane from the frozen interface
    3 reflection  train_reflection.py   Phase II: G_R + ExODE-GS against the original images, Phase I frozen
    4 render      render_vigor.py       base renders of the test views      -> test/ours_<N>/      (= "no refinement")
    5 refiner     train_refiner.py      causal autoregressive refiner        -> test/refined_<M>/   (= the full method)
    6 metrics     metrics.py            results.json over every method folder of test/

Each finished step leaves <model>/.done_<step>; a rerun skips it. Times go to <model>/timing.json.

    python run_exode.py --scene vsgd -m output/vsgd                       # scene keys: vsgd, wayve, 3dfront
    python run_exode.py -s /data/VSGD/0758 -r 2 -m output/vsgd
    python run_exode.py --scene wayve -m output/wayve_abl/full --iterations 12000 --refiner_iterations 2000

--iterations sets the iterations of each phase (default 30000); every iteration-based schedule of the phases (warm-up,
densification window, resets, learning-rate decay, time-noise annealing, first regularizer iterations) is scaled by
the same factor, so a shorter run is the same recipe compressed.

Variants (ablations): full (default), no_ode_interface (--static_interface), no_ode_reflection (--static_reflection),
no_ode_both (both). "No refinement" needs no run of its own: it is test/ours_<N>/ of any model. Every variant keeps
the mirrored virtual camera and the single glass plane.
--phase1_from <folder> copies Phase I and the glass plane of another model instead of training them (used by the
ablations that change only Phase II).
"""

import json
import os
import shutil
import subprocess
import sys
import time
from argparse import ArgumentParser

REPO = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.path.join(REPO, '..', 'vigor_gs_data')
# Scene key -> (path below the data root, resolution factor of the report)
SCENES = {
    'vsgd': ('VSGD/0758', 1),
    'wayve': ('WayveScene101/scene_001', 2),
    '3dfront': ('3D-FRONT-T/scene_1', 1),
}
VARIANTS = ('full', 'no_ode_interface', 'no_ode_reflection', 'no_ode_both')

INTERFACE_SCHEDULE = ('position_lr_max_steps', 'norm_loss_from_iter', 'reg_from_iter', 'occupancy_reset_interval', 'densify_from_iter', 'densify_until_iter', 'warm_up', 'time_noise_until_iter')
REFLECTION_SCHEDULE = ('position_lr_max_steps', 'opacity_reset_interval', 'densify_from_iter', 'densify_until_iter', 'warm_up', 'time_noise_until_iter')
DEFAULTS_30K = {
    'position_lr_max_steps': 30000,
    'norm_loss_from_iter': 700,
    'reg_from_iter': 700,
    'occupancy_reset_interval': 3000,
    'opacity_reset_interval': 3000,
    'densify_from_iter': 500,
    'densify_until_iter': 15000,
    'warm_up': 3000,
    'time_noise_until_iter': 20000,
}


def schedule_flags(iterations, names):
    """--iterations N plus every name of `names` scaled from its 30k default by N / 30000."""
    factor = iterations / 30000
    flags = ['--iterations', str(iterations), '--test_iterations', str(iterations)]
    for name in names:
        flags += [f'--{name}', str(max(1, round(DEFAULTS_30K[name] * factor)))]
    return flags


def run(cmd, log_path):
    print('\n$ ' + ' '.join(cmd), flush=True)
    with open(log_path, 'a') as log:
        log.write('\n$ ' + ' '.join(cmd) + '\n')
        log.flush()
        process = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        if process.wait() != 0:
            raise RuntimeError(f'step failed ({process.returncode}): {" ".join(cmd)} - see {log_path}')


def main():
    parser = ArgumentParser(description='ExODE-GS with causal autoregressive refinement: one scene, every stage')
    parser.add_argument('--scene', choices=sorted(SCENES), help='one of the three benchmark scenes (sets -s and -r)')
    parser.add_argument('-s', '--source_path', default='')
    parser.add_argument('-r', '--resolution', type=int, default=0)
    parser.add_argument('-m', '--model_path', required=True)
    parser.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--variant', choices=VARIANTS, default='full')
    parser.add_argument('--iterations', type=int, default=30000, help='iterations of each phase')
    parser.add_argument('--refiner_iterations', type=int, default=4000)
    parser.add_argument('--history', type=int, default=3, help='K, previous refined frames')
    parser.add_argument('--phase1_from', default='', help='copy Phase I and the glass plane from this model folder')
    parser.add_argument('--data_device', default='auto', help='cuda, cpu or auto (see utils/memory_utils.py)')
    parser.add_argument('--extra_interface', default='', help='extra flags for train_interface.py, quoted')
    parser.add_argument('--extra_reflection', default='', help='extra flags for train_reflection.py, quoted')
    parser.add_argument('--extra_refiner', default='', help='extra flags for train_refiner.py, quoted (e.g. "--flow_device cuda:1")')
    args = parser.parse_args()

    if args.scene:
        sub, resolution = SCENES[args.scene]
        args.source_path = args.source_path or os.path.join(args.data_root, sub)
        args.resolution = args.resolution or resolution
    if not args.source_path or not args.resolution:
        parser.error('give --scene, or -s and -r')
    model = os.path.abspath(args.model_path)
    os.makedirs(model, exist_ok=True)
    log_path = os.path.join(model, 'pipeline.log')
    timing_path = os.path.join(model, 'timing.json')
    timing = json.load(open(timing_path)) if os.path.exists(timing_path) else {}
    py = sys.executable

    phase1 = ['--static_interface'] if args.variant in ('no_ode_interface', 'no_ode_both') else []
    phase2 = ['--static_reflection'] if args.variant in ('no_ode_reflection', 'no_ode_both') else []
    steps = [
        ('interface', [py, 'train_interface.py', '-s', os.path.abspath(args.source_path), '-m', model, '-r', str(args.resolution), '--eval', '--data_device', args.data_device, '--quiet',
                       *schedule_flags(args.iterations, INTERFACE_SCHEDULE), *phase1, *args.extra_interface.split()]),
        ('plane', [py, 'extract_plane.py', '-m', model]),
        ('reflection', [py, 'train_reflection.py', '-m', model, '--quiet', *schedule_flags(args.iterations, REFLECTION_SCHEDULE), *phase2, *args.extra_reflection.split()]),
        ('render', [py, 'render_vigor.py', '-m', model, '--skip_train', '--quiet']),
        ('refiner', [py, 'train_refiner.py', '-m', model, '--refiner_iterations', str(args.refiner_iterations), '--history', str(args.history), '--quiet',
                     *args.extra_refiner.split()]),
        ('metrics', [py, 'metrics.py', '-m', model]),
    ]

    if args.phase1_from and not os.path.exists(os.path.join(model, '.done_plane')):
        source = os.path.abspath(args.phase1_from)
        for marker in ('interface', 'plane'):
            if not os.path.exists(os.path.join(source, f'.done_{marker}')):
                raise SystemExit(f'{source} has no finished {marker} step')
        shutil.rmtree(os.path.join(model, 'interface'), ignore_errors=True)
        shutil.copytree(os.path.join(source, 'interface'), os.path.join(model, 'interface'))
        shutil.copy(os.path.join(source, 'glass_plane.json'), os.path.join(model, 'glass_plane.json'))
        for marker in ('interface', 'plane'):
            open(os.path.join(model, f'.done_{marker}'), 'w').close()
        print(f'Phase I and the glass plane copied from {source}')

    for name, cmd in steps:
        done = os.path.join(model, f'.done_{name}')
        if os.path.exists(done):
            print(f'[skip] {name}')
            continue
        start = time.time()
        run(cmd, log_path)
        timing[name] = round((time.time() - start) / 60, 2)
        with open(timing_path, 'w') as f:
            json.dump(timing, f, indent=1)
        open(done, 'w').close()
    print(f'\nDone: {model} (minutes per step: {timing})')


if __name__ == '__main__':
    main()
