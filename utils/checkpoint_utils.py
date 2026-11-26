"""
Training checkpoints, so an interrupted Phase I, Phase II or refiner run continues where it stopped.

Every `--checkpoint_interval` iterations (default 1000) a training script writes <phase folder>/checkpoint.pth with
everything its loop needs to go on: the parameters with their optimizer state (and, for a Gaussian field, the
densification statistics and the ExODE-GS motion, which are part of `capture()`), the loop's own bookkeeping, and the
random number generator states. On start the script looks for that file and, when it belongs to the same schedule
(same total number of iterations), continues from the iteration after it. When the phase finishes, the checkpoint is
removed: the final model is saved in its usual place.

The file is written to <name>.tmp and then renamed, so a run killed while writing never leaves a broken checkpoint.
"""

import os
import random
import sys

import numpy as np
import torch

CHECKPOINT_NAME = 'checkpoint.pth'


def checkpoint_path(folder):
    return os.path.join(folder, CHECKPOINT_NAME)


def rng_state():
    state = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(folder, iteration, total_iterations, state):
    """Write the training state after `iteration` (of `total_iterations`)."""
    os.makedirs(folder, exist_ok=True)
    path = checkpoint_path(folder)
    tmp = path + '.tmp'
    torch.save({'iteration': iteration, 'total_iterations': total_iterations, 'rng': rng_state(), **state}, tmp)
    os.replace(tmp, path)
    print(f'\n[ITER {iteration}] Checkpoint saved: {path}', file=sys.stderr, flush=True)


def load_checkpoint(folder, total_iterations):
    """The saved state, or None when there is no checkpoint or it belongs to a different schedule."""
    path = checkpoint_path(folder)
    if not os.path.exists(path):
        return None
    try:
        state = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as error:  # a damaged file: start over rather than fail
        print(f'[ WARN ] Ignoring the unreadable checkpoint {path}: {error}', file=sys.stderr, flush=True)
        return None
    if state.get('total_iterations') != total_iterations:
        print(f'[ WARN ] Ignoring {path}: it is from a {state.get("total_iterations")}-iteration run, this one has {total_iterations}', file=sys.stderr, flush=True)
        return None
    print(f'Resuming from {path} after iteration {state["iteration"]} of {total_iterations}', file=sys.stderr, flush=True)
    return state


def to_cuda(value):
    """Move the tensors of a (nested) checkpoint entry to the GPU, keeping nn.Parameter as such."""
    if isinstance(value, torch.nn.Parameter):
        return torch.nn.Parameter(value.data.cuda(), requires_grad=value.requires_grad)
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, dict):
        return {k: to_cuda(v) for k, v in value.items()}
    return value


def remove_checkpoint(folder):
    for path in (checkpoint_path(folder), checkpoint_path(folder) + '.tmp'):
        if os.path.exists(path):
            os.remove(path)
