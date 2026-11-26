"""
Where the training stages keep their results, and how to load them back.

    <model_path>/interface/     Phase I: cfg_args, point_cloud/iteration_N/point_cloud.ply (the disks and their
                                ExODE-GS motion parameters A, b, omega, kappa)
    <model_path>/glass_plane.json
    <model_path>/reflection/    Phase II: same structure, plus blend/iteration_N/blend.pth (F0 and h_psi)
    <model_path>/refiner/       the causal autoregressive refiner (refiner/, train_refiner.py)
"""

import copy
import os

import torch

from scene import Scene
from scene.exode import ExODEMotion
from scene.interface_model import InterfaceGaussianModel
from scene.reflection_model import ReflectionGaussianModel
from utils.fresnel import BlendModel
from utils.system_utils import searchForMaxIteration

INTERFACE_DIR = 'interface'
REFLECTION_DIR = 'reflection'
PLANE_FILE = 'glass_plane.json'


def phase_args(dataset, phase_dir):
    """A copy of the model parameters whose model_path points at one phase's folder (what `Scene` expects)."""
    args = copy.copy(dataset)
    args.model_path = os.path.join(dataset.model_path, phase_dir)
    return args


def new_interface(dataset):
    return InterfaceGaussianModel(dataset.sh_degree, factorized=not getattr(dataset, 'ablate_factorization', False), dynamic=not getattr(dataset, 'static_interface', False))


def new_reflection(dataset):
    return ReflectionGaussianModel(dataset.sh_degree, dynamic=not getattr(dataset, 'static_reflection', False))


def load_interface(dataset, iteration=-1, shuffle=False):
    """
    Load a trained Phase I result: the cameras, the interface field and its ExODE-GS motion.

    dataset: model parameters with model_path at the model root. Returns (scene, gaussians, motion).
    """
    gaussians = new_interface(dataset)
    scene = Scene(phase_args(dataset, INTERFACE_DIR), gaussians, load_iteration=iteration, shuffle=shuffle)
    return scene, gaussians, ExODEMotion(gaussians, getattr(dataset, 'ode_chunk', 100_000))


def save_reflection(model_path, iteration, blend, gaussians=None):
    """
    Write a Phase II result under <model_path>/reflection.

    The blend parameters are always saved. The field (with its motion parameters) is left out in the shared-field
    ablation, which has no reflection field of its own.
    """
    reflection_path = os.path.join(model_path, REFLECTION_DIR)
    blend_dir = os.path.join(reflection_path, 'blend', 'iteration_{}'.format(iteration))
    os.makedirs(blend_dir, exist_ok=True)
    torch.save(blend.state_dict(), os.path.join(blend_dir, 'blend.pth'))
    if gaussians is not None:
        gaussians.save_ply(os.path.join(reflection_path, 'point_cloud', 'iteration_{}'.format(iteration), 'point_cloud.ply'))


def load_reflection(dataset, iteration=-1):
    """
    Load a trained Phase II result. Returns (gaussians, motion, blend, iteration).

    In the shared-field ablation gaussians and motion are None: the caller renders the interface field instead.
    """
    reflection_path = os.path.join(dataset.model_path, REFLECTION_DIR)
    if iteration == -1:
        iteration = searchForMaxIteration(os.path.join(reflection_path, 'blend'))

    blend = BlendModel.from_args(dataset).cuda()
    blend.load_state_dict(torch.load(os.path.join(reflection_path, 'blend', 'iteration_{}'.format(iteration), 'blend.pth')))

    if getattr(dataset, 'shared_reflection_field', False):
        return None, None, blend, iteration

    gaussians = new_reflection(dataset)
    gaussians.load_ply(os.path.join(reflection_path, 'point_cloud', 'iteration_{}'.format(iteration), 'point_cloud.ply'))
    return gaussians, ExODEMotion(gaussians, getattr(dataset, 'ode_chunk', 100_000)), blend, iteration
