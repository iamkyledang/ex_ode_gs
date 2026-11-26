"""
The reconstruction forward pass for one camera (before the causal refinement, refiner/):

    B_t        = R_2D(G_I(t), C_t)                     direct image, from the frozen interface   (render_direct)
    R_t        = R_3D(G_R(t), V(C_t, Pi))              reflection image through the virtual camera
    I_t^base   = (1 - alpha_t) B_t + alpha_t R_t       Fresnel-guided blend                      (render_composite)

G_I(t) and G_R(t) are the fields moved to time t by their own ExODE-GS motion (scene/exode.py).
"""

import torch

from gaussian_renderer.interface import render_interface
from gaussian_renderer.reflection import render_reflection
from scene.interface_model import InterfaceGaussianModel
from utils.fresnel import BlendModel, composite
from utils.virtual_camera import reflect_camera


@torch.no_grad()
def render_direct(camera, interface, interface_motion, pipe, background):
    """Render the interface field at the camera's timestamp. Nothing here carries gradients: Phase I is frozen."""
    return render_interface(camera, interface, pipe, background, geometry=interface_motion.geometry(camera.time))


def render_composite(camera, direct, glass_map, reflection_field, pipe, normal, offset, reflection_geometry=None, blend: BlendModel = None, use_virtual_camera=True):
    """
    Render the reflection and blend it with a direct image.

    direct: (3, H, W) B_t and glass_map: (1, H, W), both from `render_direct` (or its cache).
    reflection_field: the reflection field G_R. Passing the interface field instead renders that one through the
    virtual camera (the shared-field ablation: the reflection is the mirrored direct scene).
    reflection_geometry: (centers, rotations, scales) of that field at the camera's time (its ExODEMotion), or None
    for the canonical field.
    blend: decides alpha (None: pure Schlick with the default indices).
    use_virtual_camera=False renders the reflection from the real camera instead (ablation).
    Returns the reflection render's entries plus alpha, fresnel, cos_theta, h, direct, glass_map and composite.
    """
    view = reflect_camera(camera, normal, offset) if use_virtual_camera else camera
    black = torch.zeros(3, dtype=torch.float32, device='cuda')  # no Gaussian, no reflection
    if isinstance(reflection_field, InterfaceGaussianModel):
        rendered = render_interface(view, reflection_field, pipe, black, geometry=reflection_geometry)
        out = {'reflection': rendered['direct'], 'viewspace_points': rendered['viewspace_points'], 'visibility_filter': rendered['visibility_filter'], 'radii': rendered['radii']}
    else:
        out = render_reflection(view, reflection_field, pipe, black, geometry=reflection_geometry)

    blend = BlendModel() if blend is None else blend
    out.update(blend(camera, normal, offset, glass_map=glass_map))
    out['direct'] = direct
    out['glass_map'] = glass_map
    out['composite'] = composite(direct, out['reflection'], out['alpha'])
    return out
