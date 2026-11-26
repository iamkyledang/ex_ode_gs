"""
ExODE-GS: explicit ODE-guided motion of every Gaussian (replaces the deformation MLPs of both fields).

Each Gaussian i carries four motion parameters Theta_i = {A_i, b_i, omega_i, kappa_i} and follows closed-form
continuous-time dynamics in the normalized time tau in [0, 1]:

    center     d(mu_i - mu_i^0)/dtau = A_i (mu_i - mu_i^0) + b_i      (linear ODE with drift, written in the
                                                                       Gaussian's own frame centred at mu_i^0)
               exp(tau [[A_i, b_i], [0^T, 0]]) = [[M_i(tau), q_i(tau)], [0^T, 1]]
               mu_i(tau) = mu_i^0 + q_i(tau)
    rotation   R_i(tau) = exp([omega_i]_x tau) R_i^0                   (quaternion: q_i(tau) = q_omega(tau) (x) q_i^0)
    scale      s_i(tau) = s_i^0 (.) exp(kappa_i tau)                   (log s_i(tau) = log s_i^0 + kappa_i tau)

Occupancy, opacity and colour stay canonical. Every parameter starts at zero, so a new field does not move.

Regularizers (both phases):

    L_ode   = 1/N sum_i lambda_A ||A_i||_F^2 + lambda_b ||b_i / extent||^2 + lambda_w ||omega_i||^2 + lambda_k ||kappa_i||^2
    L_scale = 1/N sum_i max(0, ||log(s_i(tau) / extent)||_inf - s_max)^2

Implementation choices that keep the formulation unchanged:
  - The center ODE uses the Gaussian's local frame (its canonical center is the origin), so the motion does not
    depend on where the world origin is. In that frame mu(0) = 0, so M(tau) mu^0 vanishes and only q(tau) remains.
  - q(tau) is evaluated with the augmented 4x4 matrix exponential (torch.linalg.matrix_exp, exact for singular A),
    in chunks of `chunk` Gaussians to bound the temporaries (as in the ExODE-GS paper).
  - The rotation exponential is evaluated in closed form (Rodrigues, as a unit quaternion) and composed with the
    canonical rotation by a quaternion product, which is the same as exp([omega]_x tau) R^0.
  - b and the scales are measured in units of the scene extent inside the regularizers (b's learning rate is scaled
    by the extent too), so the same weights work for an indoor room and a street.
"""

import torch
import torch.nn.functional as F

ODE_PARAMS = ('ode_A', 'ode_b', 'ode_omega', 'ode_kappa')


def ode_param_dims(scale_dim):
    """Width of each per-Gaussian motion parameter (A is stored flattened, row major)."""
    return {'ode_A': 9, 'ode_b': 3, 'ode_omega': 3, 'ode_kappa': scale_dim}


def zero_ode_params(num_points, scale_dim, device='cuda'):
    return {name: torch.zeros((num_points, dim), device=device) for name, dim in ode_param_dims(scale_dim).items()}


def affine_drift(A, b, tau, chunk=100_000):
    """
    q(tau): the top-right column of exp(tau [[A, b], [0, 0]]), i.e. the solution of dp/dtau = A p + b, p(0) = 0.

    A: (N, 9) row-major 3x3 matrices, b: (N, 3), tau: float or (N, 1). Returns (N, 3).
    """
    n = A.shape[0]
    tau = torch.as_tensor(tau, dtype=A.dtype, device=A.device).reshape(-1, 1)
    tau = tau.expand(n, 1) if tau.shape[0] == 1 else tau
    pieces = []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        m = A.new_zeros((stop - start, 4, 4))
        m[:, :3, :3] = A[start:stop].reshape(-1, 3, 3)
        m[:, :3, 3] = b[start:stop]
        pieces.append(torch.linalg.matrix_exp(m * tau[start:stop, :, None])[:, :3, 3])
    return torch.cat(pieces, dim=0) if pieces else A.new_zeros((0, 3))


def axis_angle_quaternion(omega, tau):
    """
    exp([omega]_x tau) as a unit quaternion (w, x, y, z): (cos(theta/2), sin(theta/2) omega/|omega|), theta = |omega| tau.

    sin(theta/2)/|omega tau| is replaced by its Taylor series near 0, so the gradient is finite at omega = 0.
    """
    phi = omega * tau
    theta = phi.norm(dim=-1, keepdim=True)
    small = theta < 1e-4
    safe = torch.where(small, torch.ones_like(theta), theta)
    sinc_half = torch.where(small, 0.5 - theta**2 / 48.0, torch.sin(0.5 * safe) / safe)
    return torch.cat([torch.cos(0.5 * theta), sinc_half * phi], dim=-1)


def quaternion_multiply(q1, q2):
    """Hamilton product q1 (x) q2 of (w, x, y, z) quaternions: the rotation R(q1) R(q2)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def ode_state(xyz, rotation, log_scaling, A, b, omega, kappa, tau, chunk=100_000):
    """
    The Gaussians' geometry at time tau.

    xyz (N, 3) canonical centers, rotation (N, 4) unit quaternions, log_scaling (N, S) raw log scales, and the
    motion parameters. Returns (centers, unit quaternions, positive scales).
    """
    tau_t = torch.as_tensor(tau, dtype=xyz.dtype, device=xyz.device).reshape(-1, 1)
    centers = xyz + affine_drift(A, b, tau_t, chunk)
    rotations = F.normalize(quaternion_multiply(axis_angle_quaternion(omega, tau_t), rotation), dim=-1)
    scales = torch.exp((log_scaling + kappa * tau_t).clamp(-20.0, 10.0))
    return centers, rotations, scales


def ode_regularizer(field, weights, extent):
    """L_ode of the field, averaged over its Gaussians. `weights` = (lambda_A, lambda_b, lambda_w, lambda_k)."""
    lam_a, lam_b, lam_w, lam_k = weights
    terms = (
        lam_a * field._ode_A.pow(2).sum(-1)
        + lam_b * (field._ode_b / extent).pow(2).sum(-1)
        + lam_w * field._ode_omega.pow(2).sum(-1)
        + lam_k * field._ode_kappa.pow(2).sum(-1)
    )
    return terms.mean()


def scale_regularizer(scales, extent, s_max):
    """L_scale: penalize Gaussians whose largest |log scale| (in units of the extent) exceeds s_max."""
    excess = (torch.log(scales / extent).abs().amax(dim=-1) - s_max).clamp_min(0.0)
    return excess.pow(2).mean()


class ExODEMotion:
    """
    The motion model of one field: geometry at time t, and the two regularizers.

    `field` is a GaussianField built with motion parameters (field.dynamic). A static field (the ablations without
    ExODE-GS in G_I or G_R) has none, and `geometry` returns the canonical Gaussians at every time.
    """

    def __init__(self, field, chunk=100_000):
        self.field = field
        self.chunk = chunk

    @property
    def dynamic(self):
        return getattr(self.field, 'dynamic', False)

    def geometry(self, time, enabled=True, time_noise=0.0):
        """
        (centers, rotations, scales) at `time`. `enabled=False` (the static warm-up) or a static field returns the
        canonical geometry. `time_noise` is the std of a training-time jitter on the timestamp (annealed by the
        caller), which smooths the motion between the training frames.
        """
        f = self.field
        if not (self.dynamic and enabled):
            return f.get_xyz, f.get_rotation, f.get_scaling
        tau = float(time)
        if time_noise > 0:
            tau += float(torch.randn(1)) * time_noise
        return ode_state(f.get_xyz, f.get_rotation, f._scaling, f._ode_A, f._ode_b, f._ode_omega, f._ode_kappa, tau, self.chunk)

    def regularizers(self, scales, opt, extent):
        """{'ode': L_ode, 'scale': lambda_scale * L_scale} (weighted), empty for a static field."""
        if not self.dynamic:
            return {}
        losses = {}
        weights = (opt.lambda_ode_A, opt.lambda_ode_b, opt.lambda_ode_omega, opt.lambda_ode_kappa)
        if any(w > 0 for w in weights):
            losses['ode'] = ode_regularizer(self.field, weights, extent)
        if opt.lambda_scale > 0:
            losses['scale'] = opt.lambda_scale * scale_regularizer(scales, extent, opt.s_max)
        return losses
