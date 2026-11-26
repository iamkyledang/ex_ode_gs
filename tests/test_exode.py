"""ExODE-GS motion (scene/exode.py): the closed forms against independent references."""

import math

import torch

from scene.exode import affine_drift, axis_angle_quaternion, ode_regularizer, ode_state, quaternion_multiply, scale_regularizer
from utils.general_utils import build_rotation


def euler_drift(A, b, tau, steps=20000):
    """Integrate dp/dtau = A p + b, p(0) = 0, with small explicit steps (the reference for q(tau))."""
    p = torch.zeros_like(b, dtype=torch.float64)
    A64, b64, h = A.double().reshape(-1, 3, 3), b.double(), tau / steps
    for _ in range(steps):
        k1 = (A64 @ p[..., None])[..., 0] + b64
        k2 = (A64 @ (p + 0.5 * h * k1)[..., None])[..., 0] + b64
        p = p + h * k2
    return p


def test_drift_matches_integration_including_singular_A():
    torch.manual_seed(0)
    A = torch.randn(6, 9) * 0.8
    A[0] = 0.0  # A = 0: pure drift, q = b tau
    A[1] = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).reshape(9)  # nilpotent, singular
    b = torch.randn(6, 3)
    q = affine_drift(A, b, 0.7)
    assert torch.allclose(q.double(), euler_drift(A, b, 0.7), atol=1e-4)
    assert torch.allclose(q[0], b[0] * 0.7, atol=1e-6)


def test_drift_is_chunk_independent():
    torch.manual_seed(1)
    A, b = torch.randn(250, 9) * 0.3, torch.randn(250, 3)
    assert torch.allclose(affine_drift(A, b, 0.4, chunk=1000), affine_drift(A, b, 0.4, chunk=64), atol=1e-6)


def rodrigues(omega, tau):
    phi = omega * tau
    theta = phi.norm()
    if theta < 1e-12:
        return torch.eye(3)
    k = phi / theta
    K = torch.tensor([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return torch.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K


def test_rotation_is_exp_of_omega_times_canonical():
    torch.manual_seed(2)
    omega = torch.randn(5, 3)
    omega[0] = 0.0
    q0 = torch.nn.functional.normalize(torch.randn(5, 4), dim=-1)
    q = torch.nn.functional.normalize(quaternion_multiply(axis_angle_quaternion(omega, 0.6), q0), dim=-1)
    R, R0 = build_rotation(q.cuda() if torch.cuda.is_available() else q), build_rotation(q0.cuda() if torch.cuda.is_available() else q0)
    for i in range(5):
        expected = rodrigues(omega[i], 0.6) @ R0[i].cpu()
        assert torch.allclose(R[i].cpu(), expected, atol=1e-5)


def test_state_at_zero_time_and_zero_motion_is_canonical():
    torch.manual_seed(3)
    n = 40
    xyz, rot = torch.randn(n, 3), torch.nn.functional.normalize(torch.randn(n, 4), dim=-1)
    log_s = torch.randn(n, 2)
    A, b, w, k = torch.randn(n, 9), torch.randn(n, 3), torch.randn(n, 3), torch.randn(n, 2)
    c, r, s = ode_state(xyz, rot, log_s, A, b, w, k, 0.0)
    assert torch.allclose(c, xyz) and torch.allclose(r, rot, atol=1e-6) and torch.allclose(s, log_s.exp())
    z = torch.zeros
    c, r, s = ode_state(xyz, rot, log_s, z(n, 9), z(n, 3), z(n, 3), z(n, 2), 0.9)
    assert torch.allclose(c, xyz) and torch.allclose(r, rot, atol=1e-6) and torch.allclose(s, log_s.exp())


def test_scale_is_exponential_rate():
    log_s, kappa = torch.tensor([[0.1, -2.0, 0.5]]), torch.tensor([[0.3, -1.0, 0.0]])
    z = torch.zeros
    _, _, s = ode_state(torch.zeros(1, 3), torch.tensor([[1.0, 0, 0, 0]]), log_s, z(1, 9), z(1, 3), z(1, 3), kappa, 0.5)
    assert torch.allclose(s, log_s.exp() * torch.exp(kappa * 0.5))


def test_gradients_reach_every_motion_parameter_from_rest():
    n = 8
    params = [torch.zeros(n, d, requires_grad=True) for d in (9, 3, 3, 3)]
    xyz = torch.randn(n, 3)
    rot = torch.nn.functional.normalize(torch.randn(n, 4), dim=-1)
    c, r, s = ode_state(xyz, rot, torch.zeros(n, 3), *params, 0.5)
    (c.sum() + (r * torch.randn_like(r)).sum() + s.sum()).backward()
    A, b, w, k = (p.grad for p in params)
    assert all(torch.isfinite(g).all() for g in (A, b, w, k))
    # At rest, q = b tau + A b tau^2/2 + ...: b moves the center, A only through b, so dA = 0 while b = 0.
    assert b.abs().sum() > 0 and w.abs().sum() > 0 and k.abs().sum() > 0


class Field:
    def __init__(self, n):
        self._ode_A, self._ode_b = torch.ones(n, 9), torch.ones(n, 3) * 2.0
        self._ode_omega, self._ode_kappa = torch.ones(n, 3), torch.ones(n, 2)


def test_regularizers():
    f = Field(5)
    # 9 * 1 * lamA + 3 * (2 / 2)^2 * lamB + 3 * lamW + 2 * lamK
    assert math.isclose(ode_regularizer(f, (1.0, 1.0, 1.0, 1.0), extent=2.0).item(), 9 + 3 + 3 + 2)
    scales = torch.tensor([[1.0, 1.0], [math.exp(12.0), 1.0]])
    assert math.isclose(scale_regularizer(scales, extent=1.0, s_max=10.0).item(), (0 + 4.0) / 2, rel_tol=1e-5)
