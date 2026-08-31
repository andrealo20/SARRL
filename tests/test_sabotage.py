"""Tests designed to fail when characteristic implementation bugs are reintroduced."""

import numpy as np
import torch

from sarrl.dynamics import PlanarArm
from sarrl.rl.networks import SquashedGaussianActor


def test_sabotage_wrong_coriolis_sign_breaks_skew_identity():
    arm = PlanarArm()
    q = np.array([0.4, 1.0])
    qd = np.array([0.7, -0.9])
    h = 1e-6
    mdot = np.zeros((2, 2))
    for k in range(2):
        dq = np.zeros(2)
        dq[k] = h
        mdot += ((arm.mass_matrix(q + dq) - arm.mass_matrix(q - dq)) / (2 * h)) * qd[k]
    wrong_c = -arm.coriolis_matrix(q, qd)
    residual = mdot - 2.0 * wrong_c
    assert np.linalg.norm(residual + residual.T) > 1e-3


def test_sabotage_omitting_tanh_jacobian_changes_log_probability():
    torch.manual_seed(17)
    actor = SquashedGaussianActor(3, 2, hidden=(16, 16))
    obs = torch.randn(64, 3)
    action, correct, _ = actor.sample(obs)
    mean, log_std = actor.distribution_params(obs)
    raw = torch.atanh(torch.clamp(action, -1 + 1e-6, 1 - 1e-6))
    wrong = torch.distributions.Normal(mean, log_std.exp()).log_prob(raw).sum(-1, keepdim=True)
    assert torch.max(torch.abs(correct - wrong)).item() > 0.05
