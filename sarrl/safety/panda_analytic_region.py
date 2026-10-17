"""Geometry-derived coefficient bounds for the packaged, unconstrained Panda model."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from sarrl.controllers.panda_adaptive import PandaRegressor
from sarrl.robots.panda import PAYLOAD_RADIUS, load_panda_model, vector7
from sarrl.safety.panda_backup import readonly
from sarrl.safety.panda_state_region import PandaStateRegion, VectorBox


@dataclass(frozen=True)
class SerialCoefficientBounds:
    """Absolute first/second mass derivatives and first gravity derivatives."""

    mass_q: np.ndarray
    mass_qq: np.ndarray
    gravity_q: np.ndarray

    def __post_init__(self):
        for name, shape in (
            ("mass_q", (7, 7, 7)),
            ("mass_qq", (7, 7, 7, 7)),
            ("gravity_q", (7, 7)),
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or np.any(value < 0) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite, nonnegative and have shape {shape}")
            object.__setattr__(self, name, readonly(value))

    def bias_derivative(self, velocity_limit, damping):
        """Bound derivatives of C(q,v)v + g(q) + diag(damping)v."""
        speed = vector7(velocity_limit, "velocity_limit")
        damping = vector7(damping, "damping")
        if np.any(speed < 0) or np.any(damping < 0):
            raise ValueError("speed and damping bounds must be nonnegative")
        gamma = np.empty((7, 7, 7))
        gamma_q = np.empty((7, 7, 7, 7))
        for i, j, k in product(range(7), repeat=3):
            gamma[i, j, k] = 0.5 * (
                self.mass_q[i, j, k] + self.mass_q[i, k, j] + self.mass_q[j, k, i]
            )
            gamma_q[i, j, k] = 0.5 * (
                self.mass_qq[i, j, k] + self.mass_qq[i, k, j] + self.mass_qq[j, k, i]
            )
        position = self.gravity_q + np.einsum("ijkl,j,k->il", gamma_q, speed, speed)
        velocity = np.einsum("ilk,k->il", gamma + gamma.swapaxes(1, 2), speed)
        velocity += np.diag(damping)
        return np.column_stack((position, velocity))


def serial_coefficient_bounds(lengths, masses, inertias, gravity, last_joints):
    """Apply the serial-chain product-rule bounds documented in the design note.

    lengths[b,i] bounds the COM distance downstream of revolute joint i for
    body b. last_joints[b] identifies its last ancestor joint (zero based).
    Each body has a contiguous prefix of ancestors.
    inertias contains COM rotational-inertia spectral norms, in kg m squared.
    This algebraic helper does not validate a robot's geometric provenance.
    """
    lengths = np.asarray(lengths, dtype=float)
    masses, inertias = np.asarray(masses, dtype=float), np.asarray(inertias, dtype=float)
    count = len(masses) if masses.ndim == 1 else 0
    if (
        count == 0
        or lengths.shape != (count, 7)
        or inertias.shape != (count,)
        or not np.all(np.isfinite(lengths))
        or np.any(lengths < 0)
        or not np.all(np.isfinite(masses))
        or np.any(masses < 0)
        or not np.all(np.isfinite(inertias))
        or np.any(inertias < 0)
        or not np.isfinite(gravity)
        or gravity < 0
    ):
        raise ValueError("invalid serial-chain geometry or inertial bounds")
    last_joints = np.asarray(last_joints)
    if (
        last_joints.shape != (count,)
        or not np.issubdtype(last_joints.dtype, np.integer)
        or np.any(last_joints < 0)
        or np.any(last_joints > 6)
    ):
        raise ValueError("last_joints must contain an ancestor index for each body")
    d1, d2, g1 = np.zeros((7, 7, 7)), np.zeros((7, 7, 7, 7)), np.zeros((7, 7))
    for body in range(count):
        ancestors = range(int(last_joints[body]) + 1)
        for i, k in product(ancestors, repeat=2):
            g1[i, k] += masses[body] * gravity * lengths[body, max(i, k)]
        for order, target in ((1, d1), (2, d2)):
            for indices in product(ancestors, repeat=2 + order):
                i, j, *derivatives = indices
                for mask in product((False, True), repeat=order):
                    left = [k for k, selected in zip(derivatives, mask, strict=True) if selected]
                    right = [
                        k for k, selected in zip(derivatives, mask, strict=True) if not selected
                    ]
                    linear = masses[body] * lengths[body, max([i, *left])]
                    linear *= lengths[body, max([j, *right])]
                    # Body-frame angular columns depend only on joints after their own.
                    angular = (
                        inertias[body]
                        if all(k > i for k in left) and all(k > j for k in right)
                        else 0.0
                    )
                    target[indices] += linear + angular
    return SerialCoefficientBounds(d1, d2, g1)


def packaged_coefficient_bounds():
    """Read a fresh hash-checked packaged model; perform no physical integration."""
    model = load_panda_model()
    bodies = model.jnt_bodyid
    attachment = model.body("attachment").id
    if (
        np.any(model.jnt_type != 3)
        or np.any(model.jnt_pos != 0)
        or not np.allclose(np.linalg.norm(model.jnt_axis, axis=1), 1, atol=1e-14, rtol=0)
        or np.any(model.body_parentid[bodies[1:]] != bodies[:-1])
        or model.body_parentid[attachment] != bodies[-1]
        or np.any(model.body_jntnum[bodies] != 1)
        or model.body_mass[attachment] != 0
        or np.any(model.body_inertia[attachment] != 0)
        or np.any(model.jnt_stiffness != 0)
        or np.any(model.dof_frictionloss != 0)
        or np.any(model.body_gravcomp != 0)
        or model.opt.density != 0
        or model.opt.viscosity != 0
        or model.ntendon != 0
        or model.nplugin != 0
    ):
        raise ValueError("packaged model does not match the serial hinge-chain derivation")
    lengths = np.zeros((7, 7))
    for b, body in enumerate(bodies):
        for i in range(b + 1):
            lengths[b, i] = np.linalg.norm(model.body_ipos[body]) + sum(
                np.linalg.norm(model.body_pos[bodies[k]]) for k in range(i + 1, b + 1)
            )
    payload_lengths = np.array(
        [
            np.linalg.norm(model.body_pos[attachment])
            + np.linalg.norm(model.body_ipos[attachment])
            + sum(np.linalg.norm(model.body_pos[bodies[k]]) for k in range(i + 1, 7))
            for i in range(7)
        ]
    )
    gravity = np.linalg.norm(model.opt.gravity)
    nominal = serial_coefficient_bounds(
        lengths,
        model.body_mass[bodies],
        np.max(model.body_inertia[bodies], axis=1),
        gravity,
        np.arange(7),
    )
    payload = serial_coefficient_bounds(
        payload_lengths[None, :],
        np.ones(1),
        np.array([0.4 * PAYLOAD_RADIUS**2]),
        gravity,
        np.array([6]),
    )
    return nominal, payload, model.dof_damping.copy()


def analytic_panda_region(q_domain, v_domain, *, mass_remainder, bias_remainder, assumption):
    """Build a conditional model region; force/runtime contracts remain caller inputs.

    Derivative bounds follow packaged link geometry at every q, with velocity
    bounded by v_domain. Centers use fresh nominal/unit-payload dynamics.
    This supplies no contact exclusion, sensor bound or integration-defect bound.
    """
    if not isinstance(q_domain, VectorBox) or not isinstance(v_domain, VectorBox):
        raise ValueError("q_domain and v_domain must be VectorBox instances")
    nominal, payload, damping = packaged_coefficient_bounds()
    q, v = q_domain.center, v_domain.center
    speed = np.maximum(np.abs(v_domain.lower), np.abs(v_domain.upper))
    regressor = PandaRegressor()
    m0 = regressor.nominal.mass_matrix(q)
    mp = regressor.unit_payload.mass_matrix(q) - m0
    h0 = regressor.nominal.inverse_dynamics(q, v, np.zeros(7))
    hp = regressor.unit_payload.inverse_dynamics(q, v, np.zeros(7)) - h0
    return PandaStateRegion(
        q_domain,
        v_domain,
        q,
        v,
        m0,
        mp,
        h0,
        hp,
        nominal.mass_q,
        payload.mass_q,
        nominal.bias_derivative(speed, damping),
        payload.bias_derivative(speed, np.zeros(7)),
        mass_remainder,
        bias_remainder,
        assumption,
    )
