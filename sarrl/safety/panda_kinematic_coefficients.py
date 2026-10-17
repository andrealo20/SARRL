"""Analytic center coefficients from differentiated serial rigid-body kinematics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sarrl.robots.panda import PAYLOAD_RADIUS, vector7
from sarrl.safety.panda_analytic_region import _checked_packaged_model
from sarrl.safety.panda_backup import readonly


def _rotation(quaternion):
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _local_transform(position, quaternion, joint, axis, angle):
    """Homogeneous transform and its first/second angle derivatives."""
    transform, first, second = np.eye(4), np.zeros((7, 4, 4)), np.zeros((7, 7, 4, 4))
    fixed = _rotation(quaternion)
    transform[:3, :3], transform[:3, 3] = fixed, position
    if joint >= 0:
        x, y, z = axis
        cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        square = cross @ cross
        sine, cosine = np.sin(angle), np.cos(angle)
        transform[:3, :3] = fixed @ (np.eye(3) + sine * cross + (1 - cosine) * square)
        first[joint, :3, :3] = fixed @ (cosine * cross + sine * square)
        second[joint, joint, :3, :3] = fixed @ (-sine * cross + cosine * square)
    return transform, first, second


@dataclass(frozen=True)
class KinematicCoefficients:
    mass: np.ndarray
    mass_q: np.ndarray
    gravity: np.ndarray
    gravity_q: np.ndarray

    def __post_init__(self):
        for name, shape in (
            ("mass", (7, 7)),
            ("mass_q", (7, 7, 7)),
            ("gravity", (7,)),
            ("gravity_q", (7, 7)),
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite and have shape {shape}")
            object.__setattr__(self, name, readonly(value))

    def bias(self, velocity, damping):
        velocity, damping = vector7(velocity, "velocity"), vector7(damping, "damping")
        # Gamma_ijk = (d_k M_ij + d_j M_ik - d_i M_jk)/2.
        gamma = 0.5 * (self.mass_q + self.mass_q.swapaxes(1, 2) - self.mass_q.transpose(2, 0, 1))
        return (
            self.gravity + np.einsum("ijk,j,k->i", gamma, velocity, velocity) + damping * velocity
        )


def packaged_kinematic_coefficients(q):
    """Return nominal/unit-payload coefficients without calling MuJoCo dynamics.

    The compiled constants are read from a fresh verified model. Transforms,
    Jacobians and derivatives are propagated algebraically, with no numerical
    differentiation or physical integration. Arithmetic is ordinary floating point.
    """
    q = vector7(q, "q")
    model = _checked_packaged_model()
    transforms = np.repeat(np.eye(4)[None, :, :], model.nbody, axis=0)
    first = np.zeros((model.nbody, 7, 4, 4))
    second = np.zeros((model.nbody, 7, 7, 4, 4))
    ancestors = np.zeros((model.nbody, 7), dtype=bool)
    for body in range(1, model.nbody):
        parent = model.body_parentid[body]
        joint = model.body_jntadr[body] if model.body_jntnum[body] else -1
        ancestors[body] = ancestors[parent]
        if joint >= 0:
            ancestors[body, joint] = True
        angle = q[joint] - model.qpos0[model.jnt_qposadr[joint]] if joint >= 0 else 0
        local, dlocal, ddlocal = _local_transform(
            model.body_pos[body],
            model.body_quat[body],
            joint,
            model.jnt_axis[joint] if joint >= 0 else np.zeros(3),
            angle,
        )
        transforms[body] = transforms[parent] @ local
        first[body] = first[parent] @ local + transforms[parent] @ dlocal
        second[body] = (
            second[parent] @ local
            + first[parent, :, None] @ dlocal[None, :]
            + first[parent, None, :] @ dlocal[:, None]
            + transforms[parent] @ ddlocal
        )
    axes = np.column_stack(
        [
            transforms[body, :3, :3] @ model.jnt_axis[joint]
            for joint, body in enumerate(model.jnt_bodyid)
        ]
    )
    axes_q = np.stack(
        [
            first[body, :, :3, :3] @ model.jnt_axis[joint]
            for joint, body in enumerate(model.jnt_bodyid)
        ],
        axis=2,
    )

    def accumulate(bodies, masses, inertias, armature):
        mass, derivative = np.diag(armature), np.zeros((7, 7, 7))
        gravity, gravity_q = np.zeros(7), np.zeros((7, 7))
        for body, body_mass, inertia in zip(bodies, masses, inertias, strict=True):
            center = np.array([*model.body_ipos[body], 1])
            linear = (first[body] @ center)[:, :3].T
            linear_q = (second[body] @ center)[:, :, :3].transpose(2, 0, 1)
            rotation = transforms[body, :3, :3]
            angular = rotation.T @ axes
            angular_q = np.swapaxes(first[body, :, :3, :3], 1, 2) @ axes
            angular_q += rotation.T @ axes_q
            outside = ~ancestors[body]
            angular[:, outside], angular_q[:, :, outside] = 0, 0
            mass += body_mass * (linear.T @ linear) + angular.T @ inertia @ angular
            gravity -= body_mass * (model.opt.gravity @ linear)
            gravity_q -= body_mass * np.einsum("a,aik->ik", model.opt.gravity, linear_q)
            for k in range(7):
                term = body_mass * (linear_q[:, :, k].T @ linear)
                term += angular_q[k].T @ inertia @ angular
                derivative[:, :, k] += term + term.T
        return KinematicCoefficients(mass, derivative, gravity, gravity_q)

    bodies = model.jnt_bodyid
    inertias = []
    for body in bodies:
        rotation = _rotation(model.body_iquat[body])
        inertias.append(rotation @ np.diag(model.body_inertia[body]) @ rotation.T)
    nominal = accumulate(bodies, model.body_mass[bodies], inertias, model.dof_armature)
    payload = accumulate(
        [model.body("attachment").id], [1], [np.eye(3) * 0.4 * PAYLOAD_RADIUS**2], np.zeros(7)
    )
    return nominal, payload, model.dof_damping.copy()
