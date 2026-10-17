"""Conditional state-region bounds for the structured Panda dynamics model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from sarrl.robots.panda import vector7
from sarrl.safety.panda_backup import readonly
from sarrl.safety.panda_structured_backup import point_acceleration_interval, symmetric7


class EnclosureUnavailable(ValueError):
    """The supplied model contract cannot enclose this query."""


@dataclass(frozen=True)
class VectorBox:
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        for name in ("lower", "upper"):
            object.__setattr__(self, name, readonly(vector7(getattr(self, name), name)))
        if np.any(self.lower > self.upper):
            raise ValueError("box lower bounds exceed upper bounds")

    @classmethod
    def point(cls, value):
        return cls(value, value)

    @property
    def center(self):
        return 0.5 * self.lower + 0.5 * self.upper

    @property
    def radius(self):
        return 0.5 * self.upper - 0.5 * self.lower

    def contains(self, other, *, strict=False):
        if strict:
            return bool(np.all(other.lower > self.lower) and np.all(other.upper < self.upper))
        return bool(np.all(other.lower >= self.lower) and np.all(other.upper <= self.upper))

    def inflate(self, width):
        width = vector7(width, "inflation")
        if np.any(width < 0):
            raise ValueError("box inflation must be nonnegative")
        return VectorBox(self.lower - width, self.upper + width)


def padded_box(lower, upper):
    """Floating-point precaution, not a validated interval-arithmetic implementation."""
    scale = 1 + np.maximum(np.abs(lower), np.abs(upper))
    return VectorBox(np.asarray(lower) - 1e-11 * scale, np.asarray(upper) + 1e-11 * scale)


def matrix_vector_box(matrix, radius, vector):
    low, high = matrix - radius, matrix + radius
    products = np.array(
        [low * vector.lower, low * vector.upper, high * vector.lower, high * vector.upper]
    )
    return np.sum(np.min(products, axis=0), axis=1), np.sum(np.max(products, axis=0), axis=1)


@dataclass(frozen=True)
class PandaStateRegion:
    """Supplied derivative/remainder assumptions over a convex q/v domain.

    This class never derives these assumptions from samples or validates
    their physical provenance. The separate analytic Panda provider derives
    coefficient bounds; physical force and runtime contracts remain inputs.
    Mass derivative arrays have shape (7,7,7); bias derivatives (7,14).
    Remainders cover physical model/force errors, not controller arithmetic.
    """

    q_domain: VectorBox
    v_domain: VectorBox
    q_center: np.ndarray
    v_center: np.ndarray
    mass0: np.ndarray
    payload_mass: np.ndarray
    bias0: np.ndarray
    payload_bias: np.ndarray
    mass0_q_derivative: np.ndarray
    payload_mass_q_derivative: np.ndarray
    bias0_derivative: np.ndarray
    payload_bias_derivative: np.ndarray
    mass_remainder: np.ndarray
    bias_remainder: np.ndarray
    assumption: str

    def __post_init__(self):
        for name in ("q_center", "v_center", "bias0", "payload_bias"):
            object.__setattr__(self, name, readonly(vector7(getattr(self, name), name)))
        for name in ("mass0", "payload_mass"):
            object.__setattr__(self, name, readonly(symmetric7(getattr(self, name), name)))
        shapes = {
            "mass0_q_derivative": (7, 7, 7),
            "payload_mass_q_derivative": (7, 7, 7),
            "bias0_derivative": (7, 14),
            "payload_bias_derivative": (7, 14),
            "mass_remainder": (7, 7),
            "bias_remainder": (7,),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or not np.all(np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f"{name} must be a finite nonnegative array of shape {shape}")
            object.__setattr__(self, name, readonly(value))
        if not self.q_domain.contains(VectorBox.point(self.q_center)) or not self.v_domain.contains(
            VectorBox.point(self.v_center)
        ):
            raise ValueError("coefficient center must belong to the declared state region")
        if not isinstance(self.assumption, str) or not self.assumption.strip():
            raise ValueError("state-region assumptions must be declared")

    def _variation(self, q, v):
        if not self.q_domain.contains(q) or not self.v_domain.contains(v):
            raise EnclosureUnavailable("state_region_not_covered")
        dq = np.maximum(np.abs(q.lower - self.q_center), np.abs(q.upper - self.q_center))
        dv = np.maximum(np.abs(v.lower - self.v_center), np.abs(v.upper - self.v_center))
        state = np.concatenate((dq, dv))
        return (
            self.mass0_q_derivative @ dq,
            self.payload_mass_q_derivative @ dq,
            self.bias0_derivative @ state,
            self.payload_bias_derivative @ state,
            dv,
        )

    def torque(self, context, q, v, command_error):
        command_error = vector7(command_error, "command_error")
        if np.any(command_error < 0):
            raise ValueError("command error must be nonnegative")
        m0, mp, h0, hp, _ = self._variation(q, v)
        theta = context.parameters
        desired = VectorBox(
            -context.kp * (q.upper - context.anchor) - context.kd * v.upper,
            -context.kp * (q.lower - context.anchor) - context.kd * v.lower,
        )
        low, high = matrix_vector_box(
            self.mass0 + theta[0] * self.payload_mass, m0 + theta[0] * mp, desired
        )
        bias = self.bias0 + theta[0] * self.payload_bias
        radius = h0 + theta[0] * hp
        return padded_box(
            low + bias - radius + theta[1:] * v.lower - command_error,
            high + bias + radius + theta[1:] * v.upper + command_error,
        )

    def acceleration(self, parameters, q, v, torque):
        m0, mp, h0, hp, dv = self._variation(q, v)
        mass_radius = m0 + parameters.upper[0] * mp + self.mass_remainder
        bias_radius = (
            h0 + parameters.upper[0] * hp + parameters.upper[1:] * dv + self.bias_remainder
        )
        center_mass = (
            self.mass0 + 0.5 * (parameters.lower[0] + parameters.upper[0]) * self.payload_mass
        )
        try:
            eigenvalues, basis = eigh(self.payload_mass, center_mass)
        except np.linalg.LinAlgError as failure:
            raise EnclosureUnavailable("center_mass_family_unverified") from failure
        offsets = np.array([parameters.lower[0], parameters.upper[0]])
        offsets -= 0.5 * (parameters.lower[0] + parameters.upper[0])
        denominators = 1 + offsets[:, None] * eigenvalues
        if np.any(denominators <= 0) or not np.all(np.isfinite(denominators)):
            raise EnclosureUnavailable("center_mass_family_unverified")
        reciprocal = np.max(1 / denominators, axis=0)
        inverse_bound = np.sum(np.abs(basis[:, None, :] * basis[None, :, :]) * reciprocal, axis=2)
        comparison = inverse_bound @ mass_radius
        if np.max(np.sum(comparison, axis=1)) >= 1 - 1e-12:
            raise EnclosureUnavailable("mass_variation_unverified")
        inverse = np.linalg.solve(np.eye(7) - comparison, inverse_bound)
        residual = (np.eye(7) - comparison) @ inverse - inverse_bound
        if (
            not np.all(np.isfinite(inverse))
            or np.any(inverse < -1e-12)
            or np.linalg.norm(residual) > 1e-10 * max(1, np.linalg.norm(inverse))
        ):
            raise EnclosureUnavailable("inverse_bound_numerics")
        inverse = np.maximum(inverse, 0) * (1 + 1e-11) + 1e-12
        # Zero estimate/desired is an algebraic reference: its point law is bias0.
        low, high = point_acceleration_interval(
            self.mass0,
            self.payload_mass,
            self.payload_bias,
            self.v_center,
            np.zeros(7),
            np.zeros(8),
            parameters,
            input_difference=torque.center - self.bias0,
        )
        point_magnitude = np.maximum(np.abs(low), np.abs(high))
        radius = inverse @ (torque.radius + bias_radius + mass_radius @ point_magnitude)
        return padded_box(low - radius, high + radius)
