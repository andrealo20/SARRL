"""Pointwise parameter-dependent backup calculations, without a delay witness."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from sarrl.robots.panda import vector7
from sarrl.safety.panda_backup import readonly


def parameters8(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (8,) or not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f"{name} must contain eight finite nonnegative parameters")
    return result


def symmetric7(value, name):
    result = np.asarray(value, dtype=float)
    if (
        result.shape != (7, 7)
        or not np.all(np.isfinite(result))
        or not np.allclose(result, result.T, rtol=1e-12, atol=1e-12)
    ):
        raise ValueError(f"{name} must be finite and symmetric with shape (7, 7)")
    return result


@dataclass(frozen=True)
class PandaParameterBox:
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        for name in ("lower", "upper"):
            object.__setattr__(self, name, readonly(parameters8(getattr(self, name), name)))
        if np.any(self.lower > self.upper):
            raise ValueError("parameter box lower bounds exceed upper bounds")


@dataclass(frozen=True)
class PandaStructuredContext:
    """Immutable anchor/estimate; coefficients passed to point() must match its q/v."""

    anchor: np.ndarray
    parameters: np.ndarray
    kp: float = 25.0
    kd: float = 10.0

    def __post_init__(self):
        object.__setattr__(self, "anchor", readonly(vector7(self.anchor, "anchor")))
        object.__setattr__(self, "parameters", readonly(parameters8(self.parameters, "parameters")))
        if not np.all(np.isfinite([self.kp, self.kd])) or min(self.kp, self.kd) <= 0:
            raise ValueError("backup gains must be finite and positive")

    def point(self, q, v, mass0, payload_mass, bias0, payload_bias):
        """Raw computed torque, using current-state coefficients and retained estimates.

        This algebra does not verify coefficient provenance, clipping, a queue,
        measurement uncertainty, terminal entry or any physical trajectory.
        """
        q, v = vector7(q, "q"), vector7(v, "v")
        mass0, payload_mass = symmetric7(mass0, "mass0"), symmetric7(payload_mass, "payload_mass")
        mass = mass0 + self.parameters[0] * payload_mass
        np.linalg.cholesky(mass)
        desired = -self.kp * (q - self.anchor) - self.kd * v
        bias = vector7(bias0, "bias0") + self.parameters[0] * vector7(payload_bias, "payload_bias")
        torque = mass @ desired + bias + self.parameters[1:] * v
        if not np.all(np.isfinite(torque)):
            raise ValueError("structured torque overflow")
        return desired, torque


def point_acceleration_interval(
    mass0,
    payload_mass,
    payload_bias,
    velocity,
    desired,
    estimate,
    box,
    *,
    input_difference,
    mass_cells=16,
):
    """Analytical parameter enclosure at ONE exact state and a declared input difference.

    Mass is continuous within every cell, not a grid of assumed extrema.
    Floating-point eigensolves/guards are not interval-certified numerics.
    input_difference is delivered torque minus the same-state raw law. It
    must be supplied explicitly; this function does not reconstruct a FIFO.
    """
    if type(mass_cells) is not int or not 1 <= mass_cells <= 256:
        raise ValueError("mass cell count must be an integer from one through 256")
    mass0, payload_mass = symmetric7(mass0, "mass0"), symmetric7(payload_mass, "payload_mass")
    velocity, desired = vector7(velocity, "velocity"), vector7(desired, "desired")
    payload_bias = vector7(payload_bias, "payload_bias")
    estimate = parameters8(estimate, "estimate")
    input_difference = vector7(input_difference, "input_difference")
    center = 0.5 * (box.lower[0] + box.upper[0])
    center_mass = mass0 + center * payload_mass
    eigenvalues, basis = eigh(payload_mass, center_mass)
    endpoints = 1 + (np.array([box.lower[0], box.upper[0]])[:, None] - center) * eigenvalues
    if np.any(endpoints <= 0) or not np.all(np.isfinite(endpoints)):
        raise ValueError("mass family is not positive definite across the parameter box")
    payload_column = payload_mass @ desired + payload_bias
    payload_modes = basis.T @ payload_column
    friction_mid = 0.5 * (box.lower[1:] + box.upper[1:])
    friction_half = 0.5 * (box.upper[1:] - box.lower[1:])
    friction_modes = basis.T @ (velocity * (estimate[1:] - friction_mid) + input_difference)
    inverse_products = basis[:, None, :] * basis[None, :, :]
    low, high = np.full(7, np.inf), np.full(7, -np.inf)
    edges = np.linspace(box.lower[0], box.upper[0], mass_cells + 1)
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        masses = np.array([left, right])
        denominator = 1 + (masses[:, None] - center) * eigenvalues
        numerator = (estimate[0] - masses[:, None]) * payload_modes + friction_modes
        # Each modal ratio is affine over affine with positive denominator.
        modal = numerator / denominator
        a, b = basis * np.min(modal, axis=0), basis * np.max(modal, axis=0)
        center_low, center_high = np.sum(np.minimum(a, b), axis=1), np.sum(np.maximum(a, b), axis=1)
        reciprocal = 1 / denominator
        a = inverse_products * np.min(reciprocal, axis=0)
        b = inverse_products * np.max(reciprocal, axis=0)
        inverse_low = np.sum(np.minimum(a, b), axis=2)
        inverse_high = np.sum(np.maximum(a, b), axis=2)
        radius = np.maximum(np.abs(inverse_low), np.abs(inverse_high)) @ (
            np.abs(velocity) * friction_half
        )
        low, high = np.minimum(low, center_low - radius), np.maximum(high, center_high + radius)
    low, high = desired + low, desired + high
    if not np.all(np.isfinite([low, high])):
        raise ValueError("structured acceleration enclosure overflow")
    guard = 1e-11 * (1 + np.maximum(np.abs(low), np.abs(high)))
    return low - guard, high + guard
