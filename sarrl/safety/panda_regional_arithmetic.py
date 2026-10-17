"""Outward regional coefficient bounds for the declared normalized Panda model."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product

import numpy as np

from sarrl.robots.panda import PAYLOAD_RADIUS, model_provenance
from sarrl.safety.panda_analytic_region import _checked_packaged_model
from sarrl.safety.panda_backup import readonly
from sarrl.safety.panda_center_arithmetic import (
    _ARITHMETIC_LOCK,
    ArrayEnclosure,
    PandaCenterEnclosure,
    _compiled_signature,
    _export,
    _outward,
    enclose_panda_center,
)
from sarrl.safety.panda_state_region import VectorBox


def _center(domain):
    center = np.array(
        [
            float((Fraction(float(low)) + Fraction(float(high))) / 2)
            for low, high in zip(domain.lower, domain.upper, strict=True)
        ]
    )
    if (
        not np.all(np.isfinite(center))
        or np.any(center < domain.lower)
        or np.any(center > domain.upper)
    ):
        raise ValueError("regional center must lie inside the declared domain")
    return readonly(center)


def _radius(domain, center):
    """Cover displacement from the actual binary64 center, including midpoint rounding."""
    radius = np.empty(7)
    for i in range(7):
        c = Fraction(float(center[i]))
        distance = max(
            abs(Fraction(float(domain.lower[i])) - c), abs(Fraction(float(domain.upper[i])) - c)
        )
        radius[i] = _outward(distance, upper=True)
    return readonly(radius)


@dataclass(frozen=True)
class RegionalCoefficientEnclosure:
    mass: ArrayEnclosure
    bias: ArrayEnclosure
    mass_q_upper: np.ndarray
    mass_qq_upper: np.ndarray
    gravity_q_upper: np.ndarray
    bias_derivative_upper: np.ndarray

    def __post_init__(self):
        for name, shape in (
            ("mass_q_upper", (7, 7, 7)),
            ("mass_qq_upper", (7, 7, 7, 7)),
            ("gravity_q_upper", (7, 7)),
            ("bias_derivative_upper", (7, 14)),
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or not np.all(np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f"{name} must be a finite nonnegative array of shape {shape}")
            object.__setattr__(self, name, readonly(value))


@dataclass(frozen=True)
class PandaRegionEnclosure:
    center: PandaCenterEnclosure
    q_domain: VectorBox
    v_domain: VectorBox
    q_radius_upper: np.ndarray
    v_radius_upper: np.ndarray
    nominal: RegionalCoefficientEnclosure
    payload: RegionalCoefficientEnclosure

    def __post_init__(self):
        for name in ("q_radius_upper", "v_radius_upper"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (7,) or not np.all(np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f"{name} must be finite, nonnegative and have shape (7,)")
            object.__setattr__(self, name, readonly(value))


def _serial_bounds(lengths, masses, inertias, gravity, last_joints, arb):
    """Product-rule majorants with ball norms and no intermediate float reductions."""
    d1 = np.full((7, 7, 7), arb(0), dtype=object)
    d2 = np.full((7, 7, 7, 7), arb(0), dtype=object)
    g1 = np.full((7, 7), arb(0), dtype=object)
    for body, last in enumerate(last_joints):
        ancestors = range(last + 1)
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
                    angular = (
                        inertias[body]
                        if all(k > i for k in left) and all(k > j for k in right)
                        else arb(0)
                    )
                    target[indices] += linear + angular
    return d1, d2, g1


def _global_bounds(model, arb, exact):
    def norm(values):
        return sum(exact(x) ** 2 for x in values).sqrt()

    bodies = model.jnt_bodyid
    lengths = np.full((7, 7), arb(0), dtype=object)
    for b, body in enumerate(bodies):
        for i in range(b + 1):
            lengths[b, i] = norm(model.body_ipos[body]) + sum(
                (norm(model.body_pos[bodies[k]]) for k in range(i + 1, b + 1)), arb(0)
            )
    gravity = norm(model.opt.gravity)
    nominal = _serial_bounds(
        lengths,
        [exact(model.body_mass[b]) for b in bodies],
        [exact(max(model.body_inertia[b])) for b in bodies],
        gravity,
        list(range(7)),
        arb,
    )
    attachment = model.body("attachment").id
    payload_lengths = np.array(
        [
            [
                norm(model.body_pos[attachment])
                + norm(model.body_ipos[attachment])
                + sum((norm(model.body_pos[bodies[k]]) for k in range(i + 1, 7)), arb(0))
                for i in range(7)
            ]
        ],
        dtype=object,
    )
    payload = _serial_bounds(
        payload_lengths, [arb(1)], [(arb(2) / 5) * exact(PAYLOAD_RADIUS) ** 2], gravity, [6], arb
    )
    return nominal, payload


def _coefficient_region(center, global_bounds, q_radius, v_radius, speed, damping, arb, exact):
    d1, d2, g1 = [_export(values).upper for values in global_bounds]
    local = np.empty((7, 7, 7), dtype=object)
    for i, j, k in np.ndindex(local.shape):
        magnitude = max(abs(center.mass_q.lower[i, j, k]), abs(center.mass_q.upper[i, j, k]))
        candidate = exact(magnitude) + sum(
            (exact(d2[i, j, k, axis]) * exact(q_radius[axis]) for axis in range(7)), arb(0)
        )
        # Select between two proven upper bounds, never uncertain ball comparisons.
        bound = min(d1[i, j, k], _export(np.array([candidate], dtype=object)).upper[0])
        local[i, j, k] = exact(bound)
    local_upper = _export(local).upper
    gamma = np.empty((7, 7, 7), dtype=object)
    gamma_q = np.empty((7, 7, 7, 7), dtype=object)
    for i, j, k in np.ndindex(gamma.shape):
        gamma[i, j, k] = (local[i, j, k] + local[i, k, j] + local[j, k, i]) / 2
        for axis in range(7):
            gamma_q[i, j, k, axis] = (
                exact(d2[i, j, k, axis]) + exact(d2[i, k, j, axis]) + exact(d2[j, k, i, axis])
            ) / 2
    derivative = np.empty((7, 14), dtype=object)
    for i, axis in product(range(7), repeat=2):
        derivative[i, axis] = exact(g1[i, axis]) + sum(
            (
                gamma_q[i, j, k, axis] * exact(speed[j]) * exact(speed[k])
                for j, k in product(range(7), repeat=2)
            ),
            arb(0),
        )
        derivative[i, 7 + axis] = sum(
            ((gamma[i, axis, k] + gamma[i, k, axis]) * exact(speed[k]) for k in range(7)), arb(0)
        ) + (exact(damping[i]) if i == axis else arb(0))
    derivative_upper = _export(derivative).upper

    def expand(enclosure, radii):
        lower, upper = np.empty(enclosure.lower.shape), np.empty(enclosure.upper.shape)
        for index in np.ndindex(lower.shape):
            low = _export(np.array([exact(enclosure.lower[index]) - radii[index]], dtype=object))
            high = _export(np.array([exact(enclosure.upper[index]) + radii[index]], dtype=object))
            lower[index], upper[index] = low.lower[0], high.upper[0]
        return ArrayEnclosure(lower, upper)

    mass_radius = np.empty((7, 7), dtype=object)
    for i, j in np.ndindex(mass_radius.shape):
        mass_radius[i, j] = sum((local[i, j, k] * exact(q_radius[k]) for k in range(7)), arb(0))
    state_radius = np.concatenate((q_radius, v_radius))
    bias_radius = np.array(
        [
            sum((exact(derivative_upper[i, k]) * exact(state_radius[k]) for k in range(14)), arb(0))
            for i in range(7)
        ],
        dtype=object,
    )
    return RegionalCoefficientEnclosure(
        expand(center.mass, mass_radius),
        expand(center.bias, bias_radius),
        local_upper,
        d2,
        g1,
        derivative_upper,
    )


def enclose_panda_region(q_domain, v_domain, *, precision_bits=128):
    """Enclose ideal-model coefficients over a full box; perform no inverse or propagation.

    The model and precision-context contracts match enclose_panda_center.
    Physical remainders and uncertain additional damping are outside this result.
    """
    if not isinstance(q_domain, VectorBox) or not isinstance(v_domain, VectorBox):
        raise ValueError("regional domains must be VectorBox instances")
    center = enclose_panda_center(
        _center(q_domain), _center(v_domain), precision_bits=precision_bits
    )
    model = _checked_packaged_model()
    if _compiled_signature(model) != center.compiled_constants_sha256:
        raise ValueError("regional model constants differ from the enclosed center")
    q_radius, v_radius = _radius(q_domain, center.q), _radius(v_domain, center.velocity)
    speed = np.maximum(np.abs(v_domain.lower), np.abs(v_domain.upper))
    import flint

    def exact(value):
        numerator, denominator = float(value).as_integer_ratio()
        return flint.arb(flint.fmpq(numerator, denominator))

    with _ARITHMETIC_LOCK, flint.ctx.workprec(precision_bits):
        global0, globalp = _global_bounds(model, flint.arb, exact)
        nominal = _coefficient_region(
            center.nominal, global0, q_radius, v_radius, speed, model.dof_damping, flint.arb, exact
        )
        payload = _coefficient_region(
            center.payload, globalp, q_radius, v_radius, speed, np.zeros(7), flint.arb, exact
        )
    if model_provenance()["files"]["panda.xml"] != center.model_xml_sha256:
        raise ValueError("model source changed during regional evaluation")
    return PandaRegionEnclosure(center, q_domain, v_domain, q_radius, v_radius, nominal, payload)
