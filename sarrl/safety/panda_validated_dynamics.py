"""Offline ball arithmetic for the normalized model and continuous parameter boxes."""

from __future__ import annotations

from contextlib import contextmanager
from fractions import Fraction

import numpy as np

from sarrl.safety.panda_backup import readonly
from sarrl.safety.panda_center_arithmetic import _ARITHMETIC_LOCK, _export
from sarrl.safety.panda_regional_arithmetic import enclose_panda_region
from sarrl.safety.panda_state_region import EnclosureUnavailable, VectorBox
from sarrl.safety.panda_structured_backup import PandaParameterBox


@contextmanager
def arithmetic(precision_bits):
    if type(precision_bits) is not int or not 80 <= precision_bits <= 1024:
        raise ValueError("precision_bits must be an integer from 80 through 1024")
    import flint

    if flint.__version__ != "0.9.0":
        raise RuntimeError("Validated dynamics requires python-flint 0.9.0")
    with _ARITHMETIC_LOCK, flint.ctx.workprec(precision_bits):
        yield flint


def exact(value, backend):
    value = value if isinstance(value, Fraction) else Fraction(float(value))
    return backend.arb(backend.fmpq(value.numerator, value.denominator))


def interval(low, high, backend):
    return exact(low, backend).union(exact(high, backend))


def balls(box, backend):
    return np.array(
        [interval(box.lower[i], box.upper[i], backend) for i in np.ndindex(box.lower.shape)],
        dtype=object,
    ).reshape(box.lower.shape)


def point(values, backend):
    values = np.asarray(values)
    return np.array([exact(x, backend) for x in values.flat], dtype=object).reshape(values.shape)


def box(values):
    result = _export(values)
    return VectorBox(result.lower, result.upper)


def inflate(domain, width, *, precision_bits=128):
    width = np.asarray(width, dtype=float)
    if width.shape != (7,) or not np.all(np.isfinite(width)) or np.any(width < 0):
        raise ValueError("inflation must contain seven finite nonnegative widths")
    with arithmetic(precision_bits) as backend:
        return box(
            np.array(
                [
                    interval(-width[i], width[i], backend)
                    + interval(domain.lower[i], domain.upper[i], backend)
                    for i in range(7)
                ],
                dtype=object,
            )
        )


def step_images(q0, v0, acceleration, dt, errors, *, precision_bits=128):
    """Outward endpoint and all-partial-time images; dt is an exact binary64 input."""
    with arithmetic(precision_bits) as backend:
        q, v, a = [balls(value, backend) for value in (q0, v0, acceleration)]
        t = exact(dt, backend)
        span = interval(0, dt, backend)
        half_square = interval(Fraction(0), Fraction(float(dt)) ** 2 / 2, backend)
        eq = balls(VectorBox(-errors.position_defect, errors.position_defect), backend)
        ev = balls(VectorBox(-errors.velocity_defect, errors.velocity_defect), backend)
        return (
            box(q + span * v + half_square * a + eq),
            box(v + span * a + ev),
            box(q + t * v + (t * t / 2) * a + eq),
            box(v + t * a + ev),
        )


def mass_cells(parameters, count):
    if not isinstance(parameters, PandaParameterBox):
        raise ValueError("parameters must be a PandaParameterBox")
    if type(count) is not int or not 1 <= count <= 256:
        raise ValueError("mass cell count must be an integer from one through 256")
    low, high = Fraction(float(parameters.lower[0])), Fraction(float(parameters.upper[0]))
    # Exact rational endpoints cover the continuous interval, including every boundary.
    return tuple(
        (low + (high - low) * i / count, low + (high - low) * (i + 1) / count) for i in range(count)
    )


def solve_acceleration(
    coefficients,
    parameters,
    velocity,
    torque,
    *,
    cells=16,
    precision_bits=128,
    mass_remainder=None,
    bias_remainder=None,
):
    """Enclose held-input acceleration; cells share one mass in matrix and force terms.

    Correlations inside a cell can be lost to ball operations. Every cell is
    continuous, and any unverified solve rejects the entire query. No approximate
    eigensolve or approximate FLINT solver supplies an acceptance decision.
    """
    partitions = mass_cells(parameters, cells)
    if hasattr(coefficients, "v_domain") and not coefficients.v_domain.contains(velocity):
        raise EnclosureUnavailable("coefficient_velocity_domain_not_covered")
    mass_remainder = np.zeros((7, 7)) if mass_remainder is None else mass_remainder
    bias_remainder = np.zeros(7) if bias_remainder is None else bias_remainder
    for value, shape in ((mass_remainder, (7, 7)), (bias_remainder, (7,))):
        value = np.asarray(value)
        if value.shape != shape or not np.all(np.isfinite(value)) or np.any(value < 0):
            raise ValueError("physical remainder must be finite, nonnegative and match its shape")
    with arithmetic(precision_bits) as backend:
        m0, mp = [
            balls(group.mass, backend) for group in (coefficients.nominal, coefficients.payload)
        ]
        h0, hp = [
            balls(group.bias, backend) for group in (coefficients.nominal, coefficients.payload)
        ]
        v, u = balls(velocity, backend), balls(torque, backend)
        friction = np.array(
            [
                interval(lo, hi, backend)
                for lo, hi in zip(parameters.lower[1:], parameters.upper[1:], strict=True)
            ]
        )
        em = np.array(
            [interval(-x, x, backend) for x in np.asarray(mass_remainder).flat], dtype=object
        ).reshape(7, 7)
        eh = np.array([interval(-x, x, backend) for x in bias_remainder], dtype=object)
        answers = []
        for low, high in partitions:
            mass = interval(low, high, backend)
            matrix = m0 + mass * mp + em
            rhs = u - h0 - mass * hp - friction * v + eh
            try:
                solution = backend.arb_mat(matrix.tolist()).solve(
                    backend.arb_mat([[x] for x in rhs]), algorithm="precond"
                )
                enclosed = box(np.array([solution[i, 0] for i in range(7)], dtype=object))
            except (ZeroDivisionError, ValueError) as failure:
                raise EnclosureUnavailable("validated_mass_solve_unavailable") from failure
            answers.append(enclosed)
        return VectorBox(
            np.min([x.lower for x in answers], axis=0), np.max([x.upper for x in answers], axis=0)
        )


class PandaValidatedDynamics:
    """Recenter ideal-model coefficients on each query, outside the control runtime."""

    def __init__(
        self,
        q_domain,
        v_domain,
        *,
        mass_remainder,
        bias_remainder,
        assumption,
        cells=16,
        precision_bits=128,
    ):
        if not isinstance(q_domain, VectorBox) or not isinstance(v_domain, VectorBox):
            raise ValueError("dynamics domains must be VectorBox instances")
        if not isinstance(assumption, str) or not assumption.strip():
            raise ValueError("an explicit physical remainder assumption is required")
        mass_cells(PandaParameterBox(np.zeros(8), np.zeros(8)), cells)
        with arithmetic(precision_bits):
            pass
        self.q_domain, self.v_domain = q_domain, v_domain
        self.cells, self.precision_bits, self.assumption = cells, precision_bits, assumption
        for name, value, shape in (
            ("mass_remainder", mass_remainder, (7, 7)),
            ("bias_remainder", bias_remainder, (7,)),
        ):
            value = np.asarray(value, dtype=float)
            if value.shape != shape or not np.all(np.isfinite(value)) or np.any(value < 0):
                raise ValueError("physical remainder must be finite, nonnegative and shaped")
            setattr(self, name, readonly(value))

    def coefficients(self, q, v):
        if not self.q_domain.contains(q) or not self.v_domain.contains(v):
            raise EnclosureUnavailable("state_region_not_covered")
        return enclose_panda_region(q, v, precision_bits=self.precision_bits)

    def acceleration(self, parameters, q, v, torque):
        return solve_acceleration(
            self.coefficients(q, v),
            parameters,
            v,
            torque,
            cells=self.cells,
            precision_bits=self.precision_bits,
            mass_remainder=self.mass_remainder,
            bias_remainder=self.bias_remainder,
        )

    def torque(self, context, q, v, command_error):
        coefficients = self.coefficients(q, v)
        with arithmetic(self.precision_bits) as backend:
            position, velocity = balls(q, backend), balls(v, backend)
            desired = -exact(context.kp, backend) * (position - point(context.anchor, backend))
            desired -= exact(context.kd, backend) * velocity
            estimate = point(context.parameters, backend)
            mass = balls(coefficients.nominal.mass, backend)
            mass += estimate[0] * balls(coefficients.payload.mass, backend)
            force = balls(coefficients.nominal.bias, backend)
            force += estimate[0] * balls(coefficients.payload.bias, backend)
            force += estimate[1:] * velocity
            raw = box(mass @ desired + force)
        return inflate(raw, command_error, precision_bits=self.precision_bits)
