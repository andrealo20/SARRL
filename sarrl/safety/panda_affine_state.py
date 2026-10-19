"""Shared parameter coordinates with outward independent residuals, for offline use."""

from dataclasses import dataclass

import numpy as np

from sarrl.safety.panda_backup import readonly
from sarrl.safety.panda_center_arithmetic import ArrayEnclosure, _export
from sarrl.safety.panda_validated_dynamics import balls, exact, interval, point


@dataclass(frozen=True)
class AffineState:
    """c + G epsilon + r, with the same eight epsilon coordinates in every queue slot."""

    center: np.ndarray
    generators: np.ndarray
    residual: ArrayEnclosure

    def __post_init__(self):
        c, g = np.asarray(self.center, float), np.asarray(self.generators, float)
        if (
            c.ndim != 1
            or g.shape != (len(c), 8)
            or self.residual.lower.shape != c.shape
            or not np.all(np.isfinite(c))
            or not np.all(np.isfinite(g))
            or np.any(self.residual.lower > 0)
            or np.any(self.residual.upper < 0)
        ):
            raise ValueError("affine coordinates must be finite and have eight shared generators")
        object.__setattr__(self, "center", readonly(c))
        object.__setattr__(self, "generators", readonly(g))

    @classmethod
    def constant(cls, values):
        values = np.asarray(values, float)
        return cls(values, np.zeros((len(values), 8)), ArrayEnclosure(values * 0, values * 0))

    def hull(self, backend):
        radius = np.sum(abs(point(self.generators, backend)), axis=1)
        return _export(
            point(self.center, backend)
            + radius * interval(-1, 1, backend)
            + balls(self.residual, backend)
        )

    def take(self, indices):
        return AffineState(
            self.center[indices],
            self.generators[indices],
            ArrayEnclosure(self.residual.lower[indices], self.residual.upper[indices]),
        )


def packed(center, generators, residual, backend):
    """Round representation coefficients, transferring every rounding error to r."""
    middle = np.array([x.mid() for x in residual], dtype=object)
    centered = center + middle
    c = np.array([float(x.mid()) for x in centered])
    g = np.array([float(x.mid()) for x in generators.flat]).reshape(generators.shape)
    remainder = residual - middle + centered - point(c, backend)
    remainder += np.sum(abs(generators - point(g, backend)), axis=1) * interval(-1, 1, backend)
    return AffineState(c, g, _export(abs(remainder) * interval(-1, 1, backend)))


def concatenate(forms):
    return AffineState(
        np.concatenate([x.center for x in forms]),
        np.concatenate([x.generators for x in forms]),
        ArrayEnclosure(
            np.concatenate([x.residual.lower for x in forms]),
            np.concatenate([x.residual.upper for x in forms]),
        ),
    )


def linear_combination(terms, backend, extra=None):
    c = np.full(len(terms[0][1].center), backend.arb(0), dtype=object)
    g = np.full((len(c), 8), backend.arb(0), dtype=object)
    r = c.copy()
    for weight, form in terms:
        weight = exact(weight, backend)
        c += weight * point(form.center, backend)
        g += weight * point(form.generators, backend)
        r += weight * balls(form.residual, backend)
    if extra is not None:
        r += extra
    return packed(c, g, r, backend)


def mean_value(form, value_at_center, jacobian, backend):
    """Mean-value inclusion on the convex hull domain, retaining shared first-order terms."""
    midpoint = np.array([x.mid() for x in jacobian.flat], dtype=object).reshape(jacobian.shape)
    generators = point(form.generators, backend)
    radius = np.sum(abs(generators), axis=1)
    residual = jacobian @ balls(form.residual, backend)
    residual += (jacobian - midpoint) @ (radius * interval(-1, 1, backend))
    return packed(value_at_center, midpoint @ generators, residual, backend)
