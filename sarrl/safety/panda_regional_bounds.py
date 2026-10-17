"""Center derivatives plus analytic Hessian remainders for Panda regions."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from sarrl.robots.panda import vector7
from sarrl.safety.panda_analytic_region import (
    SerialCoefficientBounds,
    analytic_panda_region,
    packaged_coefficient_bounds,
)
from sarrl.safety.panda_kinematic_coefficients import packaged_kinematic_coefficients


def local_mass_bounds(global_bounds, center_derivative, radius):
    """Apply the mean-value bound to the first derivative over the entire box."""
    center = np.asarray(center_derivative, dtype=float)
    radius = vector7(radius, "position radius")
    if center.shape != (7, 7, 7) or not np.all(np.isfinite(center)) or np.any(radius < 0):
        raise ValueError("invalid center mass derivative or position radius")
    local = np.abs(center) + global_bounds.mass_qq @ radius
    return SerialCoefficientBounds(
        np.minimum(global_bounds.mass_q, local), global_bounds.mass_qq, global_bounds.gravity_q
    )


def regional_panda_region(q_domain, v_domain, *, mass_remainder, bias_remainder, assumption):
    """Refine mass and Coriolis-velocity bounds; keep physical contracts explicit."""
    base = analytic_panda_region(
        q_domain,
        v_domain,
        mass_remainder=mass_remainder,
        bias_remainder=bias_remainder,
        assumption=assumption,
    )
    global0, globalp, damping = packaged_coefficient_bounds()
    center0, centerp, _ = packaged_kinematic_coefficients(base.q_center)
    nominal = local_mass_bounds(global0, center0.mass_q, q_domain.radius)
    payload = local_mass_bounds(globalp, centerp.mass_q, q_domain.radius)
    speed = np.maximum(np.abs(v_domain.lower), np.abs(v_domain.upper))
    return replace(
        base,
        mass0_q_derivative=nominal.mass_q,
        payload_mass_q_derivative=payload.mass_q,
        bias0_derivative=nominal.bias_derivative(speed, damping),
        payload_bias_derivative=payload.bias_derivative(speed, np.zeros(7)),
    )
