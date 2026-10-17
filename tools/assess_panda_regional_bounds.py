"""Compare global and center/Hessian Panda regions without physical integration."""

from __future__ import annotations

import json

import numpy as np
from scipy.linalg import eigh

from sarrl.safety.panda_regional_bounds import regional_panda_region
from sarrl.safety.panda_state_region import EnclosureUnavailable, VectorBox
from sarrl.safety.panda_structured_backup import PandaParameterBox
from tools.assess_panda_analytic_region import assess as assess_global


def assess():
    baseline = assess_global()
    q = np.array(baseline["position_center_rad"])
    v = np.array(baseline["velocity_center_rad_s"])
    vr = baseline["velocity_radius_rad_s"]
    parameters = PandaParameterBox(
        baseline["physical_parameter_lower"], baseline["physical_parameter_upper"]
    )
    midpoint = 0.5 * (parameters.lower[0] + parameters.upper[0])
    offsets = parameters.lower[0] - midpoint, parameters.upper[0] - midpoint
    rows = []
    for original in baseline["rows"]:
        radius = original["position_radius_rad"]
        region = regional_panda_region(
            VectorBox(q - radius, q + radius),
            VectorBox(v - vr, v + vr),
            mass_remainder=np.zeros((7, 7)),
            bias_remainder=np.zeros(7),
            assumption="Ideal unconstrained continuous-model diagnostic only.",
        )
        m0, mp, *_ = region._variation(region.q_domain, region.v_domain)
        eigenvalues, basis = eigh(
            region.payload_mass, region.mass0 + midpoint * region.payload_mass
        )
        reciprocal = np.max(1 / (1 + np.array(offsets)[:, None] * eigenvalues), axis=0)
        inverse = np.sum(np.abs(basis[:, None, :] * basis[None, :, :]) * reciprocal, axis=2)
        result = {
            "position_radius_rad": radius,
            "global": {
                key: value for key, value in original.items() if key != "position_radius_rad"
            },
            "regional": {
                "inverse_comparison_row_norm": float(
                    np.max(np.sum(inverse @ (m0 + parameters.upper[0] * mp), axis=1))
                )
            },
        }
        torque = VectorBox.point(region.bias0 + midpoint * region.payload_bias)
        try:
            acceleration = region.acceleration(parameters, region.q_domain, region.v_domain, torque)
            result["regional"]["acceleration_max_width_rad_s2"] = float(
                np.max(acceleration.upper - acceleration.lower)
            )
        except EnclosureUnavailable as failure:
            result["regional"]["reason"] = str(failure)
        rows.append(result)
    return {
        **baseline,
        "comparison": "Global versus center/Hessian mass derivatives.",
        "rows": rows,
    }


def main():
    print(json.dumps(assess(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
