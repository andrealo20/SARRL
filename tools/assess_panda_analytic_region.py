"""Report conservatism of geometry-derived Panda regions without stepping physics."""

from __future__ import annotations

import json

import numpy as np
from scipy.linalg import eigh

from sarrl.robots.panda import HOME, model_provenance
from sarrl.safety.panda_analytic_region import analytic_panda_region
from sarrl.safety.panda_state_region import EnclosureUnavailable, VectorBox
from sarrl.safety.panda_structured_backup import PandaParameterBox


def assess():
    parameters = PandaParameterBox(np.array([0.5, *[0.5] * 7]), np.array([2.5, *[3.0] * 7]))
    rows = []
    for radius in (1e-8, 1e-6, 1e-4, 1e-3, 1e-2):
        region = analytic_panda_region(
            VectorBox(HOME - radius, HOME + radius),
            VectorBox(-np.ones(7) * 0.01, np.ones(7) * 0.01),
            mass_remainder=np.zeros((7, 7)),
            bias_remainder=np.zeros(7),
            assumption="Ideal unconstrained continuous-model diagnostic only.",
        )
        m0, mp, *_ = region._variation(region.q_domain, region.v_domain)
        eigenvalues, basis = eigh(region.payload_mass, region.mass0 + 1.5 * region.payload_mass)
        reciprocal = np.max(1 / (1 + np.array([-1, 1])[:, None] * eigenvalues), axis=0)
        inverse_bound = np.sum(np.abs(basis[:, None, :] * basis[None, :, :]) * reciprocal, axis=2)
        row = {
            "position_radius_rad": radius,
            "inverse_comparison_row_norm": float(
                np.max(np.sum(inverse_bound @ (m0 + parameters.upper[0] * mp), axis=1))
            ),
        }
        torque = VectorBox.point(region.bias0 + 1.5 * region.payload_bias)
        try:
            acceleration = region.acceleration(parameters, region.q_domain, region.v_domain, torque)
            row["acceleration_max_width_rad_s2"] = float(
                np.max(acceleration.upper - acceleration.lower)
            )
        except EnclosureUnavailable as failure:
            row["reason"] = str(failure)
        rows.append(row)
    return {
        "scope": "Static ideal-model calculation; no physical integration or terminal claim.",
        "model_xml_sha256": model_provenance()["files"]["panda.xml"],
        "position_center_rad": HOME.tolist(),
        "velocity_center_rad_s": [0.0] * 7,
        "velocity_radius_rad_s": 0.01,
        "physical_parameter_lower": parameters.lower.tolist(),
        "physical_parameter_upper": parameters.upper.tolist(),
        "force_remainders": "Zero only for this ideal unconstrained model diagnostic.",
        "sensor_and_runtime_contract": None,
        "rows": rows,
    }


def main():
    print(json.dumps(assess(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
