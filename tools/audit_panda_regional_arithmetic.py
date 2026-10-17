"""Static outward regional coefficients and uniform errors to center references."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from sarrl.controllers.panda_adaptive import PandaRegressor
from sarrl.robots.panda import HOME
from sarrl.safety.panda_regional_arithmetic import enclose_panda_region
from sarrl.safety.panda_state_region import VectorBox


def audit():
    import flint
    import mujoco

    cases = []
    regressor = PandaRegressor()
    for radius in (0.0001, 0.001, 0.01):
        region = enclose_panda_region(
            VectorBox(HOME - radius, HOME + radius),
            VectorBox(-np.ones(7) * 0.01, np.ones(7) * 0.01),
        )
        q, v = region.center.q, region.center.velocity
        m0 = regressor.nominal.mass_matrix(q)
        mp = regressor.unit_payload.mass_matrix(q) - m0
        h0, rows = regressor.evaluate(q, v, np.zeros(7))
        result = {
            "q_lower": region.q_domain.lower.tolist(),
            "q_upper": region.q_domain.upper.tolist(),
            "v_lower": region.v_domain.lower.tolist(),
            "v_upper": region.v_domain.upper.tolist(),
            "q_center": q.tolist(),
            "v_center": v.tolist(),
            "q_radius_upper": region.q_radius_upper.tolist(),
            "v_radius_upper": region.v_radius_upper.tolist(),
            "requested_radius": radius,
        }
        for name, group, mass, bias in (
            ("nominal", region.nominal, m0, h0),
            ("payload", region.payload, mp, rows[:, 0]),
        ):
            result[name] = {
                "uniform_mass_error_to_center_reference": float(
                    np.max(group.mass.absolute_error(mass))
                ),
                "uniform_bias_error_to_center_reference": float(
                    np.max(group.bias.absolute_error(bias))
                ),
                "mass_derivative_upper_max": float(np.max(group.mass_q_upper)),
                "mass_hessian_upper_max": float(np.max(group.mass_qq_upper)),
                "gravity_derivative_upper_max": float(np.max(group.gravity_q_upper)),
            }
        cases.append(result)
    root = Path(__file__).resolve().parents[1]
    sources = (
        "sarrl/safety/panda_regional_arithmetic.py",
        "sarrl/safety/panda_center_arithmetic.py",
        "sarrl/safety/panda_analytic_region.py",
        "tools/audit_panda_regional_arithmetic.py",
    )
    return {
        "schema": 1,
        "scope": "Whole-box nominal/unit-payload coefficients of the normalized ideal model.",
        "precision_bits": region.center.precision_bits,
        "model_xml_sha256": region.center.model_xml_sha256,
        "compiled_constants_sha256": region.center.compiled_constants_sha256,
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name in sources
        },
        "source_hash_normalization": "CRLF to LF",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "python_flint": flint.__version__,
        },
        "inverse_or_prefix_verified": False,
        "physical_forces_or_runtime_verified": False,
        "cases": cases,
    }


def main():
    print(json.dumps(audit(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
