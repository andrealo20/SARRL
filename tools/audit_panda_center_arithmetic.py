"""Static center-coefficient arithmetic audit with explicit ideal-model semantics."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from sarrl.controllers.panda_adaptive import PandaRegressor
from sarrl.robots.panda import HOME
from sarrl.safety.panda_center_arithmetic import enclose_panda_center
from sarrl.safety.panda_kinematic_coefficients import packaged_kinematic_coefficients


def audit():
    import flint
    import mujoco

    cases = []
    regressor = PandaRegressor()
    for offset in (0.0, 0.2, -0.3):
        q, v = HOME + offset * np.sin(np.arange(7)), np.linspace(-0.2, 0.15, 7)
        enclosure = enclose_panda_center(q, v)
        nominal, payload, damping = packaged_kinematic_coefficients(q)
        m0 = regressor.nominal.mass_matrix(q)
        mp = regressor.unit_payload.mass_matrix(q) - m0
        h0, rows = regressor.evaluate(q, v, np.zeros(7))
        result = {"q": enclosure.q.tolist(), "velocity": enclosure.velocity.tolist()}
        for name, enclosed, ordinary, direct_mass, direct_bias, friction in (
            ("nominal", enclosure.nominal, nominal, m0, h0, damping),
            ("payload", enclosure.payload, payload, mp, rows[:, 0], np.zeros(7)),
        ):
            references = {
                "mass": ordinary.mass,
                "mass_q": ordinary.mass_q,
                "gravity": ordinary.gravity,
                "bias": ordinary.bias(v, friction),
            }
            result[name] = {
                "ordinary_max_absolute_error": {
                    key: float(np.max(getattr(enclosed, key).absolute_error(value)))
                    for key, value in references.items()
                },
                "mujoco_max_absolute_error": {
                    "mass": float(np.max(enclosed.mass.absolute_error(direct_mass))),
                    "bias": float(np.max(enclosed.bias.absolute_error(direct_bias))),
                },
            }
        cases.append(result)
    root = Path(__file__).resolve().parents[1]
    sources = (
        "sarrl/safety/panda_center_arithmetic.py",
        "sarrl/safety/panda_kinematic_coefficients.py",
        "sarrl/safety/panda_analytic_region.py",
        "tools/audit_panda_center_arithmetic.py",
    )
    return {
        "schema": 1,
        "scope": "Point coefficients of the exact-input normalized rigid-body model only.",
        "geometry": "Exact binary64 constants; normalized quaternions/axes; sphere inertia 2/5 r².",
        "precision_bits": 128,
        "model_xml_sha256": enclosure.model_xml_sha256,
        "compiled_constants_sha256": enclosure.compiled_constants_sha256,
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
        "production_integration": False,
        "regional_prefix_or_terminal_verified": False,
        "cases": cases,
    }


def main():
    print(json.dumps(audit(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
