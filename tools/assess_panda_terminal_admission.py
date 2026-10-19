"""Local stability, delayed feedback delivery and model-binding admission checks."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sarrl.robots.panda import model_provenance
from sarrl.safety.panda_friction_compensation import bounded_friction_context
from sarrl.safety.panda_local_coefficients import PandaSnapshotDynamics
from sarrl.safety.panda_parameter_prefix import audit_parameter_prefix, verify_parameter_prefix
from sarrl.safety.panda_sampled_stability import enclose_sampled_stability
from sarrl.safety.panda_state_region import VectorBox
from sarrl.safety.panda_structured_backup import PandaParameterBox, PandaStructuredContext
from sarrl.safety.panda_structured_prefix import PandaPrefixConfig, PandaPrefixErrors
from sarrl.safety.panda_validated_prefix import PandaValidatedPrefix

BASELINE_SHA256 = "ac77c689474f861fa08cfeefe7be7b22a6262b101f99b8371ead41d02e8619d0"


def _json(value):
    def encode(array):
        if isinstance(array, np.ndarray):
            return array.tolist()
        raise TypeError(type(array).__name__)

    return json.dumps(value, default=encode, allow_nan=False, separators=(",", ":"))


def fingerprints():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "sarrl").rglob("*.py"))
    paths += [Path(__file__), root / "docs/panda-terminal-admission-methods.md"]
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(
            p.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for p in paths
    }


def baseline():
    root = Path(__file__).resolve().parents[1]
    data = (root / "results/panda_parameter_prefix/assessment.json.gz").read_bytes()
    if hashlib.sha256(data).hexdigest() != BASELINE_SHA256:
        raise ValueError("parameter-prefix baseline identity differs")
    return json.loads(gzip.decompress(data))


def assessment():
    import flint
    import mujoco
    import scipy

    before, provenance, prior = fingerprints(), model_provenance(), baseline()
    inputs = prior["inputs"]
    full = PandaParameterBox(**inputs["parameters"])
    original = PandaStructuredContext(**inputs["context"])
    guarded = bounded_friction_context(original, full)
    actual = np.concatenate(([original.parameters[0]], full.lower[1:]))
    if np.any(actual < full.lower) or np.any(actual > full.upper):
        raise ValueError("equilibrium witness is outside the declared physical parameter box")
    q, v = VectorBox(**inputs["q_domain"]), VectorBox(**inputs["v_domain"])
    dynamics = PandaSnapshotDynamics(
        q,
        v,
        mass_remainder=np.zeros((7, 7)),
        bias_remainder=np.zeros(7),
        assumption=inputs["assumption"],
        cells=1,
        cache_size=128,
    )
    checker = PandaValidatedPrefix(
        dynamics,
        full,
        q.lower,
        q.upper,
        v.upper,
        inputs["torque_limit"],
        PandaPrefixErrors(**inputs["errors"]),
        PandaPrefixConfig(**inputs["config"]),
    )
    arms = {}
    for name, context in (("retained_friction", original), ("bounded_friction", guarded)):
        modes = []
        for lag in range(5):
            result = enclose_sampled_stability(context, actual, lag=lag)
            if (
                result.model_xml_sha256 != prior["model_xml_sha256"]
                or result.compiled_constants_sha256 != prior["compiled_constants_sha256"]
            ):
                raise ValueError("sampled witness and baseline models differ")
            limits = np.array(inputs["torque_limit"])
            if np.any(result.equilibrium_torque.lower <= -limits) or np.any(
                result.equilibrium_torque.upper >= limits
            ):
                raise ValueError("equilibrium does not have strict unsaturated torque margin")
            modes.append(
                {
                    "lag": lag,
                    "enclosure": asdict(result),
                    "instability_verified": result.instability_verified,
                    "point_local_schur_stability_verified": result.local_schur_stability_verified,
                }
            )
        prefix = verify_parameter_prefix(
            checker,
            context,
            inputs["measured_q"],
            inputs["measured_v"],
            inputs["candidate"],
            inputs["history"],
            intervals=6,
            cells=16,
        )
        replay = audit_parameter_prefix(
            checker,
            context,
            inputs["measured_q"],
            inputs["measured_v"],
            inputs["candidate"],
            inputs["history"],
            prefix,
            intervals=6,
            cells=16,
        )
        delivery = all(
            len(case.intervals) >= case.lag + 2
            for cell in prefix.cases
            for case in cell.prefix.cases
        )
        arms[name] = {
            "context": asdict(context),
            "sampled_modes": modes,
            "prefix": asdict(prefix),
            "structural_audit": replay,
            "new_feedback_delivered_in_every_branch": delivery,
            "necessary_checks_passed": (
                all(x["point_local_schur_stability_verified"] for x in modes)
                and prefix.prefix_contained
                and delivery
            ),
        }
    if fingerprints() != before or model_provenance() != provenance:
        raise ValueError("source or model assets changed during terminal admission assessment")
    return json.loads(
        _json(
            {
                "schema": 1,
                "scope": "Necessary admission diagnostics, not a physical or terminal certificate",
                "source_sha256": before,
                "source_hash_normalization": "CRLF to LF",
                "baseline_sha256": BASELINE_SHA256,
                "model_provenance": provenance,
                "compiled_constants_sha256": prior["compiled_constants_sha256"],
                "runtime": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "mujoco": mujoco.__version__,
                    "python_flint": flint.__version__,
                },
                "baseline_inputs": inputs,
                "actual_equilibrium_parameters": actual,
                "prefix_intervals": 6,
                "outer_payload_cells": 16,
                "inner_solve_cells": 1,
                "arms": arms,
                "binding": {
                    "nonlinear_terminal_verified": False,
                    "physical_force_binding_verified": False,
                    "gaussian_deterministic_sensor_bound_available": False,
                    "partial_runtime_defect_bound_verified": False,
                    "causal_production_handoff_verified": False,
                },
                "decision": {
                    "original_candidate_promoted": False,
                    "guarded_candidate_promoted": False,
                    "integration_allowed": False,
                    "task_comparison_allowed": False,
                    "release_allowed": False,
                    "physical_step_calls": 0,
                    "task_seeds_opened": 0,
                },
            }
        )
    )


def summary(report):
    arms = {}
    for name, arm in report["arms"].items():
        cases = [case for cell in arm["prefix"]["cases"] for case in cell["prefix"]["cases"]]
        arms[name] = {
            "spectral_radius_intervals": [
                [
                    max(mode["enclosure"]["eigen_modulus"]["lower"]),
                    max(mode["enclosure"]["eigen_modulus"]["upper"]),
                ]
                for mode in arm["sampled_modes"]
            ],
            "point_local_stability_all_lags": all(
                x["point_local_schur_stability_verified"] for x in arm["sampled_modes"]
            ),
            "complete_prefix_branches": sum(x["prefix_contained"] for x in cases),
            "checked_intervals": arm["structural_audit"]["checked_intervals"],
            "prefix_interval_range": [
                min(len(x["intervals"]) for x in cases),
                max(len(x["intervals"]) for x in cases),
            ],
            "all_feedback_deliveries_verified": arm["new_feedback_delivered_in_every_branch"],
            "necessary_checks_passed": arm["necessary_checks_passed"],
        }
    return {"arms": arms, "binding": report["binding"], "decision": report["decision"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if not args.audit and args.output.exists():
        raise FileExistsError("preserve the existing report; choose a new output path")
    retained = json.loads(gzip.decompress(args.output.read_bytes())) if args.audit else None
    if retained is not None and retained.get("source_sha256") != fingerprints():
        raise ValueError("terminal admission source fingerprints differ")
    report = assessment()
    if args.audit:
        if report != retained:
            raise ValueError("terminal admission report differs from frozen recomputation")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(gzip.compress((_json(report) + "\n").encode(), mtime=0))
    print(json.dumps({"audit": args.audit, **summary(report)}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
