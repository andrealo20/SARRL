"""Frozen static numerical screen, without physical steps or task seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sarrl.controllers.panda_adaptive import PandaRegressor
from sarrl.robots.panda import HOME
from sarrl.safety.panda_prefix_audit import audit_prefix
from sarrl.safety.panda_regional_arithmetic import enclose_panda_region
from sarrl.safety.panda_state_region import EnclosureUnavailable, VectorBox
from sarrl.safety.panda_structured_backup import PandaParameterBox, PandaStructuredContext
from sarrl.safety.panda_structured_prefix import PandaPrefixConfig, PandaPrefixErrors
from sarrl.safety.panda_validated_dynamics import PandaValidatedDynamics, solve_acceleration
from sarrl.safety.panda_validated_prefix import PandaValidatedPrefix


def fingerprints():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "sarrl").rglob("*.py"))
    paths += [Path(__file__), root / "docs/panda-validated-prefix-methods.md"]
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(
            p.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for p in paths
    }


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def assessment():
    import flint
    import mujoco
    import scipy

    before = fingerprints()
    parameters = PandaParameterBox(np.array([0.5, *[0.5] * 7]), np.array([2.5, *[3.0] * 7]))
    context = PandaStructuredContext(HOME, np.array([1.5, *[1.75] * 7]))
    regressor = PandaRegressor()
    h0, rows = regressor.evaluate(HOME, np.zeros(7), np.zeros(7))
    candidate = h0 + 1.5 * rows[:, 0]
    history = [candidate.copy() for _ in range(4)]
    static = []
    for radius in (0.0001, 0.001, 0.01):
        q = VectorBox(HOME - radius, HOME + radius)
        v = VectorBox(-np.ones(7) * 0.01, np.ones(7) * 0.01)
        coefficients = enclose_panda_region(q, v)
        case = {"requested_radius": radius, "q_domain": asdict(q), "v_domain": asdict(v)}
        try:
            acceleration = solve_acceleration(
                coefficients, parameters, v, VectorBox.point(candidate)
            )
            case.update(available=True, acceleration=asdict(acceleration))
        except EnclosureUnavailable as failure:
            case.update(available=False, reason=str(failure))
        static.append(case)
    q_domain, v_domain = VectorBox(HOME - 0.5, HOME + 0.5), VectorBox(-np.ones(7), np.ones(7))
    dynamics = PandaValidatedDynamics(
        q_domain,
        v_domain,
        mass_remainder=np.zeros((7, 7)),
        bias_remainder=np.zeros(7),
        assumption="Normalized ideal unconstrained model; no sensor or discrete runtime binding",
    )
    errors = PandaPrefixErrors(*[np.zeros(7) for _ in range(5)])
    config = PandaPrefixConfig()
    torque_limit = np.array([87.0] * 4 + [12.0] * 3)
    checker = PandaValidatedPrefix(
        dynamics,
        parameters,
        q_domain.lower,
        q_domain.upper,
        v_domain.upper,
        torque_limit,
        errors,
        config,
    )
    result = checker.verify_prefix(context, HOME, np.zeros(7), candidate, history, intervals=2)
    replay = audit_prefix(
        checker, context, HOME, np.zeros(7), candidate, history, result, intervals=2
    )
    if fingerprints() != before:
        raise RuntimeError("implementation changed during numerical assessment")
    report = {
        "schema": 1,
        "scope": "Static normalized-model numerics, not a task comparison",
        "source_sha256": before,
        "source_hash_normalization": "CRLF to LF",
        "model_xml_sha256": coefficients.center.model_xml_sha256,
        "compiled_constants_sha256": coefficients.center.compiled_constants_sha256,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "mujoco": mujoco.__version__,
            "python_flint": flint.__version__,
        },
        "inputs": {
            "parameters": asdict(parameters),
            "context": asdict(context),
            "candidate": candidate,
            "history": history,
            "measured_q": HOME,
            "measured_v": np.zeros(7),
            "errors": asdict(errors),
            "config": asdict(config),
            "requested_intervals": 2,
            "mass_cells": 16,
            "precision_bits": 128,
            "q_domain": asdict(q_domain),
            "v_domain": asdict(v_domain),
            "torque_limit": torque_limit,
            "assumption": dynamics.assumption,
        },
        "static_cases": static,
        "prefix": asdict(result),
        "structural_audit": replay,
        "decision": {
            "numerical_prefix_contained": result.prefix_contained,
            "terminal_verified": False,
            "physical_force_binding_verified": False,
            "sensor_runtime_binding_verified": False,
            "causal_production_handoff_verified": False,
            "integration_allowed": False,
            "task_comparison_allowed": False,
            "candidate_promoted": False,
            "release_allowed": False,
            "physical_step_calls": 0,
            "task_seeds_opened": 0,
        },
    }
    return json.loads(json.dumps(report, default=serializable, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if not args.audit and args.output.exists():
        raise FileExistsError("preserve the existing assessment; choose a new output path")
    retained = json.loads(args.output.read_text(encoding="utf-8")) if args.audit else None
    if retained is not None and retained.get("source_sha256") != fingerprints():
        raise ValueError("assessment source fingerprints differ")
    report = assessment()
    if args.audit:
        if retained != report:
            raise ValueError("retained assessment differs from the frozen recomputation")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "audit": args.audit,
                "static_available": [x["available"] for x in report["static_cases"]],
                "prefix_cases": [
                    {"lag": x["lag"], "intervals": len(x["intervals"]), "reason": x["reason"]}
                    for x in report["prefix"]["cases"]
                ],
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
