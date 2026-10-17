"""Frozen comparison of local coefficients and persistent payload branches on used fixtures."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sarrl.safety.panda_local_coefficients import PandaLocalDynamics, enclose_local_coefficients
from sarrl.safety.panda_parameter_prefix import audit_parameter_prefix, verify_parameter_prefix
from sarrl.safety.panda_state_region import EnclosureUnavailable, VectorBox
from sarrl.safety.panda_structured_backup import PandaParameterBox, PandaStructuredContext
from sarrl.safety.panda_structured_prefix import PandaPrefixConfig, PandaPrefixErrors
from sarrl.safety.panda_validated_dynamics import solve_acceleration
from sarrl.safety.panda_validated_prefix import PandaValidatedPrefix

BASELINE_SHA256 = "cd67880ee7ad6e578a8fee33c158f73d18db98b3b35b7750f5ef6713df3fad46"


def fingerprints():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "sarrl").rglob("*.py"))
    paths += [Path(__file__), root / "docs/panda-parameter-prefix-methods.md"]
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(
            p.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for p in paths
    }


def _baseline():
    root = Path(__file__).resolve().parents[1]
    data = (
        (root / "results/panda_validated_prefix/assessment.json")
        .read_bytes()
        .replace(b"\r\n", b"\n")
    )
    if hashlib.sha256(data).hexdigest() != BASELINE_SHA256:
        raise ValueError("frozen baseline assessment differs")
    return json.loads(data)


def _json(report):
    def array(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)

    return json.dumps(report, default=array, allow_nan=False, separators=(",", ":"))


def assessment():
    import flint
    import mujoco
    import scipy

    before, baseline = fingerprints(), _baseline()
    inputs = baseline["inputs"]
    parameters = PandaParameterBox(**inputs["parameters"])
    context = PandaStructuredContext(**inputs["context"])
    q_domain = VectorBox(**inputs["q_domain"])
    v_domain = VectorBox(**inputs["v_domain"])
    errors = PandaPrefixErrors(**inputs["errors"])
    config = PandaPrefixConfig(**inputs["config"])
    static = []
    for original in baseline["static_cases"]:
        q, v = VectorBox(**original["q_domain"]), VectorBox(**original["v_domain"])
        coefficients = enclose_local_coefficients(q, v)
        if (
            coefficients.model_xml_sha256 != baseline["model_xml_sha256"]
            or coefficients.compiled_constants_sha256 != baseline["compiled_constants_sha256"]
        ):
            raise ValueError("local coefficients and baseline model identities differ")
        case = {
            "requested_radius": original["requested_radius"],
            "q_domain": asdict(q),
            "v_domain": asdict(v),
            "baseline_available": original["available"],
        }
        try:
            result = solve_acceleration(
                coefficients, parameters, v, VectorBox.point(inputs["candidate"]), cells=16
            )
            case.update(available=True, acceleration=asdict(result))
        except EnclosureUnavailable as failure:
            case.update(available=False, reason=str(failure))
        static.append(case)
    dynamics = PandaLocalDynamics(
        q_domain,
        v_domain,
        mass_remainder=np.zeros((7, 7)),
        bias_remainder=np.zeros(7),
        assumption=inputs["assumption"],
        cells=1,
    )
    checker = PandaValidatedPrefix(
        dynamics,
        parameters,
        q_domain.lower,
        q_domain.upper,
        v_domain.upper,
        inputs["torque_limit"],
        errors,
        config,
    )
    arguments = (
        checker,
        context,
        inputs["measured_q"],
        inputs["measured_v"],
        inputs["candidate"],
        inputs["history"],
    )
    prefix = verify_parameter_prefix(*arguments, intervals=inputs["requested_intervals"], cells=16)
    replay = audit_parameter_prefix(
        *arguments, prefix, intervals=inputs["requested_intervals"], cells=16
    )
    if fingerprints() != before:
        raise ValueError("implementation changed during parameter-prefix assessment")
    report = {
        "schema": 1,
        "scope": "Used-fixture numerical diagnosis; no physical or terminal guarantee",
        "source_sha256": before,
        "source_hash_normalization": "CRLF to LF",
        "baseline_sha256": BASELINE_SHA256,
        "model_xml_sha256": coefficients.model_xml_sha256,
        "compiled_constants_sha256": coefficients.compiled_constants_sha256,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "mujoco": mujoco.__version__,
            "python_flint": flint.__version__,
        },
        "inputs": inputs,
        "outer_payload_cells": 16,
        "inner_solve_cells": 1,
        "static_cases": static,
        "prefix": asdict(prefix),
        "structural_audit": replay,
        "baseline_lags": [
            {
                "lag": case["lag"],
                "checked_intervals": len(case["intervals"]),
                "reason": case["reason"],
            }
            for case in baseline["prefix"]["cases"]
        ],
        "decision": {**baseline["decision"], "numerical_prefix_contained": prefix.prefix_contained},
    }
    return json.loads(_json(report))


def summary(report):
    return {
        "static_available": [x["available"] for x in report["static_cases"]],
        "payload_cells": len(report["prefix"]["cases"]),
        "checked_intervals": report["structural_audit"]["checked_intervals"],
        "failed_branches": [
            {
                "cell": index,
                "lag": case["lag"],
                "intervals": len(case["intervals"]),
                "reason": case["reason"],
            }
            for index, cell in enumerate(report["prefix"]["cases"])
            for case in cell["prefix"]["cases"]
            if not case["prefix_contained"]
        ],
        "decision": report["decision"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if not args.audit and args.output.exists():
        raise FileExistsError("preserve the existing assessment; choose a new output path")
    retained = json.loads(gzip.decompress(args.output.read_bytes())) if args.audit else None
    if retained is not None and retained.get("source_sha256") != fingerprints():
        raise ValueError("assessment source fingerprints differ")
    report = assessment()
    if args.audit:
        if report != retained:
            raise ValueError("retained parameter assessment differs from recomputation")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(gzip.compress((_json(report) + "\n").encode("utf-8"), mtime=0))
    print(json.dumps({"audit": args.audit, **summary(report)}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
