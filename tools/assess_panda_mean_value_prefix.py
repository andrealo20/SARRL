"""Compare shared-parameter propagation with the retained bounded-friction screen."""

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
from sarrl.safety.panda_mean_value_audit import audit_mean_value_prefix
from sarrl.safety.panda_mean_value_dynamics import PandaMeanValueDynamics
from sarrl.safety.panda_mean_value_prefix import PandaMeanValuePrefix
from sarrl.safety.panda_parameter_prefix import parameter_cells
from sarrl.safety.panda_state_region import VectorBox
from sarrl.safety.panda_structured_backup import PandaParameterBox, PandaStructuredContext
from sarrl.safety.panda_structured_prefix import PandaPrefixConfig, PandaPrefixErrors

BASELINE_SHA256 = "a909e5e551bb3badb59a690c6a999ad41524fe1ef0f729a6218a08c46f655bec"


def _json(value):
    def encode(array):
        if isinstance(array, np.ndarray):
            return array.tolist()
        raise TypeError(type(array).__name__)

    return json.dumps(value, default=encode, allow_nan=False, separators=(",", ":"))


def fingerprints():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "sarrl").rglob("*.py"))
    paths += [Path(__file__), root / "docs/panda-mean-value-methods.md"]
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(
            p.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for p in paths
    }


def baseline():
    path = (
        Path(__file__).resolve().parents[1] / "results/panda_terminal_admission/assessment.json.gz"
    )
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != BASELINE_SHA256:
        raise ValueError("terminal-admission baseline identity differs")
    return json.loads(gzip.decompress(data))


def assessment():
    import flint
    import mujoco
    import scipy

    before, provenance, prior = fingerprints(), model_provenance(), baseline()
    inputs = prior["baseline_inputs"]
    full = PandaParameterBox(**inputs["parameters"])
    context = PandaStructuredContext(**prior["arms"]["bounded_friction"]["context"])
    q, v = VectorBox(**inputs["q_domain"]), VectorBox(**inputs["v_domain"])
    dynamics = PandaMeanValueDynamics(
        q,
        v,
        mass_remainder=np.zeros((7, 7)),
        bias_remainder=np.zeros(7),
        assumption=inputs["assumption"],
        cells=1,
        cache_size=128,
    )
    if dynamics._signature != prior["compiled_constants_sha256"]:
        raise ValueError("mean-value and baseline models differ")
    cells = []
    for parameters in parameter_cells(full, 16):
        checker = PandaMeanValuePrefix(
            dynamics,
            parameters,
            q.lower,
            q.upper,
            v.upper,
            inputs["torque_limit"],
            PandaPrefixErrors(**inputs["errors"]),
            PandaPrefixConfig(**inputs["config"]),
        )
        result = checker.verify_prefix(
            context,
            inputs["measured_q"],
            inputs["measured_v"],
            inputs["candidate"],
            inputs["history"],
            intervals=6,
        )
        audit = audit_mean_value_prefix(
            checker,
            context,
            inputs["measured_q"],
            inputs["measured_v"],
            inputs["candidate"],
            inputs["history"],
            result,
            intervals=6,
        )
        cells.append({"parameters": asdict(parameters), "result": asdict(result), "audit": audit})
    complete = bool(cells) and all(c["audit"]["prefix_contained"] for c in cells)
    delivery = bool(cells) and all(
        len(c["intervals"]) >= c["lag"] + 2 for cell in cells for c in cell["result"]["cases"]
    )
    if fingerprints() != before or model_provenance() != provenance:
        raise ValueError("source or model assets changed during shared-parameter assessment")
    return json.loads(
        _json(
            {
                "schema": 1,
                "scope": "Normalized ideal-model numerical diagnostic",
                "source_sha256": before,
                "source_hash_normalization": "CRLF to LF",
                "baseline_sha256": BASELINE_SHA256,
                "baseline_inputs": inputs,
                "context": asdict(context),
                "model_provenance": provenance,
                "compiled_constants_sha256": dynamics._signature,
                "runtime": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "mujoco": mujoco.__version__,
                    "python_flint": flint.__version__,
                },
                "shared_parameter_coordinates": 8,
                "prefix_intervals": 6,
                "outer_payload_cells": 16,
                "inner_solve_cells": 1,
                "cells": cells,
                "baseline_checked_intervals": prior["arms"]["bounded_friction"]["structural_audit"][
                    "checked_intervals"
                ],
                "decision": {
                    "numerical_prefix_contained": complete,
                    "new_feedback_delivered_in_every_branch": delivery,
                    "terminal_verified": False,
                    "physical_binding_verified": False,
                    "candidate_promoted": False,
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
    cases = [c for cell in report["cells"] for c in cell["result"]["cases"]]
    return {
        "checked_intervals": sum(c["audit"]["checked_intervals"] for c in report["cells"]),
        "complete_branches": sum(c["prefix_contained"] for c in cases),
        "lags": [
            {
                "lag": lag,
                "intervals": sum(len(c["intervals"]) for c in cases if c["lag"] == lag),
                "length_range": [
                    min(len(c["intervals"]) for c in cases if c["lag"] == lag),
                    max(len(c["intervals"]) for c in cases if c["lag"] == lag),
                ],
            }
            for lag in range(5)
        ],
        "failures": [
            {"cell": i, "lag": c["lag"], "reason": c["reason"]}
            for i, cell in enumerate(report["cells"])
            for c in cell["result"]["cases"]
            if not c["prefix_contained"]
        ],
        "decision": report["decision"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if not args.audit and args.output.exists():
        raise FileExistsError("preserve the existing report; choose a new output path")
    retained = json.loads(gzip.decompress(args.output.read_bytes())) if args.audit else None
    if retained is not None and retained.get("source_sha256") != fingerprints():
        raise ValueError("shared-parameter source fingerprints differ")
    report = assessment()
    if args.audit:
        if report != retained:
            raise ValueError("shared-parameter report differs from frozen recomputation")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(gzip.compress((_json(report) + "\n").encode(), mtime=0))
    print(json.dumps({"audit": args.audit, **summary(report)}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
