"""Final shared-parameter phase: four held substeps per command period."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tools.assess_panda_mean_value_prefix import _json

ROOT = Path(__file__).resolve().parents[1]
RETAINED = ROOT / "results/panda_mean_value_prefix/assessment.json.gz"
RETAINED_SHA256 = "18746bc56a77f70bde364ec068f866b6c9316f57a09ea0df24c52ec3478c19d6"
SUBSTEPS = 4
INTERVALS = 6


def fingerprints():
    paths = sorted((ROOT / "sarrl").rglob("*.py"))
    paths += [Path(__file__), ROOT / "docs/panda-substep-prefix-methods.md"]
    return {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(
            p.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for p in paths
    }


def retained():
    data = RETAINED.read_bytes()
    if hashlib.sha256(data).hexdigest() != RETAINED_SHA256:
        raise ValueError("retained shared-parameter report identity differs")
    return json.loads(gzip.decompress(data))


def decide(cells):
    cases = [c for cell in cells for c in cell["result"]["cases"]]
    complete = sum(c["prefix_contained"] for c in cases)
    return {
        "rule": "every branch completes all six held intervals, otherwise the line stops",
        "complete_branches": complete,
        "branches": len(cases),
        "held_intervals": sum(len(c["intervals"]) // SUBSTEPS for c in cases),
        "outcome": "line_continues" if cases and complete == len(cases) else "line_stops",
        "terminal_verified": False,
        "physical_binding_verified": False,
        "integration_allowed": False,
        "task_comparison_allowed": False,
        "release_allowed": False,
        "physical_step_calls": 0,
        "task_seeds_opened": 0,
    }


def assessment():
    import flint
    import mujoco
    import scipy

    from sarrl.robots.panda import model_provenance
    from sarrl.safety.panda_mean_value_audit import audit_mean_value_prefix
    from sarrl.safety.panda_mean_value_dynamics import PandaMeanValueDynamics
    from sarrl.safety.panda_mean_value_prefix import PandaMeanValuePrefix
    from sarrl.safety.panda_state_region import VectorBox
    from sarrl.safety.panda_structured_backup import PandaParameterBox, PandaStructuredContext
    from sarrl.safety.panda_structured_prefix import PandaPrefixConfig, PandaPrefixErrors

    before, provenance, prior = fingerprints(), model_provenance(), retained()
    inputs, context = prior["baseline_inputs"], PandaStructuredContext(**prior["context"])
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
        raise ValueError("model differs from the retained shared-parameter report")
    args = (inputs["measured_q"], inputs["measured_v"], inputs["candidate"], inputs["history"])
    cells = []
    for kept in prior["cells"]:
        checker = PandaMeanValuePrefix(
            dynamics,
            PandaParameterBox(**kept["parameters"]),
            q.lower,
            q.upper,
            v.upper,
            inputs["torque_limit"],
            PandaPrefixErrors(**inputs["errors"]),
            PandaPrefixConfig(**inputs["config"]),
        )
        one = checker.verify_prefix(context, *args, intervals=INTERVALS)
        if json.loads(_json(asdict(one))) != kept["result"]:
            raise ValueError("one-substep run does not reproduce the retained report")
        result = checker.verify_prefix(context, *args, intervals=INTERVALS, substeps=SUBSTEPS)
        audit = audit_mean_value_prefix(
            checker, context, *args, result, intervals=INTERVALS, substeps=SUBSTEPS
        )
        cells.append({"parameters": kept["parameters"], "result": asdict(result), "audit": audit})
    if fingerprints() != before or model_provenance() != provenance:
        raise ValueError("source or model assets changed during the substep assessment")
    return json.loads(
        _json(
            {
                "schema": 1,
                "scope": "Normalized ideal-model numerical diagnostic",
                "source_sha256": before,
                "source_hash_normalization": "CRLF to LF",
                "retained_sha256": RETAINED_SHA256,
                "retained_reproduced_with_one_substep": True,
                "baseline_inputs": inputs,
                "context": prior["context"],
                "model_provenance": provenance,
                "compiled_constants_sha256": dynamics._signature,
                "runtime": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "mujoco": mujoco.__version__,
                    "python_flint": flint.__version__,
                },
                "held_intervals": INTERVALS,
                "substeps": SUBSTEPS,
                "cells": cells,
                "decision": decide(cells),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if not args.audit and args.output.exists():
        raise FileExistsError("preserve the existing report; choose a new output path")
    kept = json.loads(gzip.decompress(args.output.read_bytes())) if args.audit else None
    if kept is not None and kept.get("source_sha256") != fingerprints():
        raise ValueError("substep source fingerprints differ")
    report = assessment()
    if args.audit:
        if report != kept:
            raise ValueError("substep report differs from frozen recomputation")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(gzip.compress((_json(report) + "\n").encode(), mtime=0))
    print(json.dumps({"audit": args.audit, **report["decision"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
