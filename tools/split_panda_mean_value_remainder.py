"""Split the shared-parameter Panda residuals by origin, from the retained report."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results/panda_mean_value_prefix/assessment.json.gz"
REPORT_SHA256 = "18746bc56a77f70bde364ec068f866b6c9316f57a09ea0df24c52ec3478c19d6"
ORIGINS = ("jacobian_acceleration", "jacobian_feedback", "taylor", "rounding")
THRESHOLD = 0.5
TOLERANCE = 1e-9


def _zero():
    return {o: np.zeros(7) for o in ORIGINS}


def _total(x):
    return sum(x.values())


def _settle(x, recorded, where):
    """Assign the recorded radius not explained by the other origins to rounding."""
    gap = recorded - _total(x)
    if np.any(gap < -TOLERANCE * np.maximum(recorded, _total(x))):
        raise ValueError(f"reconstructed radius exceeds the record at {where}")
    return {**x, "rounding": x["rounding"] + gap}


def split_branch(steps, lag, dt):
    """Carry origin-labelled residual radii through the frozen prefix recursion.

    Each step holds radii vectors recorded by the screen and, from the frozen maps,
    |J| and rad(J) with the generator radius of the hull ("acceleration" and
    "feedback", the latter None for the first interval, whose send is a constant).
    """
    q, v, queue, rows = _zero(), _zero(), [_zero() for _ in range(4)], []
    for k, s in enumerate(steps):
        qv = np.concatenate
        if s["feedback"] is None:
            sent = inherited_sent = _zero()
        else:
            mag, rad, gen = s["feedback"]
            sent = {o: mag @ qv((q[o], v[o])) for o in ORIGINS}
            sent["jacobian_feedback"] = sent["jacobian_feedback"] + rad @ gen
            sent = _settle(sent, s["sent"], f"interval {k} send")
            inherited_sent = {"all": mag @ qv((_total(q), _total(v)))}
        delivered = queue[-lag] if lag else sent
        if not np.allclose(_total(delivered), s["delivered"], rtol=1e-12, atol=0):
            raise ValueError(f"interval {k} delivers a different send than the FIFO")
        inherited_delivered = _total(queue[-lag]) if lag else _total(inherited_sent)
        mag, rad, gen = s["acceleration"]
        acc = {o: mag @ qv((q[o], v[o], np.zeros(8), delivered[o])) for o in ORIGINS}
        acc["jacobian_acceleration"] = acc["jacobian_acceleration"] + rad @ gen
        acc = _settle(acc, s["acceleration_start"], f"interval {k} acceleration")
        inherited_v = _total(v) + dt * (
            mag @ qv((_total(q), _total(v), np.zeros(8), inherited_delivered))
        )
        v_next = {o: v[o] + dt * acc[o] for o in ORIGINS}
        q_next = {o: q[o] + dt * v[o] + dt * dt / 2 * acc[o] for o in ORIGINS}
        v_next["taylor"] = v_next["taylor"] + dt * dt / 2 * s["jerk"]
        q_next["taylor"] = q_next["taylor"] + dt**3 / 6 * s["jerk"]
        v_next = _settle(v_next, s["v_end"], f"interval {k} velocity")
        q_next = _settle(q_next, s["q_end"], f"interval {k} position")
        total_v = float(np.sum(_total(v_next)))
        rows.append({"v_radius": total_v, "v_inherited": float(np.sum(inherited_v))})
        q, v, queue = q_next, v_next, [*queue[1:], sent]
    return {
        "velocity": {o: float(np.sum(v[o])) for o in ORIGINS},
        "position": {o: float(np.sum(q[o])) for o in ORIGINS},
        "steps": rows,
    }


def share(origins, key):
    return origins[key] / sum(origins.values())


def _affine(record):
    from sarrl.safety.panda_affine_state import AffineState
    from sarrl.safety.panda_center_arithmetic import ArrayEnclosure

    r = record["residual"]
    return AffineState(
        np.array(record["center"]),
        np.array(record["generators"]),
        ArrayEnclosure(np.array(r["lower"]), np.array(r["upper"])),
    )


def _same(a, b):
    pairs = (
        (a.center, b.center),
        (a.generators, b.generators),
        (a.residual.lower, b.residual.lower),
        (a.residual.upper, b.residual.upper),
    )
    return all(np.array_equal(x, y) for x, y in pairs)


def _magnitudes(form, jacobian, backend):
    from sarrl.safety.panda_center_arithmetic import _export
    from sarrl.safety.panda_validated_dynamics import point

    midpoint = np.array([x.mid() for x in jacobian.flat], dtype=object).reshape(jacobian.shape)
    gen = np.sum(abs(point(form.generators, backend)), axis=1)
    return (
        _export(abs(jacobian)).upper,
        _export(abs(jacobian - midpoint)).upper,
        _export(gen).upper,
    )


def recompute(report):
    """Rebuild every accepted interval with the frozen maps and capture their Jacobians."""
    import sarrl.safety.panda_mean_value_dynamics as maps
    from sarrl.safety.panda_mean_value_dynamics import PandaMeanValueDynamics
    from sarrl.safety.panda_mean_value_prefix import PandaMeanValuePrefix, add_error
    from sarrl.safety.panda_state_region import VectorBox
    from sarrl.safety.panda_structured_backup import PandaParameterBox, PandaStructuredContext
    from sarrl.safety.panda_structured_prefix import PandaPrefixConfig, PandaPrefixErrors
    from sarrl.safety.panda_validated_dynamics import arithmetic

    inputs = report["baseline_inputs"]
    context = PandaStructuredContext(**report["context"])
    q_domain, v_domain = VectorBox(**inputs["q_domain"]), VectorBox(**inputs["v_domain"])
    dynamics = PandaMeanValueDynamics(
        q_domain,
        v_domain,
        mass_remainder=np.zeros((7, 7)),
        bias_remainder=np.zeros(7),
        assumption=inputs["assumption"],
        cells=1,
        cache_size=128,
    )
    if dynamics._signature != report["compiled_constants_sha256"]:
        raise ValueError("compiled model differs from the retained report")
    captured, original = [], maps.mean_value

    def capture(form, value, jacobian, backend):
        captured.append((form, jacobian))
        return original(form, value, jacobian, backend)

    branches = []
    maps.mean_value = capture
    try:
        for index, cell in enumerate(report["cells"]):
            checker = PandaMeanValuePrefix(
                dynamics,
                PandaParameterBox(**cell["parameters"]),
                q_domain.lower,
                q_domain.upper,
                v_domain.upper,
                inputs["torque_limit"],
                PandaPrefixErrors(**inputs["errors"]),
                PandaPrefixConfig(**inputs["config"]),
            )
            parameters, errors = _affine(cell["result"]["parameters"]), checker.errors
            for case in cell["result"]["cases"]:
                steps = []
                with arithmetic(checker.precision_bits) as backend:
                    for k, record in enumerate(case["intervals"]):
                        q, v = _affine(record["q_start"]), _affine(record["v_start"])
                        sent, delivered = _affine(record["sent"]), _affine(record["delivered"])
                        feedback = None
                        if k:
                            rebuilt = add_error(
                                dynamics.feedback_affine(
                                    context,
                                    add_error(q, errors.position_measurement, backend),
                                    add_error(v, errors.velocity_measurement, backend),
                                    backend,
                                ),
                                errors.command_error,
                                backend,
                            )
                            if not _same(rebuilt, sent):
                                raise ValueError(f"cell {index} lag {case['lag']} send {k}")
                            feedback = _magnitudes(*captured[-1], backend)
                        acc = dynamics.acceleration_affine(parameters, q, v, delivered, backend)
                        if not _same(acc, _affine(record["acceleration_start"])):
                            raise ValueError(f"cell {index} lag {case['lag']} acceleration {k}")
                        steps.append(
                            {
                                "feedback": feedback,
                                "acceleration": _magnitudes(*captured[-1], backend),
                                "sent": sent.residual.upper,
                                "delivered": delivered.residual.upper,
                                "acceleration_start": acc.residual.upper,
                                "jerk": np.array(record["jerk_bound"]["upper"]),
                                "q_end": np.array(record["q_end"]["residual"]["upper"]),
                                "v_end": np.array(record["v_end"]["residual"]["upper"]),
                            }
                        )
                last_v = case["intervals"][-1]["v_end"]
                generator_radius = np.sum(np.abs(np.array(last_v["generators"])), axis=1)
                branches.append(
                    {
                        "cell": index,
                        "lag": case["lag"],
                        "accepted_intervals": len(steps),
                        "residual_to_generator_ratio": float(
                            np.max(last_v["residual"]["upper"]) / np.max(generator_radius)
                        ),
                        **split_branch(steps, case["lag"], inputs["config"]["control_dt"]),
                    }
                )
                captured.clear()
    finally:
        maps.mean_value = original
    return branches


def summarize(branches):
    taylor = [share(b["velocity"], "taylor") for b in branches]
    median = float(np.median(taylor))
    shares = {
        part: {o: float(np.median([share(b[part], o) for b in branches])) for o in ORIGINS}
        for part in ("velocity", "position")
    }
    return {
        "branches": len(branches),
        "median_shares": shares,
        "taylor_velocity_share_range": [float(min(taylor)), float(max(taylor))],
        "decision": {
            "metric": "median Taylor-origin share of the summed velocity residual radius",
            "threshold": THRESHOLD,
            "value": median,
            "outcome": "final_taylor_phase_justified" if median >= THRESHOLD else "line_stops",
            "task_seeds_opened": 0,
            "physical_step_calls": 0,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve the existing split; choose a new output path")
    from sarrl.evaluation.provenance import assert_source_tree_clean

    assert_source_tree_clean(ROOT)
    data = REPORT.read_bytes()
    if hashlib.sha256(data).hexdigest() != REPORT_SHA256:
        raise ValueError("retained shared-parameter report identity differs")
    import flint
    import mujoco

    branches = recompute(json.loads(gzip.decompress(data)))
    result = {
        "schema": 1,
        "scope": "Origin of retained normalized ideal-model residual radii; not a new bound",
        "report_sha256": REPORT_SHA256,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "python_flint": flint.__version__,
        },
        "origins": list(ORIGINS),
        "summary": summarize(branches),
        "branches": branches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n"
    args.output.write_bytes(gzip.compress(text.encode(), mtime=0))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
