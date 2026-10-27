"""Mechanism test: the v2.6 Panda joint filter built over the bank's tied ages only."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from sarrl.controllers.panda_delay import PandaDelayController
from sarrl.controllers.panda_filtered import PandaFilteredController
from sarrl.evaluation import panda_joint_filter as v26
from sarrl.evaluation.panda_joint_filter import CONDITIONS, CONFIG, configurations

ARMS = ("cumulative", "robust_filter", "selected_filter")
SEEDS = tuple(range(9806200, 9806220))
FAST = tuple(c for c in CONDITIONS if c.startswith("fast_"))
_FACTORIES = {
    "cumulative": (lambda: PandaDelayController(predict=False), False),
    "robust_filter": (lambda: PandaFilteredController(CONFIG), True),
    "selected_filter": (lambda: PandaFilteredController(CONFIG, selected_ages=True), True),
}


def selected_filter_protocol():
    return {
        "kind": "panda_selected_delay_joint_filter_mechanism_test",
        "arms": list(ARMS),
        "scenarios": list(CONDITIONS),
        "seeds": list(SEEDS),
        "reserved_block": [9806200, 9806299],
        "tasks": {c: asdict(configurations(c)[0]) for c in CONDITIONS},
        "interfaces": {c: asdict(configurations(c)[1]) for c in CONDITIONS},
        "filter": asdict(CONFIG),
        "change": "constraints over ages whose accumulated bank score equals the minimum",
        "invalid_age_set": "non-finite scores or empty set latch an abort",
        "unchanged": "bank, RLS, frozen affine prediction, QP, checks, abort and braking",
        "reproduction": {
            "episodes": 480,
            "reference": "results/panda_joint_filter_pilot/episodes.json",
            "arms": list(v26.ARMS),
        },
        "predictions": {
            "A": "each 20 ms fast cell: selected aborts < robust aborts",
            "B": "each 40 ms fast cell: selected aborts >= robust aborts - 2, "
            "and at least one 40 ms fast selected episode unsafe",
            "C": "risk materialises if any 0 ms fast selected episode is unsafe",
        },
        "statements": {
            "aborts_reduced": "prediction A",
            "no_new_violations": "no paired task with selected unsafe and robust safe",
            "no_success_loss": "no paired task with robust success and selected failure",
        },
        "advancement": "the four v2.6 criteria, selected_filter against cumulative",
    }


def run_episode(seed, scenario, arm, *, retain_trace=False):
    """v2.8 arms by name; the v2.6 arm names reproduce the retained v2.6 records."""
    if arm in v26.ARMS:
        return v26.run_episode(seed, scenario, arm, retain_trace=retain_trace)
    if arm not in ARMS:
        raise ValueError("unknown Panda selected-filter arm")
    factory, filtered = _FACTORIES[arm]
    return v26.episode(seed, scenario, arm, factory, filtered=filtered, retain_trace=retain_trace)


def _lookup(records):
    lookup = {(r["scenario"], r["seed"], r["arm"]): r for r in records}
    seeds = sorted({r["seed"] for r in records})
    expected = {(c, s, a) for c in CONDITIONS for s in seeds for a in ARMS}
    if not seeds or len(lookup) != len(records) or set(lookup) != expected:
        raise ValueError("selected-filter analysis requires a complete paired factorial")
    return lookup, seeds


def _aborts(lookup, condition, seeds, arm):
    return sum(lookup[condition, s, arm]["filter"]["aborted"] for s in seeds)


def decide(records):
    lookup, seeds = _lookup(records)
    cells = {c: {arm: _aborts(lookup, c, seeds, arm) for arm in ARMS[1:]} for c in CONDITIONS}

    def unsafe(condition, arm):
        return [lookup[condition, s, arm]["unsafe_episode"] for s in seeds]

    d20 = [c for c in FAST if c.endswith("d20")]
    d40 = [c for c in FAST if c.endswith("d40")]
    d00 = [c for c in FAST if c.endswith("d00")]
    prediction_a = all(cells[c]["selected_filter"] < cells[c]["robust_filter"] for c in d20)
    prediction_b = all(
        cells[c]["selected_filter"] >= cells[c]["robust_filter"] - 2 for c in d40
    ) and any(any(unsafe(c, "selected_filter")) for c in d40)
    risk_c = any(any(unsafe(c, "selected_filter")) for c in d00)
    pairs = [
        (lookup[c, s, "robust_filter"], lookup[c, s, "selected_filter"])
        for c in CONDITIONS
        for s in seeds
    ]
    new_unsafe = sum(b["unsafe_episode"] and not a["unsafe_episode"] for a, b in pairs)
    removed_unsafe = sum(a["unsafe_episode"] and not b["unsafe_episode"] for a, b in pairs)
    lost = sum(a["success"] and not b["success"] for a, b in pairs)
    gained = sum(b["success"] and not a["success"] for a, b in pairs)
    excluded_true_age = 0
    for (condition, _seed, arm), row in lookup.items():
        if arm == "selected_filter":
            true_age = configurations(condition)[1].command_delay_steps
            excluded_true_age += any(true_age not in ages for ages in row["filter"]["age_sets"])
    relabelled = [
        {**r, "arm": "bank_align" if r["arm"] == "cumulative" else "bank_filtered"}
        for r in records
        if r["arm"] in ("cumulative", "selected_filter")
    ]
    return {
        "mechanism": {
            "prediction_A": prediction_a,
            "prediction_B": prediction_b,
            "risk_C_materialised": risk_c,
            "aborts_reduced": prediction_a,
            "no_new_violations": new_unsafe == 0,
            "no_success_loss": lost == 0,
            "paired_selected_unsafe_robust_safe": new_unsafe,
            "paired_robust_unsafe_selected_safe": removed_unsafe,
            "paired_robust_success_selected_failure": lost,
            "paired_selected_success_robust_failure": gained,
            "zero_ms_abort_reduction": "structural; reported, not evidence",
            "aborts_per_cell": cells,
            "selected_episodes_excluding_true_age": excluded_true_age,
        },
        "advancement": v26.evaluate_gate(relabelled),
        "meaning": "mechanism test on twenty paired tasks per cell; no safety certificate",
    }


def summarize(records):
    lookup, seeds = _lookup(records)
    cells = {}
    for condition in CONDITIONS:
        for arm in ARMS:
            rows = [lookup[condition, s, arm] for s in seeds]
            filtered = arm != "cumulative"
            abort_steps = [
                r["filter"]["abort_step"] for r in rows if filtered and r["filter"]["aborted"]
            ]
            cells[f"{condition}/{arm}"] = {
                "episodes": len(rows),
                "successes": sum(r["success"] for r in rows),
                "unsafe_episodes": sum(r["unsafe_episode"] for r in rows),
                "aborts": sum(r["filter"]["aborted"] for r in rows) if filtered else 0,
                "abort_steps": sorted(abort_steps),
                "abort_reasons": sorted(
                    reason
                    for r in rows
                    if filtered and r["filter"]["aborted"]
                    for reason, count in r["filter"]["reasons"].items()
                    if reason not in ("projected", "unchanged")
                ),
                "mean_tracking_rms_rad": np.mean(
                    [r["tracking_rms_rad"] for r in rows], axis=0
                ).tolist(),
            }
    return {
        "episodes": len(records),
        "physical_steps": sum(r["physical_steps"] for r in records),
        "cells": cells,
        "decision": decide(records),
    }
