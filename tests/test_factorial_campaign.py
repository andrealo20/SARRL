import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from sarrl.evaluation.adaptive_pilot import PilotEpisode
from sarrl.evaluation.factorial_campaign import (
    V21_ARM_LABELS,
    V21_DECISION_EPISODES,
    V21_DECISION_SEED_START,
    V21_REPRODUCTION_ARMS,
    V21_REPRODUCTION_EPISODES,
    V21_REPRODUCTION_REFERENCE,
    V21_SCENARIOS,
    analyze,
    v21_cells,
    v21_config,
    v21_protocol_dict,
    v21_seeds,
)

RATES = {  # success probabilities per arm, mirroring the pilot
    "fixed_hocbf": 0.10,
    "fixed_hocbf_adaptivemodel": 0.10,
    "adaptive_hocbf_fixedmodel": 0.60,
    "adaptive_hocbf": 0.90,
}


def _episode(arm, scenario, seed, success, unsafe=False):
    estimator = arm != "fixed_hocbf"
    return PilotEpisode(
        arm=arm,
        scenario=scenario,
        seed=seed,
        origin="official",
        plant="mujoco",
        outcome="success" if success else "timeout",
        steps=60 if success else 250,
        final_distance=0.04 if success else 0.5,
        reward=-1.0,
        max_speed=1.0,
        max_command_torque=10.0,
        fault_seen=scenario == "motor_fault",
        success=success,
        unsafe_episode=unsafe,
        normalized_violation_max=0.1 if unsafe else 0.0,
        safety_infeasible=False,
        safety_intervention_fraction=0.3,
        true_delay=1,
        selected_lag=1 if estimator else None,
        converged_at_step=None,
        prediction_error_rms=(0.1, 0.2) if estimator else None,
        estimated_parameters=None,
        true_parameters=[[1.0] * 7, [1.0] * 7],
        selected_time_constant=0.03 if estimator else None,
        true_time_constant=0.02,
        true_armature=0.05,
        control_steps=100 if estimator else 0,
        model_fallback_steps=0,
    )


def _campaign(rates=RATES, id_certificate_shift=0.0):
    rng = np.random.default_rng(0)
    episodes = []
    for arm, scenario, seed, _ in v21_cells():
        if seed < V21_DECISION_SEED_START:
            episodes.append(_episode(arm, scenario, seed, success=seed % 2 == 0))
            continue
        rate = rates[arm]
        if scenario == "id_reference" and arm == "adaptive_hocbf_fixedmodel":
            rate = rates["adaptive_hocbf"]  # the certificate makes no difference in distribution
        if scenario == "id_reference" and arm == "adaptive_hocbf":
            rate += id_certificate_shift
        episodes.append(_episode(arm, scenario, seed, success=bool(rng.random() < rate)))
    return episodes


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / "episodes.jsonl"
    lines = []
    for arm in V21_REPRODUCTION_ARMS:
        for scenario in V21_SCENARIOS:
            for seed in v21_seeds("reproduction"):
                lines.append(json.dumps(asdict(_episode(arm, scenario, seed, seed % 2 == 0))))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_cells_cover_reproduction_and_decision_blocks_in_order():
    cells = list(v21_cells())
    reproduction = len(V21_REPRODUCTION_ARMS) * V21_REPRODUCTION_EPISODES * len(V21_SCENARIOS)
    decision = len(V21_ARM_LABELS) * V21_DECISION_EPISODES * len(V21_SCENARIOS)
    assert len(cells) == reproduction + decision == 600 + 22800
    assert len(set(cells)) == len(cells)
    assert set(v21_seeds("decision")) == set(range(54200, 56100))
    assert set(v21_seeds("reproduction")) == set(range(52200, 52300))
    assert {plant for *_, plant in cells} == {"mujoco"}


def test_protocol_is_serialisable_and_names_the_factors():
    protocol = v21_protocol_dict()
    json.dumps(protocol)
    assert protocol["factors"] == {
        "controller_model": ["fixed", "identified"],
        "certificate_model": ["fixed", "identified"],
    }
    assert protocol["safety_endpoints"]["decision_weight"] is False
    assert protocol["decision"]["order"][0] == "estimator_validity"
    assert v21_config().actuator_time_constants == (0.0, 0.01, 0.03, 0.06)


def test_confirmed_when_the_three_statements_hold(reference):
    report = analyze(_campaign(), reference)
    assert report["decision"] == "confirmed"
    assert report["failed_statements"] == []
    assert report["reproduction_check"]["matched"] == 600
    effects = report["success"]["motor_fault"]
    assert effects["controller_at_fixed_certificate"]["difference"] > 0.4
    assert effects["certificate_at_identified_controller"]["difference"] > 0.2
    assert effects["interaction"]["ci95_low"] > 0.0
    assert set(report["safety"]["id_reference"]) >= {
        "unsafe_episode",
        "joint_position_violation_max_rad",
        "safety_intervention_fraction",
    }
    assert report["cell_summary"]["adaptive_hocbf/ood_compound"]["episodes"] == 1900


def test_partial_when_a_statement_fails(reference):
    rates = dict(RATES, adaptive_hocbf_fixedmodel=0.88)  # certificate adds nothing
    report = analyze(_campaign(rates), reference)
    assert report["decision"] == "partial"
    assert report["failed_statements"] == ["certificate_gain_under_mismatch"]
    shifted = analyze(_campaign(id_certificate_shift=0.08), reference)
    assert "certificate_equivalence_in_distribution" in shifted["failed_statements"]


def test_guarded_estimates_force_inconclusive(reference):
    episodes = [
        replace(e, model_fallback_steps=50)
        if e.arm != "fixed_hocbf" and e.seed % 3 == 0 and e.seed >= V21_DECISION_SEED_START
        else e
        for e in _campaign()
    ]
    report = analyze(episodes, reference)
    assert report["decision"] == "inconclusive"
    assert report["estimator"]["guarded_episode_fraction"] == pytest.approx(1 / 3, abs=0.01)


def test_reordered_campaign_and_bad_reproduction_are_refused(reference):
    episodes = _campaign()
    with pytest.raises(ValueError):
        analyze(episodes[1:] + episodes[:1], reference)
    broken = [replace(e, steps=e.steps + 1) if e.seed == 52200 else e for e in episodes]
    with pytest.raises(ValueError, match="do not reproduce"):
        analyze(broken, reference)


def test_reference_must_hold_the_retained_corner_rows():
    retained = Path(V21_REPRODUCTION_REFERENCE)
    if not retained.exists():
        pytest.skip("retained v2.0 evidence unavailable")
    from sarrl.evaluation.factorial_campaign import load_reference

    reference = load_reference(retained)
    assert len(reference) == 600
    assert all(row["plant"] == "mujoco" for row in reference.values())
