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
    V21_DESCRIPTIVE_ARM,
    V21_DESCRIPTIVE_EPISODES,
    V21_REPRODUCTION_ARMS,
    V21_REPRODUCTION_EPISODES,
    V21_REPRODUCTION_REFERENCE,
    V21_SCENARIOS,
    analyze,
    arm_stack_and_compensation,
    load_reference,
    reproduction_check,
    v21_cells,
    v21_config,
    v21_decision_cells,
    v21_protocol_dict,
    v21_reproduction_cells,
    v21_seeds,
)

RATES = {  # success probabilities per arm, mirroring the pilot
    "fixed_hocbf": 0.10,
    "fixed_hocbf_adaptivemodel": 0.10,
    "adaptive_hocbf_fixedmodel": 0.60,
    "adaptive_hocbf": 0.90,
    V21_DESCRIPTIVE_ARM: 0.80,
}


def _episode(arm, scenario, seed, success, unsafe=False, position_excess=0.0):
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
        estimated_parameters=[[1.0] * 7, [1.0] * 7] if estimator else None,
        true_parameters=[[1.0] * 7, [1.0] * 7],
        selected_time_constant=0.03 if estimator else None,
        true_time_constant=0.02,
        true_armature=0.05,
        control_steps=100 if estimator else 0,
        model_fallback_steps=0,
        joint_position_violation_max_rad=position_excess,
    )


def _campaign(rates=RATES, id_certificate_shift=0.0, unsafe_rate=0.0):
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
        unsafe = bool(rng.random() < unsafe_rate)
        # One draw per seed shared by every arm: outcomes are comonotone across
        # arms, as paired episodes on the same plant draw are in practice.
        draw = np.random.default_rng(seed).random()
        episodes.append(
            _episode(
                arm,
                scenario,
                seed,
                success=bool(draw < rate),
                unsafe=unsafe,
                position_excess=0.2 if unsafe and arm == "adaptive_hocbf_fixedmodel" else 0.01,
            )
        )
    return episodes


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / "episodes.jsonl"
    lines = []
    for arm in V21_REPRODUCTION_ARMS:
        for scenario in V21_SCENARIOS:
            for seed in v21_seeds("reproduction"):
                record = asdict(_episode(arm, scenario, seed, seed % 2 == 0))
                # The retained v2.0 rows predate the fields added afterwards.
                for name in (
                    "residual_rms",
                    "obstacle_present",
                    "obstacle_violation_max_m",
                    "obstacle_contact",
                    "obstacle_contact_steps",
                    "obstacle_contact_geoms",
                    "joint_position_violation_max_rad",
                    "joint_velocity_violation_max_rad_s",
                    "first_unsafe_step",
                ):
                    record.pop(name)
                lines.append(json.dumps(record))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_reproduction_block_precedes_every_decision_cell():
    cells = list(v21_cells())
    reproduction = list(v21_reproduction_cells())
    decision = list(v21_decision_cells())
    assert cells == reproduction + decision
    assert len(reproduction) == len(V21_REPRODUCTION_ARMS) * V21_REPRODUCTION_EPISODES * 3 == 600
    expected_decision = len(V21_ARM_LABELS) * V21_DECISION_EPISODES * 3
    expected_descriptive = V21_DESCRIPTIVE_EPISODES * 3
    assert len(decision) == expected_decision + expected_descriptive == 22800 + 300
    assert len(set(cells)) == len(cells)
    assert set(v21_seeds("decision")) == set(range(54200, 56100))
    assert set(v21_seeds("reproduction")) == set(range(52200, 52300))
    assert all(seed < V21_DECISION_SEED_START for _, _, seed, _ in reproduction)
    assert all(seed >= V21_DECISION_SEED_START for _, _, seed, _ in decision)


def test_arm_labels_map_to_stacks_and_compensation():
    assert arm_stack_and_compensation("adaptive_hocbf") == ("adaptive_hocbf", True)
    assert arm_stack_and_compensation("fixed_hocbf") == ("fixed_hocbf", True)
    assert arm_stack_and_compensation(V21_DESCRIPTIVE_ARM) == ("adaptive_hocbf", False)


def test_protocol_is_serialisable_and_names_the_factors():
    protocol = v21_protocol_dict()
    json.dumps(protocol)
    assert protocol["factors"]["controller_stack"]["levels"] == ["fixed", "identified"]
    assert protocol["safety_endpoints"]["decision_weight"] is False
    assert protocol["decision"]["order"][0] == "estimator_validity"
    assert protocol["decision"]["estimator_validity"]["applied_per"] == "arm and scenario"
    assert v21_config().actuator_time_constants == (0.0, 0.01, 0.03, 0.06)


def test_confirmed_when_the_three_statements_hold(reference):
    report = analyze(_campaign(unsafe_rate=0.1), reference)
    assert report["decision"] == "confirmed"
    assert report["failed_statements"] == []
    assert report["reproduction_check"]["matched"] == 600
    assert "true_parameters" in report["reproduction_check"]["fields"]
    effects = report["success"]["motor_fault"]
    assert effects["controller_at_fixed_certificate"]["ci95_low"] > 0.2
    assert effects["certificate_at_identified_controller"]["ci95_low"] > 0.1
    assert effects["interaction"]["ci95_low"] > 0.0
    safety = report["safety"]["id_reference"]
    assert set(safety) == {
        "operational",
        "violation_rate",
        "severity_paired",
        "severity_conditional",
    }
    conditional = safety["severity_conditional"]["adaptive_hocbf_fixedmodel"]
    assert conditional["joint_position_violation_max_rad"]["mean"] == pytest.approx(0.2)
    assert safety["violation_rate"]["unsafe_beyond_tolerance"]["difference"] < 0.0
    assert report["cell_summary"]["adaptive_hocbf/ood_compound"]["episodes"] == 1900
    assert report["descriptive_summary"][f"{V21_DESCRIPTIVE_ARM}/id_reference"]["episodes"] == 100


def test_partial_when_a_statement_fails(reference):
    rates = dict(RATES, adaptive_hocbf_fixedmodel=0.85)  # certificate adds little
    report = analyze(_campaign(rates), reference)
    assert report["decision"] == "partial"
    assert report["failed_statements"] == ["certificate_gain_under_mismatch"]
    shifted = analyze(_campaign(id_certificate_shift=0.08), reference)
    assert "certificate_equivalence_in_distribution" in shifted["failed_statements"]
    # A point estimate above the margin with a lower bound below it does not pass.
    weak = dict(RATES, adaptive_hocbf_fixedmodel=0.795)  # +10.5 pp, lower bound below +10
    assert (
        "certificate_gain_under_mismatch"
        in analyze(_campaign(weak), reference)["failed_statements"]
    )


def test_guarded_estimates_in_one_cell_force_inconclusive(reference):
    episodes = [
        replace(e, model_fallback_steps=50)
        if e.arm == "fixed_hocbf_adaptivemodel"
        and e.scenario == "ood_compound"
        and e.seed % 3 == 0
        and e.seed >= V21_DECISION_SEED_START
        else e
        for e in _campaign()
    ]
    report = analyze(episodes, reference)
    assert report["decision"] == "inconclusive"
    assert report["invalid_estimator_cells"] == ["fixed_hocbf_adaptivemodel/ood_compound"]


def test_reordered_campaign_and_bad_reproduction_are_refused(reference):
    episodes = _campaign()
    with pytest.raises(ValueError):
        analyze(episodes[1:] + episodes[:1], reference)
    broken = [
        replace(e, true_parameters=[[1.0] * 7, [1.1] * 7]) if e.seed == 52200 else e
        for e in episodes
    ]
    with pytest.raises(ValueError, match="do not reproduce"):
        analyze(broken, reference)
    check = reproduction_check(broken, reference)
    assert check["mismatches"][0][3] == "true_parameters"


def test_reference_must_hold_the_retained_corner_rows():
    retained = Path(V21_REPRODUCTION_REFERENCE)
    if not retained.exists():
        pytest.skip("retained v2.0 evidence unavailable")
    reference = load_reference(retained)
    assert len(reference) == 600
    assert all(row["plant"] == "mujoco" for row in reference.values())


def test_current_code_reproduces_retained_rows_field_by_field():
    """A small slice of the reproduction block against the retained v2.0 log."""
    pytest.importorskip("mujoco")
    retained = Path(V21_REPRODUCTION_REFERENCE)
    if not retained.exists():
        pytest.skip("retained v2.0 evidence unavailable")
    from sarrl.evaluation.adaptive_pilot import run_case
    from sarrl.evaluation.factorial_campaign import V21_COMPENSATE_DELAY, V21_PLANT_OPTIONS

    reference = load_reference(retained)
    rows = []
    for arm in V21_REPRODUCTION_ARMS:
        for scenario in V21_SCENARIOS:
            seed = v21_seeds("reproduction")[0]
            rows.append(
                run_case(
                    arm,
                    scenario,
                    seed,
                    "official",
                    v21_config(),
                    V21_COMPENSATE_DELAY,
                    "mujoco",
                    V21_PLANT_OPTIONS,
                )
            )
    from sarrl.evaluation.factorial_campaign import _values_match

    for episode in rows:
        retained_row = reference[(episode.arm, episode.scenario, episode.seed)]
        ours = asdict(episode)
        for name, value in retained_row.items():
            assert _values_match(ours[name], value), (episode.arm, episode.scenario, name)
