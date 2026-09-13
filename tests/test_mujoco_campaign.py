import json
from pathlib import Path

import numpy as np
import pytest

from sarrl.evaluation.adaptive_campaign import V19_REPRODUCTION_REFERENCE
from sarrl.evaluation.adaptive_pilot import PilotEpisode
from sarrl.evaluation.mujoco_campaign import (
    V20_ARMS,
    V20_DESCRIPTIVE_EPISODES,
    V20_PLANT,
    V20_PRIMARY_ARMS,
    V20_PRIMARY_EPISODES,
    V20_PRIMARY_SEED_START,
    V20_SCENARIOS,
    V20_TRANSFER_EPISODES,
    analyze,
    v20_cells,
    v20_config,
    v20_protocol_dict,
    v20_seeds,
)


def _episode(arm, scenario, seed, plant, success, unsafe=False, distance=None):
    adaptive = arm.startswith("adaptive")
    return PilotEpisode(
        arm=arm,
        scenario=scenario,
        seed=seed,
        origin="official",
        plant=plant,
        outcome="success" if success else "timeout",
        steps=100,
        final_distance=0.04 if success else 0.5 if distance is None else distance,
        success=success,
        unsafe_episode=unsafe,
        normalized_violation_max=0.1 if unsafe else 0.0,
        safety_infeasible=False,
        safety_intervention_fraction=0.1,
        true_delay=1,
        selected_lag=1 if adaptive else None,
        converged_at_step=None,
        prediction_error_rms=(0.1, 0.2) if adaptive else None,
        estimated_parameters=None,
        true_parameters=[[1.0] * 7, [1.0] * 7],
        selected_time_constant=0.03 if adaptive else None,
        true_time_constant=0.02,
        true_armature=0.05,
    )


def _campaign(adaptive_success, fixed_success, adaptive_unsafe=0.0, fixed_unsafe=0.0):
    rng = np.random.default_rng(0)
    episodes = []
    for arm, scenario, seed, plant in v20_cells():
        adaptive = arm.startswith("adaptive")
        if seed < V20_PRIMARY_SEED_START:
            # Transfer block: deterministic rows that match the synthetic reference.
            episodes.append(_episode(arm, scenario, seed, plant, False, distance=0.5))
            continue
        episodes.append(
            _episode(
                arm,
                scenario,
                seed,
                plant,
                success=bool(rng.random() < (adaptive_success if adaptive else fixed_success)),
                unsafe=bool(rng.random() < (adaptive_unsafe if adaptive else fixed_unsafe)),
            )
        )
    return episodes


@pytest.fixture
def reference(tmp_path):
    lines = ["scenario,controller,seed,reward,steps,success,final_distance"]
    for scenario in V20_SCENARIOS:
        for seed in v20_seeds("transfer"):
            lines.append(f"{scenario},A0_computed_torque,{seed},-1.0,250,False,0.5")
    path = tmp_path / "heldout_episodes.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_cells_cover_transfer_decision_and_descriptive_blocks():
    cells = list(v20_cells())
    transfer = 2 * V20_TRANSFER_EPISODES * len(V20_SCENARIOS)
    decision = len(V20_PRIMARY_ARMS) * V20_PRIMARY_EPISODES * len(V20_SCENARIOS)
    descriptive = 2 * V20_DESCRIPTIVE_EPISODES * len(V20_SCENARIOS)
    assert len(cells) == transfer + decision + descriptive == 600 + 11400 + 600
    assert len(set(cells)) == len(cells)
    plants = {plant for _, _, seed, plant in cells if seed >= V20_PRIMARY_SEED_START}
    assert plants == {V20_PLANT}
    assert {plant for _, _, seed, plant in cells if seed < V20_PRIMARY_SEED_START} == {
        "analytical",
        V20_PLANT,
    }
    assert set(v20_seeds("decision")) == set(range(52200, 54100))
    assert all(arm in V20_ARMS for arm, _, _, _ in cells)


def test_protocol_names_the_plant_options_and_the_grid():
    protocol = v20_protocol_dict()
    assert protocol["plant"] == "mujoco"
    assert protocol["plant_options"]["actuator_time_constant_range"] == (0.01, 0.05)
    assert protocol["estimator"]["actuator_time_constants"] == [0.0, 0.01, 0.03, 0.06]
    assert protocol["decision"]["order"][0] == "safety_veto"
    json.dumps(protocol)
    assert v20_config().actuator_time_constants == (0.0, 0.01, 0.03, 0.06)


def test_go_requires_gain_and_non_inferiority(reference):
    report = analyze(_campaign(0.9, 0.1), reference)
    assert report["decision"] == "go"
    assert report["transfer_check"]["reproduction_of_retained_a0"]["matched"] == 300
    assert report["transfer_check"]["plant_difference"]["id_reference"]["pairs"] == 100
    assert report["estimator"]["time_constant_abs_error_mean_s"] == pytest.approx(0.01)
    assert report["decision_summary"]["adaptive_hocbf/motor_fault"]["episodes"] == 1900
    assert report["descriptive_summary"]["adaptive/id_reference"]["episodes"] == 100


def test_veto_and_inconclusive_paths(reference):
    assert analyze(_campaign(0.9, 0.1, 0.3, 0.0), reference)["decision"] == "no_go_safety"
    assert analyze(_campaign(0.25, 0.2), reference)["decision"] == "inconclusive"


def test_reordered_or_incomplete_campaign_is_rejected(reference):
    episodes = _campaign(0.9, 0.1)
    with pytest.raises(ValueError):
        analyze(episodes[1:], reference)
    with pytest.raises(ValueError):
        analyze(episodes[1:] + episodes[:1], reference)


def test_analysis_with_the_retained_reference():
    retained = Path(V19_REPRODUCTION_REFERENCE)
    if not retained.exists():
        pytest.skip("retained v1.3 evidence unavailable")
    report = analyze(_campaign(0.9, 0.1), retained)
    # Synthetic rows do not reproduce the retained outcomes; the check reports that.
    assert report["transfer_check"]["reproduction_of_retained_a0"]["compared"] == 300
