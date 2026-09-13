import json
from dataclasses import asdict

import numpy as np
import pytest

from sarrl.evaluation.adaptive_campaign import (
    V19_ARMS,
    V19_DESCRIPTIVE_EPISODES,
    V19_PRIMARY_EPISODES,
    V19_SCENARIOS,
    V19_SEED_START,
    analyze,
    load_episodes,
    paired_difference,
    v19_cells,
    v19_episodes,
    v19_protocol_dict,
)
from sarrl.evaluation.adaptive_pilot import PilotEpisode


def _episode(arm, scenario, seed, success, unsafe=False, abort=False):
    return PilotEpisode(
        arm=arm,
        scenario=scenario,
        seed=seed,
        origin="official",
        outcome="abort" if abort else ("success" if success else "timeout"),
        steps=100,
        final_distance=0.04 if success else 0.5,
        success=success,
        unsafe_episode=unsafe,
        normalized_violation_max=0.1 if unsafe else 0.0,
        safety_infeasible=abort,
        safety_intervention_fraction=0.1,
        true_delay=1,
        selected_lag=1 if arm.startswith("adaptive") else None,
        converged_at_step=None,
        prediction_error_rms=None,
        estimated_parameters=None,
        true_parameters=[[1.0] * 7, [1.0] * 7],
    )


def _campaign(adaptive_success, fixed_success, adaptive_unsafe=0.0, fixed_unsafe=0.0):
    rng = np.random.default_rng(0)
    episodes = []
    for arm, scenario, seed in v19_cells():
        adaptive = arm.startswith("adaptive")
        p_success = adaptive_success if adaptive else fixed_success
        p_unsafe = adaptive_unsafe if adaptive else fixed_unsafe
        episodes.append(
            _episode(
                arm,
                scenario,
                seed,
                success=bool(rng.random() < p_success),
                unsafe=bool(rng.random() < p_unsafe),
            )
        )
    return episodes


def test_cells_enumerate_every_arm_scenario_and_seed_once():
    cells = list(v19_cells())
    expected = sum(v19_episodes(arm) for arm in V19_ARMS) * len(V19_SCENARIOS)
    assert len(cells) == expected == (2 * 2000 + 2 * 100) * 3
    assert len(set(cells)) == len(cells)
    seeds = {seed for _, _, seed in cells}
    assert min(seeds) == V19_SEED_START and len(seeds) == V19_PRIMARY_EPISODES
    assert sum(1 for arm, _, _ in cells if arm == "fixed") == 3 * V19_DESCRIPTIVE_EPISODES


def test_protocol_dict_is_json_friendly_and_names_the_primary_contrast():
    protocol = v19_protocol_dict()
    assert protocol["primary_contrast"]["treatment"] == "adaptive_hocbf"
    assert protocol["primary_contrast"]["reference"] == "fixed_hocbf"
    assert protocol["decision"]["order"][0] == "safety_veto"
    assert protocol["training"] is False


def test_paired_difference_is_exact_on_deterministic_rows():
    treatment = [_episode("adaptive_hocbf", "id_reference", s, s % 2 == 0) for s in range(10)]
    reference = [_episode("fixed_hocbf", "id_reference", s, False) for s in range(10)]
    result = paired_difference(treatment, reference, lambda r: r.success, np.random.default_rng(1))
    assert result["difference"] == pytest.approx(0.5)
    assert result["ci95_low"] <= 0.5 <= result["ci95_high"]
    with pytest.raises(ValueError):
        paired_difference(treatment[:5], reference, lambda r: r.success, np.random.default_rng(1))


def test_go_when_success_gain_is_large_and_safety_holds():
    report = analyze(_campaign(adaptive_success=0.9, fixed_success=0.1))
    assert report["decision"] == "go"
    assert report["safety_vetoes"] == []
    assert report["summary"]["adaptive_hocbf/id_reference"]["episodes"] == V19_PRIMARY_EPISODES
    assert report["summary"]["fixed/id_reference"]["episodes"] == V19_DESCRIPTIVE_EPISODES
    assert all(report["unsafe_non_inferior_at_margin"].values())


def test_safety_veto_precedes_the_primary_endpoint():
    report = analyze(
        _campaign(adaptive_success=0.9, fixed_success=0.1, adaptive_unsafe=0.3, fixed_unsafe=0.0)
    )
    assert report["decision"] == "no_go_safety"
    assert set(report["safety_vetoes"]) <= set(V19_SCENARIOS) and report["safety_vetoes"]
    assert report["primary_met"] is True


def test_small_gain_is_inconclusive():
    report = analyze(_campaign(adaptive_success=0.25, fixed_success=0.2))
    assert report["decision"] == "inconclusive"


def test_equal_unsafe_rates_do_not_trigger_the_veto():
    report = analyze(
        _campaign(adaptive_success=0.9, fixed_success=0.1, adaptive_unsafe=0.1, fixed_unsafe=0.1)
    )
    assert report["decision"] == "go" and report["safety_vetoes"] == []


def test_moderate_unsafe_increase_triggers_the_veto():
    report = analyze(
        _campaign(adaptive_success=0.9, fixed_success=0.1, adaptive_unsafe=0.16, fixed_unsafe=0.1)
    )
    assert report["decision"] == "no_go_safety"


def test_incomplete_campaign_is_rejected():
    episodes = _campaign(0.9, 0.1)
    with pytest.raises(ValueError):
        analyze(episodes[:-1])
    duplicated = episodes[:-1] + [episodes[0]]
    with pytest.raises(ValueError):
        analyze(duplicated)


def test_go_needs_non_inferiority_not_only_the_absence_of_harm():
    # Deterministic: adaptive unsafe on 250 of 2,000 seeds, fixed on 220 disjoint seeds
    # (+1.5 pp). At this size the paired interval spans roughly [-0.5, +3.5] pp, so
    # no harm is detected and non-inferiority at +3 pp is not shown either.
    episodes = []
    for arm, scenario, seed in v19_cells():
        offset = seed - V19_SEED_START
        if arm == "adaptive_hocbf":
            unsafe = offset < 250
        elif arm == "fixed_hocbf":
            unsafe = 1000 <= offset < 1220
        else:
            unsafe = False
        success = arm.startswith("adaptive")
        episodes.append(_episode(arm, scenario, seed, success=success, unsafe=unsafe))
    report = analyze(episodes)
    assert report["safety_vetoes"] == []
    assert report["primary_met"] is True
    assert report["decision"] == "inconclusive"
    assert any("non-inferiority" in reason for reason in report["inconclusive_reasons"])


def test_episode_log_round_trips_through_the_loader(tmp_path):
    episodes = _campaign(0.9, 0.1)[:5]
    episodes[0] = PilotEpisode(**{**asdict(episodes[0]), "prediction_error_rms": (0.1, 0.2)})
    path = tmp_path / "episodes.jsonl"
    path.write_text("".join(json.dumps(asdict(e)) + "\n" for e in episodes))
    assert load_episodes(path) == episodes
    path.write_text(json.dumps({"arm": "fixed"}) + "\n")
    with pytest.raises(ValueError):
        load_episodes(path)
