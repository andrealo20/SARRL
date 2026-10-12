import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from sarrl.evaluation.adaptive_campaign import load_episodes
from sarrl.evaluation.adaptive_pilot import PilotEpisode
from sarrl.evaluation.margin_campaign import (
    V22_ARM_LABELS,
    V22_CONSTANT,
    V22_COVARIANCE,
    V22_DECISION_EPISODES,
    V22_DECISION_SEED_START,
    V22_MARGINS,
    V22_NONE,
    V22_REPRODUCTION_ARMS,
    V22_REPRODUCTION_EPISODES,
    V22_REPRODUCTION_REFERENCE,
    V22_SCENARIOS,
    analyze,
    load_reference,
    margin_diagnostics,
    reproduction_check,
    v22_cells,
    v22_config,
    v22_decision_cells,
    v22_protocol_dict,
    v22_reproduction_cells,
    v22_seeds,
)

# Success and unsafe-episode probabilities per arm, mirroring the pilot.
SUCCESS = {V22_NONE: 0.91, V22_COVARIANCE: 0.90, V22_CONSTANT: 0.90}
UNSAFE = {V22_NONE: 0.13, V22_COVARIANCE: 0.01, V22_CONSTANT: 0.04}
MEAN_MARGIN = {V22_COVARIANCE: (4.0, 2.0), V22_CONSTANT: (2.0, 2.0)}
# Fields the retained v2.1 rows predate.
ADDED_AFTER_V21 = (
    "certificate_margin",
    "margin_steps",
    "margin_reduced_steps",
    "margin_dropped_steps",
    "margin_mean_command",
)


def _episode(arm, scenario, seed, success, unsafe=False):
    margined = arm != V22_NONE
    return PilotEpisode(
        arm=arm,
        scenario=scenario,
        seed=seed,
        origin="official",
        plant="mujoco",
        outcome="success" if success else "timeout",
        steps=80 if success else 250,
        final_distance=0.04 if success else 0.5,
        reward=-1.0,
        max_speed=1.0,
        max_command_torque=10.0,
        fault_seen=scenario == "motor_fault",
        success=success,
        unsafe_episode=unsafe,
        normalized_violation_max=0.1 if unsafe else 0.0,
        safety_infeasible=False,
        safety_intervention_fraction=0.5 if margined else 0.4,
        true_delay=1,
        selected_lag=1,
        converged_at_step=None,
        prediction_error_rms=(0.1, 0.2),
        estimated_parameters=[[1.0] * 7, [1.0] * 7],
        true_parameters=[[1.0] * 7, [1.0] * 7],
        selected_time_constant=0.03,
        true_time_constant=0.02,
        true_armature=0.05,
        control_steps=100,
        model_fallback_steps=0,
        joint_velocity_violation_max_rad_s=0.6 if unsafe else 0.0,
        certificate_margin=V22_MARGINS[arm],
        margin_steps=100 if margined else 0,
        margin_reduced_steps=20 if margined else 0,
        margin_dropped_steps=1 if margined and seed % 10 == 0 else 0,
        margin_mean_command=MEAN_MARGIN.get(arm),
    )


def _campaign(success=SUCCESS, unsafe=UNSAFE):
    episodes = []
    for arm, scenario, seed, _ in v22_cells():
        if seed < V22_DECISION_SEED_START:
            episodes.append(_episode(arm, scenario, seed, success=seed % 2 == 0))
            continue
        # One draw per seed shared by every arm: outcomes are comonotone across
        # arms, as paired episodes on the same plant draw are in practice.
        draw = np.random.default_rng(seed).random(2)
        episodes.append(
            _episode(
                arm,
                scenario,
                seed,
                success=bool(draw[0] < success[arm]),
                unsafe=bool(draw[1] < unsafe[arm]),
            )
        )
    return episodes


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / "episodes.jsonl"
    lines = []
    for arm in V22_REPRODUCTION_ARMS:
        for scenario in V22_SCENARIOS:
            for seed in v22_seeds("reproduction"):
                record = asdict(_episode(arm, scenario, seed, seed % 2 == 0))
                for name in ADDED_AFTER_V21:
                    record.pop(name)
                lines.append(json.dumps(record))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_reproduction_block_precedes_every_decision_cell():
    cells = list(v22_cells())
    reproduction = list(v22_reproduction_cells())
    decision = list(v22_decision_cells())
    assert cells == reproduction + decision
    assert len(reproduction) == len(V22_REPRODUCTION_ARMS) * V22_REPRODUCTION_EPISODES * 3 == 300
    assert len(decision) == len(V22_ARM_LABELS) * V22_DECISION_EPISODES * 3 == 17100
    assert len(set(cells)) == len(cells)
    assert set(v22_seeds("decision")) == set(range(56200, 58100))
    assert set(v22_seeds("reproduction")) == set(range(54200, 54300))
    assert all(seed < V22_DECISION_SEED_START for _, _, seed, _ in reproduction)
    assert all(seed >= V22_DECISION_SEED_START for _, _, seed, _ in decision)
    with pytest.raises(ValueError):
        v22_seeds("descriptive")


def test_protocol_is_serialisable_and_fixes_the_margins():
    protocol = v22_protocol_dict()
    json.dumps(protocol)
    assert protocol["margins"] == {V22_NONE: 0.0, V22_COVARIANCE: 0.25, V22_CONSTANT: 2.0}
    assert protocol["decision"]["order"] == ["estimator_validity", "statements"]
    statements = protocol["statements"]
    assert statements["margin_removes_violations"]["upper_95_below"] == -0.05
    assert statements["no_material_success_cost"]["lower_95_above"] == -0.03
    assert statements["covariance_shape_matters"]["scenarios"] == ["id_reference"]
    assert protocol["seeds"]["unopened_range"] == [56200, 58099]
    assert v22_config().actuator_time_constants == (0.0, 0.01, 0.03, 0.06)


def test_registry_reserves_the_decision_block_once():
    registry = json.loads(Path("docs/seed_registry.json").read_text())
    reserved = [r for r in registry["ranges"] if (r["low"], r["high"]) == (56200, 58099)]
    assert [r["status"] for r in reserved] == ["reserved"]
    others = [r for r in registry["ranges"] if r not in reserved]
    assert not any(r["low"] <= 58099 and 56200 <= r["high"] for r in others)


def test_confirmed_when_the_three_statements_hold(reference):
    report = analyze(_campaign(), reference)
    assert report["decision"] == "confirmed"
    assert report["failed_statements"] == []
    assert report["reproduction_check"]["matched"] == 300
    assert "certificate_margin" not in report["reproduction_check"]["fields"]
    margin = report["contrasts"]["id_reference"]["covariance_minus_none"]
    assert margin["unsafe_episode"]["ci95_high"] < -0.05
    assert margin["success"]["ci95_low"] > -0.03
    shape = report["contrasts"]["id_reference"]["covariance_minus_constant"]
    assert shape["unsafe_episode"]["difference"] == pytest.approx(-0.03, abs=0.01)
    assert set(report["contrasts"]["ood_compound"]) == {
        "covariance_minus_none",
        "constant_minus_none",
        "covariance_minus_constant",
    }
    conditional = report["severity_conditional"]["id_reference"][V22_NONE]
    assert conditional["joint_velocity_violation_max_rad_s"]["mean"] == pytest.approx(0.6)
    diagnostics = report["margin_diagnostics"][f"{V22_COVARIANCE}/motor_fault"]
    assert diagnostics["mean_enforced_margin_per_joint"] == [4.0, 2.0]
    assert diagnostics["halved_step_fraction"] == pytest.approx(0.2)
    assert report["cell_summary"][f"{V22_CONSTANT}/ood_compound"]["episodes"] == 1900


def test_shape_statement_fails_alone_when_the_two_margins_tie(reference):
    tied = dict(UNSAFE, **{V22_CONSTANT: UNSAFE[V22_COVARIANCE]})
    report = analyze(_campaign(unsafe=tied), reference)
    assert report["decision"] == "margin_confirmed_shape_not"
    assert report["failed_statements"] == ["covariance_shape_matters"]


def test_a_success_cost_or_a_small_reduction_is_not_confirmed(reference):
    costly = dict(SUCCESS, **{V22_COVARIANCE: 0.85})
    report = analyze(_campaign(success=costly), reference)
    assert report["decision"] == "not_confirmed"
    assert report["failed_statements"] == ["no_material_success_cost"]
    # A point estimate past the margin with an upper bound short of it does not pass.
    weak = dict(UNSAFE, **{V22_COVARIANCE: 0.075, V22_CONSTANT: 0.10})
    report = analyze(_campaign(unsafe=weak), reference)
    assert "margin_removes_violations" in report["failed_statements"]
    assert report["decision"] == "not_confirmed"


def test_guarded_estimates_in_one_cell_force_inconclusive(reference):
    episodes = [
        replace(e, model_fallback_steps=50)
        if e.arm == V22_CONSTANT
        and e.scenario == "ood_compound"
        and e.seed % 3 == 0
        and e.seed >= V22_DECISION_SEED_START
        else e
        for e in _campaign()
    ]
    report = analyze(episodes, reference)
    assert report["decision"] == "inconclusive"
    assert report["invalid_estimator_cells"] == [f"{V22_CONSTANT}/ood_compound"]


def test_reordered_mislabelled_or_unreproduced_campaigns_are_refused(reference):
    episodes = _campaign()
    with pytest.raises(ValueError, match="planned cells"):
        analyze(episodes[1:] + episodes[:1], reference)
    wrong = [
        replace(e, certificate_margin=0.5) if e.arm == V22_COVARIANCE and e.seed == 56200 else e
        for e in episodes
    ]
    with pytest.raises(ValueError, match="wrong margin"):
        analyze(wrong, reference)
    broken = [
        replace(e, true_parameters=[[1.0] * 7, [1.1] * 7]) if e.seed == 54200 else e
        for e in episodes
    ]
    with pytest.raises(ValueError, match="does not reproduce"):
        analyze(broken, reference)
    check = reproduction_check(broken, reference)
    assert check["mismatches"][0][3] == "true_parameters"


def test_margin_diagnostics_weight_by_margin_steps():
    rows = [
        replace(_episode(V22_COVARIANCE, "id_reference", 1, True), margin_mean_command=(1.0, 1.0)),
        replace(
            _episode(V22_COVARIANCE, "id_reference", 2, True),
            margin_steps=300,
            margin_mean_command=(5.0, 3.0),
        ),
    ]
    diagnostics = margin_diagnostics(rows)
    assert diagnostics["mean_enforced_margin_per_joint"] == [4.0, 2.5]
    assert diagnostics["margin_steps"] == 400
    assert diagnostics["dropped_step_fraction"] == 0.0
    empty = margin_diagnostics([_episode(V22_NONE, "id_reference", 1, True)])
    assert empty["mean_enforced_margin_per_joint"] is None
    assert empty["halved_step_fraction"] is None


def test_episode_log_round_trips_the_margin_fields(tmp_path):
    episode = _episode(V22_COVARIANCE, "motor_fault", 56201, True)
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps(asdict(episode)) + "\n")
    assert load_episodes(path) == [episode]


def test_reference_holds_the_retained_rows():
    retained = Path(V22_REPRODUCTION_REFERENCE)
    if not retained.exists():
        pytest.skip("retained v2.1 evidence unavailable")
    reference = load_reference(retained)
    assert len(reference) == 300
    assert all(row["plant"] == "mujoco" for row in reference.values())
    assert not set(ADDED_AFTER_V21) & set(next(iter(reference.values())))


def test_current_code_reproduces_retained_rows_field_by_field():
    """A small slice of the reproduction block against the retained v2.1 log."""
    pytest.importorskip("mujoco")
    retained = Path(V22_REPRODUCTION_REFERENCE)
    if not retained.exists():
        pytest.skip("retained v2.1 evidence unavailable")
    from sarrl.evaluation.adaptive_pilot import run_case
    from sarrl.evaluation.factorial_campaign import _values_match
    from sarrl.evaluation.margin_campaign import V22_COMPENSATE_DELAY, V22_PLANT_OPTIONS

    reference = load_reference(retained)
    for scenario in V22_SCENARIOS:
        seed = v22_seeds("reproduction")[0]
        episode = run_case(
            V22_NONE,
            scenario,
            seed,
            "official",
            v22_config(),
            V22_COMPENSATE_DELAY,
            "mujoco",
            V22_PLANT_OPTIONS,
            certificate_margin=V22_MARGINS[V22_NONE],
        )
        ours = asdict(episode)
        for name, value in reference[(V22_NONE, scenario, seed)].items():
            assert _values_match(ours[name], value), (scenario, name)
