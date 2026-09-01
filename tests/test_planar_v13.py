from sarrl.evaluation import (
    V13_CONDITIONS,
    V13_EPISODES,
    V13_EVALUATION_SEED,
    V13_TRAINING_SEEDS,
    v13_protocol_dict,
    v13_scenarios,
)
from tools.run_planar_v13 import run_campaign, validate_inputs


def test_v13_protocol_freezes_disjoint_evaluation_population():
    protocol = v13_protocol_dict()

    assert V13_EVALUATION_SEED == 50_000
    assert V13_EPISODES == 100
    assert V13_TRAINING_SEEDS == (0, 1, 2, 3, 4)
    assert protocol["evaluation"]["seed_start"] == 50_000
    assert protocol["evaluation"]["episodes_per_model_scenario"] == 100
    assert protocol["evaluation"]["paired_across_scenarios"] is True
    assert 50_000 > 40_099


def test_v13_conditions_reuse_every_retained_v12_policy_family():
    assert V13_CONDITIONS == ("A0", "A2", "A3", "A4", "A5", "A6")
    assert v13_protocol_dict()["excluded_condition"] == {
        "key": "A1",
        "reason": "selected policy checkpoints were not retained",
    }


def test_v13_scenarios_include_strict_payload_ood_and_exposed_fault():
    scenarios = {scenario.key: scenario for scenario in v13_scenarios()}

    assert set(scenarios) == {"id_reference", "ood_compound", "motor_fault"}
    assert scenarios["id_reference"].randomization.payload_range == (0.0, 1.0)
    assert scenarios["ood_compound"].randomization.payload_range[0] > 1.0
    assert scenarios["ood_compound"].randomization.action_delay_max == 3
    assert scenarios["motor_fault"].fault is not None
    assert scenarios["motor_fault"].fault.start_step == 20
    assert scenarios["motor_fault"].fault.motor_gain_multiplier == (1.0, 0.55)


def test_v13_statistics_are_model_level_and_scenario_paired():
    statistics = v13_protocol_dict()["statistics"]

    assert statistics["multi_seed_spread"] == "sample_sd_ddof_1"
    assert statistics["paired_scenario_difference"] == "paired_bootstrap_95"
    assert statistics["bootstrap_samples"] == 10_000


def test_v13_input_validation_requires_five_of_each_artifact(tmp_path):
    try:
        validate_inputs(
            tmp_path,
            [0, 1, 2, 3, 4],
            [],
            [tmp_path / "a3.pt"] * 5,
            [tmp_path / "context.pt"] * 5,
            [tmp_path / "ensemble.pt"] * 5,
        )
    except ValueError as exc:
        assert "A2 requires exactly 5 checkpoints" in str(exc)
    else:
        raise AssertionError("missing A2 artifacts must be rejected")


def test_v13_a0_smoke_retains_paired_scenario_outputs(tmp_path):
    run_campaign(
        root=tmp_path,
        out=tmp_path / "out",
        inputs=[],
        conditions=["A0"],
        scenario_keys=["id_reference", "ood_compound"],
        evaluation_seed=50_000,
        episodes=2,
    )

    out = tmp_path / "out"
    assert len((out / "heldout_episodes.csv").read_text().splitlines()) == 5
    assert len((out / "summary.csv").read_text().splitlines()) == 3
    assert len((out / "robustness_deltas.csv").read_text().splitlines()) == 2
    assert '"A0"' in (out / "aggregate.json").read_text()
