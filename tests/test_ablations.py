import pytest

from tools.run_planar_ablations import CONDITIONS, build_protocol


def _protocol(**overrides):
    kwargs = {
        "seeds": [0, 1, 2, 3, 4],
        "steps": 200_000,
        "validation_seed": 20_000,
        "validation_episodes": 30,
        "heldout_seed": 40_000,
        "heldout_episodes": 100,
    }
    kwargs.update(overrides)
    return build_protocol(**kwargs)


def test_v12_ablation_matrix_is_fixed_and_complete():
    assert [condition.key for condition in CONDITIONS] == [
        "A0",
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
    ]
    assert CONDITIONS[0].label == "Computed torque"
    assert CONDITIONS[2].label == "Residual SAC"
    assert CONDITIONS[-1].label == "Full adaptive stack"


def test_v12_protocol_freezes_retained_training_and_evaluation_seeds():
    protocol = _protocol()

    assert protocol["training"]["seeds"] == [0, 1, 2, 3, 4]
    assert protocol["training"]["steps_per_seed"] == 200_000

    assert protocol["validation"]["seed_start"] == 20_000
    assert protocol["validation"]["episodes"] == 30

    assert protocol["heldout"]["seed_start"] == 40_000
    assert protocol["heldout"]["episodes_per_policy"] == 100


def test_v12_protocol_freezes_domain_randomization():
    dr = _protocol()["domain_randomization"]

    assert dr == {
        "mass_fraction": 0.15,
        "friction_fraction": 0.30,
        "motor_gain_fraction": 0.15,
        "payload_range": [0.0, 1.0],
        "action_delay_max": 2,
    }


def test_v12_protocol_rejects_validation_heldout_leakage():
    with pytest.raises(
        ValueError,
        match="validation and held-out seed ranges must not overlap",
    ):
        _protocol(
            validation_seed=20_000,
            validation_episodes=100,
            heldout_seed=20_050,
        )


def test_v12_protocol_rejects_duplicate_training_seeds():
    with pytest.raises(ValueError, match="training seeds must be unique"):
        _protocol(seeds=[0, 1, 1, 2])


def test_v12_statistics_require_sample_sd_and_paired_comparison():
    statistics = _protocol()["statistics"]

    assert statistics["multi_seed_spread"] == "sample_sd_ddof_1"
    assert statistics["episode_success_interval"] == "wilson_95"
    assert statistics["paired_comparison"] == "paired_bootstrap_95"
