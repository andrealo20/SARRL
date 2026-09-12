import numpy as np
import pytest

from sarrl.controllers import AdaptiveNominalConfig, ComputedTorqueController
from sarrl.dynamics import PlanarArm
from sarrl.evaluation.adaptive_pilot import (
    ARMS,
    NominalStack,
    pilot_cases,
    run_case,
    summarize,
)
from sarrl.safety import HOCBFSafetyFilter, SafetyConfig


def test_pilot_cases_keep_historical_seeds_first_and_fresh_seeds_apart():
    cases = pilot_cases(3)
    assert len(cases) == 15
    assert cases[0] == ("id_reference", 9801800, "historical")
    assert all(origin in ("historical", "fresh") for _, _, origin in cases)
    seeds = [seed for _, seed, _ in cases]
    assert len(set(seeds)) == len(seeds)
    assert len(pilot_cases(3, historical=False)) == 9
    with pytest.raises(ValueError):
        pilot_cases(101)


def test_nominal_stack_without_filter_returns_the_clipped_baseline():
    stack = NominalStack(ComputedTorqueController(PlanarArm()))
    state = np.array([0.3, -0.2, 0.0, 0.0])
    result = stack.command(None, state, np.array([1.0, 0.5]))
    assert result.executable and not result.safety_certified
    assert np.all(np.abs(result.torque) <= 40.0)
    np.testing.assert_array_equal(result.raw_residual, np.zeros(2))


def test_nominal_stack_with_filter_certifies_or_aborts():
    nominal = PlanarArm()
    stack = NominalStack(
        ComputedTorqueController(nominal), HOCBFSafetyFilter(nominal, SafetyConfig())
    )
    result = stack.command(None, np.array([0.3, -0.2, 0.0, 0.0]), np.array([1.0, 0.5]))
    assert result.executable and result.safety_certified
    assert stack.config.require_safety


def test_fixed_filtered_arm_reproduces_the_retained_null_residual_stop():
    # Historical Z episode on ID 9801800: timeout at 250 steps, 0.593683 m.
    config = AdaptiveNominalConfig()
    episode = run_case("fixed_hocbf", "id_reference", 9801800, "historical", config)
    assert episode.outcome == "timeout" and episode.steps == 250
    assert abs(episode.final_distance - 0.593683) < 1e-5
    assert episode.selected_lag is None and episode.prediction_error_rms is None


def test_adaptive_filtered_arm_reaches_the_same_target():
    config = AdaptiveNominalConfig()
    episode = run_case("adaptive_hocbf", "id_reference", 9801800, "historical", config)
    assert episode.outcome == "success"
    assert episode.selected_lag == episode.true_delay == 2
    assert max(episode.prediction_error_rms) < 1.0
    assert not episode.unsafe_episode


def test_summary_counts_every_arm():
    config = AdaptiveNominalConfig()
    episodes = [run_case(arm, "motor_fault", 9802001, "historical", config) for arm in ARMS]
    table = summarize(episodes)
    assert set(table) == {f"{o}/motor_fault/{a}" for o in ("historical", "all") for a in ARMS}
    row = table["historical/motor_fault/adaptive_hocbf"]
    assert row["episodes"] == 1 and row["lag_correct"] in (0, 1)
    assert table["historical/motor_fault/fixed"]["lag_correct"] is None
