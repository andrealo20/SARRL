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


def test_measured_state_is_sampled_once_per_step_and_noisy():
    from dataclasses import replace

    from sarrl.envs import PlanarReachEnv
    from sarrl.evaluation.adaptive_pilot import MeasuredState, PlantOptions, make_env
    from sarrl.evaluation.planar_v13 import v13_scenarios

    spec = {s.key: s for s in v13_scenarios()}["id_reference"]
    env = make_env("analytical", spec, PlantOptions(sensor_noise_std=1e-2))
    assert isinstance(env, PlanarReachEnv)
    env.reset(seed=9803006)
    measured = MeasuredState(env)
    first = measured()
    assert np.array_equal(first, measured()), "same step, same measurement"
    assert not np.array_equal(first, env.state), "measurement carries noise"
    env.step_torque(np.zeros(2))
    assert not np.array_equal(first, measured()), "new step, new measurement"
    with pytest.raises(ValueError):
        make_env("analytical", spec, PlantOptions(armature=0.05))
    assert replace(spec.randomization, sensor_noise_std=0.0) == spec.randomization


class _ZeroPolicy:
    def act(self, observation, deterministic=True):
        return np.zeros(2)


class _ConstantPolicy:
    def act(self, observation, deterministic=True):
        return np.array([0.5, -0.5])


def test_residual_arm_with_a_zero_policy_matches_the_nominal_arm():
    config = AdaptiveNominalConfig()
    nominal = run_case("adaptive_hocbf", "id_reference", 9801800, "historical", config, True)
    residual = run_case(
        "adaptive_hocbf_residual",
        "id_reference",
        9801800,
        "historical",
        config,
        True,
        policy=_ZeroPolicy(),
    )
    assert residual.arm == "adaptive_hocbf_residual"
    assert residual.residual_rms == 0.0 and nominal.residual_rms is None
    assert residual.steps == nominal.steps and residual.outcome == nominal.outcome
    assert residual.final_distance == nominal.final_distance


def test_residual_arm_records_the_residual_and_needs_a_policy():
    config = AdaptiveNominalConfig()
    episode = run_case(
        "fixed_hocbf_residual",
        "id_reference",
        9801800,
        "historical",
        config,
        policy=_ConstantPolicy(),
    )
    assert episode.residual_rms == pytest.approx(np.sqrt(2 * 4.0**2))
    assert summarize([episode])["all/id_reference/fixed_hocbf_residual"][
        "mean_residual_rms"
    ] == pytest.approx(episode.residual_rms)
    with pytest.raises(ValueError, match="residual arm"):
        run_case("adaptive_hocbf_residual", "id_reference", 9801800, "historical", config)
    with pytest.raises(ValueError, match="residual arm"):
        run_case(
            "adaptive_hocbf", "id_reference", 9801800, "historical", config, policy=_ZeroPolicy()
        )


def test_certificate_only_estimator_guards_once_per_step_and_learns_from_the_executed_command():
    from unittest.mock import patch

    from sarrl.evaluation.adaptive_pilot import build_stack

    config = AdaptiveNominalConfig()
    controller, stack = build_stack("fixed_hocbf_adaptivemodel", config, True)
    assert isinstance(controller, ComputedTorqueController)
    assert stack.estimator is not None and stack.estimator is not controller
    assert stack.safety_filter.model.controller is stack.estimator
    assert stack.compensate_delay is False
    state = np.array([0.1, -0.2, 0.3, 0.4])
    with patch.object(stack.estimator, "begin_step", wraps=stack.estimator.begin_step) as guard:
        result = stack.command(None, state, np.array([0.5, 0.5]))
    assert guard.call_count == 1
    np.testing.assert_array_equal(guard.call_args[0][0], state[:2])
    # The pre-filter candidate is the fixed computed torque at the current state,
    # untouched by the estimate; the executed command may differ through the filter.
    expected = controller.command(state[:2], state[2:], np.array([0.5, 0.5]))
    np.testing.assert_allclose(result.baseline_torque, expected)
    from sarrl.controllers import AdaptiveNominalController
    from sarrl.evaluation.adaptive_pilot import MeasuredState, PlantOptions
    from sarrl.safety import HOCBFSafetyFilter

    # With sensor noise on, the estimator must see the cached noisy measurement
    # before and after each step (the same sample every consumer sees) and the
    # torque the filter returned, which is what the plant executed.
    calls, measured, executed = [], [], []
    observe = AdaptiveNominalController.observe
    measure = MeasuredState.__call__
    project = HOCBFSafetyFilter.filter

    def spy_observe(self, before, after, torque):
        calls.append((np.array(before), np.array(after), np.array(torque)))
        return observe(self, before, after, torque)

    def spy_measure(self, env=None):
        value = measure(self, env)
        if not measured or not np.array_equal(measured[-1], value):
            measured.append(value.copy())
        return value

    def spy_filter(self, state, candidate, obstacles=()):
        result = project(self, state, candidate, obstacles)
        executed.append(np.array(result.torque))
        return result

    with (
        patch.object(AdaptiveNominalController, "observe", spy_observe),
        patch.object(MeasuredState, "__call__", spy_measure),
        patch.object(HOCBFSafetyFilter, "filter", spy_filter),
    ):
        episode = run_case(
            "fixed_hocbf_adaptivemodel",
            "id_reference",
            9801800,
            "historical",
            config,
            True,
            options=PlantOptions(sensor_noise_std=1e-3),
        )
    assert episode.selected_lag is not None and len(calls) == episode.steps
    assert len(measured) == episode.steps + 1 and len(executed) == episode.steps
    for step, (before, after, torque) in enumerate(calls):
        np.testing.assert_array_equal(before, measured[step])
        np.testing.assert_array_equal(after, measured[step + 1])
        np.testing.assert_array_equal(torque, executed[step])
    # The measurements are noisy, so they differ from any exact plant state.
    assert not np.array_equal(measured[0], measured[1])


def test_fixed_certificate_arm_drives_the_identified_command_with_a_nominal_certificate():
    from sarrl.evaluation.adaptive_pilot import build_stack

    controller, stack = build_stack("adaptive_hocbf_fixedmodel", AdaptiveNominalConfig(), True)
    assert stack.estimator is None  # the controller is the estimator
    assert stack.compensate_delay is True
    assert isinstance(stack.safety_filter.model, PlanarArm)
    identified, identified_stack = build_stack("adaptive_hocbf", AdaptiveNominalConfig(), True)
    assert identified_stack.safety_filter.model.controller is identified
