import copy

import numpy as np
import pytest

from sarrl.envs import DomainRandomization, PlanarReachEnv
from sarrl.envs.adaptive_margin_residual import AdaptiveMarginResidualEnv
from sarrl.evaluation.adaptive_pilot import MeasuredState, NominalStack, build_arm
from sarrl.evaluation.margin_campaign import v22_config


def environment():
    return AdaptiveMarginResidualEnv(
        PlanarReachEnv(
            mode="torque",
            randomization=DomainRandomization(
                mass_fraction=0.15,
                friction_fraction=0.3,
                motor_gain_fraction=0.15,
                payload_range=(0.0, 1.0),
                action_delay_max=2,
                sensor_noise_std=0.001,
            ),
        ),
        v22_config(),
    )


def test_zero_residual_matches_retained_covariance_stack_commands_and_physics():
    wrapped = environment()
    wrapped.reset(seed=9803000)
    base = PlanarReachEnv(mode="torque", randomization=wrapped.env.randomization)
    base.reset(seed=9803000)
    controller, certificate, estimator = build_arm("adaptive_hocbf_margin", v22_config(), 0.25)
    controller.reset()
    stack = NominalStack(controller, certificate, compensate_delay=True, estimator=estimator)
    measured = MeasuredState(base)
    observation = base._observation()
    for _ in range(30):
        state = measured()
        command = stack.command(observation, state, base.q_des)
        result = wrapped.step(np.zeros(2))
        np.testing.assert_allclose(result[-1]["projected_torque"], command.torque, atol=2e-6)
        assert result[-1]["safety_infeasible"] == (not command.executable)
        if not command.executable:
            break
        observation, reward, terminated, truncated, info = base.step_torque(
            command.torque, baseline=command.baseline_torque
        )
        controller.observe(state, measured(), command.torque)
        np.testing.assert_array_equal(wrapped.env.state, base.state)
        assert result[1:4] == (reward, terminated, truncated)
        if terminated or truncated:
            break


def test_margin_checkpoint_restores_next_transition_and_rejects_different_kappa():
    first = environment()
    first.reset(seed=9803001)
    for _ in range(4):
        result = first.step(np.zeros(2))
        if result[2] or result[3]:
            break
    state = copy.deepcopy(first.state_dict())
    second = environment()
    second.load_state_dict(state)
    expected, actual = first.step(np.array([0.05, -0.05])), second.step(np.array([0.05, -0.05]))
    np.testing.assert_array_equal(expected[0], actual[0])
    assert expected[1:4] == actual[1:4]
    np.testing.assert_array_equal(first.safety_filter.applied, second.safety_filter.applied)
    assert first.safety_filter.margin_steps == second.safety_filter.margin_steps
    different = AdaptiveMarginResidualEnv(
        PlanarReachEnv(mode="torque", randomization=first.env.randomization),
        v22_config(),
        certificate_kappa=0.5,
    )
    with pytest.raises(ValueError, match="configuration"):
        different.load_state_dict(state)


def test_resumed_sac_training_matches_uninterrupted_parameters(tmp_path):
    pytest.importorskip("mujoco")
    import torch

    from sarrl.rl.adaptive_residual_training import ResidualTrainingConfig, train

    cfg = ResidualTrainingConfig(
        steps=12, start_steps=0, hidden=(8, 8), batch_size=4, replay_capacity=32, checkpoint_every=6
    )
    signature = {"purpose": "checkpoint continuation test"}
    train(9803002, tmp_path / "continuous", signature, config=cfg)
    train(9803002, tmp_path / "resumed", signature, config=cfg, stop_after=6)
    train(9803002, tmp_path / "resumed", signature, config=cfg)
    first = torch.load(tmp_path / "continuous/final.pt", weights_only=False, map_location="cpu")
    second = torch.load(tmp_path / "resumed/final.pt", weights_only=False, map_location="cpu")
    assert first["update_steps"] == second["update_steps"]
    for name in ("actor", "q1", "q2", "q1_target", "q2_target"):
        for key, value in first[name].items():
            torch.testing.assert_close(value, second[name][key], rtol=0, atol=0)
    assert (tmp_path / "continuous/episodes.csv").read_bytes() == (
        tmp_path / "resumed/episodes.csv"
    ).read_bytes()


def test_zero_policy_inference_preserves_retained_margin_episode():
    pytest.importorskip("mujoco")
    from dataclasses import asdict

    from sarrl.evaluation.adaptive_pilot import run_case
    from sarrl.evaluation.margin_campaign import V22_PLANT_OPTIONS
    from sarrl.evaluation.panda_adaptation import reproduction_mismatches
    from sarrl.evaluation.planar_residual import ARMS, run_episode

    class ZeroPolicy:
        def act(self, observation, deterministic=False):
            return np.zeros(2)

    expected = asdict(
        run_case(
            "adaptive_hocbf_margin",
            "id_reference",
            9803002,
            "heldout",
            v22_config(),
            True,
            "mujoco",
            V22_PLANT_OPTIONS,
            certificate_margin=0.25,
        )
    )
    actual, trace = run_episode(
        9803002, "id_reference", ARMS[2], policy=ZeroPolicy(), retain_trace=True
    )
    actual["arm"], actual["residual_rms"] = expected["arm"], None
    assert not reproduction_mismatches(actual, expected)
    assert trace and all(frame["raw_residual"] == [0.0, 0.0] for frame in trace)


def test_crossed_bootstrap_cannot_hide_safety_cost_or_select_one_model():
    from sarrl.evaluation.planar_residual import ARMS, SCENARIOS, analyze

    rows = [
        {
            "scenario": c,
            "seed": s,
            "arm": a,
            "success": a.startswith("residual_"),
            "unsafe_episode": False,
            "safety_infeasible": False,
            "final_distance": 0.1,
        }
        for c in SCENARIOS
        for s in (9803000, 9803001)
        for a in ARMS
    ]
    result = analyze(rows, bootstrap_draws=30)
    assert result["decision"]["residual_advances"]
    assert result["primary_success"]["lower95"] == result["primary_success"]["upper95"] == 1.0
    for row in rows:
        row["unsafe_episode"] = row["arm"].startswith("residual_")
    result = analyze(rows, bootstrap_draws=30)
    assert not result["decision"]["residual_advances"]
    assert {f["criterion"] for f in result["decision"]["failures"]} == {"unsafe_margin"}
    with pytest.raises(ValueError, match="factorial"):
        analyze(rows[:-1], bootstrap_draws=30)


def test_crossed_resampling_matches_explicit_model_task_oracle():
    from sarrl.evaluation.planar_residual import ARMS, BOOTSTRAP_SEED, SCENARIOS, analyze

    differences = np.array([[1, 0], [1, -1], [0, -1], [1, 0], [0, 0]])
    rows = []
    for condition in SCENARIOS:
        for index, seed in enumerate((9803000, 9803001)):
            for arm in ARMS:
                base = index == 1
                success = (
                    base
                    if arm in ARMS[:2]
                    else bool(base + differences[ARMS.index(arm) - 2, index])
                )
                rows.append(
                    {
                        "scenario": condition,
                        "seed": seed,
                        "arm": arm,
                        "success": success,
                        "unsafe_episode": False,
                        "safety_infeasible": False,
                        "final_distance": 0.1,
                    }
                )
    actual = analyze(rows, bootstrap_draws=50)["primary_success"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    models = rng.integers(0, 5, size=(50, 5))
    tasks = rng.integers(0, 2, size=(50, 2))
    values = [
        sum(differences[m, t] for m in model for t in task) / 10
        for model, task in zip(models, tasks, strict=True)
    ]
    np.testing.assert_allclose(
        [actual["lower95"], actual["upper95"]], np.quantile(values, [0.025, 0.975])
    )


def test_study_runner_refuses_alternate_outputs_and_invalid_workers(tmp_path):
    from tools.run_planar_residual import main

    for args in (["--output", str(tmp_path / "other")], ["--workers", "6"]):
        with pytest.raises(SystemExit) as error:
            main(args)
        assert error.value.code == 2
    assert not (tmp_path / "other").exists()


def test_study_commands_help_without_optional_simulator():
    import subprocess
    import sys

    for module in ("tools.run_planar_residual", "tools.audit_planar_residual"):
        script = (
            f"import sys; sys.modules['mujoco']=None; from {module} import main; main(['--help'])"
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
