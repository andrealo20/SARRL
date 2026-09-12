from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from sarrl.controllers import ComputedTorqueController
from sarrl.dynamics import PlanarArm
from sarrl.envs import PlanarReachEnv
from sarrl.evaluation import evaluate_safety_episodes, planar_safety_config, v13_scenarios
from sarrl.evaluation.residual_diagnosis import (
    ZeroResidualPolicy,
    compare_historical,
    describe,
    evaluate_recorded,
    exact,
    invalid_data,
    paired_contrast,
    physical_balance,
    plain,
)
from sarrl.runtime import ControlStackConfig, SARRLControlStack
from sarrl.safety import HOCBFSafetyFilter


def test_early_success_and_noise_state():
    from sarrl.envs import DomainRandomization

    class SuccessEnv(PlanarReachEnv):
        def reset(self, seed=None, target=None):
            super().reset(seed=seed, target=target)
            self.state = np.zeros(4)
            self.q_des = np.zeros(2)
            self.target = self.arm.forward_kinematics(self.state[:2])
            return self._observation(), self._info_base()

    env = SuccessEnv(mode="torque", randomization=DomainRandomization(sensor_noise_std=0.001))
    record = evaluate_recorded(
        ZeroResidualPolicy(), env, seed=78, scenario="engineering", controller="Z"
    )
    assert record["episode"]["steps"] == 1
    assert record["summary"]["outcome"] == "success"
    assert record["trace"][0]["normalized_action"] == [0.0, 0.0]
    # Exactly one new observation draws noise in the one executed step.
    noise_rng = np.random.default_rng()
    noise_rng.bit_generator.state = record["initial"]["noise_rng_state"]
    noise_rng.normal(0.0, 0.001, size=4)
    exact(record["final"]["noise_rng_state"], noise_rng.bit_generator.state, "noise draw count")
    print("Additional unit fixture: 1 attempt in this invocation.")


def test_runner_cpu_actor_without_optimizer_or_cuda(tmp_path, monkeypatch):
    import torch

    from sarrl.rl.networks import SquashedGaussianActor
    from tools.run_residual_diagnosis import FrozenCPUActor

    actor = SquashedGaussianActor(8, 2, (8, 8))
    checkpoint = tmp_path / "synthetic.pt"
    torch.save(
        {
            "checkpoint_version": 1,
            "obs_dim": 8,
            "action_dim": 2,
            "config": {"hidden": [8, 8]},
            "actor": actor.state_dict(),
        },
        checkpoint,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **kw: pytest.fail("optimizer created"))
    rng_before = torch.get_rng_state().clone()
    policy = FrozenCPUActor(checkpoint)
    assert torch.equal(rng_before, torch.get_rng_state())
    assert all(p.device.type == "cpu" and not p.requires_grad for p in policy.actor.parameters())
    observation = np.linspace(-1, 1, 8, dtype=np.float32)
    with torch.no_grad():
        expected = actor.deterministic(torch.as_tensor(observation).unsqueeze(0))[0].numpy()
    assert np.array_equal(policy.act(observation), expected)
    with pytest.raises(ValueError, match="deterministic"):
        policy.act(observation, deterministic=False)


def _mock_result(controller, scenario, seed):
    return {
        "status": "valid",
        "initial": {"state": [0, 0, 0, 0]},
        "initial_observation": [0] * 8,
        "episode": {
            "controller": controller,
            "scenario": scenario,
            "seed": seed,
            "steps": 1,
            "success": False,
            "final_distance": 0.2,
        },
        "safety": {"safety_infeasible": False},
        "summary": {"outcome": "timeout"},
        "trace": [
            {
                "attempt": 0,
                "executed": True,
                "distance_m": 0.2,
                "speed_rad_s": 0.0,
                "command": {"raw_residual": [0, 0], "safety_correction": 0.0},
            }
        ],
    }


@pytest.fixture
def fake_runner(tmp_path, monkeypatch):
    from tools import run_residual_diagnosis as runner

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    official = tmp_path / "results/penalty_ablation"
    (official / "failure_audit").mkdir(parents=True)
    monkeypatch.setattr(runner, "OFFICIAL", official)
    checkpoint = tmp_path / "synthetic.pt"
    checkpoint.write_bytes(b"not a model; loader mocked")
    models = [
        {
            "condition": condition,
            "training_seed": seed,
            "path": "synthetic.pt",
            "sha256": runner.sha(checkpoint),
        }
        for condition in ("P0_inloop_reference", "P1_inloop_half_penalty")
        for seed in range(30, 35)
    ]
    protocol = tmp_path / "protocol.md"
    protocol.write_text("synthetic runner contract")
    plan = {
        "models": models,
        "cases": {k: list(v) for k, v in runner.CASES.items()},
        "protocol_sha256": runner.sha(protocol),
        "amendment_sha256": runner.sha(protocol),
        "execution": {"actor_device": "cuda:0", "synthetic": True},
    }
    historical = [
        _historic(_mock_result(c, s, seed))
        for c, s, seed, model in runner.cell_specs(plan)
        if model is not None
    ]
    runner.write_new(official / "failure_audit/trajectories.json", historical)
    calls = []

    def evaluate(policy, env, *, seed, scenario, controller):
        calls.append((controller, scenario, seed))
        return _mock_result(controller, scenario, seed)

    monkeypatch.setattr(runner, "evaluate_recorded", evaluate)

    class FakeCUDAActor(ZeroResidualPolicy):
        def __init__(self, path):
            pass

        def validate_device(self):
            pass

    monkeypatch.setattr(runner, "FrozenCUDAActor", FakeCUDAActor)
    monkeypatch.setattr(runner, "execution_metadata", lambda: dict(plan["execution"]))

    def forbidden(*args, **kwargs):
        pytest.fail("physical operations are forbidden in runner transaction tests")

    monkeypatch.setattr(PlanarReachEnv, "reset", forbidden)
    monkeypatch.setattr(PlanarReachEnv, "step_torque", forbidden)
    monkeypatch.setattr(PlanarArm, "step_rk4", forbidden)
    # No env.reset or env.step occurs in any of these transaction tests.
    return runner, plan, protocol, tmp_path / "results/residual_diagnosis_test", calls


def test_runner_complete_matrix_and_no_overwrite(fake_runner):
    runner, plan, protocol, output, calls = fake_runner
    runner.execute(plan, output, protocol, protocol)
    assert len(calls) == len(set(calls)) == 66
    assert sum(key[0] == "Z" for key in calls) == 6
    assert len(runner.read(output / "contrasts.json")) == 60
    complete = runner.read(output / "complete.json")
    assert all(
        runner.sha(output / name) == expected for name, expected in complete["hashes"].items()
    )
    with pytest.raises(FileExistsError, match="already complete"):
        runner.execute(plan, output, protocol, protocol)
    assert len(calls) == 66
    with pytest.raises(FileExistsError):
        runner.write_new(output / "summary.json", {})


def test_runner_invalid_cell_cannot_be_retried(fake_runner, monkeypatch):
    runner, plan, protocol, output, calls = fake_runner

    def invalid(*args, **kwargs):
        calls.append("invalid")
        raise ValueError("deliberate numerical mismatch")

    monkeypatch.setattr(runner, "evaluate_recorded", invalid)
    with pytest.raises(ValueError, match="deliberate numerical"):
        runner.execute(plan, output, protocol, protocol)
    assert len(list(output.glob("*.started.json"))) == 1
    assert len(list(output.glob("*.invalid.json"))) == 1
    with pytest.raises(ValueError, match="needs inspection"):
        runner.execute(plan, output, protocol, protocol)
    assert calls == ["invalid"]


def test_runner_interrupted_cell_cannot_be_retried(fake_runner, monkeypatch):
    runner, plan, protocol, output, calls = fake_runner

    def interrupted(*args, **kwargs):
        calls.append("interrupted")
        raise KeyboardInterrupt()

    monkeypatch.setattr(runner, "evaluate_recorded", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runner.execute(plan, output, protocol, protocol)
    with pytest.raises(ValueError, match="needs inspection"):
        runner.execute(plan, output, protocol, protocol)
    assert calls == ["interrupted"]


def test_runner_cli_cannot_execute_implicitly():
    from tools import run_residual_diagnosis as runner

    with pytest.raises(SystemExit) as exc:
        runner.main(["run", "--protocol", "missing.md", "--amendment", "missing.md"])
    assert exc.value.code == 2
    with pytest.raises(ValueError, match="new results"):
        runner.validate_output_path(runner.OFFICIAL)


def test_runner_resumes_only_never_started_cells(fake_runner, monkeypatch):
    runner, plan, protocol, output, calls = fake_runner
    original_specs = runner.cell_specs

    def stop_between_cells(plan):
        yield next(original_specs(plan))
        raise KeyboardInterrupt()

    monkeypatch.setattr(runner, "cell_specs", stop_between_cells)
    with pytest.raises(KeyboardInterrupt):
        runner.execute(plan, output, protocol, protocol)
    assert len(calls) == 1
    monkeypatch.setattr(runner, "cell_specs", original_specs)
    runner.execute(plan, output, protocol, protocol)
    assert len(calls) == len(set(calls)) == 66


class FixedPolicy:
    def __init__(self):
        self.calls = 0

    def act(self, observation, deterministic=True):
        assert deterministic
        self.calls += 1
        return np.array([0.2, -0.1], dtype=np.float32)


class FixtureEnv(PlanarReachEnv):
    """Explicit engineering states, never held-out performance observations."""

    def __init__(self, specification, *, infeasible=False):
        super().__init__(
            mode="torque",
            max_steps=25,
            randomization=specification.randomization,
            fault=specification.fault,
        )
        self.infeasible = infeasible
        self.initial_fixture = None
        self.attempt_rewards = []

    def reset(self, seed=None, target=None):
        super().reset(seed=seed, target=target)
        self.state = np.array([0.1, 0.2, 0.0, 0.0])
        self.q_des = self.state[:2].copy()
        self.target = np.array([1.0, 1.0])
        self.action_delay = 2 if self.randomization.action_delay_max >= 3 or self.fault else 0
        self._command_queue = [np.zeros(2) for _ in range(self.action_delay)]
        if self.infeasible:
            self.state = np.array([3.05, 0.0, 7.0, 0.0])
        observation = self._observation()
        self.initial_fixture = deepcopy(self.state_dict())
        return observation, self._info_base()

    def step_torque(self, *args, **kwargs):
        result = super().step_torque(*args, **kwargs)
        self.attempt_rewards.append(result[1])
        return result


@pytest.fixture(scope="module")
def observations():
    """Four ON/OFF pairs, at most 200 total physical command attempts."""
    specifications = list(v13_scenarios())
    fixtures = [(s, False) for s in specifications] + [(specifications[0], True)]
    collected = []
    total_attempts = 0
    for index, (specification, infeasible) in enumerate(fixtures):
        env_on = FixtureEnv(specification, infeasible=infeasible)
        policy_on = FixedPolicy()
        record = evaluate_recorded(
            policy_on, env_on, seed=71 + index, scenario=specification.key, controller="synthetic"
        )
        env_off = FixtureEnv(specification, infeasible=infeasible)
        policy_off = FixedPolicy()
        nominal = PlanarArm()
        stack = SARRLControlStack(
            ComputedTorqueController(nominal),
            policy_off,
            ControlStackConfig(require_safety=True),
            safety_filter=HOCBFSafetyFilter(nominal, planar_safety_config()),
        )
        rows, diagnostics = evaluate_safety_episodes(
            stack,
            HOCBFSafetyFilter(nominal, planar_safety_config()),
            env_off,
            episodes=1,
            seed=71 + index,
            scenario=specification.key,
            controller="synthetic",
        )
        expected_safety = asdict(diagnostics[0])
        if rows[0].steps == 0:
            expected_safety["executed_constraint_margin_min"] = None
        exact(record["episode"], asdict(rows[0]), "instrumentation episode parity")
        exact(record["safety"], expected_safety, "instrumentation safety parity")
        exact(record["initial"], env_off.initial_fixture, "initial and RNG parity")
        exact(record["final"], env_off.state_dict(), "final and RNG parity")
        exact(env_on.attempt_rewards, env_off.attempt_rewards, "per-step reward parity")
        assert policy_on.calls == policy_off.calls == record["safety"]["command_attempts"]
        total_attempts += policy_on.calls + policy_off.calls
        collected.append(record)
    assert total_attempts <= 200
    print(f"Engineering budget: {total_attempts}/200 attempts; no trained policy loaded.")
    return collected


def test_noninterference_and_delayed_fault_boundary(observations):
    id_record, ood, fault, abort = observations
    assert id_record["initial"]["action_delay"] == 0
    assert ood["initial"]["action_delay"] == 2
    assert fault["episode"]["steps"] == 25
    assert not fault["trace"][19]["after"]["fault_active"]
    boundary = fault["trace"][20]
    assert not boundary["before"]["fault_active"]
    assert boundary["after"]["fault_active"]
    assert np.allclose(
        boundary["after"]["motor_gain"], np.asarray(boundary["before"]["motor_gain"]) * [1.0, 0.55]
    )
    assert abort["summary"]["outcome"] == "abort"
    assert abort["episode"]["steps"] == 0
    assert len(abort["trace"]) == 1
    assert abort["trace"][0]["physical"] is None
    assert abort["initial"] == abort["final"]
    assert all(r["summary"]["tail50"] is None for r in observations)


def test_zero_residual_and_json_fail_closed():
    assert np.array_equal(ZeroResidualPolicy().act(np.ones(8)), np.zeros(2))
    with pytest.raises(ValueError, match="non-finite"):
        plain({"state": np.array([np.nan])})


def _historic(record):
    return {
        "episode": deepcopy(record["episode"]),
        "safety": deepcopy(record["safety"]),
        "tail50_mean_distance_m": 999.0,
        "trace": [
            {
                "attempt": t["attempt"],
                "executed": t["executed"],
                "distance_m": t["distance_m"],
                "speed_rad_s": t["speed_rad_s"],
                "raw_residual": deepcopy(t["command"]["raw_residual"]),
                "correction": t["command"]["safety_correction"],
            }
            for t in record["trace"]
        ],
    }


def test_historical_comparison_checks_abort_and_continuous_data(observations):
    for record in observations:
        compare_historical(record, _historic(record))
    history = _historic(observations[0])
    history["trace"][0]["raw_residual"][0] += 0.1
    with pytest.raises(ValueError, match="historical residual"):
        compare_historical(observations[0], history)
    history = _historic(observations[-1])
    history["safety"]["safety_infeasible"] = False
    with pytest.raises(ValueError, match="historical abort"):
        compare_historical(observations[-1], history)


def test_pairing_rejects_wrong_state_or_case(observations):
    record = observations[0]
    zero = deepcopy(record)
    zero["episode"]["controller"] = "Z"
    contrast = paired_contrast(record, zero)
    assert contrast["zero_key"] == [record["episode"]["scenario"], record["episode"]["seed"]]
    zero["initial"]["command_queue"] = [[1.0, 1.0]]
    with pytest.raises(ValueError, match="paired initial"):
        paired_contrast(record, zero)


def test_balance_detects_torque_and_fault_corruption(observations):
    transition = observations[2]["trace"][20]
    command = SimpleNamespace(
        **{k: np.asarray(v) if isinstance(v, list) else v for k, v in transition["command"].items()}
    )
    info = {
        "pre_step_state": transition["before"]["state"],
        "pre_step_acceleration": transition["physical"]["qdd"],
        "commanded_torque_exact": transition["physical"]["commanded"],
        "delayed_torque_exact": transition["physical"]["delayed"],
        "plant_input_torque": transition["physical"]["applied"],
        "actuator_scaled_torque": transition["physical"]["applied"],
    }
    result = physical_balance(
        transition["before"],
        transition["after"],
        command,
        info,
        PlanarArm(),
        np.asarray(transition["nominal_unclipped"]),
    )
    assert result["checks_passed"]
    bad = deepcopy(info)
    bad["plant_input_torque"][0] += 0.1
    with pytest.raises(ValueError, match="applied torque"):
        physical_balance(
            transition["before"],
            transition["after"],
            command,
            bad,
            PlanarArm(),
            np.asarray(transition["nominal_unclipped"]),
        )


def test_tail_definition_from_synthetic_rows():
    record = _mock_result("synthetic", "engineering", 0)
    record["initial"].update({"arm_params": asdict(PlanarArm().params), "target": [1.0, 1.0]})
    record["trace"][0].update(
        {"after": {"state": [0.0] * 4}, "physical": {"qdd": [0.0, 0.0], "net": [0.0, 0.0]}}
    )
    record["trace"] = [deepcopy(record["trace"][0]) for _ in range(50)]
    record["episode"]["success"] = False
    record["safety"]["safety_infeasible"] = False
    for trace in record["trace"]:
        trace["distance_m"] = 0.2
        trace["speed_rad_s"] = 0.001
    assert describe(record)["off_target_stop"]
    record["trace"][0]["speed_rad_s"] = 0.061
    assert not describe(record)["off_target_stop"]
    record["trace"] = record["trace"][:1]
    record["episode"]["success"] = True
    assert describe(record)["outcome"] == "success"
    assert describe(record)["tail50"] is None


def test_invalid_serialization_preserves_small_errors():
    import json

    precise = np.float64(1.000000000041)
    value = invalid_data({"state": np.array([precise, np.nan, np.inf, -np.inf])})
    restored = json.loads(json.dumps(value, allow_nan=False))
    assert restored["state"][0] == precise
    assert restored["state"][1:] == [
        {"nonfinite": "nan"},
        {"nonfinite": "inf"},
        {"nonfinite": "-inf"},
    ]
