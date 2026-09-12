"""Synthetic algebra, callback and storage checks with all plant operations forbidden."""

import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from sarrl.controllers import ComputedTorqueController
from sarrl.controllers.integral_nominal import IntegralNominalController
from sarrl.dynamics import PlanarArm
from sarrl.envs import PlanarReachEnv
from sarrl.evaluation import nominal_integral as evaluation
from sarrl.evaluation.residual_diagnosis import plain
from tools import run_nominal_integral as runner


@pytest.fixture(autouse=True)
def forbid_physics(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("plant reset or integration is forbidden in integral tests")

    monkeypatch.setattr(PlanarReachEnv, "reset", forbidden)
    monkeypatch.setattr(PlanarReachEnv, "step_torque", forbidden)
    monkeypatch.setattr(PlanarArm, "step_rk4", forbidden)


class AlgebraModel:
    def __init__(self, mass=None):
        self.mass = np.eye(2) if mass is None else np.asarray(mass, dtype=float)

    def mass_matrix(self, q):
        return self.mass.copy()

    def inverse_dynamics(self, q, qd, acceleration, include_friction):
        return self.mass @ acceleration


def issue(controller, error=(0.1, -0.2)):
    command = controller.command([0, 0], [0, 0], error)
    return command, controller.pending["token"]


def test_pure_command_copies_inputs_and_snapshot_and_matches_legacy():
    model = PlanarArm()
    integral, legacy = IntegralNominalController(model), ComputedTorqueController(model)
    for state in ([0.0, 0.0, 0.0, 0.0], [2.9, -2.9, 0.3, -0.7], [-1, 2, 4, -5]):
        integral.reset()
        state = np.asarray(state)
        target = np.array([-3.9, 1.0])
        result = integral.command(state[:2], state[2:], target)
        np.testing.assert_array_equal(result, legacy.command(state[:2], state[2:], target))
        np.testing.assert_array_equal(integral.z, [0, 0])
        snapshot = integral.pending
        expected = snapshot["e_pre"].copy()
        target[:] = 0
        result[:] = 0
        snapshot["e_pre"][:] = 0
        np.testing.assert_array_equal(integral.pending["e_pre"], expected)
        with pytest.raises(ValueError, match="pending"):
            issue(integral)


def test_integral_uses_pre_step_mass_error_and_explicit_dt():
    model = AlgebraModel([[2, 1], [1, 3]])
    controller = IntegralNominalController(model)
    b, token = issue(controller)
    model.mass[:] = np.eye(2) * 99
    row = controller.commit(token, b, b)
    np.testing.assert_allclose(row["z_after"], [0.072, -0.144], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(row["back_calculation"], [0, 0])
    np.testing.assert_array_equal(row["M_nominal_pre"], [[2, 1], [1, 3]])
    with pytest.raises(ValueError, match="duplicate"):
        controller.commit(token, b, b)
    row["z_after"][:] = 123
    np.testing.assert_allclose(controller.z, [0.072, -0.144])


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_saturation_backcalculation_sign_and_coupled_projection(sign):
    mass = np.array([[2.0, 1.0], [1.0, 3.0]])
    controller = IntegralNominalController(AlgebraModel(mass))
    b, token = issue(controller, [sign * 2, sign * 2])
    raw = controller.pending["tau_raw"]
    h = b - sign * np.array([1.0, 0.0])
    row = controller.commit(token, h, h)
    np.testing.assert_allclose(row["aw_hocbf"], sign * np.array([-2.4, 0.8]))
    np.testing.assert_allclose(row["aw_clip"], 4 * np.linalg.solve(mass, b - raw))
    np.testing.assert_allclose(row["back_calculation"], 4 * np.linalg.solve(mass, h - raw))
    assert sign * row["aw_clip"][0] < 0


def test_final_send_clip_is_separate_and_invalid_commit_preserves_pending():
    controller = IntegralNominalController(AlgebraModel())
    b, token = issue(controller)
    h = np.array([40.0 + 1e-8, -40.0 - 1e-8])
    with pytest.raises(ValueError, match="clipping"):
        controller.commit(token, h, b)
    np.testing.assert_array_equal(controller.z, [0, 0])
    assert controller.pending["token"] == token
    sent = np.clip(h, -40, 40)
    row = controller.commit(token, h, sent)
    np.testing.assert_allclose(row["aw_sendclip"], 4 * (sent - h), rtol=0, atol=1e-15)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_integral_cap_without_nominal_clipping(sign):
    controller = IntegralNominalController(AlgebraModel(np.eye(2) * 0.01))
    # Repeated algebraic transactions are synthetic vectors, never plant transitions.
    for _ in range(60):
        b, token = issue(controller, [sign, sign])
        row = controller.commit(token, b, b)
    np.testing.assert_array_equal(controller.z, [sign * 36] * 2)
    assert np.all(sign * row["integral_clipping"] < 0)


def test_abort_and_reset_reject_old_token_even_same_attempt_index():
    controller = IntegralNominalController(AlgebraModel())
    b, old = issue(controller)
    row = controller.abort(old)
    assert row["commit"] is False
    for field in (
        "u_sent",
        "aw_clip",
        "aw_hocbf",
        "aw_sendclip",
        "back_calculation",
        "dz",
        "z_unclipped_next",
    ):
        assert row[field] is None
    np.testing.assert_array_equal(row["z_after"], row["z_before"])
    with pytest.raises(ValueError):
        issue(controller)
    controller.reset()
    b, new = issue(controller)
    assert old[1] == new[1] == 0 and old[0] != new[0]
    with pytest.raises(ValueError, match="stale"):
        controller.commit(old, b, b)
    controller.commit(new, b, b)
    assert plain(controller.state_dict())["pending"] is None


@pytest.mark.parametrize("bad", [[0, np.nan], [np.inf, 0], [0], [[0, 0]]])
def test_bad_input_creates_no_pending_transaction(bad):
    controller = IntegralNominalController(AlgebraModel())
    with pytest.raises(ValueError, match="finite pair"):
        issue(controller, bad)
    assert controller.pending is None


class SyntheticEnv:
    """A fixed observation source, with no dynamics engine or numerical integrator."""

    mode, dt, torque_limit = "torque", 0.02, 40.0

    def __init__(self, outcome):
        self.arm = PlanarArm()
        self.state = np.zeros(4)
        self.q_des = np.array([0.1, -0.1])
        self.target = self.arm.forward_kinematics([0, 0]) + np.array([0.2, 0])
        self.steps = self.resets = 0
        self.outcome = outcome
        self.terminal_after = 1

    def state_dict(self):
        return {
            "state": self.state.copy(),
            "steps": self.steps,
            "q_des": self.q_des.copy(),
            "target": self.target.copy(),
            "arm_params": vars(self.arm.params).copy(),
            "noise_rng_state": {"state": 123},
            "command_queue": [],
        }

    def reset(self, seed):
        self.resets += 1
        return np.zeros(8, dtype=np.float32), {}

    def _observation(self):
        pytest.fail("adapter requested an extra observation")

    def step_torque(self, torque, baseline):
        self.steps += 1
        terminal = self.steps >= self.terminal_after
        success = terminal and self.outcome == "success"
        sent = np.clip(torque, -40, 40)
        distance = float(np.linalg.norm(self.target - self.arm.forward_kinematics([0, 0])))
        reward = -distance - 0.0002 * float(sent @ sent) + (10 if success else 0)
        return (
            np.ones(8, dtype=np.float32),
            reward,
            success,
            terminal and not success,
            {
                "commanded_torque_exact": sent,
                "commanded_torque": sent,
                "success": success,
                "distance": distance,
            },
        )


def setup_recorder(monkeypatch, outcome):
    env = SyntheticEnv(outcome)
    recorder = evaluation.IntegralRecorder(env)

    def projection(state, candidate, obstacles=()):
        success = outcome != "abort"
        result = SimpleNamespace(torque=candidate.copy(), success=success, correction_norm=0.0)
        recorder.safety.calls += 1
        recorder.safety.last = {
            "state": state.copy(),
            "candidate": candidate.copy(),
            "result": {"torque": candidate.copy(), "success": success},
        }
        return result

    def balance(before, after, command, info, nominal, raw):
        assert after["steps"] == before["steps"] + 1
        np.testing.assert_array_equal(raw, recorder.baseline.pending["tau_raw"])
        np.testing.assert_array_equal(
            raw,
            recorder.baseline.pending["nominal_pd_unclipped"]
            + recorder.baseline.pending["integral_torque"],
        )
        return {"commanded": info["commanded_torque_exact"], "qdd": [0, 0], "net": [0, 0]}

    monkeypatch.setattr(recorder.safety, "filter", projection)
    monkeypatch.setattr(evaluation, "physical_balance", balance)
    return recorder


@pytest.mark.parametrize("outcome", ["success", "timeout", "abort"])
def test_canonical_callback_terminal_commit_and_call_counts(monkeypatch, outcome):
    recorder = setup_recorder(monkeypatch, outcome)
    record = recorder.evaluate(seed=7, scenario="synthetic")
    assert record["summary"]["outcome"] == outcome
    assert recorder.env.resets == recorder.policy.calls == recorder.safety.calls == 1
    assert recorder.env.steps == int(outcome != "abort")
    assert len(record["trace"]) == 1
    row = record["trace"][0]
    assert row["integral"]["commit"] == (outcome != "abort")
    assert record["integral_final"]["pending"] is None
    if outcome == "abort":
        assert row["before"] == row["after"]
        assert row["physical"] is row["integral"]["u_sent"] is None
    else:
        np.testing.assert_allclose(row["integral"]["z_after"], [0.072, -0.072])
    with pytest.raises(ValueError, match="only one episode"):
        recorder.evaluate(seed=7, scenario="synthetic")
    assert recorder.env.resets == recorder.policy.calls == recorder.safety.calls == 1


def test_callback_failure_cannot_commit_or_hide_partial_evidence(monkeypatch):
    recorder = setup_recorder(monkeypatch, "success")
    original = recorder.observe

    def corrupted(event):
        event["info"]["commanded_torque_exact"] = [0.0, 0.0]
        original(event)

    monkeypatch.setattr(recorder, "observe", corrupted)
    with pytest.raises(evaluation.DiagnosticFailure) as failure:
        recorder.evaluate(seed=0, scenario="synthetic")
    assert failure.value.partial["status"] == "invalid"
    assert failure.value.partial["integral_final"]["pending"] is not None
    assert failure.value.partial["command_context"]["observation"] == [0.0] * 8
    assert failure.value.partial["command_context"]["before"]["steps"] == 0
    np.testing.assert_array_equal(recorder.baseline.z, [0, 0])


def test_second_synthetic_callback_records_nonzero_integral_torque(monkeypatch):
    recorder = setup_recorder(monkeypatch, "timeout")
    recorder.env.terminal_after = 2
    record = recorder.evaluate(seed=7, scenario="synthetic")
    first, second = record["trace"]
    assert not first["truncated"] and second["truncated"]
    np.testing.assert_array_equal(second["integral"]["z_before"], first["integral"]["z_after"])
    expected_integral = recorder.nominal.mass_matrix([0, 0]) @ np.array([0.072, -0.072])
    assert np.linalg.norm(expected_integral) > 0
    np.testing.assert_allclose(second["integral_torque"], expected_integral)
    np.testing.assert_allclose(
        second["nominal_unclipped"], np.array(second["nominal_pd_unclipped"]) + expected_integral
    )
    assert recorder.policy.calls == recorder.safety.calls == recorder.env.steps == 2
    assert recorder.env.resets == 1


def test_r0_dispatches_directly_to_legacy(monkeypatch):
    env = object()
    calls = []

    def legacy(policy, actual_env, **kwargs):
        calls.append(kwargs)
        assert actual_env is env
        assert type(policy) is evaluation.ZeroResidualPolicy
        return "legacy"

    monkeypatch.setattr(evaluation, "evaluate_recorded", legacy)
    assert evaluation.evaluate_nominal(env, seed=3, scenario="case", controller="R0") == "legacy"
    assert calls == [{"seed": 3, "scenario": "case", "controller": "R0"}]


def record_for(controller, scenario="id_reference", seed=9801800):
    return {
        "schema_version": 1,
        "status": "valid",
        "identity": None,
        "episode": {
            "controller": controller,
            "scenario": scenario,
            "seed": seed,
            "steps": 1,
            "success": True,
            "reward": 0.0,
            "final_distance": 0.0,
        },
        "safety": {
            "controller": controller,
            "unsafe_episode": False,
            "safety_infeasible": False,
            "normalized_violation_max": 0.0,
        },
        "initial": {"state": [0.0] * 4, "noise_rng_state": {"state": 1, "x": 0.1}},
        "initial_observation": [0.0] * 8,
        "final": {"state": [0.0] * 4},
        "configuration": {"kp": [36.0, 36.0]},
        "trace": [{"executed": True, "state": [0.0] * 4}],
        "summary": {"outcome": "success", "final_speed_rad_s": 0.0},
    }


def test_reference_full_schema_remaps_only_copies():
    new, old = record_for("R0"), record_for("Z")
    new["trace"][0]["integral"] = None
    saved = deepcopy((new, old))
    evaluation.compare_reference(new, old)
    assert (new, old) == saved
    new["trace"][0]["state"][0] = 1e-11
    evaluation.compare_reference(new, old)
    new["trace"][0]["state"][0] = 1e-8
    with pytest.raises(ValueError, match="trace"):
        evaluation.compare_reference(new, old)


@pytest.mark.parametrize("corruption", ["rng", "bool", "key", "model", "label", "shape", "extra"])
def test_reference_rejects_structure_discrete_rng_and_identity_changes(corruption):
    new, old = record_for("R0"), record_for("Z")
    if corruption == "rng":
        new["initial"]["noise_rng_state"]["x"] += 1e-15
    elif corruption == "bool":
        new["trace"][0]["executed"] = 1
    elif corruption == "key":
        new["configuration"]["new"] = 0
    elif corruption == "model":
        old["identity"] = {"path": "model"}
    elif corruption == "label":
        new["safety"]["controller"] = "I1"
    elif corruption == "shape":
        new["trace"][0]["state"].pop()
    else:
        new["mystery"] = None
    with pytest.raises(ValueError):
        evaluation.compare_reference(new, old)


def matrix():
    return [record_for(c, s, seed) for c, s, seed in runner.cell_specs()]


def test_decision_order_matrix_and_nonmotivating_regression():
    records = matrix()
    assert evaluation.decision(records) == "target_cases_passed"
    assert evaluation.decision(records[:-1]) == "incomplete"
    assert evaluation.decision(records[:-1] + records[:1]) == "incomplete"
    records[3]["episode"]["success"] = False  # Other case does not change the frozen rule.
    assert evaluation.decision(records) == "target_cases_passed"
    records[1]["episode"]["success"] = False
    assert evaluation.decision(records) == "hypothesis_not_met"
    records[3]["safety"]["unsafe_episode"] = True
    assert evaluation.decision(records) == "no_go_safety"
    records[-1]["status"] = "invalid"
    assert evaluation.decision(records) == "incomplete"


@pytest.mark.parametrize("kind", ["abort", "unsafe_increase", "threshold"])
def test_safety_veto_and_tolerance(kind):
    records = matrix()
    if kind == "abort":
        records[1]["safety"]["safety_infeasible"] = True
    else:
        for r in records[:2]:
            r["safety"].update(unsafe_episode=True, normalized_violation_max=0.1)
        records[1]["safety"]["normalized_violation_max"] += (
            1e-8 if kind == "unsafe_increase" else 1e-11
        )
    assert evaluation.decision(records) == (
        "target_cases_passed" if kind == "threshold" else "no_go_safety"
    )


@pytest.fixture
def synthetic_runner(tmp_path, monkeypatch):
    root = tmp_path
    history = root / "history"
    history.mkdir()
    protocol = root / "protocol.md"
    protocol.write_text("synthetic protocol", encoding="utf-8")
    plan = {"protocol_sha256": runner.sha(protocol)}
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "HISTORY", history)
    monkeypatch.setattr(runner, "build_plan", lambda p: plan)
    calls = []
    monkeypatch.setattr(runner, "PlanarReachEnv", lambda **kwargs: object())

    def evaluate(env, *, seed, scenario, controller):
        calls.append((controller, scenario, seed))
        result = record_for(controller, scenario, seed)
        if controller == "I1":
            result["integral_summary"] = {}
        return result

    monkeypatch.setattr(runner, "evaluate_nominal", evaluate)
    for _controller, scenario, seed in runner.cell_specs():
        path = history / (runner.cell_name("Z", scenario, seed) + ".json")
        if not path.exists():
            runner.write_new(path, record_for("Z", scenario, seed))
    return plan, root / "results" / runner.OUTPUT_NAME, protocol, calls


def test_runner_exact_order_complete_and_no_overwrite(synthetic_runner):
    plan, output, protocol, calls = synthetic_runner
    runner.execute(plan, output, protocol)
    assert calls == list(runner.cell_specs())
    assert [c[1] for c in calls[::4]] == ["id_reference", "ood_compound", "motor_fault"]
    report = runner.read(output / "report.json")
    assert len(report["pairs"]) == 6 and len(report["other_four_cases"]) == 4
    assert report["decision"] == "target_cases_passed"
    with pytest.raises(FileExistsError):
        runner.execute(plan, output, protocol)
    assert len(calls) == 12


@pytest.mark.parametrize("marker", ["started.json", "invalid.json", "json.tmp", "json"])
def test_runner_checks_later_interruption_before_any_new_cell(synthetic_runner, marker):
    plan, output, protocol, calls = synthetic_runner
    output.mkdir(parents=True)
    runner.write_new(output / "manifest.json", plan)
    name = runner.cell_name(*list(runner.cell_specs())[7])
    (output / f"{name}.{marker}").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="inspection"):
        runner.execute(plan, output, protocol)
    assert calls == []


def test_runner_technical_parity_failure_stops_and_preserves_invalid(synthetic_runner, monkeypatch):
    plan, output, protocol, calls = synthetic_runner
    original = runner.evaluate_nominal

    def mismatch(*args, **kwargs):
        result = original(*args, **kwargs)
        result["final"]["state"][0] = 1.0
        return result

    monkeypatch.setattr(runner, "evaluate_nominal", mismatch)
    with pytest.raises(ValueError):
        runner.execute(plan, output, protocol)
    assert len(calls) == 1
    assert runner.read(output / "incomplete.json")["decision"] == "incomplete"
    assert len(list(output.glob("*.invalid.json"))) == 1
    with pytest.raises(ValueError, match="incomplete"):
        runner.execute(plan, output, protocol)
    assert len(calls) == 1


def test_runner_resumes_only_never_started_cells(synthetic_runner, monkeypatch):
    plan, output, protocol, calls = synthetic_runner
    real_write = runner.write_new

    def interrupted(path, payload):
        if path.name == runner.cell_name(*list(runner.cell_specs())[1]) + ".started.json":
            raise KeyboardInterrupt("synthetic interruption before second reservation")
        return real_write(path, payload)

    monkeypatch.setattr(runner, "write_new", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runner.execute(plan, output, protocol)
    assert len(calls) == 1
    monkeypatch.setattr(runner, "write_new", real_write)
    runner.execute(plan, output, protocol)
    assert calls == list(runner.cell_specs())


def test_runner_cli_has_no_implicit_execution(monkeypatch):
    monkeypatch.setattr(runner, "build_plan", lambda *args: pytest.fail("unexpected preflight"))
    with pytest.raises(SystemExit):
        runner.main(["run", "--protocol", "missing"])


def test_report_suppresses_differences_at_unequal_terminal_times():
    records = matrix()
    for record in records[1::2]:
        record["integral_summary"] = {}
    records[1]["episode"]["steps"] = 2
    report = runner.comparisons(records)
    assert report[0]["final_distance_difference_m"] is None
    assert report[1]["final_distance_difference_m"] == 0.0


def test_retained_z_schema_is_accepted_without_replaying_trajectories():
    first = runner.HISTORY / (runner.cell_name("Z", "id_reference", 9801800) + ".json")
    if not first.exists():
        pytest.skip("retained local diagnostic episode records unavailable")
    checked = set()
    for _, scenario, seed in runner.cell_specs():
        if (scenario, seed) in checked:
            continue
        old = runner.read(runner.HISTORY / (runner.cell_name("Z", scenario, seed) + ".json"))
        new = deepcopy(old)
        new["episode"]["controller"] = new["safety"]["controller"] = "R0"
        evaluation.compare_reference(new, old)
        assert old["episode"]["controller"] == "Z"
        checked.add((scenario, seed))
    assert len(checked) == 6


@pytest.mark.parametrize("corruption", ["hash", "reservation", "pair", "order"])
def test_resume_validates_all_retained_cells_before_new_work(
    synthetic_runner,
    monkeypatch,
    corruption,
):
    plan, output, protocol, calls = synthetic_runner
    real_write = runner.write_new
    keys = list(runner.cell_specs())

    def interrupted(path, payload):
        if path.name == runner.cell_name(*keys[2]) + ".started.json":
            raise KeyboardInterrupt("synthetic interruption before third reservation")
        return real_write(path, payload)

    monkeypatch.setattr(runner, "write_new", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runner.execute(plan, output, protocol)
    monkeypatch.setattr(runner, "write_new", real_write)
    import json

    name = runner.cell_name(*keys[1])
    if corruption == "hash":
        (output / f"{name}.json").write_text("{}", encoding="utf-8")
    elif corruption == "reservation":
        (output / f"{name}.started.json").write_text("{}", encoding="utf-8")
    elif corruption == "order":
        for suffix in ("json", "complete.json", "started.json"):
            (output / (runner.cell_name(*keys[0]) + "." + suffix)).unlink()
    else:
        path = output / f"{name}.json"
        result = runner.read(path)
        result["initial_observation"][0] = 1.0
        path.write_text(json.dumps(result), encoding="utf-8")
        marker = output / f"{name}.complete.json"
        payload = runner.read(marker)
        payload["sha256"] = runner.sha(path)
        marker.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        runner.execute(plan, output, protocol)
    assert len(calls) == 2


@pytest.fixture
def synthetic_preflight(tmp_path, monkeypatch):
    """Exercise real hash checks against a temporary repository and six fake references."""
    root = tmp_path / "repository"
    history = root / "results" / "retained"
    history.mkdir(parents=True)
    protocol = root / "protocol.md"
    protocol.write_text("synthetic frozen protocol", encoding="utf-8")
    source_names = (
        "sarrl/controllers/computed_torque.py",
        "tools/run_nominal_integral.py",
        "tools/run_residual_diagnosis.py",
        "tests/test_nominal_integral.py",
        "pyproject.toml",
    )
    for name in source_names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source: " + name, encoding="utf-8")
    legacy = root / source_names[0]
    runner.write_new(
        history / "manifest.json",
        {
            "source_hashes": {source_names[0]: runner.sha(legacy)},
        },
    )
    hashes = {"manifest.json": runner.sha(history / "manifest.json")}
    for _, scenario, seed in runner.cell_specs():
        name = runner.cell_name("Z", scenario, seed) + ".json"
        if name not in hashes:
            runner.write_new(history / name, {"synthetic": True, "seed": seed})
            hashes[name] = runner.sha(history / name)
    runner.write_new(history / "complete.json", {"hashes": hashes})
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "HISTORY", history)
    monkeypatch.setattr(runner, "__file__", str(root / "tools/run_nominal_integral.py"))
    monkeypatch.setattr(runner, "PROTOCOL_HASH", runner.sha(protocol))
    monkeypatch.setattr(runner, "HISTORY_HASH", runner.sha(history / "complete.json"))
    monkeypatch.setattr(runner, "assert_repository_import_root", lambda p: root)
    monkeypatch.setattr(runner, "runtime_metadata", lambda p: {"synthetic": True})
    monkeypatch.setattr(runner.torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(runner.torch, "get_num_interop_threads", lambda: 1)

    def forbidden(*args, **kwargs):
        pytest.fail("preflight must not construct an environment or evaluate a controller")

    monkeypatch.setattr(runner, "PlanarReachEnv", forbidden)
    monkeypatch.setattr(runner, "evaluate_nominal", forbidden)
    return root, history, protocol, legacy


def test_real_preflight_hashes_sources_and_six_references(synthetic_preflight):
    root, history, protocol, legacy = synthetic_preflight
    plan = runner.build_plan(protocol)
    assert plan["cells"] == [list(key) for key in runner.cell_specs()]
    assert plan["protocol_sha256"] == runner.sha(protocol)
    assert plan["history_complete_sha256"] == runner.sha(history / "complete.json")
    assert plan["source_hashes"]["sarrl/controllers/computed_torque.py"] == runner.sha(legacy)
    assert len(plan["history_hashes"]) == 6
    assert plan["episodes"] == 12 and plan["max_physical_steps"] == 3000
    assert plan["execution"]["device"] == "cpu"
    assert not (root / "results" / runner.OUTPUT_NAME).exists()


@pytest.mark.parametrize(
    "target,message",
    [
        ("protocol", "reviewed protocol changed"),
        ("complete", "historical completion changed"),
        ("manifest", "historical manifest changed"),
        ("legacy", "legacy source changed"),
        ("reference", "historical reference changed"),
    ],
)
def test_real_preflight_rejects_altered_input(synthetic_preflight, target, message):
    root, history, protocol, legacy = synthetic_preflight
    path = {
        "protocol": protocol,
        "complete": history / "complete.json",
        "manifest": history / "manifest.json",
        "legacy": legacy,
        "reference": history / "Z__id_reference__9801800.json",
    }[target]
    # Every altered file belongs to this test's temporary repository.
    assert path.is_relative_to(root)
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\nsynthetic corruption\n")
    with pytest.raises(ValueError, match=message):
        runner.build_plan(protocol)
    assert not (root / "results" / runner.OUTPUT_NAME).exists()


@pytest.mark.parametrize("intraop,interop", [(2, 1), (1, 2)])
def test_execution_metadata_rejects_multiple_threads(monkeypatch, intraop, interop):
    monkeypatch.setattr(runner.torch, "get_num_threads", lambda: intraop)
    monkeypatch.setattr(runner.torch, "get_num_interop_threads", lambda: interop)
    with pytest.raises(ValueError, match="one numerical thread"):
        runner.execution_metadata()


def test_cli_plan_prints_real_preflight_without_execution(synthetic_preflight, monkeypatch, capsys):
    root, _, protocol, _ = synthetic_preflight
    monkeypatch.setattr(runner.torch, "set_num_threads", lambda n: None)
    monkeypatch.setattr(runner, "execute", lambda *args: pytest.fail("plan attempted execution"))
    runner.main(["plan", "--protocol", str(protocol)])
    plan = json.loads(capsys.readouterr().out)
    assert plan == runner.build_plan(protocol)
    assert not (root / "results" / runner.OUTPUT_NAME).exists()
