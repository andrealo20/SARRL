import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from sarrl.controllers.panda_nominal import PandaComputedTorqueController  # noqa: E402
from sarrl.envs.panda_reach import (  # noqa: E402
    PandaReachConfig,
    PandaReachEnv,
    quintic_reference,
)
from sarrl.evaluation.panda_nominal import run_episode  # noqa: E402
from sarrl.robots.panda import HOME, PandaDynamics, load_panda_model, model_provenance  # noqa: E402


def test_panda_assets_and_torque_actuation():
    source = model_provenance()
    assert source["license"] == "Apache-2.0"
    model = load_panda_model()
    assert (model.nq, model.nv, model.nu) == (7, 7, 7)
    np.testing.assert_array_equal(model.actuator_gainprm[:, 0], np.ones(7))
    np.testing.assert_array_equal(model.actuator_biasprm, np.zeros((7, 10)))
    np.testing.assert_array_equal(model.actuator_gear[:, 0], np.ones(7))
    np.testing.assert_array_equal(model.actuator_ctrlrange[:, 1], [87] * 4 + [12] * 3)
    data = mujoco.MjData(model)
    data.qpos[:] = HOME
    data.ctrl[:] = np.arange(7) - 3
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qfrc_actuator, data.ctrl, atol=1e-12)


def test_panda_inertia_is_symmetric_positive_definite():
    dynamics = PandaDynamics()
    rng = np.random.default_rng(9804100)
    for _ in range(12):
        matrix = dynamics.mass_matrix(HOME + rng.uniform(-0.2, 0.2, 7))
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
        assert np.min(np.linalg.eigvalsh(matrix)) > 0.0


def test_panda_inverse_and_forward_dynamics_agree_without_contacts():
    dynamics = PandaDynamics()
    plant = load_panda_model()
    data = mujoco.MjData(plant)
    rng = np.random.default_rng(9804101)
    for _ in range(12):
        q = HOME + rng.uniform(-0.15, 0.15, 7)
        qd, qdd = rng.uniform(-0.2, 0.2, (2, 7))
        data.qpos[:], data.qvel[:] = q, qd
        data.ctrl[:] = dynamics.inverse_dynamics(q, qd, qdd)
        mujoco.mj_forward(plant, data)
        assert data.ncon == 0
        np.testing.assert_allclose(data.qacc, qdd, atol=1e-10)


def test_panda_gravity_compensation_holds_a_stationary_pose():
    env = PandaReachEnv()
    env.reset(seed=9804102, start=HOME, goal=HOME)
    controller = PandaComputedTorqueController()
    for _ in range(100):
        env.step_torque(controller.command(env.data.qpos, env.data.qvel, HOME))
    np.testing.assert_allclose(env.data.qpos, HOME, atol=1e-10)
    np.testing.assert_allclose(env.data.qvel, 0.0, atol=1e-10)
    assert env.diagnostics()["contact_substeps"] == 0


def test_quintic_reference_has_zero_endpoint_derivatives():
    goal = HOME + 0.2
    for time, expected in ((-1, HOME), (0, HOME), (2, goal), (3, goal)):
        q, qd, qdd = quintic_reference(HOME, goal, time, 2.0)
        np.testing.assert_allclose(q, expected, atol=1e-12)
        np.testing.assert_allclose(qd, 0, atol=1e-12)
        np.testing.assert_allclose(qdd, 0, atol=1e-12)
    q, qd, qdd = quintic_reference(HOME, goal, 1.0, 2.0)
    np.testing.assert_allclose(q, HOME + 0.1)
    np.testing.assert_allclose(qd, np.full(7, 0.1875))
    np.testing.assert_allclose(qdd, 0)


def test_panda_reset_and_rollout_are_reproducible():
    first, first_trace = run_episode(9804103, retain_trace=True)
    second, second_trace = run_episode(9804103, retain_trace=True)
    assert first == second
    assert first_trace == second_trace
    assert first["success"]
    assert not first["unsafe_episode"]
    assert first["controller_saturated_steps"] == 0
    assert len(first_trace) == 301


def test_panda_nominal_model_does_not_share_plant_parameters():
    env = PandaReachEnv()
    env.reset(seed=9804104)
    controller = PandaComputedTorqueController()
    nominal_mass = controller.dynamics.model.body_mass.copy()
    before = controller.command(HOME, np.zeros(7), HOME)
    env.model.body_mass[:] *= 1.2
    mujoco.mj_setConst(env.model, env.data)
    np.testing.assert_array_equal(controller.dynamics.model.body_mass, nominal_mass)
    np.testing.assert_array_equal(controller.command(HOME, np.zeros(7), HOME), before)


def test_panda_torque_limits_and_invalid_commands():
    env = PandaReachEnv()
    env.reset(seed=9804105)
    initial = env.state.copy()
    with pytest.raises(ValueError):
        env.step_torque(np.full(7, np.nan))
    np.testing.assert_array_equal(env.state, initial)
    assert env.steps == 0
    env.step_torque(np.full(7, 1000.0))
    np.testing.assert_array_equal(env.data.ctrl, env.torque_limit)
    assert env.saturated_steps == 1
    controller = PandaComputedTorqueController()
    torque = controller.command(HOME, np.zeros(7), HOME + 10)
    assert np.all(np.abs(torque) <= controller.dynamics.torque_limit)


def test_panda_rejects_bad_timing_and_out_of_bounds_paths():
    with pytest.raises(ValueError):
        PandaReachEnv(PandaReachConfig(control_dt=0.003))
    with pytest.raises(ValueError):
        PandaReachEnv(PandaReachConfig(timestep=np.nan))
    env = PandaReachEnv()
    with pytest.raises(ValueError):
        env.reset(start=HOME, goal=np.full(7, 20.0))
    with pytest.raises(RuntimeError):
        env.step_torque(np.zeros(7))


def test_panda_contact_observation_prevents_success():
    env = PandaReachEnv()
    env.reset(seed=9804106, start=HOME, goal=HOME)
    # A non-adjacent-link collision inside the individual joint ranges.
    env.data.qpos[:] = [
        2.79458829,
        0.78280070,
        0.06212423,
        -2.72770236,
        2.34605150,
        2.03699258,
        0.98303476,
    ]
    mujoco.mj_forward(env.model, env.data)
    assert env.data.ncon > 0
    env.step_torque(np.zeros(7))
    diagnostics = env.diagnostics()
    assert diagnostics["contact_substeps"] > 0
    assert diagnostics["unsafe_episode"]
    assert not diagnostics["success"]


def test_panda_runner_refuses_overwrite_and_unregistered_seeds(tmp_path):
    from tools.run_panda_nominal import main

    with pytest.raises(SystemExit) as exc:
        main(["--episodes", "1", "--seed-start", "50000", "--output", str(tmp_path / "new")])
    assert exc.value.code == 2
    assert not (tmp_path / "new").exists()
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("retained")
    with pytest.raises(SystemExit) as exc:
        main(["--output", str(tmp_path)])
    assert exc.value.code == 2
    assert sentinel.read_text() == "retained"
