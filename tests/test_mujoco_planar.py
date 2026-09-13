import numpy as np
import pytest

from sarrl.controllers import ComputedTorqueController
from sarrl.dynamics import PlanarArm, PlanarArmParams
from sarrl.envs import PlanarReachEnv
from sarrl.evaluation.planar_v13 import v13_scenarios

mujoco = pytest.importorskip("mujoco")

from sarrl.envs.mujoco_planar import MujocoPlanarReachEnv, planar_arm_xml  # noqa: E402


def _scenario(key):
    return {s.key: s for s in v13_scenarios()}[key]


def test_rigid_body_dynamics_match_the_analytical_model_without_dry_friction():
    rng = np.random.default_rng(0)
    params = PlanarArmParams(
        m1=1.1, m2=0.8, i1=0.09, i2=0.07, payload_mass=0.9, viscous=(0.05, 0.03), coulomb=(0.0, 0.0)
    )
    arm = PlanarArm(params)
    model = mujoco.MjModel.from_xml_string(planar_arm_xml(params, 0.002, "implicitfast"))
    data = mujoco.MjData(model)
    for _ in range(100):
        q, qd, tau = rng.uniform(-3, 3, 2), rng.uniform(-4, 4, 2), rng.uniform(-30, 30, 2)
        data.qpos[:] = q
        data.qvel[:] = qd
        data.qfrc_applied[:] = tau
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.qacc, arm.forward_dynamics(q, qd, tau), atol=1e-5)


def test_reset_samples_the_same_episode_as_the_analytical_environment():
    spec = _scenario("ood_compound")
    analytical = PlanarReachEnv(mode="torque", randomization=spec.randomization)
    engine = MujocoPlanarReachEnv(mode="torque", randomization=spec.randomization)
    for seed in (9803100, 9803101):
        oa, ia = analytical.reset(seed=seed)
        om, im = engine.reset(seed=seed)
        np.testing.assert_array_equal(oa, om)
        np.testing.assert_array_equal(analytical.state, engine.state)
        np.testing.assert_array_equal(analytical.target, engine.target)
        assert analytical.arm.params == engine.arm.params
        assert ia["action_delay"] == im["action_delay"]
        np.testing.assert_array_equal(ia["motor_gain"], im["motor_gain"])
        # The engine carries the sampled plant.
        body = engine._body["payload"]
        assert engine.model.body_mass[body] == pytest.approx(engine.payload_mass, abs=1e-8)
        np.testing.assert_allclose(engine.model.dof_damping, engine.arm.params.viscous)


def test_step_semantics_match_delay_gain_and_info_fields():
    spec = _scenario("id_reference")
    analytical = PlanarReachEnv(mode="torque", randomization=spec.randomization)
    engine = MujocoPlanarReachEnv(mode="torque", randomization=spec.randomization)
    analytical.reset(seed=9803003)
    engine.reset(seed=9803003)
    controller = ComputedTorqueController(PlanarArm())
    for _ in range(30):
        u = controller.command(analytical.state[:2], analytical.state[2:], analytical.q_des)
        _, ra, _, _, ia = analytical.step_torque(u)
        _, rm, _, _, im = engine.step_torque(u)
        np.testing.assert_array_equal(ia["delayed_torque_exact"], im["delayed_torque_exact"])
        np.testing.assert_array_equal(ia["applied_torque"], im["applied_torque"])
        assert im["plant"] == "mujoco"
        # One step apart the two plants stay close; dry friction is the only model difference.
        assert np.max(np.abs(analytical.state - engine.state)) < 0.2
        engine.state = analytical.state.copy()
        engine._sync_plant()
    assert engine.steps == analytical.steps == 30


def test_fault_changes_the_engine_payload_and_gain():
    spec = _scenario("motor_fault")
    engine = MujocoPlanarReachEnv(
        mode="torque", randomization=spec.randomization, fault=spec.fault
    )
    engine.reset(seed=9803200)
    gain_before = engine.motor_gain.copy()
    for _ in range(spec.fault.start_step + 1):
        engine.step_torque(np.zeros(2))
    assert engine._fault_active
    np.testing.assert_allclose(engine.motor_gain, gain_before * np.array([1.0, 0.55]))
    body = engine._body["payload"]
    assert engine.model.body_mass[body] == pytest.approx(max(engine.payload_mass, 1e-9))


def test_state_dict_round_trip_keeps_the_engine_in_sync():
    spec = _scenario("id_reference")
    engine = MujocoPlanarReachEnv(mode="torque", randomization=spec.randomization)
    engine.reset(seed=9803004)
    for _ in range(5):
        engine.step_torque(np.array([5.0, -2.0]))
    snapshot = engine.state_dict()
    restored = MujocoPlanarReachEnv.from_state_dict(snapshot)
    np.testing.assert_array_equal(restored.state, engine.state)
    np.testing.assert_allclose(restored.data.qpos, engine.data.qpos)
    assert restored.constructor_config() == engine.constructor_config()


def test_dt_must_be_a_multiple_of_the_timestep():
    with pytest.raises(ValueError):
        MujocoPlanarReachEnv(timestep=0.003)


def test_armature_and_actuator_lag_change_the_plant_response():
    spec = _scenario("id_reference")
    plain = MujocoPlanarReachEnv(mode="torque", randomization=spec.randomization)
    heavy = MujocoPlanarReachEnv(
        mode="torque", randomization=spec.randomization, armature=0.05, actuator_time_constant=0.03
    )
    plain.reset(seed=9803005)
    heavy.reset(seed=9803005)
    torque = np.array([10.0, 5.0])
    _, _, _, _, info_plain = plain.step_torque(torque)
    _, _, _, _, info_heavy = heavy.step_torque(torque)
    # Reflected inertia lowers the acceleration; the lag delivers less than commanded at first.
    acc_plain = np.abs(info_plain["pre_step_acceleration"])
    acc_heavy = np.abs(info_heavy["pre_step_acceleration"])
    assert np.all(acc_heavy <= acc_plain + 1e-9)
    assert np.all(np.abs(info_heavy["delivered_torque"]) < np.abs(info_heavy["applied_torque"]))
    assert heavy.constructor_config()["armature"] == 0.05
    with pytest.raises(ValueError):
        MujocoPlanarReachEnv(armature=-1.0)
