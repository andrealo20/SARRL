import numpy as np
import pytest

from sarrl.controllers import AdaptiveNominalConfig, ComputedTorqueController
from sarrl.dynamics import PlanarArm
from sarrl.envs import DomainRandomization, PlanarReachEnv
from sarrl.envs.planar_reach import ObstacleSpec
from sarrl.evaluation.adaptive_pilot import NominalStack, PlantOptions, run_case
from sarrl.evaluation.planar_v12 import planar_safety_config
from sarrl.evaluation.safety_audit import evaluate_safety_episodes, obstacle_violation
from sarrl.safety import HOCBFSafetyFilter

SEED = 9803310  # pilot range


def _randomization():
    return DomainRandomization(
        mass_fraction=0.15,
        friction_fraction=0.30,
        motor_gain_fraction=0.15,
        payload_range=(0.0, 1.0),
        action_delay_max=2,
    )


def test_obstacle_leaves_plant_draws_and_target_unchanged():
    plain = PlanarReachEnv(mode="torque", randomization=_randomization())
    with_obstacle = PlanarReachEnv(
        mode="torque", randomization=_randomization(), obstacle=ObstacleSpec()
    )
    for seed in (SEED, SEED + 1, SEED + 2):
        obs_a, _ = plain.reset(seed=seed)
        obs_b, info = with_obstacle.reset(seed=seed)
        assert np.array_equal(obs_a, obs_b)
        assert np.array_equal(plain.target, with_obstacle.target)
        assert np.array_equal(plain.motor_gain, with_obstacle.motor_gain)
        assert plain.arm.params == with_obstacle.arm.params
        assert plain.obstacles == [] and len(with_obstacle.obstacles) == 1
        assert info["obstacles"][0]["radius"] == 0.08


def test_obstacle_is_placed_outward_of_the_nominal_path_and_clear_of_endpoints():
    env = PlanarReachEnv(mode="torque", randomization=_randomization(), obstacle=ObstacleSpec())
    spec = env.obstacle
    for seed in range(SEED, SEED + 10):
        env.reset(seed=seed)
        path = env.nominal_path()
        obstacle = env.obstacles[0]
        center = np.asarray(obstacle.center)
        barrier = obstacle.radius + obstacle.margin
        gaps = np.linalg.norm(path - center, axis=1) - barrier
        # The barrier protrudes into the path by the sampled overlap, never more.
        assert -spec.overlap_range[1] - 1e-6 <= gaps.min() <= -spec.overlap_range[0] + 0.02
        assert np.linalg.norm(center - path[0]) > barrier + spec.clearance
        assert np.linalg.norm(center - env.target) > barrier + spec.clearance
        # Same seed, same obstacle.
        env.reset(seed=seed)
        assert env.obstacles[0] == obstacle


def test_obstacle_violation_and_clearance_agree():
    env = PlanarReachEnv(mode="torque", randomization=_randomization(), obstacle=ObstacleSpec())
    env.reset(seed=SEED)
    obstacle = env.obstacles[0]
    arm = PlanarArm()
    q = arm.inverse_kinematics(np.asarray(obstacle.center))
    excess, normalized = obstacle_violation(q, arm, env.obstacles)
    assert excess == pytest.approx(obstacle.radius + obstacle.margin, abs=1e-6)
    assert normalized == pytest.approx(1.0, abs=1e-6)
    assert env.obstacle_clearance(q) == pytest.approx(-obstacle.radius, abs=1e-6)
    assert obstacle_violation(env.state[:2], arm, env.obstacles)[0] == 0.0


def test_audit_scores_the_obstacle_and_the_filter_certifies_against_it():
    env = PlanarReachEnv(mode="torque", randomization=_randomization(), obstacle=ObstacleSpec())
    nominal = PlanarArm()
    unfiltered = NominalStack(ComputedTorqueController(nominal))
    filtered = NominalStack(
        ComputedTorqueController(nominal), HOCBFSafetyFilter(nominal, planar_safety_config())
    )
    observer = HOCBFSafetyFilter(nominal, planar_safety_config())
    _, plain = evaluate_safety_episodes(unfiltered, observer, env, 5, SEED)
    _, certified = evaluate_safety_episodes(filtered, observer, env, 5, SEED)
    assert any(row.obstacle_violation_max_m > 0.0 for row in plain)
    assert all(row.obstacle_violation_max_m <= 1e-3 for row in certified)
    assert all(row.obstacle_contact_geoms == () for row in certified)


def test_pilot_records_obstacle_fields():
    options = PlantOptions(obstacle=ObstacleSpec())
    episode = run_case(
        "adaptive_hocbf",
        "id_reference",
        SEED,
        "fresh",
        AdaptiveNominalConfig(),
        True,
        options=options,
    )
    assert episode.obstacle_present
    assert episode.obstacle_violation_max_m <= 1e-3
    plain = run_case(
        "fixed", "id_reference", SEED, "fresh", AdaptiveNominalConfig(), options=options
    )
    assert (
        plain.obstacle_present
        and not run_case(
            "fixed", "id_reference", SEED, "fresh", AdaptiveNominalConfig()
        ).obstacle_present
    )


def test_mujoco_reports_engine_contacts_with_the_obstacle():
    pytest.importorskip("mujoco")
    from sarrl.envs.mujoco_planar import MujocoPlanarReachEnv

    env = MujocoPlanarReachEnv(
        mode="torque", randomization=_randomization(), obstacle=ObstacleSpec()
    )
    env.reset(seed=SEED)
    obstacle = env.obstacles[0]
    # Drive the tip straight into the obstacle by resetting the target onto it;
    # the reset samples a fresh obstacle for that route, and the arm must hit one.
    env.reset(seed=SEED, target=np.asarray(obstacle.center))
    controller = ComputedTorqueController(PlanarArm())
    contact = False
    for _ in range(env.max_steps):
        _, _, terminated, truncated, info = env.step_torque(
            controller.command(env.state[:2], env.state[2:], env.q_des)
        )
        contact = contact or info["obstacle_contact"]
        if terminated or truncated:
            break
    assert contact
    # The obstacle-free engine keeps its previous model: nothing collides.
    plain = MujocoPlanarReachEnv(mode="torque", randomization=_randomization())
    assert plain.constructor_config().get("obstacle") is None
    assert plain.model.ngeom == 2
