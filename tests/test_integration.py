import numpy as np

from sarrl.envs import PlanarReachEnv


def test_nominal_residual_controller_can_reach_fixed_target_with_zero_residual():
    env = PlanarReachEnv(mode="residual", max_steps=250)
    obs, _ = env.reset(seed=0, target=np.array([1.1, 0.6]))
    del obs
    success = False
    for _ in range(250):
        _, _, terminated, truncated, info = env.step(np.zeros(2))
        if terminated or truncated:
            success = bool(info["success"])
            break
    assert success
    assert info["distance"] <= env.success_radius
