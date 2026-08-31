import numpy as np

from sarrl.controllers import ComputedTorqueController
from sarrl.dynamics import PlanarArm


def test_computed_torque_converges_on_fixed_joint_target():
    arm = PlanarArm()
    ctl = ComputedTorqueController(arm)
    q_des = np.array([0.8, -0.5])
    x = np.array([-0.2, 0.3, 0.0, 0.0])
    for _ in range(400):
        tau = ctl.command(x[:2], x[2:], q_des)
        x = arm.step_rk4(x, tau, 0.01)
    error = np.arctan2(np.sin(q_des - x[:2]), np.cos(q_des - x[:2]))
    assert np.linalg.norm(error) < 2e-3
    assert np.linalg.norm(x[2:]) < 1e-2


def test_controller_respects_torque_limits():
    arm = PlanarArm()
    ctl = ComputedTorqueController(arm, torque_limit=(5.0, 7.0))
    tau = ctl.command(np.zeros(2), np.zeros(2), np.array([3.0, -3.0]))
    assert abs(tau[0]) <= 5.0
    assert abs(tau[1]) <= 7.0
