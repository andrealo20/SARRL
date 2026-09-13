"""The planar reaching benchmark with its plant simulated by MuJoCo.

Everything that defines an episode (seeded randomisation of masses, friction,
payload, motor gain and delay; target sampling; observation, reward and
success; fault injection; the command chain of clipping, delay and gain) is
inherited unchanged from `PlanarReachEnv`. Only the plant changes: joint
accelerations and the integration step come from MuJoCo instead of the
analytical rigid-body equations and RK4. The controller never sees the
engine; it keeps working from `state`, exactly as on the analytical plant.

Differences that are inherent to the engine and therefore part of what this
environment tests: the integrator (`implicitfast` with a 2 ms internal step
by default), MuJoCo's constraint-based dry friction (`frictionloss`) in place
of the smoothed `tanh` Coulomb term, and a payload modelled as a body with a
small non-zero inertia because MuJoCo requires one.
"""

from __future__ import annotations

import numpy as np

from sarrl.dynamics import PlanarArmParams
from sarrl.envs.planar_reach import DomainRandomization, FaultSpec, PlanarReachEnv

try:  # MuJoCo is an optional dependency of the repository.
    import mujoco
except ImportError:  # pragma: no cover - exercised only without the extra installed
    mujoco = None

PAYLOAD_INERTIA_FLOOR = 1e-9
PAYLOAD_MASS_FLOOR = 1e-9


def planar_arm_xml(params: PlanarArmParams, timestep: float, integrator: str) -> str:
    """MuJoCo model of the two-link arm with the analytical benchmark's geometry.

    Gravity acts along -y in the plane of motion and joint angles are measured
    from the +x axis, matching `PlanarArm.gravity_vector`. Inertias are given
    about each link's centre of mass, as in `PlanarArmParams`.
    """
    p = params
    payload = max(p.payload_mass, PAYLOAD_MASS_FLOOR)
    floor = PAYLOAD_INERTIA_FLOOR
    return f"""
<mujoco model="sarrl_planar_arm">
  <option timestep="{timestep}" gravity="0 {-p.gravity} 0" integrator="{integrator}"/>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="joint1" type="hinge" axis="0 0 1" damping="{p.viscous[0]}"
             frictionloss="{p.coulomb[0]}" limited="false"/>
      <inertial pos="{p.lc1} 0 0" mass="{p.m1}" diaginertia="{p.i1} {p.i1} {p.i1}"/>
      <geom type="capsule" fromto="0 0 0 {p.l1} 0 0" size="0.03" mass="0"
            contype="0" conaffinity="0"/>
      <body name="link2" pos="{p.l1} 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1" damping="{p.viscous[1]}"
               frictionloss="{p.coulomb[1]}" limited="false"/>
        <inertial pos="{p.lc2} 0 0" mass="{p.m2}" diaginertia="{p.i2} {p.i2} {p.i2}"/>
        <geom type="capsule" fromto="0 0 0 {p.l2} 0 0" size="0.03" mass="0"
              contype="0" conaffinity="0"/>
        <body name="payload" pos="{p.l2} 0 0">
          <inertial pos="0 0 0" mass="{payload}" diaginertia="{floor} {floor} {floor}"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class MujocoPlanarReachEnv(PlanarReachEnv):
    """`PlanarReachEnv` whose plant is advanced by MuJoCo."""

    def __init__(
        self,
        mode: str = "torque",
        dt: float = 0.02,
        max_steps: int = 250,
        torque_limit: float = 40.0,
        residual_limit: float = 8.0,
        success_radius: float = 0.05,
        randomization: DomainRandomization | None = None,
        fault: FaultSpec | None = None,
        timestep: float = 0.002,
        integrator: str = "implicitfast",
    ):
        if mujoco is None:
            raise ImportError("MujocoPlanarReachEnv needs the 'mujoco' package")
        super().__init__(
            mode=mode,
            dt=dt,
            max_steps=max_steps,
            torque_limit=torque_limit,
            residual_limit=residual_limit,
            success_radius=success_radius,
            randomization=randomization,
            fault=fault,
        )
        substeps = self.dt / timestep
        if abs(substeps - round(substeps)) > 1e-9 or round(substeps) < 1:
            raise ValueError("dt must be a positive integer multiple of the MuJoCo timestep")
        self.timestep = float(timestep)
        self.substeps = int(round(substeps))
        self.integrator = integrator
        self.model = mujoco.MjModel.from_xml_string(
            planar_arm_xml(self.nominal_arm.params, self.timestep, integrator)
        )
        self.data = mujoco.MjData(self.model)
        self._body = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("link1", "link2", "payload")
        }
        self._sync_plant()

    # Plant synchronisation ---------------------------------------------------

    def _sync_plant(self) -> None:
        """Write the current `self.arm` parameters and `self.state` into MuJoCo."""
        p = self.arm.params
        model = self.model
        for name, mass, inertia in (
            ("link1", p.m1, p.i1),
            ("link2", p.m2, p.i2),
            ("payload", max(p.payload_mass, PAYLOAD_MASS_FLOOR), PAYLOAD_INERTIA_FLOOR),
        ):
            body = self._body[name]
            model.body_mass[body] = mass
            model.body_inertia[body] = (inertia, inertia, inertia)
        model.dof_damping[:] = p.viscous
        model.dof_frictionloss[:] = p.coulomb
        mujoco.mj_setConst(model, self.data)
        self.data.qpos[:] = self.state[:2]
        self.data.qvel[:] = self.state[2:]
        self.data.qacc[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(model, self.data)

    def reset(self, seed: int | None = None, target=None):
        result = super().reset(seed=seed, target=target)
        self._sync_plant()
        return result

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self._sync_plant()

    def _activate_fault_if_due(self) -> None:
        was_active = self._fault_active
        super()._activate_fault_if_due()
        if self._fault_active and not was_active:
            self._sync_plant()

    # Plant step --------------------------------------------------------------

    def plant_acceleration(self, applied) -> np.ndarray:
        """Joint acceleration MuJoCo attributes to the current state and torque."""
        self.data.qfrc_applied[:] = applied
        mujoco.mj_forward(self.model, self.data)
        return np.array(self.data.qacc, dtype=np.float64)

    def step_torque(self, commanded, baseline=None):
        """Same contract as `PlanarReachEnv.step_torque`, with MuJoCo as the plant."""
        self._activate_fault_if_due()
        commanded = np.asarray(commanded, dtype=np.float64)
        if commanded.shape != (2,) or not np.all(np.isfinite(commanded)):
            raise ValueError("commanded torque must be a finite vector of shape (2,)")
        commanded = np.clip(commanded, -self.torque_limit, self.torque_limit)
        if baseline is None:
            baseline = np.zeros(2, dtype=np.float64)
        baseline = np.asarray(baseline, dtype=np.float64)
        if baseline.shape != (2,) or not np.all(np.isfinite(baseline)):
            raise ValueError("baseline torque must be a finite vector of shape (2,)")

        delayed = self._delayed_command(commanded)
        applied = delayed * self.motor_gain
        pre_step_state = self.state.copy()
        pre_step_acceleration = self.plant_acceleration(applied)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        self.state = np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float64)
        self.steps += 1

        q, qd = self.state[:2], self.state[2:]
        ee = self.arm.forward_kinematics(q)
        distance = float(np.linalg.norm(self.target - ee))
        success = distance <= self.success_radius and float(np.linalg.norm(qd)) <= 0.35
        terminated = bool(success)
        truncated = self.steps >= self.max_steps and not terminated
        reward = -distance - 0.01 * float(qd @ qd) - 0.0002 * float(commanded @ commanded)
        if success:
            reward += 10.0

        info = self._info_base()
        info.update(
            {
                "distance": distance,
                "success": success,
                "baseline_torque": baseline.astype(np.float32),
                "commanded_torque": commanded.astype(np.float32),
                "delayed_torque": delayed.astype(np.float32),
                "applied_torque": applied.astype(np.float32),
                "commanded_torque_exact": commanded.copy(),
                "delayed_torque_exact": delayed.copy(),
                "actuator_scaled_torque": applied.copy(),
                "plant_input_torque": applied.copy(),
                "pre_step_state": pre_step_state,
                "pre_step_acceleration": pre_step_acceleration,
                "plant": "mujoco",
            }
        )
        return self._observation(), float(reward), terminated, truncated, info

    def constructor_config(self) -> dict:
        config = super().constructor_config()
        config.update({"timestep": self.timestep, "integrator": self.integrator})
        return config
