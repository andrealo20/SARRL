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
from sarrl.envs.planar_reach import DomainRandomization, FaultSpec, ObstacleSpec, PlanarReachEnv

try:  # MuJoCo is an optional dependency of the repository.
    import mujoco
except ImportError:  # pragma: no cover - exercised only without the extra installed
    mujoco = None

PAYLOAD_INERTIA_FLOOR = 1e-9
PAYLOAD_MASS_FLOOR = 1e-9
# Where the obstacle geom rests in episodes that carry no obstacle.
OBSTACLE_PARKING = (100.0, 100.0, 0.0)


def planar_arm_xml(
    params: PlanarArmParams,
    timestep: float,
    integrator: str,
    armature: float = 0.0,
    obstacle: ObstacleSpec | None = None,
) -> str:
    """MuJoCo model of the two-link arm with the analytical benchmark's geometry.

    Gravity acts along -y in the plane of motion and joint angles are measured
    from the +x axis, matching `PlanarArm.gravity_vector`. Inertias are given
    about each link's centre of mass, as in `PlanarArmParams`.

    Without an obstacle nothing collides, as in every retained campaign. With
    one, the link capsules and a tip sphere collide with a static cylinder
    whose position is set per episode; the arm never collides with itself.
    """
    p = params
    payload = max(p.payload_mass, PAYLOAD_MASS_FLOOR)
    floor = PAYLOAD_INERTIA_FLOOR
    arm_collision = 'contype="2" conaffinity="1"' if obstacle else 'contype="0" conaffinity="0"'
    tip = (
        f'<geom name="tip" type="sphere" size="{obstacle.tip_radius}" mass="0" {arm_collision}/>'
        if obstacle
        else ""
    )
    parking = f"{OBSTACLE_PARKING[0]} {OBSTACLE_PARKING[1]}"
    cylinder = (
        f'<geom name="obstacle" type="cylinder" pos="{parking} 0" '
        f'size="{obstacle.radius} 0.2" contype="1" conaffinity="2"/>'
        if obstacle
        else ""
    )
    return f"""
<mujoco model="sarrl_planar_arm">
  <option timestep="{timestep}" gravity="0 {-p.gravity} 0" integrator="{integrator}"/>
  <worldbody>
    {cylinder}
    <body name="link1" pos="0 0 0">
      <joint name="joint1" type="hinge" axis="0 0 1" damping="{p.viscous[0]}"
             frictionloss="{p.coulomb[0]}" armature="{armature}" limited="false"/>
      <inertial pos="{p.lc1} 0 0" mass="{p.m1}" diaginertia="{p.i1} {p.i1} {p.i1}"/>
      <geom name="link1" type="capsule" fromto="0 0 0 {p.l1} 0 0" size="0.03" mass="0"
            {arm_collision}/>
      <body name="link2" pos="{p.l1} 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1" damping="{p.viscous[1]}"
               frictionloss="{p.coulomb[1]}" armature="{armature}" limited="false"/>
        <inertial pos="{p.lc2} 0 0" mass="{p.m2}" diaginertia="{p.i2} {p.i2} {p.i2}"/>
        <geom name="link2" type="capsule" fromto="0 0 0 {p.l2} 0 0" size="0.03" mass="0"
              {arm_collision}/>
        <body name="payload" pos="{p.l2} 0 0">
          <inertial pos="0 0 0" mass="{payload}" diaginertia="{floor} {floor} {floor}"/>
          {tip}
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
        armature: float = 0.0,
        actuator_time_constant: float = 0.0,
        armature_range: tuple[float, float] | None = None,
        actuator_time_constant_range: tuple[float, float] | None = None,
        obstacle: ObstacleSpec | None = None,
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
            obstacle=obstacle,
        )
        substeps = self.dt / timestep
        if abs(substeps - round(substeps)) > 1e-9 or round(substeps) < 1:
            raise ValueError("dt must be a positive integer multiple of the MuJoCo timestep")
        if armature < 0.0 or actuator_time_constant < 0.0:
            raise ValueError("armature and actuator_time_constant must be non-negative")
        self.timestep = float(timestep)
        self.substeps = int(round(substeps))
        self.integrator = integrator
        # Reflected motor inertia on each joint and a first-order lag between the
        # torque the actuator is asked for and the torque it delivers. Neither is
        # part of the controller's parametrisation.
        self.armature = float(armature)
        self.actuator_time_constant = float(actuator_time_constant)
        # Constructor values, kept apart from the per-episode draws so that the
        # constructor configuration stays constant across episodes.
        self._armature_base = float(armature)
        self._actuator_time_constant_base = float(actuator_time_constant)
        # Optional per-episode randomisation of the actuator, drawn from a generator
        # seeded apart from the benchmark's so that targets and plant draws stay
        # identical to the analytical environment for the same seed.
        for name, bounds in (
            ("armature_range", armature_range),
            ("actuator_time_constant_range", actuator_time_constant_range),
        ):
            if bounds is not None:
                low, high = bounds
                if not (0.0 <= low <= high) or not np.isfinite(high):
                    raise ValueError(f"{name} must be finite, non-negative and ordered")
        self.armature_range = tuple(armature_range) if armature_range else None
        self.actuator_time_constant_range = (
            tuple(actuator_time_constant_range) if actuator_time_constant_range else None
        )
        self._actuator_rng = np.random.default_rng(0)
        self._actuator_torque = np.zeros(2, dtype=np.float64)
        self.model = mujoco.MjModel.from_xml_string(
            planar_arm_xml(
                self.nominal_arm.params, self.timestep, integrator, self.armature, self.obstacle
            )
        )
        self.data = mujoco.MjData(self.model)
        self._body = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("link1", "link2", "payload")
        }
        self._obstacle_geom = (
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle")
            if self.obstacle is not None
            else None
        )
        self._contact_geoms: set[str] = set()
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
        model.dof_armature[:] = self.armature
        if self._obstacle_geom is not None:
            position = self.obstacles[0].center if self.obstacles else OBSTACLE_PARKING[:2]
            model.geom_pos[self._obstacle_geom] = (position[0], position[1], 0.0)
        mujoco.mj_setConst(model, self.data)
        self.data.qpos[:] = self.state[:2]
        self.data.qvel[:] = self.state[2:]
        self.data.qacc[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(model, self.data)

    def reset(self, seed: int | None = None, target=None):
        result = super().reset(seed=seed, target=target)
        if seed is not None:
            self._actuator_rng = np.random.default_rng(seed ^ 0x5A5A5A5A)
        if self.armature_range is not None:
            self.armature = float(self._actuator_rng.uniform(*self.armature_range))
        if self.actuator_time_constant_range is not None:
            self.actuator_time_constant = float(
                self._actuator_rng.uniform(*self.actuator_time_constant_range)
            )
        self._actuator_torque = np.zeros(2, dtype=np.float64)
        self._sync_plant()
        return result

    def _deliver(self, applied: np.ndarray) -> np.ndarray:
        """Torque the actuator delivers during the coming substep."""
        if self.actuator_time_constant == 0.0:
            self._actuator_torque = applied.copy()
        else:
            alpha = self.timestep / (self.actuator_time_constant + self.timestep)
            self._actuator_torque = self._actuator_torque + alpha * (
                applied - self._actuator_torque
            )
        return self._actuator_torque

    def state_dict(self) -> dict:
        state = super().state_dict()
        state.update(
            {
                "armature_value": self.armature,
                "actuator_time_constant_value": self.actuator_time_constant,
                "actuator_torque": self._actuator_torque.copy(),
                "actuator_rng_state": self._actuator_rng.bit_generator.state,
            }
        )
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        if "armature_value" in state:
            self.armature = float(state["armature_value"])
            self.actuator_time_constant = float(state["actuator_time_constant_value"])
            torque = np.asarray(state["actuator_torque"], dtype=np.float64)
            if torque.shape != (2,) or not np.all(np.isfinite(torque)):
                raise ValueError("invalid actuator torque state")
            self._actuator_torque = torque.copy()
            self._actuator_rng.bit_generator.state = state["actuator_rng_state"]
        self._sync_plant()

    def _activate_fault_if_due(self) -> None:
        was_active = self._fault_active
        super()._activate_fault_if_due()
        if self._fault_active and not was_active:
            self._sync_plant()

    # Plant step --------------------------------------------------------------

    def _record_contacts(self) -> None:
        """Note which arm geoms the engine reports in contact with the obstacle."""
        if self._obstacle_geom is None:
            return
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self._obstacle_geom in (contact.geom1, contact.geom2):
                other = contact.geom2 if contact.geom1 == self._obstacle_geom else contact.geom1
                self._contact_geoms.add(
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other)
                )

    def _obstacle_info(self) -> dict:
        info = super()._obstacle_info()
        if self._obstacle_geom is not None:
            # The engine's contacts replace the geometric tip test, and can
            # involve the links, which the barrier does not cover.
            info["obstacle_contact"] = bool(self._contact_geoms)
            info["obstacle_contact_geoms"] = tuple(sorted(self._contact_geoms))
        return info

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
        pre_step_acceleration = self.plant_acceleration(self._actuator_torque)
        self._contact_geoms = set()
        for _ in range(self.substeps):
            self.data.qfrc_applied[:] = self._deliver(applied)
            mujoco.mj_step(self.model, self.data)
            self._record_contacts()
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
        info.update(self._obstacle_info())
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
                "delivered_torque": self._actuator_torque.copy(),
                "armature": self.armature,
                "actuator_time_constant": self.actuator_time_constant,
                "plant": "mujoco",
            }
        )
        return self._observation(), float(reward), terminated, truncated, info

    def constructor_config(self) -> dict:
        config = super().constructor_config()
        config.update(
            {
                "timestep": self.timestep,
                "integrator": self.integrator,
                "armature": self._armature_base,
                "actuator_time_constant": self._actuator_time_constant_base,
                "armature_range": self.armature_range,
                "actuator_time_constant_range": self.actuator_time_constant_range,
            }
        )
        return config
