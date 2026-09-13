"""Computed torque on a command-space plant model identified online.

The planar arm is linear in its inertial and friction parameters, and the
actuator scales the delayed command by a per-joint motor gain. Dividing the
joint-i torque equation by that gain gives a relation that is still linear in
the unknowns and whose left-hand side is the command the controller sent:

    u_i(k - L) = Y_i(q, qd, qdd) (theta / g_i) + friction_i(qd) / g_i

Each joint therefore has its own parameter vector, already expressed in the
units the controller needs, and no separate gain estimate is required. The
unknown actuator delay L is handled by running one recursive least-squares
estimator per candidate lag and using the lag with the smallest recent
prediction error. Acceleration is the finite difference of the measured
velocity across one step, evaluated with the regressor at the midpoint state,
so the estimator only consumes quantities a real controller could observe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sarrl.dynamics import PlanarArm, PlanarArmParams

from .computed_torque import _angle_error

PARAMETER_NAMES = ("m1", "m2", "i1", "i2", "payload", "viscous", "coulomb")


def _vector(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite pair")
    return result


class CommandRegressor:
    """Rows of the torque equation that multiply the per-joint parameter vector."""

    def __init__(self, geometry: PlanarArmParams | None = None):
        self.geometry = geometry or PlanarArmParams()
        self.geometry.validate()

    def inertial_matrices(self, q):
        """Mass-matrix contribution of each inertial parameter, shape (5, 2, 2)."""
        p = self.geometry
        c2 = np.cos(q[1])
        m2_block = np.array(
            [
                [p.l1**2 + p.lc2**2 + 2.0 * p.l1 * p.lc2 * c2, p.lc2**2 + p.l1 * p.lc2 * c2],
                [p.lc2**2 + p.l1 * p.lc2 * c2, p.lc2**2],
            ]
        )
        payload_block = np.array(
            [
                [p.l1**2 + p.l2**2 + 2.0 * p.l1 * p.l2 * c2, p.l2**2 + p.l1 * p.l2 * c2],
                [p.l2**2 + p.l1 * p.l2 * c2, p.l2**2],
            ]
        )
        return np.stack(
            [
                np.array([[p.lc1**2, 0.0], [0.0, 0.0]]),
                m2_block,
                np.array([[1.0, 0.0], [0.0, 0.0]]),
                np.ones((2, 2)),
                payload_block,
            ]
        )

    def rows(self, q, qd, qdd):
        """Regressor rows Y such that Y[i] @ theta_i equals the joint-i torque."""
        q = _vector(q, "q")
        qd = _vector(qd, "qd")
        qdd = _vector(qdd, "qdd")
        p = self.geometry
        s2 = np.sin(q[1])
        c1 = np.cos(q[0])
        c12 = np.cos(q[0] + q[1])
        inertia = self.inertial_matrices(q) @ qdd  # (5, 2)
        # Coriolis and gravity columns, indexed (parameter, joint).
        coriolis = np.zeros((5, 2))
        for index, coefficient in ((1, p.l1 * p.lc2), (4, p.l1 * p.l2)):
            h = coefficient * s2
            coriolis[index] = (-h * (2.0 * qd[0] * qd[1] + qd[1] ** 2), h * qd[0] ** 2)
        gravity = np.zeros((5, 2))
        gravity[0] = (p.lc1 * p.gravity * c1, 0.0)
        gravity[1] = (p.l1 * p.gravity * c1 + p.lc2 * p.gravity * c12, p.lc2 * p.gravity * c12)
        gravity[4] = (p.l1 * p.gravity * c1 + p.l2 * p.gravity * c12, p.l2 * p.gravity * c12)
        inertial = (inertia + coriolis + gravity).T  # (2, 5)
        friction = np.stack([qd, np.tanh(qd / p.friction_smoothing)], axis=1)  # (2, 2)
        return np.concatenate([inertial, friction], axis=1)

    @staticmethod
    def parameters(params: PlanarArmParams, motor_gain=(1.0, 1.0)):
        """Per-joint command-space parameter vectors of a known plant, shape (2, 7)."""
        gain = _vector(motor_gain, "motor gain")
        rows = []
        for joint in range(2):
            theta = np.array(
                [
                    params.m1,
                    params.m2,
                    params.i1,
                    params.i2,
                    params.payload_mass,
                    params.viscous[joint],
                    params.coulomb[joint],
                ]
            )
            rows.append(theta / gain[joint])
        return np.stack(rows)


@dataclass(frozen=True)
class AdaptiveNominalConfig:
    kp: tuple[float, float] = (36.0, 36.0)
    kd: tuple[float, float] = (12.0, 12.0)
    torque_limit: tuple[float, float] = (40.0, 40.0)
    dt: float = 0.02
    forgetting: float = 1.0
    process_noise: float = 0.02
    max_lag: int = 3
    lag_min_updates: int = 10
    # Candidate first-order actuator time constants, in seconds. Each is paired
    # with every lag into one hypothesis whose regression target is the command
    # the actuator would deliver under that lag and time constant. Zero is the
    # ideal actuator and reproduces the v1.9 behaviour.
    actuator_time_constants: tuple[float, ...] = (0.0,)
    actuator_substeps: int = 10
    selection_memory: float = 1.0
    payload_prior: float = 0.0
    filter_gate: bool = False
    gate_threshold: float = 1.0
    gate_min_updates: int = 5
    prediction_limit: float = 5.0
    conditioning_floor: float = 0.05
    prior_std: tuple[float, ...] = (1.0, 1.0, 0.1, 0.1, 2.0, 0.1, 0.04)
    lower: tuple[float, ...] = (0.2, 0.2, 0.01, 0.01, 0.0, 0.0, 0.0)
    upper: tuple[float, ...] = (5.0, 5.0, 0.5, 0.5, 6.0, 0.5, 0.2)

    def validate(self) -> None:
        if not 0.0 < self.forgetting <= 1.0 or not 0.0 < self.selection_memory <= 1.0:
            raise ValueError("forgetting and selection_memory must lie in (0, 1]")
        if not np.isfinite(self.process_noise) or self.process_noise < 0.0:
            raise ValueError("process_noise must be finite and non-negative")
        if self.max_lag < 0 or self.dt <= 0.0:
            raise ValueError("max_lag must be non-negative and dt positive")
        if not np.isfinite(self.payload_prior) or self.payload_prior < 0.0:
            raise ValueError("payload_prior must be finite and non-negative")
        if not np.isfinite(self.gate_threshold) or self.gate_threshold <= 0.0:
            raise ValueError("gate_threshold must be positive")
        if self.gate_min_updates < 1 or self.lag_min_updates < 1:
            raise ValueError("gate_min_updates and lag_min_updates must be at least one")
        for name in ("prior_std", "lower", "upper"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != (7,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must hold seven finite values")
        if np.any(np.asarray(self.prior_std) <= 0.0):
            raise ValueError("prior_std must be positive")
        if np.any(np.asarray(self.lower) >= np.asarray(self.upper)):
            raise ValueError("lower bounds must be below upper bounds")
        if self.prediction_limit <= 0.0 or not 0.0 < self.conditioning_floor < 1.0:
            raise ValueError("prediction_limit must be positive and conditioning_floor in (0, 1)")
        taus = np.asarray(self.actuator_time_constants, dtype=np.float64)
        if taus.ndim != 1 or taus.size == 0 or np.any(taus < 0.0) or not np.all(np.isfinite(taus)):
            raise ValueError("actuator_time_constants must be non-negative finite values")
        if self.actuator_substeps < 1:
            raise ValueError("actuator_substeps must be at least one")


class AdaptiveNominalController:
    """Computed torque whose model is re-estimated from each executed command."""

    def __init__(self, model: PlanarArm, config: AdaptiveNominalConfig | None = None):
        self.model = model
        self.config = config or AdaptiveNominalConfig()
        self.config.validate()
        self.regressor = CommandRegressor(model.params)
        self.kp = np.asarray(self.config.kp, dtype=np.float64)
        self.kd = np.asarray(self.config.kd, dtype=np.float64)
        self.torque_limit = np.asarray(self.config.torque_limit, dtype=np.float64)
        self.prior = CommandRegressor.parameters(model.params)
        # The prior payload is a design choice: the nominal arm carries none,
        # but the randomised plant usually does.
        self.prior[:, 4] = self.config.payload_prior
        self._lower = np.asarray(self.config.lower, dtype=np.float64)
        self._upper = np.asarray(self.config.upper, dtype=np.float64)
        self._p0 = np.diag(np.asarray(self.config.prior_std, dtype=np.float64) ** 2)
        self.reset()

    # Estimator state -------------------------------------------------------

    def reset(self) -> None:
        cfg = self.config
        # Hypothesis h = tau_index * (max_lag + 1) + lag; the default grid of one
        # zero time constant makes hypotheses and lags coincide.
        self.hypotheses = [
            (lag, float(tau))
            for tau in cfg.actuator_time_constants
            for lag in range(cfg.max_lag + 1)
        ]
        count = len(self.hypotheses)
        self.theta = np.repeat(self.prior[None], count, axis=0)  # (count, 2, 7)
        self.covariance = np.repeat(np.stack([self._p0, self._p0])[None], count, axis=0)
        self.error_score = np.zeros(count)
        self.recent_score = np.zeros(count)
        self._delivered = np.zeros((count, 2))
        self.updates = 0
        self._converged = False
        self.prediction_fallbacks = 0
        self.model_fallbacks = 0
        self.model_fallback_steps = 0
        self.control_steps = 0
        self._step_uses_nominal = False
        self._sent: list[np.ndarray] = []

    def state_dict(self) -> dict:
        """Every mutable field, so a training session can checkpoint mid-episode."""
        return {
            "theta": self.theta.copy(),
            "covariance": self.covariance.copy(),
            "error_score": self.error_score.copy(),
            "recent_score": self.recent_score.copy(),
            "delivered": self._delivered.copy(),
            "updates": int(self.updates),
            "converged": bool(self._converged),
            "prediction_fallbacks": int(self.prediction_fallbacks),
            "model_fallbacks": int(self.model_fallbacks),
            "model_fallback_steps": int(self.model_fallback_steps),
            "control_steps": int(self.control_steps),
            "step_uses_nominal": bool(self._step_uses_nominal),
            "sent": [x.copy() for x in self._sent],
        }

    def load_state_dict(self, state: dict) -> None:
        count = len(self.hypotheses)
        shapes = {
            "theta": (count, 2, 7),
            "covariance": (count, 2, 7, 7),
            "error_score": (count,),
            "recent_score": (count,),
            "delivered": (count, 2),
        }
        values = {}
        for name, shape in shapes.items():
            value = np.asarray(state[name], dtype=np.float64)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"invalid estimator checkpoint field {name}")
            values[name] = value.copy()
        sent = [np.asarray(x, dtype=np.float64).copy() for x in state["sent"]]
        if any(x.shape != (2,) or not np.all(np.isfinite(x)) for x in sent):
            raise ValueError("invalid estimator command history")
        self.theta = values["theta"]
        self.covariance = values["covariance"]
        self.error_score = values["error_score"]
        self.recent_score = values["recent_score"]
        self._delivered = values["delivered"]
        self.updates = int(state["updates"])
        self._converged = bool(state["converged"])
        self.prediction_fallbacks = int(state["prediction_fallbacks"])
        self.model_fallbacks = int(state["model_fallbacks"])
        self.model_fallback_steps = int(state["model_fallback_steps"])
        self.control_steps = int(state["control_steps"])
        self._step_uses_nominal = bool(state["step_uses_nominal"])
        self._sent = sent

    @property
    def hypothesis(self) -> int:
        """Index of the (lag, time constant) hypothesis that explains the data best."""
        if self.updates == 0:
            return 0
        return int(np.argmin(self.error_score))

    @property
    def lag(self) -> int:
        """Actuator lag of the selected hypothesis."""
        return self.hypotheses[self.hypothesis][0]

    @property
    def time_constant(self) -> float:
        """Actuator time constant of the selected hypothesis."""
        return self.hypotheses[self.hypothesis][1]

    def _delivered_mean(self, state, command, tau):
        """Mean torque an actuator with time constant tau delivers over one control step.

        Mirrors a first-order filter advanced on `actuator_substeps` sub-steps
        with the command held constant; returns the mean and the final state.
        """
        if tau == 0.0:
            return command.copy(), command.copy()
        substep = self.config.dt / self.config.actuator_substeps
        alpha = substep / (tau + substep)
        total = np.zeros(2)
        current = state.copy()
        for _ in range(self.config.actuator_substeps):
            current = current + alpha * (command - current)
            total += current
        return total / self.config.actuator_substeps, current

    @property
    def prediction_lag(self) -> int:
        """Lag trusted for state prediction; zero until every hypothesis has data.

        With seven parameters and a live covariance, each hypothesis fits its
        first few targets almost exactly, so the argmin over lags is noise until
        the regressors have accumulated. Using the best-fitting parameters for
        the command is harmless then; walking the state through a wrong queue
        is not, so prediction waits.
        """
        if self.updates < self.config.lag_min_updates:
            return 0
        return self.lag

    @property
    def converged(self) -> bool:
        """Latched once the selected lag's recent squared innovation drops below the gate.

        The recent score is an exponential average of the squared innovation
        with memory 0.9, kept apart from the cumulative lag-selection score.

        The latch matters after an in-episode plant change: the innovation rises
        again while the estimate re-adapts, and during those steps a briefly stale
        estimate is still closer to the plant than the nominal model.
        """
        return self._converged

    @property
    def parameters(self) -> np.ndarray:
        """Per-joint command-space parameters of the selected hypothesis, shape (2, 7)."""
        return self.theta[self.hypothesis].copy()

    def observe(self, state_before, state_after, sent_torque) -> dict:
        """Update every lag hypothesis with one executed transition."""
        before = np.asarray(state_before, dtype=np.float64)
        after = np.asarray(state_after, dtype=np.float64)
        if before.shape != (4,) or after.shape != (4,):
            raise ValueError("states must have shape (4,)")
        if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
            raise ValueError("states must be finite")
        sent = _vector(sent_torque, "sent torque")
        self._sent.append(sent.copy())
        cfg = self.config
        mid = 0.5 * (before + after)
        acceleration = (after[2:] - before[2:]) / cfg.dt
        phi = self.regressor.rows(mid[:2], mid[2:], acceleration)  # (2, 7)
        innovations = np.zeros((len(self.hypotheses), 2))
        for lag, (delay, tau) in enumerate(self.hypotheses):
            index = len(self._sent) - 1 - delay
            # The plant queue starts with zero commands, exactly as the environment does.
            delayed = self._sent[index] if index >= 0 else np.zeros(2)
            target, self._delivered[lag] = self._delivered_mean(self._delivered[lag], delayed, tau)
            for joint in range(2):
                row = phi[joint]
                theta = self.theta[lag, joint]
                cov = self.covariance[lag, joint]
                error = float(target[joint] - row @ theta)
                innovations[lag, joint] = error
                gain = cov @ row / (cfg.forgetting + row @ cov @ row)
                theta = np.clip(theta + gain * error, self._lower, self._upper)
                # Random-walk parameters: the additive term keeps the gain alive
                # after convergence so that an abrupt plant change is tracked
                # within a few excited steps instead of the slow 1/forgetting regrowth.
                cov = (cov - np.outer(gain, row @ cov)) / cfg.forgetting
                cov = 0.5 * (cov + cov.T) + cfg.process_noise * self._p0
                self.theta[lag, joint] = theta
                self.covariance[lag, joint] = cov
            # The actuator lag does not change within an episode, so the score
            # accumulates by default: a wrong-lag hypothesis with a live covariance
            # keeps chasing the data with small one-step errors, and only their
            # sum over the episode separates it from the right one.
            squared = float(innovations[lag] @ innovations[lag])
            self.error_score[lag] = cfg.selection_memory * self.error_score[lag] + squared
            self.recent_score[lag] = 0.9 * self.recent_score[lag] + 0.1 * squared
        self.updates += 1
        settled = self.recent_score[self.hypothesis] < cfg.gate_threshold
        if self.updates >= cfg.gate_min_updates and settled:
            self._converged = True
        return {
            "lag": self.lag,
            "time_constant": self.time_constant,
            "innovation": innovations[self.hypothesis].copy(),
            "error_score": self.error_score.copy(),
        }

    # Control law -----------------------------------------------------------

    def begin_step(self, q) -> bool:
        """Decide once per control step whether the estimate is usable.

        Called with the state the step starts from (the measured one when the
        interface is measured). The decision holds for every model query of
        the step: filter rows, drift, and every RK4 stage of the prediction.
        Returns True when the nominal model is used for the step.
        """
        q = _vector(q, "q")
        self.control_steps += 1
        estimated = self.estimated_model()._estimated_mass_matrix(q)
        nominal = self.model.mass_matrix(q)
        det = float(np.linalg.det(estimated))
        floor = self.config.conditioning_floor * float(np.linalg.det(nominal))
        self._step_uses_nominal = not np.isfinite(det) or det < floor
        if self._step_uses_nominal:
            self.model_fallbacks += 1
            self.model_fallback_steps += 1
        return self._step_uses_nominal

    def command(self, q, qd, q_des, qd_des=(0.0, 0.0), qdd_des=(0.0, 0.0)) -> np.ndarray:
        q = _vector(q, "q")
        qd = _vector(qd, "qd")
        q_des = _vector(q_des, "q_des")
        qd_des = _vector(qd_des, "qd_des")
        qdd_des = _vector(qdd_des, "qdd_des")
        qdd_cmd = qdd_des + self.kd * (qd_des - qd) + self.kp * _angle_error(q_des, q)
        rows = self.regressor.rows(q, qd, qdd_cmd)
        theta = self.theta[self.hypothesis]
        tau = np.einsum("ij,ij->i", rows, theta)
        return np.clip(tau, -self.torque_limit, self.torque_limit)

    def predict_state(self, state, steps: int | None = None) -> np.ndarray:
        """Propagate the state through the commands still queued in the actuator.

        With an identified lag L, the command computed now acts L steps from now,
        after the L commands already sent have taken effect. Integrating the
        estimated model over those known commands gives the state at which the
        new command will act, so both the control law and the filter can be
        evaluated there instead of at the current state.
        """
        x = np.asarray(state, dtype=np.float64)
        if x.shape != (4,) or not np.all(np.isfinite(x)):
            raise ValueError("state must be a finite vector of shape (4,)")
        lag = self.prediction_lag if steps is None else int(steps)
        if lag == 0:
            return x.copy()
        model = self.estimated_model()
        history = len(self._sent)
        start = x.copy()
        tau = self.time_constant
        delivered = self._delivered[self.hypothesis].copy()
        for offset in range(lag, 0, -1):
            index = history - offset
            queued = self._sent[index] if index >= 0 else np.zeros(2)
            applied, delivered = self._delivered_mean(delivered, queued, tau)
            x = _rk4(model, x, applied, self.config.dt)
        if not np.all(np.isfinite(x)) or np.max(np.abs(x - start)) > self.config.prediction_limit:
            # A degenerate estimate must not propagate; no compensation is safer
            # than a wild prediction.
            self.prediction_fallbacks += 1
            return start
        return x

    def estimated_model(self) -> EstimatedCommandModel:
        """Live model view for a safety filter, expressed in command units."""
        return EstimatedCommandModel(self)


def _rk4(model, state, command, dt):
    def derivative(x):
        return np.concatenate([x[2:], model.forward_dynamics(x[:2], x[2:], command)])

    k1 = derivative(state)
    k2 = derivative(state + 0.5 * dt * k1)
    k3 = derivative(state + 0.5 * dt * k2)
    k4 = derivative(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class EstimatedCommandModel:
    """Plant model in command coordinates, read live from the estimator.

    Exposes the subset of the `PlanarArm` interface the HOCBF filter uses. The
    mass matrix has row i scaled by 1 / g_i, so `forward_dynamics` maps the
    command sent to the actuator to the joint acceleration it should produce.
    """

    def __init__(self, controller: AdaptiveNominalController):
        self.controller = controller
        self.kinematics = controller.model

    @property
    def params(self) -> PlanarArmParams:
        return self.kinematics.params

    @property
    def uses_nominal(self) -> bool:
        """Nominal model while gated before convergence, or when the estimate is degenerate.

        The two per-joint parameter vectors are estimated independently, so the
        command-space mass matrix they imply need not stay well conditioned.
        When its determinant falls below a fraction of the nominal one, the
        filter and the prediction use the nominal model for that step.
        """
        controller = self.controller
        if controller.config.filter_gate and not controller.converged:
            return True
        return controller._step_uses_nominal

    def _estimated_mass_matrix(self, q):
        blocks = self.controller.regressor.inertial_matrices(np.asarray(q, dtype=np.float64))
        theta = self._theta()[:, :5]
        return np.stack(
            [np.tensordot(theta[joint], blocks[:, joint, :], axes=1) for joint in range(2)]
        )

    def _theta(self) -> np.ndarray:
        return self.controller.theta[self.controller.hypothesis]

    def mass_matrix(self, q) -> np.ndarray:
        q = _vector(q, "q")
        if self.uses_nominal:
            return self.kinematics.mass_matrix(q)
        blocks = self.controller.regressor.inertial_matrices(q)  # (5, 2, 2)
        theta = self._theta()[:, :5]  # (2, 5)
        return np.stack(
            [np.tensordot(theta[joint], blocks[:, joint, :], axes=1) for joint in range(2)]
        )

    def bias(self, q, qd) -> np.ndarray:
        """Command needed at zero acceleration: Coriolis, gravity and friction."""
        rows = self.controller.regressor.rows(q, qd, np.zeros(2))
        return np.einsum("ij,ij->i", rows, self._theta())

    def inverse_dynamics(self, q, qd, qdd, include_friction: bool = True) -> np.ndarray:
        if self.uses_nominal:
            return self.kinematics.inverse_dynamics(q, qd, qdd, include_friction=include_friction)
        rows = self.controller.regressor.rows(q, qd, qdd)
        theta = self._theta()
        if not include_friction:
            theta = theta.copy()
            theta[:, 5:] = 0.0
        return np.einsum("ij,ij->i", rows, theta)

    def forward_dynamics(self, q, qd, tau, include_friction: bool = True) -> np.ndarray:
        tau = _vector(tau, "tau")
        bias = self.inverse_dynamics(q, qd, np.zeros(2), include_friction=include_friction)
        return np.linalg.solve(self.mass_matrix(q), tau - bias)

    def forward_kinematics(self, q):
        return self.kinematics.forward_kinematics(q)

    def jacobian(self, q):
        return self.kinematics.jacobian(q)

    def jacobian_dot_times_qd(self, q, qd):
        return self.kinematics.jacobian_dot_times_qd(q, qd)
