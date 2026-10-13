"""Causal command-delay identification and queued-command prediction for the Panda."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sarrl.controllers.panda_adaptive import (
    PandaAdaptiveController,
    PandaEstimatorConfig,
    PandaRLS,
    parameters8,
)
from sarrl.robots.panda import mujoco, vector7


@dataclass(frozen=True)
class PandaDelayConfig:
    max_delay_steps: int = 4
    control_dt: float = 0.01
    prediction_substeps: int = 5
    max_position_shift_rad: float = 0.5
    max_velocity_shift_rad_s: float = 3.0

    def validate(self):
        if type(self.max_delay_steps) is not int or self.max_delay_steps < 0:
            raise ValueError("maximum delay must be a non-negative integer")
        if type(self.prediction_substeps) is not int or self.prediction_substeps < 1:
            raise ValueError("prediction substeps must be a positive integer")
        values = [self.control_dt, self.max_position_shift_rad, self.max_velocity_shift_rad_s]
        if not np.all(np.isfinite(values)) or np.any(np.asarray(values) <= 0.0):
            raise ValueError("prediction intervals and shift limits must be finite and positive")


def _cached_mass_matrix(dynamics):
    matrix = np.empty((7, 7))
    for joint, basis in enumerate(np.eye(7)):
        column = np.empty(7)
        mujoco.mj_mulM(dynamics.model, dynamics.data, column, basis)
        matrix[:, joint] = column
    return matrix


class PandaDelayController(PandaAdaptiveController):
    """One bounded RLS per delay hypothesis; no plant object or delivered torque input."""

    def __init__(
        self,
        config: PandaDelayConfig | None = None,
        estimator_config: PandaEstimatorConfig | None = None,
        *,
        known_delay_steps: int | None = None,
        predict: bool = False,
        kp=100.0,
        kd=20.0,
    ):
        self.delay_config = config or PandaDelayConfig()
        self.delay_config.validate()
        if type(predict) is not bool:
            raise ValueError("predict must be a boolean")
        if known_delay_steps is not None and (
            type(known_delay_steps) is not int
            or not 0 <= known_delay_steps <= self.delay_config.max_delay_steps
        ):
            raise ValueError("known delay must be an integer within the configured hypothesis bank")
        super().__init__(estimator_config, kp=kp, kd=kd)
        self.lags = (
            tuple(range(self.delay_config.max_delay_steps + 1))
            if known_delay_steps is None
            else (known_delay_steps,)
        )
        self.estimators = [PandaRLS(estimator_config) for _ in self.lags]
        self.scores = np.zeros(len(self.lags))
        self.selected = 0
        self.estimator = self.estimators[0]
        self.dynamics.estimator = self.estimator
        self.known_delay_steps = known_delay_steps
        self.predict = predict
        self.sent_history = []
        self.selected_history = []
        self.selection_switches = 0
        self.prediction_calls = self.prediction_substeps = self.prediction_fallbacks = 0
        self.last_lookahead_steps = 0
        self.last_control_state = None
        self.last_prediction_fallback = False

    @property
    def selected_delay_steps(self):
        return self.lags[self.selected]

    def observe_transition(self, q, qd, q_next, qd_next, sent_command, dt):
        if not np.isfinite(dt) or not np.isclose(
            dt, self.delay_config.control_dt, rtol=0, atol=1e-12
        ):
            raise ValueError("transition interval must match the configured control period")
        q, qd = vector7(q, "q"), vector7(qd, "qd")
        q_next, qd_next = vector7(q_next, "q_next"), vector7(qd_next, "qd_next")
        sent = vector7(sent_command, "sent_command")
        if np.any(np.abs(sent) > self.dynamics.torque_limit + 1e-12):
            raise ValueError("the command history requires sent clipped commands")
        nominal, rows = self.regressor.evaluate(
            0.5 * (q + q_next), 0.5 * (qd + qd_next), (qd_next - qd) / dt
        )
        self.sent_history.append(sent.copy())
        for index, (lag, estimator) in enumerate(zip(self.lags, self.estimators, strict=True)):
            history_index = len(self.sent_history) - 1 - lag
            aligned = self.sent_history[history_index] if history_index >= 0 else np.zeros(7)
            residual = aligned - nominal
            innovation = residual - rows @ estimator.parameters
            self.scores[index] += float(innovation @ innovation)
            estimator.update(rows, residual)
        selected = int(np.argmin(self.scores))
        self.selection_switches += int(selected != self.selected)
        self.selected = selected
        self.estimator = self.estimators[selected]
        self.dynamics.estimator = self.estimator
        self.selected_history.append(self.selected_delay_steps)

    def estimated_acceleration(self, q, qd, torque, parameters=None):
        theta = parameters8(self.estimator.parameters if parameters is None else parameters)
        nominal, rows = self.regressor.evaluate(q, qd, np.zeros(7))
        m0 = _cached_mass_matrix(self.regressor.nominal)
        m1 = _cached_mass_matrix(self.regressor.unit_payload)
        matrix = m0 + theta[0] * (m1 - m0)
        return np.linalg.solve(matrix, vector7(torque, "torque") - nominal - rows @ theta)

    def _integrate_interval(self, state, torque, initial):
        cfg = self.delay_config
        dt = cfg.control_dt / cfg.prediction_substeps

        def derivative(x):
            return np.concatenate([x[7:], self.estimated_acceleration(x[:7], x[7:], torque)])

        for _ in range(cfg.prediction_substeps):
            k1 = derivative(state)
            k2 = derivative(state + 0.5 * dt * k1)
            k3 = derivative(state + 0.5 * dt * k2)
            k4 = derivative(state + dt * k3)
            state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            self.prediction_substeps += 1
            if (
                not np.all(np.isfinite(state))
                or np.max(np.abs(state[:7] - initial[:7])) > cfg.max_position_shift_rad
                or np.max(np.abs(state[7:] - initial[7:])) > cfg.max_velocity_shift_rad_s
            ):
                raise ValueError("predicted state left the configured displacement limits")
        return state

    def prepare_control(self, q, qd):
        initial = np.concatenate([vector7(q, "q"), vector7(qd, "qd")])
        state = initial.copy()
        lag = self.selected_delay_steps if self.predict else 0
        self.last_prediction_fallback = False
        if lag:
            self.prediction_calls += 1
            try:
                for offset in range(lag, 0, -1):
                    index = len(self.sent_history) - offset
                    queued = self.sent_history[index] if index >= 0 else np.zeros(7)
                    state = self._integrate_interval(state, queued, initial)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                state = initial.copy()
                lag = 0
                self.prediction_fallbacks += 1
                self.last_prediction_fallback = True
        self.last_lookahead_steps = lag
        self.last_control_state = state.copy()
        return state[:7].copy(), state[7:].copy(), lag * self.delay_config.control_dt

    def control_diagnostics(self):
        return {
            "selected_delay_steps": self.selected_delay_steps,
            "prediction_lookahead_s": self.last_lookahead_steps * self.delay_config.control_dt,
            "prediction_fallback": self.last_prediction_fallback,
            "control_state": self.last_control_state.tolist()
            if self.last_control_state is not None
            else None,
        }

    def delay_diagnostics(self):
        return {
            "lags": list(self.lags),
            "scores": self.scores.tolist(),
            "selected_delay_steps": self.selected_delay_steps,
            "selected_history": self.selected_history.copy(),
            "selection_switches": self.selection_switches,
            "candidate_estimates": [e.parameters.tolist() for e in self.estimators],
            "candidate_bound_hits": [e.bound_hits for e in self.estimators],
            "prediction_calls": self.prediction_calls,
            "prediction_virtual_substeps": self.prediction_substeps,
            "prediction_fallbacks": self.prediction_fallbacks,
        }
