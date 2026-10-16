"""Cumulative Panda alignment with a queue-aware retained backup supervisor."""

from __future__ import annotations

import numpy as np

from sarrl.controllers.panda_delay import PandaDelayController
from sarrl.robots.panda import vector7
from sarrl.safety.panda_backup import PandaBackupConfig
from sarrl.safety.panda_supervisor import PandaBackupSupervisor


class PandaBackupController(PandaDelayController):
    def __init__(self, config=None):
        cfg = config or PandaBackupConfig()
        cfg.validate()
        if cfg.control_dt != 0.01 or cfg.max_delay_steps != 4:
            raise ValueError("Panda backup controller retains the five-age, 10 ms reference bank")
        super().__init__(predict=False)
        ranges = self.regressor.nominal.model.jnt_range
        self.supervisor = PandaBackupSupervisor(
            ranges[:, 0], ranges[:, 1], self.dynamics.torque_limit, cfg
        )
        self.pending_sent = None
        self.last_nominal_command = None

    @property
    def aborted(self):
        return self.supervisor.aborted

    @property
    def joint_filter(self):
        return self.supervisor.joint_filter

    def command(self, q, qd, q_ref, qd_ref=None, qdd_ref=None):
        if self.pending_sent is not None:
            raise RuntimeError(
                "observe the previous sent-command transition before another command"
            )
        q, qd = vector7(q, "q"), vector7(qd, "qd")
        self.last_nominal_command = None
        if self.aborted:
            sent = self.supervisor.command(q, qd, None, None, None, self.sent_history[-4:])
        else:
            try:
                nominal = super().command(q, qd, q_ref, qd_ref, qdd_ref)
                self.last_nominal_command = vector7(nominal, "nominal").copy()
                theta = self.estimator.parameters.copy()
                bias = self.regressor.inverse_dynamics(q, qd, np.zeros(7), theta)
                mass = self.regressor.mass_matrix(q, theta)
                np.linalg.cholesky(mass)
                matrix = np.linalg.solve(mass, np.eye(7))
                drift = -matrix @ bias
                self.supervisor.witness.policy(q, matrix, drift)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as failure:
                sent = self.supervisor.command(
                    q,
                    qd,
                    None,
                    None,
                    None,
                    self.sent_history[-4:],
                    model_error=type(failure).__name__,
                )
            else:
                sent = self.supervisor.command(
                    q, qd, nominal, matrix, drift, self.sent_history[-4:]
                )
        self.pending_sent = sent.copy()
        return sent

    def observe_transition(self, q, qd, q_next, qd_next, sent_command, dt):
        if self.pending_sent is None:
            raise RuntimeError("a transition requires one pending sent command")
        if not np.array_equal(vector7(sent_command, "sent_command"), self.pending_sent):
            raise ValueError("observe the supervisor's sent torque, never the delivered torque")
        super().observe_transition(q, qd, q_next, qd_next, sent_command, dt)
        self.pending_sent = None

    def backup_diagnostics(self):
        return self.supervisor.diagnostics()
