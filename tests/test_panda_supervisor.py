from dataclasses import replace

import numpy as np
import pytest

from sarrl.safety.panda_backup import PandaBackupConfig
from sarrl.safety.panda_supervisor import PandaBackupSupervisor, policy_identity


def supervisor(config=None):
    return PandaBackupSupervisor(np.full(7, -2.0), np.full(7, 2.0), np.full(7, 20.0), config)


def test_startup_verifies_backup_before_accepting_a_proposal():
    control = supervisor()
    sent = control.command(np.zeros(7), np.zeros(7), np.zeros(7), np.eye(7), np.zeros(7), [])
    np.testing.assert_array_equal(sent, np.zeros(7))
    assert control.last["initial_witness"]["accepted"]
    assert control.last["proposal_witness"]["accepted"]
    assert control.last["mode"] == "task"
    assert control.policy_origin == "proposal" and control.policy_step == 0
    assert control.witness_calls == 2 and control.steps == 1


def test_rejection_uses_last_accepted_context_despite_a_new_estimated_model():
    control = supervisor(replace(PandaBackupConfig(), horizon_steps=1))
    first = control.command(np.zeros(7), np.zeros(7), np.zeros(7), np.eye(7), np.zeros(7), [])
    retained, digest = control.retained_policy, policy_identity(control.retained_policy)
    q, v = np.full(7, 0.001), np.ones(7)
    sent = control.command(q, v, np.zeros(7), 0.5 * np.eye(7), np.full(7, 0.2), [first])
    assert control.abort_reason == "proposal_witness_rejected" and control.abort_step == 1
    assert control.retained_policy is retained
    assert control.last["retained_policy_sha256"] == digest
    np.testing.assert_allclose(sent, -25 * q - 10 * v)
    assert control.last["mode"] == "backup"
    count = control.witness_calls
    followup = control.command(q, v, None, None, None, [first, sent])
    np.testing.assert_array_equal(followup, sent)
    assert control.witness_calls == count and control.backup_steps == 2


def test_failed_initial_witness_is_emergency_and_never_retained_as_accepted():
    control = supervisor()
    q = np.full(7, 1.98)
    sent = control.command(q, np.zeros(7), np.zeros(7), np.eye(7), np.zeros(7), [])
    assert control.abort_reason == "unrecoverable_initial_witness"
    assert control.retained_policy is None and control.emergency_policy is not None
    assert control.last["mode"] == "emergency" and control.last["proposal_witness"] is None
    assert not control.last["initial_witness"]["accepted"]
    followup = control.command(q, np.full(7, 3.0), None, None, None, [sent])
    np.testing.assert_array_equal(followup, np.full(7, -20))
    assert control.last["saturated"] and control.saturated_steps == 1


def test_unbounded_initial_feedback_is_not_a_witnessed_backup():
    control = supervisor()
    sent = control.command(np.zeros(7), np.full(7, 3), np.zeros(7), np.eye(7), np.zeros(7), [])
    assert control.abort_reason == "unrecoverable_initial_torque"
    assert control.last["initial_witness"] is None and control.witness_calls == 0
    assert control.last["mode"] == "emergency"
    np.testing.assert_array_equal(sent, np.full(7, -20))


def test_a_model_error_without_a_retained_policy_sends_bounded_zero_emergency():
    control = supervisor()
    sent = control.command(np.zeros(7), np.zeros(7), None, None, None, [], model_error="ValueError")
    assert control.abort_reason == "estimated_model_error"
    assert control.last["mode"] == "emergency"
    np.testing.assert_array_equal(sent, np.zeros(7))


def test_a_model_error_after_acceptance_preserves_the_previous_policy():
    control = supervisor()
    sent = control.command(np.zeros(7), np.zeros(7), np.zeros(7), np.eye(7), np.zeros(7), [])
    retained = control.retained_policy
    followup = control.command(
        np.zeros(7), np.ones(7), None, None, None, [sent], model_error="ValueError"
    )
    np.testing.assert_array_equal(followup, np.full(7, -10))
    assert control.retained_policy is retained and control.last["mode"] == "backup"


def test_projection_rejection_latches_and_records_requested_and_clipped_backup():
    control = supervisor()
    control.command(np.zeros(7), np.zeros(7), np.zeros(7), np.eye(7), np.zeros(7), [])
    v = np.full(7, 2.07)
    control.command(np.zeros(7), v, np.full(7, -20), np.eye(7), np.zeros(7), [np.full(7, 20)])
    assert control.abort_reason == "projection_pending_command_violation"
    assert control.last["proposal_witness"] is None
    assert control.last["saturated"]
    assert min(control.last["raw_torque_nm"]) < -20
    assert min(control.last["torque_nm"]) == -20


def test_controller_observes_only_its_sent_command_and_updates_all_five_estimators():
    pytest.importorskip("mujoco")
    from sarrl.controllers.panda_backup import PandaBackupController
    from sarrl.robots.panda import HOME

    control = PandaBackupController()
    sent = control.command(HOME, np.zeros(7), HOME)
    with pytest.raises(RuntimeError, match="previous"):
        control.command(HOME, np.zeros(7), HOME)
    with pytest.raises(ValueError, match="sent torque"):
        control.observe_transition(HOME, np.zeros(7), HOME, np.zeros(7), sent + 0.001, 0.01)
    control.observe_transition(HOME, np.zeros(7), HOME, np.zeros(7), sent, 0.01)
    assert all(estimator.updates == 1 for estimator in control.estimators)
    np.testing.assert_array_equal(control.sent_history[0], sent)
    with pytest.raises(RuntimeError, match="pending"):
        control.observe_transition(HOME, np.zeros(7), HOME, np.zeros(7), sent, 0.01)


def test_controller_keeps_actuating_after_abort_without_recomputing_the_model(monkeypatch):
    pytest.importorskip("mujoco")
    from sarrl.controllers.panda_backup import PandaBackupController
    from sarrl.robots.panda import HOME

    control = PandaBackupController()
    sent = control.command(HOME, np.full(7, 4), HOME)
    assert control.aborted
    control.observe_transition(HOME, np.full(7, 4), HOME, np.full(7, 4), sent, 0.01)

    def prohibited(*args, **kwargs):
        raise AssertionError("fallback must not recompute the estimated inverse dynamics")

    monkeypatch.setattr(control.regressor, "inverse_dynamics", prohibited)
    followup = control.command(HOME, np.zeros(7), HOME)
    assert control.last_nominal_command is None
    assert control.supervisor.steps == 2
    assert np.all(np.abs(followup) <= control.dynamics.torque_limit)
