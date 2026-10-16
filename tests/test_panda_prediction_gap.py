from copy import deepcopy

import numpy as np
import pytest

from sarrl.evaluation.panda_prediction_gap import acceleration_replacements, prediction_gap
from sarrl.robots.panda import VELOCITY_LIMIT
from sarrl.safety.panda_backup import PandaBackupWitness
from sarrl.safety.panda_supervisor import witness_record


def synthetic(lag=0, offset=0.0, bias=0.0):
    check = PandaBackupWitness(np.full(7, -2.0), np.full(7, 2.0), np.full(7, 20.0))
    candidate = np.zeros(7)
    result = check.verify(np.zeros(7), np.full(7, offset), candidate, np.eye(7), np.zeros(7), [])
    assert result.accepted
    record = witness_record(result, candidate)
    q, v, frames, sent_history = np.zeros(7), np.zeros(7), [], []
    unsafe = False
    for index in range(300):
        measured = v + offset
        sent = candidate if not index else result.policy.bounded_torque(q, measured)
        sent_history.append(sent)
        delivered = sent_history[index - lag] if index >= lag else np.zeros(7)
        a = delivered + bias
        samples = []
        for t in (0.002, 0.004, 0.006, 0.008, 0.01):
            position, speed = q + t * v + 0.5 * t * t * a, v + t * a
            unsafe |= bool(np.any(np.abs(position) > 2) or np.any(np.abs(speed) > VELOCITY_LIMIT))
            samples.append({"q": position.tolist(), "qd": speed.tolist(), "contacts": 0})
        frames.append(
            {
                "q": q.tolist(),
                "qd": v.tolist(),
                "measured_q": q.tolist(),
                "measured_qd": measured.tolist(),
                "estimated_parameters": [0.0] * 8,
                "torque_nm": sent.tolist(),
                "delivered_torque_nm": delivered.tolist(),
                "physical_substeps": samples,
                "supervisor": {
                    "policy_origin": "proposal",
                    "policy_step": 0,
                    "retained_policy_sha256": record["policy_sha256"],
                    "mode": "task" if not index else "backup",
                    "proposal_witness": record if not index else None,
                },
            }
        )
        q, v = np.asarray(samples[-1]["q"]), np.asarray(samples[-1]["qd"])
    frames.append({"q": q.tolist(), "qd": v.tolist()})
    row = {
        "scenario": "synthetic",
        "seed": 9805800,
        "arm": "bank_backup",
        "unsafe_episode": unsafe,
        "backup": {"abort_step": 1},
        "interface_config": {"command_delay_steps": lag},
        "task_config": {"control_dt": 0.01, "timestep": 0.002},
        "true_parameters": {"payload_mass": 1.0, "viscous_extra": [0.2] * 7},
    }
    return row, frames


@pytest.mark.parametrize(
    "lag,offset,bias",
    [(0, 0.0, 0.0), (0, 0.02, 0.0), (0, 0.0, 500.0), (2, 0.02, 500.0), (4, 0.0, 0.5)],
)
def test_telescoping_velocity_identity_with_delay_measurement_offset_and_discrepancy(
    lag, offset, bias
):
    row, frames = synthetic(lag, offset, bias)
    report, curve = prediction_gap(row, frames, np.full(7, -2), np.full(7, 2), VELOCITY_LIMIT)
    assert len(curve) == 1500
    assert report["maximum_identity_residual_rad_s"] < 1e-9
    np.testing.assert_allclose(report["initial_velocity_offset_rad_s"], -offset, atol=1e-14)
    assert (report["first_violation"] is not None) == row["unsafe_episode"]
    if not bias and not offset:
        assert report["maximum_local_one_period_velocity_discrepancy_rad_s"] == 0


@pytest.mark.parametrize("field", ["policy_hash", "delivery", "mode", "nonfinite"])
def test_inconsistent_saved_provenance_and_states_are_rejected(field):
    row, frames = synthetic()
    changed = deepcopy(frames)
    if field == "policy_hash":
        changed[1]["supervisor"]["retained_policy_sha256"] = "invalid"
    elif field == "delivery":
        changed[2]["delivered_torque_nm"][0] = 1
    elif field == "mode":
        changed[2]["supervisor"]["mode"] = "task"
    else:
        changed[2]["physical_substeps"][0]["qd"][0] = float("nan")
    with pytest.raises((ValueError, AssertionError)):
        prediction_gap(row, changed, np.full(7, -2), np.full(7, 2), VELOCITY_LIMIT)


def test_acceleration_replacements_use_state_and_parameter_substitutions_only():
    row, frames = synthetic(bias=500)
    report, _ = prediction_gap(row, frames, np.full(7, -2), np.full(7, 2), VELOCITY_LIMIT)

    class Regressor:
        def mass_matrix(self, q, theta):
            return np.eye(7) * (1 + theta[0])

        def inverse_dynamics(self, q, v, acceleration, theta):
            return 2 * q + np.asarray(theta[1:]) * v

    result = acceleration_replacements(report, row, frames, Regressor())
    point = report["first_violation"]
    q, v = np.asarray(point["interval_start_q"]), np.asarray(point["interval_start_qd"])
    np.testing.assert_allclose(result["refreshed_state_acceleration_rad_s2"], -2 * q)
    np.testing.assert_allclose(result["oracle_acceleration_rad_s2"], (-2 * q - 0.2 * v) / 2)
    np.testing.assert_allclose(result["identity_residual_rad_s2"], 0, atol=1e-12)
