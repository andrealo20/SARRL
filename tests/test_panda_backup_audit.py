from copy import deepcopy

import numpy as np
import pytest

from sarrl.evaluation.panda_backup_audit import audit_witness
from sarrl.safety.panda_backup import PandaBackupWitness
from sarrl.safety.panda_supervisor import witness_record


def fixture(pending=False):
    check = PandaBackupWitness(np.full(7, -2.0), np.full(7, 2.0), np.full(7, 20.0))
    q, v, candidate = np.zeros(7), np.full(7, 0.2), np.full(7, 0.2)
    mass = np.eye(7) + 0.02 * np.ones((7, 7))
    history = []
    if pending:
        v[0], candidate[0] = 2.07, -20
        history = [np.full(7, 20)]
    result = check.verify(q, v, candidate, np.linalg.inv(mass), np.zeros(7), history)
    record = witness_record(result, candidate)
    arguments = (
        q,
        v,
        candidate,
        history,
        check.bounds.lower,
        check.bounds.upper,
        check.bounds.velocity,
        check.bounds.limit,
    )
    return record, arguments


@pytest.mark.parametrize("pending", [False, True])
def test_independent_audit_matches_accepted_and_rejected_witnesses(pending, monkeypatch):
    record, arguments = fixture(pending)

    def prohibited(*args, **kwargs):
        raise AssertionError("auditor must not call the acceptance implementation")

    monkeypatch.setattr(PandaBackupWitness, "verify", prohibited)
    report = audit_witness(record, *arguments)
    assert report["witnesses"] == 1 and report["delay_cases"] == 5
    assert report["checked_intervals"] > 0


@pytest.mark.parametrize("tamper", ["decision", "radius", "energy", "policy", "candidate", "lag"])
def test_independent_audit_rejects_altered_claims(tamper):
    record, arguments = fixture()
    changed = deepcopy(record)
    if tamper == "decision":
        changed["accepted"] = not changed["accepted"]
    elif tamper == "radius":
        changed["cases"][0]["terminal_radius_squared"] += 1
    elif tamper == "energy":
        changed["cases"][0]["terminal_energy"] += 1
    elif tamper == "policy":
        changed["policy"]["anchor"][0] += 0.1
    elif tamper == "candidate":
        changed["candidate_torque_nm"][0] += 0.1
    else:
        changed["cases"][0]["lag"] = 1
    with pytest.raises((ValueError, AssertionError)):
        audit_witness(changed, *arguments)
