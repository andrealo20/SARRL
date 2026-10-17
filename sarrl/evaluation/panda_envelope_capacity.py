"""Necessary pointwise error bounds for the conditional Panda backup tube."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from sarrl.robots.panda import VELOCITY_LIMIT, vector7
from sarrl.safety.panda_backup import PandaBackupConfig, PandaBackupWitness
from sarrl.safety.panda_backup_tube import PandaBackupEnvelope, PandaBackupTube


def affine_box_error(matrix, drift, frozen_matrix, frozen_drift, torque_limit):
    """Exact componentwise maximum at one state/model, not over a state domain."""
    matrix, frozen_matrix = (np.asarray(value, dtype=float) for value in (matrix, frozen_matrix))
    if any(
        value.shape != (7, 7) or not np.all(np.isfinite(value)) for value in (matrix, frozen_matrix)
    ):
        raise ValueError("point acceleration matrices must be finite with shape (7, 7)")
    delta = matrix - frozen_matrix
    bias = vector7(drift, "drift") - vector7(frozen_drift, "frozen_drift")
    limit = vector7(torque_limit, "torque_limit")
    if np.any(limit <= 0):
        raise ValueError("torque limits must be positive")
    floor = np.abs(bias) + np.abs(delta) @ limit
    # Each row has its own attainable extremum. The rows are not one shared command.
    corners = np.where(bias >= 0, 1.0, -1.0)[:, None] * np.sign(delta) * limit
    if not np.all(np.isfinite(floor)) or not np.all(np.isfinite(corners)):
        raise ValueError("pointwise error bound overflow")
    return floor, corners


def optimistic_capacity(policy, error_floor, lower, upper):
    """Largest available physical domain, exact sensors and zero propagation defects.

    A rejected case rules out this sufficient scalar-tube test for any
    componentwise envelope that covers the full torque box and these points.
    A feasible case does not establish such an envelope.
    """
    config = PandaBackupConfig(**policy["config"])
    witness = PandaBackupWitness(lower, upper, policy["torque_limit"], config)
    envelope = PandaBackupEnvelope(
        lower,
        upper,
        VELOCITY_LIMIT,
        error_floor,
        np.zeros(7),
        np.zeros(7),
        np.zeros(7),
        np.zeros(7),
        "Pointwise necessary error floor only; not a validated domain enclosure",
    )
    tube = PandaBackupTube(witness, envelope)
    cases = tube.assess(policy["anchor"], policy["matrix"], policy["drift"])
    return [asdict(case) for case in cases]
