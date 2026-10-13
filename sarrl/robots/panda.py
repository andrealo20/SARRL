"""Torque-controlled, seven-joint Panda model with a fixed base and no gripper."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "panda"
HOME = np.array([0.0, -0.35, 0.0, -1.8, 0.0, 1.5, 0.6])
VELOCITY_LIMIT = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])


def vector7(value, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (7,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite vector of shape (7,)")
    return value


def model_provenance() -> dict:
    """Verify the packaged model and collision geometry against their recorded hashes."""
    source = json.loads((ASSETS / "source.json").read_text())
    for relative, expected in source["files"].items():
        actual = hashlib.sha256((ASSETS / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Panda asset hash mismatch: {relative}")
    return source


def load_panda_model(timestep: float = 0.002):
    if mujoco is None:
        raise ImportError("Panda simulation needs the mujoco extra: pip install -e '.[mujoco]'")
    if not np.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("timestep must be finite and positive")
    model_provenance()
    model = mujoco.MjModel.from_xml_path(str(ASSETS / "panda.xml"))
    model.opt.timestep = timestep
    if (model.nq, model.nv, model.nu) != (7, 7, 7):
        raise RuntimeError("Panda model must have exactly seven joints and seven controls")
    return model


class PandaDynamics:
    """A private nominal model; no plant object or randomized parameters are consumed."""

    def __init__(self, timestep: float = 0.002):
        self.model = load_panda_model(timestep)
        self.data = mujoco.MjData(self.model)
        self.torque_limit = self.model.actuator_ctrlrange[:, 1].copy()

    def update(self, q, qd) -> None:
        self.data.qpos[:] = vector7(q, "q")
        self.data.qvel[:] = vector7(qd, "qd")
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def inverse_dynamics(self, q, qd, qdd) -> np.ndarray:
        self.update(q, qd)
        result = np.empty(7)
        # mj_mulM includes the reflected rotor inertia and works across MuJoCo versions.
        mujoco.mj_mulM(self.model, self.data, result, vector7(qdd, "qdd"))
        return result + self.data.qfrc_bias - self.data.qfrc_passive

    def mass_matrix(self, q) -> np.ndarray:
        self.update(q, np.zeros(7))
        result = np.empty((7, 7))
        for joint, basis in enumerate(np.eye(7)):
            column = np.empty(7)
            mujoco.mj_mulM(self.model, self.data, column, basis)
            result[:, joint] = column
        return result

    def tip_position(self, q) -> np.ndarray:
        self.update(q, np.zeros(7))
        site = self.model.site("attachment_site").id
        return self.data.site_xpos[site].copy()
