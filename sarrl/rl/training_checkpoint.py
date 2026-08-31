"""Atomic checkpoints for reproducible off-policy training continuation."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

CHECKPOINT_VERSION = 2
_SUPPORTED_VERSIONS = {1, 2}


def _load_payload(path, map_location="cpu") -> dict:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    version = payload.get("checkpoint_version")
    if version not in _SUPPORTED_VERSIONS:
        raise ValueError("unsupported training checkpoint version")
    return payload


def save_training_checkpoint(path, agent, replay, env, loop_state: dict) -> None:
    """Save every mutable state needed by the current training loop.

    Version 2 includes Python's RNG and the environment constructor
    configuration. Together with the agent, replay and NumPy RNG states this
    lets a session be reconstructed without relying on matching CLI defaults.
    """
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "agent": agent.state_dict(include_optimizers=True),
        "replay": replay.state_dict(),
        "environment": env.state_dict(),
        "loop_state": dict(loop_state),
        "numpy_global_state": np.random.get_state(),
        "python_random_state": random.getstate(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _restore_global_rng(payload: dict) -> None:
    if payload.get("numpy_global_state") is not None:
        np.random.set_state(payload["numpy_global_state"])
    if payload.get("python_random_state") is not None:
        random.setstate(payload["python_random_state"])


def load_training_checkpoint(path, agent, replay, env) -> dict:
    """Restore into already-constructed components.

    Version-1 checkpoints remain readable. Version 2 additionally verifies
    the full environment constructor configuration before restoring state.
    """
    payload = _load_payload(path, map_location=agent.device)
    agent.load_state_dict(payload["agent"], load_optimizers=True, restore_rng=True)
    replay.load_state_dict(payload["replay"])
    env.load_state_dict(payload["environment"])
    _restore_global_rng(payload)
    loop = dict(payload["loop_state"])
    loop["checkpoint_version"] = int(payload["checkpoint_version"])
    return loop


def load_training_session(path):
    """Reconstruct agent, replay buffer and environment directly from a checkpoint.

    This is the preferred resume path for v2 checkpoints because network
    architecture, replay capacity, domain randomization and fault settings are
    all taken from the saved session rather than current command-line defaults.
    """
    from sarrl.envs import PlanarReachEnv
    from sarrl.rl.replay_buffer import ReplayBuffer
    from sarrl.rl.sac import SACAgent

    payload = _load_payload(path, map_location="cpu")
    if int(payload["checkpoint_version"]) < 2:
        raise ValueError(
            "training checkpoint v1 cannot reconstruct the environment exactly; "
            "use load_training_checkpoint with matching components"
        )
    agent = SACAgent.from_state_dict(
        payload["agent"], seed=0, load_optimizers=True, restore_rng=True
    )
    replay = ReplayBuffer.from_state_dict(payload["replay"])
    env = PlanarReachEnv.from_state_dict(payload["environment"])
    _restore_global_rng(payload)
    loop = dict(payload["loop_state"])
    loop["checkpoint_version"] = int(payload["checkpoint_version"])
    return agent, replay, env, loop
