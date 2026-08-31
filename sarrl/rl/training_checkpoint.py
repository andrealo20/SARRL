"""Atomic checkpoint for exact off-policy training continuation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


CHECKPOINT_VERSION = 1


def save_training_checkpoint(path, agent, replay, env, loop_state: dict) -> None:
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "agent": agent.state_dict(include_optimizers=True),
        "replay": replay.state_dict(),
        "environment": env.state_dict(),
        "loop_state": dict(loop_state),
        "numpy_global_state": np.random.get_state(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_training_checkpoint(path, agent, replay, env) -> dict:
    payload = torch.load(Path(path), map_location=agent.device, weights_only=False)
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported training checkpoint version")
    agent.load_state_dict(payload["agent"], load_optimizers=True, restore_rng=True)
    replay.load_state_dict(payload["replay"])
    env.load_state_dict(payload["environment"])
    if payload.get("numpy_global_state") is not None:
        np.random.set_state(payload["numpy_global_state"])
    return dict(payload["loop_state"])
