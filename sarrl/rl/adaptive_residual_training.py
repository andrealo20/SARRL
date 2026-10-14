"""Resumable residual training on the fixed identified/covariance-margin baseline."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from sarrl.envs.adaptive_margin_residual import AdaptiveMarginResidualEnv
from sarrl.envs.mujoco_planar import MujocoPlanarReachEnv
from sarrl.envs.planar_reach import DomainRandomization
from sarrl.evaluation.margin_campaign import V22_PLANT_OPTIONS, v22_config
from sarrl.rl.replay_buffer import ReplayBuffer
from sarrl.rl.sac import SACAgent, SACConfig
from sarrl.rl.training_checkpoint import load_training_checkpoint, save_training_checkpoint
from sarrl.utils import seed_everything


@dataclass(frozen=True)
class ResidualTrainingConfig:
    steps: int = 200000
    start_steps: int = 5000
    hidden: tuple[int, int] = (256, 256)
    batch_size: int = 256
    replay_capacity: int = 200000
    checkpoint_every: int = 25000

    def validate(self):
        if any(
            type(v) is not int or v <= 0
            for v in (
                self.steps,
                self.batch_size,
                self.replay_capacity,
                self.checkpoint_every,
                *self.hidden,
            )
        ):
            raise ValueError("training sizes must be positive integers")
        if type(self.start_steps) is not int or not 0 <= self.start_steps < self.steps:
            raise ValueError("warmup must precede the final training step")
        if self.batch_size > self.replay_capacity or len(self.hidden) != 2:
            raise ValueError("invalid replay batch or architecture")


def training_environment():
    options = V22_PLANT_OPTIONS
    dr = DomainRandomization(
        mass_fraction=0.15,
        friction_fraction=0.30,
        motor_gain_fraction=0.15,
        payload_range=(0.0, 1.0),
        action_delay_max=2,
        sensor_noise_std=options.sensor_noise_std,
    )
    base = MujocoPlanarReachEnv(
        mode="torque",
        randomization=dr,
        armature_range=options.armature_range,
        actuator_time_constant_range=options.actuator_time_constant_range,
    )
    return AdaptiveMarginResidualEnv(base, v22_config(), certificate_kappa=0.25)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, payload):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=1) + "\n")
    temporary.replace(path)


def train(seed, output, signature, *, config=None, stop_after=None):
    """Resume the same identity and retain the final policy without validation selection."""
    cfg = config or ResidualTrainingConfig()
    cfg.validate()
    output = Path(output)
    expected = json.loads(json.dumps({"seed": seed, "config": asdict(cfg), "signature": signature}))
    if (output / "complete.json").exists():
        marker = json.loads((output / "complete.json").read_text())
        if marker["identity"] != expected or marker["steps"] != cfg.steps:
            raise RuntimeError("completed training identity differs")
        if any(sha(output / name) != digest for name, digest in marker["hashes"].items()):
            raise RuntimeError("completed training hashes differ")
        return marker
    if output.exists():
        if (
            not (output / "manifest.json").exists()
            or json.loads((output / "manifest.json").read_text()) != expected
        ):
            raise RuntimeError("resume requires the original training identity")
    else:
        output.mkdir(parents=True)
        atomic_json(output / "manifest.json", expected)
    torch.set_num_threads(1)
    seed_everything(seed)
    env = training_environment()
    agent = SACAgent(env.observation_space.shape[0], 2, SACConfig(hidden=cfg.hidden), seed=seed)
    if agent.device.type != "cpu":
        raise RuntimeError("this training protocol requires CPU execution")
    replay = ReplayBuffer(env.observation_space.shape[0], 2, cfg.replay_capacity, seed)
    checkpoint = output / "training_checkpoint.pt"
    if checkpoint.exists():
        loop = load_training_checkpoint(checkpoint, agent, replay, env)
        if loop["identity"] != expected:
            raise RuntimeError("checkpoint identity differs")
        start, observation, episode_reward, rows = (
            loop["step"],
            loop["observation"],
            loop["episode_reward"],
            loop["rows"],
        )
        executed, abort_calls = loop["executed_control_steps"], loop["abort_calls"]
    else:
        observation, _ = env.reset(seed=seed)
        start, episode_reward, rows = 0, 0.0, []
        executed, abort_calls = 0, 0
    if start > cfg.steps:
        raise RuntimeError("checkpoint exceeds the fixed budget")
    endpoint = cfg.steps if stop_after is None else min(cfg.steps, stop_after)
    if endpoint < start:
        raise ValueError("training endpoint precedes its checkpoint")

    def loop_state(step):
        return {
            "identity": expected,
            "step": step,
            "observation": observation,
            "episode_reward": episode_reward,
            "rows": rows,
            "executed_control_steps": executed,
            "abort_calls": abort_calls,
        }

    for step in range(start + 1, endpoint + 1):
        action = env.sample_action() if step <= cfg.start_steps else agent.act(observation)
        following, reward, terminated, truncated, info = env.step(action)
        executed += int(not info["safety_infeasible"])
        abort_calls += int(info["safety_infeasible"])
        replay.add(observation, action, reward, following, terminated)
        observation = following
        episode_reward += reward
        if len(replay) >= cfg.batch_size and step > cfg.start_steps:
            agent.update(replay.sample(cfg.batch_size))
        if terminated or truncated:
            rows.append(
                [
                    len(rows) + 1,
                    step,
                    float(episode_reward),
                    bool(info["success"]),
                    float(info["distance"]),
                    bool(info["safety_infeasible"]),
                ]
            )
            observation, _ = env.reset()
            episode_reward = 0.0
        if step % 1000 == 0:
            print(
                f"seed={seed} step={step}/{cfg.steps} "
                f"episodes={len(rows)} updates={agent.update_steps}",
                flush=True,
            )
        if step % cfg.checkpoint_every == 0:
            save_training_checkpoint(checkpoint, agent, replay, env, loop_state(step))
    save_training_checkpoint(checkpoint, agent, replay, env, loop_state(endpoint))
    if endpoint != cfg.steps:
        return {"completed": False, "steps": endpoint}
    temporary = output / "final.pt.tmp"
    agent.save(temporary)
    temporary.replace(output / "final.pt")
    temporary = output / "episodes.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "step", "reward", "success", "distance", "abort"])
        writer.writerows(rows)
    temporary.replace(output / "episodes.csv")
    marker = {
        "identity": expected,
        "steps": cfg.steps,
        "updates": agent.update_steps,
        "episodes": len(rows),
        "executed_control_steps": executed,
        "abort_calls": abort_calls,
        "selected": "final.pt",
        "hashes": {
            name: sha(output / name)
            for name in ("manifest.json", "final.pt", "episodes.csv", "training_checkpoint.pt")
        },
    }
    atomic_json(output / "complete.json", marker)
    return marker
