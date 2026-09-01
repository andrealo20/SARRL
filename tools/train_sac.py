#!/usr/bin/env python3
"""Train from-scratch SAC on the analytical SARRL reaching environment."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch

from sarrl.adaptation import AdaptiveContextEnv, DynamicsContextEncoder
from sarrl.envs.planar_reach import DomainRandomization, PlanarReachEnv
from sarrl.evaluation import (
    assert_repository_import_root,
    evaluate_policy,
    write_run_manifest,
)
from sarrl.rl import (
    ReplayBuffer,
    SACAgent,
    SACConfig,
    load_training_session,
    save_training_checkpoint,
)
from sarrl.utils import seed_everything


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_env(env):
    if isinstance(env, AdaptiveContextEnv):
        return env.env
    return env


def _environment_mode(env) -> str:
    return _base_env(env).mode


def _validation_env(env):
    """Construct an independent deterministic validation environment."""
    base = _base_env(env)

    val_base = PlanarReachEnv(
        mode=base.mode,
        dt=base.dt,
        max_steps=base.max_steps,
        torque_limit=base.torque_limit,
        residual_limit=base.residual_limit,
        success_radius=base.success_radius,
        randomization=base.randomization,
        fault=base.fault,
    )

    if not isinstance(env, AdaptiveContextEnv):
        return val_base

    # deepcopy preserves the frozen encoder exactly without consuming the
    # global torch RNG that drives stochastic SAC actions.
    encoder = copy.deepcopy(env.encoder)
    return AdaptiveContextEnv(
        val_base,
        encoder,
        device="cpu",
    )


def _load_context_encoder(path: Path) -> DynamicsContextEncoder:
    """Load a frozen runtime encoder without perturbing SAC's CPU RNG."""
    if not path.is_file():
        raise FileNotFoundError(f"context checkpoint not found: {path}")

    with torch.random.fork_rng(devices=[]):
        encoder = DynamicsContextEncoder.load(
            path,
            map_location="cpu",
        )

    encoder.eval()

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    return encoder


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["torque", "residual"], default="residual")
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start-steps", type=int, default=5_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, nargs=2, default=(256, 256), metavar=("H1", "H2"))
    p.add_argument("--update-every", type=int, default=1)
    p.add_argument("--replay-capacity", type=int, default=200_000)
    p.add_argument("--randomize", action="store_true")
    p.add_argument(
        "--context-checkpoint",
        default=None,
        help=(
            "Frozen DynamicsContextEncoder checkpoint. "
            "When supplied, residual SAC receives obs + causal context latent."
        ),
    )
    p.add_argument("--output", default="results/run_seed0")
    p.add_argument("--resume", default=None)
    p.add_argument("--checkpoint-every", type=int, default=50_000)
    p.add_argument("--validate-every", type=int, default=25_000)
    p.add_argument("--validation-episodes", type=int, default=30)
    p.add_argument("--validation-seed", type=int, default=20_000)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    assert_repository_import_root(root)
    if args.steps <= 0 or args.start_steps < 0 or args.batch_size <= 0:
        raise SystemExit("steps/batch-size must be positive and start-steps non-negative")
    if any(h <= 0 for h in args.hidden) or args.update_every <= 0 or args.replay_capacity <= 0:
        raise SystemExit("hidden sizes, update-every and replay-capacity must be positive")
    if args.checkpoint_every < 0 or args.validate_every < 0:
        raise SystemExit("checkpoint-every and validate-every must be non-negative")
    if args.validation_episodes <= 0 or args.validation_seed < 0:
        raise SystemExit("validation episodes must be positive and seed non-negative")

    if args.context_checkpoint is not None and args.mode != "residual":
        raise SystemExit("--context-checkpoint requires --mode residual")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.resume is not None:
        agent, replay, env, loop = load_training_session(args.resume)
        step0 = int(loop["step"])
        episode = int(loop["episode"])
        ep_reward = float(loop["ep_reward"])
        obs = np.asarray(loop["obs"], dtype=np.float32)
        rows = list(loop.get("rows", []))
        validation_rows = list(loop.get("validation_rows", []))
        best_key = tuple(loop.get("best_key", (-np.inf, -np.inf)))
        trainer_cfg = dict(loop.get("trainer_config", {}))
        batch_size = int(trainer_cfg.get("batch_size", args.batch_size))
        start_steps = int(trainer_cfg.get("start_steps", args.start_steps))
        update_every = int(trainer_cfg.get("update_every", args.update_every))
        context_checkpoint_path = trainer_cfg.get("context_checkpoint")
        context_checkpoint_sha256 = trainer_cfg.get("context_checkpoint_sha256")

        if step0 > args.steps:
            raise SystemExit("resume checkpoint exceeds requested --steps")
        stored_context_sha256 = trainer_cfg.get("context_checkpoint_sha256")

        if isinstance(env, AdaptiveContextEnv):
            if stored_context_sha256 is None:
                raise SystemExit(
                    "adaptive-context resume checkpoint is missing context checkpoint provenance"
                )

            if args.context_checkpoint is not None:
                supplied_context = Path(args.context_checkpoint).resolve()

                if not supplied_context.is_file():
                    raise SystemExit(f"context checkpoint not found: {supplied_context}")

                supplied_sha256 = _sha256(supplied_context)

                if supplied_sha256 != stored_context_sha256:
                    raise SystemExit(
                        "refusing adaptive-context resume with a different context checkpoint"
                    )

        elif args.context_checkpoint is not None:
            raise SystemExit(
                "cannot add a context encoder while resuming a non-context training session"
            )

        print(
            "resuming exact session: "
            f"mode={_environment_mode(env)} "
            f"hidden={agent.config.hidden} replay={replay.capacity} "
            f"batch={batch_size} update_every={update_every} "
            f"context={isinstance(env, AdaptiveContextEnv)}"
        )
    else:
        seed_everything(args.seed)
        dr = (
            DomainRandomization(
                mass_fraction=0.15,
                friction_fraction=0.30,
                motor_gain_fraction=0.15,
                payload_range=(0.0, 1.0),
                action_delay_max=2,
            )
            if args.randomize
            else DomainRandomization()
        )
        base_env = PlanarReachEnv(
            mode=args.mode,
            randomization=dr,
        )

        context_checkpoint_path = None
        context_checkpoint_sha256 = None

        if args.context_checkpoint is not None:
            context_checkpoint_path = Path(args.context_checkpoint).resolve()

            context_checkpoint_sha256 = _sha256(context_checkpoint_path)

            encoder = _load_context_encoder(context_checkpoint_path)

            env = AdaptiveContextEnv(
                base_env,
                encoder,
                device="cpu",
            )
        else:
            env = base_env

        agent = SACAgent(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            SACConfig(hidden=tuple(args.hidden)),
            seed=args.seed,
        )
        replay = ReplayBuffer(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            args.replay_capacity,
            args.seed,
        )
        obs, _ = env.reset(seed=args.seed)
        ep_reward = 0.0
        episode = 0
        step0 = 0
        rows = []
        validation_rows = []
        best_key = (-np.inf, -np.inf)
        batch_size = args.batch_size
        start_steps = args.start_steps
        update_every = args.update_every

    trainer_config = {
        "batch_size": batch_size,
        "start_steps": start_steps,
        "update_every": update_every,
        "context_checkpoint": (
            None if context_checkpoint_path is None else str(context_checkpoint_path)
        ),
        "context_checkpoint_sha256": context_checkpoint_sha256,
    }
    val_env = _validation_env(env)
    write_run_manifest(
        out / "run_manifest.json",
        {
            "requested_steps": args.steps,
            "seed": args.seed,
            "resume": args.resume,
            "agent_config": {
                "gamma": agent.config.gamma,
                "tau": agent.config.tau,
                "actor_lr": agent.config.actor_lr,
                "critic_lr": agent.config.critic_lr,
                "alpha_lr": agent.config.alpha_lr,
                "init_alpha": agent.config.init_alpha,
                "hidden": list(agent.config.hidden),
            },
            "environment": env.constructor_config(),
            "context": {
                "enabled": isinstance(env, AdaptiveContextEnv),
                "checkpoint": trainer_config["context_checkpoint"],
                "checkpoint_sha256": trainer_config["context_checkpoint_sha256"],
                "runtime_device": ("cpu" if isinstance(env, AdaptiveContextEnv) else None),
                "latent_dim": (
                    env.config.latent_dim if isinstance(env, AdaptiveContextEnv) else None
                ),
            },
            "replay_capacity": replay.capacity,
            "trainer": trainer_config,
            "validation": {
                "every": args.validate_every,
                "episodes": args.validation_episodes,
                "seed": args.validation_seed,
            },
        },
        root=Path(__file__).resolve().parents[1],
    )

    def loop_state(step: int) -> dict:
        return {
            "step": step,
            "episode": episode,
            "ep_reward": ep_reward,
            "obs": obs,
            "rows": rows,
            "validation_rows": validation_rows,
            "best_key": list(best_key),
            "trainer_config": trainer_config,
        }

    def validate(step: int) -> None:
        nonlocal best_key
        result = evaluate_policy(
            agent,
            val_env,
            episodes=args.validation_episodes,
            seed=args.validation_seed,
        )
        key = result.selection_key
        validation_rows.append(
            (
                step,
                result.successes,
                result.episodes,
                result.success_rate,
                result.reward_mean,
                result.reward_std,
                result.final_distance_mean,
            )
        )
        print(
            f"validation step={step:8d} success={result.successes}/{result.episodes} "
            f"reward={result.reward_mean:.2f} distance={result.final_distance_mean:.4f}"
        )
        if key > best_key:
            best_key = key
            agent.save(out / "best.pt")

    for step in range(step0 + 1, args.steps + 1):
        action = env.sample_action() if step <= start_steps else agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        replay.add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        ep_reward += reward

        if len(replay) >= batch_size and step > start_steps and step % update_every == 0:
            metrics = agent.update(replay.sample(batch_size))
        else:
            metrics = {}

        if terminated or truncated:
            episode += 1
            rows.append((episode, step, ep_reward, int(info["success"]), info["distance"]))
            if episode % 20 == 0:
                alpha = metrics.get("alpha", float("nan"))
                print(
                    f"episode={episode:5d} step={step:8d} reward={ep_reward:9.2f} "
                    f"success={int(info['success'])} alpha={alpha:.3f}"
                )
            obs, _ = env.reset()
            ep_reward = 0.0

        if args.validate_every and step % args.validate_every == 0:
            validate(step)

        if args.checkpoint_every and step % args.checkpoint_every == 0:
            save_training_checkpoint(
                out / f"train_step{step}.pt", agent, replay, env, loop_state(step)
            )

    if args.validate_every and (not validation_rows or validation_rows[-1][0] != args.steps):
        validate(args.steps)

    agent.save(out / "final.pt")
    save_training_checkpoint(out / "training_final.pt", agent, replay, env, loop_state(args.steps))
    with (out / "episodes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "step", "reward", "success", "final_distance"])
        w.writerows(rows)
    with (out / "validation.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "step",
                "successes",
                "episodes",
                "success_rate",
                "reward_mean",
                "reward_std",
                "final_distance_mean",
            ]
        )
        w.writerows(validation_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
