#!/usr/bin/env python3
"""Train from-scratch SAC on the analytical SARRL reaching environment."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sarrl.adaptation import AdaptiveContextEnv, DynamicsContextEncoder
from sarrl.envs import (
    DomainRandomization,
    PlanarReachEnv,
    SafetyProjectedEnv,
)
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


def _save_agent_atomically(agent, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    agent.save(temporary)
    temporary.replace(path)


def _write_csv_atomically(path: Path, header: list[str], rows: list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    temporary.replace(path)


def _write_json_atomically(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _base_env(env):
    while isinstance(env, (AdaptiveContextEnv, SafetyProjectedEnv)):
        env = env.env
    return env


def _environment_mode(env) -> str:
    return _base_env(env).mode


def _validation_env(
    env,
    *,
    safety_projected: bool = False,
    infeasible_reward: float = -500.0,
):
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

    if isinstance(env, AdaptiveContextEnv):
        if safety_projected:
            raise ValueError("safety projection with adaptive context is not supported")

        # deepcopy preserves the frozen encoder exactly without consuming the
        # global torch RNG that drives stochastic SAC actions.
        encoder = copy.deepcopy(env.encoder)
        return AdaptiveContextEnv(
            val_base,
            encoder,
            device="cpu",
        )

    if safety_projected:
        config = env.safety_config if isinstance(env, SafetyProjectedEnv) else None
        return SafetyProjectedEnv(
            val_base,
            safety_config=config,
            infeasible_reward=infeasible_reward,
        )

    return val_base


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
        "--training-hocbf",
        action="store_true",
        help="Apply the required HOCBF projection to every training action.",
    )
    p.add_argument(
        "--validation-hocbf",
        action="store_true",
        help="Evaluate validation checkpoints through the required HOCBF.",
    )
    p.add_argument("--infeasible-reward", type=float, default=-500.0)
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
    if (args.training_hocbf or args.validation_hocbf) and args.mode != "residual":
        raise SystemExit("HOCBF training and validation require --mode residual")
    if args.context_checkpoint is not None and (
        args.training_hocbf or args.validation_hocbf
    ):
        raise SystemExit("HOCBF training is not combined with adaptive context")
    if not np.isfinite(args.infeasible_reward) or args.infeasible_reward >= 0.0:
        raise SystemExit("--infeasible-reward must be finite and negative")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.resume is not None:
        agent, replay, env, loop = load_training_session(args.resume)
        step0 = int(loop["step"])
        episode = int(loop["episode"])
        ep_reward = float(loop["ep_reward"])
        obs = np.asarray(loop["obs"], dtype=np.float32)
        rows = list(loop.get("rows", []))
        safety_rows = list(loop.get("safety_rows", []))
        validation_rows = list(loop.get("validation_rows", []))
        best_key = tuple(loop.get("best_key", (-np.inf, -np.inf)))
        best_step = loop.get("best_step")
        best_step = None if best_step is None else int(best_step)
        if best_step is None and validation_rows and np.isfinite(best_key).all():
            matching_steps = [
                int(row[0])
                for row in validation_rows
                if (float(row[3]), float(row[4])) == best_key
            ]
            if matching_steps:
                best_step = min(matching_steps)
        trainer_cfg = dict(loop.get("trainer_config", {}))
        batch_size = int(trainer_cfg.get("batch_size", args.batch_size))
        start_steps = int(trainer_cfg.get("start_steps", args.start_steps))
        update_every = int(trainer_cfg.get("update_every", args.update_every))
        context_checkpoint_path = trainer_cfg.get("context_checkpoint")
        context_checkpoint_sha256 = trainer_cfg.get("context_checkpoint_sha256")
        training_hocbf = bool(trainer_cfg.get("training_hocbf", False))
        validation_hocbf = bool(trainer_cfg.get("validation_hocbf", False))
        infeasible_reward = float(
            trainer_cfg.get("infeasible_reward", args.infeasible_reward)
        )

        if step0 > args.steps:
            raise SystemExit("resume checkpoint exceeds requested --steps")
        stored_context_sha256 = trainer_cfg.get("context_checkpoint_sha256")

        if args.training_hocbf != training_hocbf:
            raise SystemExit("resume --training-hocbf does not match the checkpoint")
        if args.validation_hocbf != validation_hocbf:
            raise SystemExit("resume --validation-hocbf does not match the checkpoint")
        if args.infeasible_reward != infeasible_reward:
            raise SystemExit("resume --infeasible-reward does not match the checkpoint")

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
            f"context={isinstance(env, AdaptiveContextEnv)} "
            f"training_hocbf={training_hocbf} validation_hocbf={validation_hocbf}"
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
        training_hocbf = args.training_hocbf
        validation_hocbf = args.validation_hocbf
        infeasible_reward = args.infeasible_reward

        if args.context_checkpoint is not None:
            context_checkpoint_path = Path(args.context_checkpoint).resolve()

            context_checkpoint_sha256 = _sha256(context_checkpoint_path)

            encoder = _load_context_encoder(context_checkpoint_path)

            env = AdaptiveContextEnv(
                base_env,
                encoder,
                device="cpu",
            )
        elif training_hocbf:
            env = SafetyProjectedEnv(
                base_env,
                infeasible_reward=infeasible_reward,
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
        safety_rows = []
        validation_rows = []
        best_key = (-np.inf, -np.inf)
        best_step = None
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
        "training_hocbf": training_hocbf,
        "validation_hocbf": validation_hocbf,
        "infeasible_reward": infeasible_reward,
    }
    val_env = _validation_env(
        env,
        safety_projected=validation_hocbf,
        infeasible_reward=infeasible_reward,
    )
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
            "safety": {
                "training_hocbf": training_hocbf,
                "validation_hocbf": validation_hocbf,
                "infeasible_reward": infeasible_reward,
                "training_environment": (
                    env.constructor_config() if training_hocbf else None
                ),
                "validation_environment": (
                    val_env.constructor_config() if validation_hocbf else None
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
            "safety_rows": safety_rows,
            "validation_rows": validation_rows,
            "best_key": list(best_key),
            "best_step": best_step,
            "trainer_config": trainer_config,
        }

    def validate(step: int) -> None:
        nonlocal best_key, best_step
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
            best_step = step
            _save_agent_atomically(agent, out / "best.pt")

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
            if training_hocbf:
                safety_rows.append(
                    (
                        episode,
                        step,
                        int(info["safety_infeasible"]),
                        int(info["safety_command_attempts"]),
                        int(info["safety_certified_steps"]),
                        int(info["safety_intervention_steps"]),
                        float(info["safety_correction_sum"]),
                        float(info["safety_correction_max"]),
                    )
                )
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

    _save_agent_atomically(agent, out / "final.pt")
    save_training_checkpoint(
        out / "training_final.pt", agent, replay, env, loop_state(args.steps)
    )
    _write_csv_atomically(
        out / "episodes.csv",
        ["episode", "step", "reward", "success", "final_distance"],
        rows,
    )
    _write_csv_atomically(
        out / "validation.csv",
        [
            "step",
            "successes",
            "episodes",
            "success_rate",
            "reward_mean",
            "reward_std",
            "final_distance_mean",
        ],
        validation_rows,
    )
    if training_hocbf:
        _write_csv_atomically(
            out / "training_safety.csv",
            [
                "episode",
                "step",
                "safety_infeasible",
                "command_attempts",
                "safety_certified_steps",
                "safety_intervention_steps",
                "safety_correction_sum",
                "safety_correction_max",
            ],
            safety_rows,
        )
    if args.validate_every:
        best_path = out / "best.pt"
        if not best_path.is_file() or best_step is None:
            raise RuntimeError("training completed without a selected validation checkpoint")
        _write_json_atomically(
            out / "selection.json",
            {
                "selection_key": {
                    "success_rate": float(best_key[0]),
                    "reward_mean": float(best_key[1]),
                },
                "best_step": best_step,
                "checkpoint": "best.pt",
                "checkpoint_sha256": _sha256(best_path),
                "tie_break": "success_rate_then_reward_then_earliest_step",
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
