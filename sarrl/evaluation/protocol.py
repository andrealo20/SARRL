"""Reproducible evaluation protocol and statistical summaries."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EpisodeResult:
    scenario: str
    controller: str
    seed: int
    reward: float
    steps: int
    success: bool
    final_distance: float
    max_speed: float
    max_command_torque: float
    fault_seen: bool


@dataclass(frozen=True)
class AggregateMetrics:
    n: int
    successes: int
    success_rate: float
    success_ci95_low: float
    success_ci95_high: float
    reward_mean: float
    reward_std: float
    final_distance_mean: float
    success_steps_mean: float | None
    max_speed_mean: float
    max_command_torque_mean: float


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0 or successes < 0 or successes > n:
        raise ValueError("invalid binomial counts")
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def aggregate(results: list[EpisodeResult]) -> AggregateMetrics:
    if not results:
        raise ValueError("cannot aggregate an empty result set")
    successes = sum(int(r.success) for r in results)
    low, high = wilson_interval(successes, len(results))
    rewards = np.asarray([r.reward for r in results], dtype=np.float64)
    distances = np.asarray([r.final_distance for r in results], dtype=np.float64)
    speeds = np.asarray([r.max_speed for r in results], dtype=np.float64)
    torques = np.asarray([r.max_command_torque for r in results], dtype=np.float64)
    success_steps = [r.steps for r in results if r.success]
    return AggregateMetrics(
        n=len(results),
        successes=successes,
        success_rate=successes / len(results),
        success_ci95_low=low,
        success_ci95_high=high,
        reward_mean=float(rewards.mean()),
        reward_std=float(rewards.std()),
        final_distance_mean=float(distances.mean()),
        success_steps_mean=float(np.mean(success_steps)) if success_steps else None,
        max_speed_mean=float(speeds.mean()),
        max_command_torque_mean=float(torques.mean()),
    )


def paired_success_difference(
    a: list[EpisodeResult], b: list[EpisodeResult], bootstrap: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Paired bootstrap CI for success-rate difference a-b on identical seeds."""
    if bootstrap <= 0:
        raise ValueError("bootstrap must be positive")
    amap = {r.seed: int(r.success) for r in a}
    bmap = {r.seed: int(r.success) for r in b}
    if amap.keys() != bmap.keys() or not amap:
        raise ValueError("paired comparisons require the same non-empty seed set")
    seeds = sorted(amap)
    diff = np.asarray([amap[s] - bmap[s] for s in seeds], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(diff), size=(bootstrap, len(diff)))
    boot = diff[draws].mean(axis=1)
    return float(diff.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def write_episode_csv(path, results: list[EpisodeResult]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        list(asdict(results[0]).keys())
        if results
        else list(EpisodeResult.__dataclass_fields__)
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))


def write_summary_json(
    path, grouped: dict[str, AggregateMetrics], metadata: dict | None = None
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": dict(metadata or {}),
        "groups": {key: asdict(value) for key, value in grouped.items()},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def repository_commit(root: str | Path = ".") -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
