"""Run the small Panda nominal pilot; refuse to overwrite existing outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import sarrl
from sarrl.envs.panda_reach import PandaReachConfig
from sarrl.evaluation.panda_nominal import run_episode, summarize
from sarrl.robots.panda import model_provenance, mujoco

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "sarrl/robots/panda.py",
    "sarrl/controllers/panda_nominal.py",
    "sarrl/envs/panda_reach.py",
    "sarrl/evaluation/panda_nominal.py",
    "tools/run_panda_nominal.py",
    "pyproject.toml",
    "sarrl/assets/panda/source.json",
    "docs/seed_registry.json",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=9804000)
    parser.add_argument("--output", type=Path, default=Path("results/panda_nominal_pilot"))
    args = parser.parse_args(argv)
    if Path(sarrl.__file__).resolve().parent.parent != ROOT:
        parser.error("the imported SARRL package does not match this runner's checkout")
    if not 1 <= args.episodes <= 100 or not 9804000 <= args.seed_start <= 9804100 - args.episodes:
        parser.error("use 1..100 episodes inside the registered Panda pilot block 9804000..9804099")
    config = PandaReachConfig()
    config.validate()
    provenance = model_provenance()
    # The pilot may be rerun into a new directory, but no existing evidence is replaced.
    output = args.output.resolve()
    if output.exists():
        parser.error(f"output already exists: {output}; choose a new directory")
    if mujoco is None:
        parser.error("install the mujoco extra before running the Panda pilot")
    registry = json.loads((ROOT / "docs/seed_registry.json").read_text())
    if not any(
        r["label"] == "Panda nominal engineering pilot"
        and r["low"] <= args.seed_start
        and r["high"] >= args.seed_start + args.episodes - 1
        for r in registry["ranges"]
    ):
        parser.error("the Panda pilot seeds are absent from the seed registry")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    manifest = {
        "kind": "exploratory_nominal_engineering_pilot",
        "config": asdict(config),
        "controller": {"name": "computed_torque", "kp": 100.0, "kd": 20.0},
        "seeds": list(range(args.seed_start, args.seed_start + args.episodes)),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "platform": platform.platform(),
        },
        "source_commit": commit,
        "dirty_paths": dirty,
        "source_sha256": {p: sha(ROOT / p) for p in SOURCES},
        "model": provenance,
    }
    output.mkdir(parents=True)
    write_json(output / "manifest.json", manifest)
    episodes = []
    for i, seed in enumerate(manifest["seeds"]):
        record, trace = run_episode(seed, config, retain_trace=i == 0)
        episodes.append(record)
        if trace:
            write_json(output / "first_episode_trace.json", trace)
        print(
            f"seed {seed}: success={record['success']} "
            f"joint_error={record['joint_error_max_rad']:.6f} rad "
            f"contacts={record['contact_substeps']}",
            flush=True,
        )
    summary = summarize(episodes, config)
    write_json(output / "episodes.json", episodes)
    write_json(output / "summary.json", summary)
    write_json(
        output / "complete.json",
        {
            "episodes": len(episodes),
            "hashes": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()},
        },
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["successes"] == len(episodes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
