"""Experiment provenance helpers."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import scipy
import torch

from .protocol import repository_commit


def runtime_metadata(root: str | Path = ".") -> dict:
    return {
        "git_commit": repository_commit(root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }


def write_run_manifest(
    path, config: dict, root: str | Path = ".", extra: dict | None = None
) -> None:
    payload = {
        "config": dict(config),
        "runtime": runtime_metadata(root),
        "extra": dict(extra or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def seed_ranges_overlap(start_a: int, n_a: int, start_b: int, n_b: int) -> bool:
    if min(start_a, start_b) < 0 or n_a <= 0 or n_b <= 0:
        raise ValueError("seed starts must be non-negative and lengths positive")
    end_a = start_a + n_a
    end_b = start_b + n_b
    return max(start_a, start_b) < min(end_a, end_b)
