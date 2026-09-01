"""Experiment provenance helpers."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import scipy
import torch

from .protocol import repository_commit


def imported_repository_root() -> Path:
    """Return the checkout from which the imported SARRL package originates."""
    return Path(__file__).resolve().parents[2]


def assert_repository_import_root(root: str | Path) -> Path:
    """Fail if Python imported SARRL from a different checkout."""
    expected = Path(root).resolve()
    imported = imported_repository_root()

    if imported != expected:
        raise RuntimeError(
            "SARRL import/check-out mismatch: "
            f"script checkout={expected}, imported package={imported}. "
            "Reinstall the active checkout with `python -m pip install --no-deps -e .`."
        )

    return imported


def repository_dirty_paths(root: str | Path) -> list[str]:
    """Return Git working-tree paths reported as modified/untracked."""
    root = Path(root).resolve()

    try:
        output = subprocess.check_output(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot inspect repository status at {root}") from exc

    paths: list[str] = []

    for line in output.splitlines():
        if not line:
            continue

        path = line[3:]

        # Porcelain v1 rename/copy representation.
        if " -> " in path:
            path = path.rsplit(" -> ", 1)[1]

        # Git may quote unusual paths. Quoted paths are conservatively
        # considered dirty; stripping the outer quotes improves readability.
        if len(path) >= 2 and path[0] == path[-1] == '"':
            path = path[1:-1]

        paths.append(path)

    return paths


def source_tree_dirty_paths(
    root: str | Path,
    ignored_prefixes: tuple[str, ...] = ("results/", "artifacts/"),
) -> list[str]:
    """Return dirty paths outside experiment-output directories."""
    dirty = repository_dirty_paths(root)

    def ignored(path: str) -> bool:
        return any(
            path == prefix.rstrip("/") or path.startswith(prefix) for prefix in ignored_prefixes
        )

    return [path for path in dirty if not ignored(path)]


def assert_source_tree_clean(root: str | Path) -> None:
    """Require committed source code before an official training campaign."""
    dirty = source_tree_dirty_paths(root)

    if dirty:
        preview = ", ".join(dirty[:10])
        if len(dirty) > 10:
            preview += f", ... (+{len(dirty) - 10} more)"

        raise RuntimeError("refusing official training with uncommitted source changes: " + preview)


def runtime_metadata(root: str | Path = ".") -> dict:
    return {
        "git_commit": repository_commit(root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "imported_repository_root": str(imported_repository_root()),
    }


def write_run_manifest(
    path,
    config: dict,
    root: str | Path = ".",
    extra: dict | None = None,
) -> None:
    payload = {
        "config": dict(config),
        "runtime": runtime_metadata(root),
        "extra": dict(extra or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def seed_ranges_overlap(
    start_a: int,
    n_a: int,
    start_b: int,
    n_b: int,
) -> bool:
    if min(start_a, start_b) < 0 or n_a <= 0 or n_b <= 0:
        raise ValueError("seed starts must be non-negative and lengths positive")

    end_a = start_a + n_a
    end_b = start_b + n_b

    return max(start_a, start_b) < min(end_a, end_b)
