import json
import subprocess

import pytest

from sarrl.evaluation import adaptive_campaign
from tools import run_adaptive_campaign as runner


def _git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def _sealed_repo(tmp_path):
    """Freeze commit with the frozen paths, then a sealing commit outside them."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    for path in runner.V19_FROZEN_PATHS:
        target = tmp_path / path
        if path.endswith((".md", ".toml")):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("frozen\n")
        else:
            target.mkdir(parents=True, exist_ok=True)
            (target / "module.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "freeze"], cwd=tmp_path, check=True)
    frozen = _git(tmp_path, "rev-parse", "HEAD")
    seal = tmp_path / runner.V19_SEAL_FILE
    seal.write_text(json.dumps({"frozen_source_commit": frozen}) + "\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seal"], cwd=tmp_path, check=True)
    return frozen


def test_sealing_commit_after_the_freeze_passes(tmp_path):
    frozen = _sealed_repo(tmp_path)
    result = runner.verify_frozen_sources(tmp_path)
    assert result["sealed_commit"] == frozen
    assert result["head"] != frozen
    assert set(result["path_trees"]) == set(runner.V19_FROZEN_PATHS)


def test_missing_or_modified_seal_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="not been sealed"):
        _sealed_repo_without_seal(tmp_path)
    (tmp_path / "b").mkdir()
    _sealed_repo(tmp_path / "b")
    (tmp_path / "b" / runner.V19_SEAL_FILE).write_text("{\"frozen_source_commit\": \"0\"}\n")
    with pytest.raises(RuntimeError, match="modified or not committed"):
        runner.verify_frozen_sources(tmp_path / "b")


def _sealed_repo_without_seal(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    runner.verify_frozen_sources(tmp_path)


def test_frozen_path_change_after_the_seal_is_rejected(tmp_path):
    _sealed_repo(tmp_path)
    (tmp_path / "sarrl" / "module.py").write_text("VALUE = 2\n")
    with pytest.raises(RuntimeError, match="local modifications"):
        runner.verify_frozen_sources(tmp_path)
    subprocess.run(["git", "commit", "-q", "-am", "drift"], cwd=tmp_path, check=True)
    with pytest.raises(RuntimeError, match="frozen paths differ"):
        runner.verify_frozen_sources(tmp_path)


def test_untracked_file_inside_frozen_paths_is_rejected(tmp_path):
    _sealed_repo(tmp_path)
    (tmp_path / "tools" / "scratch.py").write_text("x = 1\n")
    with pytest.raises(RuntimeError, match="untracked"):
        runner.verify_frozen_sources(tmp_path)


def test_validated_prefix_accepts_ordered_logs_and_rejects_reordered_ones(tmp_path):
    cells = list(adaptive_campaign.v19_cells())
    log = tmp_path / "episodes.jsonl"
    records = [runner.run_cell(cell) for cell in cells[:2]]
    log.write_text("".join(json.dumps(r) + "\n" for r in records))
    assert runner.validated_prefix(log, cells) == 2
    log.write_text("".join(json.dumps(r) + "\n" for r in reversed(records)))
    with pytest.raises(RuntimeError, match="planned cell order"):
        runner.validated_prefix(log, cells)
    assert runner.validated_prefix(tmp_path / "missing.jsonl", cells) == 0


def test_run_cell_returns_a_serialisable_official_record():
    record = runner.run_cell(("fixed", "id_reference", adaptive_campaign.V19_PRIMARY_SEED_START))
    assert record["origin"] == "official"
    assert record["arm"] == "fixed" and record["seed"] == adaptive_campaign.V19_PRIMARY_SEED_START
    assert record["selected_lag"] is None
