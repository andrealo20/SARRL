import subprocess

import pytest

from sarrl.evaluation import adaptive_campaign
from tools import run_adaptive_campaign as runner


def _git(*args):
    return subprocess.run(
        ["git", *args], cwd=runner.ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_unsealed_protocol_refuses_to_run(monkeypatch):
    monkeypatch.setattr(runner, "V19_FROZEN_SOURCE_COMMIT", None)
    with pytest.raises(RuntimeError, match="not been sealed"):
        runner.verify_frozen_sources()


def test_sealed_commit_equal_to_head_passes_when_frozen_paths_are_clean(monkeypatch):
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=all", "--", *runner.V19_FROZEN_PATHS)
    if status:
        pytest.skip("frozen paths carry local changes in this checkout")
    monkeypatch.setattr(runner, "V19_FROZEN_SOURCE_COMMIT", head)
    frozen = runner.verify_frozen_sources()
    assert frozen["head"] == frozen["sealed_commit"] == head
    assert set(frozen["path_trees"]) == set(runner.V19_FROZEN_PATHS)


def test_sealed_commit_differing_from_head_is_rejected(monkeypatch):
    parent = _git("rev-parse", "HEAD~1")
    diff = subprocess.run(
        ["git", "diff", "--quiet", parent, "HEAD", "--", *runner.V19_FROZEN_PATHS],
        cwd=runner.ROOT,
        check=False,
    )
    if diff.returncode == 0:
        pytest.skip("HEAD~1 has identical frozen paths")
    monkeypatch.setattr(runner, "V19_FROZEN_SOURCE_COMMIT", parent)
    with pytest.raises(RuntimeError, match="frozen paths differ"):
        runner.verify_frozen_sources()


def test_run_cell_returns_a_serialisable_official_record():
    record = runner.run_cell(("fixed", "id_reference", adaptive_campaign.V19_SEED_START))
    assert record["origin"] == "official"
    assert record["arm"] == "fixed" and record["seed"] == adaptive_campaign.V19_SEED_START
    assert record["selected_lag"] is None
