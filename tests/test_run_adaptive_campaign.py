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
    frozen = _sealed_repo(tmp_path / "b")
    seal = tmp_path / "b" / runner.V19_SEAL_FILE
    seal.write_text(json.dumps({"frozen_source_commit": frozen, "note": "edited"}) + "\n")
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


def test_seal_must_be_a_full_commit_id_that_is_an_ancestor_of_head(tmp_path):
    _sealed_repo(tmp_path)
    seal = tmp_path / runner.V19_SEAL_FILE
    for bad in ("HEAD", "abc123", "0" * 40):
        seal.write_text(json.dumps({"frozen_source_commit": bad}) + "\n")
        subprocess.run(["git", "commit", "-q", "-am", "reseal"], cwd=tmp_path, check=True)
        with pytest.raises(RuntimeError, match="commit"):
            runner.verify_frozen_sources(tmp_path)


def _journal_record(cell, seed_override=None):
    # A record with the planned shape, computed on a pilot seed so that no
    # decision seed is ever executed by the test suite.
    arm, scenario, seed = cell
    record = runner.run_cell((arm, scenario, 9803000))
    record["seed"] = seed if seed_override is None else seed_override
    record["session"] = "test"
    return record


def test_journal_records_accept_unordered_planned_cells_and_reject_others(tmp_path):
    cells = list(adaptive_campaign.v19_cells())
    planned = set(cells)
    journal = tmp_path / "journal.jsonl"
    records = [_journal_record(cells[1]), _journal_record(cells[0])]
    journal.write_text("".join(json.dumps(r) + "\n" for r in records))
    done = runner.journal_records(journal, planned)
    assert set(done) == {cells[0], cells[1]}
    assert "session" not in done[cells[0]]
    journal.write_text(json.dumps(records[0]) + "\n" + json.dumps(records[0]) + "\n")
    with pytest.raises(RuntimeError, match="unique"):
        runner.journal_records(journal, planned)
    journal.write_text(json.dumps(_journal_record(cells[0], seed_override=1)) + "\n")
    with pytest.raises(RuntimeError, match="planned"):
        runner.journal_records(journal, planned)
    assert runner.journal_records(tmp_path / "missing.jsonl", planned) == {}


def test_campaign_lock_is_exclusive(tmp_path):
    with runner.CampaignLock(tmp_path):
        with pytest.raises(RuntimeError, match="campaign lock"):
            with runner.CampaignLock(tmp_path):
                pass
    with runner.CampaignLock(tmp_path):
        pass


def test_run_cell_on_a_pilot_seed_returns_an_official_record():
    record = runner.run_cell(("fixed", "id_reference", 9803000))
    assert record["origin"] == "official"
    assert record["arm"] == "fixed" and record["seed"] == 9803000
    assert record["selected_lag"] is None


def test_decision_seed_range_is_the_documented_one():
    start = adaptive_campaign.V19_PRIMARY_SEED_START
    end = start + adaptive_campaign.V19_PRIMARY_EPISODES - 1
    assert (start, end) == (50200, 52099)
