from pathlib import Path

import pytest

import sarrl.evaluation.provenance as provenance
from sarrl.evaluation import (
    assert_repository_import_root,
    assert_source_tree_clean,
    imported_repository_root,
    source_tree_dirty_paths,
)


def test_imported_repository_root_matches_active_checkout():
    root = Path(__file__).resolve().parents[1]

    assert imported_repository_root() == root.resolve()
    assert assert_repository_import_root(root) == root.resolve()


def test_import_root_guard_rejects_different_checkout(tmp_path):
    with pytest.raises(
        RuntimeError,
        match="import/check-out mismatch",
    ):
        assert_repository_import_root(tmp_path)


def test_source_tree_filter_ignores_experiment_outputs(monkeypatch):
    def fake_status(*args, **kwargs):
        return (
            " M tools/train_sac.py\n"
            "?? results/planar_ablations/run.csv\n"
            "?? artifacts/checkpoint.pt\n"
        )

    monkeypatch.setattr(
        provenance.subprocess,
        "check_output",
        fake_status,
    )

    assert source_tree_dirty_paths(".") == [
        "tools/train_sac.py",
    ]


def test_source_tree_guard_rejects_source_changes(monkeypatch):
    def fake_status(*args, **kwargs):
        return " M sarrl/rl/sac.py\n"

    monkeypatch.setattr(
        provenance.subprocess,
        "check_output",
        fake_status,
    )

    with pytest.raises(
        RuntimeError,
        match="uncommitted source changes",
    ):
        assert_source_tree_clean(".")
