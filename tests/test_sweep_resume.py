import json

import pytest

from tools.run_sac_sweep import _resume_plan


def _manifest(
    path,
    requested_steps,
    commit="abc",
    context_sha256=None,
):
    path.write_text(
        json.dumps(
            {
                "config": {
                    "requested_steps": requested_steps,
                    "context": {
                        "checkpoint_sha256": context_sha256,
                    },
                },
                "runtime": {"git_commit": commit},
            }
        )
    )


def test_resume_plan_starts_fresh_when_no_checkpoint_exists(tmp_path):
    checkpoint, complete = _resume_plan(tmp_path, 200_000, "abc")
    assert checkpoint is None
    assert complete is False


def test_resume_plan_reuses_completed_matching_run(tmp_path):
    _manifest(tmp_path / "run_manifest.json", 200_000)
    (tmp_path / "training_final.pt").touch()

    checkpoint, complete = _resume_plan(tmp_path, 200_000, "abc")

    assert checkpoint is None
    assert complete is True


def test_resume_plan_extends_from_previous_final_checkpoint(tmp_path):
    _manifest(tmp_path / "run_manifest.json", 100_000)
    final = tmp_path / "training_final.pt"
    final.touch()

    checkpoint, complete = _resume_plan(tmp_path, 200_000, "abc")

    assert checkpoint == final
    assert complete is False


def test_resume_plan_uses_latest_periodic_checkpoint(tmp_path):
    _manifest(tmp_path / "run_manifest.json", 200_000)

    old = tmp_path / "train_step50000.pt"
    latest = tmp_path / "train_step100000.pt"
    old.touch()
    latest.touch()

    checkpoint, complete = _resume_plan(tmp_path, 200_000, "abc")

    assert checkpoint == latest
    assert complete is False


def test_resume_plan_refuses_cross_commit_resume(tmp_path):
    _manifest(tmp_path / "run_manifest.json", 200_000, commit="old")
    (tmp_path / "train_step50000.pt").touch()

    with pytest.raises(ValueError, match="different git commits"):
        _resume_plan(tmp_path, 200_000, "new")


def test_resume_plan_accepts_matching_context_checkpoint(tmp_path):
    _manifest(
        tmp_path / "run_manifest.json",
        200_000,
        context_sha256="context-a",
    )
    (tmp_path / "training_final.pt").touch()

    checkpoint, complete = _resume_plan(
        tmp_path,
        200_000,
        "abc",
        "context-a",
    )

    assert checkpoint is None
    assert complete is True


def test_resume_plan_refuses_different_context_checkpoint(tmp_path):
    _manifest(
        tmp_path / "run_manifest.json",
        200_000,
        context_sha256="context-a",
    )
    (tmp_path / "train_step50000.pt").touch()

    with pytest.raises(
        ValueError,
        match="different context checkpoint",
    ):
        _resume_plan(
            tmp_path,
            200_000,
            "abc",
            "context-b",
        )


def test_resume_plan_refuses_adding_context_to_old_run(tmp_path):
    _manifest(
        tmp_path / "run_manifest.json",
        200_000,
        context_sha256=None,
    )
    (tmp_path / "train_step50000.pt").touch()

    with pytest.raises(
        ValueError,
        match="different context checkpoint",
    ):
        _resume_plan(
            tmp_path,
            200_000,
            "abc",
            "context-a",
        )
