import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from sarrl.evaluation.factorial_campaign import V21_DESCRIPTIVE_ARM  # noqa: E402
from tools import run_factorial_campaign as runner  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_completion_marker_is_valid_only_with_matching_hashes(tmp_path):
    for name in runner.OUTPUT_FILES:
        (tmp_path / name).write_text(name)
    marker = tmp_path / "complete.json"
    marker.write_text(
        json.dumps({"hashes": {name: _sha(tmp_path / name) for name in runner.OUTPUT_FILES}})
    )
    assert runner.valid_completion_marker(marker)
    (tmp_path / "decision.json").write_text("changed")
    assert not runner.valid_completion_marker(marker)
    marker.write_text('{"hashes": {"manifest.js')  # truncated
    assert not runner.valid_completion_marker(marker)


def test_decision_range_is_unopened_and_registered_while_used_ranges_are_refused(monkeypatch):
    if not (ROOT / ".git").exists():
        pytest.skip("needs the git repository")
    calls = []

    def clean_scan(root, low, high, revision="HEAD", seen=None):
        calls.append(revision)
        return {"files_scanned": 1, "hits": 0, "files_with_hits": []}

    monkeypatch.setattr(runner, "scan_revision", clean_scan)
    report = runner.seed_range_is_unopened(ROOT, 54200, 56099)
    assert report["reserved_entry"]["label"] == "v2.1 decision block"
    assert calls[0] == "HEAD" and any(c.startswith("v2.0") for c in calls)
    # Registered ranges are refused even when no committed artifact records them.
    with pytest.raises(RuntimeError, match="registered"):
        runner.seed_range_is_unopened(ROOT, 9803000, 9803000)
    with pytest.raises(RuntimeError, match="registered"):
        runner.seed_range_is_unopened(ROOT, 52200, 52200)
    # A committed-artifact hit in any revision refuses the range.
    monkeypatch.setattr(
        runner,
        "scan_revision",
        lambda *a, **k: {"files_scanned": 1, "hits": 1, "files_with_hits": ["x.csv"]},
    )
    with pytest.raises(RuntimeError, match="already used"):
        runner.seed_range_is_unopened(ROOT, 54200, 56099)


def test_run_cell_labels_the_descriptive_arm_and_disables_compensation():
    record = runner.run_cell((V21_DESCRIPTIVE_ARM, "id_reference", 9803320, "mujoco"))
    assert record["arm"] == V21_DESCRIPTIVE_ARM
    assert record["selected_lag"] is not None
    assert record["plant"] == "mujoco"
