import hashlib
import json
from pathlib import Path

import pytest

from tools import campaign_guards as guards

ROOT = Path(__file__).resolve().parents[1]
FILES = ("a.json", "b.jsonl")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_completion_marker_is_valid_only_with_matching_hashes(tmp_path):
    for name in FILES:
        (tmp_path / name).write_text(name)
    marker = tmp_path / "complete.json"
    marker.write_text(json.dumps({"hashes": {name: _sha(tmp_path / name) for name in FILES}}))
    assert guards.valid_completion_marker(marker, FILES)
    (tmp_path / "b.jsonl").write_text("changed")
    assert not guards.valid_completion_marker(marker, FILES)
    marker.write_text('{"hashes": {"a.js')  # truncated
    assert not guards.valid_completion_marker(marker, FILES)
    marker.write_text(json.dumps({"hashes": {"a.json": _sha(tmp_path / "a.json")}}))
    assert not guards.valid_completion_marker(marker, FILES)  # missing output


def _key(record):
    return (record["arm"], record["seed"])


def test_journal_drops_only_a_truncated_final_record(tmp_path):
    journal = tmp_path / "journal.jsonl"
    fields = {"arm", "seed", "value", "session"}
    planned = {("a", 1), ("a", 2), ("a", 3)}
    good = [json.dumps({"arm": "a", "seed": s, "value": s, "session": "x"}) for s in (1, 2)]
    journal.write_text("\n".join(good) + "\n" + '{"arm": "a", "seed": 3, "val')
    done = guards.journal_records(journal, planned, fields, _key)
    assert set(done) == {("a", 1), ("a", 2)}
    # A malformed record that is not the last one stops the resume.
    journal.write_text(good[0] + "\n" + '{"arm": "a"' + "\n" + good[1] + "\n")
    with pytest.raises(RuntimeError, match="malformed"):
        guards.journal_records(journal, planned, fields, _key)
    # Duplicates, unplanned cells and unexpected fields stop it too.
    journal.write_text(good[0] + "\n" + good[0] + "\n")
    with pytest.raises(RuntimeError, match="unique"):
        guards.journal_records(journal, planned, fields, _key)
    journal.write_text(json.dumps({"arm": "b", "seed": 1, "value": 0, "session": "x"}) + "\n")
    with pytest.raises(RuntimeError, match="planned"):
        guards.journal_records(journal, planned, fields, _key)
    journal.write_text(json.dumps({"arm": "a", "seed": 1, "session": "x"}) + "\n")
    with pytest.raises(RuntimeError, match="fields"):
        guards.journal_records(journal, planned, fields, _key)


def test_reproduction_failure_opens_no_decision_cell():
    executed = []

    def execute(block):
        executed.append(list(block))

    with pytest.raises(RuntimeError, match="no decision seed opened"):
        guards.run_two_phases(
            ["r1", "r2"], ["d1"], execute, lambda: {"mismatches": [["r1", "field"]]}
        )
    assert executed == [["r1", "r2"]]
    executed.clear()
    report = guards.run_two_phases(["r1"], ["d1", "d2"], execute, lambda: {"mismatches": []})
    assert executed == [["r1"], ["d1", "d2"]] and report == {"mismatches": []}


def test_decision_range_is_checked_against_history_and_registry(monkeypatch, tmp_path):
    if not (ROOT / ".git").exists():
        pytest.skip("needs the git repository")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "ranges": [
                    {"label": "used", "low": 100, "high": 199, "status": "used"},
                    {"label": "new", "low": 200, "high": 299, "status": "reserved"},
                ]
            }
        )
    )
    trees = [("c1", "t1"), ("c2", "t2")]
    monkeypatch.setattr(guards, "reachable_trees", lambda root: trees)
    monkeypatch.setattr(guards, "committed_blobs", lambda root, tree: [("x.csv", tree)])
    monkeypatch.setattr(guards, "scan_blobs", lambda root, entries, low, high, seen: [])
    report = guards.seed_range_is_unopened(tmp_path, 200, 299, "registry.json")
    assert report["commits_scanned"] == 2 and report["distinct_trees"] == ["t1", "t2"]
    assert report["reserved_entry"]["label"] == "new"
    with pytest.raises(RuntimeError, match="registered"):
        guards.seed_range_is_unopened(tmp_path, 150, 250, "registry.json")
    with pytest.raises(RuntimeError, match="reserved exactly once"):
        guards.seed_range_is_unopened(tmp_path, 300, 399, "registry.json")
    monkeypatch.setattr(
        guards, "scan_blobs", lambda root, entries, low, high, seen: [("x.csv", "row 1", 250)]
    )
    with pytest.raises(RuntimeError, match="already used in commit c1"):
        guards.seed_range_is_unopened(tmp_path, 200, 299, "registry.json")


def test_reachable_trees_lists_distinct_trees_of_the_real_repository():
    if not (ROOT / ".git").exists():
        pytest.skip("needs the git repository")
    trees = guards.reachable_trees(ROOT)
    assert len(trees) > 50
    assert len({tree for _, tree in trees}) == len(trees)
    assert all(len(commit) == 40 and len(tree) == 40 for commit, tree in trees)
