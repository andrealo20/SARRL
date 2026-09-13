import json
import subprocess

from tools.scan_seed_usage import is_seed_key, scan_revision


def _repo(tmp_path, files):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "x"], cwd=tmp_path, check=True)


def test_seed_key_recognition_covers_repository_schemas():
    for name in (
        "seed",
        "episode_seed",
        "training_seeds",
        "common_episode_seeds",
        "evaluation_seed_start",
        "heldout_seed_start",
        "bootstrap_rng_seed",
        "SEED",
    ):
        assert is_seed_key(name), name
    for name in ("step", "reward", "episodes", "final_distance"):
        assert not is_seed_key(name), name


def test_scan_reads_committed_blobs_and_expands_ranges(tmp_path):
    _repo(
        tmp_path,
        {
            "a.csv": "episode,step,seed\n1,50141,7\n2,50979,50150\n",
            "b.json": json.dumps(
                {
                    "evaluation": {"seed": 50100, "steps": 50200},
                    "heldout": {"heldout_seed_start": 51990, "episodes": 20},
                    "data": {"data_seed_start": 40000, "data_seed_end": 40010},
                    "training_seeds": [30, 31, 51000],
                }
            ),
            "c.jsonl": json.dumps({"seed": 3}) + "\n" + json.dumps({"x_seed": 51999}) + "\n",
        },
    )
    summary = scan_revision(tmp_path, 50100, 51999)
    assert summary["files_scanned"] == 3
    # heldout_seed_start counts once as a value and once as the start of a range.
    assert summary["hits"] == 6
    assert summary["distinct_seeds"] == [50100, 50150, 51000, 51990, 51999]
    assert summary["files_with_hits"] == ["a.csv", "b.json", "c.jsonl"]
    assert any("range 51990..52009" in example[1] for example in summary["examples"])


def test_scan_ignores_working_tree_edits(tmp_path):
    _repo(tmp_path, {"a.csv": "seed\n50500\n"})
    (tmp_path / "a.csv").write_text("seed\n1\n")
    summary = scan_revision(tmp_path, 50100, 51999)
    assert summary["hits"] == 1 and summary["distinct_seeds"] == [50500]
