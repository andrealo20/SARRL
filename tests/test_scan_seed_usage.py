import json
import subprocess

from tools.scan_seed_usage import is_seed_key, scan_tracked


def test_seed_key_recognition_ignores_step_and_reward_columns():
    assert is_seed_key("seed") and is_seed_key("episode_seed") and is_seed_key("seed_start")
    assert not is_seed_key("step") and not is_seed_key("reward")
    assert not is_seed_key("episodes_total")


def test_scan_counts_only_seed_columns_and_keys(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "a.csv").write_text("episode,step,seed\n1,50141,7\n2,50979,50150\n")
    (tmp_path / "b.json").write_text(json.dumps({"evaluation": {"seed": 50100, "steps": 50200}}))
    lines = [json.dumps({"seed": 3}), json.dumps({"x_seed": 51999})]
    (tmp_path / "c.jsonl").write_text("\n".join(lines) + "\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    summary = scan_tracked(tmp_path, 50100, 51999)
    assert summary["files_scanned"] == 3
    assert summary["distinct_seeds"] == [50100, 50150, 51999]
    assert summary["hits"] == 3
