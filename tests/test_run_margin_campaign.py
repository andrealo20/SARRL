import csv

import pytest

pytest.importorskip("mujoco")

from sarrl.evaluation.adaptive_campaign import load_episodes  # noqa: E402
from sarrl.evaluation.margin_campaign import (  # noqa: E402
    V22_CONSTANT,
    V22_COVARIANCE,
    V22_MARGINS,
    V22_NONE,
)
from tools import run_margin_campaign as runner  # noqa: E402


def test_run_cell_applies_each_arm_margin_and_writes_split_csv_columns(tmp_path):
    seed = 9803320  # pilot range, outside every official block
    records = [runner.run_cell((arm, "id_reference", seed, "mujoco")) for arm in V22_MARGINS]
    for record in records:
        assert record["certificate_margin"] == V22_MARGINS[record["arm"]]
        assert record["plant"] == "mujoco" and record["selected_lag"] is not None
    by_arm = {record["arm"]: record for record in records}
    assert by_arm[V22_NONE]["margin_steps"] == 0
    assert by_arm[V22_NONE]["margin_mean_command"] is None
    assert by_arm[V22_COVARIANCE]["margin_steps"] > 0
    assert by_arm[V22_CONSTANT]["margin_mean_command"][0] <= 2.0

    log = tmp_path / "episodes.jsonl"
    log.write_text("".join(runner.json.dumps(record) + "\n" for record in records))
    episodes = load_episodes(log)
    runner.write_csv(tmp_path / "episodes.csv", episodes)
    with (tmp_path / "episodes.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["arm"] for row in rows] == list(V22_MARGINS)
    assert "margin_mean_command" not in rows[0]
    assert rows[0]["margin_mean_command_joint1"] == ""
    assert float(rows[1]["margin_mean_command_joint1"]) > 0.0
    assert float(rows[1]["prediction_error_rms_joint2"]) >= 0.0
