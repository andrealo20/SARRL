import json
from pathlib import Path

import numpy as np
import pytest

from sarrl.evaluation.panda_backup_integration import selected_tasks, timing_summary
from tools.run_panda_backup_integration import selected


def test_integration_selection_uses_only_the_fixed_thirty_two_retained_tasks():
    root = Path(__file__).resolve().parents[1]
    rows = json.loads((root / "results/panda_joint_filter_pilot/episodes.json").read_text())
    tasks = selected_tasks(rows)
    assert tasks == selected(rows)
    assert len(tasks) == 32
    assert sum(seed == 9805800 for condition, seed in tasks) == 12
    assert all(9805800 <= seed < 9805820 for condition, seed in tasks)
    altered = rows + [
        {
            "arm": "bank_filtered",
            "scenario": "fast_clean_d40",
            "seed": 9805900,
            "unsafe_episode": True,
        }
    ]
    with pytest.raises(ValueError):
        selected_tasks(altered)


def test_timing_summary_preserves_overruns_and_rejects_incomplete_cycles():
    values = np.ones(300)
    values[-1] = 11
    report = timing_summary(values)
    assert report["over_10_ms"] == 1 and report["maximum_ms"] == 11
    with pytest.raises(ValueError):
        timing_summary(values[:-1])
    values[0] = np.nan
    with pytest.raises(ValueError):
        timing_summary(values)
