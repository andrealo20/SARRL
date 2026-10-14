import json
import shutil
from pathlib import Path

import pytest

from tools.render_panda_demo import load_pair, tip_paths


def test_demo_uses_first_retained_paired_case():
    seed, records, traces, times = load_pair(Path("results/panda_adaptation_pilot"))
    assert seed == 9804300
    assert records["fixed"]["seed"] == records["adaptive"]["seed"] == seed
    assert len(traces["fixed"]) == len(traces["adaptive"]) == 301
    assert times[0] == 0 and times[-1] == 3
    assert not records["fixed"]["success"]
    assert records["adaptive"]["success"]


def test_demo_rejects_modified_evidence(tmp_path):
    source = Path("results/panda_adaptation_pilot")
    names = json.loads((source / "complete.json").read_text())["hashes"]
    for name in ("complete.json", *names):
        shutil.copyfile(source / name, tmp_path / name)
    with (tmp_path / "episodes.json").open("ab") as file:
        file.write(b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_pair(tmp_path)


def test_tip_closeup_preserves_recorded_endpoint_errors():
    pytest.importorskip("mujoco")
    import numpy as np

    from sarrl.robots.panda import load_panda_model

    _, records, traces, _ = load_pair(Path("results/panda_adaptation_pilot"))
    paths = tip_paths(load_panda_model(), traces)
    target = np.array(records["fixed"]["target_xyz_m"])
    np.testing.assert_allclose(paths["reference"][-1], target, atol=1e-12)
    for arm in ("fixed", "adaptive"):
        error = np.linalg.norm(paths[arm][-1] - target)
        assert error == pytest.approx(records[arm]["tip_error_m"], abs=1e-12)
    assert np.linalg.norm(paths["fixed"][-1] - paths["adaptive"][-1]) > 0.037
