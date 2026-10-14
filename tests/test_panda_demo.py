import json
import shutil
from pathlib import Path

import pytest

from tools.render_panda_demo import load_pair


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
