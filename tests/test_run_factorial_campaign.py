import pytest

pytest.importorskip("mujoco")

from sarrl.evaluation.factorial_campaign import V21_DESCRIPTIVE_ARM  # noqa: E402
from tools import run_factorial_campaign as runner  # noqa: E402


def test_run_cell_labels_the_descriptive_arm_and_disables_compensation():
    record = runner.run_cell((V21_DESCRIPTIVE_ARM, "id_reference", 9803320, "mujoco"))
    assert record["arm"] == V21_DESCRIPTIVE_ARM
    assert record["selected_lag"] is not None
    assert record["plant"] == "mujoco"
    plain = runner.run_cell(("adaptive_hocbf", "id_reference", 9803320, "mujoco"))
    assert plain["arm"] == "adaptive_hocbf"
