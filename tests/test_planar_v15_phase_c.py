import numpy as np
import pytest

from tools.run_planar_v15_phase_c import _bootstrap_difference


def test_phase_c_bootstrap_uses_frozen_five_by_one_hundred_pairing():
    matrix = np.ones((5, 100), dtype=np.float64)

    assert _bootstrap_difference(matrix) == pytest.approx((1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="5x100"):
        _bootstrap_difference(np.ones((5, 99), dtype=np.float64))
