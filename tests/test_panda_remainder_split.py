import numpy as np
import pytest

from tools.split_panda_mean_value_remainder import share, split_branch

DT = 0.01


def _maps(width):
    zero = (np.zeros((7, width)), np.zeros((7, width)), np.zeros(width))
    return zero


def _taylor_steps():
    """Two intervals whose only residual source is the held-flow Taylor term."""
    jerk, q, v, steps = np.ones(7), np.zeros(7), np.zeros(7), []
    for k in range(2):
        v_end, q_end = v + DT**2 / 2 * jerk, q + DT * v + DT**3 / 6 * jerk
        steps.append(
            {
                "feedback": None if k == 0 else _maps(14),
                "acceleration": _maps(29),
                "sent": np.zeros(7),
                "delivered": np.zeros(7),
                "acceleration_start": np.zeros(7),
                "jerk": jerk,
                "q_end": q_end,
                "v_end": v_end,
            }
        )
        q, v = q_end, v_end
    return steps


def test_taylor_only_branch_is_attributed_to_taylor():
    result = split_branch(_taylor_steps(), lag=0, dt=DT)
    assert share(result["velocity"], "taylor") == pytest.approx(1.0)
    assert share(result["position"], "taylor") == pytest.approx(1.0)
    assert result["steps"][1]["v_inherited"] == pytest.approx(result["steps"][0]["v_radius"])


def test_variation_term_keeps_its_origin_through_propagation():
    steps = _taylor_steps()
    mag, rad, gen = _maps(29)
    rad[:, 0], gen[0] = 1.0, 2.0  # acceleration variation of 2 per joint
    steps[0]["acceleration"] = (mag, rad, gen)
    steps[0]["acceleration_start"] = np.full(7, 2.0)
    steps[0]["v_end"] = steps[0]["v_end"] + DT * 2.0
    steps[0]["q_end"] = steps[0]["q_end"] + DT**2 / 2 * 2.0
    steps[1]["v_end"] = steps[1]["v_end"] + DT * 2.0
    steps[1]["q_end"] = steps[1]["q_end"] + DT * DT * 2.0 + DT**2 / 2 * 2.0
    result = split_branch(steps, lag=0, dt=DT)
    velocity = result["velocity"]
    assert velocity["jacobian_acceleration"] == pytest.approx(7 * DT * 2.0)
    assert sum(velocity.values()) == pytest.approx(float(np.sum(steps[1]["v_end"])))


def test_reconstruction_wider_than_the_record_is_rejected():
    steps = _taylor_steps()
    steps[0]["v_end"] = steps[0]["v_end"] / 2
    with pytest.raises(ValueError, match="exceeds the record"):
        split_branch(steps, lag=0, dt=DT)
