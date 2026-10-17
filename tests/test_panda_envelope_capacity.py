import itertools

import numpy as np
import pytest

from sarrl.evaluation.panda_envelope_capacity import affine_box_error, optimistic_capacity


def test_exact_box_maximum_matches_all_128_corners_and_its_attaining_commands():
    delta = np.arange(49).reshape(7, 7) / 100 - 0.25
    drift = np.linspace(-0.5, 0.5, 7)
    limit = np.linspace(1, 2, 7)
    floor, attaining = affine_box_error(delta, drift, np.zeros((7, 7)), np.zeros(7), limit)
    torques = np.array(list(itertools.product((-1, 1), repeat=7))) * limit
    brute = np.max(np.abs(torques @ delta.T + drift), axis=0)
    np.testing.assert_allclose(floor, brute, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(np.abs(np.diag(delta @ attaining.T) + drift), floor)
    assert np.all(np.abs(attaining) <= limit)


def test_identical_models_have_zero_error_including_zero_coefficients():
    floor, torques = affine_box_error(np.eye(7), np.ones(7), np.eye(7), np.ones(7), np.ones(7))
    np.testing.assert_array_equal(floor, np.zeros(7))
    np.testing.assert_array_equal(torques, np.zeros((7, 7)))


@pytest.mark.parametrize(
    "argument,value",
    [
        (0, np.zeros((6, 6))),
        (1, np.full(7, np.nan)),
        (2, np.full((7, 7), np.inf)),
        (3, np.zeros(8)),
        (4, np.zeros(7)),
    ],
)
def test_invalid_box_bound_arguments_are_rejected(argument, value):
    values = [np.eye(7), np.zeros(7), np.eye(7), np.zeros(7), np.ones(7)]
    values[argument] = value
    with pytest.raises(ValueError):
        affine_box_error(*values)


def policy():
    return {
        "config": {
            "control_dt": 0.01,
            "max_delay_steps": 4,
            "horizon_steps": 200,
            "kp": 25.0,
            "kd": 10.0,
        },
        "anchor": np.zeros(7),
        "matrix": np.eye(7),
        "drift": np.zeros(7),
        "torque_limit": np.full(7, 20.0),
    }


def test_larger_error_floor_cannot_recover_a_rejected_scalar_tube():
    p, lower, upper = policy(), np.full(7, -2.0), np.full(7, 2.0)
    zero = optimistic_capacity(p, np.zeros(7), lower, upper)
    first = optimistic_capacity(p, np.full(7, 1.0), lower, upper)
    larger = optimistic_capacity(p, np.full(7, 2.0), lower, upper)
    assert all(case["feasible"] for case in zero)
    assert not any(case["feasible"] for case in first)
    for a, b in zip(first, larger, strict=True):
        assert b["minimum_radius"] >= a["minimum_radius"]
        assert b["admissible_radius"] <= a["admissible_radius"]
        assert not b["feasible"]


def test_smaller_domain_cannot_recover_a_rejected_scalar_tube():
    p, error = policy(), np.ones(7)
    wide = optimistic_capacity(p, error, np.full(7, -2.0), np.full(7, 2.0))
    narrow = optimistic_capacity(p, error, np.full(7, -0.1), np.full(7, 0.1))
    for a, b in zip(wide, narrow, strict=True):
        assert b["admissible_radius"] <= a["admissible_radius"]
        assert not b["feasible"]
