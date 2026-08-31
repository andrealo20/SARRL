import numpy as np
import pytest

from sarrl.rl import ReplayBuffer


def _fill(buf):
    for i in range(20):
        obs = np.array([i, i + 1], dtype=np.float32)
        buf.add(obs, np.array([i / 20], dtype=np.float32), float(i), obs + 1, i % 3 == 0)


def test_replay_capacity_wraps():
    b = ReplayBuffer(2, 1, capacity=5, seed=0)
    _fill(b)
    assert len(b) == 5


def test_replay_sampling_is_seed_reproducible():
    a = ReplayBuffer(2, 1, 20, seed=9)
    b = ReplayBuffer(2, 1, 20, seed=9)
    _fill(a)
    _fill(b)
    sa, sb = a.sample(8), b.sample(8)
    for key in sa:
        np.testing.assert_array_equal(sa[key], sb[key])


def test_replay_rejects_oversized_batch():
    b = ReplayBuffer(2, 1, 5)
    with pytest.raises(ValueError):
        b.sample(1)
