import json

import numpy as np
import pytest

from sarrl.evaluation import (
    V15_PHASE_A_TRAINING_SEEDS,
    canonical_float,
    canonical_json,
    derive_reference_uncertainties,
    summarize_episode,
)


def _episodes():
    rows = []
    for policy in ("A2", "A3"):
        for seed in V15_PHASE_A_TRAINING_SEEDS:
            for offset in range(100):
                value = float(seed + offset + (0 if policy == "A2" else 100) + 1)
                rows.append(
                    summarize_episode(
                        policy=policy,
                        training_seed=seed,
                        ensemble_seed=seed,
                        episode_seed=60_000 + offset,
                        uncertainty=np.full(10, value),
                        error=np.arange(10),
                        attempted_pairs=10,
                        terminated=False,
                        truncated=True,
                    )
                )
    return rows


def test_reference_uncertainty_equally_pools_a2_and_a3_episode_medians():
    result = derive_reference_uncertainties(_episodes())
    assert len(result) == 5
    assert result[0]["u_ref"] == 100.5
    assert result[0]["pooled_episode_medians"] == 200
    assert result[0]["a2_scale_eligible_episodes"] == 100
    assert result[0]["a3_scale_eligible_episodes"] == 100


def test_reference_uncertainty_fails_closed_after_any_omitted_pair():
    rows = _episodes()
    row = rows[0]
    rows[0] = type(row)(
        **{
            **row.__dict__,
            "attempted_pairs": row.attempted_pairs + 1,
            "excluded_nonfinite_pairs": 1,
        }
    )
    with pytest.raises(ValueError, match="error-independent"):
        derive_reference_uncertainties(rows)


def test_canonical_serialization_uses_sorted_keys_and_frozen_float_format():
    payload = {"z": -0.0, "a": [True, 1.25, "x"]}
    encoded = canonical_json(payload)
    assert encoded == '{"a":[true,1.25,"x"],"z":0}'
    assert json.loads(encoded) == {"a": [True, 1.25, "x"], "z": 0}
    assert canonical_float(np.nextafter(1.0, 2.0)) == "1.0000000000000002"
