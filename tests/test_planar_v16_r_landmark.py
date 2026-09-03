import csv
import gzip
from pathlib import Path

import pytest

from tools.run_planar_v16_r_landmark import (
    CONDITION,
    EXPECTED_ROWS_IN_WINDOW,
    POPULATION,
    SCHEMA,
    WINDOW_FIRST_STEP,
    WINDOW_LAST_STEP,
    build_landmark_rows,
    key_of,
    load_ensemble_seeds,
    load_outcomes,
)


def _write_gate_episodes(path: Path, keys: list[tuple]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "population", "condition", "training_seed", "ensemble_seed",
                "scenario", "episode_seed",
            ]
        )
        for population, condition, training_seed, scenario, episode_seed in keys:
            w.writerow(
                [population, condition, training_seed, training_seed, scenario, episode_seed]
            )


def _write_safety_diagnostics(path: Path, rows: list[dict]) -> None:
    fields = [
        "population", "condition", "training_seed", "scenario", "seed",
        "unsafe_episode", "safety_infeasible",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_raw(path: Path, keys: list[tuple], n_steps: int = 25) -> None:
    fields = [
        "population", "condition", "training_seed", "ensemble_seed", "scenario",
        "episode_seed", "step", "uncertainty_norm", "uncertainty_ratio",
    ]
    with gzip.open(path, "wt", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for population, condition, training_seed, scenario, episode_seed in keys:
            for step in range(n_steps):
                w.writerow(
                    {
                        "population": population,
                        "condition": condition,
                        "training_seed": training_seed,
                        "ensemble_seed": training_seed,
                        "scenario": scenario,
                        "episode_seed": episode_seed,
                        "step": step,
                        "uncertainty_norm": float(step),
                        "uncertainty_ratio": float(step) / 2.0,
                    }
                )
            # a few post-window rows the landmark builder must ignore
            for step in range(n_steps, n_steps + 5):
                w.writerow(
                    {
                        "population": population,
                        "condition": condition,
                        "training_seed": training_seed,
                        "ensemble_seed": training_seed,
                        "scenario": scenario,
                        "episode_seed": episode_seed,
                        "step": step,
                        "uncertainty_norm": 9999.0,
                        "uncertainty_ratio": 9999.0,
                    }
                )


KEYS = [
    ("safety", CONDITION, "0", "id_reference", "50000"),
    ("safety", CONDITION, "0", "ood_compound", "50000"),
]


@pytest.fixture
def fixtures(tmp_path):
    raw = tmp_path / "transitions.csv.gz"
    gate = tmp_path / "gate_episodes.csv"
    safety = tmp_path / "safety_diagnostics.csv"
    _write_raw(raw, KEYS)
    _write_gate_episodes(gate, KEYS)
    _write_safety_diagnostics(
        safety,
        [
            {
                "population": "safety", "condition": CONDITION, "training_seed": "0",
                "scenario": "id_reference", "seed": "50000",
                "unsafe_episode": "True", "safety_infeasible": "False",
            },
            {
                "population": "safety", "condition": CONDITION, "training_seed": "0",
                "scenario": "ood_compound", "seed": "50000",
                "unsafe_episode": "False", "safety_infeasible": "True",
            },
        ],
    )
    return raw, gate, safety


def test_window_bounds_are_frozen():
    assert WINDOW_FIRST_STEP == 0
    assert WINDOW_LAST_STEP == 24
    assert EXPECTED_ROWS_IN_WINDOW == 25


def test_schema_matches_the_frozen_plan():
    assert SCHEMA == (
        "population", "condition", "training_seed", "ensemble_seed", "scenario",
        "episode_seed", "window_first_step", "window_last_step", "n_rows_in_window",
        "uncertainty_norm_landmark_median", "uncertainty_ratio_landmark_median",
        "unsafe_episode", "safety_infeasible", "operational_failure",
    )


def test_landmark_median_ignores_rows_outside_the_window(fixtures):
    raw, gate, safety = fixtures
    outcomes = load_outcomes(safety)
    ensembles = load_ensemble_seeds(gate)
    rows = build_landmark_rows(raw, outcomes, ensembles)
    assert len(rows) == 2
    for row in rows:
        assert row["n_rows_in_window"] == 25
        # steps 0..24 -> median 12.0; the 9999 sentinel rows must be excluded
        assert row["uncertainty_norm_landmark_median"] == pytest.approx(12.0)
        assert row["uncertainty_ratio_landmark_median"] == pytest.approx(6.0)


def test_operational_failure_is_the_union(fixtures):
    raw, gate, safety = fixtures
    outcomes = load_outcomes(safety)
    ensembles = load_ensemble_seeds(gate)
    rows = build_landmark_rows(raw, outcomes, ensembles)
    by_scenario = {r["scenario"]: r for r in rows}
    assert by_scenario["id_reference"]["unsafe_episode"] is True
    assert by_scenario["id_reference"]["safety_infeasible"] is False
    assert by_scenario["id_reference"]["operational_failure"] is True
    assert by_scenario["ood_compound"]["unsafe_episode"] is False
    assert by_scenario["ood_compound"]["safety_infeasible"] is True
    assert by_scenario["ood_compound"]["operational_failure"] is True


def test_short_episode_raises_instead_of_silently_truncating(tmp_path):
    raw = tmp_path / "transitions.csv.gz"
    gate = tmp_path / "gate_episodes.csv"
    safety = tmp_path / "safety_diagnostics.csv"
    short_keys = [("safety", CONDITION, "0", "id_reference", "50000")]
    _write_raw(raw, short_keys, n_steps=10)  # fewer than 25 rows in the window
    _write_gate_episodes(gate, short_keys)
    _write_safety_diagnostics(
        safety,
        [
            {
                "population": "safety", "condition": CONDITION, "training_seed": "0",
                "scenario": "id_reference", "seed": "50000",
                "unsafe_episode": "False", "safety_infeasible": "False",
            }
        ],
    )
    outcomes = load_outcomes(safety)
    ensembles = load_ensemble_seeds(gate)
    with pytest.raises(RuntimeError, match="expected exactly 25"):
        build_landmark_rows(raw, outcomes, ensembles)


def test_mismatched_keys_between_raw_and_outcomes_raise(fixtures):
    raw, gate, safety = fixtures
    outcomes = load_outcomes(safety)
    ensembles = load_ensemble_seeds(gate)
    dropped_key = {
        "population": "safety",
        "condition": CONDITION,
        "training_seed": "0",
        "scenario": "id_reference",
        "episode_seed": "50000",
    }
    outcomes.pop(key_of(dropped_key))
    with pytest.raises(RuntimeError, match="key mismatch"):
        build_landmark_rows(raw, outcomes, ensembles)


def test_other_conditions_and_populations_are_filtered_out(tmp_path):
    raw = tmp_path / "transitions.csv.gz"
    gate = tmp_path / "gate_episodes.csv"
    safety = tmp_path / "safety_diagnostics.csv"
    keys = KEYS + [("heldout", "A4c", "0", "id_reference", "40000")]
    _write_raw(raw, keys)
    _write_gate_episodes(gate, keys)
    _write_safety_diagnostics(
        safety,
        [
            {
                "population": p, "condition": c, "training_seed": ts,
                "scenario": sc, "seed": es,
                "unsafe_episode": "False", "safety_infeasible": "False",
            }
            for p, c, ts, sc, es in keys
        ],
    )
    outcomes = load_outcomes(safety)
    assert (POPULATION, CONDITION, "0", "id_reference", "50000") in outcomes
    assert ("heldout", "A4c", "0", "id_reference", "40000") not in outcomes
    ensembles = load_ensemble_seeds(gate)
    rows = build_landmark_rows(raw, outcomes, ensembles)
    assert len(rows) == 2
    assert {r["condition"] for r in rows} == {CONDITION}
