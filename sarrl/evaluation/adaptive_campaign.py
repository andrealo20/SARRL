"""Preregistered v1.9 campaign: adaptive against fixed nominal control behind the HOCBF.

Everything that defines the decision lives in this module and is frozen at the
source commit that runs the campaign: arms, seeds, endpoints, bootstrap,
thresholds and the order in which the rule is applied. The runner only
orchestrates episodes and writes the files; the analysis reads them back.
"""

from __future__ import annotations

import numpy as np

from sarrl.controllers import AdaptiveNominalConfig
from sarrl.evaluation.adaptive_pilot import PilotEpisode

V19_ARMS = ("fixed", "fixed_hocbf", "adaptive", "adaptive_hocbf")
V19_PRIMARY_ARMS = ("fixed_hocbf", "adaptive_hocbf")
V19_SCENARIOS = ("id_reference", "ood_compound", "motor_fault")
V19_PRIMARY_SCENARIOS = ("id_reference", "motor_fault")
V19_SEED_START = 50000
V19_EPISODES_PER_CELL = 100
V19_BOOTSTRAP_DRAWS = 10_000
V19_BOOTSTRAP_SEED = 190_000
V19_SUCCESS_GAIN = 0.20
V19_UNSAFE_MARGIN = 0.02
V19_COMPENSATE_DELAY = True


def v19_config() -> AdaptiveNominalConfig:
    """The frozen estimator configuration: library defaults, nothing tuned after the pilot."""
    return AdaptiveNominalConfig()


def v19_protocol_dict() -> dict:
    """JSON-friendly record of the frozen protocol, written into the manifest."""
    config = v19_config()
    return {
        "release_target": "v1.9.0",
        "campaign": "adaptive_nominal_v19",
        "training": False,
        "arms": list(V19_ARMS),
        "primary_contrast": {
            "treatment": "adaptive_hocbf",
            "reference": "fixed_hocbf",
            "scenarios": list(V19_PRIMARY_SCENARIOS),
            "metric": "success",
        },
        "seeds": {
            scenario: [V19_SEED_START, V19_EPISODES_PER_CELL] for scenario in V19_SCENARIOS
        },
        "bootstrap": {
            "type": "paired_over_episode_seeds",
            "draws": V19_BOOTSTRAP_DRAWS,
            "seed": V19_BOOTSTRAP_SEED,
        },
        "decision": {
            "order": ["safety_veto", "primary_success", "otherwise_inconclusive"],
            "safety_veto": {
                "metric": "unsafe_episode",
                "scenarios": list(V19_SCENARIOS),
                "upper_95_limit": V19_UNSAFE_MARGIN,
            },
            "primary_success": {
                "minimum_gain": V19_SUCCESS_GAIN,
                "lower_95_above": 0.0,
                "scenarios": list(V19_PRIMARY_SCENARIOS),
            },
        },
        "estimator": {
            "process_noise": config.process_noise,
            "forgetting": config.forgetting,
            "max_lag": config.max_lag,
            "lag_min_updates": config.lag_min_updates,
            "selection_memory": config.selection_memory,
            "payload_prior": config.payload_prior,
            "filter_gate": config.filter_gate,
            "prior_std": list(config.prior_std),
            "lower": list(config.lower),
            "upper": list(config.upper),
            "compensate_delay": V19_COMPENSATE_DELAY,
        },
    }


def v19_cells():
    """Every (arm, scenario, seed) in the fixed order the runner executes."""
    for scenario in V19_SCENARIOS:
        for offset in range(V19_EPISODES_PER_CELL):
            for arm in V19_ARMS:
                yield arm, scenario, V19_SEED_START + offset


def paired_difference(rows_treatment, rows_reference, metric, rng):
    """Mean difference and percentile interval over seed-paired bootstrap draws."""
    treatment = {r.seed: float(metric(r)) for r in rows_treatment}
    reference = {r.seed: float(metric(r)) for r in rows_reference}
    if treatment.keys() != reference.keys() or not treatment:
        raise ValueError("paired contrast needs the same non-empty seed set")
    seeds = sorted(treatment)
    differences = np.asarray([treatment[s] - reference[s] for s in seeds])
    draws = rng.integers(0, len(seeds), size=(V19_BOOTSTRAP_DRAWS, len(seeds)))
    distribution = differences[draws].mean(axis=1)
    return {
        "difference": float(differences.mean()),
        "ci95_low": float(np.quantile(distribution, 0.025)),
        "ci95_high": float(np.quantile(distribution, 0.975)),
        "treatment_rate": float(np.mean(list(treatment.values()))),
        "reference_rate": float(np.mean(list(reference.values()))),
        "pairs": len(seeds),
    }


def cell_summary(rows: list[PilotEpisode]) -> dict:
    lags = [r for r in rows if r.selected_lag is not None]
    return {
        "episodes": len(rows),
        "success_rate": float(np.mean([r.success for r in rows])),
        "unsafe_episode_rate": float(np.mean([r.unsafe_episode for r in rows])),
        "abort_rate": float(np.mean([r.outcome == "abort" for r in rows])),
        "timeout_rate": float(np.mean([r.outcome == "timeout" for r in rows])),
        "median_final_distance_m": float(np.median([r.final_distance for r in rows])),
        "normalized_violation_max": float(max(r.normalized_violation_max for r in rows)),
        "intervention_fraction_mean": float(
            np.mean([r.safety_intervention_fraction for r in rows])
        ),
        "lag_identified_fraction": (
            float(np.mean([r.selected_lag == r.true_delay for r in lags])) if lags else None
        ),
    }


def analyze(episodes: list[PilotEpisode]) -> dict:
    """Apply the frozen rule. Vetoes are checked before the primary endpoint."""
    by_cell: dict[tuple[str, str], list[PilotEpisode]] = {}
    for episode in episodes:
        by_cell.setdefault((episode.arm, episode.scenario), []).append(episode)
    expected = {(arm, scenario) for arm in V19_ARMS for scenario in V19_SCENARIOS}
    if set(by_cell) != expected:
        raise ValueError("campaign is incomplete: missing cells")
    for key, rows in by_cell.items():
        if len(rows) != V19_EPISODES_PER_CELL or len({r.seed for r in rows}) != len(rows):
            raise ValueError(f"cell {key} does not hold {V19_EPISODES_PER_CELL} unique seeds")

    rng = np.random.default_rng(V19_BOOTSTRAP_SEED)
    summary = {f"{arm}/{scenario}": cell_summary(rows) for (arm, scenario), rows in by_cell.items()}
    contrasts = {}
    for scenario in V19_SCENARIOS:
        treatment = by_cell[("adaptive_hocbf", scenario)]
        reference = by_cell[("fixed_hocbf", scenario)]
        contrasts[scenario] = {
            "success": paired_difference(treatment, reference, lambda r: r.success, rng),
            "unsafe_episode": paired_difference(
                treatment, reference, lambda r: r.unsafe_episode, rng
            ),
            "abort": paired_difference(
                treatment, reference, lambda r: r.outcome == "abort", rng
            ),
        }

    vetoes = [
        scenario
        for scenario in V19_SCENARIOS
        if contrasts[scenario]["unsafe_episode"]["ci95_high"] > V19_UNSAFE_MARGIN
    ]
    primary_met = all(
        contrasts[s]["success"]["difference"] >= V19_SUCCESS_GAIN
        and contrasts[s]["success"]["ci95_low"] > 0.0
        for s in V19_PRIMARY_SCENARIOS
    )
    if vetoes:
        decision = "no_go_safety"
    elif primary_met:
        decision = "go"
    else:
        decision = "inconclusive"
    return {
        "decision": decision,
        "safety_vetoes": vetoes,
        "primary_met": primary_met,
        "contrasts": contrasts,
        "summary": summary,
        "protocol": v19_protocol_dict(),
    }
