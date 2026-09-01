"""Frozen protocol for SARRL v1.3 planar OOD and fault robustness."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from sarrl.envs import DomainRandomization, FaultSpec

V13_EVALUATION_SEED = 50_000
V13_EPISODES = 100
V13_TRAINING_SEEDS = (0, 1, 2, 3, 4)
V13_CONDITIONS = ("A0", "A2", "A3", "A4", "A5", "A6")


@dataclass(frozen=True)
class RobustnessScenario:
    key: str
    label: str
    randomization: DomainRandomization
    fault: FaultSpec | None = None

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "randomization": asdict(self.randomization),
            "fault": asdict(self.fault) if self.fault is not None else None,
        }


def v13_scenarios() -> tuple[RobustnessScenario, ...]:
    """Return paired ID, compound-OOD and abrupt motor-fault scenarios."""
    in_distribution = DomainRandomization(
        mass_fraction=0.15,
        friction_fraction=0.30,
        motor_gain_fraction=0.15,
        payload_range=(0.0, 1.0),
        action_delay_max=2,
    )
    return (
        RobustnessScenario(
            "id_reference",
            "In-distribution reference",
            in_distribution,
        ),
        RobustnessScenario(
            "ood_compound",
            "Compound OOD dynamics",
            DomainRandomization(
                mass_fraction=0.30,
                friction_fraction=0.50,
                motor_gain_fraction=0.25,
                payload_range=(1.25, 1.75),
                action_delay_max=3,
            ),
        ),
        RobustnessScenario(
            "motor_fault",
            "Abrupt joint-2 motor loss",
            in_distribution,
            FaultSpec(
                start_step=20,
                motor_gain_multiplier=(1.0, 0.55),
            ),
        ),
    )


def v13_protocol_dict() -> dict:
    """Return the JSON-friendly frozen v1.3 evaluation protocol."""
    return {
        "release_target": "v1.3.0",
        "campaign": "planar_ood_fault_robustness",
        "training": "reuse_v1.2_artifacts_without_retraining",
        "training_seeds": list(V13_TRAINING_SEEDS),
        "evaluation": {
            "seed_start": V13_EVALUATION_SEED,
            "episodes_per_model_scenario": V13_EPISODES,
            "paired_across_scenarios": True,
            "reference_scenario": "id_reference",
        },
        "conditions": list(V13_CONDITIONS),
        "excluded_condition": {
            "key": "A1",
            "reason": "selected policy checkpoints were not retained",
        },
        "scenarios": [scenario.to_dict() for scenario in v13_scenarios()],
        "statistics": {
            "multi_seed_spread": "sample_sd_ddof_1",
            "episode_success_interval": "wilson_95",
            "paired_scenario_difference": "paired_bootstrap_95",
            "bootstrap_samples": 10_000,
        },
    }
