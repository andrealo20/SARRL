"""Saved-transition diagnostics for Panda delay hypotheses."""

from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from sarrl.controllers.panda_delay import PandaDelayController
from sarrl.controllers.panda_score_memory import PandaScoreMemoryConfig, PandaScoreMemoryController

BANK_ARMS = ("bank_align", "bank_exp", "bank_window")
PHASES = {"before": (0, 100), "moving_after": (100, 200), "settling": (200, 300)}


def information_diagnostics(design):
    """Report raw and column-normalized singular values; no calibrated uncertainty."""
    matrix = np.asarray(design, dtype=float).reshape(-1, 8)
    if not len(matrix) or not np.all(np.isfinite(matrix)):
        raise ValueError("design must contain finite rows")
    norms = np.linalg.norm(matrix, axis=0)
    normalized = matrix / np.where(norms > 0, norms, 1)
    singular = np.linalg.svd(matrix, compute_uv=False)
    scaled = np.linalg.svd(normalized, compute_uv=False)
    rank = int(np.linalg.matrix_rank(matrix))
    scaled_rank = int(np.linalg.matrix_rank(normalized))
    return {
        "rank": rank,
        "column_norms": norms.tolist(),
        "singular_values": singular.tolist(),
        "condition": float(singular[0] / singular[-1]) if rank == 8 else None,
        "normalized_rank": scaled_rank,
        "normalized_singular_values": scaled.tolist(),
        "normalized_condition": float(scaled[0] / scaled[-1]) if scaled_rank == 8 else None,
    }


def bounded_fit(design, residual, upper):
    """Retrospective within-phase fit, including future samples in that phase."""
    matrix = np.asarray(design, dtype=float).reshape(-1, 8)
    values = np.asarray(residual, dtype=float).reshape(-1)
    if len(matrix) != len(values) or not np.all(np.isfinite(values)):
        raise ValueError("residual must match the finite design rows")
    fit = lsq_linear(
        matrix, values, bounds=(np.zeros(8), np.asarray(upper)), method="bvls", tol=1e-12
    )
    if not fit.success:
        raise RuntimeError(f"bounded diagnostic fit failed: {fit.message}")
    error = values - matrix @ fit.x
    return {
        "parameters": fit.x.tolist(),
        "mean_squared_nm2": float(np.mean(error**2)),
        "active_bounds": fit.active_mask.tolist(),
    }


def suffix_selection(innovations, start):
    """Reset only score accumulation on the saved input history."""
    values = np.asarray(innovations, dtype=float)
    if values.ndim != 2 or not 0 <= start < len(values):
        raise ValueError("start must index a two-dimensional innovation history")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("innovations must be finite and nonnegative")
    return np.argmin(np.cumsum(values[start:], axis=0), axis=1)


def replay_bank(row, trace):
    """Replay causal estimators on saved inputs, without stepping a plant."""
    arm = row["arm"]
    if arm not in BANK_ARMS or len(trace) != 301:
        raise ValueError("diagnosis requires a bank arm with 301 saved states")
    if arm == "bank_align":
        bank = PandaDelayController(predict=False)
    else:
        rule = "exponential" if arm == "bank_exp" else "window"
        bank = PandaScoreMemoryController(PandaScoreMemoryConfig(rule, 50))
    bank.sent_history = [np.array(u, dtype=float) for u in row["initial_command_history"]]
    sent = np.array([t["torque_nm"] for t in trace[:-1]])
    history = np.concatenate([np.array(row["initial_command_history"]), sent])
    truth = np.array(
        [row["true_parameters"]["payload_mass"], *row["true_parameters"]["viscous_extra"]]
    )
    designs, physical_designs, residuals, actual_residuals = [], [], [], []
    innovations, true_innovations, parameters, scores, ages = [], [], [], [], []
    for index, (current, following) in enumerate(zip(trace[:-1], trace[1:], strict=True)):
        np.testing.assert_array_equal(current["estimated_parameters"], bank.estimator.parameters)
        if "delay_scores" in current:
            np.testing.assert_array_equal(current["delay_scores"], bank.scores)
        q, qd = np.array(current["measured_q"]), np.array(current["measured_qd"])
        nq, nv = np.array(following["measured_q"]), np.array(following["measured_qd"])
        nominal, design = bank.regressor.evaluate((q + nq) / 2, (qd + nv) / 2, (nv - qd) / 0.01)
        pq, pv = np.array(current["q"]), np.array(current["qd"])
        pnq, pnv = np.array(following["q"]), np.array(following["qd"])
        _, physical = bank.regressor.evaluate((pq + pnq) / 2, (pv + pnv) / 2, (pnv - pv) / 0.01)
        aligned = np.array([history[4 + index - lag] for lag in bank.lags])
        target = aligned - nominal
        theta = np.array([estimator.parameters.copy() for estimator in bank.estimators])
        errors = target - np.einsum("ji,ki->kj", design, theta)
        squared = np.sum(errors**2, axis=1)
        if "score_innovation_squared" in current:
            np.testing.assert_allclose(
                squared, current["score_innovation_squared"], rtol=1e-12, atol=1e-10
            )
        age = current["applied_age_steps"]
        actual = np.array(current["delivered_torque_nm"])
        np.testing.assert_array_equal(actual, history[4 + index - age])
        designs.append(design)
        physical_designs.append(physical)
        residuals.append(target)
        actual_residuals.append(actual - nominal)
        innovations.append(squared)
        true_innovations.append(np.sum((target - design @ truth) ** 2, axis=1))
        parameters.append(theta)
        ages.append(age)
        bank.observe_transition(q, qd, nq, nv, sent[index], 0.01)
        scores.append(bank.scores.copy())
    diagnostic = row["delay_estimator"]
    for field in (
        "scores",
        "candidate_estimates",
        "candidate_bound_hits",
        "selected_history",
        "selection_switches",
    ):
        np.testing.assert_array_equal(diagnostic[field], bank.delay_diagnostics()[field])
    np.testing.assert_array_equal(
        row["estimator"]["estimated_parameters"], bank.estimator.parameters
    )
    designs, physical_designs, residuals = map(np.array, (designs, physical_designs, residuals))
    innovations, true_innovations, parameters, scores, ages = map(
        np.array, (innovations, true_innovations, parameters, scores, ages)
    )
    phases = {}
    for name, (start, end) in PHASES.items():
        part = slice(start, end)
        batch = [
            bounded_fit(designs[part], residuals[part, lag], bank.estimators[lag].upper)
            for lag in bank.lags
        ]
        phases[name] = {
            "transitions": [start, end],
            "applied_ages": sorted(set(ages[part].tolist())),
            "measured_information": information_diagnostics(designs[part]),
            "physical_information": information_diagnostics(physical_designs[part]),
            "candidate_preupdate_mean_squared_nm2": np.mean(innovations[part] / 7, axis=0).tolist(),
            "true_parameter_mean_squared_nm2": np.mean(true_innovations[part] / 7, axis=0).tolist(),
            "bounded_phase_fits": batch,
            "known_age_phase_fit": bounded_fit(
                designs[part], np.array(actual_residuals)[part], bank.estimators[0].upper
            ),
            "candidate_parameters_at_start": parameters[start].tolist(),
        }
    reset = suffix_selection(innovations, 100)
    selected = np.array(bank.selected_history)
    final_age = int(ages[-1])
    return {
        "scenario": row["scenario"],
        "seed": row["seed"],
        "arm": arm,
        "true_parameters": truth.tolist(),
        "lags": list(bank.lags),
        "final_applied_age": final_age,
        "final_selected_age": int(selected[-1]),
        "correct_fraction_after_switch": float(np.mean(selected[100:] == ages[100:])),
        "cumulative_suffix_final_age": int(reset[-1]),
        "cumulative_suffix_correct_fraction": float(np.mean(reset == ages[100:])),
        "candidate_scores_at_switch": scores[99].tolist(),
        "candidate_scores_final": scores[-1].tolist(),
        "candidate_parameters_final": [e.parameters.tolist() for e in bank.estimators],
        "phases": phases,
    }
