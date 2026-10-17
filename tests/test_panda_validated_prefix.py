"""Rational bounds, continuous parameters, dense trajectories and witness tampering."""

from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from sarrl.safety.panda_center_arithmetic import ArrayEnclosure
from sarrl.safety.panda_prefix_audit import audit_prefix
from sarrl.safety.panda_state_region import EnclosureUnavailable, VectorBox
from sarrl.safety.panda_structured_backup import PandaParameterBox, PandaStructuredContext
from sarrl.safety.panda_structured_prefix import PandaPrefixConfig, PandaPrefixErrors
from sarrl.safety.panda_validated_dynamics import (
    PandaValidatedDynamics,
    arithmetic,
    balls,
    box,
    inflate,
    mass_cells,
    solve_acceleration,
    step_images,
)
from sarrl.safety.panda_validated_prefix import PandaValidatedPrefix


def enclosure(values):
    return ArrayEnclosure(values, values)


def parameters():
    return PandaParameterBox(np.array([0.5, *[0.5] * 7]), np.array([2.5, *[3.0] * 7]))


def errors():
    return PandaPrefixErrors(*[np.zeros(7) for _ in range(5)])


def coefficients(m0=1, mp=0.25):
    return SimpleNamespace(
        nominal=SimpleNamespace(mass=enclosure(np.eye(7) * m0), bias=enclosure(np.zeros(7))),
        payload=SimpleNamespace(mass=enclosure(np.eye(7) * mp), bias=enclosure(np.ones(7) * 0.5)),
    )


@pytest.mark.parametrize("count", [1, 3, 16, 256])
def test_mass_partitions_cover_exact_continuous_range(count):
    cells = mass_cells(parameters(), count)
    assert cells[0][0] == Fraction(1, 2) and cells[-1][1] == Fraction(5, 2)
    assert all(a[1] == b[0] for a, b in zip(cells[:-1], cells[1:], strict=True))


@pytest.mark.parametrize("count", [0, 257, True, 1.5])
def test_invalid_partition_count(count):
    with pytest.raises(ValueError):
        mass_cells(parameters(), count)


@pytest.mark.parametrize("bits", [79, 1025, True, 128.0])
def test_invalid_precision(bits):
    with pytest.raises(ValueError):
        with arithmetic(bits):
            pass


def test_inflation_and_time_images_cover_exact_rational_subnormal_operations():
    pytest.importorskip("flint")
    tiny = np.nextafter(0.0, np.inf)
    q = VectorBox.point(np.full(7, 3 * tiny))
    inflated = inflate(q, np.full(7, tiny))
    assert np.all(inflated.lower <= 2 * tiny) and np.all(inflated.upper >= 4 * tiny)
    v, a = VectorBox.point(np.full(7, 0.1)), VectorBox.point(np.full(7, -0.3))
    qi, vi, qe, ve = step_images(q, v, a, 0.01, errors())
    t, x, y, z = map(Fraction, [0.01, 3 * tiny, 0.1, -0.3])
    for k in range(101):
        time = t * k / 100
        qvalue, vvalue = x + time * y + time * time / 2 * z, y + time * z
        assert Fraction(float(qi.lower[0])) <= qvalue <= Fraction(float(qi.upper[0]))
        assert Fraction(float(vi.lower[0])) <= vvalue <= Fraction(float(vi.upper[0]))
    assert Fraction(float(qe.lower[0])) <= x + t * y + t * t / 2 * z <= Fraction(float(qe.upper[0]))
    assert Fraction(float(ve.lower[0])) <= y + t * z <= Fraction(float(ve.upper[0]))


def test_validated_shared_mass_solve_covers_rational_solutions_and_refines_cells():
    pytest.importorskip("flint")
    qv = VectorBox.point(np.ones(7))
    torque = VectorBox.point(np.full(7, 2.0))
    wide = solve_acceleration(coefficients(), parameters(), qv, torque, cells=1)
    refined = solve_acceleration(coefficients(), parameters(), qv, torque, cells=16)
    for im in range(33):
        mass = Fraction(1, 2) + Fraction(im, 16)
        for friction in (Fraction(1, 2), Fraction(7, 4), Fraction(3)):
            expected = (2 - mass / 2 - friction) / (1 + mass / 4)
            assert (
                Fraction(float(refined.lower[0])) <= expected <= Fraction(float(refined.upper[0]))
            )
    assert np.all(refined.upper - refined.lower < wide.upper - wide.lower)
    with pytest.raises(ValueError):
        refined.lower[0] = 0


def test_unverified_matrix_and_nonfinite_remainder_fail_closed_and_restore_precision():
    flint = pytest.importorskip("flint")
    old = flint.ctx.prec
    with pytest.raises(EnclosureUnavailable, match="validated_mass_solve_unavailable"):
        solve_acceleration(
            coefficients(0, 0),
            parameters(),
            VectorBox.point(np.zeros(7)),
            VectorBox.point(np.zeros(7)),
            precision_bits=256,
        )
    with pytest.raises(ValueError, match="remainder"):
        solve_acceleration(
            coefficients(),
            parameters(),
            VectorBox.point(np.zeros(7)),
            VectorBox.point(np.zeros(7)),
            bias_remainder=np.full(7, np.inf),
        )
    assert flint.ctx.prec == old


class Manufactured(PandaValidatedDynamics):
    """Fixed coupled mass, analytic nonlinear bias; no Panda model constants."""

    def coefficients(self, q, v):
        if not self.q_domain.contains(q) or not self.v_domain.contains(v):
            raise EnclosureUnavailable("state_region_not_covered")
        with arithmetic(self.precision_bits) as backend:
            qb, vb = balls(q, backend), balls(v, backend)
            h = box(qb / 5 + vb * vb / 50)
        direction = np.linspace(-0.1, 0.2, 7)
        return SimpleNamespace(
            nominal=SimpleNamespace(
                mass=enclosure(np.eye(7) + np.outer(direction, direction)),
                bias=ArrayEnclosure(h.lower, h.upper),
            ),
            payload=SimpleNamespace(mass=enclosure(np.eye(7) / 4), bias=enclosure(np.full(7, 0.5))),
        )


@pytest.fixture(scope="module")
def fixture():
    pytest.importorskip("flint")
    domain = VectorBox(-np.ones(7), np.ones(7))
    dynamics = Manufactured(
        domain,
        domain,
        mass_remainder=np.zeros((7, 7)),
        bias_remainder=np.zeros(7),
        assumption="Manufactured ideal model",
    )
    noise = PandaPrefixErrors(
        np.full(7, 1e-6), np.full(7, 2e-6), np.full(7, 3e-7), np.full(7, 4e-7), np.full(7, 2e-6)
    )
    checker = PandaValidatedPrefix(
        dynamics,
        parameters(),
        domain.lower,
        domain.upper,
        np.ones(7),
        np.full(7, 20),
        noise,
        PandaPrefixConfig(),
    )
    context = PandaStructuredContext(np.zeros(7), np.array([1.5, *[1.75] * 7]))
    q, v = np.linspace(-0.01, 0.01, 7), np.linspace(0.02, -0.01, 7)
    candidate, history = np.full(7, 0.75), [np.full(7, value) for value in (0.2, 0.4, 0.6)]
    result = checker.verify_prefix(context, q, v, candidate, history, intervals=3)
    assert result.prefix_contained and not result.terminal_verified
    return checker, context, q, v, candidate, history, result


@pytest.mark.parametrize("mass,friction", [(0.5, 0.5), (1.5, 1.75), (2.5, 3)])
def test_full_prefix_contains_independent_dense_trajectories_for_every_lag(fixture, mass, friction):
    from scipy.integrate import solve_ivp

    checker, context, q0, v0, candidate, history, result = fixture
    direction = np.linspace(-0.1, 0.2, 7)
    m0 = np.eye(7) + np.outer(direction, direction)
    matrix = m0 + mass * np.eye(7) / 4
    for case in result.cases:
        q, v = q0 + checker.errors.position_measurement, v0 - checker.errors.velocity_measurement
        queue = [np.zeros(7), *history]
        for index, record in enumerate(case.intervals):
            if index == 0:
                sent = candidate
            else:
                qm, vm = (
                    q - checker.errors.position_measurement,
                    v + checker.errors.velocity_measurement,
                )
                desired = -context.kp * qm - context.kd * vm
                sent = (m0 + 1.5 * np.eye(7) / 4) @ desired + qm / 5 + vm**2 / 50
                sent += 0.75 + 1.75 * vm + checker.errors.command_error
            delivered = queue[-case.lag] if case.lag else sent

            def rhs(t, state, delivered=delivered):
                q, v = state[:7], state[7:]
                acceleration = np.linalg.solve(
                    matrix, delivered - q / 5 - v**2 / 50 - mass / 2 - friction * v
                )
                return np.r_[v, acceleration]

            trajectory = solve_ivp(
                rhs, [0, 0.01], np.r_[q, v], rtol=1e-11, atol=1e-13, t_eval=np.linspace(0, 0.01, 21)
            ).y
            assert np.all(trajectory[:7] >= record.q_tube.lower[:, None])
            assert np.all(trajectory[:7] <= record.q_tube.upper[:, None])
            assert np.all(trajectory[7:] >= record.v_tube.lower[:, None])
            assert np.all(trajectory[7:] <= record.v_tube.upper[:, None])
            q = trajectory[:7, -1] + checker.errors.position_defect
            v = trajectory[7:, -1] - checker.errors.velocity_defect
            assert record.q_end.contains(VectorBox.point(q))
            assert record.v_end.contains(VectorBox.point(v))
            queue = [*queue[1:], sent]


def test_auditor_checks_retained_prefix_without_calling_search(fixture, monkeypatch):
    checker, context, q, v, candidate, history, result = fixture
    monkeypatch.setattr(checker, "verify_prefix", lambda *a, **kw: pytest.fail("search in audit"))
    report = audit_prefix(checker, context, q, v, candidate, history, result, intervals=3)
    assert report["checked_intervals"] == 15 and report["prefix_contained"]
    assert not report["terminal_verified"] and not report["physical_binding_verified"]


@pytest.mark.parametrize("kind", ["lag", "queue", "acceleration", "endpoint", "flag", "candidate"])
def test_auditor_rejects_incomplete_or_modified_records(fixture, kind):
    checker, context, q, v, candidate, history, result = fixture
    case, record = result.cases[-1], result.cases[-1].intervals[0]
    if kind == "lag":
        altered = replace(result, cases=result.cases[:-1])
    else:
        if kind == "queue":
            record = replace(record, delivered=VectorBox.point(np.ones(7)))
        elif kind == "acceleration":
            record = replace(record, acceleration=VectorBox.point(np.zeros(7)))
        elif kind == "endpoint":
            record = replace(record, q_end=VectorBox.point(np.zeros(7)))
        elif kind == "candidate":
            record = replace(record, sent=VectorBox.point(np.zeros(7)))
        case = replace(
            case,
            intervals=(record, *case.intervals[1:]),
            prefix_contained=False if kind == "flag" else case.prefix_contained,
        )
        altered = replace(result, cases=(*result.cases[:-1], case))
    with pytest.raises(ValueError, match="prefix audit"):
        audit_prefix(checker, context, q, v, candidate, history, altered, intervals=3)


def test_actual_panda_region_solve_covers_independent_point_linear_systems():
    pytest.importorskip("flint")
    pytest.importorskip("mujoco")
    from sarrl.robots.panda import HOME
    from sarrl.safety.panda_center_arithmetic import enclose_panda_center
    from sarrl.safety.panda_regional_arithmetic import enclose_panda_region

    q, v = VectorBox(HOME - 0.001, HOME + 0.001), VectorBox(-np.ones(7) * 0.01, np.ones(7) * 0.01)
    region = enclose_panda_region(q, v)
    torque = VectorBox.point(np.array([0, -12, 0, 20, 0, 1, 0]))
    acceleration = solve_acceleration(region, parameters(), v, torque)
    for state_q, state_v in ((q.lower, v.upper), (q.upper, v.lower)):
        center = enclose_panda_center(state_q, state_v, precision_bits=256)
        with arithmetic(256) as backend:
            m0, mp = balls(center.nominal.mass, backend), balls(center.payload.mass, backend)
            h0, hp = balls(center.nominal.bias, backend), balls(center.payload.bias, backend)
            for mass, friction in ((0.5, 0.5), (1.5, 1.75), (2.5, 3)):
                rhs = (
                    balls(torque, backend)
                    - h0
                    - mass * hp
                    - friction * balls(VectorBox.point(state_v), backend)
                )
                answer = backend.arb_mat((m0 + mass * mp).tolist()).solve(
                    backend.arb_mat([[x] for x in rhs]), algorithm="lu"
                )
                inner = box(np.array([answer[i, 0] for i in range(7)], dtype=object))
                assert acceleration.contains(inner)


def test_velocity_box_must_match_the_coefficient_domain():
    data = coefficients()
    data.v_domain = VectorBox.point(np.zeros(7))
    with pytest.raises(EnclosureUnavailable, match="coefficient_velocity_domain_not_covered"):
        solve_acceleration(
            data, parameters(), VectorBox.point(np.ones(7)), VectorBox.point(np.zeros(7))
        )


def test_screen_cli_preserves_existing_report_and_rejects_source_mismatch(tmp_path, monkeypatch):
    import json
    import sys

    from tools import assess_panda_validated_prefix as tool

    path = tmp_path / "assessment.json"
    path.write_text(json.dumps({"source_sha256": {"different": "source"}}), encoding="utf-8")
    monkeypatch.setattr(tool, "assessment", lambda: pytest.fail("unexpected numerical run"))
    monkeypatch.setattr(sys, "argv", ["screen", "--output", str(path)])
    with pytest.raises(FileExistsError):
        tool.main()
    monkeypatch.setattr(sys, "argv", ["screen", "--audit", "--output", str(path)])
    with pytest.raises(ValueError, match="fingerprints differ"):
        tool.main()


def test_no_prefix_can_clip_an_escaping_image_or_accept_an_outside_initial_state(fixture):
    checker, context, q, v, candidate, history, _ = fixture
    for position, velocity, command in (
        (np.full(7, 1.1), v, candidate),
        (np.full(7, 0.99999), np.full(7, 0.1), np.full(7, 20)),
    ):
        result = checker.verify_prefix(context, position, velocity, command, history, intervals=1)
        assert not result.prefix_contained
        assert all(not case.prefix_contained for case in result.cases)
        assert all(case.reason == "whole_interval_bounds_not_established" for case in result.cases)


def test_raw_feedback_torque_cannot_be_clipped_to_pass(fixture):
    checker, _, q, v, candidate, history, _ = fixture
    context = PandaStructuredContext(np.ones(7) * 100, np.ones(8))
    result = checker.verify_prefix(context, q, v, candidate, history, intervals=2)
    assert not result.prefix_contained
    assert all(case.reason == "raw_feedback_torque_bounds_not_established" for case in result.cases)
