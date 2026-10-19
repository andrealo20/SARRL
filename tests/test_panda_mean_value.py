"""Shared-coordinate arithmetic, held-flow truncation and Panda differential checks."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from sarrl.safety.panda_affine_state import AffineState, concatenate, linear_combination, mean_value
from sarrl.safety.panda_center_arithmetic import ArrayEnclosure, _export
from sarrl.safety.panda_mean_value_prefix import physical_coordinates, taylor_endpoint
from sarrl.safety.panda_state_region import VectorBox
from sarrl.safety.panda_structured_backup import PandaParameterBox, PandaStructuredContext
from sarrl.safety.panda_validated_dynamics import arithmetic, balls, exact, interval, point


def coordinates():
    return physical_coordinates(
        PandaParameterBox(np.array([0.5, *[0.5] * 7]), np.array([0.625, *[3.0] * 7]))
    )


@pytest.mark.parametrize("kind", ["shape", "nonfinite", "residual"])
def test_invalid_affine_state_is_rejected_without_optional_backends(kind):
    c, g = np.zeros(7), np.zeros((7, 8))
    residual = ArrayEnclosure(np.zeros(7), np.zeros(7))
    if kind == "shape":
        g = g[:, :7]
    elif kind == "nonfinite":
        c[0] = np.inf
    else:
        residual = ArrayEnclosure(np.ones(7), np.ones(7))
    with pytest.raises(ValueError, match="affine coordinates"):
        AffineState(c, g, residual)


def test_continuous_parameters_keep_eight_distinct_immutable_coordinates():
    parameters = coordinates()
    assert parameters.generators.shape == (8, 8)
    np.testing.assert_array_equal(np.diag(parameters.generators), [0.0625, *[1.25] * 7])
    with pytest.raises(ValueError):
        parameters.generators[0, 0] = 1


def fiber(form, epsilon, backend):
    return _export(
        point(form.center, backend)
        + point(form.generators, backend) @ point(epsilon, backend)
        + balls(form.residual, backend)
    )


def test_shared_coordinates_cancel_when_independent_hulls_cannot():
    pytest.importorskip("flint")
    parameters = coordinates()
    form = parameters.take(slice(0, 7))
    with arithmetic(128) as backend:
        difference = linear_combination(((1, form), (-1, form)), backend)
        assert np.max(difference.hull(backend).upper - difference.hull(backend).lower) == 0
        assert np.max(form.hull(backend).upper - form.hull(backend).lower) > 2


def test_mean_value_quadratic_covers_each_shared_parameter_fiber():
    pytest.importorskip("flint")
    form = coordinates()
    with arithmetic(128) as backend:
        x = form.hull(backend)
        derivative = point(np.eye(8), backend) * (2 * balls(x, backend))
        enclosed = mean_value(form, point(form.center, backend) ** 2, derivative, backend)
        for fraction in (Fraction(-1), Fraction(-1, 3), Fraction(0), Fraction(1, 7), Fraction(1)):
            eps = np.full(8, float(fraction))
            result = fiber(enclosed, eps, backend)
            for i in range(8):
                actual = (
                    Fraction(float(form.center[i]))
                    + Fraction(float(form.generators[i, i])) * Fraction(float(eps[i]))
                ) ** 2
                assert (
                    Fraction(float(result.lower[i])) <= actual <= Fraction(float(result.upper[i]))
                )


def test_remainder_transfer_covers_rounding_at_nonbinary_period():
    pytest.importorskip("flint")
    form = AffineState.constant(np.full(7, 0.1))
    with arithmetic(128) as backend:
        output = linear_combination(((Fraction(1, 3), form),), backend)
        result = output.hull(backend)
        actual = Fraction(0.1) / 3
        assert Fraction(float(result.lower[0])) <= actual <= Fraction(float(result.upper[0]))


def test_second_order_endpoint_has_exact_polynomial_integral_remainder():
    pytest.importorskip("flint")
    with arithmetic(128) as backend:
        q, v, a = [AffineState.constant(np.full(7, x)) for x in (0.25, -0.125, 0.75)]
        t = Fraction(0.01)
        qnext, vnext = taylor_endpoint(q, v, a, point(np.full(7, 2), backend), 0.01, backend)
        for jerk in (-2, -1, 0, 1, 2):
            exact_q = (
                Fraction(0.25) + t * Fraction(-0.125) + t * t / 2 * Fraction(0.75) + t**3 / 6 * jerk
            )
            exact_v = Fraction(-0.125) + t * Fraction(0.75) + t * t / 2 * jerk
            for form, value in ((qnext, exact_q), (vnext, exact_v)):
                hull = form.hull(backend)
                assert Fraction(float(hull.lower[0])) <= value <= Fraction(float(hull.upper[0]))


@pytest.fixture(scope="module")
def panda():
    pytest.importorskip("flint")
    pytest.importorskip("mujoco")
    from sarrl.robots.panda import HOME
    from sarrl.safety.panda_mean_value_dynamics import PandaMeanValueDynamics

    return PandaMeanValueDynamics(
        VectorBox(HOME - 0.5, HOME + 0.5),
        VectorBox(-np.ones(7), np.ones(7)),
        mass_remainder=np.zeros((7, 7)),
        bias_remainder=np.zeros(7),
        assumption="Normalized ideal model",
        cells=1,
    ), HOME


def test_differential_gravity_and_mass_agree_with_existing_point_coefficients(panda):
    dynamics, home = panda
    from sarrl.safety.panda_kinematic_coefficients import packaged_kinematic_coefficients

    nominal, payload, _ = packaged_kinematic_coefficients(home)
    with arithmetic(128) as backend:
        _, mass, mq, hq, _ = dynamics.derivatives(
            VectorBox.point(home), VectorBox.point(np.zeros(7)), exact(1.5, backend), backend
        )
        for enclosed, reference in (
            (mass, nominal.mass + 1.5 * payload.mass),
            (mq, nominal.mass_q + 1.5 * payload.mass_q),
            (hq, nominal.gravity_q + 1.5 * payload.gravity_q),
        ):
            assert np.max(_export(enclosed).absolute_error(reference)) < 1e-10


def test_affine_panda_acceleration_contains_independent_parameter_corner_values(panda):
    dynamics, home = panda
    from sarrl.safety.panda_kinematic_coefficients import packaged_kinematic_coefficients

    nominal, payload, damping = packaged_kinematic_coefficients(home)
    velocity = np.linspace(-0.01, 0.02, 7)
    torque = nominal.gravity + 1.5 * payload.gravity
    with arithmetic(128) as backend:
        form = dynamics.acceleration_affine(
            coordinates(),
            AffineState.constant(home),
            AffineState.constant(velocity),
            AffineState.constant(torque),
            backend,
        )
        for pattern in (0, 37, 85, 170, 255):
            eps = np.array([1 if pattern & (1 << j) else -1 for j in range(8)])
            parameters = coordinates().center + coordinates().generators @ eps
            mass, b = parameters[0], parameters[1:]
            actual = np.linalg.solve(
                nominal.mass + mass * payload.mass,
                torque
                - nominal.bias(velocity, damping)
                - mass * payload.bias(velocity, np.zeros(7))
                - b * velocity,
            )
            enclosed = fiber(form, eps, backend)
            assert np.all(actual >= enclosed.lower - 1e-10) and np.all(
                actual <= enclosed.upper + 1e-10
            )


def test_zero_velocity_feedback_preserves_the_existing_point_law(panda):
    dynamics, home = panda
    context = PandaStructuredContext(home, np.array([1.5, *[0.5] * 7]))
    with arithmetic(128) as backend:
        value = dynamics.feedback_affine(
            context, AffineState.constant(home), AffineState.constant(np.zeros(7)), backend
        )
        old = dynamics.torque(
            context, VectorBox.point(home), VectorBox.point(np.zeros(7)), np.zeros(7)
        )
        assert np.max(value.hull(backend).absolute_error(old.center)) < 1e-10


def test_concatenation_preserves_primary_coordinate_identity():
    first = coordinates()
    second = first.take(slice(0, 7))
    combined = concatenate((first, second))
    np.testing.assert_array_equal(combined.generators[:8], first.generators)
    np.testing.assert_array_equal(combined.generators[8:], second.generators)


@pytest.fixture(scope="module")
def manufactured():
    pytest.importorskip("flint")
    from sarrl.safety.panda_mean_value_prefix import PandaMeanValuePrefix
    from sarrl.safety.panda_structured_prefix import PandaPrefixConfig, PandaPrefixErrors
    from sarrl.safety.panda_validated_dynamics import PandaValidatedDynamics, box

    class HeldMassForcing(PandaValidatedDynamics):
        def acceleration(self, parameters, q, v, torque):
            with arithmetic(128) as backend:
                return box(
                    balls(torque, backend)
                    + interval(parameters.lower[0], parameters.upper[0], backend)
                )

        def acceleration_affine(self, parameters, q, v, torque, backend):
            mass = AffineState(
                np.repeat(parameters.center[:1], 7),
                np.repeat(parameters.generators[:1], 7, axis=0),
                ArrayEnclosure(
                    np.repeat(parameters.residual.lower[:1], 7),
                    np.repeat(parameters.residual.upper[:1], 7),
                ),
            )
            return linear_combination(((1, torque), (1, mass)), backend)

        def feedback_affine(self, context, q, v, backend):
            return linear_combination(((-1, q), (-1, v)), backend)

        def jerk(self, parameters, q, v, torque, acceleration, backend):
            return point(np.zeros(7), backend)

    domain = VectorBox(-np.ones(7), np.ones(7))
    dynamics = HeldMassForcing(
        domain,
        domain,
        mass_remainder=np.zeros((7, 7)),
        bias_remainder=np.zeros(7),
        assumption="Manufactured constant held acceleration",
        cells=1,
    )
    physical = PandaParameterBox(np.array([0.5, *[0.5] * 7]), np.array([0.625, *[3.0] * 7]))
    checker = PandaMeanValuePrefix(
        dynamics,
        physical,
        domain.lower,
        domain.upper,
        domain.upper,
        np.full(7, 20),
        PandaPrefixErrors(*[np.zeros(7) for _ in range(5)]),
        PandaPrefixConfig(),
    )
    context = PandaStructuredContext(np.zeros(7), np.array([0.5625, *[0.5] * 7]), kp=1, kd=1)
    history = [np.zeros(7) for _ in range(4)]
    result = checker.verify_prefix(
        context, np.zeros(7), np.zeros(7), np.zeros(7), history, intervals=6
    )
    return checker, context, history, result


def test_manufactured_held_flow_covers_exact_trajectories_and_queue_fibers(manufactured):
    checker, _, _, result = manufactured
    assert result.prefix_contained and not result.terminal_verified
    t = Fraction(checker.config.control_dt)
    with arithmetic(128) as backend:
        for case in result.cases:
            for epsilon in (-1, -0.25, 0, 0.5, 1):
                mass = Fraction(0.5625) + Fraction(0.0625) * Fraction(epsilon)
                q, v, queue = Fraction(0), Fraction(0), [Fraction(0)] * 4
                eps = np.array([epsilon, *[0] * 7])
                for step, record in enumerate(case.intervals):
                    sent = Fraction(0) if step == 0 else -q - v
                    delivered = queue[-case.lag] if case.lag else sent
                    for form, actual in ((record.sent, sent), (record.delivered, delivered)):
                        enclosed = fiber(form, eps, backend)
                        assert (
                            Fraction(float(enclosed.lower[0]))
                            <= actual
                            <= Fraction(float(enclosed.upper[0]))
                        )
                    acceleration = delivered + mass
                    for k in range(11):
                        partial = t * k / 10
                        for tube, actual in (
                            (record.q_tube, q + partial * v + partial**2 / 2 * acceleration),
                            (record.v_tube, v + partial * acceleration),
                        ):
                            assert (
                                Fraction(float(tube.lower[0]))
                                <= actual
                                <= Fraction(float(tube.upper[0]))
                            )
                    q, v = q + t * v + t * t / 2 * acceleration, v + t * acceleration
                    for form, actual in ((record.q_end, q), (record.v_end, v)):
                        enclosed = fiber(form, eps, backend)
                        assert (
                            Fraction(float(enclosed.lower[0]))
                            <= actual
                            <= Fraction(float(enclosed.upper[0]))
                        )
                    queue = [*queue[1:], sent]
            assert np.any(case.intervals[-1].sent.generators != 0)


def test_audit_uses_exact_rational_endpoint_checks_without_calling_search(
    manufactured, monkeypatch
):
    from sarrl.safety.panda_mean_value_audit import audit_mean_value_prefix
    from sarrl.safety.panda_mean_value_prefix import PandaMeanValuePrefix

    checker, context, history, result = manufactured
    monkeypatch.setattr(
        PandaMeanValuePrefix, "verify_prefix", lambda *a, **kw: pytest.fail("search called")
    )
    checked = audit_mean_value_prefix(
        checker, context, np.zeros(7), np.zeros(7), np.zeros(7), history, result, intervals=6
    )
    assert checked["checked_intervals"] == 30


@pytest.mark.parametrize(
    "kind", ["generator", "queue", "endpoint", "jerk", "parameter", "lag", "terminal"]
)
def test_audit_rejects_changed_shared_coordinates_and_false_flags(manufactured, kind):
    from sarrl.safety.panda_mean_value_audit import audit_mean_value_prefix

    checker, context, history, result = manufactured
    case = result.cases[0]
    record = case.intervals[0]
    if kind in ("generator", "endpoint"):
        form = record.q_end
        value = form.generators.copy() if kind == "generator" else form.center.copy()
        value.flat[0] += 0.01
        altered = replace(form, **{"generators" if kind == "generator" else "center": value})
        record = replace(record, q_end=altered)
    elif kind == "queue":
        record = replace(record, delivered=AffineState.constant(np.ones(7)))
    elif kind == "jerk":
        record = replace(record, jerk_bound=ArrayEnclosure(-np.ones(7), -np.ones(7)))
    elif kind == "parameter":
        result = replace(result, parameters=AffineState.constant(np.ones(8)))
    elif kind == "lag":
        result = replace(result, cases=result.cases[1:])
    else:
        result = replace(result, terminal_verified=True)
    if kind in ("generator", "endpoint", "queue", "jerk"):
        case = replace(case, intervals=(record, *case.intervals[1:]))
        result = replace(result, cases=(case, *result.cases[1:]))
    with pytest.raises(ValueError, match="prefix audit"):
        audit_mean_value_prefix(
            checker, context, np.zeros(7), np.zeros(7), np.zeros(7), history, result, intervals=6
        )


@pytest.mark.parametrize("kind", ["physical", "runtime", "history"])
def test_unsupported_derivative_contracts_fail_closed(panda, kind):
    from sarrl.safety.panda_mean_value_dynamics import PandaMeanValueDynamics
    from sarrl.safety.panda_mean_value_prefix import PandaMeanValuePrefix
    from sarrl.safety.panda_structured_prefix import PandaPrefixConfig, PandaPrefixErrors

    dynamics, _ = panda
    domain = dynamics.v_domain
    with pytest.raises(ValueError):
        if kind == "physical":
            PandaMeanValueDynamics(
                dynamics.q_domain,
                domain,
                mass_remainder=np.ones((7, 7)),
                bias_remainder=np.zeros(7),
                assumption="Unsupported derivatives",
            )
        else:
            errors = [np.zeros(7) for _ in range(5)]
            if kind == "runtime":
                errors[2] = np.ones(7) * 1e-6
            checker = PandaMeanValuePrefix(
                dynamics,
                PandaParameterBox(np.zeros(8), np.ones(8)),
                dynamics.q_domain.lower,
                dynamics.q_domain.upper,
                domain.upper,
                np.full(7, 87),
                PandaPrefixErrors(*errors),
                PandaPrefixConfig(),
            )
            checker.verify_prefix(None, np.zeros(7), np.zeros(7), np.zeros(7), [], intervals=6)


def test_nonzero_velocity_jacobian_covers_independent_central_differences(panda):
    dynamics, home = panda
    from sarrl.safety.panda_kinematic_coefficients import packaged_kinematic_coefficients

    velocity = np.linspace(-0.015, 0.02, 7)
    nominal, payload, _ = packaged_kinematic_coefficients(home)
    torque = nominal.gravity + 1.5 * payload.gravity
    actual = np.array([0.5625, *[1.75] * 7])
    with arithmetic(128) as backend:
        jacobian = _export(
            dynamics.acceleration_jacobian(
                PandaParameterBox(actual - 0.0001, actual + 0.0001),
                VectorBox(home - 1e-5, home + 1e-5),
                VectorBox(velocity - 1e-5, velocity + 1e-5),
                VectorBox(torque - 1e-5, torque + 1e-5),
                backend,
            )
        )

    def acceleration(z):
        q, v, parameters, u = z[:7], z[7:14], z[14:22], z[22:]
        nominal, payload, damping = packaged_kinematic_coefficients(q)
        return np.linalg.solve(
            nominal.mass + parameters[0] * payload.mass,
            u
            - nominal.bias(v, damping)
            - parameters[0] * payload.bias(v, np.zeros(7))
            - parameters[1:] * v,
        )

    z = np.concatenate((home, velocity, actual, torque))
    for j in range(29):
        shift = np.zeros(29)
        shift[j] = 1e-6
        derivative = (acceleration(z + shift) - acceleration(z - shift)) / (2e-6)
        # Ordinary differences are only a cross-check, never an admission bound.
        assert np.all(derivative >= jacobian.lower[:, j] - 1e-5)
        assert np.all(derivative <= jacobian.upper[:, j] + 1e-5)


def test_shared_provider_rejects_compiled_mutation_on_a_cached_query(panda):
    dynamics, home = panda
    saved = dynamics._snapshot.dof_damping[0]
    try:
        dynamics.jets(VectorBox.point(home), VectorBox.point(np.zeros(7)))
        dynamics._snapshot.dof_damping[0] += 0.25
        with pytest.raises(ValueError, match="snapshot changed"):
            dynamics.jets(VectorBox.point(home), VectorBox.point(np.zeros(7)))
    finally:
        dynamics._snapshot.dof_damping[0] = saved


def test_differential_solve_failure_restores_precision(panda, monkeypatch):
    import flint

    import sarrl.safety.panda_mean_value_dynamics as implementation

    dynamics, home = panda
    before = flint.ctx.prec

    def failure(*args):
        raise ValueError("injected solve failure")

    monkeypatch.setattr(implementation, "solve", failure)
    with pytest.raises(ValueError, match="injected solve"):
        with arithmetic(256) as backend:
            dynamics.acceleration_jacobian(
                PandaParameterBox(np.ones(8), np.ones(8)),
                VectorBox.point(home),
                VectorBox.point(np.zeros(7)),
                VectorBox.point(np.zeros(7)),
                backend,
            )
    assert flint.ctx.prec == before


def test_shared_screen_preserves_reports_and_rejects_source_changes(tmp_path, monkeypatch):
    import gzip
    import json
    import sys

    from tools import assess_panda_mean_value_prefix as tool

    path = tmp_path / "assessment.json.gz"
    path.write_bytes(gzip.compress(json.dumps({"source_sha256": {"wrong": "source"}}).encode()))
    retained = path.read_bytes()
    monkeypatch.setattr(tool, "assessment", lambda: pytest.fail("unexpected numerical run"))
    monkeypatch.setattr(sys, "argv", ["screen", "--output", str(path)])
    with pytest.raises(FileExistsError):
        tool.main()
    monkeypatch.setattr(sys, "argv", ["screen", "--audit", "--output", str(path)])
    with pytest.raises(ValueError, match="fingerprints differ"):
        tool.main()
    assert path.read_bytes() == retained


def test_shared_screen_checks_the_retained_admission_identity():
    from tools.assess_panda_mean_value_prefix import baseline

    report = baseline()
    assert not report["decision"]["guarded_candidate_promoted"]
    assert report["arms"]["bounded_friction"]["structural_audit"]["checked_intervals"] == 229


def test_auditor_rejects_out_of_range_known_sends_even_before_replaying(manufactured):
    from sarrl.safety.panda_mean_value_audit import audit_mean_value_prefix

    checker, context, history, result = manufactured
    history = [np.full(7, 100), *history[1:]]
    with pytest.raises(ValueError, match="known input torque limit"):
        audit_mean_value_prefix(
            checker, context, np.zeros(7), np.zeros(7), np.zeros(7), history, result, intervals=6
        )


def test_signed_ball_magnitude_record_uses_zero_lower_and_unchanged_upper(
    manufactured, monkeypatch
):
    from sarrl.safety.panda_mean_value_audit import audit_mean_value_prefix

    checker, context, history, _ = manufactured

    def crossing_zero(*args):
        backend = args[-1]
        return np.full(7, interval(-1, 1, backend), dtype=object)

    monkeypatch.setattr(checker.region, "jerk", crossing_zero)
    result = checker.verify_prefix(
        context, np.zeros(7), np.zeros(7), np.zeros(7), history, intervals=6
    )
    first = result.cases[0].intervals[0]
    assert np.all(first.jerk_bound.lower == 0) and np.all(first.jerk_bound.upper >= 1)
    assert (
        audit_mean_value_prefix(
            checker, context, np.zeros(7), np.zeros(7), np.zeros(7), history, result, intervals=6
        )["checked_intervals"]
        == 30
    )


def test_held_substeps_cover_exact_trajectories_and_pass_the_audit(manufactured):
    from sarrl.safety.panda_mean_value_audit import audit_mean_value_prefix

    checker, context, history, _ = manufactured
    result = checker.verify_prefix(
        context, np.zeros(7), np.zeros(7), np.zeros(7), history, intervals=6, substeps=4
    )
    assert result.prefix_contained and all(len(c.intervals) == 24 for c in result.cases)
    h = Fraction(checker.config.control_dt) / 4
    with arithmetic(128) as backend:
        for case in result.cases:
            for epsilon in (-1, 0, 1):
                mass = Fraction(0.5625) + Fraction(0.0625) * Fraction(epsilon)
                q, v, queue = Fraction(0), Fraction(0), [Fraction(0)] * 4
                eps = np.array([epsilon, *[0] * 7])
                for step, record in enumerate(case.intervals):
                    if step % 4 == 0:
                        sent = Fraction(0) if step == 0 else -q - v
                        delivered = queue[-case.lag] if case.lag else sent
                    acceleration = delivered + mass
                    q, v = q + h * v + h * h / 2 * acceleration, v + h * acceleration
                    for form, actual in ((record.q_end, q), (record.v_end, v)):
                        enclosed = fiber(form, eps, backend)
                        assert (
                            Fraction(float(enclosed.lower[0]))
                            <= actual
                            <= Fraction(float(enclosed.upper[0]))
                        )
                    if step % 4 == 3:
                        queue = [*queue[1:], sent]
    checked = audit_mean_value_prefix(
        checker,
        context,
        np.zeros(7),
        np.zeros(7),
        np.zeros(7),
        history,
        result,
        intervals=6,
        substeps=4,
    )
    assert checked["checked_intervals"] == 120
    case = result.cases[2]
    record = replace(case.intervals[5], sent=AffineState.constant(np.ones(7)))
    case = replace(case, intervals=(*case.intervals[:5], record, *case.intervals[6:]))
    with pytest.raises(ValueError, match="prefix audit"):
        audit_mean_value_prefix(
            checker,
            context,
            np.zeros(7),
            np.zeros(7),
            np.zeros(7),
            history,
            replace(result, cases=(*result.cases[:2], case, *result.cases[3:])),
            intervals=6,
            substeps=4,
        )


def test_substeps_must_partition_the_held_period_exactly(manufactured):
    checker = manufactured[0]
    assert checker.substep_period(4) * 4 == checker.config.control_dt
    for count in (0, 3, 1.0):
        with pytest.raises(ValueError, match="substep"):
            checker.substep_period(count)


def test_substep_phase_continues_only_when_every_branch_completes():
    from tools.assess_panda_substep_prefix import decide

    case = {"prefix_contained": True, "intervals": [None] * 24}
    cells = [{"result": {"cases": [case] * 5}} for _ in range(16)]
    assert decide(cells)["outcome"] == "line_continues"
    partial = {"prefix_contained": False, "intervals": [None] * 23}
    cells[3] = {"result": {"cases": [case] * 4 + [partial]}}
    decision = decide(cells)
    assert (decision["outcome"], decision["complete_branches"]) == ("line_stops", 79)
    assert decision["held_intervals"] == 79 * 6 + 5
    assert not decision["integration_allowed"] and not decision["release_allowed"]


def test_substep_phase_preserves_reports_and_rejects_source_changes(tmp_path, monkeypatch):
    import gzip
    import json
    import sys

    from tools import assess_panda_substep_prefix as tool

    path = tmp_path / "assessment.json.gz"
    path.write_bytes(gzip.compress(json.dumps({"source_sha256": {"wrong": "source"}}).encode()))
    kept = path.read_bytes()
    monkeypatch.setattr(tool, "assessment", lambda: pytest.fail("unexpected numerical run"))
    monkeypatch.setattr(sys, "argv", ["screen", "--output", str(path)])
    with pytest.raises(FileExistsError):
        tool.main()
    monkeypatch.setattr(sys, "argv", ["screen", "--audit", "--output", str(path)])
    with pytest.raises(ValueError, match="fingerprints differ"):
        tool.main()
    assert path.read_bytes() == kept
