import numpy as np
import pytest

from sippy.identification.algorithms.ararmax import ARARMAXAlgorithm
from sippy.identification.algorithms.bj import BJAlgorithm
from sippy.identification.algorithms.gen import GENAlgorithm
from sippy.identification.algorithms.oe import OEAlgorithm
from sippy.identification.algorithms.opt_support import _schur_reflection_coefficients
from sippy.utils.compiled_utils import (
    create_regression_matrix_armax_compiled,
    matrix_operations_a_compiled,
)


@pytest.mark.parametrize(
    ("module_name", "algorithm", "orders"),
    [
        (
            "ararmax",
            ARARMAXAlgorithm(),
            {"na": 1, "nb": 1, "nc": 1, "nd": 1, "nf": 1, "nk": 1},
        ),
        ("bj", BJAlgorithm(), {"nb": 1, "nc": 1, "nd": 1, "nf": 1, "nk": 1}),
        ("oe", OEAlgorithm(), {"nb": 1, "nf": 1, "nk": 1}),
        (
            "gen",
            GENAlgorithm(),
            {"na": 1, "nb": 1, "nc": 1, "nd": 1, "nf": 1, "nk": 1},
        ),
    ],
)
def test_prediction_error_estimators_require_casadi(
    monkeypatch, module_name, algorithm, orders
):
    monkeypatch.setattr(
        f"sippy.identification.algorithms.{module_name}.CASADI_AVAILABLE", False
    )
    rng = np.random.default_rng(5)
    u = rng.normal(size=(1, 80))
    y = rng.normal(size=(1, 80))

    with pytest.raises(RuntimeError, match="CasADi is required"):
        algorithm.identify(y=y, u=u, **orders)


def test_armax_compiled_regressor_rejects_missing_innovations():
    u = np.arange(20.0).reshape(1, -1)
    y = np.arange(20.0).reshape(1, -1)

    with pytest.raises(ValueError, match="innovation estimates"):
        create_regression_matrix_armax_compiled(
            u=u,
            y=y,
            na=1,
            nb=1,
            nc=1,
            nk=1,
            ny=1,
            nu=1,
            N=20,
        )


def test_compiled_state_input_regression_is_deterministic_least_squares():
    rng = np.random.default_rng(16)
    inputs = rng.normal(size=(2, 30))
    states = np.zeros((2, 30))
    expected_a = np.array([[0.7, 0.1], [-0.2, 0.5]])
    expected_b = np.array([[0.4, -0.1], [0.2, 0.3]])
    for sample in range(29):
        states[:, sample + 1] = (
            expected_a @ states[:, sample] + expected_b @ inputs[:, sample]
        )

    actual_a, actual_b = matrix_operations_a_compiled(
        X_fd=states,
        O_i=np.eye(2),
        n=2,
        B_recalc=True,
        u=inputs,
        f=0,
        N=30,
    )

    np.testing.assert_allclose(actual_a, expected_a, atol=1e-12)
    np.testing.assert_allclose(actual_b, expected_b, atol=1e-12)


def test_schur_reflection_coefficients_accept_stable_polynomial_above_unit_norm():
    coefficients = np.array([-1.5, 0.56])

    reflection = _schur_reflection_coefficients(coefficients, radius=0.95)

    assert np.max(np.abs(reflection)) < 1.0
    assert np.max(np.abs(np.roots(np.r_[1.0, coefficients]))) < 0.95


def test_schur_reflection_coefficients_reject_pole_outside_requested_margin():
    coefficients = np.poly([0.97, 0.7])[1:]

    reflection = _schur_reflection_coefficients(coefficients, radius=0.95)

    assert np.max(np.abs(reflection)) > 1.0


def test_second_order_oe_stability_constraint_is_feasible_for_stable_model():
    rng = np.random.default_rng(88)
    sample_count = 120
    u = rng.normal(size=(1, sample_count))
    y = np.zeros((1, sample_count))
    for sample in range(2, sample_count):
        y[0, sample] = (
            1.5 * y[0, sample - 1] - 0.56 * y[0, sample - 2] + 0.4 * u[0, sample - 1]
        )

    model = OEAlgorithm().identify(
        y=y,
        u=u,
        nb=1,
        nf=2,
        nk=1,
        max_iterations=100,
        stability_constraint=True,
        stability_margin=0.95,
    )

    poles = np.roots(model.G_tf.den[0][0])
    assert np.max(np.abs(poles)) < 0.95
    np.testing.assert_allclose(np.sort(poles), [0.7, 0.8], atol=1e-6)
