import numpy as np
import pytest

from sippy import systems as control
from sippy.identification.algorithms.n4sid import N4SIDAlgorithm
from sippy.identification.base import IdentificationResult, StateSpaceModel


def _model(**kwargs):
    values = {
        "A": np.array([[0.2]]),
        "B": np.array([[1.0]]),
        "C": np.array([[1.0]]),
        "D": np.array([[0.0]]),
        "K": np.zeros((1, 1)),
        "Q": np.eye(1),
        "R": np.eye(1),
        "S": np.zeros((1, 1)),
        "ts": 0.25,
        "Vn": 0.01,
    }
    values.update(kwargs)
    return StateSpaceModel(**values)


def test_identification_result_is_the_canonical_class_name():
    assert StateSpaceModel is IdentificationResult
    assert _model().__class__.__name__ == "IdentificationResult"


def test_discrete_modal_properties_use_continuous_equivalent_poles():
    sample_time = 0.2
    radius = 0.8
    angle = 0.35
    A = radius * np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    model = _model(
        A=A,
        B=np.ones((2, 1)),
        C=np.ones((1, 2)),
        K=np.zeros((2, 1)),
        Q=np.eye(2),
        S=np.zeros((2, 1)),
        ts=sample_time,
    )

    continuous_pole = (np.log(radius) + 1j * angle) / sample_time
    expected_frequency = abs(continuous_pole) / (2 * np.pi)
    expected_damping = -continuous_pole.real / abs(continuous_pole)

    np.testing.assert_allclose(
        model.get_natural_frequencies(), [expected_frequency, expected_frequency]
    )
    np.testing.assert_allclose(
        model.get_damping_ratios(), [expected_damping, expected_damping]
    )


def test_finalized_polynomial_result_uses_transfer_function_as_canonical_model():
    transfer_function = control.tf([0.4], [1.0, -0.75], dt=0.25)
    u = np.zeros((1, 40))
    u[0, 0] = 1.0
    expected = control.forced_response(transfer_function, U=u, squeeze=False).outputs
    model = _model(G_tf=transfer_function)

    model.finalize_identification(
        method="ARX",
        input_data=u,
        output_data=expected,
        covariance_source=None,
        kalman_gain_source=None,
    )

    _, actual = model.simulate(u)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert model.deterministic_model is model.G_tf
    assert model.method == "ARX"
    assert model.ninputs == 1
    assert model.noutputs == 1


def test_finalization_replaces_placeholder_stochastic_values_with_unavailable():
    u = np.ones((1, 8))
    y = np.arange(8, dtype=float).reshape(1, -1)
    model = _model(Yid=np.zeros_like(y))

    model.finalize_identification(
        method="FIR",
        input_data=u,
        output_data=y,
        covariance_source=None,
        kalman_gain_source=None,
    )

    assert model.K is None
    assert model.Q is None
    assert model.R is None
    assert model.S is None
    np.testing.assert_allclose(model.residuals(), y)
    np.testing.assert_allclose(model.residual_covariance, np.cov(y, bias=False))
    assert model.Vn == pytest.approx(np.mean(y**2))
    assert not model.supports("one_step_prediction")
    assert model.supports("simulation")


def test_fit_reports_per_output_nrmse_and_aggregate_score():
    y = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
    fitted = y - np.array([[0.0, 0.1, -0.1, 0.0], [0.0, 0.2, -0.2, 0.0]])
    model = _model(
        C=np.ones((2, 1)),
        D=np.zeros((2, 1)),
        K=np.zeros((1, 2)),
        R=np.eye(2),
        S=np.zeros((1, 2)),
        Yid=fitted,
    )
    model.finalize_identification(
        method="ARMAX",
        input_data=np.zeros((1, y.shape[1])),
        output_data=y,
        covariance_source=None,
        kalman_gain_source=None,
    )

    fit = model.fit()

    assert fit["nrmse"].shape == (2,)
    assert fit["score"] == pytest.approx(np.mean(fit["nrmse"]))
    assert np.all(fit["nrmse"] <= 1.0)


def test_diagonal_innovations_model_supports_generic_one_step_prediction():
    process = control.tf([0.5], [1.0, -0.7], dt=1.0)
    innovations = control.tf([1.0, 0.25], [1.0, -0.2], dt=1.0)
    rng = np.random.default_rng(42)
    u = rng.standard_normal((1, 100))
    e = 0.05 * rng.standard_normal((1, 100))
    process_output = control.forced_response(process, U=u, squeeze=False).outputs
    noise_output = control.forced_response(innovations, U=e, squeeze=False).outputs
    y = process_output + noise_output
    model = _model(G_tf=process, H_tf=innovations)
    model.finalize_identification(
        method="ARMAX",
        input_data=u,
        output_data=y,
        covariance_source=None,
        kalman_gain_source=None,
    )

    prediction = model.predict(u=u, y=y)

    np.testing.assert_allclose(y - prediction, e, atol=1e-12, rtol=1e-12)
    assert model.supports("one_step_prediction")


def test_common_time_response_methods_use_the_canonical_process_model():
    process = control.tf([0.3], [1.0, -0.6], dt=0.5)
    model = _model(G_tf=process, ts=0.5)
    model.finalize_identification(
        method="OE",
        input_data=np.ones((1, 20)),
        output_data=np.zeros((1, 20)),
        covariance_source=None,
        kalman_gain_source=None,
    )

    impulse = model.impulse_response(20)
    step = model.step_response(20)

    assert impulse.outputs.shape == (1, 1, 20)
    assert step.outputs.shape == (1, 1, 20)
    np.testing.assert_allclose(step.outputs[0, 0, -3:], 0.75, atol=1e-3)


def test_nonparametric_result_has_the_same_methods_with_honest_capabilities():
    model = StateSpaceModel(
        A=np.empty((0, 0)),
        B=np.empty((0, 1)),
        C=np.empty((1, 0)),
        D=np.zeros((1, 1)),
        K=np.empty((0, 1)),
        Q=np.empty((0, 0)),
        R=np.eye(1),
        S=np.empty((0, 1)),
        ts=1.0,
        Vn=0.0,
        identification_info={
            "method": "FD",
            "estimator": "welch",
            "frequency_response": {
                "omega_real": np.array([0.0, 1.0]),
                "G": np.ones((2, 1, 1), dtype=complex),
            },
        },
        is_parametric=False,
    )
    model.finalize_identification(
        method="FD",
        input_data=np.ones((1, 16)),
        output_data=np.ones((1, 16)),
        covariance_source=None,
        kalman_gain_source=None,
    )

    assert model.supports("frequency_response")
    assert not model.supports("simulation")
    assert not model.supports("stability")
    with pytest.raises(NotImplementedError, match="Simulation"):
        model.simulate(np.ones((1, 8)))


def test_subspace_covariances_are_returned_in_physical_output_units():
    rng = np.random.default_rng(8)
    u = rng.standard_normal((1, 320))
    y = np.zeros((1, 320))
    for sample in range(1, y.shape[1]):
        y[0, sample] = (
            0.72 * y[0, sample - 1]
            + 0.4 * u[0, sample - 1]
            + 0.03 * rng.standard_normal()
        )

    base = N4SIDAlgorithm().identify(y=y, u=u, ss_f=10, ss_fixed_order=2, tsample=1.0)
    scale = 4.0
    scaled = N4SIDAlgorithm().identify(
        y=scale * y, u=u, ss_f=10, ss_fixed_order=2, tsample=1.0
    )

    np.testing.assert_allclose(scaled.Q, base.Q, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(scaled.R, scale**2 * base.R, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(scaled.S, scale * base.S, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(scaled.K, base.K / scale, rtol=1e-10, atol=1e-12)

    second_output = np.zeros(y.shape[1])
    for sample in range(1, second_output.size):
        second_output[sample] = (
            0.55 * second_output[sample - 1]
            - 0.2 * u[0, sample - 1]
            + 0.02 * rng.standard_normal()
        )
    y_mimo = np.vstack([y[0], second_output])
    output_scale = np.diag([2.0, 5.0])
    base_mimo = N4SIDAlgorithm().identify(
        y=y_mimo, u=u, ss_f=10, ss_fixed_order=3, tsample=1.0
    )
    scaled_mimo = N4SIDAlgorithm().identify(
        y=output_scale @ y_mimo,
        u=u,
        ss_f=10,
        ss_fixed_order=3,
        tsample=1.0,
    )

    np.testing.assert_allclose(scaled_mimo.Q, base_mimo.Q, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(
        scaled_mimo.R,
        output_scale @ base_mimo.R @ output_scale,
        rtol=1e-9,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        scaled_mimo.S, base_mimo.S @ output_scale, rtol=1e-9, atol=1e-11
    )
    np.testing.assert_allclose(
        scaled_mimo.K,
        base_mimo.K @ np.linalg.inv(output_scale),
        rtol=1e-9,
        atol=1e-11,
    )
