from dataclasses import replace

import numpy as np
import pytest

from sippy import get_frequency_response_uncertainty, get_model_uncertainty
from sippy.identification.algorithms.frequency_domain import FrequencyDomainAlgorithm
from sippy.identification.base import StateSpaceModel
from sippy.identification.uncertainty import (
    FrequencyResponseUncertainty,
    estimate_frequency_response_uncertainty,
)


def make_siso_model(dt: float = 0.2) -> StateSpaceModel:
    return StateSpaceModel(
        A=np.array([[0.65]]),
        B=np.array([[0.8]]),
        C=np.array([[1.0]]),
        D=np.array([[0.0]]),
        K=np.zeros((1, 1)),
        Q=np.eye(1),
        R=np.eye(1),
        S=np.zeros((1, 1)),
        ts=dt,
        Vn=0.0,
    )


def make_siso_data(
    *, n_samples: int = 4096, noise: float = 0.05, seed: int = 42
) -> tuple[StateSpaceModel, np.ndarray, np.ndarray]:
    model = make_siso_model()
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(n_samples)
    _, y = model.simulate(u[None, :])
    y = y[0] + noise * rng.standard_normal(n_samples)
    return model, u, y


def make_mimo_model(dt: float = 0.1) -> StateSpaceModel:
    return StateSpaceModel(
        A=np.diag([0.45, 0.7]),
        B=np.array([[0.8, 0.2], [0.1, 0.6]]),
        C=np.eye(2),
        D=np.zeros((2, 2)),
        K=np.zeros((2, 2)),
        Q=np.eye(2),
        R=np.eye(2),
        S=np.zeros((2, 2)),
        ts=dt,
        Vn=np.zeros((2, 2)),
    )


def test_uncertainty_uses_actual_segment_degrees_of_freedom():
    model, u, y = make_siso_data(n_samples=2048)

    result = model.get_model_uncertainty(u, y, nperseg=256, smoothing_bins=5)

    assert result.n_segments == 8
    assert result.degrees_of_freedom == 7
    assert result.empirical_response.shape == (129, 1, 1)
    assert result.model_response.shape == (129, 1, 1)
    assert result.magnitude_standard_error_db.shape == (129, 1, 1)


def test_empirical_uncertainty_does_not_fabricate_a_model_response():
    _, u, y = make_siso_data(n_samples=2048)

    result = estimate_frequency_response_uncertainty(
        u,
        y,
        dt=0.2,
        nperseg=256,
    )

    assert result.model_response is None
    with pytest.raises(ValueError, match="No model response"):
        _ = result.model_magnitude_db


def test_confidence_intervals_are_centered_on_the_empirical_response():
    model, u, y = make_siso_data(n_samples=2048)

    result = model.get_frequency_response_uncertainty(
        u,
        y,
        nperseg=256,
        confidence_levels=(0.95,),
    )
    magnitude_lower, magnitude_upper = result.magnitude_confidence_interval(0.95)
    phase_lower, phase_upper = result.phase_confidence_interval(0.95)

    np.testing.assert_allclose(
        (magnitude_lower + magnitude_upper) / 2,
        result.empirical_magnitude_db,
    )
    np.testing.assert_allclose(
        (phase_lower + phase_upper) / 2,
        result.empirical_phase_deg,
    )


def test_phase_validation_error_uses_the_nearest_phase_branch():
    _, u, y = make_siso_data(n_samples=2048)
    result = estimate_frequency_response_uncertainty(
        u,
        y,
        dt=0.2,
        nperseg=256,
    )
    shape = result.empirical_response.shape
    empirical_response = np.full(shape, np.exp(1j * np.deg2rad(170.0)))
    model_response = np.full(shape, np.exp(1j * np.deg2rad(-170.0)))
    result = replace(
        result,
        empirical_response=empirical_response,
        model_response=model_response,
    )

    np.testing.assert_allclose(result.phase_validation_error_deg, -20.0)
    np.testing.assert_allclose(result.model_validation_phase_deg, 190.0)


def test_validation_envelope_is_model_centered_and_includes_empirical_error():
    true_model, u, y = make_siso_data(n_samples=4096, noise=0.05, seed=19)
    wrong_model = StateSpaceModel(
        A=np.array([[0.25]]),
        B=np.array([[0.2]]),
        C=np.array([[1.0]]),
        D=np.array([[0.0]]),
        K=np.zeros((1, 1)),
        Q=np.eye(1),
        R=np.eye(1),
        S=np.zeros((1, 1)),
        ts=0.2,
        Vn=0.0,
    )

    result = wrong_model.get_frequency_response_uncertainty(
        u,
        y,
        nperseg=256,
        confidence_levels=(0.95,),
    )
    empirical_lower, empirical_upper = result.magnitude_confidence_interval(0.95)
    validation_lower, validation_upper = result.magnitude_validation_envelope(0.95)
    validation_half_width = (validation_upper - validation_lower) / 2
    expected_half_width = np.maximum(
        np.abs(result.model_magnitude_db - empirical_lower),
        np.abs(empirical_upper - result.model_magnitude_db),
    )

    np.testing.assert_allclose(
        (validation_lower + validation_upper) / 2,
        result.model_magnitude_db,
    )
    np.testing.assert_allclose(validation_half_width, expected_half_width)
    assert np.all(validation_lower <= empirical_lower + 1e-12)
    assert np.all(validation_upper >= empirical_upper - 1e-12)

    true_result = true_model.get_frequency_response_uncertainty(
        u,
        y,
        nperseg=256,
        confidence_levels=(0.95,),
    )
    true_empirical_lower, true_empirical_upper = (
        true_result.magnitude_confidence_interval(0.95)
    )
    true_validation_lower, true_validation_upper = (
        true_result.magnitude_validation_envelope(0.95)
    )
    np.testing.assert_allclose(empirical_lower, true_empirical_lower)
    np.testing.assert_allclose(empirical_upper, true_empirical_upper)
    assert np.median(validation_upper - validation_lower) > np.median(
        true_validation_upper - true_validation_lower
    )

    empirical_phase_lower, empirical_phase_upper = result.phase_confidence_interval(
        0.95
    )
    validation_phase_lower, validation_phase_upper = result.phase_validation_envelope(
        0.95
    )
    np.testing.assert_allclose(
        (validation_phase_lower + validation_phase_upper) / 2,
        result.model_validation_phase_deg,
        atol=1e-12,
    )
    assert np.all(validation_phase_lower <= empirical_phase_lower + 1e-12)
    assert np.all(validation_phase_upper >= empirical_phase_upper - 1e-12)


def test_uncertainty_supports_records_shorter_than_legacy_fft_length():
    model, u, y = make_siso_data(n_samples=400)

    result = model.get_model_uncertainty(u, y, nperseg=80, smoothing_bins=3)

    assert result.frequency_hz.shape == (41,)
    assert np.all(np.isfinite(result.magnitude_standard_error_db))


def test_uncertainty_preserves_sample_time_in_frequency_axis():
    model, u, y = make_siso_data(n_samples=2048)

    result = model.get_model_uncertainty(u, y, nperseg=256)

    assert result.frequency_hz[-1] == pytest.approx(1.0 / (2.0 * model.ts))
    assert result.omega[-1] == pytest.approx(np.pi / model.ts)


def test_uncertainty_widens_when_output_noise_increases():
    model, u, y_low = make_siso_data(n_samples=8192, noise=0.02, seed=7)
    _, _, y_high = make_siso_data(n_samples=8192, noise=0.8, seed=7)

    low = model.get_model_uncertainty(u, y_low, nperseg=512, smoothing_bins=5)
    high = model.get_model_uncertainty(u, y_high, nperseg=512, smoothing_bins=5)
    interior = slice(2, -2)

    assert np.median(high.magnitude_standard_error_db[interior]) > 2 * np.median(
        low.magnitude_standard_error_db[interior]
    )
    assert np.median(high.coherence[interior]) < np.median(low.coherence[interior])


def test_uncertainty_confidence_band_covers_known_siso_response():
    model, u, y = make_siso_data(n_samples=16384, noise=0.1, seed=12)

    result = model.get_model_uncertainty(
        u,
        y,
        nperseg=1024,
        smoothing_bins=7,
        confidence_levels=(0.95,),
    )
    lower, upper = result.magnitude_confidence_interval(0.95)
    model_magnitude = 20 * np.log10(np.maximum(np.abs(result.model_response), 1e-300))
    interior = slice(2, -2)
    covered = (model_magnitude[interior] >= lower[interior]) & (
        model_magnitude[interior] <= upper[interior]
    )

    assert np.mean(covered) > 0.8


def test_uncertainty_supports_correlated_mimo_inputs():
    model = make_mimo_model()
    rng = np.random.default_rng(18)
    n_samples = 8192
    u0 = rng.standard_normal(n_samples)
    u1 = 0.7 * u0 + 0.7 * rng.standard_normal(n_samples)
    u = np.vstack([u0, u1])
    _, y = model.simulate(u)
    y = y + 0.1 * rng.standard_normal(y.shape)

    result = model.get_model_uncertainty(u, y, nperseg=512, smoothing_bins=5)

    assert result.empirical_response.shape == (257, 2, 2)
    assert result.model_response.shape == (257, 2, 2)
    assert result.coherence.shape == (257, 2)
    assert result.signal_to_noise_ratio.shape == (257, 2)
    assert result.magnitude_standard_error_db.shape == (257, 2, 2)
    assert np.all(np.isfinite(result.empirical_response))


def test_frequency_domain_result_uses_identified_response_for_shared_methods():
    _, u, y = make_siso_data(n_samples=4096, noise=0.05)
    model = FrequencyDomainAlgorithm().identify(
        y=y,
        u=u,
        fd_method="welch",
        nperseg=256,
        smoothing_window=5,
    )
    stored = model.identification_info["frequency_response"]

    response = model.frequency_response()

    assert model.G is None
    np.testing.assert_allclose(response.omega, stored["omega_real"])
    np.testing.assert_allclose(response.frdata, np.transpose(stored["G"], (1, 2, 0)))
    with pytest.raises(NotImplementedError, match="non-parametric"):
        model.simulate(u)


def test_correlation_frequency_domain_result_uses_shared_uncertainty_engine():
    _, u, y = make_siso_data(n_samples=4096, noise=0.05)
    model = FrequencyDomainAlgorithm().identify(y=y, u=u)

    result = model.get_model_uncertainty(u, y, nperseg=256)

    assert result.model_response.shape == result.empirical_response.shape
    assert np.all(np.isfinite(result.model_response))


def test_standalone_uncertainty_supports_raw_fir_coefficients():
    model, u, y = make_siso_data(n_samples=4096, noise=0.05)
    impulse = np.zeros(80)
    impulse[1:] = 0.8 * 0.65 ** np.arange(78, -1, -1)[::-1]

    result = get_frequency_response_uncertainty(
        u,
        y,
        impulse,
        dt=model.ts,
        nperseg=256,
    )

    assert isinstance(result, FrequencyResponseUncertainty)
    assert result.model_response.shape == (129, 1, 1)
    expected = np.transpose(model.frequency_response(result.omega).frdata, (2, 0, 1))
    np.testing.assert_allclose(result.model_response, expected, atol=1e-11)

    compatibility_result = get_model_uncertainty(
        u,
        y,
        impulse,
        dt=model.ts,
        nperseg=256,
    )
    np.testing.assert_allclose(
        compatibility_result.empirical_response,
        result.empirical_response,
    )
    np.testing.assert_allclose(
        compatibility_result.model_response, result.model_response
    )


def test_uncertainty_rejects_overlapping_jackknife_segments():
    model, u, y = make_siso_data(n_samples=2048)

    with pytest.raises(ValueError, match="non-overlapping"):
        model.get_model_uncertainty(u, y, nperseg=256, noverlap=128)


def test_uncertainty_requires_enough_segments_for_mimo_input_rank():
    model = make_mimo_model()
    rng = np.random.default_rng(23)
    u = rng.standard_normal((2, 1024))
    _, y = model.simulate(u)

    with pytest.raises(ValueError, match="full-rank leave-one-out"):
        model.get_model_uncertainty(u, y, nperseg=512)


@pytest.mark.parametrize(
    "method_name",
    [
        "frequency_response",
        "get_frequency_response_uncertainty",
        "get_model_uncertainty",
        "get_fir_coefficients",
        "get_step_response",
        "simulate",
        "is_stable",
    ],
)
def test_parametric_and_frequency_domain_results_share_analysis_methods(method_name):
    parametric = make_siso_model()
    _, u, y = make_siso_data(n_samples=1024)
    nonparametric = FrequencyDomainAlgorithm().identify(
        y=y, u=u, fd_method="welch", nperseg=128
    )

    assert callable(getattr(parametric, method_name))
    assert callable(getattr(nonparametric, method_name))
