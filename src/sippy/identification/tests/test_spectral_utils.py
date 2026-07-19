"""
Tests for the shared spectral analysis utilities.

Ported from the frequency-domain PR and extended with ground-truth accuracy
checks, convention regressions (cross-correlation lag sign, spectrum
frequency alignment), and MIMO cross-spectral estimation.
"""

import numpy as np
import pytest
from scipy import signal

from sippy.utils.spectral_utils import (
    autocorrelation_fft,
    compute_coherence,
    compute_correlations_fft,
    compute_cross_spectrum_welch,
    compute_csd_matrices,
    compute_frequency_response,
    compute_output_spectrum,
    compute_power_spectrum_welch,
    compute_spectra_from_correlation,
    create_hamming_window,
    create_window,
    denormalize_frequency,
    estimate_frf_mimo,
    extract_magnitude_phase,
    smooth_frequency_response,
    validate_signal_pair,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def siso_system_data(rng):
    """White-noise input through a known second-order filter."""
    N = 4000
    b = [0.0, 0.5, 0.3]
    a = [1.0, -1.2, 0.35]
    u = rng.standard_normal(N)
    y = signal.lfilter(b, a, u)
    return u, y, b, a


class TestWelchSpectra:
    def test_power_spectrum_returns_arrays(self, rng):
        x = rng.standard_normal(2048)
        freqs, Pxx = compute_power_spectrum_welch(x, dt=1.0, nperseg=256)
        assert isinstance(freqs, np.ndarray)
        assert isinstance(Pxx, np.ndarray)
        assert len(freqs) == len(Pxx)

    def test_power_spectrum_positive(self, rng):
        x = rng.standard_normal(2048)
        _, Pxx = compute_power_spectrum_welch(x, nperseg=256)
        assert np.all(Pxx >= 0)

    def test_power_spectrum_frequency_range(self, rng):
        x = rng.standard_normal(2048)
        freqs, _ = compute_power_spectrum_welch(x, dt=0.5, nperseg=256)
        assert freqs[0] == 0
        assert np.isclose(freqs[-1], 1.0)  # Nyquist = 1/(2*dt)

    def test_cross_spectrum_complex(self, rng):
        u = rng.standard_normal(2048)
        y = np.roll(u, 3)
        freqs, Puy = compute_cross_spectrum_welch(u, y, nperseg=256)
        assert np.iscomplexobj(Puy)
        assert len(freqs) == len(Puy)


class TestCorrelations:
    def test_returns_tuple_with_expected_lengths(self, rng):
        u = rng.standard_normal(500)
        y = rng.standard_normal(500)
        tau, R_u, R_uy = compute_correlations_fft(u, y, max_lag=50)
        assert len(tau) == 101
        assert len(R_u) == 101
        assert len(R_uy) == 101
        assert tau[0] == -50 and tau[-1] == 50

    def test_autocorrelation_even_and_peaks_at_zero(self, rng):
        u = rng.standard_normal(2000)
        tau, R_u, _ = compute_correlations_fft(u, u, max_lag=40)
        center = len(tau) // 2
        np.testing.assert_allclose(R_u, R_u[::-1], atol=1e-10)
        assert np.argmax(R_u) == center

    def test_default_max_lag(self, rng):
        u = rng.standard_normal(200)
        y = rng.standard_normal(200)
        tau, R_u, R_uy = compute_correlations_fft(u, y)
        assert len(tau) == 2 * 199 + 1

    def test_cross_correlation_causal_lag_sign(self, rng):
        """R_uy(tau) = E[u(t) y(t+tau)]: a causal response must show up at
        POSITIVE lags. Regression for the flipped-lag bug that conjugated
        the cross-spectrum."""
        N = 5000
        b = [0.0, 0.5, 0.3]
        a = [1.0, -1.2, 0.35]
        u = rng.standard_normal(N)
        y = signal.lfilter(b, a, u)
        tau, _, R_uy = compute_correlations_fft(u, y, max_lag=10)
        impulse = signal.lfilter(b, a, np.r_[1.0, np.zeros(10)])
        causal = R_uy[tau >= 0]
        np.testing.assert_allclose(causal, impulse, atol=0.1)
        # Anti-causal side should be near zero
        assert np.max(np.abs(R_uy[tau < 0])) < 0.1

    def test_autocorrelation_fft_matches_pair_function(self, rng):
        x = rng.standard_normal(1000)
        _, R_from_pair, _ = compute_correlations_fft(x, x, max_lag=30)
        R_direct = autocorrelation_fft(x, max_lag=30)
        np.testing.assert_allclose(R_direct, R_from_pair, atol=1e-12)


class TestSpectraFromCorrelation:
    def test_returns_sorted_omega_and_real_phi_u(self, rng):
        u = rng.standard_normal(1000)
        y = rng.standard_normal(1000)
        _, R_u, R_uy = compute_correlations_fft(u, y, max_lag=64)
        Phi_u, Phi_uy, omega = compute_spectra_from_correlation(R_u, R_uy)
        assert np.all(np.diff(omega) > 0)
        assert np.isrealobj(Phi_u)
        assert np.iscomplexobj(Phi_uy)
        assert len(Phi_u) == len(Phi_uy) == len(omega) == len(R_u)

    def test_white_noise_spectrum_flat_and_positive(self, rng):
        """With the lag-0-first FFT ordering the white-noise power spectrum
        must be ~flat around the signal variance. Regression for the missing
        ifftshift that contaminated spectra with a linear phase."""
        u = rng.standard_normal(20000)
        _, R_u, _ = compute_correlations_fft(u, u, max_lag=100)
        lag_window = create_window(len(R_u), "hamming")
        Phi_u, _, _ = compute_spectra_from_correlation(
            R_u * lag_window, R_u * lag_window
        )
        assert np.all(Phi_u > 0)
        assert 0.7 < np.mean(Phi_u) < 1.3
        assert np.std(Phi_u) < 0.35

    def test_output_spectrum_aligned_with_input_spectrum(self, rng):
        """Phi_y must use the same ascending-frequency ordering as Phi_u:
        a sinusoid at w0 must peak at the matching omega bins."""
        N = 4000
        w0 = 1.0
        t = np.arange(N)
        y = np.sin(w0 * t) + 0.01 * rng.standard_normal(N)
        max_lag = 200
        Phi_y = compute_output_spectrum(y, max_lag)
        omega = np.sort(np.fft.fftfreq(2 * max_lag + 1, d=1.0) * 2 * np.pi)
        peaks = omega[np.argsort(Phi_y)[-2:]]
        np.testing.assert_allclose(np.abs(peaks), w0, atol=0.05)


class TestFrequencyResponseAndCoherence:
    def test_frf_no_nans(self, rng):
        Phi_u = np.abs(rng.standard_normal(100)) + 0.1
        Phi_uy = rng.standard_normal(100) + 1j * rng.standard_normal(100)
        G = compute_frequency_response(Phi_uy, Phi_u)
        assert not np.any(np.isnan(G))

    def test_frf_handles_zero_input_power(self):
        Phi_u = np.zeros(10)
        Phi_uy = np.ones(10, dtype=complex)
        G = compute_frequency_response(Phi_uy, Phi_u)
        assert np.all(np.isfinite(G))

    def test_coherence_range(self, rng):
        Phi_u = np.abs(rng.standard_normal(100)) + 0.1
        Phi_y = np.abs(rng.standard_normal(100)) + 0.1
        Phi_uy = rng.standard_normal(100) + 1j * rng.standard_normal(100)
        coh = compute_coherence(Phi_uy, Phi_u, Phi_y)
        assert np.all(coh >= 0) and np.all(coh <= 1)

    def test_coherence_high_for_noise_free_system(self, siso_system_data):
        u, y, _, _ = siso_system_data
        max_lag = 200
        _, R_u, R_uy = compute_correlations_fft(u, y, max_lag)
        lw = create_window(len(R_u), "hamming")
        Phi_u, Phi_uy, _ = compute_spectra_from_correlation(R_u * lw, R_uy * lw)
        Phi_y = compute_output_spectrum(y, max_lag, lag_window=lw)
        coh = compute_coherence(Phi_uy, Phi_u, Phi_y)
        assert np.mean(coh) > 0.9

    def test_correlation_pipeline_frf_accuracy(self, siso_system_data):
        """End-to-end Blackman-Tukey estimate must match freqz of the true
        system."""
        u, y, b, a = siso_system_data
        max_lag = 400
        _, R_u, R_uy = compute_correlations_fft(u, y, max_lag)
        lw = create_window(len(R_u), "hamming")
        Phi_u, Phi_uy, omega = compute_spectra_from_correlation(R_u * lw, R_uy * lw)
        G = compute_frequency_response(Phi_uy, Phi_u)
        mask = (omega > 0.1) & (omega < 3.0)
        _, G_true = signal.freqz(b, a, worN=omega[mask])
        rel_err = np.abs(G[mask] - G_true) / np.abs(G_true)
        assert np.median(rel_err) < 0.05


class TestWindows:
    def test_create_window_types(self):
        for wt in ["hann", "hamming", "blackman", "bartlett", "none"]:
            w = create_window(64, wt)
            assert len(w) == 64
        np.testing.assert_array_equal(create_window(10, "none"), np.ones(10))

    def test_create_window_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown window_type"):
            create_window(64, "kaiser")

    def test_hamming_window_normalized_and_odd(self):
        w = create_hamming_window(10, normalize=True)
        assert len(w) == 11  # made odd
        assert np.isclose(np.sum(w), 1.0)

    def test_hamming_window_unnormalized(self):
        w = create_hamming_window(11, normalize=False)
        assert np.sum(w) > 1.0


class TestSmoothing:
    def test_smoothing_reduces_variance(self, rng):
        G = np.ones(500, dtype=complex) + 0.3 * (
            rng.standard_normal(500) + 1j * rng.standard_normal(500)
        )
        w = create_hamming_window(21)
        G_s = smooth_frequency_response(G, w)
        assert np.std(np.abs(G_s)) < np.std(np.abs(G))

    def test_smoothing_stable_across_phase_wrap(self):
        """Constant-magnitude FRF with phase winding through +/- pi must keep
        its magnitude after smoothing. Regression for smoothing the wrapped
        phase directly."""
        omega = np.linspace(-np.pi, np.pi, 400)
        G = np.exp(1j * 5 * omega)  # |G| = 1 everywhere, phase wraps 5 times
        w = create_hamming_window(11)
        G_s = smooth_frequency_response(G, w)
        interior = np.abs(G_s[20:-20])
        assert np.all(interior > 0.9)

    def test_smoothing_unbiased_at_edges(self):
        G = np.full(100, 2.0 + 0.0j)
        w = create_hamming_window(11)
        G_s = smooth_frequency_response(G, w)
        np.testing.assert_allclose(np.abs(G_s), 2.0, rtol=1e-10)


class TestMagnitudePhase:
    def test_returns_tuple_consistent_with_input(self, rng):
        G = rng.standard_normal(50) + 1j * rng.standard_normal(50)
        mag_db, phase_deg = extract_magnitude_phase(G)
        assert mag_db.shape == G.shape
        assert phase_deg.shape == G.shape
        np.testing.assert_allclose(10 ** (mag_db / 20), np.abs(G) + 1e-12, rtol=1e-6)

    def test_multidimensional_unwrap_along_frequency_axis(self):
        omega = np.linspace(0, np.pi, 200)
        G = np.exp(1j * 8 * omega)[:, None, None] * np.ones((1, 2, 2))
        _, phase_deg = extract_magnitude_phase(G)
        # Unwrapped phase should be monotonically increasing, no 360 jumps
        assert np.all(np.diff(phase_deg[:, 0, 0]) > 0)


class TestSignalValidation:
    def test_valid_pair_passes(self, rng):
        u = rng.standard_normal(200)
        y = rng.standard_normal(200)
        u2, y2 = validate_signal_pair(u, y)
        assert len(u2) == len(y2) == 200

    def test_mismatched_length_raises(self, rng):
        with pytest.raises(ValueError, match="same length"):
            validate_signal_pair(rng.standard_normal(200), rng.standard_normal(150))

    def test_short_data_raises(self, rng):
        with pytest.raises(ValueError, match="at least"):
            validate_signal_pair(rng.standard_normal(50), rng.standard_normal(50))

    def test_nan_raises(self, rng):
        u = rng.standard_normal(200)
        u[10] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            validate_signal_pair(u, rng.standard_normal(200))

    def test_inf_raises(self, rng):
        u = rng.standard_normal(200)
        u[10] = np.inf
        with pytest.raises(ValueError, match="infinite"):
            validate_signal_pair(u, rng.standard_normal(200))

    def test_constant_signal_warns(self):
        with pytest.warns(RuntimeWarning, match="low variance"):
            validate_signal_pair(np.ones(200), np.ones(200))


class TestFrequencyDenormalization:
    def test_nyquist(self):
        omega = np.array([np.pi])
        omega_rad, freq_hz = denormalize_frequency(omega, dt=0.1)
        assert np.isclose(omega_rad[0], np.pi / 0.1)
        assert np.isclose(freq_hz[0], 5.0)  # Nyquist for fs = 10 Hz


class TestMIMOSpectra:
    @pytest.fixture
    def mimo_data(self, rng):
        """Two correlated inputs driving one output through known filters."""
        N = 8000
        b0 = [0.0, 0.5, 0.3]
        a0 = [1.0, -1.2, 0.35]
        b1 = [0.0, 0.4, -0.2]
        a1 = [1.0, -0.5]
        u0 = rng.standard_normal(N)
        # u1 deliberately correlated with u0
        u1 = 0.6 * signal.lfilter([1, 0.5], [1], u0) + 0.8 * rng.standard_normal(N)
        u = np.vstack([u0, u1])
        y = signal.lfilter(b0, a0, u0) + signal.lfilter(b1, a1, u1)
        return u, y[None, :], (b0, a0), (b1, a1)

    def test_csd_matrices_shapes(self, mimo_data):
        u, y, _, _ = mimo_data
        freqs, S_uu, S_uy, S_yy = compute_csd_matrices(u, y, nperseg=256)
        F = len(freqs)
        assert S_uu.shape == (F, 2, 2)
        assert S_uy.shape == (F, 2, 1)
        assert S_yy.shape == (F, 1)

    def test_csd_matrix_hermitian_with_positive_diagonal(self, mimo_data):
        u, y, _, _ = mimo_data
        _, S_uu, _, _ = compute_csd_matrices(u, y, nperseg=256)
        np.testing.assert_allclose(
            S_uu, np.conj(np.transpose(S_uu, (0, 2, 1))), atol=1e-12
        )
        diag = np.real(np.diagonal(S_uu, axis1=1, axis2=2))
        assert np.all(diag >= 0)

    def test_mimo_frf_accuracy_with_correlated_inputs(self, mimo_data):
        """The H1 matrix solve must recover both channel FRFs even though
        the inputs are cross-correlated (a naive per-channel Phi_uy/Phi_u
        ratio is biased here)."""
        u, y, ch0, ch1 = mimo_data
        frf = estimate_frf_mimo(u, y, nperseg=512)
        omega = frf["omega"]
        mask = (omega > 0.1) & (omega < 3.0)
        for k, (b, a) in enumerate([ch0, ch1]):
            _, G_true = signal.freqz(b, a, worN=omega[mask])
            rel_err = np.abs(frf["G"][mask, 0, k] - G_true) / np.abs(G_true)
            assert np.median(rel_err) < 0.1, f"channel {k}"

    def test_naive_siso_ratio_is_biased_on_correlated_inputs(self, mimo_data):
        """Documents WHY the matrix solve matters: the SISO ratio for
        channel 0 alone is substantially worse on the same data."""
        u, y, ch0, _ = mimo_data
        frf = estimate_frf_mimo(u, y, nperseg=512)
        omega = frf["omega"]
        mask = (omega > 0.1) & (omega < 3.0)
        _, G_true = signal.freqz(*ch0, worN=omega[mask])

        naive = frf["S_uy"][:, 0, 0] / np.real(frf["S_uu"][:, 0, 0])
        naive_err = np.median(np.abs(naive[mask] - G_true) / np.abs(G_true))
        mimo_err = np.median(np.abs(frf["G"][mask, 0, 0] - G_true) / np.abs(G_true))
        assert mimo_err < naive_err / 2

    def test_multiple_coherence_near_one_noise_free(self, mimo_data):
        u, y, _, _ = mimo_data
        frf = estimate_frf_mimo(u, y, nperseg=512)
        assert np.mean(frf["coherence"]) > 0.95
        assert np.all(frf["coherence"] >= 0)
        assert np.all(frf["coherence"] <= 1)

    def test_multiple_coherence_drops_with_noise(self, mimo_data, rng):
        u, y, _, _ = mimo_data
        y_noisy = y + 2.0 * rng.standard_normal(y.shape)
        frf_clean = estimate_frf_mimo(u, y, nperseg=512)
        frf_noisy = estimate_frf_mimo(u, y_noisy, nperseg=512)
        assert np.mean(frf_noisy["coherence"]) < np.mean(frf_clean["coherence"])

    def test_two_outputs(self, rng):
        N = 6000
        u = rng.standard_normal((2, N))
        y0 = signal.lfilter([0.0, 1.0], [1.0, -0.5], u[0])
        y1 = signal.lfilter([0.0, 0.7], [1.0, -0.3], u[1])
        frf = estimate_frf_mimo(u, np.vstack([y0, y1]), nperseg=512)
        F = len(frf["freq_hz"])
        assert frf["G"].shape == (F, 2, 2)
        assert frf["coherence"].shape == (F, 2)
        omega = frf["omega"]
        mask = (omega > 0.1) & (omega < 3.0)
        _, G_true = signal.freqz([0.0, 1.0], [1.0, -0.5], worN=omega[mask])
        rel_err = np.abs(frf["G"][mask, 0, 0] - G_true) / np.abs(G_true)
        assert np.median(rel_err) < 0.1
        # cross-channels should be near zero (independent inputs)
        assert np.median(np.abs(frf["G"][mask, 0, 1])) < 0.1
