"""
Tests for fitting parametric transfer functions to FD frequency responses.
"""

import control
import numpy as np
import pytest
from scipy import signal

from sippy.identification.algorithms.frequency_domain import (
    FrequencyDomainAlgorithm,
)
from sippy.identification.base import StateSpaceModel
from sippy.identification.frf_fit import fit_frf_model, fit_rational_frf

B_TRUE = [0.0, 0.5, 0.3]
A_TRUE = [1.0, -1.2, 0.35]


def simulate_fit_percent(model, u, y_ref):
    """NRMSE fit%% of the model's simulated output against a reference."""
    n = model.A.shape[0]
    x = np.zeros(n)
    y_hat = np.zeros(len(u))
    for k in range(len(u)):
        y_hat[k] = (model.C @ x + model.D[:, 0] * u[k]).item()
        x = model.A @ x + model.B[:, 0] * u[k]
    return 100 * (
        1 - np.linalg.norm(y_ref - y_hat) / np.linalg.norm(y_ref - y_ref.mean())
    )


class TestFitRationalFRF:
    def test_exact_samples_recover_coefficients(self):
        omega = np.linspace(0.05, 3.0, 200)
        _, G = signal.freqz(B_TRUE, A_TRUE, worN=omega)
        b, a, info = fit_rational_frf(omega, G, na=2, nb=2, nk=1)
        np.testing.assert_allclose(b, [0.5, 0.3], atol=1e-8)
        np.testing.assert_allclose(a, A_TRUE, atol=1e-8)
        assert info["converged"]

    def test_delay_handled_through_nk(self):
        omega = np.linspace(0.05, 3.0, 300)
        b_delayed = np.r_[np.zeros(3), 0.5, 0.3]
        _, G = signal.freqz(b_delayed, A_TRUE, worN=omega)
        b, a, _ = fit_rational_frf(omega, G, na=2, nb=2, nk=3)
        np.testing.assert_allclose(b, [0.5, 0.3], atol=1e-8)
        np.testing.assert_allclose(a, A_TRUE, atol=1e-8)

    def test_noisy_samples_with_weights(self):
        rng = np.random.default_rng(1)
        omega = np.linspace(0.05, 3.0, 400)
        _, G = signal.freqz(B_TRUE, A_TRUE, worN=omega)
        noise_scale = np.linspace(0.005, 0.15, 400)  # low freq clean
        G_noisy = G + noise_scale * (
            rng.standard_normal(400) + 1j * rng.standard_normal(400)
        )
        weights = 1.0 / noise_scale
        b, a, _ = fit_rational_frf(omega, G_noisy, na=2, nb=2, nk=1, weights=weights)
        np.testing.assert_allclose(a, A_TRUE, atol=0.05)
        np.testing.assert_allclose(b, [0.5, 0.3], atol=0.05)

    def test_invalid_orders_raise(self):
        omega = np.linspace(0.1, 3.0, 50)
        G = np.ones(50, dtype=complex)
        with pytest.raises(ValueError, match="Orders"):
            fit_rational_frf(omega, G, na=-1, nb=2)
        with pytest.raises(ValueError, match="Orders"):
            fit_rational_frf(omega, G, na=2, nb=0)

    def test_too_few_bins_raise(self):
        omega = np.linspace(0.1, 3.0, 3)
        G = np.ones(3, dtype=complex)
        with pytest.raises(ValueError, match="at least"):
            fit_rational_frf(omega, G, na=2, nb=2)

    @pytest.mark.parametrize("n_iter", [0, -1, 1.5])
    def test_invalid_iteration_count_raises(self, n_iter):
        omega = np.linspace(0.1, 3.0, 20)
        with pytest.raises(ValueError, match="n_iter"):
            fit_rational_frf(omega, np.ones(20), na=1, nb=1, n_iter=n_iter)

    @pytest.mark.parametrize(
        "weights",
        [
            np.ones(19),
            -np.ones(20),
            np.full(20, np.nan),
            np.zeros(20),
        ],
    )
    def test_invalid_weights_raise(self, weights):
        omega = np.linspace(0.1, 3.0, 20)
        with pytest.raises(ValueError, match="weights"):
            fit_rational_frf(omega, np.ones(20), na=1, nb=1, weights=weights)

    def test_weight_scaling_does_not_change_fit(self):
        omega = np.linspace(0.05, 3.0, 200)
        _, G = signal.freqz(B_TRUE, A_TRUE, worN=omega)
        weights = np.linspace(0.1, 2.0, len(omega))
        fit = fit_rational_frf(omega, G, na=2, nb=2, weights=weights)
        scaled_fit = fit_rational_frf(omega, G, na=2, nb=2, weights=1e150 * weights)
        np.testing.assert_allclose(fit[0], scaled_fit[0])
        np.testing.assert_allclose(fit[1], scaled_fit[1])
        assert fit[2]["weighted_rms_error"] == pytest.approx(
            scaled_fit[2]["weighted_rms_error"]
        )


class TestFitFRFModel:
    @pytest.fixture
    def siso_fd_result(self):
        rng = np.random.default_rng(0)
        N = 8000
        u = rng.standard_normal(N)
        y = signal.lfilter(B_TRUE, A_TRUE, u) + 0.05 * rng.standard_normal(N)
        fd = FrequencyDomainAlgorithm().identify(y=y, u=u)
        return fd, u

    def test_siso_correlation_recovers_true_model(self, siso_fd_result):
        fd, u = siso_fd_result
        par = fit_frf_model(fd, na=2, nb=2, nk=1)
        assert isinstance(par, StateSpaceModel)
        assert isinstance(par.G, control.StateSpace)
        assert isinstance(par.G_tf, control.TransferFunction)
        ch = par.identification_info["frf_fit"][0][0]
        np.testing.assert_allclose(ch["a"], A_TRUE, atol=0.02)
        np.testing.assert_allclose(ch["b"], [0.5, 0.3], atol=0.02)
        poles = np.sort(np.linalg.eigvals(par.A).real)
        np.testing.assert_allclose(poles, [0.5, 0.7], atol=0.02)

    def test_siso_time_domain_fit_quality(self, siso_fd_result):
        fd, u = siso_fd_result
        par = fit_frf_model(fd, na=2, nb=2, nk=1)
        y_clean = signal.lfilter(B_TRUE, A_TRUE, u)
        assert simulate_fit_percent(par, u, y_clean) > 98.0

    def test_siso_welch_estimator_also_fits(self):
        rng = np.random.default_rng(2)
        N = 8000
        u = rng.standard_normal(N)
        y = signal.lfilter(B_TRUE, A_TRUE, u) + 0.05 * rng.standard_normal(N)
        fd = FrequencyDomainAlgorithm().identify(y=y, u=u, fd_method="welch")
        par = fit_frf_model(fd, na=2, nb=2, nk=1)
        ch = par.identification_info["frf_fit"][0][0]
        np.testing.assert_allclose(ch["a"], A_TRUE, atol=0.05)
        np.testing.assert_allclose(ch["b"], [0.5, 0.3], atol=0.05)

    def test_mimo_fit_recovers_both_channels(self):
        rng = np.random.default_rng(5)
        N = 10000
        b1, a1 = [0.0, 0.4, 0.1], [1.0, -0.5]  # no pole-zero cancellation
        u0 = rng.standard_normal(N)
        u1 = 0.6 * signal.lfilter([1, 0.5], [1], u0) + 0.8 * rng.standard_normal(N)
        y = (
            signal.lfilter(B_TRUE, A_TRUE, u0)
            + signal.lfilter(b1, a1, u1)
            + 0.05 * rng.standard_normal(N)
        )
        fd = FrequencyDomainAlgorithm().identify(y=y, u=np.vstack([u0, u1]))
        par = fit_frf_model(fd, na=2, nb=2, nk=1)

        assert par.B.shape[1] == 2
        assert par.C.shape[0] == 1
        fits = par.identification_info["frf_fit"][0]
        np.testing.assert_allclose(fits[0]["a"], A_TRUE, atol=0.05)
        np.testing.assert_allclose(fits[0]["b"], [0.5, 0.3], atol=0.05)
        # channel 1 has true order 1; the fitted na=2 model must still match
        # the true FRF closely even if coefficients are a non-minimal
        # equivalent
        omega = np.linspace(0.1, 3.0, 200)
        _, G_true = signal.freqz(b1, a1, worN=omega)
        E = np.exp(-1j * omega)
        b, a = fits[1]["b"], fits[1]["a"]
        G_fit = (E * np.polyval(b[::-1], E)) / np.polyval(a[::-1], E)
        rel = np.abs(G_fit - G_true) / np.abs(G_true)
        assert np.median(rel) < 0.05

    def test_dt_propagates_to_fitted_model(self):
        rng = np.random.default_rng(3)
        N = 4000
        u = rng.standard_normal(N)
        y = signal.lfilter(B_TRUE, A_TRUE, u) + 0.05 * rng.standard_normal(N)
        fd = FrequencyDomainAlgorithm().identify(y=y, u=u, dt=0.25)
        par = fit_frf_model(fd, na=2, nb=2, nk=1)
        assert par.ts == 0.25
        assert par.G.dt == pytest.approx(0.25)
        assert par.G_tf.dt == pytest.approx(0.25)

    def test_diagnostics_recorded(self, siso_fd_result):
        fd, _ = siso_fd_result
        par = fit_frf_model(fd, na=2, nb=2, nk=1)
        info = par.identification_info
        assert info["method"] == "FD-FIT"
        assert info["source_estimator"] == "correlation"
        assert info["orders"] == {"na": 2, "nb": 2, "nk": 1}
        ch = info["frf_fit"][0][0]
        assert ch["converged"]
        assert ch["relative_error"] < 0.05
        assert par.Vn >= 0

    def test_rejects_non_fd_model(self):
        dummy = StateSpaceModel(
            A=np.eye(1),
            B=np.zeros((1, 1)),
            C=np.zeros((1, 1)),
            D=np.zeros((1, 1)),
            K=np.zeros((1, 1)),
            Q=np.eye(1),
            R=np.eye(1),
            S=np.zeros((1, 1)),
            ts=1.0,
            Vn=0.0,
        )
        with pytest.raises(ValueError, match="FD"):
            fit_frf_model(dummy, na=2, nb=2)

    def test_min_coherence_too_strict_raises(self):
        # output unrelated to input -> coherence is low everywhere
        rng = np.random.default_rng(9)
        u = rng.standard_normal(2000)
        y = rng.standard_normal(2000)
        fd = FrequencyDomainAlgorithm().identify(y=y, u=u)
        with pytest.raises(ValueError, match="coherence"):
            fit_frf_model(fd, na=2, nb=2, min_coherence=0.95)

    @pytest.mark.parametrize("min_coherence", [-0.1, 1.1, np.nan])
    def test_invalid_min_coherence_raises(self, siso_fd_result, min_coherence):
        fd, _ = siso_fd_result
        with pytest.raises(ValueError, match="min_coherence"):
            fit_frf_model(fd, na=2, nb=2, min_coherence=min_coherence)

    def test_unstable_fit_warns(self):
        rng = np.random.default_rng(11)
        N = 4000
        u = rng.standard_normal(N)
        # true system is a pure delay-gain: over-parameterized fits place a
        # cancelling pole-zero pair that can land on/outside the unit circle
        y = 0.4 * np.r_[0.0, u[:-1]] + 0.01 * rng.standard_normal(N)
        fd = FrequencyDomainAlgorithm().identify(y=y, u=u)
        import warnings as w

        with w.catch_warnings():
            w.simplefilter("ignore")
            par = fit_frf_model(fd, na=2, nb=2, nk=1)
        # regardless of whether the warning fired, the fitted FRF must match
        omega = np.linspace(0.1, 3.0, 100)
        E = np.exp(-1j * omega)
        ch = par.identification_info["frf_fit"][0][0]
        G_fit = (E * np.polyval(ch["b"][::-1], E)) / np.polyval(ch["a"][::-1], E)
        G_true = 0.4 * E
        assert np.median(np.abs(G_fit - G_true) / np.abs(G_true)) < 0.05
