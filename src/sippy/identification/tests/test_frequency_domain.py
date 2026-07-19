"""
Tests for the non-parametric frequency-domain identification algorithm.

Covers factory integration, parameter validation, both estimators
(correlation / Welch H1), SISO and MIMO accuracy against known transfer
functions, quality assessment, and edge cases.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import signal

from sippy.identification import SystemIdentification, SystemIdentificationConfig
from sippy.identification.algorithms.frequency_domain import (
    FrequencyDomainAlgorithm,
)
from sippy.identification.base import StateSpaceModel
from sippy.identification.factory import AlgorithmFactory
from sippy.identification.iddata import IDData

B_TRUE = [0.0, 0.5, 0.3]
A_TRUE = [1.0, -1.2, 0.35]


def make_siso_data(N=4000, noise=0.05, seed=42):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(N)
    y = signal.lfilter(B_TRUE, A_TRUE, u) + noise * rng.standard_normal(N)
    return u, y


def median_relative_frf_error(G_est, omega, b, a):
    _, G_true = signal.freqz(b, a, worN=omega)
    return np.median(np.abs(G_est - G_true) / np.abs(G_true))


class TestFactoryIntegration:
    def test_aliases_registered(self):
        for name in ["FD", "FREQUENCY_DOMAIN", "FREQ_DOMAIN"]:
            assert AlgorithmFactory.is_registered(name)

    def test_case_insensitive(self):
        assert AlgorithmFactory.is_registered("fd")
        algo = AlgorithmFactory.create("frequency_domain")
        assert isinstance(algo, FrequencyDomainAlgorithm)

    def test_listed_in_available_algorithms(self):
        assert "FD" in AlgorithmFactory.list_algorithms()


class TestParameterValidation:
    def setup_method(self):
        self.algo = FrequencyDomainAlgorithm()

    def test_defaults_valid(self):
        assert self.algo.validate_parameters() is True

    def test_invalid_fd_method(self):
        with pytest.raises(ValueError, match="fd_method"):
            self.algo.validate_parameters(fd_method="etfe")

    def test_smoothing_window_too_small(self):
        with pytest.raises(ValueError, match="smoothing_window"):
            self.algo.validate_parameters(smoothing_window=2)

    def test_coherence_threshold_out_of_range(self):
        with pytest.raises(ValueError, match="coherence_threshold"):
            self.algo.validate_parameters(coherence_threshold=1.5)
        with pytest.raises(ValueError, match="coherence_threshold"):
            self.algo.validate_parameters(coherence_threshold=0.0)

    def test_invalid_window_type(self):
        with pytest.raises(ValueError, match="window_type"):
            self.algo.validate_parameters(window_type="kaiser")

    def test_invalid_lag_window(self):
        with pytest.raises(ValueError, match="lag_window"):
            self.algo.validate_parameters(lag_window="boxcar")

    def test_invalid_nperseg(self):
        with pytest.raises(ValueError, match="nperseg"):
            self.algo.validate_parameters(nperseg=4)

    def test_invalid_max_lag(self):
        with pytest.raises(ValueError, match="max_lag"):
            self.algo.validate_parameters(max_lag=0)


class TestCorrelationEstimator:
    @pytest.fixture
    def result(self):
        u, y = make_siso_data()
        algo = FrequencyDomainAlgorithm()
        return algo.identify(y=y, u=u)

    def test_returns_state_space_model_with_placeholders(self, result):
        assert isinstance(result, StateSpaceModel)
        assert result.A.shape == (1, 1)
        assert result.n == 1

    def test_estimator_and_dimensions_recorded(self, result):
        info = result.identification_info
        assert info["estimator"] == "correlation"
        assert info["n_inputs"] == 1
        assert info["n_outputs"] == 1

    def test_frequency_response_fields(self, result):
        fr = result.identification_info["frequency_response"]
        for key in [
            "omega",
            "omega_real",
            "freq_hz",
            "G_raw",
            "G_smooth",
            "magnitude_db",
            "phase_deg",
            "coherence",
            "Phi_u",
            "Phi_y",
            "Phi_uy",
            "R_u",
            "R_uy",
            "tau",
        ]:
            assert key in fr, key
        assert np.iscomplexobj(fr["G_smooth"])
        assert np.isrealobj(fr["Phi_u"])
        assert np.iscomplexobj(fr["Phi_uy"])
        assert len(fr["omega"]) == len(fr["G_smooth"])

    def test_omega_symmetric_and_sorted(self, result):
        omega = result.identification_info["frequency_response"]["omega"]
        assert np.all(np.diff(omega) > 0)
        assert omega[0] < 0 < omega[-1]

    def test_magnitude_phase_consistent_with_G(self, result):
        fr = result.identification_info["frequency_response"]
        np.testing.assert_allclose(
            10 ** (fr["magnitude_db"] / 20),
            np.abs(fr["G_smooth"]) + 1e-12,
            rtol=1e-6,
        )

    def test_coherence_in_valid_range_and_informative(self, result):
        coh = result.identification_info["frequency_response"]["coherence"]
        assert np.all(coh >= 0) and np.all(coh <= 1)
        # must NOT be identically 1 (degenerate periodogram regression)
        assert np.std(coh) > 1e-3

    def test_frf_accuracy_against_true_system(self, result):
        fr = result.identification_info["frequency_response"]
        omega = fr["omega"]
        mask = (omega > 0.1) & (omega < 3.0) & (fr["coherence"] > 0.9)
        assert mask.sum() > 50
        err = median_relative_frf_error(
            fr["G_smooth"][mask], omega[mask], B_TRUE, A_TRUE
        )
        assert err < 0.05

    def test_max_lag_controls_resolution(self):
        u, y = make_siso_data()
        algo = FrequencyDomainAlgorithm()
        result = algo.identify(y=y, u=u, max_lag=100)
        fr = result.identification_info["frequency_response"]
        assert len(fr["omega"]) == 2 * 100 + 1

    def test_explicit_theta_of_delay_visible_in_phase(self):
        """A pure delay adds linear phase: estimate for delayed system must
        show steeper phase slope."""
        rng = np.random.default_rng(3)
        N = 4000
        u = rng.standard_normal(N)
        y_delayed = signal.lfilter(np.r_[np.zeros(5), 0.5], [1.0, -0.5], u)
        y_prompt = signal.lfilter([0.0, 0.5], [1.0, -0.5], u)
        algo = FrequencyDomainAlgorithm()
        fr_d = algo.identify(y=y_delayed, u=u).identification_info["frequency_response"]
        fr_p = algo.identify(y=y_prompt, u=u).identification_info["frequency_response"]
        omega = fr_d["omega"]
        i = np.argmin(np.abs(omega - 0.5))
        slope_d = np.angle(fr_d["G_smooth"][i])
        slope_p = np.angle(fr_p["G_smooth"][i])
        assert slope_d < slope_p  # extra delay = more negative phase

    def test_remove_mean_handles_dc_offset(self):
        u, y = make_siso_data()
        algo = FrequencyDomainAlgorithm()
        r_offset = algo.identify(y=y + 10.0, u=u + 5.0)
        fr = r_offset.identification_info["frequency_response"]
        omega = fr["omega"]
        mask = (omega > 0.1) & (omega < 3.0) & (fr["coherence"] > 0.9)
        err = median_relative_frf_error(
            fr["G_smooth"][mask], omega[mask], B_TRUE, A_TRUE
        )
        assert err < 0.05

    def test_data_taper_option_runs(self):
        u, y = make_siso_data()
        algo = FrequencyDomainAlgorithm()
        for wt in ["hann", "hamming", "blackman"]:
            result = algo.identify(y=y, u=u, window_type=wt)
            assert isinstance(result, StateSpaceModel)


class TestWelchEstimator:
    def test_siso_welch_accuracy(self):
        u, y = make_siso_data()
        algo = FrequencyDomainAlgorithm()
        result = algo.identify(y=y, u=u, fd_method="welch")
        info = result.identification_info
        assert info["estimator"] == "welch"
        fr = info["frequency_response"]
        omega = fr["omega"]
        mask = (omega > 0.1) & (omega < 3.0) & (fr["coherence"][:, 0] > 0.9)
        err = median_relative_frf_error(
            fr["G"][mask, 0, 0], omega[mask], B_TRUE, A_TRUE
        )
        assert err < 0.05

    def test_welch_output_shapes(self):
        u, y = make_siso_data(N=2000)
        result = FrequencyDomainAlgorithm().identify(
            y=y, u=u, fd_method="welch", nperseg=256
        )
        fr = result.identification_info["frequency_response"]
        F = len(fr["freq_hz"])
        assert fr["G"].shape == (F, 1, 1)
        assert fr["coherence"].shape == (F, 1)
        assert fr["nperseg"] == 256

    def test_nperseg_defaults_to_eighth_of_data(self):
        u, y = make_siso_data(N=2000)
        result = FrequencyDomainAlgorithm().identify(y=y, u=u, fd_method="welch")
        assert result.identification_info["frequency_response"]["nperseg"] == 250


class TestMIMOIdentification:
    @pytest.fixture
    def mimo_data(self):
        rng = np.random.default_rng(7)
        N = 8000
        u0 = rng.standard_normal(N)
        u1 = 0.6 * signal.lfilter([1, 0.5], [1], u0) + 0.8 * rng.standard_normal(N)
        u = np.vstack([u0, u1])
        b1, a1 = [0.0, 0.4, -0.2], [1.0, -0.5]
        y0 = (
            signal.lfilter(B_TRUE, A_TRUE, u0)
            + signal.lfilter(b1, a1, u1)
            + 0.05 * rng.standard_normal(N)
        )
        y1 = signal.lfilter([0.0, 0.8], [1.0, -0.6], u0) + 0.05 * rng.standard_normal(N)
        return u, np.vstack([y0, y1]), (b1, a1)

    def test_auto_selects_welch_for_mimo(self, mimo_data):
        u, y, _ = mimo_data
        result = FrequencyDomainAlgorithm().identify(y=y, u=u)
        info = result.identification_info
        assert info["estimator"] == "welch"
        assert info["n_inputs"] == 2
        assert info["n_outputs"] == 2

    def test_correlation_method_rejects_mimo(self, mimo_data):
        u, y, _ = mimo_data
        with pytest.raises(ValueError, match="SISO"):
            FrequencyDomainAlgorithm().identify(y=y, u=u, fd_method="correlation")

    def test_mimo_frf_shapes_and_accuracy(self, mimo_data):
        u, y, (b1, a1) = mimo_data
        result = FrequencyDomainAlgorithm().identify(y=y, u=u, nperseg=512)
        fr = result.identification_info["frequency_response"]
        F = len(fr["freq_hz"])
        assert fr["G"].shape == (F, 2, 2)
        assert fr["coherence"].shape == (F, 2)

        omega = fr["omega"]
        mask = (omega > 0.1) & (omega < 3.0)
        # y0 <- u0 through the reference system, y0 <- u1 through (b1, a1),
        # despite u0/u1 being correlated
        for out, inp, b, a in [
            (0, 0, B_TRUE, A_TRUE),
            (0, 1, b1, a1),
            (1, 0, [0.0, 0.8], [1.0, -0.6]),
        ]:
            err = median_relative_frf_error(fr["G"][mask, out, inp], omega[mask], b, a)
            assert err < 0.1, f"G[{out},{inp}]"

    def test_per_output_quality_metrics(self, mimo_data):
        u, y, _ = mimo_data
        result = FrequencyDomainAlgorithm().identify(y=y, u=u)
        q = result.identification_info["quality_metrics"]
        assert len(q["per_output"]) == 2
        for qo in q["per_output"]:
            assert 0 <= qo["mean_coherence"] <= 1
            assert "quality_label" in qo


class TestQualityAssessment:
    def test_metrics_present_and_consistent(self):
        u, y = make_siso_data()
        result = FrequencyDomainAlgorithm().identify(y=y, u=u)
        q = result.identification_info["quality_metrics"]
        for key in [
            "mean_coherence",
            "min_coherence",
            "max_coherence",
            "median_coherence",
            "fraction_reliable",
            "threshold",
            "quality_label",
            "is_reliable",
        ]:
            assert key in q, key
        assert q["min_coherence"] <= q["median_coherence"] <= q["max_coherence"]
        coh = result.identification_info["frequency_response"]["coherence"]
        assert np.isclose(q["mean_coherence"], np.mean(coh))

    def test_low_snr_degrades_quality(self):
        u, y_clean = make_siso_data(noise=0.0)
        _, y_noisy = make_siso_data(noise=3.0)
        algo = FrequencyDomainAlgorithm()
        q_clean = algo.identify(y=y_clean, u=u).identification_info["quality_metrics"]
        q_noisy = algo.identify(y=y_noisy, u=u).identification_info["quality_metrics"]
        assert q_noisy["mean_coherence"] < q_clean["mean_coherence"]
        assert q_clean["quality_label"] == "EXCELLENT"


class TestSystemIdentificationIntegration:
    def test_config_method_fd(self):
        u, y = make_siso_data()
        ident = SystemIdentification(SystemIdentificationConfig(method="FD"))
        model = ident.identify(y=y, u=u)
        assert model.identification_info["method"] == "FD"

    def test_kwargs_override(self):
        u, y = make_siso_data()
        ident = SystemIdentification(SystemIdentificationConfig(method="ARX"))
        model = ident.identify(y=y, u=u, method="FD", fd_method="welch")
        assert model.identification_info["estimator"] == "welch"

    def test_iddata_object(self):
        u, y = make_siso_data()
        df = pd.DataFrame({"u": u, "y": y})
        data = IDData(df, inputs=["u"], outputs=["y"], tsample=0.1)
        model = FrequencyDomainAlgorithm().identify(iddata=data)
        fr = model.identification_info["frequency_response"]
        assert model.ts == 0.1
        # Nyquist should reflect dt = 0.1 s
        assert np.isclose(np.max(fr["freq_hz"]), 5.0, rtol=0.05)

    def test_iddata_and_arrays_agree(self):
        u, y = make_siso_data()
        df = pd.DataFrame({"u": u, "y": y})
        data = IDData(df, inputs=["u"], outputs=["y"], tsample=1.0)
        algo = FrequencyDomainAlgorithm()
        fr_a = algo.identify(y=y, u=u).identification_info["frequency_response"]
        fr_b = algo.identify(iddata=data).identification_info["frequency_response"]
        np.testing.assert_allclose(fr_a["G_smooth"], fr_b["G_smooth"])


class TestEdgeCases:
    def test_short_data_raises(self):
        rng = np.random.default_rng(0)
        u = rng.standard_normal(50)
        y = rng.standard_normal(50)
        algo = FrequencyDomainAlgorithm()
        with pytest.raises(ValueError, match="at least"):
            algo.identify(y=y, u=u, fd_method="correlation")
        with pytest.raises(ValueError, match="at least"):
            algo.identify(y=np.vstack([y, y]), u=np.vstack([u, u]))

    def test_mismatched_lengths_raise(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="same number of samples"):
            FrequencyDomainAlgorithm().identify(
                y=rng.standard_normal(200), u=rng.standard_normal(150)
            )

    def test_nan_and_inf_raise(self):
        u, y = make_siso_data(N=500)
        algo = FrequencyDomainAlgorithm()
        y_nan = y.copy()
        y_nan[5] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            algo.identify(y=y_nan, u=u)
        u_inf = u.copy()
        u_inf[5] = np.inf
        with pytest.raises(ValueError, match="[Ii]nf"):
            algo.identify(y=y, u=u_inf)

    def test_invalid_dt_raises(self):
        u, y = make_siso_data(N=500)
        with pytest.raises(ValueError, match="positive"):
            FrequencyDomainAlgorithm().identify(y=y, u=u, dt=-1.0)

    def test_constant_signal_warns(self):
        u, _ = make_siso_data(N=500)
        with pytest.warns(RuntimeWarning, match="low variance"):
            FrequencyDomainAlgorithm().identify(y=np.ones(500), u=u)
