"""
Non-parametric frequency-domain system identification.

Estimates the frequency response function G(e^iw) directly from input-output
data without assuming a parametric model structure. Two estimators are
provided:

- "correlation": the classical correlation (Blackman-Tukey) method for SISO
  data — correlations R_u, R_uy are estimated, transformed to spectra, and
  the FRF is the spectral ratio, followed by Hamming smoothing.
- "welch": Welch segment-averaged cross-spectral density matrices with the
  H1 estimator. Works for SISO and MIMO data; correlated inputs are handled
  by solving the full Wiener-Hopf system per frequency.

References:
    - Ljung, L. (1999). System Identification: Theory for the User, ch. 6.
    - Pintelon & Schoukens (2012). System Identification: A Frequency
      Domain Approach.
"""

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ...utils.spectral_utils import (
    compute_coherence,
    compute_correlations_fft,
    compute_frequency_response,
    compute_output_spectrum,
    compute_spectra_from_correlation,
    create_hamming_window,
    create_window,
    denormalize_frequency,
    estimate_frf_mimo,
    extract_magnitude_phase,
    smooth_frequency_response,
    validate_signal_pair,
)
from ..base import IdentificationAlgorithm, StateSpaceModel, resolve_identification_data

if TYPE_CHECKING:
    from ..iddata import IDData

VALID_ESTIMATORS = ("auto", "correlation", "welch")
VALID_WINDOW_TYPES = ("none", "hann", "hamming", "blackman")
VALID_LAG_WINDOWS = ("none", "hann", "hamming", "blackman", "bartlett")


class FrequencyDomainAlgorithm(IdentificationAlgorithm):
    """
    Non-parametric frequency-domain identification (correlation / Welch H1).

    Because the result is a frequency response rather than a parametric
    model, the returned StateSpaceModel carries placeholder 1x1 matrices;
    the estimate lives in ``model.identification_info``:

        info["frequency_response"]  frequency grid, complex FRF, magnitude,
                                    phase, coherence, and the raw spectra
        info["quality_metrics"]     coherence-based quality assessment

    Parameters (all via kwargs, config-merged by SystemIdentification):
        fd_method: "auto" (default), "correlation", or "welch".
            "auto" picks "correlation" for SISO data and "welch" otherwise.
        max_lag: maximum correlation lag (correlation method;
            default max(32, min(N // 10, 512)))
        lag_window: lag window for the Blackman-Tukey estimate,
            "hamming" (default) / "hann" / "blackman" / "bartlett" / "none"
        smoothing_window: frequency-smoothing width in bins, odd (default 11);
            applied to the FRF (correlation) or the CSD estimates (welch)
        coherence_threshold: minimum acceptable coherence (default 0.8)
        window_type: data taper "none"/"hann"/"hamming"/"blackman" (default "none")
        remove_mean: remove DC before estimation (default True)
        nperseg: Welch segment length (default max(64, min(1024, N // 8)))
        welch_window: Welch window function (default "hann")
        noverlap: Welch segment overlap (default nperseg // 2)
        dt / tsample: sampling interval in seconds (default 1.0)
    """

    def validate_parameters(self, **kwargs) -> bool:
        """Validate algorithm-specific parameters."""
        fd_method = kwargs.get("fd_method", "auto")
        if fd_method not in VALID_ESTIMATORS:
            raise ValueError(
                f"fd_method must be one of {VALID_ESTIMATORS}, got {fd_method!r}"
            )

        smoothing_window = kwargs.get("smoothing_window", 11)
        if smoothing_window < 3:
            raise ValueError("smoothing_window must be >= 3")

        coherence_threshold = kwargs.get("coherence_threshold", 0.8)
        if not 0 < coherence_threshold <= 1:
            raise ValueError("coherence_threshold must be in (0, 1]")

        window_type = kwargs.get("window_type", "none")
        if window_type not in VALID_WINDOW_TYPES:
            raise ValueError(f"Unknown window_type: {window_type}")

        lag_window = kwargs.get("lag_window", "hamming")
        if lag_window not in VALID_LAG_WINDOWS:
            raise ValueError(f"Unknown lag_window: {lag_window}")

        nperseg = kwargs.get("nperseg")
        if nperseg is not None and nperseg < 8:
            raise ValueError("nperseg must be >= 8")

        max_lag = kwargs.get("max_lag")
        if max_lag is not None and max_lag < 1:
            raise ValueError("max_lag must be >= 1")

        return True

    def identify(
        self,
        y: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
        iddata: Optional["IDData"] = None,
        **kwargs,
    ) -> StateSpaceModel:
        """
        Perform non-parametric frequency-domain identification.

        Args:
            y: Output data, shape (l, N) or (N,)
            u: Input data, shape (m, N) or (N,)
            iddata: IDData object with input/output data
            **kwargs: Algorithm parameters (see class docstring)

        Returns:
            StateSpaceModel with placeholder matrices and the frequency
            response estimate in ``identification_info``.
        """
        self.validate_parameters(**kwargs)

        dt = float(kwargs.get("dt") or kwargs.get("tsample") or 1.0)
        y_arr, u_arr, dt = resolve_identification_data(y, u, iddata, tsample=dt)
        if dt <= 0:
            raise ValueError(f"Sampling interval must be positive, got {dt}")

        n_outputs, N = y_arr.shape
        n_inputs = u_arr.shape[0]

        if np.any(np.isnan(y_arr)) or np.any(np.isnan(u_arr)):
            raise ValueError("Input/output contains NaN values")
        if np.any(np.isinf(y_arr)) or np.any(np.isinf(u_arr)):
            raise ValueError("Input/output contains infinite values")

        fd_method = kwargs.get("fd_method", "auto")
        if fd_method == "auto":
            fd_method = "correlation" if n_inputs == 1 and n_outputs == 1 else "welch"
        if fd_method == "correlation" and (n_inputs > 1 or n_outputs > 1):
            raise ValueError(
                "fd_method='correlation' supports only SISO data; "
                "use fd_method='welch' for MIMO identification"
            )

        remove_mean = kwargs.get("remove_mean", True)
        if remove_mean:
            y_arr = y_arr - np.mean(y_arr, axis=1, keepdims=True)
            u_arr = u_arr - np.mean(u_arr, axis=1, keepdims=True)

        if fd_method == "correlation":
            results = self._identify_correlation(u_arr[0], y_arr[0], dt, N, **kwargs)
        else:
            results = self._identify_welch(u_arr, y_arr, dt, N, **kwargs)

        results["n_inputs"] = n_inputs
        results["n_outputs"] = n_outputs

        # Non-parametric methods do not produce state-space matrices;
        # placeholders keep the factory return contract.
        return StateSpaceModel(
            A=np.eye(1),
            B=np.zeros((1, 1)),
            C=np.zeros((1, 1)),
            D=np.zeros((1, 1)),
            K=np.zeros((1, 1)),
            Q=np.eye(1),
            R=np.eye(1),
            S=np.zeros((1, 1)),
            ts=dt,
            Vn=0.0,
            identification_info=results,
        )

    def _identify_correlation(
        self, u: np.ndarray, y: np.ndarray, dt: float, N: int, **kwargs
    ) -> dict[str, Any]:
        """
        Correlation (Blackman-Tukey) method for SISO data.

        1. Estimate R_u(tau), R_uy(tau) via FFT, truncated at max_lag
        2. Apply a lag window and transform to spectra Phi_u, Phi_uy, Phi_y
        3. FRF as the spectral ratio G = Phi_uy / Phi_u
        4. Hamming-window frequency smoothing
        5. Coherence and quality assessment

        Truncating the correlations well below N and windowing them is what
        makes this a consistent Blackman-Tukey estimator: at max_lag = N-1
        with no lag window the spectra degenerate to raw periodograms, the
        FRF estimate does not converge, and the coherence is identically 1.
        """
        u, y = validate_signal_pair(u, y, min_length=100)

        window_type = kwargs.get("window_type", "none")
        if window_type != "none":
            taper = create_window(N, window_type)
            u = u * taper
            y = y * taper

        max_lag = kwargs.get("max_lag")
        if max_lag is None:
            max_lag = max(32, min(N // 10, 512))
        max_lag = min(int(max_lag), N - 1)

        tau, R_u, R_uy = compute_correlations_fft(u, y, max_lag)

        lag_window_type = kwargs.get("lag_window", "hamming")
        if lag_window_type != "none":
            lag_window = create_window(len(R_u), lag_window_type)
            R_u_w = R_u * lag_window
            R_uy_w = R_uy * lag_window
        else:
            lag_window = None
            R_u_w, R_uy_w = R_u, R_uy

        Phi_u, Phi_uy, omega = compute_spectra_from_correlation(R_u_w, R_uy_w)
        Phi_y = compute_output_spectrum(y, max_lag, lag_window=lag_window)

        G_raw = compute_frequency_response(Phi_uy, Phi_u)

        smoothing_window = int(kwargs.get("smoothing_window", 11))
        window = create_hamming_window(smoothing_window, normalize=True)
        G_smooth = smooth_frequency_response(G_raw, window)

        coherence = compute_coherence(Phi_uy, Phi_u, Phi_y)
        coherence_threshold = kwargs.get("coherence_threshold", 0.8)
        quality_metrics = _assess_quality(coherence, coherence_threshold)

        magnitude_db, phase_deg = extract_magnitude_phase(G_smooth)
        omega_real, freq_hz = denormalize_frequency(omega, dt)

        return {
            "method": "FD",
            "estimator": "correlation",
            "frequency_response": {
                "omega": omega,
                "omega_real": omega_real,
                "freq_hz": freq_hz,
                "G_raw": G_raw,
                "G_smooth": G_smooth,
                "magnitude_db": magnitude_db,
                "phase_deg": phase_deg,
                "coherence": coherence,
                "Phi_u": Phi_u,
                "Phi_y": Phi_y,
                "Phi_uy": Phi_uy,
                "R_u": R_u,
                "R_uy": R_uy,
                "tau": tau,
            },
            "quality_metrics": quality_metrics,
        }

    def _identify_welch(
        self, u: np.ndarray, y: np.ndarray, dt: float, N: int, **kwargs
    ) -> dict[str, Any]:
        """Welch/H1 estimator for SISO or MIMO data."""
        if N < 100:
            raise ValueError(
                f"Need at least 100 samples for reliable identification, got {N}"
            )
        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(y))):
            raise ValueError("Input/output contains NaN or infinite values")

        nperseg = kwargs.get("nperseg")
        if nperseg is None:
            # N // 8 with 50% overlap gives ~15 segment averages; combined
            # with CSD frequency smoothing this keeps the MIMO solve
            # low-variance at low SNR without losing much resolution.
            nperseg = max(64, min(1024, N // 8))
        nperseg = min(int(nperseg), N)

        frf = estimate_frf_mimo(
            u,
            y,
            dt=dt,
            nperseg=nperseg,
            window=kwargs.get("welch_window", "hann"),
            noverlap=kwargs.get("noverlap"),
            smoothing_bins=int(kwargs.get("smoothing_window", 11)),
        )

        magnitude_db, phase_deg = extract_magnitude_phase(frf["G"])
        coherence_threshold = kwargs.get("coherence_threshold", 0.8)
        quality_metrics = _assess_quality(frf["coherence"], coherence_threshold)
        quality_metrics["per_output"] = [
            _assess_quality(frf["coherence"][:, j], coherence_threshold)
            for j in range(frf["coherence"].shape[1])
        ]

        return {
            "method": "FD",
            "estimator": "welch",
            "frequency_response": {
                "omega": frf["omega"],
                "omega_real": frf["omega"] / dt,
                "freq_hz": frf["freq_hz"],
                "G": frf["G"],
                "magnitude_db": magnitude_db,
                "phase_deg": phase_deg,
                "coherence": frf["coherence"],
                "S_uu": frf["S_uu"],
                "S_uy": frf["S_uy"],
                "S_yy": frf["S_yy"],
                "nperseg": nperseg,
            },
            "quality_metrics": quality_metrics,
        }


def _assess_quality(coherence: np.ndarray, threshold: float) -> dict[str, Any]:
    """Assess estimate quality from the coherence function."""
    mean_coh = float(np.mean(coherence))

    if mean_coh >= 0.9:
        quality_label = "EXCELLENT"
    elif mean_coh >= 0.8:
        quality_label = "GOOD"
    elif mean_coh >= 0.7:
        quality_label = "ACCEPTABLE"
    else:
        quality_label = "POOR"

    return {
        "mean_coherence": mean_coh,
        "min_coherence": float(np.min(coherence)),
        "max_coherence": float(np.max(coherence)),
        "median_coherence": float(np.median(coherence)),
        "fraction_reliable": float(np.mean(coherence >= threshold)),
        "threshold": threshold,
        "quality_label": quality_label,
        "is_reliable": mean_coh >= threshold,
    }
