"""
Shared spectral analysis utilities for frequency-domain system identification.

This module provides the spectral estimation building blocks used by the
non-parametric frequency-domain identification algorithm and by model
uncertainty analysis:

- FFT-based auto/cross-correlation estimation
- Blackman-Tukey style spectra from correlations (SISO correlation method)
- Welch power/cross spectra (scipy-backed)
- MIMO cross-spectral density matrices and frequency-response-matrix
  estimation (H1 estimator, handles correlated inputs)
- Coherence (ordinary and multiple), spectral smoothing, windowing helpers
"""

import warnings

import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq, irfft, rfft, rfftfreq


def compute_power_spectrum_welch(
    x: np.ndarray, dt: float = 1.0, nperseg: int = 1024, window: str = "hann"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density using Welch's method.

    Parameters
    ----------
    x : ndarray
        Input signal (1D array)
    dt : float
        Sampling interval (seconds)
    nperseg : int
        Length of each segment
    window : str
        Window function ('hann', 'hamming', 'blackman', ...)

    Returns
    -------
    freqs : ndarray
        Frequency array (Hz)
    Pxx : ndarray
        Power spectral density
    """
    freqs, Pxx = scipy_signal.welch(x, fs=1 / dt, nperseg=nperseg, window=window)
    return freqs, Pxx


def compute_cross_spectrum_welch(
    u: np.ndarray,
    y: np.ndarray,
    dt: float = 1.0,
    nperseg: int = 1024,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute cross-spectral density using Welch's method.

    Parameters
    ----------
    u : ndarray
        Input signal
    y : ndarray
        Output signal
    dt : float
        Sampling interval (seconds)
    nperseg : int
        Length of each segment
    window : str
        Window function

    Returns
    -------
    freqs : ndarray
        Frequency array (Hz)
    Puy : ndarray
        Complex cross-spectral density E[conj(U) Y]
    """
    freqs, Puy = scipy_signal.csd(u, y, fs=1 / dt, nperseg=nperseg, window=window)
    return freqs, Puy


def autocorrelation_fft(x: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """
    Biased autocorrelation estimate R_x(tau) for tau in [-max_lag, max_lag].

    Uses real FFTs with zero-padding to avoid circular artifacts.

    Parameters
    ----------
    x : ndarray
        Signal (1D array)
    max_lag : int, optional
        Maximum lag (default: len(x) - 1)

    Returns
    -------
    R_x : ndarray
        Autocorrelation, length 2*max_lag + 1, centred on lag 0
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    if max_lag is None:
        max_lag = N - 1
    max_lag = min(max_lag, N - 1)

    x_fft = rfft(x, n=2 * N)
    R_full = irfft(x_fft * np.conj(x_fft), n=2 * N) / N
    return np.concatenate([R_full[-max_lag:], R_full[: max_lag + 1]])


def compute_correlations_fft(
    u: np.ndarray, y: np.ndarray, max_lag: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute input autocorrelation and input-output cross-correlation via FFT.

    O(N log N) with zero-padding to avoid circular correlation artifacts.
    Estimates are biased (normalized by N), the standard choice for
    spectral estimation because it guarantees a positive semidefinite
    correlation sequence.

    Parameters
    ----------
    u : ndarray
        Input signal
    y : ndarray
        Output signal
    max_lag : int, optional
        Maximum lag to compute (default: len(u) - 1)

    Returns
    -------
    tau : ndarray
        Lag vector [-max_lag, ..., 0, ..., +max_lag]
    R_u : ndarray
        Input autocorrelation
    R_uy : ndarray
        Input-output cross-correlation R_uy(tau) = E[u(t) y(t + tau)]
    """
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)
    N = len(u)
    if max_lag is None:
        max_lag = N - 1
    max_lag = min(max_lag, N - 1)

    u_fft = rfft(u, n=2 * N)
    y_fft = rfft(y, n=2 * N)

    # conj on the first factor so that positive lags mean "y lags u":
    # R_uy(tau) = E[u(t) y(t + tau)] peaks at tau > 0 for causal systems.
    R_u_full = irfft(np.conj(u_fft) * u_fft, n=2 * N) / N
    R_uy_full = irfft(np.conj(u_fft) * y_fft, n=2 * N) / N

    tau = np.arange(-max_lag, max_lag + 1)
    R_u = np.concatenate([R_u_full[-max_lag:], R_u_full[: max_lag + 1]])
    R_uy = np.concatenate([R_uy_full[-max_lag:], R_uy_full[: max_lag + 1]])

    return tau, R_u, R_uy


def compute_spectra_from_correlation(
    R_u: np.ndarray, R_uy: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute power and cross spectra from correlations (Blackman-Tukey).

    Theory:
        Phi_u(w)  = sum_tau R_u(tau)  exp(-i w tau)   [real, even]
        Phi_uy(w) = sum_tau R_uy(tau) exp(-i w tau)   [complex]

    Parameters
    ----------
    R_u : ndarray
        Input autocorrelation (centred on lag 0)
    R_uy : ndarray
        Input-output cross-correlation (centred on lag 0)

    Returns
    -------
    Phi_u : ndarray
        Input power spectrum (real)
    Phi_uy : ndarray
        Cross spectrum (complex)
    omega : ndarray
        Normalized frequency vector, ascending in [-pi, pi)
    """
    N_fft = len(R_u)

    # The correlations are centred on lag 0; rotate so lag 0 sits at index 0
    # as the FFT expects, otherwise every spectrum picks up a linear phase
    # exp(i*omega*max_lag).
    Phi_u_raw = fft(np.fft.ifftshift(R_u), n=N_fft)
    Phi_uy_raw = fft(np.fft.ifftshift(R_uy), n=N_fft)

    omega = fftfreq(N_fft, d=1.0) * 2 * np.pi

    idx = np.argsort(omega)
    omega = omega[idx]
    Phi_u = np.real(Phi_u_raw[idx])
    Phi_uy = Phi_uy_raw[idx]

    return Phi_u, Phi_uy, omega


def compute_output_spectrum(
    y: np.ndarray,
    max_lag: int | None = None,
    lag_window: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute output power spectrum from the output autocorrelation.

    Uses the same Blackman-Tukey construction and ascending-frequency
    ordering as :func:`compute_spectra_from_correlation`, so the result is
    sample-aligned with Phi_u/Phi_uy for coherence computation.

    Parameters
    ----------
    y : ndarray
        Output signal
    max_lag : int, optional
        Maximum correlation lag (default: len(y) - 1). The returned
        spectrum has length 2*max_lag + 1.
    lag_window : ndarray, optional
        Lag window of length 2*max_lag + 1 applied to the autocorrelation
        before the transform (use the same window as for Phi_u/Phi_uy).

    Returns
    -------
    Phi_y : ndarray
        Output power spectrum, ascending frequency order
    """
    R_y = autocorrelation_fft(y, max_lag)
    if lag_window is not None:
        R_y = R_y * lag_window
    N_fft = len(R_y)
    Phi_y_raw = np.real(fft(np.fft.ifftshift(R_y), n=N_fft))
    omega = fftfreq(N_fft, d=1.0)
    return Phi_y_raw[np.argsort(omega)]


def compute_frequency_response(
    cross_spectrum: np.ndarray, input_spectrum: np.ndarray
) -> np.ndarray:
    """
    Estimate frequency response from spectra: G(e^iw) = Phi_uy(w) / Phi_u(w).

    Parameters
    ----------
    cross_spectrum : ndarray
        Input-output cross spectrum (complex or real)
    input_spectrum : ndarray
        Input power spectrum (real)

    Returns
    -------
    G : ndarray
        Complex frequency response
    """
    epsilon = max(1e-12 * np.max(np.abs(input_spectrum)), 1e-300)
    return cross_spectrum / (input_spectrum + epsilon)


def compute_coherence(
    cross_spectrum: np.ndarray,
    input_spectrum: np.ndarray,
    output_spectrum: np.ndarray,
) -> np.ndarray:
    """
    Compute coherence function gamma^2(w) = |Phi_uy|^2 / (Phi_u * Phi_y).

    gamma^2 = 1 indicates a perfect linear noise-free relationship;
    lower values indicate noise, nonlinearity, or unmeasured disturbances.
    All three spectra must share the same frequency ordering.

    Parameters
    ----------
    cross_spectrum : ndarray
        Input-output cross spectrum (complex)
    input_spectrum : ndarray
        Input power spectrum (real)
    output_spectrum : ndarray
        Output power spectrum (real)

    Returns
    -------
    coherence : ndarray
        Coherence gamma^2(w), clipped to [0, 1]
    """
    min_len = min(len(cross_spectrum), len(input_spectrum), len(output_spectrum))
    cross_spectrum = cross_spectrum[:min_len]
    input_spectrum = input_spectrum[:min_len]
    output_spectrum = output_spectrum[:min_len]

    numerator = np.abs(cross_spectrum) ** 2
    denominator = input_spectrum * output_spectrum

    epsilon = max(1e-12 * np.max(denominator), 1e-300)
    coherence = numerator / (denominator + epsilon)

    return np.clip(coherence, 0.0, 1.0)


def create_window(N: int, window_type: str) -> np.ndarray:
    """
    Create tapering window to reduce FFT leakage.

    Parameters
    ----------
    N : int
        Window length
    window_type : str
        Window type ('hann', 'hamming', 'blackman', 'none')

    Returns
    -------
    window : ndarray
        Window function
    """
    if window_type == "hann":
        return np.hanning(N)
    elif window_type == "hamming":
        return np.hamming(N)
    elif window_type == "blackman":
        return np.blackman(N)
    elif window_type == "bartlett":
        return np.bartlett(N)
    elif window_type == "none":
        return np.ones(N)
    else:
        raise ValueError(f"Unknown window_type: {window_type}")


def create_hamming_window(window_size: int, normalize: bool = True) -> np.ndarray:
    """
    Create a (optionally unit-sum) Hamming window for spectral smoothing.

    Parameters
    ----------
    window_size : int
        Size of window (made odd for symmetry)
    normalize : bool
        Whether to normalize window to sum to 1

    Returns
    -------
    window : ndarray
        Hamming window
    """
    if window_size % 2 == 0:
        window_size += 1

    window = np.hamming(window_size)
    if normalize:
        window = window / np.sum(window)

    return window


def smooth_frequency_response(G: np.ndarray, window: np.ndarray) -> np.ndarray:
    """
    Apply frequency-averaging (Daniell-style) smoothing to a complex FRF.

    Smooths real and imaginary parts, which is well-behaved across phase
    wrap-arounds at +/- pi (smoothing the wrapped phase directly is not).
    Edge effects of the convolution are corrected by normalizing with the
    local window mass so the endpoints are unbiased.

    Parameters
    ----------
    G : ndarray
        Complex frequency response
    window : ndarray
        Non-negative smoothing window (need not be normalized)

    Returns
    -------
    G_smooth : ndarray
        Smoothed complex frequency response
    """
    window = np.asarray(window, dtype=float)
    coverage = np.convolve(np.ones(len(G)), window, mode="same")
    real_smooth = np.convolve(np.real(G), window, mode="same") / coverage
    imag_smooth = np.convolve(np.imag(G), window, mode="same") / coverage
    return real_smooth + 1j * imag_smooth


def extract_magnitude_phase(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract magnitude (dB) and unwrapped phase (degrees) from a complex FRF.

    For multidimensional inputs the phase is unwrapped along the first axis
    (the frequency axis).

    Parameters
    ----------
    G : ndarray
        Complex frequency response

    Returns
    -------
    magnitude_db : ndarray
        Magnitude in dB = 20*log10|G|
    phase_deg : ndarray
        Unwrapped phase in degrees
    """
    magnitude_db = 20 * np.log10(np.abs(G) + 1e-12)
    phase_deg = np.degrees(np.unwrap(np.angle(G), axis=0))
    return magnitude_db, phase_deg


def validate_signal_pair(
    u: np.ndarray, y: np.ndarray, min_length: int = 100, warn_constant: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate and flatten a SISO input-output signal pair.

    Parameters
    ----------
    u : ndarray
        Input signal
    y : ndarray
        Output signal
    min_length : int
        Minimum acceptable signal length
    warn_constant : bool
        Whether to warn if signals have very low variance

    Returns
    -------
    u_valid, y_valid : ndarray
        Validated 1D signals

    Raises
    ------
    ValueError
        If signals contain NaN/inf, differ in length, or are too short
    """
    u = np.asarray(u, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()

    if np.any(np.isnan(u)) or np.any(np.isnan(y)):
        raise ValueError("Input/output contains NaN values")
    if np.any(np.isinf(u)) or np.any(np.isinf(y)):
        raise ValueError("Input/output contains infinite values")

    if len(u) != len(y):
        raise ValueError(
            f"Input and output must have same length: {len(u)} != {len(y)}"
        )

    if len(u) < min_length:
        raise ValueError(
            f"Need at least {min_length} samples for reliable identification, "
            f"got {len(u)}"
        )

    if warn_constant:
        if np.std(u) < 1e-10 or np.std(y) < 1e-10:
            warnings.warn(
                "Input or output signal has very low variance (constant signal). "
                "Frequency response estimation may be unreliable.",
                RuntimeWarning,
            )

    return u, y


def denormalize_frequency(
    omega: np.ndarray, dt: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert normalized frequency (rad/sample) to physical units.

    Parameters
    ----------
    omega : ndarray
        Normalized frequency (rad/sample)
    dt : float
        Sampling interval (seconds)

    Returns
    -------
    omega_rad : ndarray
        Angular frequency (rad/s)
    freq_hz : ndarray
        Frequency (Hz)
    """
    omega_rad = omega / dt
    freq_hz = omega_rad / (2 * np.pi)
    return omega_rad, freq_hz


def compute_csd_matrices(
    u: np.ndarray,
    y: np.ndarray,
    dt: float = 1.0,
    nperseg: int = 1024,
    window: str = "hann",
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate the full set of Welch cross-spectral density matrices for
    multi-input multi-output data.

    The windowed FFT is computed once per input and output channel, then all
    channel-pair spectra are formed with batched contractions. This avoids the
    repeated FFT work incurred by separate Welch/CSD calls.

    Parameters
    ----------
    u : ndarray
        Inputs, shape (m, N) or (N,)
    y : ndarray
        Outputs, shape (l, N) or (N,)
    dt : float
        Sampling interval (seconds)
    nperseg : int
        Welch segment length
    window : str
        Window function name
    noverlap : int, optional
        Segment overlap (default: nperseg // 2)

    Returns
    -------
    freqs : ndarray
        One-sided frequency array (Hz), length F
    S_uu : ndarray, shape (F, m, m)
        Input CSD matrix, S_uu[f, i, j] = E[conj(U_i) U_j]
    S_uy : ndarray, shape (F, m, l)
        Input-output CSD matrix, S_uy[f, i, j] = E[conj(U_i) Y_j]
    S_yy : ndarray, shape (F, l)
        Output auto-spectra
    """
    freqs, S_uu_segments, S_uy_segments, S_yy_segments = compute_csd_segment_matrices(
        u,
        y,
        dt=dt,
        nperseg=nperseg,
        window=window,
        noverlap=noverlap,
    )
    return (
        freqs,
        np.mean(S_uu_segments, axis=0),
        np.mean(S_uy_segments, axis=0),
        np.mean(S_yy_segments, axis=0),
    )


def compute_csd_segment_matrices(
    u: np.ndarray,
    y: np.ndarray,
    dt: float = 1.0,
    nperseg: int = 1024,
    window: str = "hann",
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the individual Welch-segment spectra before averaging."""
    u = np.atleast_2d(np.asarray(u, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    if u.shape[1] != y.shape[1]:
        raise ValueError("Input and output must share the same number of samples")
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(y)):
        raise ValueError("Input and output must contain only finite values")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Sampling interval must be positive and finite, got {dt}")

    n_samples = u.shape[1]
    nperseg = int(nperseg)
    if nperseg < 1:
        raise ValueError("nperseg must be a positive integer")
    if nperseg > n_samples:
        warnings.warn(
            f"nperseg = {nperseg} is greater than input length = {n_samples}, "
            f"using nperseg = {n_samples}",
            UserWarning,
        )
        nperseg = n_samples
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = int(noverlap)
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError("noverlap must be non-negative and less than nperseg")

    taper = scipy_signal.get_window(window, nperseg)
    step = nperseg - noverlap

    def segment_ffts(x: np.ndarray) -> np.ndarray:
        segments = np.lib.stride_tricks.sliding_window_view(x, nperseg, axis=-1)[
            :, ::step, :
        ]
        segments = segments - np.mean(segments, axis=-1, keepdims=True)
        return rfft(segments * taper, axis=-1)

    U = segment_ffts(u)
    Y = segment_ffts(y)
    S_uu = np.einsum("msf,nsf->sfmn", np.conj(U), U, optimize=True)
    S_uy = np.einsum("msf,lsf->sfml", np.conj(U), Y, optimize=True)
    S_yy = np.transpose(np.abs(Y) ** 2, (1, 2, 0))

    fs = 1.0 / dt
    spectral_scale = np.full(U.shape[-1], 2.0 / (fs * np.sum(taper**2)))
    spectral_scale[0] *= 0.5
    if nperseg % 2 == 0:
        spectral_scale[-1] *= 0.5
    S_uu *= spectral_scale[None, :, None, None]
    S_uy *= spectral_scale[None, :, None, None]
    S_yy *= spectral_scale[None, :, None]
    freqs = rfftfreq(nperseg, dt)

    return freqs, S_uu, S_uy, S_yy


def smooth_csd_along_frequency(S: np.ndarray, smoothing_bins: int) -> np.ndarray:
    """
    Daniell-style smoothing of a CSD array along the frequency (first) axis.

    Averaging neighbouring frequency bins of the auto/cross spectra adds
    effective averages on top of Welch segment averaging, which sharply
    reduces the variance of FRF estimates derived from them (important for
    MIMO solves at low SNR). Edge bins are normalized by the local window
    mass so they stay unbiased.

    Parameters
    ----------
    S : ndarray
        Spectra with frequency on axis 0 (any trailing shape), real or complex
    smoothing_bins : int
        Hamming window width in bins (made odd); values < 3 return S unchanged

    Returns
    -------
    S_smooth : ndarray
        Smoothed spectra, same shape as S
    """
    if smoothing_bins is None or smoothing_bins < 3:
        return S

    from scipy.ndimage import convolve1d

    window = create_hamming_window(smoothing_bins, normalize=True)
    coverage = np.convolve(np.ones(S.shape[0]), window, mode="same")
    coverage = coverage.reshape((-1,) + (1,) * (S.ndim - 1))

    smoothed = convolve1d(np.real(S), window, axis=0, mode="constant")
    if np.iscomplexobj(S):
        smoothed = smoothed + 1j * convolve1d(
            np.imag(S), window, axis=0, mode="constant"
        )
    return smoothed / coverage


def solve_frf_from_spectra(
    S_uu: np.ndarray, S_uy: np.ndarray, S_yy: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the MIMO H1 response, coherence, residual spectrum, and SNR."""
    _, n_inputs, _ = S_uu.shape
    scale = np.mean(np.real(np.trace(S_uu, axis1=1, axis2=2))) / max(n_inputs, 1)
    regularization = (1e-12 * scale if scale > 0 else 1e-30) * np.eye(n_inputs)
    projected = np.linalg.solve(S_uu + regularization, S_uy)
    response = np.transpose(projected, (0, 2, 1))

    explained = np.real(np.einsum("fml,fml->fl", np.conj(S_uy), projected))
    residual = np.maximum(np.real(S_yy) - explained, 0.0)
    spectrum_scale = max(float(np.max(np.real(S_yy))), 1e-300)
    epsilon = 1e-12 * spectrum_scale
    coherence = np.clip(explained / (np.real(S_yy) + epsilon), 0.0, 1.0)
    snr = np.maximum(explained, 0.0) / (residual + epsilon)
    return response, coherence, residual, snr


def estimate_frf_mimo(
    u: np.ndarray,
    y: np.ndarray,
    dt: float = 1.0,
    nperseg: int = 1024,
    window: str = "hann",
    noverlap: int | None = None,
    smoothing_bins: int | None = None,
) -> dict:
    """
    Estimate the MIMO frequency response matrix with the H1 estimator.

    Solves, at every frequency, the Wiener-Hopf system

        S_uy(w) = S_uu(w) G(w)^T

    so correlated inputs are handled correctly (a per-channel SISO ratio
    Phi_uy/Phi_u is biased whenever inputs are cross-correlated). Also
    returns the multiple coherence per output,

        gamma_j^2(w) = S_uy[:, j]^H S_uu^{-1} S_uy[:, j] / S_yy[j],

    the MIMO generalization of ordinary coherence.

    Parameters
    ----------
    u : ndarray
        Inputs, shape (m, N) or (N,)
    y : ndarray
        Outputs, shape (l, N) or (N,)
    dt : float
        Sampling interval (seconds)
    nperseg : int
        Welch segment length
    window : str
        Window function name
    noverlap : int, optional
        Segment overlap (default: nperseg // 2)
    smoothing_bins : int, optional
        If >= 3, apply Daniell smoothing of this width to the CSD estimates
        along frequency before solving (variance reduction)

    Returns
    -------
    dict with keys:
        freq_hz : ndarray (F,) one-sided frequencies in Hz
        omega : ndarray (F,) normalized frequency (rad/sample)
        G : ndarray (F, l, m) complex frequency response matrix
        coherence : ndarray (F, l) multiple coherence per output
        residual_spectrum, signal_to_noise_ratio : ndarray (F, l)
        S_uu, S_uy, S_yy : the underlying (smoothed) CSD estimates
    """
    freqs, S_uu, S_uy, S_yy = compute_csd_matrices(
        u, y, dt=dt, nperseg=nperseg, window=window, noverlap=noverlap
    )
    if smoothing_bins is not None and smoothing_bins >= 3:
        S_uu = smooth_csd_along_frequency(S_uu, smoothing_bins)
        S_uy = smooth_csd_along_frequency(S_uy, smoothing_bins)
        S_yy = smooth_csd_along_frequency(S_yy, smoothing_bins)
    G, coherence, residual_spectrum, snr = solve_frf_from_spectra(S_uu, S_uy, S_yy)

    omega = 2 * np.pi * freqs * dt

    return {
        "freq_hz": freqs,
        "omega": omega,
        "G": G,
        "coherence": coherence,
        "residual_spectrum": residual_spectrum,
        "signal_to_noise_ratio": snr,
        "S_uu": S_uu,
        "S_uy": S_uy,
        "S_yy": S_yy,
    }
