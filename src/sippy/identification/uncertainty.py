from dataclasses import dataclass, replace

import numpy as np
from scipy import stats

from ..utils.spectral_utils import (
    compute_csd_segment_matrices,
    smooth_csd_along_frequency,
    solve_frf_from_spectra,
)


@dataclass(frozen=True)
class FrequencyResponseUncertainty:
    """Empirical Welch/H1 frequency-response uncertainty from validation data."""

    omega: np.ndarray
    frequency_hz: np.ndarray
    empirical_response: np.ndarray
    model_response: np.ndarray
    coherence: np.ndarray
    residual_spectrum: np.ndarray
    signal_to_noise_ratio: np.ndarray
    magnitude_standard_error_db: np.ndarray
    phase_standard_error_deg: np.ndarray
    confidence_levels: np.ndarray
    magnitude_confidence_lower_db: np.ndarray
    magnitude_confidence_upper_db: np.ndarray
    phase_confidence_lower_deg: np.ndarray
    phase_confidence_upper_deg: np.ndarray
    n_segments: int
    degrees_of_freedom: int
    nperseg: int
    window: str

    @property
    def empirical_magnitude_db(self) -> np.ndarray:
        return 20 * np.log10(np.maximum(np.abs(self.empirical_response), 1e-300))

    @property
    def model_magnitude_db(self) -> np.ndarray:
        return 20 * np.log10(np.maximum(np.abs(self.model_response), 1e-300))

    @property
    def empirical_phase_deg(self) -> np.ndarray:
        return np.degrees(np.unwrap(np.angle(self.empirical_response), axis=0))

    @property
    def model_phase_deg(self) -> np.ndarray:
        return np.degrees(np.unwrap(np.angle(self.model_response), axis=0))

    def magnitude_confidence_interval(
        self, level: float
    ) -> tuple[np.ndarray, np.ndarray]:
        index = self._confidence_index(level)
        return (
            self.magnitude_confidence_lower_db[index],
            self.magnitude_confidence_upper_db[index],
        )

    def phase_confidence_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        index = self._confidence_index(level)
        return (
            self.phase_confidence_lower_deg[index],
            self.phase_confidence_upper_deg[index],
        )

    def with_model_response(
        self, model_response: np.ndarray
    ) -> "FrequencyResponseUncertainty":
        response = np.asarray(model_response, dtype=complex)
        if response.shape != self.empirical_response.shape:
            raise ValueError(
                "Model response shape must match the empirical frequency response"
            )
        return replace(self, model_response=response)

    def _confidence_index(self, level: float) -> int:
        matches = np.flatnonzero(np.isclose(self.confidence_levels, level))
        if matches.size == 0:
            available = ", ".join(f"{item:g}" for item in self.confidence_levels)
            raise ValueError(
                f"Confidence level {level:g} was not calculated; available: {available}"
            )
        return int(matches[0])


def _jackknife_standard_error(values: np.ndarray) -> np.ndarray:
    count = values.shape[0]
    centered = values - np.mean(values, axis=0, keepdims=True)
    return np.sqrt((count - 1) / count * np.sum(centered**2, axis=0))


def estimate_frequency_response_uncertainty(
    u: np.ndarray,
    y: np.ndarray,
    *,
    dt: float = 1.0,
    nperseg: int | None = None,
    window: str = "hann",
    noverlap: int = 0,
    smoothing_bins: int = 5,
    confidence_levels: tuple[float, ...] = (0.68, 0.95),
) -> FrequencyResponseUncertainty:
    """Estimate H1 FRF confidence bands with a delete-one-segment jackknife."""
    u_array = np.atleast_2d(np.asarray(u, dtype=float))
    y_array = np.atleast_2d(np.asarray(y, dtype=float))
    if u_array.shape[1] != y_array.shape[1]:
        raise ValueError("Input and output must share the same number of samples")
    if u_array.shape[1] < 64:
        raise ValueError("Need at least 64 samples for frequency-response uncertainty")
    if noverlap != 0:
        raise ValueError(
            "Jackknife confidence intervals require non-overlapping Welch segments"
        )
    if nperseg is None:
        nperseg = max(16, min(1024, u_array.shape[1] // 8))
    if not isinstance(nperseg, (int, np.integer)) or nperseg < 8:
        raise ValueError("nperseg must be an integer of at least 8")
    if not isinstance(smoothing_bins, (int, np.integer)) or smoothing_bins < 1:
        raise ValueError("smoothing_bins must be a positive integer")

    levels = np.asarray(confidence_levels, dtype=float)
    if levels.ndim != 1 or levels.size == 0:
        raise ValueError(
            "confidence_levels must be a nonempty one-dimensional sequence"
        )
    if not np.all(np.isfinite(levels)) or np.any((levels <= 0) | (levels >= 1)):
        raise ValueError(
            "confidence_levels must contain values strictly between 0 and 1"
        )
    if np.unique(levels).size != levels.size:
        raise ValueError("confidence_levels must not contain duplicates")

    freqs, S_uu_segments, S_uy_segments, S_yy_segments = compute_csd_segment_matrices(
        u_array,
        y_array,
        dt=dt,
        nperseg=int(nperseg),
        window=window,
        noverlap=0,
    )
    n_segments = S_uu_segments.shape[0]
    minimum_segments = max(4, u_array.shape[0] + 2)
    if n_segments < minimum_segments:
        raise ValueError(
            f"Need at least {minimum_segments} non-overlapping Welch segments "
            "for jackknife uncertainty and a full-rank leave-one-out input "
            f"spectrum; got {n_segments}. Reduce nperseg or collect more data."
        )

    def average_and_smooth(segments: np.ndarray) -> np.ndarray:
        averaged = np.mean(segments, axis=0)
        return smooth_csd_along_frequency(averaged, int(smoothing_bins))

    S_uu = average_and_smooth(S_uu_segments)
    S_uy = average_and_smooth(S_uy_segments)
    S_yy = average_and_smooth(S_yy_segments)
    response, coherence, residual_spectrum, snr = solve_frf_from_spectra(
        S_uu, S_uy, S_yy
    )

    total_uu = np.sum(S_uu_segments, axis=0)
    total_uy = np.sum(S_uy_segments, axis=0)
    total_yy = np.sum(S_yy_segments, axis=0)
    leave_one_out = []
    for segment in range(n_segments):
        loo_uu = (total_uu - S_uu_segments[segment]) / (n_segments - 1)
        loo_uy = (total_uy - S_uy_segments[segment]) / (n_segments - 1)
        loo_yy = (total_yy - S_yy_segments[segment]) / (n_segments - 1)
        loo_uu = smooth_csd_along_frequency(loo_uu, int(smoothing_bins))
        loo_uy = smooth_csd_along_frequency(loo_uy, int(smoothing_bins))
        loo_yy = smooth_csd_along_frequency(loo_yy, int(smoothing_bins))
        loo_response, _, _, _ = solve_frf_from_spectra(loo_uu, loo_uy, loo_yy)
        leave_one_out.append(loo_response)
    jackknife_response = np.asarray(leave_one_out)

    floor = 1e-300
    magnitude_db = 20 * np.log10(np.maximum(np.abs(response), floor))
    jackknife_magnitude_db = 20 * np.log10(
        np.maximum(np.abs(jackknife_response), floor)
    )
    magnitude_se = _jackknife_standard_error(jackknife_magnitude_db)

    phase_deg = np.degrees(np.unwrap(np.angle(response), axis=0))
    reference_phase = np.exp(-1j * np.angle(response))
    phase_deviation = np.degrees(
        np.angle(
            jackknife_response / np.maximum(np.abs(response), floor) * reference_phase
        )
    )
    phase_se = _jackknife_standard_error(phase_deviation)

    degrees_of_freedom = n_segments - 1
    critical = stats.t.ppf((1.0 + levels) / 2.0, degrees_of_freedom)
    expansion = (slice(None),) + (None,) * magnitude_se.ndim
    magnitude_half_width = critical[expansion] * magnitude_se[None, ...]
    phase_half_width = critical[expansion] * phase_se[None, ...]

    return FrequencyResponseUncertainty(
        omega=2 * np.pi * freqs,
        frequency_hz=freqs,
        empirical_response=response,
        model_response=response.copy(),
        coherence=coherence,
        residual_spectrum=residual_spectrum,
        signal_to_noise_ratio=snr,
        magnitude_standard_error_db=magnitude_se,
        phase_standard_error_deg=phase_se,
        confidence_levels=levels,
        magnitude_confidence_lower_db=magnitude_db[None, ...] - magnitude_half_width,
        magnitude_confidence_upper_db=magnitude_db[None, ...] + magnitude_half_width,
        phase_confidence_lower_deg=phase_deg[None, ...] - phase_half_width,
        phase_confidence_upper_deg=phase_deg[None, ...] + phase_half_width,
        n_segments=n_segments,
        degrees_of_freedom=degrees_of_freedom,
        nperseg=int(nperseg),
        window=window,
    )
