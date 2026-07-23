"""High-pass detrending filter for process data."""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.signal import filtfilt, firwin, kaiserord

from .base import FilterConfig, IFilter


class HighPassFilter(IFilter):
    """
    High-pass filter for removing low-frequency trends from time series data.

    A zero-phase Kaiser FIR estimates the slow low-pass trend. The returned
    signal is the processed input minus that trend, matching the historical
    SIPPY detrending contract.
    """

    _MAX_AUTO_TAPS = 1001

    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Initialize high-pass filter.

        Parameters:
        -----------
        config : FilterConfig, optional
            Filter configuration parameters
        """
        super().__init__(config)

        self._ripple_db = 65.0
        self._width_factor = 0.5
        self._last_design: dict[str, float | int] | None = None

    def apply_filter(
        self,
        data: pd.DataFrame,
        tss: Optional[float] = None,
        multiplier: Optional[float] = None,
        slices: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Apply high-pass filter to input data.

        Parameters:
        -----------
        data : pd.DataFrame
            Input time series data to filter
        tss : float, optional
            Time to steady state in minutes (overrides config)
        multiplier : float, optional
            Filter timestep multiplier for filter cutoff calculation
            (overrides config)
        slices : dict, optional
            Data slice definitions for bad data handling (overrides config)
        **kwargs
            Additional parameters (ignored for this filter)

        Returns:
        --------
        pd.DataFrame
            High-pass filtered data

        Raises:
        ------
        ValueError
            If data validation or filtering parameters are invalid
        """
        self._validate_input(data)
        selected_slices = self._resolve_slices(slices)
        processed_data = self._process_slices(data, selected_slices)
        sample_time = self._calculate_sampling_time(processed_data)
        sampling_frequency = 1.0 / sample_time
        nyquist_frequency = sampling_frequency / 2.0
        mult_factor = self.config.multiplier if multiplier is None else multiplier
        if (
            isinstance(mult_factor, bool)
            or not np.isfinite(mult_factor)
            or mult_factor <= 0
        ):
            raise ValueError("Multiplier must be positive and finite")

        tss_minutes = self.config.tss if tss is None else tss
        if tss_minutes is None:
            tss_minutes = 1.0
        if (
            isinstance(tss_minutes, bool)
            or not np.isfinite(tss_minutes)
            or tss_minutes <= 0
        ):
            raise ValueError("Time to steady state must be positive and finite")

        cutoff = self.config.cutoff
        if cutoff is None:
            cutoff = 1.0 / (2.0 * float(tss_minutes) * 60.0 * float(mult_factor))
        if not 0.0 < cutoff < nyquist_frequency:
            raise ValueError(
                f"Cutoff frequency must be between 0 and Nyquist "
                f"({nyquist_frequency:g} Hz); got {cutoff:g} Hz"
            )

        transition_width = min(
            self._width_factor * cutoff,
            self._width_factor * (nyquist_frequency - cutoff),
        )
        normalized_width = transition_width / nyquist_frequency
        requested_taps, beta = kaiserord(self._ripple_db, normalized_width)
        if self.config.order is None:
            maximum_taps = min(
                self._MAX_AUTO_TAPS,
                max(3, (len(processed_data) - 1) // 2),
            )
            num_taps = min(requested_taps, maximum_taps)
        else:
            num_taps = self.config.order + 1
        if len(processed_data) <= 2 * num_taps:
            raise ValueError(
                f"High-pass filtering with {num_taps} taps requires more than "
                f"{2 * num_taps} samples; got {len(processed_data)}"
            )

        try:
            coefficients = firwin(
                numtaps=num_taps,
                cutoff=cutoff,
                window=("kaiser", beta),
                pass_zero=True,
                fs=sampling_frequency,
            )
            pad_length = min(3 * (num_taps - 1), len(processed_data) - 1)
            trend_values = filtfilt(
                coefficients,
                1.0,
                processed_data.to_numpy(),
                axis=0,
                padlen=pad_length,
            )
        except Exception as error:
            raise ValueError(
                f"Failed to apply high-pass detrending filter: {error}"
            ) from error
        trend = pd.DataFrame(
            trend_values,
            index=processed_data.index,
            columns=processed_data.columns,
        )
        self._last_design = {
            "sample_time_seconds": sample_time,
            "sampling_frequency_hz": sampling_frequency,
            "nyquist_frequency_hz": nyquist_frequency,
            "cutoff_frequency_hz": cutoff,
            "transition_width_hz": transition_width,
            "requested_taps": requested_taps,
            "order": num_taps - 1,
            "num_taps": num_taps,
            "pad_length": pad_length,
        }

        # Handle slices for trend restoration
        if selected_slices:
            for slice_info in selected_slices.values():
                if slice_info["type"] == "bad":
                    if slice_info.get("isGlobal", False):
                        # Restore original data for bad slices (global)
                        start, end = slice_info["start"], slice_info["end"]
                        trend.iloc[start:end, :] = processed_data.iloc[start:end, :]
                    else:
                        # Restore only specified tags
                        if "tags" in slice_info:
                            start, end = slice_info["start"], slice_info["end"]
                            valid_tags = [
                                tag
                                for tag in slice_info["tags"]
                                if tag in trend.columns
                            ]
                            for tag in valid_tags:
                                col_idx = trend.columns.get_loc(tag)
                                trend.iloc[start:end, col_idx] = processed_data.iloc[
                                    start:end, col_idx
                                ]

        self.data_manager.add_data("input", data, type="original")
        self.data_manager.add_data("trend", trend, type="filtered_trend")
        output = processed_data - trend
        self.data_manager.add_data("output", output, type="highpass_output")
        return output

    def get_filter_info(self) -> dict:
        """
        Get information about this filter instance.

        Returns:
        --------
        dict
            Filter parameters and design information
        """
        return {
            "type": "HighPassFilter",
            "ripple_db": self._ripple_db,
            "width_factor": self._width_factor,
            "description": "Zero-phase Kaiser FIR trend subtraction",
            "suitable_for": "Removing low-frequency drift from process data",
            "design": None if self._last_design is None else self._last_design.copy(),
        }
