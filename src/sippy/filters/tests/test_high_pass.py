import numpy as np
import pandas as pd
import pytest

from sippy.filters import FilterConfig, FilterFactory


def component_amplitude(
    values: np.ndarray, frequency: float, sample_time: float
) -> float:
    time = np.arange(values.size) * sample_time
    basis = np.column_stack(
        [np.sin(2 * np.pi * frequency * time), np.cos(2 * np.pi * frequency * time)]
    )
    coefficients, *_ = np.linalg.lstsq(basis, values, rcond=None)
    return float(np.linalg.norm(coefficients))


@pytest.mark.parametrize("sample_time", [10.0, 60.0])
def test_high_pass_removes_slow_trend_and_preserves_dynamic_signal(sample_time):
    sample_count = 4096
    low_frequency = 5e-5
    high_frequency = 2e-3
    time = np.arange(sample_count) * sample_time
    low_component = 2.0 * np.sin(2 * np.pi * low_frequency * time)
    high_component = 0.4 * np.sin(2 * np.pi * high_frequency * time)
    index = pd.date_range(
        "2026-01-01",
        periods=sample_count,
        freq=pd.Timedelta(seconds=sample_time),
    )
    data = pd.DataFrame({"signal": low_component + high_component}, index=index)
    original = data.copy(deep=True)
    detrending_filter = FilterFactory.create(
        "highpass",
        FilterConfig(cutoff=2.5e-4, order=100),
    )

    result = detrending_filter.apply_filter(data)
    trend = detrending_filter.data_manager.get_data("trend")

    pd.testing.assert_frame_equal(data, original)
    assert result.index.equals(data.index)
    assert result.columns.equals(data.columns)
    assert trend.index.equals(data.index)
    assert trend.columns.equals(data.columns)
    assert (
        component_amplitude(result["signal"].to_numpy(), low_frequency, sample_time)
        < 0.2
    )
    assert (
        component_amplitude(result["signal"].to_numpy(), high_frequency, sample_time)
        > 0.3
    )
    assert (
        component_amplitude(trend["signal"].to_numpy(), low_frequency, sample_time)
        > 1.8
    )
    assert (
        component_amplitude(trend["signal"].to_numpy(), high_frequency, sample_time)
        < 0.1
    )


def test_high_pass_uses_tss_in_minutes_and_reports_physical_design_units():
    index = pd.date_range("2026-01-01", periods=1200, freq="1min")
    data = pd.DataFrame({"signal": np.linspace(0.0, 1.0, len(index))}, index=index)
    detrending_filter = FilterFactory.create(
        "highpass",
        FilterConfig(tss=120, multiplier=3, order=100),
    )

    detrending_filter.apply_filter(data)
    design = detrending_filter.get_filter_info()["design"]

    assert design["sample_time_seconds"] == 60.0
    assert design["sampling_frequency_hz"] == pytest.approx(1.0 / 60.0)
    assert design["nyquist_frequency_hz"] == pytest.approx(1.0 / 120.0)
    assert design["cutoff_frequency_hz"] == pytest.approx(1.0 / (2 * 120 * 60 * 3))
    assert design["order"] == 100
    assert design["num_taps"] == 101


def test_high_pass_bounds_automatic_tap_count():
    index = pd.date_range("2026-01-01", periods=2500, freq="1min")
    data = pd.DataFrame({"signal": np.arange(len(index), dtype=float)}, index=index)
    detrending_filter = FilterFactory.create(
        "highpass",
        FilterConfig(tss=120, multiplier=3),
    )

    detrending_filter.apply_filter(data)
    design = detrending_filter.get_filter_info()["design"]

    assert design["requested_taps"] > design["num_taps"]
    assert design["num_taps"] == 1001


def test_high_pass_rejects_tap_count_too_large_for_record():
    index = pd.date_range("2026-01-01", periods=200, freq="1min")
    data = pd.DataFrame({"signal": np.arange(len(index), dtype=float)}, index=index)
    detrending_filter = FilterFactory.create(
        "highpass",
        FilterConfig(cutoff=1e-3, order=100),
    )

    with pytest.raises(ValueError, match="requires more than 202 samples"):
        detrending_filter.apply_filter(data)


def test_high_pass_preserves_slice_alignment_and_zeros_bad_channel_span():
    index = pd.date_range("2026-01-01", periods=800, freq="1min")
    data = pd.DataFrame(
        {
            "first": np.sin(np.linspace(0.0, 20.0, len(index))),
            "second": np.cos(np.linspace(0.0, 20.0, len(index))),
        },
        index=index,
    )
    original = data.copy(deep=True)
    slices = {
        "invalid": {
            "type": "bad",
            "isGlobal": False,
            "start": 100,
            "end": 110,
            "tags": ["first"],
        }
    }
    detrending_filter = FilterFactory.create(
        "highpass",
        FilterConfig(cutoff=5e-4, order=50, slices=slices),
    )

    result = detrending_filter.apply_filter(data)

    pd.testing.assert_frame_equal(data, original)
    assert result.index.equals(data.index)
    assert result.columns.equals(data.columns)
    np.testing.assert_allclose(result["first"].iloc[100:110], 0.0)
    assert np.any(np.abs(result["second"].iloc[100:110]) > 0.0)


def test_high_pass_handles_bad_channel_span_at_start_of_record():
    index = pd.date_range("2026-01-01", periods=800, freq="1min")
    data = pd.DataFrame(
        {
            "first": np.sin(np.linspace(0.0, 20.0, len(index))),
            "second": np.cos(np.linspace(0.0, 20.0, len(index))),
        },
        index=index,
    )
    slices = {
        "startup": {
            "type": "bad",
            "isGlobal": False,
            "start": 0,
            "end": 10,
            "tags": ["first"],
        }
    }
    detrending_filter = FilterFactory.create(
        "highpass",
        FilterConfig(cutoff=5e-4, order=50, slices=slices),
    )

    result = detrending_filter.apply_filter(data)

    assert np.isfinite(result.to_numpy()).all()
    np.testing.assert_allclose(result["first"].iloc[:10], 0.0)
    assert np.any(np.abs(result["first"].iloc[10:]) > 0.0)


@pytest.mark.parametrize("filter_name", ["highpass", "difference", "zeromean", "none"])
def test_explicit_empty_slices_override_configured_slices(filter_name):
    index = pd.date_range("2026-01-01", periods=800, freq="1min")
    data = pd.DataFrame(
        {
            "first": np.sin(np.linspace(0.0, 20.0, len(index))),
            "second": np.cos(np.linspace(0.0, 20.0, len(index))),
        },
        index=index,
    )
    slices = {
        "invalid": {
            "type": "bad",
            "isGlobal": False,
            "start": 100,
            "end": 110,
            "tags": ["first"],
        }
    }
    design = {"cutoff": 5e-4, "order": 50} if filter_name == "highpass" else {}
    configured_filter = FilterFactory.create(
        filter_name,
        FilterConfig(slices=slices, **design),
    )
    baseline_filter = FilterFactory.create(filter_name, FilterConfig(**design))

    result = configured_filter.apply_filter(data, slices={})
    expected = baseline_filter.apply_filter(data)

    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "configuration",
    [
        {"cutoff": 0.0},
        {"cutoff": np.inf},
        {"order": 0},
        {"order": True},
        {"tss": 0.0},
        {"multiplier": np.nan},
    ],
)
def test_filter_config_rejects_invalid_design_values(configuration):
    with pytest.raises(ValueError):
        FilterConfig(**configuration)
