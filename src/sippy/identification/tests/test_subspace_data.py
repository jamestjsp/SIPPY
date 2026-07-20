import numpy as np
import pytest

from sippy.identification.algorithms.subspace_data import (
    _numerical_rank,
    prepare_subspace_data,
)
from sippy.utils.simulation_utils import ordinate_sequence


def test_prepare_subspace_data_aligns_input_output_and_reference_blocks():
    rng = np.random.default_rng(41)
    y = rng.standard_normal((2, 80))
    u = rng.standard_normal((3, 80))
    reference = rng.standard_normal((2, 80))

    data = prepare_subspace_data(
        y,
        u,
        future_horizon=6,
        past_offset=4,
        reference=reference,
        scale=False,
    )

    expected_yf, expected_yp = ordinate_sequence(y, 6, 4)
    expected_uf, expected_up = ordinate_sequence(u, 6, 4)
    expected_rf, expected_rp = ordinate_sequence(reference, 6, 4)
    np.testing.assert_array_equal(data.future_outputs, expected_yf)
    np.testing.assert_array_equal(data.past_outputs, expected_yp)
    np.testing.assert_array_equal(data.future_inputs, expected_uf)
    np.testing.assert_array_equal(data.past_inputs, expected_up)
    np.testing.assert_array_equal(data.future_references, expected_rf)
    np.testing.assert_array_equal(data.past_references, expected_rp)
    np.testing.assert_array_equal(
        data.past_data,
        np.vstack((expected_up, expected_yp)),
    )
    assert data.usable_columns == 80 - 4 - 6 + 1

    ranked = prepare_subspace_data(
        y,
        u,
        future_horizon=4,
        past_offset=4,
        reference=reference,
        scale=False,
    )
    assert ranked.ranks.input_persistently_exciting
    assert ranked.ranks.reference_informative


def test_prepare_subspace_data_scales_copies_and_reconstructs_channels():
    y = np.array([[2.0, -2.0, 4.0, -4.0, 1.0, -1.0]])
    u = np.array([[3.0, -3.0, 6.0, -6.0, 1.5, -1.5]])
    original_y = y.copy()
    original_u = u.copy()

    data = prepare_subspace_data(y, u, future_horizon=2, past_offset=2)

    np.testing.assert_array_equal(y, original_y)
    np.testing.assert_array_equal(u, original_u)
    np.testing.assert_allclose(data.outputs * data.output_scale[:, None], y)
    np.testing.assert_allclose(data.inputs * data.input_scale[:, None], u)
    np.testing.assert_allclose(np.std(data.outputs, axis=1), 1.0)
    np.testing.assert_allclose(np.std(data.inputs, axis=1), 1.0)


def test_prepare_subspace_data_supports_a_distinct_predictor_past_depth():
    signal = np.arange(80, dtype=float).reshape(2, 40)
    data = prepare_subspace_data(
        signal,
        signal + 1.0,
        future_horizon=4,
        past_offset=7,
        past_block_rows=7,
        scale=False,
    )

    assert data.past_block_rows == 7
    assert data.past_outputs.shape == (14, 30)
    np.testing.assert_array_equal(data.past_outputs[:2], signal[:, :30])
    np.testing.assert_array_equal(data.past_outputs[-2:], signal[:, 6:36])
    np.testing.assert_array_equal(data.future_outputs[:2], signal[:, 7:37])

    with pytest.raises(ValueError, match="cannot exceed"):
        prepare_subspace_data(
            signal,
            signal,
            future_horizon=4,
            past_offset=3,
            past_block_rows=4,
        )


def test_prepare_subspace_data_reports_rank_without_rejecting_legacy_path():
    samples = np.arange(60, dtype=float)
    u = np.vstack((samples, 2.0 * samples))
    y = np.vstack((np.sin(samples), np.cos(samples)))

    data = prepare_subspace_data(
        y,
        u,
        future_horizon=4,
        past_offset=4,
        scale=False,
    )

    assert data.ranks.input_rank < data.ranks.input_rows
    assert not data.ranks.input_persistently_exciting

    with pytest.raises(ValueError, match="not persistently exciting"):
        prepare_subspace_data(
            y,
            u,
            future_horizon=4,
            past_offset=4,
            scale=False,
            require_persistent_excitation=True,
        )


def test_numerical_rank_matches_direct_svd_near_the_rank_boundary():
    base = np.linspace(0.5, 1.5, 64)
    matrix = np.vstack((base, base + 1e-16 * np.arange(base.size)))

    expected = int(np.linalg.matrix_rank(matrix))

    assert expected == 1
    assert _numerical_rank(matrix) == expected


def test_prepare_subspace_data_distinguishes_short_and_unexciting_records():
    with pytest.raises(ValueError, match="Need at least 10"):
        prepare_subspace_data(
            np.ones((1, 9)),
            np.ones((1, 9)),
            future_horizon=5,
            past_offset=5,
        )

    data = prepare_subspace_data(
        np.ones((1, 20)),
        np.ones((1, 20)),
        future_horizon=5,
        past_offset=5,
    )
    assert data.usable_columns == 11
    assert not data.ranks.input_persistently_exciting


def test_prepare_subspace_data_validates_alignment_and_finite_values():
    with pytest.raises(ValueError, match="same sample count"):
        prepare_subspace_data(
            np.ones((1, 30)),
            np.ones((1, 29)),
            future_horizon=4,
            past_offset=4,
        )
    with pytest.raises(ValueError, match="reference.*sample count"):
        prepare_subspace_data(
            np.ones((1, 30)),
            np.ones((1, 30)),
            future_horizon=4,
            past_offset=4,
            reference=np.ones((1, 29)),
        )
    with pytest.raises(ValueError, match="finite"):
        prepare_subspace_data(
            np.array([[1.0, np.nan, 2.0, 3.0, 4.0]]),
            np.ones((1, 5)),
            future_horizon=2,
            past_offset=2,
        )
