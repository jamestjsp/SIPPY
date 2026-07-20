import numpy as np
import pytest

from sippy.identification.algorithms import subspace_core as subspace_core_module
from sippy.identification.algorithms.subspace_core import (
    SubspaceCoreAlgorithm,
    _project_onto_reference_row_space,
    _two_stage_ort_projection,
)
from sippy.identification.algorithms.subspace_data import prepare_subspace_data
from sippy.utils.simulation_utils import Z_dot_PIort, impile, ordinate_sequence


def _explicit_projection(y, u, horizon):
    Yf, Yp = ordinate_sequence(y, horizon, horizon)
    Uf, Up = ordinate_sequence(u, horizon, horizon)
    Zp = impile(Up, Yp)
    projected_outputs = Z_dot_PIort(Yf, Uf)
    projected_past = Z_dot_PIort(Zp, Uf)
    projection = projected_outputs @ np.linalg.pinv(projected_past) @ Zp
    return Yf, Uf, projection, projected_outputs


@pytest.mark.parametrize("weights", ["N4SID", "MOESP", "CVA"])
def test_lq_compression_matches_explicit_projection(weights):
    rng = np.random.default_rng(120)
    sample_count = 500
    horizon = 8
    u = rng.normal(size=(2, sample_count))
    y = np.zeros((2, sample_count))
    for sample in range(2, sample_count):
        y[:, sample] = (
            np.array([[0.72, 0.08], [-0.04, 0.61]]) @ y[:, sample - 1]
            + np.array([[0.5, -0.1], [0.2, 0.35]]) @ u[:, sample - 1]
            + 0.02 * rng.normal(size=2)
        )

    _, singular_values, _, weighting, projection = SubspaceCoreAlgorithm.svd_weighted(
        y, u, horizon, 2, weights
    )
    _, Uf, explicit_projection, projected_outputs = _explicit_projection(y, u, horizon)

    np.testing.assert_allclose(
        projection.materialize(),
        explicit_projection,
        rtol=2e-9,
        atol=2e-10,
    )

    if weights == "N4SID":
        expected_matrix = explicit_projection
    elif weights == "MOESP":
        expected_matrix = Z_dot_PIort(explicit_projection, Uf)
    else:
        covariance = projected_outputs @ projected_outputs.T
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
        tolerance = (
            max(covariance.shape)
            * np.finfo(np.float64).eps
            * max(float(eigenvalues[-1]), 0.0)
        )
        retained = eigenvalues > tolerance
        inverse_square_root = (
            eigenvectors[:, retained] * (1.0 / np.sqrt(eigenvalues[retained]))
        ) @ eigenvectors[:, retained].T
        expected_matrix = Z_dot_PIort(inverse_square_root @ explicit_projection, Uf)
        assert weighting is not None

    expected_singular_values = np.linalg.svd(
        expected_matrix, full_matrices=False, compute_uv=False
    )
    np.testing.assert_allclose(
        singular_values,
        expected_singular_values,
        rtol=2e-9,
        atol=2e-10,
    )


def test_subspace_compression_has_structural_performance_gate(monkeypatch):
    rng = np.random.default_rng(121)
    sample_count = 20_000
    horizon = 12
    u = rng.normal(size=(2, sample_count))
    y = rng.normal(size=(2, sample_count))
    factorization_calls = 0
    svd_shapes = []
    original_qr = np.linalg.qr
    original_svd = np.linalg.svd

    def tracked_qr(*args, **kwargs):
        nonlocal factorization_calls
        factorization_calls += 1
        return original_qr(*args, **kwargs)

    def tracked_svd(matrix, *args, **kwargs):
        svd_shapes.append(matrix.shape)
        return original_svd(matrix, *args, **kwargs)

    monkeypatch.setattr(subspace_core_module.np.linalg, "qr", tracked_qr)
    monkeypatch.setattr(subspace_core_module.np.linalg, "svd", tracked_svd)
    monkeypatch.setattr(subspace_core_module, "NUMBA_AVAILABLE", False)
    monkeypatch.setattr(
        subspace_core_module,
        "Z_dot_PIort",
        lambda *_args, **_kwargs: pytest.fail("explicit projection was used"),
    )

    _, _, _, _, projection = SubspaceCoreAlgorithm.svd_weighted(y, u, horizon, 2, "CVA")

    stacked_row_count = horizon * (2 * u.shape[0] + 2 * y.shape[0])
    assert factorization_calls == 1
    assert svd_shapes
    assert all(max(shape) <= stacked_row_count for shape in svd_shapes)
    assert projection.projector.shape[1] == horizon * (u.shape[0] + y.shape[0])


def test_lq_compression_preserves_underdetermined_mimo_behavior():
    rng = np.random.default_rng(122)
    sample_count = 150
    horizon = 20
    u = rng.normal(size=(2, sample_count))
    y = rng.normal(size=(2, sample_count))

    _, singular_values, _, _, projection = SubspaceCoreAlgorithm.svd_weighted(
        y, u, horizon, 2, "N4SID"
    )
    _, _, explicit_projection, _ = _explicit_projection(y, u, horizon)

    np.testing.assert_allclose(
        projection.materialize(),
        explicit_projection,
        rtol=2e-9,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        singular_values,
        np.linalg.svd(explicit_projection, compute_uv=False, full_matrices=False),
        rtol=2e-9,
        atol=2e-10,
    )


def _reference_projection_fixture(*, sample_count=1200, horizon=8):
    rng = np.random.default_rng(123)
    reference = rng.normal(size=(4, sample_count))
    u = np.zeros((2, sample_count))
    y = np.zeros((2, sample_count))
    for sample in range(2, sample_count):
        u[:, sample] = (
            np.array([[0.65, 0.1], [-0.05, 0.55]]) @ u[:, sample - 1]
            + np.array([[0.5, 0.0, 0.2, -0.1], [0.1, 0.4, -0.15, 0.25]])
            @ reference[:, sample]
        )
        y[:, sample] = (
            np.array([[0.7, 0.05], [-0.08, 0.62]]) @ y[:, sample - 1]
            + np.array([[0.4, -0.1], [0.15, 0.3]]) @ u[:, sample - 1]
            + 0.15 * rng.normal(size=2)
        )
    return prepare_subspace_data(
        y,
        u,
        future_horizon=horizon,
        past_offset=horizon,
        reference=reference,
        scale=False,
    )


def test_reference_lq_projection_matches_explicit_row_space_projection():
    data = _reference_projection_fixture()
    stage = _project_onto_reference_row_space(data)

    assert stage.diagnostics.usable
    reference_hankel = np.vstack((data.past_references, data.future_references))
    signals = np.vstack((data.future_inputs, data.past_data, data.future_outputs))
    expected = signals @ np.linalg.pinv(reference_hankel) @ reference_hankel

    np.testing.assert_allclose(stage.materialize(), expected, rtol=2e-9, atol=2e-10)
    assert max(stage.reference_factor.shape) <= reference_hankel.shape[0]
    assert stage.coefficient_map.shape == (signals.shape[0], reference_hankel.shape[0])


def test_two_stage_ort_matches_two_explicit_orthogonal_projections():
    data = _reference_projection_fixture()
    result = _two_stage_ort_projection(data)

    assert result.diagnostics.usable
    first = result.reference_projection.materialize()
    input_rows = data.future_inputs.shape[0]
    past_rows = data.past_data.shape[0]
    projected_inputs = first[:input_rows]
    projected_past = first[input_rows : input_rows + past_rows]
    projected_outputs = first[input_rows + past_rows :]
    outputs_orthogonal_to_inputs = Z_dot_PIort(projected_outputs, projected_inputs)
    past_orthogonal_to_inputs = Z_dot_PIort(projected_past, projected_inputs)
    expected = (
        outputs_orthogonal_to_inputs
        @ np.linalg.pinv(past_orthogonal_to_inputs)
        @ projected_past
    )

    np.testing.assert_allclose(
        result.projection.materialize(),
        expected,
        rtol=3e-9,
        atol=3e-10,
    )


def test_reference_projection_rejects_rank_deficient_exogenous_channels():
    data = _reference_projection_fixture()
    duplicated = np.vstack((data.references[:1], data.references[:1]))
    deficient = prepare_subspace_data(
        data.outputs,
        data.inputs,
        future_horizon=data.future_horizon,
        past_offset=data.past_offset,
        reference=duplicated,
        scale=False,
    )

    result = _two_stage_ort_projection(deficient)

    assert not result.diagnostics.usable
    assert result.diagnostics.reason == "reference_rank_deficient"


def test_ort_does_not_substitute_input_or_past_output_for_missing_reference():
    source = _reference_projection_fixture()
    without_reference = prepare_subspace_data(
        source.outputs,
        source.inputs,
        future_horizon=source.future_horizon,
        past_offset=source.past_offset,
        scale=False,
    )

    result = _two_stage_ort_projection(without_reference)

    assert not result.diagnostics.usable
    assert result.diagnostics.reason == "reference_missing"
    assert result.reference_projection is None


def test_reference_must_independently_excite_projected_plant_inputs():
    rng = np.random.default_rng(124)
    samples = 1000
    data = prepare_subspace_data(
        rng.normal(size=(1, samples)),
        np.zeros((1, samples)),
        future_horizon=8,
        past_offset=8,
        reference=rng.normal(size=(2, samples)),
        scale=False,
    )

    result = _two_stage_ort_projection(data)

    assert not result.diagnostics.usable
    assert result.diagnostics.reason == "projected_input_rank_deficient"


def test_two_stage_ort_factors_stay_bounded_by_hankel_rows():
    data = _reference_projection_fixture(sample_count=20_000, horizon=12)
    result = _two_stage_ort_projection(data)

    assert result.diagnostics.usable
    reference_rows = result.diagnostics.reference_rows
    assert max(result.reference_projection.reference_factor.shape) <= reference_rows
    assert result.reference_projection.coefficient_map.shape[1] == reference_rows
    assert max(result.compact_projection.shape) <= reference_rows
    assert max(result.projection.projector.shape) <= reference_rows
