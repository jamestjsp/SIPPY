import numpy as np
import pytest

from sippy.identification.algorithms import subspace_core as subspace_core_module
from sippy.identification.algorithms.subspace_core import SubspaceCoreAlgorithm
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
