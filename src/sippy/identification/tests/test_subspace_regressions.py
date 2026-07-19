import numpy as np
import pytest
from scipy.linalg import solve_discrete_are

from sippy.identification.algorithms.subspace_core import SubspaceCoreAlgorithm
from sippy.identification.factory import create_algorithm
from sippy.utils.signal_utils import rescale
from sippy.utils.simulation_utils import K_calc


def test_k_calc_solves_generalized_dare_with_cross_covariance():
    A = np.array([[0.8, 0.1], [0.0, 0.7]])
    C = np.array([[1.0, 0.2]])
    Q = np.diag([0.05, 0.02])
    R = np.array([[0.3]])
    S = np.array([[0.01], [0.005]])

    covariance = solve_discrete_are(A.T, C.T, Q, R, s=S)
    numerator = A @ covariance @ C.T + S
    expected = np.linalg.solve((C @ covariance @ C.T + R).T, numerator.T).T

    gain, calculated = K_calc(A, C, Q, R, S)

    assert calculated
    assert np.linalg.norm(gain) > 0.0
    np.testing.assert_allclose(gain, expected, rtol=1e-12, atol=1e-12)


def test_force_a_stability_handles_multi_input_data():
    M = np.array([[1.2, 0.1, -0.2], [1.0, 0.0, 0.0]])
    observability = np.array([[1.0], [0.5]])
    state_sequence = np.linspace(0.0, 1.0, 8).reshape(1, -1)
    inputs = np.vstack(
        (
            np.linspace(-1.0, 1.0, 10),
            np.array([1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.1, -0.1, 0.0, 0.2]),
        )
    )

    stabilized, residuals, forced = SubspaceCoreAlgorithm.force_a_stability(
        M=M,
        n=1,
        Ob=observability,
        l=1,
        X_fd=state_sequence,
        N=8,
        u=inputs,
        f=1,
    )

    assert forced
    assert np.max(np.abs(np.linalg.eigvals(stabilized[:1, :1]))) < 1.0
    assert residuals.shape == (1, 7)
    assert np.all(np.isfinite(stabilized))


def test_rescale_constant_signal_is_finite_and_reversible():
    signal = np.full(20, 3.5)

    scale, scaled = rescale(signal)

    assert scale == 1.0
    assert np.all(np.isfinite(scaled))
    np.testing.assert_allclose(scaled * scale, signal)


@pytest.mark.parametrize("method", ["N4SID", "MOESP", "CVA"])
def test_subspace_identification_returns_nonzero_kalman_gain(method):
    rng = np.random.default_rng(42)
    sample_count = 600
    u = rng.normal(size=(1, sample_count))
    innovations = rng.normal(scale=0.1, size=sample_count)
    y = np.zeros((1, sample_count))
    for sample in range(1, sample_count):
        y[0, sample] = (
            0.7 * y[0, sample - 1] + 0.4 * u[0, sample - 1] + innovations[sample]
        )

    model = create_algorithm(method).identify(
        y=y,
        u=u,
        ss_f=10,
        ss_fixed_order=1,
    )

    assert np.all(np.isfinite(model.K))
    assert np.linalg.norm(model.K) > 1e-3


def test_cva_handles_rank_deficient_output_covariance():
    rng = np.random.default_rng(81)
    sample_count = 300
    u = rng.normal(size=(1, sample_count))
    y_base = np.zeros(sample_count)
    for sample in range(1, sample_count):
        y_base[sample] = 0.65 * y_base[sample - 1] + 0.4 * u[0, sample - 1]
    y = np.vstack((y_base, y_base))

    model = create_algorithm("CVA").identify(
        y=y,
        u=u,
        ss_f=10,
        ss_fixed_order=1,
    )

    assert np.all(np.isfinite(model.A))
    assert np.all(np.isfinite(model.B))
    assert np.all(np.isfinite(model.C))
    assert np.all(np.isfinite(model.D))
    assert model.K.shape == (1, 2)
    assert np.all(np.isfinite(model.K))
    np.testing.assert_allclose(model.C[0], model.C[1], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("method", ["PARSIM-K", "PARSIM-S", "PARSIM-P"])
def test_parsim_handles_a_constant_input_channel(method):
    rng = np.random.default_rng(7)
    sample_count = 500
    u = np.vstack((np.full(sample_count, 2.0), rng.normal(size=sample_count)))
    y = np.zeros((1, sample_count))
    for sample in range(1, sample_count):
        y[0, sample] = 0.6 * y[0, sample - 1] + 0.4 * u[1, sample - 1]

    model = create_algorithm(method).identify(
        y=y,
        u=u,
        ss_f=10,
        ss_fixed_order=1,
    )

    assert np.all(np.isfinite(model.A))
    assert np.all(np.isfinite(model.B))
    assert np.isfinite(model.Vn)
