import threading
import time

import numpy as np
import pytest
from scipy.linalg import solve_discrete_are

from sippy.identification.algorithms import subspace_core as subspace_core_module
from sippy.identification.algorithms.subspace_core import SubspaceCoreAlgorithm
from sippy.identification.algorithms.subspace_weighting import cva_weighted_svd
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


def test_cva_whitens_the_conditional_consistent_subspace():
    rng = np.random.default_rng(82)
    consistent = rng.normal(size=(6, 30))
    conditional_outputs = rng.normal(size=(6, 90))

    U, singular_values, Vh, square_root, diagnostics = cva_weighted_svd(
        consistent,
        conditional_outputs,
    )

    whitened_outputs = np.linalg.solve(square_root, conditional_outputs)
    np.testing.assert_allclose(
        whitened_outputs @ whitened_outputs.T,
        np.eye(6),
        rtol=2e-10,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        U @ np.diag(singular_values) @ Vh,
        np.linalg.solve(square_root, consistent),
        rtol=2e-10,
        atol=2e-10,
    )
    assert diagnostics.applied == "CVA"
    assert diagnostics.fallback_reason is None


def test_singular_cva_falls_back_to_the_same_unweighted_consistent_subspace():
    rng = np.random.default_rng(83)
    consistent = rng.normal(size=(4, 20))
    conditional_base = rng.normal(size=(2, 60))
    conditional_outputs = np.vstack(
        (conditional_base, conditional_base[0:1], conditional_base[1:2])
    )

    U, singular_values, Vh, square_root, diagnostics = cva_weighted_svd(
        consistent,
        conditional_outputs,
    )

    assert square_root is None
    np.testing.assert_allclose(
        U @ np.diag(singular_values) @ Vh,
        consistent,
        rtol=2e-10,
        atol=2e-10,
    )
    assert diagnostics.applied == "unweighted"
    assert diagnostics.covariance_rank == 2
    assert diagnostics.fallback_reason == "conditional_covariance_rank_deficient"


def test_subspace_order_selection_uses_causal_innovation_likelihood():
    rng = np.random.default_rng(44)
    sample_count = 60
    u = rng.normal(size=(1, sample_count))
    state = np.zeros((2, sample_count))
    y = np.zeros((1, sample_count))
    A = np.array([[1.5, -0.56], [1.0, 0.0]])
    B = np.array([[0.4], [0.0]])
    for sample in range(sample_count - 1):
        state[:, sample + 1] = A @ state[:, sample] + B[:, 0] * u[0, sample]
    y[:] = state[:1] + 0.5 * rng.normal(size=(1, sample_count))

    selected_A, *_ = SubspaceCoreAlgorithm.select_order(
        y,
        u,
        f=6,
        weights="N4SID",
        method="BIC",
        orders=[1, 6],
        n_jobs=1,
    )

    assert selected_A.shape == (2, 2)


def test_innovation_information_criterion_uses_mimo_log_determinant():
    errors = np.array(
        [
            [1.0, -1.0, 2.0, -2.0, 0.5],
            [0.2, 0.8, -0.4, -1.2, 1.5],
        ]
    )
    parameter_count = 7
    sample_count = errors.shape[1]
    covariance = errors @ errors.T / sample_count
    _, log_determinant = np.linalg.slogdet(covariance)
    expected = sample_count * log_determinant + parameter_count * np.log(sample_count)

    actual = subspace_core_module._innovation_information_criterion(
        errors,
        parameter_count,
        "BIC",
    )

    assert actual == pytest.approx(expected)


def test_automatic_horizon_and_order_candidates_respect_data_constraints():
    horizons = subspace_core_module._default_horizon_candidates(
        500,
        2,
        2,
        reference_count=4,
    )

    assert len(horizons) >= 2
    for horizon in horizons:
        usable_columns = 500 - 2 * horizon + 1
        assert usable_columns >= (2 + 2) * (horizon + 1)
        assert usable_columns >= 2 * 4 * horizon

    assert subspace_core_module._default_horizon_candidates(
        500,
        2,
        2,
        reference_count=4,
        explicit_horizon=7,
    ) == (7,)

    orders, effective_rank = (
        subspace_core_module._candidate_orders_from_singular_values(
            np.array([10.0, 4.0, 0.3, 0.02, 1e-8]),
            horizon=5,
        )
    )
    assert all(1 <= order < 5 for order in orders)
    assert effective_rank == 4
    assert subspace_core_module._candidate_orders_from_singular_values(
        np.array([10.0, 4.0, 0.3]),
        horizon=5,
        explicit_order=2,
    )[0] == (2,)


def test_dimension_selection_reuses_causal_innovation_scoring_on_fixed_tail(
    monkeypatch,
):
    rng = np.random.default_rng(45)
    samples = 300
    u = rng.normal(size=(1, samples))
    y = np.zeros((1, samples))
    for sample in range(1, samples):
        y[0, sample] = (
            0.7 * y[0, sample - 1] + 0.4 * u[0, sample - 1] + 0.03 * rng.normal()
        )

    candidates = (
        subspace_core_module.DimensionCandidate(
            horizon=8,
            order=1,
            singular_values=np.array([5.0, 0.1]),
            effective_rank=2,
            singular_gap=50.0,
            A=np.array([[0.7]]),
            B=np.array([[0.4]]),
            C=np.array([[1.0]]),
            D=np.zeros((1, 1)),
            K=np.array([[0.7]]),
            parameter_count=4,
            initial_state=np.zeros(1),
        ),
        subspace_core_module.DimensionCandidate(
            horizon=12,
            order=2,
            singular_values=np.array([5.0, 2.0, 0.1]),
            effective_rank=3,
            singular_gap=20.0,
            A=np.diag([0.2, 0.1]),
            B=np.zeros((2, 1)),
            C=np.array([[1.0, 0.0]]),
            D=np.zeros((1, 1)),
            K=np.zeros((2, 1)),
            parameter_count=12,
            initial_state=np.zeros(2),
        ),
    )
    calls = 0
    original = subspace_core_module._innovation_information_criterion

    def tracked_criterion(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        subspace_core_module,
        "_innovation_information_criterion",
        tracked_criterion,
    )

    selection = subspace_core_module._select_dimension_candidate(
        candidates,
        y,
        u,
        method="BIC",
        validation_fraction=0.2,
    )

    assert calls == len(candidates)
    assert selection.candidate.order == 1
    assert selection.validation_start == 240


def test_subspace_order_selection_honors_parallel_n_jobs(monkeypatch):
    rng = np.random.default_rng(16)
    sample_count = 120
    u = rng.normal(size=(1, sample_count))
    y = np.zeros((1, sample_count))
    for sample in range(1, sample_count):
        y[0, sample] = 0.7 * y[0, sample - 1] + 0.3 * u[0, sample - 1]

    sequential = SubspaceCoreAlgorithm.select_order(
        y,
        u,
        f=8,
        weights="N4SID",
        method="BIC",
        orders=[1, 4],
        n_jobs=1,
    )

    thread_ids = set()
    thread_lock = threading.Lock()
    original = subspace_core_module._causal_prediction_errors

    def tracked_prediction_errors(*args, **kwargs):
        with thread_lock:
            thread_ids.add(threading.get_ident())
        time.sleep(0.01)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        subspace_core_module,
        "_causal_prediction_errors",
        tracked_prediction_errors,
    )

    parallel = SubspaceCoreAlgorithm.select_order(
        y,
        u,
        f=8,
        weights="N4SID",
        method="BIC",
        orders=[1, 4],
        n_jobs=2,
    )

    assert len(thread_ids) > 1
    for sequential_value, parallel_value in zip(sequential, parallel):
        np.testing.assert_allclose(
            parallel_value,
            sequential_value,
            rtol=1e-9,
            atol=1e-10,
        )


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
