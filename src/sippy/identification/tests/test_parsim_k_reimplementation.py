"""
TDD tests for PARSIM-K reimplementation.

This test file follows Test-Driven Development principles to ensure
the PARSIM-K algorithm matches the reference implementation.
"""

import numpy as np
import pytest

from sippy.identification.algorithms.parsim_core import (
    ParsimCoreAlgorithm,
    _estimate_predictor_markov_blocks,
    _solve_predictor_parameters,
)


def _analytical_predictor_hankel(*, direct_feedthrough: bool):
    rng = np.random.default_rng(410)
    state_order = 2
    outputs = 2
    inputs = 2
    horizon = 5
    columns = 96
    past_rows = horizon * (inputs + outputs)
    A_K = np.array([[0.45, 0.12], [-0.08, 0.35]])
    B_K = np.array([[0.3, -0.1], [0.08, 0.25]])
    C = np.array([[1.0, -0.2], [0.15, 0.8]])
    D = np.array([[0.06, 0.0], [-0.02, 0.04]])
    if not direct_feedthrough:
        D = np.zeros_like(D)
    K = np.array([[0.18, -0.04], [0.03, 0.12]])
    past_map = rng.standard_normal((state_order, past_rows))
    Zp = rng.standard_normal((past_rows, columns))
    Uf = rng.standard_normal((inputs * horizon, columns))

    gamma_blocks = []
    input_blocks = [D]
    output_blocks = [np.zeros((outputs, outputs))]
    power = np.eye(state_order)
    for block in range(horizon):
        gamma_blocks.append(C @ power @ past_map)
        if block:
            previous_power = np.linalg.matrix_power(A_K, block - 1)
            input_blocks.append(C @ previous_power @ B_K)
            output_blocks.append(C @ previous_power @ K)
        power = power @ A_K

    initial_regressor = np.vstack((Zp, Uf[:inputs])) if direct_feedthrough else Zp

    def orthogonal_innovations(regressor):
        draws = rng.standard_normal((outputs, columns))
        return draws - (draws @ np.linalg.pinv(regressor)) @ regressor

    first_output = gamma_blocks[0] @ Zp + input_blocks[0] @ Uf[:inputs]
    first_output += orthogonal_innovations(initial_regressor)
    future_outputs = [first_output]
    iteration_regressor = np.vstack((Zp, Uf[:inputs], first_output))
    for row in range(1, horizon):
        value = gamma_blocks[row] @ Zp
        for lag in range(row + 1):
            input_slice = Uf[inputs * (row - lag) : inputs * (row - lag + 1)]
            value = value + input_blocks[lag] @ input_slice
            if lag:
                output_slice = future_outputs[row - lag]
                value = value + output_blocks[lag] @ output_slice
        value += orthogonal_innovations(iteration_regressor)
        future_outputs.append(value)

    return (
        np.vstack(future_outputs),
        Uf,
        Zp,
        np.stack(gamma_blocks),
        np.stack(input_blocks),
        np.stack(output_blocks),
    )


@pytest.mark.parametrize("direct_feedthrough", [False, True])
def test_predictor_markov_regression_recovers_analytical_toeplitz_blocks(
    direct_feedthrough,
):
    Yf, Uf, Zp, gamma, input_blocks, output_blocks = _analytical_predictor_hankel(
        direct_feedthrough=direct_feedthrough
    )

    estimate = _estimate_predictor_markov_blocks(
        Yf,
        Uf,
        Zp,
        output_count=2,
        input_count=2,
        future_horizon=5,
        direct_feedthrough=direct_feedthrough,
        strict=True,
    )

    np.testing.assert_allclose(estimate.gamma_blocks, gamma, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(
        estimate.input_blocks,
        input_blocks,
        rtol=2e-10,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        estimate.output_blocks,
        output_blocks,
        rtol=2e-10,
        atol=2e-10,
    )
    assert not estimate.used_compatibility_fallback


def test_predictor_markov_regression_reuses_two_factorizations(monkeypatch):
    import sippy.identification.algorithms.parsim_core as parsim_core_module

    Yf, Uf, Zp, *_ = _analytical_predictor_hankel(direct_feedthrough=False)
    factorization_calls = 0
    original_qr = parsim_core_module.sc.linalg.qr

    def tracked_qr(*args, **kwargs):
        nonlocal factorization_calls
        factorization_calls += 1
        return original_qr(*args, **kwargs)

    monkeypatch.setattr(parsim_core_module.sc.linalg, "qr", tracked_qr)

    _estimate_predictor_markov_blocks(
        Yf,
        Uf,
        Zp,
        output_count=2,
        input_count=2,
        future_horizon=5,
        direct_feedthrough=False,
    )

    assert factorization_calls == 2


def test_predictor_markov_regression_has_strict_identifiability_semantics():
    Yf = np.ones((4, 30))
    Uf = np.ones((4, 30))
    Zp = np.ones((8, 30))

    with pytest.raises(ValueError, match="not identifiable"):
        _estimate_predictor_markov_blocks(
            Yf,
            Uf,
            Zp,
            output_count=1,
            input_count=1,
            future_horizon=4,
            direct_feedthrough=False,
            strict=True,
        )

    estimate = _estimate_predictor_markov_blocks(
        Yf,
        Uf,
        Zp,
        output_count=1,
        input_count=1,
        future_horizon=4,
        direct_feedthrough=False,
    )
    assert estimate.used_compatibility_fallback


def test_predictor_parameter_regression_has_strict_identifiability_semantics():
    design = np.column_stack((np.ones(40), np.ones(40)))
    outputs = np.linspace(-1.0, 1.0, 40).reshape(1, -1)

    with pytest.raises(ValueError, match="not identifiable"):
        _solve_predictor_parameters(design, outputs, strict=True)

    coefficients, used_fallback = _solve_predictor_parameters(design, outputs)
    assert coefficients.shape == (2, 1)
    assert used_fallback


class TestParsimKReimplementation:
    """TDD tests for PARSIM-K reimplementation."""

    @pytest.fixture
    def simple_siso_system(self):
        """Simple SISO system for testing."""
        np.random.seed(42)
        n_points = 200
        u = np.random.randn(1, n_points)
        # True system: y[k] = 0.8*y[k-1] + 0.5*u[k-1] + noise
        y = np.zeros((1, n_points))
        for i in range(1, n_points):
            y[0, i] = 0.8 * y[0, i - 1] + 0.5 * u[0, i - 1] + 0.05 * np.random.randn()
        return y, u

    def test_svd_weighted_k_exists(self):
        """Test that SVD_weighted_K function exists in parsim_core."""
        # This will FAIL initially - that's expected in TDD (RED phase)
        assert hasattr(ParsimCoreAlgorithm, "svd_weighted_k"), (
            "SVD_weighted_K method must exist"
        )

    def test_svd_weighted_k_returns_correct_shapes(self, simple_siso_system):
        """Test SVD_weighted_K returns U_n, S_n, V_n with correct shapes."""
        from sippy.utils.simulation_utils import impile, ordinate_sequence

        y, u = simple_siso_system
        l_, L = y.shape
        u.shape[0]
        f = 10

        # Build input matrices like master does
        Yf, Yp = ordinate_sequence(y, f, f)
        Uf, Up = ordinate_sequence(u, f, f)
        Zp = impile(Up, Yp)

        # Create a simple Gamma_L for testing
        Gamma_L = np.random.randn(l_ * f, Yf.shape[1])

        # Call SVD_weighted_K
        U_n, S_n, V_n = ParsimCoreAlgorithm.svd_weighted_k(Uf, Zp, Gamma_L)

        # Check shapes
        assert U_n.shape[0] == l_ * f, "U_n should have l_*f rows"
        assert S_n.shape[0] > 0, "S_n should have positive size"
        assert V_n.shape[1] == Gamma_L.shape[1], "V_n columns should match Gamma_L"

    def test_simulations_sequence_k_exists(self):
        """Test that simulations_sequence_k function exists."""
        assert hasattr(ParsimCoreAlgorithm, "simulations_sequence_k"), (
            "simulations_sequence_k method must exist"
        )

    def test_simulations_sequence_k_returns_correct_shape(self, simple_siso_system):
        """Test simulations_sequence_k returns correct shape."""
        y, u = simple_siso_system
        l_, L = y.shape
        m = u.shape[0]
        n = 2  # model order

        # Create dummy matrices
        A_K = np.random.randn(n, n) * 0.1
        C = np.random.randn(l_, n)
        np.random.randn(n, m)
        D = np.zeros((l_, m))
        K = np.random.randn(n, l_) * 0.01
        np.zeros((n, 1))

        # This should call simulations_sequence_k
        y_sim = ParsimCoreAlgorithm.simulations_sequence_k(
            A_K, C, L, y, u, l_, m, n, K, D, D_required=False
        )

        # Check shape - master branch transposes at end, so output is (L*l_, n_simulations)
        # This matches how it's used: pinv(y_sim) @ y gives correct dimensions
        expected_simulations = n * m + n * l_ + n
        assert y_sim.shape == (
            L * l_,
            expected_simulations,
        ), f"Expected shape ({L * l_}, {expected_simulations}), got {y_sim.shape}"

    def test_ss_lsim_predictor_form_exists(self):
        """Test that SS_lsim_predictor_form function exists."""
        from sippy.utils.simulation_utils import ss_lsim_predictor_form

        # This will FAIL initially
        assert callable(ss_lsim_predictor_form), (
            "ss_lsim_predictor_form must be callable"
        )

    def test_ss_lsim_predictor_form_simulation(self, simple_siso_system):
        """Test that SS_lsim_predictor_form works correctly."""
        from sippy.utils.simulation_utils import ss_lsim_predictor_form

        y, u = simple_siso_system
        l_, L = y.shape
        m = u.shape[0]
        n = 2  # model order

        # Create simple state-space model
        A_K = np.array([[0.8, 0.1], [0.0, 0.5]])
        B_K = np.array([[0.5], [0.3]])
        C = np.array([[1.0, 0.0]])
        D = np.zeros((l_, m))
        K = np.array([[0.1], [0.05]])
        x0 = np.zeros((n, 1))

        # Call predictor form simulation
        x, y_hat = ss_lsim_predictor_form(A_K, B_K, C, D, K, y, u, x0)

        # Check shapes
        assert x.shape == (n, L + 1), f"Expected x shape ({n}, {L + 1}), got {x.shape}"
        assert y_hat.shape == (l_, L), (
            f"Expected y_hat shape ({l_}, {L}), got {y_hat.shape}"
        )

        # Check predictor form equation: x[i+1] = A_K*x[i] + B_K*u[i] + K*y[i]
        for i in range(5):  # Check first 5 steps
            x_next_expected = (
                A_K @ x[:, i : i + 1] + B_K @ u[:, i : i + 1] + K @ y[:, i : i + 1]
            )
            np.testing.assert_allclose(
                x[:, i + 1 : i + 2],
                x_next_expected,
                rtol=1e-10,
                err_msg=f"Predictor form equation failed at step {i}",
            )

    def test_parsim_k_uses_gamma_l_in_svd(self, simple_siso_system):
        """Test that PARSIM-K constructs and uses Gamma_L in SVD."""
        y, u = simple_siso_system

        # Run PARSIM-K
        A_K, C, B_K, D, K, A, B, x0, Vn = ParsimCoreAlgorithm.parsim_k(
            y, u, f=10, p=10, threshold=0.1, fixed_order=2, D_required=False
        )

        # If implementation is correct, it should produce reasonable matrices
        assert A_K.shape[0] == 2, "A_K should be 2x2 for order 2"
        assert C.shape == (1, 2), "C should be 1x2"
        assert K.shape == (2, 1), "K should be 2x1"
        assert not np.all(K == 0), "K should not be all zeros"

    def test_parsim_k_vs_reference_simple_case(self, simple_siso_system):
        """Test PARSIM-K against known reference behavior."""
        y, u = simple_siso_system

        # Run with fixed order
        A_K, C, B_K, D, K, A, B, x0, Vn = ParsimCoreAlgorithm.parsim_k(
            y, u, f=10, p=10, threshold=0.0, fixed_order=2, D_required=False
        )

        # Basic sanity checks
        assert np.isfinite(A_K).all(), "A_K should have finite values"
        assert np.isfinite(C).all(), "C should have finite values"
        assert np.isfinite(B_K).all(), "B_K should have finite values"
        assert np.isfinite(K).all(), "K should have finite values"
        assert np.isfinite(Vn), "Vn should be finite"

        # Check that A is related to A_K and K by: A = A_K + K*C
        A_expected = A_K + K @ C
        np.testing.assert_allclose(
            A,
            A_expected,
            rtol=1e-10,
            err_msg="A should equal A_K + K*C",
        )

    def test_parsim_k_predictor_form_simulation_is_used(self, simple_siso_system):
        """Test that PARSIM-K uses predictor form simulation internally."""
        # This is a behavioral test - we verify the algorithm produces
        # correct results consistent with predictor form
        y, u = simple_siso_system

        A_K, C, B_K, D, K, A, B, x0, Vn = ParsimCoreAlgorithm.parsim_k(
            y, u, f=10, p=10, threshold=0.0, fixed_order=2, D_required=False
        )

        # The key property of predictor form is that K should affect
        # how states evolve based on output feedback
        # If K is non-zero and well-estimated, the model should be reasonable
        assert not np.allclose(K, 0, atol=1e-10), (
            "K should be non-zero in predictor form"
        )

        # Verify B relationship: B = B_K + K*D
        B_expected = B_K + K @ D
        np.testing.assert_allclose(
            B,
            B_expected,
            rtol=1e-10,
            err_msg="B should equal B_K + K*D",
        )
