"""
PARSIM algorithms core implementation.
"""

from dataclasses import dataclass

import numpy as np
import scipy as sc

try:
    from joblib import Parallel, delayed

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from ...utils.simulation_utils import (
    Vn_mat,
    check_inputs,
    check_types,
    impile,
    reducingOrder,
    simulate_ss_system,
)
from .subspace_data import prepare_subspace_data

# Import compiled utilities for performance
try:
    from ...utils.compiled_utils import (
        NUMBA_AVAILABLE,
        Z_dot_PIort_compiled,
        matrix_operations_a_compiled,
        pinv_compiled_svd,
        subspace_weighted_svd_compiled,
    )
except ImportError:
    subspace_weighted_svd_compiled = None
    Z_dot_PIort_compiled = None
    matrix_operations_a_compiled = None
    NUMBA_AVAILABLE = False


def _full_rank_from_diagonal(triangular, sample_count):
    diagonal = np.abs(np.diag(triangular))
    if diagonal.size == 0:
        return False
    scale = float(np.max(diagonal))
    tolerance = max(sample_count, *triangular.shape) * np.finfo(np.float64).eps * scale
    return scale > 0.0 and bool(np.all(diagonal > tolerance))


@dataclass(frozen=True)
class _ReusableRightLeastSquares:
    regressor: np.ndarray
    q: np.ndarray
    r: np.ndarray
    pivots: np.ndarray
    full_rank: bool

    @classmethod
    def factor(cls, regressor):
        q, r, pivots = sc.linalg.qr(
            regressor.T,
            mode="economic",
            pivoting=True,
            check_finite=False,
        )
        full_rank = r.shape[0] >= regressor.shape[0] and _full_rank_from_diagonal(
            r[: regressor.shape[0]], regressor.shape[1]
        )
        return cls(regressor, q, r, pivots, full_rank)

    def solve(self, target):
        if not self.full_rank:
            return target @ np.linalg.pinv(self.regressor)
        row_count = self.regressor.shape[0]
        transformed = self.q.T @ target.T
        pivoted = sc.linalg.solve_triangular(
            self.r[:row_count],
            transformed[:row_count],
            lower=False,
            check_finite=False,
        )
        coefficients = np.empty_like(pivoted)
        coefficients[self.pivots] = pivoted
        return coefficients.T


@dataclass(frozen=True)
class PredictorMarkovEstimate:
    gamma_blocks: np.ndarray
    input_blocks: np.ndarray
    output_blocks: np.ndarray
    used_compatibility_fallback: bool

    @property
    def gamma_matrix(self) -> np.ndarray:
        return self.gamma_blocks.reshape(
            self.gamma_blocks.shape[0] * self.gamma_blocks.shape[1],
            self.gamma_blocks.shape[2],
        )


def _estimate_predictor_markov_blocks(
    Yf,
    Uf,
    Zp,
    *,
    output_count,
    input_count,
    future_horizon,
    direct_feedthrough,
    strict=False,
):
    column_count = Zp.shape[1]
    expected_output_rows = output_count * future_horizon
    expected_input_rows = input_count * future_horizon
    if Yf.shape != (expected_output_rows, column_count):
        raise ValueError("future output Hankel matrix has incompatible dimensions")
    if Uf.shape != (expected_input_rows, column_count):
        raise ValueError("future input Hankel matrix has incompatible dimensions")

    gamma_blocks = np.empty((future_horizon, output_count, Zp.shape[0]), dtype=float)
    input_blocks = np.zeros((future_horizon, output_count, input_count), dtype=float)
    output_blocks = np.zeros((future_horizon, output_count, output_count), dtype=float)

    initial_regressor = np.vstack((Zp, Uf[:input_count])) if direct_feedthrough else Zp
    initial_solver = _ReusableRightLeastSquares.factor(initial_regressor)
    initial_coefficients = initial_solver.solve(Yf[:output_count])
    gamma_blocks[0] = initial_coefficients[:, : Zp.shape[0]]
    if direct_feedthrough:
        input_blocks[0] = initial_coefficients[:, Zp.shape[0] :]

    iteration_solver = None
    if future_horizon > 1:
        iteration_solver = _ReusableRightLeastSquares.factor(
            np.vstack((Zp, Uf[:input_count], Yf[:output_count]))
        )
        for row in range(1, future_horizon):
            adjusted_output = Yf[output_count * row : output_count * (row + 1)].copy()
            for lag in range(row):
                input_start = input_count * (row - lag)
                adjusted_output -= (
                    input_blocks[lag] @ Uf[input_start : input_start + input_count]
                )
                if lag:
                    output_start = output_count * (row - lag)
                    adjusted_output -= (
                        output_blocks[lag]
                        @ Yf[output_start : output_start + output_count]
                    )

            coefficients = iteration_solver.solve(adjusted_output)
            gamma_blocks[row] = coefficients[:, : Zp.shape[0]]
            markov_start = Zp.shape[0]
            input_blocks[row] = coefficients[
                :, markov_start : markov_start + input_count
            ]
            output_blocks[row] = coefficients[:, markov_start + input_count :]

    used_fallback = not initial_solver.full_rank or (
        iteration_solver is not None and not iteration_solver.full_rank
    )
    if strict and used_fallback:
        raise ValueError(
            "predictor Markov regression is not identifiable from the supplied data"
        )
    return PredictorMarkovEstimate(
        gamma_blocks=gamma_blocks,
        input_blocks=input_blocks,
        output_blocks=output_blocks,
        used_compatibility_fallback=used_fallback,
    )


def _solve_predictor_parameters(design, outputs, *, strict=False):
    target = np.asarray(outputs, dtype=float).reshape(1, -1)
    if design.shape[0] != target.shape[1]:
        raise ValueError(
            "predictor design and output data have incompatible dimensions"
        )
    solver = _ReusableRightLeastSquares.factor(design.T)
    if strict and not solver.full_rank:
        raise ValueError(
            "predictor parameter regression is not identifiable from the supplied data"
        )
    return solver.solve(target).T, not solver.full_rank


def _build_parsim_p_gamma_l(Yf, Uf, Zp, f, l_, m):
    regressor_rows = Zp.shape[0] + Uf.shape[0]
    stacked = np.vstack((Zp, Uf, Yf))
    L = np.linalg.qr(stacked.T, mode="r").T
    if L.shape[1] < regressor_rows:
        raise ValueError(
            "Not enough data columns for PARSIM-P LQ regression; "
            f"need at least {regressor_rows}, got {L.shape[1]}"
        )

    output_start = regressor_rows
    gamma_blocks = []
    for i in range(f):
        prefix_rows = Zp.shape[0] + m * (i + 1)
        triangular = L[:prefix_rows, :prefix_rows]
        output_block = L[
            output_start + l_ * i : output_start + l_ * (i + 1),
            :prefix_rows,
        ]
        if _full_rank_from_diagonal(triangular, stacked.shape[1]):
            coefficients = sc.linalg.solve_triangular(
                triangular.T,
                output_block.T,
                lower=False,
                check_finite=False,
            ).T
        else:
            regressor = impile(Zp, Uf[: m * (i + 1)])
            coefficients = Yf[l_ * i : l_ * (i + 1)] @ np.linalg.pinv(regressor)
        gamma_blocks.append(coefficients[:, : Zp.shape[0]])
    return np.vstack(gamma_blocks)


class ParsimCoreAlgorithm:
    """Core PARSIM algorithms implementation."""

    @staticmethod
    def parsim_k(
        y,
        u,
        f=20,
        p=20,
        threshold=0.1,
        max_order=np.nan,
        fixed_order=np.nan,
        D_required=False,
        B_recalc=False,
        strict_identifiability=False,
    ):
        """
        PARSIM-K algorithm implementation.

        Parameters:
        -----------
        y : ndarray
            Output data (outputs x time_steps)
        u : ndarray
            Input data (inputs x time_steps)
        f : int
            Future horizon
        p : int
            Past horizon
        threshold : float
            Singular value threshold
        max_order : float
            Maximum order
        fixed_order : float
            Fixed order
        D_required : bool
            Whether D matrix is required
        B_recalc : bool
            Whether to recalculate B matrix
        strict_identifiability : bool
            Raise instead of using the explicit PARSIM-K compatibility fallback
            when a required regression is rank deficient

        Returns:
        --------
        A_K, C, B_K, D, K, A, B, x0, Vn : ndarrays
            System matrices and initial state
        """
        y = 1.0 * np.atleast_2d(y)
        u = 1.0 * np.atleast_2d(u)
        l_, L = y.shape
        m = u[:, 0].size

        if not check_types(threshold, max_order, fixed_order, f, p):
            return (
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.inf,
            )

        threshold, max_order = check_inputs(threshold, max_order, fixed_order, f)

        data = prepare_subspace_data(
            y,
            u,
            future_horizon=f,
            past_offset=p,
            past_block_rows=p,
        )
        y = data.outputs
        u = data.inputs
        Ustd = data.input_scale
        Ystd = data.output_scale
        Yf = data.future_outputs
        Uf = data.future_inputs
        Zp = data.past_data
        markov_estimate = _estimate_predictor_markov_blocks(
            Yf,
            Uf,
            Zp,
            output_count=l_,
            input_count=m,
            future_horizon=f,
            direct_feedthrough=D_required,
            strict=strict_identifiability,
        )
        Gamma_L = markov_estimate.gamma_matrix

        # CRITICAL FIX: Use PARSIM-K specific SVD with Gamma_L (not N4SID's svd_weighted)
        # Reference: master/sippy_unipi/Parsim_methods.py line 233
        U_n, S_n, V_n = ParsimCoreAlgorithm.svd_weighted_k(Uf, Zp, Gamma_L)
        U_n, S_n, V_n = reducingOrder(U_n, S_n, V_n, threshold, max_order)

        n = S_n.size
        S_n_diag = np.diag(S_n)
        Ob_K = np.dot(U_n, np.sqrt(S_n_diag))

        # Estimate A_K carefully
        if l_ * (f - 1) >= n and n > 0:
            try:
                A_K = np.dot(
                    pinv_compiled_svd(Ob_K[0 : l_ * (f - 1), :])
                    if NUMBA_AVAILABLE
                    else np.linalg.pinv(Ob_K[0 : l_ * (f - 1), :]),
                    Ob_K[l_:, :],
                )
            except (np.linalg.LinAlgError, ValueError):
                A_K = np.linalg.pinv(Ob_K[0 : l_ * (f - 1), :]) @ Ob_K[l_:, :]
        else:
            raise ValueError(
                "PARSIM-K future horizon is too short for the identified order"
            )

        C = Ob_K[0:l_, :]

        # CRITICAL FIX: Use simulations_sequence_k for parameter estimation
        # This uses predictor form simulation
        # Reference: master/sippy_unipi/Parsim_methods.py line 240
        K_placeholder = np.zeros((n, l_))
        D_placeholder = np.zeros((l_, m))
        y_sim = ParsimCoreAlgorithm.simulations_sequence_k(
            A_K, C, L, y, u, l_, m, n, K_placeholder, D_placeholder, D_required
        )

        # Solve for parameters using least squares
        vect, _ = _solve_predictor_parameters(
            y_sim,
            y,
            strict=strict_identifiability,
        )
        Y_estimate = np.dot(y_sim, vect)
        Vn = Vn_mat(y.reshape((L * l_, 1)), Y_estimate)

        # Extract parameters from vect
        B_K = vect[0 : n * m, :].reshape((n, m))
        if D_required:
            D = vect[n * m : n * m + l_ * m, :].reshape((l_, m))
            K = vect[n * m + l_ * m : n * m + l_ * m + n * l_, :].reshape((n, l_))
            x0 = vect[n * m + l_ * m + n * l_ : :, :].reshape((n, 1))
        else:
            D = np.zeros((l_, m))
            K = vect[n * m : n * m + n * l_, :].reshape((n, l_))
            x0 = vect[n * m + n * l_ : :, :].reshape((n, 1))

        # Calculate A matrix
        A = A_K + np.dot(K, C)

        # Optional B recalculation using process form
        # Reference: master/sippy_unipi/Parsim_methods.py lines 256-263
        if B_recalc:
            # Helper function to create simulation matrix for B recalc
            def recalc_K(A, C, D, u):
                y_sim = []
                n_ord = A[:, 0].size
                m_input, L_u = u.shape
                l_out = C[:, 0].size
                n_simulations = n_ord + n_ord * m_input
                vect = np.zeros((n_simulations, 1))
                for i in range(n_simulations):
                    vect[i, 0] = 1.0
                    B_i = vect[0 : n_ord * m_input, :].reshape((n_ord, m_input))
                    x0_i = vect[n_ord * m_input : :, :].reshape((n_ord, 1))
                    _, y_i = simulate_ss_system(A, B_i, C, D, u, x0=x0_i)
                    y_sim.append(y_i.reshape((1, L_u * l_out)))
                    vect[i, 0] = 0.0
                y_matrix = 1.0 * y_sim[0]
                for j in range(n_simulations - 1):
                    y_matrix = impile(y_matrix, y_sim[j + 1])
                y_matrix = y_matrix.T
                return y_matrix

            y_sim = recalc_K(A, C, D, u)
            vect, _ = _solve_predictor_parameters(
                y_sim,
                y,
                strict=strict_identifiability,
            )
            Y_estimate = np.dot(y_sim, vect)
            Vn = Vn_mat(y.reshape((L * l_, 1)), Y_estimate)
            B = vect[0 : n * m, :].reshape((n, m))
            x0 = vect[n * m : :, :].reshape((n, 1))
            B_K = B - np.dot(K, D)
        else:
            B = B_K + np.dot(K, D)

        # Rescale back to original units
        for j in range(m):
            B_K[:, j] = B_K[:, j] / Ustd[j]
            D[:, j] = D[:, j] / Ustd[j]
        for j in range(l_):
            K[:, j] = K[:, j] / Ystd[j]
            C[j, :] = C[j, :] * Ystd[j]
            D[j, :] = D[j, :] * Ystd[j]
        B = B_K + np.dot(K, D)

        return A_K, C, B_K, D, K, A, B, x0, Vn

    @staticmethod
    def parsim_s(
        y,
        u,
        f=20,
        p=20,
        threshold=0.1,
        max_order=np.nan,
        fixed_order=np.nan,
        D_required=False,
    ):
        """
        PARSIM-S algorithm implementation.

        Parameters:
        -----------
        y : ndarray
            Output data (outputs x time_steps)
        u : ndarray
            Input data (inputs x time_steps)
        f : int
            Future horizon
        p : int
            Past horizon
        threshold : float
            Singular value threshold
        max_order : float
            Maximum order
        fixed_order : float
            Fixed order
        D_required : bool
            Whether D matrix is required

        Returns:
        --------
        A_K, C, B_K, D, K, A, B, x0, Vn : ndarrays
            System matrices and initial state
        """
        y = 1.0 * np.atleast_2d(y)
        u = 1.0 * np.atleast_2d(u)
        l_, L = y.shape
        m = u[:, 0].size

        if not check_types(threshold, max_order, fixed_order, f, p):
            return (
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.inf,
            )

        threshold, max_order = check_inputs(threshold, max_order, fixed_order, f)

        data = prepare_subspace_data(
            y,
            u,
            future_horizon=f,
            past_offset=p,
        )
        y = data.outputs
        u = data.inputs
        Ustd = data.input_scale
        Ystd = data.output_scale
        Yf = data.future_outputs
        Uf = data.future_inputs
        Zp = data.past_data

        # Initial matrices
        regression_solver = _ReusableRightLeastSquares.factor(impile(Zp, Uf[0:m, :]))
        M = regression_solver.solve(Yf[0:l_, :])
        Gamma_L = M[:, 0 : (m + l_) * f]
        H = M[:, (m + l_) * f :]

        # Helper function for y_tilde estimation S
        def estimating_y_S(H, Uf, Yf, i, m, l_):
            y_tilde = np.dot(H[0:l_, :], Uf[m * i : m * (i + 1), :])
            for j in range(1, i):
                y_tilde = y_tilde + np.dot(
                    H[l_ * j : l_ * (j + 1), :], Uf[m * (i - j) : m * (i - j + 1), :]
                )
            return y_tilde

        # Build matrices for each horizon
        for i in range(1, f):
            y_tilde = estimating_y_S(H, Uf, Yf, i, m, l_)
            M = regression_solver.solve(Yf[l_ * i : l_ * (i + 1)] - y_tilde)
            Gamma_L = impile(Gamma_L, M[:, 0 : (m + l_) * f])
            H = impile(H, M[:, (m + l_) * f :])

        # CRITICAL FIX: Use PARSIM-specific SVD weighting (not N4SID's SVD)
        # Reference: master/sippy_unipi/Parsim_methods.py line 384 (now 459)
        U_n, S_n, V_n = ParsimCoreAlgorithm.svd_weighted_k(Uf, Zp, Gamma_L)
        U_n, S_n, V_n = reducingOrder(U_n, S_n, V_n, threshold, max_order)

        # CRITICAL FIX: Use QR-based Kalman gain estimation
        # Reference: master/sippy_unipi/Parsim_methods.py lines 461-462 (now 386-387)
        A, C, A_K, K, n = ParsimCoreAlgorithm.ak_c_estimating_s_p(
            U_n, S_n, V_n, l_, f, m, Zp, Uf, Yf
        )

        # CRITICAL FIX: Use systematic predictor form simulation
        # Reference: master/sippy_unipi/Parsim_methods.py lines 464-465 (now 389-390)
        y_sim = ParsimCoreAlgorithm.simulations_sequence_s(
            A_K, C, L, K, y, u, l_, m, n, D_required
        )

        # Solve for parameters using least squares
        # Reference: master/sippy_unipi/Parsim_methods.py lines 467-476 (now 392-401)
        vect = np.dot(np.linalg.pinv(y_sim), y.reshape((L * l_, 1)))
        Y_estimate = np.dot(y_sim, vect)
        Vn = Vn_mat(y.reshape((L * l_, 1)), Y_estimate)

        # Extract parameters from vect
        B_K = vect[0 : n * m, :].reshape((n, m))
        if D_required:
            D = vect[n * m : n * m + l_ * m, :].reshape((l_, m))
            x0 = vect[n * m + l_ * m :, :].reshape((n, 1))
        else:
            D = np.zeros((l_, m))
            x0 = vect[n * m :, :].reshape((n, 1))

        # Calculate B matrix
        B = B_K + np.dot(K, D)

        # Rescale back to original units
        for j in range(m):
            B_K[:, j] = B_K[:, j] / Ustd[j]
            D[:, j] = D[:, j] / Ustd[j]
        for j in range(l_):
            K[:, j] = K[:, j] / Ystd[j]
            C[j, :] = C[j, :] * Ystd[j]
            D[j, :] = D[j, :] * Ystd[j]
        B = B_K + np.dot(K, D)

        return A_K, C, B_K, D, K, A, B, x0, Vn

    @staticmethod
    def parsim_p(
        y,
        u,
        f=20,
        p=20,
        threshold=0.1,
        max_order=np.nan,
        fixed_order=np.nan,
        D_required=False,
    ):
        """
        PARSIM-P algorithm implementation with expanding window approach.

        Key difference from PARSIM-S: The Uf window expands with each iteration,
        providing progressively more input information for better parameter estimation.

        Reference: master/sippy_unipi/Parsim_methods.py lines 597-670

        Parameters:
        -----------
        y : ndarray
            Output data (outputs x time_steps)
        u : ndarray
            Input data (inputs x time_steps)
        f : int
            Future horizon
        p : int
            Past horizon
        threshold : float
            Singular value threshold
        max_order : float
            Maximum order
        fixed_order : float
            Fixed order
        D_required : bool
            Whether D matrix is required

        Returns:
        --------
        A_K, C, B_K, D, K, A, B, x0, Vn : ndarrays
            System matrices and initial state
        """
        y = 1.0 * np.atleast_2d(y)
        u = 1.0 * np.atleast_2d(u)
        l_, L = y.shape
        m = u[:, 0].size

        if not check_types(threshold, max_order, fixed_order, f, p):
            return (
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.inf,
            )

        threshold, max_order = check_inputs(threshold, max_order, fixed_order, f)

        data = prepare_subspace_data(
            y,
            u,
            future_horizon=f,
            past_offset=p,
        )
        y = data.outputs
        u = data.inputs
        Ustd = data.input_scale
        Ystd = data.output_scale
        Yf = data.future_outputs
        Uf = data.future_inputs
        Zp = data.past_data

        Gamma_L = _build_parsim_p_gamma_l(Yf, Uf, Zp, f, l_, m)

        # SVD for order estimation - use PARSIM-K weighted SVD
        # Master line 644
        U_n, S_n, V_n = ParsimCoreAlgorithm.svd_weighted_k(Uf, Zp, Gamma_L)
        U_n, S_n, V_n = reducingOrder(U_n, S_n, V_n, threshold, max_order)

        # Use same QR-based K estimation as PARSIM-S
        # Master lines 646-647
        A, C, A_K, K, n = ParsimCoreAlgorithm.ak_c_estimating_s_p(
            U_n, S_n, V_n, l_, f, m, Zp, Uf, Yf
        )

        # Simulation using predictor form (master lines 649-651)
        # Use simulations_sequence_S (K is fixed, estimate B_K, D, x0)
        y_sim = ParsimCoreAlgorithm.simulations_sequence_s(
            A_K, C, L, K, y, u, l_, m, n, D_required
        )

        # Parameter estimation (master lines 652-654)
        vect = np.dot(
            pinv_compiled_svd(y_sim) if NUMBA_AVAILABLE else np.linalg.pinv(y_sim),
            y.reshape((L * l_, 1)),
        )
        Y_estimate = np.dot(y_sim, vect)
        Vn = Vn_mat(y.reshape((L * l_, 1)), Y_estimate)

        # Extract parameters (master lines 655-661)
        B_K = vect[0 : n * m, :].reshape((n, m))
        if D_required:
            D = vect[n * m : n * m + l_ * m, :].reshape((l_, m))
            x0 = vect[n * m + l_ * m :, :].reshape((n, 1))
        else:
            D = np.zeros((l_, m))
            x0 = vect[n * m :, :].reshape((n, 1))

        # Rescale back to original units (master lines 662-668)
        for j in range(m):
            B_K[:, j] = B_K[:, j] / Ustd[j]
            D[:, j] = D[:, j] / Ustd[j]
        for j in range(l_):
            K[:, j] = K[:, j] / Ystd[j]
            C[j, :] = C[j, :] * Ystd[j]
            D[j, :] = D[j, :] * Ystd[j]

        # Calculate B matrix (master line 669)
        B = B_K + np.dot(K, D)

        return A_K, C, B_K, D, K, A, B, x0, Vn

    @staticmethod
    def svd_weighted_k(Uf, Zp, Gamma_L):
        """
        PARSIM-K specific weighted SVD.

        This is different from N4SID's SVD weighting - it uses PARSIM-specific
        weighting based on Z_dot_PIort(Zp, Uf) instead of the N4SID weights.

        Reference: master/sippy_unipi/Parsim_methods.py lines 76-79

        Parameters:
        -----------
        Uf : ndarray
            Future input ordinate sequence
        Zp : ndarray
            Past data matrix (stacked Up and Yp)
        Gamma_L : ndarray
            Extended observability matrix from PARSIM-K iteration

        Returns:
        --------
        U_n : ndarray
            Left singular vectors
        S_n : ndarray
            Singular values
        V_n : ndarray
            Right singular vectors
        """
        from ...utils.simulation_utils import Z_dot_PIort

        # Edge case: Check for empty or degenerate matrices
        if Gamma_L.size == 0 or Gamma_L.shape[0] == 0 or Gamma_L.shape[1] == 0:
            # Return empty SVD components with consistent shapes
            return (
                np.zeros((Gamma_L.shape[0], 0)),
                np.array([]),
                np.zeros((0, Gamma_L.shape[1])),
            )

        try:
            # PARSIM-K weighting: W2 = sqrtm((Zp - Zp*Uf^T*pinv(Uf^T)) * Zp^T)
            W2 = sc.linalg.sqrtm(np.dot(Z_dot_PIort(Zp, Uf), Zp.T)).real

            # Check for NaN or Inf in W2
            if not np.all(np.isfinite(W2)):
                # Fallback to unweighted SVD
                U_n, S_n, V_n = np.linalg.svd(Gamma_L, full_matrices=False)
                return U_n, S_n, V_n

            # Weighted SVD: svd(Gamma_L * W2)
            weighted_matrix = np.dot(Gamma_L, W2)

            # Check for numerical issues
            if not np.all(np.isfinite(weighted_matrix)):
                # Fallback to unweighted SVD
                U_n, S_n, V_n = np.linalg.svd(Gamma_L, full_matrices=False)
                return U_n, S_n, V_n

            U_n, S_n, V_n = np.linalg.svd(weighted_matrix, full_matrices=False)

        except (np.linalg.LinAlgError, ValueError):
            # Fallback to unweighted SVD on any linear algebra errors
            U_n, S_n, V_n = np.linalg.svd(Gamma_L, full_matrices=False)

        return U_n, S_n, V_n

    @staticmethod
    def _simulate_single_parameter_k(
        i, n_simulations, vect, A_K, C, D_required, y, u, l_, m, n, L
    ):
        """
        Simulate a single parameter configuration for PARSIM-K.

        This is a helper function designed to be thread-safe for parallel execution.
        Each call simulates the system with a single unit vector for one parameter.

        Parameters:
        -----------
        i : int
            Parameter index (which parameter to set to 1.0)
        n_simulations : int
            Total number of simulations
        vect : ndarray
            Parameter vector template (will be copied, not modified)
        A_K, C : ndarrays
            System matrices
        D_required : bool
            Whether D matrix is estimated
        y, u : ndarrays
            Output and input data (read-only)
        l_, m, n : ints
            System dimensions
        L : int
            Number of time steps

        Returns:
        --------
        y_hat_flat : ndarray
            Flattened output simulation (1 x L*l_)
        """
        from ...utils.simulation_utils import ss_lsim_predictor_form

        # Create local copy of vect to avoid race conditions
        vect_local = vect.copy()
        vect_local[i, 0] = 1.0

        if D_required:
            B_K = vect_local[0 : n * m, :].reshape((n, m))
            D_i = vect_local[n * m : n * m + l_ * m, :].reshape((l_, m))
            K_i = vect_local[n * m + l_ * m : n * m + l_ * m + n * l_, :].reshape(
                (n, l_)
            )
            x0 = vect_local[n * m + l_ * m + n * l_ : :, :].reshape((n, 1))
        else:
            B_K = vect_local[0 : n * m, :].reshape((n, m))
            D_i = np.zeros((l_, m))
            K_i = vect_local[n * m : n * m + n * l_, :].reshape((n, l_))
            x0 = vect_local[n * m + n * l_ : :, :].reshape((n, 1))

        # Simulate using predictor form
        _, y_hat = ss_lsim_predictor_form(A_K, B_K, C, D_i, K_i, y, u, x0)
        return y_hat.reshape((1, L * l_))

    @staticmethod
    def _simulate_single_parameter_s(
        i, n_simulations, vect, A_K, C, K, D_required, y, u, l_, m, n, L
    ):
        """
        Simulate a single parameter configuration for PARSIM-S.

        This is a helper function designed to be thread-safe for parallel execution.
        Each call simulates the system with a single unit vector for one parameter.
        Note: K is FIXED (not estimated) unlike PARSIM-K.

        Parameters:
        -----------
        i : int
            Parameter index (which parameter to set to 1.0)
        n_simulations : int
            Total number of simulations
        vect : ndarray
            Parameter vector template (will be copied, not modified)
        A_K, C, K : ndarrays
            System matrices (K is fixed)
        D_required : bool
            Whether D matrix is estimated
        y, u : ndarrays
            Output and input data (read-only)
        l_, m, n : ints
            System dimensions
        L : int
            Number of time steps

        Returns:
        --------
        y_hat_flat : ndarray
            Flattened output simulation (1 x L*l_)
        """
        from ...utils.simulation_utils import SS_lsim_predictor_form

        # Create local copy of vect to avoid race conditions
        vect_local = vect.copy()
        vect_local[i, 0] = 1.0

        if D_required:
            B_K = vect_local[0 : n * m, :].reshape((n, m))
            D = vect_local[n * m : n * m + l_ * m, :].reshape((l_, m))
            x0 = vect_local[n * m + l_ * m :, :].reshape((n, 1))
        else:
            B_K = vect_local[0 : n * m, :].reshape((n, m))
            D = np.zeros((l_, m))
            x0 = vect_local[n * m :, :].reshape((n, 1))

        # Simulate predictor form with FIXED K
        _, y_hat = SS_lsim_predictor_form(A_K, B_K, C, D, K, y, u, x0)
        return y_hat.reshape((1, L * l_))

    @staticmethod
    def simulations_sequence_k(A_K, C, L, y, u, l_, m, n, K, D, D_required=False):
        """
        Create simulation matrix for PARSIM-K parameter estimation.

        This function creates a regression matrix by simulating the system
        with different unit vectors for B_K, K, D, and x0 parameters.
        Uses predictor form simulation: x[i+1] = A_K*x[i] + B_K*u[i] + K*y[i]

        PERFORMANCE: Uses parallel execution via joblib when available and
        n_simulations >= 20, achieving 3-6x speedup on multi-core systems.

        Reference: master/sippy_unipi/Parsim_methods.py lines 82-120

        Parameters:
        -----------
        A_K : ndarray
            State matrix in predictor form (n x n)
        C : ndarray
            Output matrix (l x n)
        L : int
            Number of time steps
        y : ndarray
            Output data (l x L)
        u : ndarray
            Input data (m x L)
        l_ : int
            Number of outputs
        m : int
            Number of inputs
        n : int
            Model order
        K : ndarray
            Kalman gain (n x l) - placeholder, overwritten in simulations
        D : ndarray
            Feedthrough matrix (l x m) - placeholder
        D_required : bool
            Whether to estimate D matrix

        Returns:
        --------
        y_matrix : ndarray
            Simulation matrix (L*l x n_simulations) - transposed for least squares
        """
        from ...utils.simulation_utils import impile

        # Calculate number of simulations needed
        if D_required:
            # Parameters to estimate: B_K (n*m), D (l*m), K (n*l), x0 (n)
            n_simulations = n * m + l_ * m + n * l_ + n
        else:
            # Parameters to estimate: B_K (n*m), K (n*l), x0 (n)
            n_simulations = n * m + n * l_ + n

        # Create parameter vector template
        vect = np.zeros((n_simulations, 1))

        # Adaptive threshold: use parallel for n_simulations >= 20
        # Below this threshold, overhead dominates any speedup
        use_parallel = JOBLIB_AVAILABLE and n_simulations >= 20

        if use_parallel:
            # Parallel execution using joblib with processes for true parallelism
            # prefer="processes" avoids GIL and achieves real CPU parallelism
            y_sim_list = Parallel(n_jobs=-1, prefer="processes")(
                delayed(ParsimCoreAlgorithm._simulate_single_parameter_k)(
                    i, n_simulations, vect, A_K, C, D_required, y, u, l_, m, n, L
                )
                for i in range(n_simulations)
            )
        else:
            # Sequential execution fallback
            from ...utils.simulation_utils import ss_lsim_predictor_form

            y_sim_list = []
            for i in range(n_simulations):
                vect[i, 0] = 1.0

                if D_required:
                    B_K = vect[0 : n * m, :].reshape((n, m))
                    D_i = vect[n * m : n * m + l_ * m, :].reshape((l_, m))
                    K_i = vect[n * m + l_ * m : n * m + l_ * m + n * l_, :].reshape(
                        (n, l_)
                    )
                    x0 = vect[n * m + l_ * m + n * l_ : :, :].reshape((n, 1))
                else:
                    B_K = vect[0 : n * m, :].reshape((n, m))
                    D_i = np.zeros((l_, m))
                    K_i = vect[n * m : n * m + n * l_, :].reshape((n, l_))
                    x0 = vect[n * m + n * l_ : :, :].reshape((n, 1))

                # Simulate using predictor form
                _, y_hat = ss_lsim_predictor_form(A_K, B_K, C, D_i, K_i, y, u, x0)
                y_sim_list.append(y_hat.reshape((1, L * l_)))
                vect[i, 0] = 0.0

        # Stack all simulations into a matrix
        # Each y_sim_list[i] has shape (1, L*l_), impile stacks vertically giving (n_simulations, L*l_)
        # Transpose to (L*l_, n_simulations) for least squares: pinv(y_sim) @ y
        y_matrix = 1.0 * y_sim_list[0]
        for j in range(n_simulations - 1):
            y_matrix = impile(y_matrix, y_sim_list[j + 1])
        y_matrix = y_matrix.T

        return y_matrix

    @staticmethod
    def ak_c_estimating_s_p(U_n, S_n, V_n, l_, f, m, Zp, Uf, Yf):
        """
        Estimate A, C, A_K, and K matrices for PARSIM-S and PARSIM-P using QR decomposition.

        This function uses rigorous QR decomposition to estimate the Kalman gain K,
        which is the correct approach from the reference implementation.

        Reference: master/sippy_unipi/Parsim_methods.py lines 85-101 (AK_C_estimating_S_P function)

        Parameters:
        -----------
        U_n, S_n, V_n : ndarrays
            SVD decomposition from svd_weighted_k
        l_ : int
            Number of outputs
        f : int
            Future horizon
        m : int
            Number of inputs
        Zp, Uf, Yf : ndarrays
            Data matrices from ordinate sequences

        Returns:
        --------
        A : ndarray
            State matrix (n x n)
        C : ndarray
            Output matrix (l x n)
        A_K : ndarray
            Predictor form state matrix (n x n)
        K : ndarray
            Kalman gain matrix (n x l)
        n : int
            Model order
        """

        n = S_n.size

        # Construct observability matrix
        Ob_f = np.dot(U_n, np.diag(np.sqrt(S_n)))

        # Estimate A from observability matrix shift property
        A = np.dot(
            pinv_compiled_svd(Ob_f[0 : l_ * (f - 1), :])
            if NUMBA_AVAILABLE
            else np.linalg.pinv(Ob_f[0 : l_ * (f - 1), :]),
            Ob_f[l_:, :],
        )

        # Extract C from first block of observability matrix
        C = Ob_f[0:l_, :]

        # QR-based Kalman gain estimation
        # Stack [Zp; Uf; Yf] and perform QR decomposition
        stacked_matrix = impile(impile(Zp, Uf), Yf).T
        # The R factor must extend past the (2m+l)*f block for G_f to be
        # non-empty, i.e. enough windowed samples for the innovation estimate.
        required = (2 * m + l_) * f + l_
        if stacked_matrix.shape[0] < required:
            raise ValueError(
                f"Insufficient data for PARSIM QR step: {stacked_matrix.shape[0]} "
                f"windowed samples, need at least {required}. "
                "Reduce ss_f/ss_p or provide more samples."
            )
        Q, R = np.linalg.qr(stacked_matrix)
        Q = Q.T
        R = R.T

        # Extract relevant block from R matrix
        # G_f contains innovation covariance information
        G_f = R[(2 * m + l_) * f :, (2 * m + l_) * f :]
        F = G_f[0:l_, 0:l_]

        # Compute Kalman gain K using QR decomposition result
        # K = Ob_f^+ * G_f[l_:, 0:l_] * F^-1
        K = np.dot(
            (
                pinv_compiled_svd(Ob_f[0 : l_ * (f - 1), :])
                if NUMBA_AVAILABLE
                else np.linalg.pinv(Ob_f[0 : l_ * (f - 1), :])
            )
            @ G_f[l_:, 0:l_],
            np.linalg.inv(F),
        )

        # Compute predictor form A_K = A - K*C
        A_K = A - np.dot(K, C)

        return A, C, A_K, K, n

    @staticmethod
    def simulations_sequence_s(A_K, C, L, K, y, u, l_, m, n, D_required):
        """
        Systematic simulation for PARSIM-S parameter estimation using predictor form.

        Simulates the predictor form system with unit vectors for all parameters
        (B_K, D, x0) to build regression matrix for least squares. Note that K
        is FIXED (already estimated), unlike PARSIM-K where K is also estimated.

        PERFORMANCE: Uses parallel execution via joblib when available and
        n_simulations >= 20, achieving 3-6x speedup on multi-core systems.

        Reference: master/sippy_unipi/Parsim_methods.py lines 48-82 (simulations_sequence_S function)

        Parameters:
        -----------
        A_K : ndarray
            Predictor form A matrix (n x n)
        C : ndarray
            Output matrix (l x n)
        L : int
            Number of time points
        K : ndarray
            Kalman gain matrix (n x l) - FIXED, not estimated
        y, u : ndarrays
            Output (l x L) and input (m x L) data
        l_, m, n : ints
            System dimensions (outputs, inputs, states)
        D_required : bool
            Whether D matrix is included in estimation

        Returns:
        --------
        y_matrix : ndarray
            Simulation matrix (L*l x n_simulations) for least squares
        """
        from ...utils.simulation_utils import impile

        # Calculate number of simulations needed
        if D_required:
            # Parameters to estimate: B_K (n*m), D (l*m), x0 (n)
            # Note: K is NOT estimated, it's fixed
            n_simulations = n * m + l_ * m + n
        else:
            # Parameters to estimate: B_K (n*m), x0 (n)
            n_simulations = n * m + n

        # Create parameter vector template
        vect = np.zeros((n_simulations, 1))

        # Adaptive threshold: use parallel for n_simulations >= 20
        # Below this threshold, overhead dominates any speedup
        use_parallel = JOBLIB_AVAILABLE and n_simulations >= 20

        if use_parallel:
            # Parallel execution using joblib with processes for true parallelism
            # prefer="processes" avoids GIL and achieves real CPU parallelism
            y_sim_list = Parallel(n_jobs=-1, prefer="processes")(
                delayed(ParsimCoreAlgorithm._simulate_single_parameter_s)(
                    i, n_simulations, vect, A_K, C, K, D_required, y, u, l_, m, n, L
                )
                for i in range(n_simulations)
            )
        else:
            # Sequential execution fallback
            from ...utils.simulation_utils import SS_lsim_predictor_form

            y_sim_list = []
            for i in range(n_simulations):
                vect[i, 0] = 1.0

                if D_required:
                    B_K = vect[0 : n * m, :].reshape((n, m))
                    D = vect[n * m : n * m + l_ * m, :].reshape((l_, m))
                    x0 = vect[n * m + l_ * m :, :].reshape((n, 1))
                else:
                    B_K = vect[0 : n * m, :].reshape((n, m))
                    D = np.zeros((l_, m))
                    x0 = vect[n * m :, :].reshape((n, 1))

                # Simulate predictor form with FIXED K
                _, y_hat = SS_lsim_predictor_form(A_K, B_K, C, D, K, y, u, x0)
                y_sim_list.append(y_hat.reshape((1, L * l_)))

                vect[i, 0] = 0.0

        # Stack all simulations into regression matrix
        y_matrix = 1.0 * y_sim_list[0]
        for j in range(n_simulations - 1):
            y_matrix = impile(y_matrix, y_sim_list[j + 1])

        y_matrix = y_matrix.T
        return y_matrix
