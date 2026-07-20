"""
Core subspace identification algorithms implementation.
"""

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import numpy as np
from numpy.linalg import pinv
from scipy import signal

from ...utils.simulation_utils import (
    K_calc,
    Vn_mat,
    Z_dot_PIort,
    check_inputs,
    check_types,
    impile,
    reducingOrder,
    simulate_ss_system,
    ss_lsim_predictor_form,
)
from .subspace_data import SubspaceData, prepare_subspace_data

# Import compiled utilities for performance
try:
    from ...utils.compiled_utils import (
        NUMBA_AVAILABLE,
        Z_dot_PIort_compiled,
        covariance_symmetric_compiled,
        pinv_compiled_svd,
        subspace_weighted_svd_compiled,
    )
except ImportError:
    NUMBA_AVAILABLE = False
    Z_dot_PIort_compiled = None
    covariance_symmetric_compiled = None
    subspace_weighted_svd_compiled = None


def _causal_prediction_errors(A, B, C, D, K, y, u, initial_state, use_compiled=True):
    predictor_a = A - K @ C
    predictor_b = B - K @ D
    if use_compiled:
        _, prediction = ss_lsim_predictor_form(
            predictor_a,
            predictor_b,
            C,
            D,
            K,
            y,
            u,
            np.asarray(initial_state, dtype=float).reshape(-1, 1),
        )
    else:
        predictor_input = np.vstack((u, y)).T
        input_matrix = np.hstack((predictor_b, K))
        feedthrough = np.hstack((D, np.zeros((C.shape[0], C.shape[0]))))
        _, prediction, _ = signal.dlsim(
            (predictor_a, input_matrix, C, feedthrough, 1.0),
            predictor_input,
            x0=initial_state,
        )
        prediction = prediction.T
    return y - prediction


def _innovation_information_criterion(errors, parameter_count, method):
    sample_count = errors.shape[1]
    if sample_count == 0 or not np.all(np.isfinite(errors)):
        return np.inf

    covariance = errors @ errors.T / sample_count
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(float(eigenvalues[-1]), np.finfo(np.float64).tiny)
    eigenvalue_floor = max(
        np.finfo(np.float64).tiny,
        np.finfo(np.float64).eps * scale,
    )
    log_determinant = float(np.sum(np.log(np.maximum(eigenvalues, eigenvalue_floor))))
    likelihood = sample_count * log_determinant

    if method == "AIC":
        return likelihood + 2 * parameter_count
    if method == "AICc":
        denominator = sample_count - parameter_count - 1
        if denominator <= 0:
            return np.inf
        correction = 2 * parameter_count * (parameter_count + 1) / denominator
        return likelihood + 2 * parameter_count + correction
    if method == "BIC":
        return likelihood + parameter_count * np.log(sample_count)
    raise ValueError(f"Unknown method: {method}")


@dataclass(frozen=True)
class _LQProjection:
    projector: np.ndarray
    past_data: np.ndarray
    materialize_first: bool = False

    def materialize(self):
        return self.projector @ self.past_data

    def state_sequence(self, observability_inverse):
        if self.materialize_first:
            return observability_inverse @ self.materialize()
        return (observability_inverse @ self.projector) @ self.past_data


@dataclass(frozen=True)
class ORTProjectionDiagnostics:
    usable: bool
    reason: str | None
    reference_rank: int
    reference_rows: int
    projected_input_rank: int
    projected_input_rows: int
    deterministic_regressor_rank: int
    deterministic_regressor_rows: int


@dataclass(frozen=True)
class _ReferenceRowProjection:
    diagnostics: ORTProjectionDiagnostics
    reference_hankel: np.ndarray
    reference_factor: np.ndarray
    coefficient_map: np.ndarray
    compact_future_inputs: np.ndarray
    compact_past_data: np.ndarray
    compact_future_outputs: np.ndarray

    def materialize(self):
        return self.coefficient_map @ self.reference_hankel

    def expand_compact(self, compact):
        reference_coordinates = np.linalg.solve(
            self.reference_factor,
            self.reference_hankel,
        )
        return compact @ reference_coordinates


@dataclass(frozen=True)
class _TwoStageORTProjection:
    diagnostics: ORTProjectionDiagnostics
    reference_projection: _ReferenceRowProjection | None
    projection: _LQProjection | None
    compact_projection: np.ndarray | None
    compact_moesp: np.ndarray | None
    compact_projected_outputs: np.ndarray | None


def _warn_for_unusable_reference(diagnostics):
    if diagnostics.reason == "reference_missing":
        return
    warnings.warn(
        "Measured exogenous reference is unusable for two-stage ORT "
        f"({diagnostics.reason}); using the predictor-form estimator",
        UserWarning,
        stacklevel=2,
    )


def _numerically_full_rank(triangular, sample_count):
    diagonal = np.abs(np.diag(triangular))
    if diagonal.size == 0:
        return False
    scale = float(np.max(diagonal))
    tolerance = max(sample_count, *triangular.shape) * np.finfo(np.float64).eps * scale
    return scale > 0.0 and bool(np.all(diagonal > tolerance))


def _explicit_subspace_projection(Uf, Zp, Yf):
    if NUMBA_AVAILABLE and Z_dot_PIort_compiled is not None:
        projected_outputs = Z_dot_PIort_compiled(Yf, Uf)
        projected_past = Z_dot_PIort_compiled(Zp, Uf)
    else:
        projected_outputs = Z_dot_PIort(Yf, Uf)
        projected_past = Z_dot_PIort(Zp, Uf)
    if NUMBA_AVAILABLE and pinv_compiled_svd is not None:
        projected_past_inverse = pinv_compiled_svd(projected_past)
    else:
        projected_past_inverse = pinv(projected_past)
    projector = projected_outputs @ projected_past_inverse
    projection = _LQProjection(projector, Zp, materialize_first=True)
    materialized = projection.materialize()
    return (
        projection,
        materialized,
        Z_dot_PIort(materialized, Uf),
        projected_outputs,
    )


def _lq_compress_subspace_data(Uf, Zp, Yf):
    input_rows = Uf.shape[0]
    past_rows = Zp.shape[0]
    stacked = np.vstack((Uf, Zp, Yf))
    L = np.linalg.qr(stacked.T, mode="r").T

    past_start = input_rows
    output_start = input_rows + past_rows
    if L.shape[1] < output_start:
        return _explicit_subspace_projection(Uf, Zp, Yf)

    L11 = L[:input_rows, :input_rows]
    L22 = L[past_start:output_start, input_rows:output_start]
    if not (
        _numerically_full_rank(L11, stacked.shape[1])
        and _numerically_full_rank(L22, stacked.shape[1])
    ):
        return _explicit_subspace_projection(Uf, Zp, Yf)

    L32 = L[output_start:, input_rows:output_start]
    projector = L32 @ pinv(L22)
    compact_past = L[past_start:output_start, :output_start]
    compact_projection = projector @ compact_past
    compact_moesp = compact_projection[:, input_rows:]
    compact_projected_outputs = L[output_start:, input_rows:]
    return (
        _LQProjection(projector, Zp),
        compact_projection,
        compact_moesp,
        compact_projected_outputs,
    )


def _unusable_ort_projection(data, reason):
    return _TwoStageORTProjection(
        diagnostics=ORTProjectionDiagnostics(
            usable=False,
            reason=reason,
            reference_rank=data.ranks.reference_rank,
            reference_rows=data.ranks.reference_rows,
            projected_input_rank=0,
            projected_input_rows=(
                data.past_inputs.shape[0] + data.future_inputs.shape[0]
            ),
            deterministic_regressor_rank=0,
            deterministic_regressor_rows=(
                data.future_inputs.shape[0] + data.past_data.shape[0]
            ),
        ),
        reference_projection=None,
        projection=None,
        compact_projection=None,
        compact_moesp=None,
        compact_projected_outputs=None,
    )


def _project_onto_reference_row_space(data: SubspaceData):
    if data.past_references is None or data.future_references is None:
        return _unusable_ort_projection(data, "reference_missing")
    if not data.ranks.reference_informative:
        return _unusable_ort_projection(data, "reference_rank_deficient")

    reference_hankel = np.vstack((data.past_references, data.future_references))
    signals = np.vstack((data.future_inputs, data.past_data, data.future_outputs))
    stacked = np.vstack((reference_hankel, signals))
    factor = np.linalg.qr(stacked.T, mode="r").T
    reference_rows = reference_hankel.shape[0]
    if factor.shape[1] < reference_rows:
        return _unusable_ort_projection(data, "reference_sample_count_insufficient")

    reference_factor = factor[:reference_rows, :reference_rows]
    if not _numerically_full_rank(reference_factor, data.usable_columns):
        return _unusable_ort_projection(data, "reference_rank_deficient")

    compact = factor[reference_rows:, :reference_rows]
    future_input_rows = data.future_inputs.shape[0]
    past_rows = data.past_data.shape[0]
    compact_future_inputs = compact[:future_input_rows]
    compact_past_data = compact[future_input_rows : future_input_rows + past_rows]
    compact_future_outputs = compact[future_input_rows + past_rows :]

    past_input_rows = data.past_inputs.shape[0]
    projected_input = np.vstack(
        (compact_past_data[:past_input_rows], compact_future_inputs)
    )
    projected_input_rank = int(np.linalg.matrix_rank(projected_input))
    projected_input_rows = projected_input.shape[0]
    deterministic_regressor = np.vstack((compact_future_inputs, compact_past_data))
    deterministic_regressor_rank = int(np.linalg.matrix_rank(deterministic_regressor))
    deterministic_regressor_rows = deterministic_regressor.shape[0]
    diagnostics = ORTProjectionDiagnostics(
        usable=(
            projected_input_rank == projected_input_rows
            and deterministic_regressor_rank == deterministic_regressor_rows
        ),
        reason=None,
        reference_rank=data.ranks.reference_rank,
        reference_rows=reference_rows,
        projected_input_rank=projected_input_rank,
        projected_input_rows=projected_input_rows,
        deterministic_regressor_rank=deterministic_regressor_rank,
        deterministic_regressor_rows=deterministic_regressor_rows,
    )
    if projected_input_rank < projected_input_rows:
        diagnostics = replace(
            diagnostics,
            reason="projected_input_rank_deficient",
        )
    elif deterministic_regressor_rank < deterministic_regressor_rows:
        diagnostics = replace(
            diagnostics,
            reason="reference_deterministic_regressor_rank_deficient",
        )

    coefficient_map = np.linalg.solve(reference_factor.T, compact.T).T
    return _ReferenceRowProjection(
        diagnostics=diagnostics,
        reference_hankel=reference_hankel,
        reference_factor=reference_factor,
        coefficient_map=coefficient_map,
        compact_future_inputs=compact_future_inputs,
        compact_past_data=compact_past_data,
        compact_future_outputs=compact_future_outputs,
    )


def _two_stage_ort_projection(data: SubspaceData):
    first_stage = _project_onto_reference_row_space(data)
    if isinstance(first_stage, _TwoStageORTProjection):
        return first_stage
    if not first_stage.diagnostics.usable:
        return _TwoStageORTProjection(
            diagnostics=first_stage.diagnostics,
            reference_projection=first_stage,
            projection=None,
            compact_projection=None,
            compact_moesp=None,
            compact_projected_outputs=None,
        )

    second_stage = _lq_compress_subspace_data(
        first_stage.compact_future_inputs,
        first_stage.compact_past_data,
        first_stage.compact_future_outputs,
    )
    compact_state_projection = second_stage[0].materialize()
    full_projection = _LQProjection(
        np.linalg.solve(
            first_stage.reference_factor.T,
            compact_state_projection.T,
        ).T,
        first_stage.reference_hankel,
    )
    return _TwoStageORTProjection(
        diagnostics=first_stage.diagnostics,
        reference_projection=first_stage,
        projection=full_projection,
        compact_projection=second_stage[1],
        compact_moesp=second_stage[2],
        compact_projected_outputs=second_stage[3],
    )


def _weighted_projection_svd(
    compact_projection,
    compact_moesp,
    compact_projected_outputs,
    weights,
):
    if weights == "MOESP":
        weighting = None
        U_n, S_n, V_n = np.linalg.svd(compact_moesp, full_matrices=False)
    elif weights == "CVA":
        covariance_product = compact_projected_outputs @ compact_projected_outputs.T
        covariance = 0.5 * (covariance_product + covariance_product.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        largest_eigenvalue = max(float(eigenvalues[-1]), 0.0)
        tolerance = (
            max(covariance.shape) * np.finfo(np.float64).eps * largest_eigenvalue
        )
        retained = eigenvalues > tolerance
        if not np.any(retained):
            warnings.warn("CVA weighting failed, falling back to N4SID")
            weighting = None
            U_n, S_n, V_n = np.linalg.svd(compact_projection, full_matrices=False)
        else:
            retained_vectors = eigenvectors[:, retained]
            retained_eigenvalues = eigenvalues[retained]
            square_roots = np.sqrt(retained_eigenvalues)
            weighting = (retained_vectors * square_roots) @ retained_vectors.T
            inverse_square_root = (
                retained_vectors * (1.0 / square_roots)
            ) @ retained_vectors.T
            U_n, S_n, V_n = np.linalg.svd(
                inverse_square_root @ compact_moesp,
                full_matrices=False,
            )
    elif weights == "N4SID":
        weighting = None
        U_n, S_n, V_n = np.linalg.svd(compact_projection, full_matrices=False)
    else:
        raise ValueError(f"Unknown weighting method: {weights}")
    return U_n, S_n, V_n, weighting


class SubspaceCoreAlgorithm:
    """Core subspace identification algorithms implementation."""

    @staticmethod
    def svd_weighted(y, u, f, l, weights="N4SID", data: SubspaceData | None = None):
        """
        Perform weighted SVD for subspace algorithms.

        Parameters:
        -----------
        y : ndarray
            Output data (outputs x time_steps)
        u : ndarray
            Input data (inputs x time_steps)
        f : int
            Future horizon
        l : int
            Number of outputs
        weights : str
            Weighting method ('N4SID', 'MOESP', 'CVA')

        Returns:
        --------
        U_n, S_n, V_n : ndarray
            SVD components
        W1 : ndarray or None
            Square-root factor of the CVA output weighting (None means the
            weighting is the identity); the SVD is taken of ``W1^-1 O_i``.
        projection : _LQProjection
            Compact map used to reconstruct the selected-order state sequence.
        """
        if data is None:
            data = prepare_subspace_data(
                y,
                u,
                future_horizon=f,
                past_offset=f,
                scale=False,
            )
        Yf = data.future_outputs
        Uf = data.future_inputs
        Zp = data.past_data

        projection, compact_projection, compact_moesp, compact_projected_outputs = (
            _lq_compress_subspace_data(Uf, Zp, Yf)
        )

        U_n, S_n, V_n, W1 = _weighted_projection_svd(
            compact_projection,
            compact_moesp,
            compact_projected_outputs,
            weights,
        )

        return U_n, S_n, V_n, W1, projection

    @staticmethod
    def svd_weighted_ort(data: SubspaceData, weights="N4SID"):
        ort = _two_stage_ort_projection(data)
        if not ort.diagnostics.usable:
            _warn_for_unusable_reference(ort.diagnostics)
            return None, ort
        decomposition = _weighted_projection_svd(
            ort.compact_projection,
            ort.compact_moesp,
            ort.compact_projected_outputs,
            weights,
        )
        return (*decomposition, ort.projection), ort

    @staticmethod
    def algorithm_1(
        y,
        u,
        l,
        m,
        f,
        N,
        U_n,
        S_n,
        V_n,
        W1,
        projection,
        threshold,
        max_order,
        D_required,
        regression_y=None,
        regression_u=None,
    ):
        """
        Algorithm 1 from subspace identification literature.

        Parameters:
        -----------
        y : ndarray
            Output data
        u : ndarray
            Input data
        l, m, f, N : int
            System dimensions
        U_n, S_n, V_n : ndarray
            SVD components
        W1 : ndarray or None
            Weighting matrix
        projection : _LQProjection
            Compact map from past data to the oblique projection.
        threshold : float
            Truncation threshold
        max_order : int
            Maximum order
        D_required : bool
            Whether D matrix is required

        Returns:
        --------
        Ob : ndarray
            Observability matrix
        X_fd : ndarray
            State sequence
        M : ndarray
            System matrix
        n : int
            System order
        residuals : ndarray
            Residuals
        """
        U_n, S_n, V_n = reducingOrder(U_n, S_n, V_n, threshold, max_order)
        V_n = V_n.T
        n = S_n.size
        sqrt_singular = np.diag(np.sqrt(S_n))

        if W1 is None:  # W1 is identity
            Ob = np.dot(U_n, sqrt_singular)
        else:
            # W1 holds the square-root factor; undoing the inverse-sqrt
            # weighting is a plain multiplication.
            Ob = np.dot(W1, np.dot(U_n, sqrt_singular))

        # Fast pinv for Ob
        try:
            Ob_pinv = (
                pinv_compiled_svd(Ob)
                if NUMBA_AVAILABLE and pinv_compiled_svd is not None
                else np.linalg.pinv(Ob)
            )
        except Exception:
            Ob_pinv = np.linalg.pinv(Ob)
        X_fd = projection.state_sequence(Ob_pinv)
        # Ensure contiguous memory for optimal performance with compiled functions
        X_fd_slice1 = np.ascontiguousarray(X_fd[:, 1:N])
        y_slice = np.ascontiguousarray(
            y[:, f : f + N - 1] if regression_y is None else regression_y[:, : N - 1]
        )
        Sxterm = impile(X_fd_slice1, y_slice)

        X_fd_slice2 = np.ascontiguousarray(X_fd[:, 0 : N - 1])
        u_slice = np.ascontiguousarray(
            u[:, f : f + N - 1] if regression_u is None else regression_u[:, : N - 1]
        )
        Dxterm = impile(X_fd_slice2, u_slice)

        if D_required:
            try:
                Dxinv = (
                    pinv_compiled_svd(Dxterm)
                    if NUMBA_AVAILABLE and pinv_compiled_svd is not None
                    else np.linalg.pinv(Dxterm)
                )
            except Exception:
                Dxinv = np.linalg.pinv(Dxterm)
            M = np.dot(Sxterm, Dxinv)
        else:
            M = np.zeros((n + l, n + m))
            try:
                Dxinv = (
                    pinv_compiled_svd(Dxterm)
                    if NUMBA_AVAILABLE and pinv_compiled_svd is not None
                    else np.linalg.pinv(Dxterm)
                )
            except Exception:
                Dxinv = np.linalg.pinv(Dxterm)
            M[0:n, :] = np.dot(Sxterm[0:n], Dxinv)
            try:
                Dxinv_state = (
                    pinv_compiled_svd(Dxterm[0:n, :])
                    if NUMBA_AVAILABLE and pinv_compiled_svd is not None
                    else np.linalg.pinv(Dxterm[0:n, :])
                )
            except Exception:
                Dxinv_state = np.linalg.pinv(Dxterm[0:n, :])
            M[n::, 0:n] = np.dot(Sxterm[n::], Dxinv_state)

        residuals = Sxterm - np.dot(M, Dxterm)
        return Ob, X_fd, M, n, residuals

    @staticmethod
    def force_a_stability(M, n, Ob, l, X_fd, N, u, f):
        """
        Force A matrix stability if needed.

        Parameters:
        -----------
        M : ndarray
            System matrix
        n : int
            System order
        Ob : ndarray
            Observability matrix
        l : int
            Number of outputs
        X_fd : ndarray
            State sequence
        N : int
            Number of data points
        u : ndarray
            Input data
        f : int
            Future horizon

        Returns:
        --------
        M : ndarray
            Modified system matrix
        res : ndarray
            Residuals
        Forced_A : bool
            Whether A was forced stable
        """
        Forced_A = False
        if np.max(np.abs(np.linalg.eigvals(M[0:n, 0:n]))) >= 1.0:
            Forced_A = True
            warnings.warn("Forcing A stability")
            try:
                Ob_pinv = (
                    pinv_compiled_svd(Ob)
                    if NUMBA_AVAILABLE and pinv_compiled_svd is not None
                    else np.linalg.pinv(Ob)
                )
            except Exception:
                Ob_pinv = np.linalg.pinv(Ob)
            M[0:n, 0:n] = np.dot(Ob_pinv, impile(Ob[l::, :], np.zeros((l, n))))

            # Ensure contiguous memory for sliced arrays
            u_slice_det = np.ascontiguousarray(u[:, f : f + N - 1])
            if np.linalg.matrix_rank(u_slice_det) == u_slice_det.shape[0]:
                X_fd_next = np.ascontiguousarray(X_fd[:, 1:N])
                X_fd_curr = np.ascontiguousarray(X_fd[:, 0 : N - 1])
                try:
                    Uinv = (
                        pinv_compiled_svd(u_slice_det)
                        if NUMBA_AVAILABLE and pinv_compiled_svd is not None
                        else np.linalg.pinv(u_slice_det)
                    )
                except Exception:
                    Uinv = np.linalg.pinv(u_slice_det)
                B_new = np.dot(X_fd_next - np.dot(M[0:n, 0:n], X_fd_curr), Uinv)
                M[0:n, n::] = B_new
            else:
                warnings.warn("Cannot compute B matrix due to singular input data")

        # Ensure contiguous memory for residual calculation
        X_fd_next = np.ascontiguousarray(X_fd[:, 1:N])
        X_fd_curr = np.ascontiguousarray(X_fd[:, 0 : N - 1])
        u_slice_res = np.ascontiguousarray(u[:, f : f + N - 1])
        res = (
            X_fd_next
            - np.dot(M[0:n, 0:n], X_fd_curr)
            - np.dot(M[0:n, n::], u_slice_res)
        )
        return M, res, Forced_A

    @staticmethod
    def extract_matrices(M, n):
        """
        Extract state-space matrices from augmented system matrix.

        Parameters:
        -----------
        M : ndarray
            System matrix
        n : int
            System order

        Returns:
        --------
        A, B, C, D : ndarray
            State-space matrices
        """
        A = M[0:n, 0:n]
        B = M[0:n, n::]
        C = M[n::, 0:n]
        D = M[n::, n::]
        return A, B, C, D

    @staticmethod
    def olsims(
        y,
        u,
        f,
        weights="N4SID",
        threshold=0.1,
        max_order=np.nan,
        fixed_order=np.nan,
        D_required=False,
        A_stability=False,
    ):
        """
        Main subspace identification implementation.

        Parameters:
        -----------
        y : ndarray
            Output data (outputs x time_steps)
        u : ndarray
            Input data (inputs x time_steps)
        f : int
            Future horizon
        weights : str
            Weighting method ('N4SID', 'MOESP', 'CVA')
        threshold : float
            Truncation threshold
        max_order : float or int
            Maximum order
        fixed_order : float or int
            Fixed order
        D_required : bool
            Whether D matrix is required
        A_stability : bool
            Whether to force A stability

        Returns:
        --------
        A, B, C, D : ndarray
            State-space matrices
        Vn : float
            Noise variance
        Q, R, S : ndarray
            Covariance matrices
        K : ndarray
            Kalman gain
        """
        if not check_types(threshold, max_order, fixed_order, f):
            raise ValueError("Invalid parameters for subspace identification")

        threshold, max_order = check_inputs(threshold, max_order, fixed_order, f)
        data = prepare_subspace_data(
            y,
            u,
            future_horizon=f,
            past_offset=f,
        )
        y = data.outputs
        u = data.inputs
        l, L = y.shape
        m = u.shape[0]
        N = data.usable_columns
        Ustd = data.input_scale
        Ystd = data.output_scale

        # Perform weighted SVD
        U_n, S_n, V_n, W1, O_i = SubspaceCoreAlgorithm.svd_weighted(
            y, u, f, l, weights, data=data
        )

        # Algorithm 1: extract system matrices
        Ob, X_fd, M, n, residuals = SubspaceCoreAlgorithm.algorithm_1(
            y, u, l, m, f, N, U_n, S_n, V_n, W1, O_i, threshold, max_order, D_required
        )

        # Force A stability if requested
        if A_stability:
            M, residuals[0:n, :], _ = SubspaceCoreAlgorithm.force_a_stability(
                M, n, Ob, l, X_fd, N, u, f
            )

        # Extract state-space matrices
        A, B, C, D = SubspaceCoreAlgorithm.extract_matrices(M, n)

        # Calculate covariances using optimized symmetric computation
        if NUMBA_AVAILABLE and covariance_symmetric_compiled is not None:
            try:
                Covariances = covariance_symmetric_compiled(residuals, ddof=1)
            except Exception:
                # Fallback to original
                Covariances = np.dot(residuals, residuals.T) / (N - 1)
        else:
            Covariances = np.dot(residuals, residuals.T) / (N - 1)
        Q = Covariances[0:n, 0:n]
        R = Covariances[n::, n::]
        S = Covariances[0:n, n::]

        # Simulate to evaluate model
        X_states, Y_estimate = simulate_ss_system(A, B, C, D, u)
        Vn = Vn_mat(y, Y_estimate)

        # Calculate Kalman gain
        K, K_calculated = K_calc(A, C, Q, R, S)

        # Rescale matrices back to original units
        for j in range(m):
            B[:, j] = B[:, j] / Ustd[j]
            D[:, j] = D[:, j] / Ustd[j]

        for j in range(l):
            C[j, :] = C[j, :] * Ystd[j]
            D[j, :] = D[j, :] * Ystd[j]
            if K_calculated:
                K[:, j] = K[:, j] / Ystd[j]

        output_scale = np.diag(Ystd)
        R = output_scale @ R @ output_scale
        S = S @ output_scale

        return A, B, C, D, Vn, Q, R, S, K

    @staticmethod
    def olsims_ort(
        y,
        u,
        reference,
        f,
        weights="N4SID",
        threshold=0.1,
        max_order=np.nan,
        fixed_order=np.nan,
        D_required=False,
    ):
        if not check_types(threshold, max_order, fixed_order, f):
            raise ValueError("Invalid parameters for subspace identification")

        threshold, max_order = check_inputs(threshold, max_order, fixed_order, f)
        data = prepare_subspace_data(
            y,
            u,
            future_horizon=f,
            past_offset=f,
            reference=reference,
        )
        y = data.outputs
        u = data.inputs
        output_count, _ = y.shape
        input_count = u.shape[0]
        usable_columns = data.usable_columns

        decomposition, ort = SubspaceCoreAlgorithm.svd_weighted_ort(data, weights)
        if decomposition is None:
            return None, ort.diagnostics
        U_n, S_n, V_n, W1, projection = decomposition

        deterministic = ort.reference_projection.materialize()
        future_input_rows = data.future_inputs.shape[0]
        past_rows = data.past_data.shape[0]
        deterministic_inputs = deterministic[:input_count]
        output_start = future_input_rows + past_rows
        deterministic_outputs = deterministic[
            output_start : output_start + output_count
        ]

        Ob, X_fd, M, order, residuals = SubspaceCoreAlgorithm.algorithm_1(
            y,
            u,
            output_count,
            input_count,
            f,
            usable_columns,
            U_n,
            S_n,
            V_n,
            W1,
            projection,
            threshold,
            max_order,
            D_required,
            regression_y=deterministic_outputs,
            regression_u=deterministic_inputs,
        )
        A, B, C, D = SubspaceCoreAlgorithm.extract_matrices(M, order)
        covariance = residuals @ residuals.T / (usable_columns - 1)
        Q = covariance[:order, :order]
        R = covariance[order:, order:]
        S = covariance[:order, order:]
        _, output_estimate = simulate_ss_system(A, B, C, D, u)
        Vn = Vn_mat(y, output_estimate)
        K, kalman_gain_calculated = K_calc(A, C, Q, R, S)

        for channel in range(input_count):
            B[:, channel] /= data.input_scale[channel]
            D[:, channel] /= data.input_scale[channel]
        for channel in range(output_count):
            C[channel, :] *= data.output_scale[channel]
            D[channel, :] *= data.output_scale[channel]
            if kalman_gain_calculated:
                K[:, channel] /= data.output_scale[channel]

        output_scale = np.diag(data.output_scale)
        R = output_scale @ R @ output_scale
        S = S @ output_scale
        return (A, B, C, D, Vn, Q, R, S, K), ort.diagnostics

    @staticmethod
    def select_order(
        y,
        u,
        f=20,
        weights="N4SID",
        method="AIC",
        orders=[1, 10],
        ss_threshold=0.1,
        D_required=False,
        A_stability=False,
        n_jobs=-1,
    ):
        """
        Select optimal model order using information criteria.

        Parameters:
        -----------
        y : ndarray
            Output data
        u : ndarray
            Input data
        f : int
            Future horizon
        weights : str
            Weighting method
        method : str
            Information criterion ('AIC', 'AICc', 'BIC')
        orders : list
            Order range [min, max]
        ss_threshold : float
            Singular value threshold
        D_required : bool
            Whether D matrix is required
        A_stability : bool
            Whether to force A stability
        n_jobs : int, optional
            Number of shared-memory prediction-error scoring workers. ``-1``
            selects the compiled sequential path when available and otherwise
            uses the available CPUs. A positive value requests that many workers.

        Returns:
        --------
        A, B, C, D : ndarray
            State-space matrices at optimal order
        Vn : float
            Noise variance
        Q, R, S : ndarray
            Covariance matrices
        K : ndarray
            Kalman gain
        """
        y = 1.0 * np.atleast_2d(y)
        u = 1.0 * np.atleast_2d(u)
        min_ord = min(orders)
        l, L = y.shape
        m, L = u.shape

        if not check_types(0.0, np.nan, np.nan, f):
            raise ValueError("Invalid parameters")

        if min_ord < 1:
            warnings.warn("The minimum model order will be set to 1")
            min_ord = 1

        max_ord = max(orders) + 1
        if f < min_ord:
            warnings.warn(
                f"The horizon must be larger than the model order, min_order set to f={f}"
            )
            min_ord = f
        if f < max_ord - 1:
            warnings.warn(
                f"The horizon must be larger than the model order, max_order set to f={f}"
            )
            max_ord = f + 1

        IC_old = np.inf
        data = prepare_subspace_data(
            y,
            u,
            future_horizon=f,
            past_offset=f,
        )
        y = data.outputs
        u = data.inputs
        N = data.usable_columns
        Ustd = data.input_scale
        Ystd = data.output_scale

        # Perform SVD
        U_n, S_n, V_n, W1, O_i = SubspaceCoreAlgorithm.svd_weighted(
            y, u, f, l, weights, data=data
        )

        if n_jobs != 1 and not (n_jobs == -1 or n_jobs > 0):
            raise ValueError(f"n_jobs must be -1 or positive integer, got {n_jobs}")

        order_range = list(range(min_ord, max_ord))

        def build_candidate(i):
            Ob, X_fd, M, n, residuals = SubspaceCoreAlgorithm.algorithm_1(
                y,
                u,
                l,
                m,
                f,
                N,
                U_n,
                S_n,
                V_n,
                W1,
                O_i,
                ss_threshold,
                i,
                D_required,
            )

            forced_a = False
            if A_stability:
                M, state_residuals, forced_a = SubspaceCoreAlgorithm.force_a_stability(
                    M,
                    n,
                    Ob,
                    l,
                    X_fd,
                    N,
                    u,
                    f,
                )
                residuals[:n, :] = state_residuals

            A, B, C, D = SubspaceCoreAlgorithm.extract_matrices(M, n)
            covariance = residuals @ residuals.T / residuals.shape[1]
            Q = covariance[:n, :n]
            R = covariance[n:, n:]
            S = covariance[:n, n:]
            K, _ = K_calc(A, C, Q, R, S)
            return i, n, A, B, C, D, K, forced_a

        candidates = [build_candidate(i) for i in order_range]

        def evaluate_candidate(candidate):
            i, n, A, B, C, D, K, forced_a = candidate
            prediction_errors = _causal_prediction_errors(
                A,
                B,
                C,
                D,
                K,
                y,
                u,
                np.zeros(n),
                use_compiled=worker_count == 1,
            )
            errors = prediction_errors[:, 2 * f :]

            K_par = n * (m + 2 * l) + l * (l + 1) // 2
            if D_required:
                K_par = K_par + l * m
            criterion = _innovation_information_criterion(errors, K_par, method)
            return i, criterion, forced_a

        worker_count = 1
        if len(order_range) > 1 and n_jobs != 1:
            if n_jobs == -1:
                requested_workers = 1 if NUMBA_AVAILABLE else (os.cpu_count() or 1)
            else:
                requested_workers = n_jobs
            worker_count = min(len(order_range), requested_workers)

        if worker_count == 1:
            results = map(evaluate_candidate, candidates)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                results = list(executor.map(evaluate_candidate, candidates))

        n_min = min_ord
        for i, IC, forced_a in results:
            if forced_a:
                warnings.warn(f"A stability forced at n={i}")
            if IC < IC_old:
                n_min = i
                IC_old = IC

        warnings.warn(f"The suggested order is: n={n_min}")

        # Final identification with selected order
        Ob, X_fd, M, n, residuals = SubspaceCoreAlgorithm.algorithm_1(
            y, u, l, m, f, N, U_n, S_n, V_n, W1, O_i, ss_threshold, n_min, D_required
        )

        if A_stability:
            _, _, _ = SubspaceCoreAlgorithm.force_a_stability(
                M, n, Ob, l, X_fd, N, u, f
            )

        A, B, C, D = SubspaceCoreAlgorithm.extract_matrices(M, n)
        # Use optimized symmetric covariance computation
        if NUMBA_AVAILABLE and covariance_symmetric_compiled is not None:
            try:
                Covariances = covariance_symmetric_compiled(residuals, ddof=1)
            except Exception:
                Covariances = np.dot(residuals, residuals.T) / (N - 1)
        else:
            Covariances = np.dot(residuals, residuals.T) / (N - 1)
        X_states, Y_estimate = simulate_ss_system(A, B, C, D, u)
        Vn = Vn_mat(y, Y_estimate)

        Q = Covariances[0:n, 0:n]
        R = Covariances[n::, n::]
        S = Covariances[0:n, n::]

        K, K_calculated = K_calc(A, C, Q, R, S)

        # Rescale back to original units
        for j in range(m):
            B[:, j] = B[:, j] / Ustd[j]
            D[:, j] = D[:, j] / Ustd[j]

        for j in range(l):
            C[j, :] = C[j, :] * Ystd[j]
            D[j, :] = D[j, :] * Ystd[j]
            if K_calculated:
                K[:, j] = K[:, j] / Ystd[j]

        output_scale = np.diag(Ystd)
        R = output_scale @ R @ output_scale
        S = S @ output_scale

        return A, B, C, D, Vn, Q, R, S, K
