"""
Core subspace identification algorithms implementation.
"""

import warnings

import numpy as np
import scipy as sc
from numpy.linalg import pinv

from ...utils.signal_utils import information_criterion, rescale
from ...utils.simulation_utils import (
    K_calc,
    Vn_mat,
    Z_dot_PIort,
    check_inputs,
    check_types,
    impile,
    ordinate_sequence,
    reducingOrder,
    simulate_ss_system,
)

# Import compiled utilities for performance
try:
    from ...utils.compiled_utils import (
        NUMBA_AVAILABLE,
        Z_dot_PIort_compiled,
        covariance_symmetric_compiled,
        information_criterion_compiled,
        pinv_compiled_svd,
        rescale_compiled,
        subspace_weighted_svd_compiled,
    )
except ImportError:
    NUMBA_AVAILABLE = False
    Z_dot_PIort_compiled = None
    covariance_symmetric_compiled = None
    information_criterion_compiled = None
    rescale_compiled = None
    subspace_weighted_svd_compiled = None


class SubspaceCoreAlgorithm:
    """Core subspace identification algorithms implementation."""

    @staticmethod
    def svd_weighted(y, u, f, l, weights="N4SID"):
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
        O_i : ndarray
            Extended observability matrix
        """
        Yf, Yp = ordinate_sequence(y, f, f)
        Uf, Up = ordinate_sequence(u, f, f)
        Zp = impile(Up, Yp)

        # Use compiled Z_dot_PIort when available
        if NUMBA_AVAILABLE and Z_dot_PIort_compiled is not None:
            try:
                YfdotPIort_Uf = Z_dot_PIort_compiled(Yf, Uf)
                ZpdotPIort_Uf = Z_dot_PIort_compiled(Zp, Uf)
            except Exception:
                # Fallback to original
                YfdotPIort_Uf = Z_dot_PIort(Yf, Uf)
                ZpdotPIort_Uf = Z_dot_PIort(Zp, Uf)
        else:
            YfdotPIort_Uf = Z_dot_PIort(Yf, Uf)
            ZpdotPIort_Uf = Z_dot_PIort(Zp, Uf)

        # Use compiled pseudoinverse when available for performance
        try:
            if NUMBA_AVAILABLE and pinv_compiled_svd is not None:
                Zpinv = pinv_compiled_svd(ZpdotPIort_Uf)
            else:
                Zpinv = pinv(ZpdotPIort_Uf)
        except Exception:
            Zpinv = pinv(ZpdotPIort_Uf)
        O_i = np.dot(np.dot(YfdotPIort_Uf, Zpinv), Zp)

        if weights == "MOESP":
            W1 = None
            if NUMBA_AVAILABLE and Z_dot_PIort_compiled is not None:
                try:
                    OidotPIort_Uf = Z_dot_PIort_compiled(O_i, Uf)
                except Exception:
                    OidotPIort_Uf = Z_dot_PIort(O_i, Uf)
            else:
                OidotPIort_Uf = Z_dot_PIort(O_i, Uf)
            U_n, S_n, V_n = np.linalg.svd(OidotPIort_Uf, full_matrices=False)

        elif weights == "CVA":
            YfdotPIort_Uf_YfdotPIort_Uf_T = np.dot(YfdotPIort_Uf, YfdotPIort_Uf.T)
            if YfdotPIort_Uf_YfdotPIort_Uf_T.shape[0] == 0:
                warnings.warn("CVA weighting failed, falling back to N4SID")
                W1 = None
                U_n, S_n, V_n = np.linalg.svd(O_i, full_matrices=False)
            else:
                sqrt_term = sc.linalg.sqrtm(YfdotPIort_Uf_YfdotPIort_Uf_T)
                sqrt_term_real = sqrt_term.real
                W1 = sqrt_term_real
                W1dotOi = np.linalg.solve(sqrt_term_real, O_i)
                if NUMBA_AVAILABLE and Z_dot_PIort_compiled is not None:
                    try:
                        W1_dot_Oi_dot_PIort_Uf = Z_dot_PIort_compiled(W1dotOi, Uf)
                    except Exception:
                        W1_dot_Oi_dot_PIort_Uf = Z_dot_PIort(W1dotOi, Uf)
                else:
                    W1_dot_Oi_dot_PIort_Uf = Z_dot_PIort(W1dotOi, Uf)
                U_n, S_n, V_n = np.linalg.svd(
                    W1_dot_Oi_dot_PIort_Uf, full_matrices=False
                )

        elif weights == "N4SID":
            W1 = None  # is identity
            U_n, S_n, V_n = np.linalg.svd(O_i, full_matrices=False)
        else:
            raise ValueError(f"Unknown weighting method: {weights}")

        return U_n, S_n, V_n, W1, O_i

    @staticmethod
    def algorithm_1(
        y, u, l, m, f, N, U_n, S_n, V_n, W1, O_i, threshold, max_order, D_required
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
        O_i : ndarray
            Extended observability matrix
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
        X_fd = np.dot(Ob_pinv, O_i)
        # Ensure contiguous memory for optimal performance with compiled functions
        X_fd_slice1 = np.ascontiguousarray(X_fd[:, 1:N])
        y_slice = np.ascontiguousarray(y[:, f : f + N - 1])
        Sxterm = impile(X_fd_slice1, y_slice)

        X_fd_slice2 = np.ascontiguousarray(X_fd[:, 0 : N - 1])
        u_slice = np.ascontiguousarray(u[:, f : f + N - 1])
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
        y = 1.0 * np.atleast_2d(y)
        u = 1.0 * np.atleast_2d(u)
        l, L = y.shape
        m = u[:, 0].size

        if not check_types(threshold, max_order, fixed_order, f):
            raise ValueError("Invalid parameters for subspace identification")

        threshold, max_order = check_inputs(threshold, max_order, fixed_order, f)
        N = L - 2 * f + 1

        if N <= 0:
            raise ValueError(
                f"Not enough data points. Need at least {2 * f + 1} points, got {L}"
            )

        # Standardize inputs and outputs
        Ustd = np.zeros(m)
        Ystd = np.zeros(l)
        for j in range(m):
            if NUMBA_AVAILABLE and rescale_compiled is not None:
                Ustd[j], u[j] = rescale_compiled(u[j])
            else:
                Ustd[j], u[j] = rescale(u[j])
        for j in range(l):
            if NUMBA_AVAILABLE and rescale_compiled is not None:
                Ystd[j], y[j] = rescale_compiled(y[j])
            else:
                Ystd[j], y[j] = rescale(y[j])

        # Perform weighted SVD
        U_n, S_n, V_n, W1, O_i = SubspaceCoreAlgorithm.svd_weighted(y, u, f, l, weights)

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
            Kept for backward compatibility; order evaluation shares one SVD
            and is always sequential. Must be -1 or a positive integer.

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
        N = L - 2 * f + 1

        # Standardize data
        Ustd = np.zeros(m)
        Ystd = np.zeros(l)
        for j in range(m):
            if NUMBA_AVAILABLE and rescale_compiled is not None:
                Ustd[j], u[j] = rescale_compiled(u[j])
            else:
                Ustd[j], u[j] = rescale(u[j])
        for j in range(l):
            if NUMBA_AVAILABLE and rescale_compiled is not None:
                Ystd[j], y[j] = rescale_compiled(y[j])
            else:
                Ystd[j], y[j] = rescale(y[j])

        # Perform SVD
        U_n, S_n, V_n, W1, O_i = SubspaceCoreAlgorithm.svd_weighted(y, u, f, l, weights)

        if n_jobs != 1 and not (n_jobs == -1 or n_jobs > 0):
            raise ValueError(f"n_jobs must be -1 or positive integer, got {n_jobs}")

        order_range = list(range(min_ord, max_ord))

        # The SVD above is shared across all candidate orders, so each order
        # evaluation is just a truncation plus small least-squares solves —
        # too cheap to justify spawning worker processes (n_jobs is kept for
        # backward compatibility but evaluation is always sequential).
        n_min = min_ord
        for i in order_range:
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

            if A_stability:
                _, _, ForcedA = SubspaceCoreAlgorithm.force_a_stability(
                    M, n, Ob, l, X_fd, N, u, f
                )
                if ForcedA:
                    warnings.warn(f"A stability forced at n={n}")

            # One-step output-equation residual variance for the
            # likelihood-based information criteria (prediction error,
            # not free-run simulation error).
            output_residuals = residuals[n:, :]
            Vn = max(float(np.mean(output_residuals**2)), np.finfo(np.float64).tiny)

            # Calculate number of parameters
            K_par = n * l + m * n
            if D_required:
                K_par = K_par + l * m

            # Use compiled information criterion if available
            if NUMBA_AVAILABLE and information_criterion_compiled is not None:
                method_map = {"AIC": 0, "AICc": 1, "BIC": 2}
                method_code = method_map.get(method, 0)
                IC = information_criterion_compiled(K_par, L, Vn, method_code)
            else:
                IC = information_criterion(K_par, L, Vn, method)
            if IC < IC_old:
                n_min = i
                IC_old = IC

        print(f"The suggested order is: n={n_min}")

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
