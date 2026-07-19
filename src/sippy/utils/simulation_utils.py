"""
System simulation and analysis utilities.
"""

import warnings

import numpy as np
from scipy import signal
from scipy.linalg import solve_discrete_are

from sippy import systems as control

# Import compiled utilities for performance
try:
    from .compiled_utils import (
        NUMBA_AVAILABLE,
        Vn_mat_compiled,
        covariance_symmetric_compiled,
        extract_matrices_batch_compiled,
        # Enhanced Phase 1-3 functions
        impile_advanced_compiled,
        impile_compiled,
        kalc_riccati_compiled,
        ordinate_sequence_compiled,
        pinv_compiled_svd,
        reducingOrder_compiled,
        reducingOrder_fast_compiled,
        simulate_ss_system_compiled,
        ss_lsim_predictor_form_compiled,
        vn_mat_parallel_compiled,
    )
except ImportError:
    # Fallback if compiled_utils is not available
    ordinate_sequence_compiled = None
    simulate_ss_system_compiled = None
    impile_compiled = None
    reducingOrder_compiled = None
    Vn_mat_compiled = None
    impile_advanced_compiled = None
    reducingOrder_fast_compiled = None
    kalc_riccati_compiled = None
    vn_mat_parallel_compiled = None
    covariance_symmetric_compiled = None
    extract_matrices_batch_compiled = None
    pinv_compiled_svd = None
    NUMBA_AVAILABLE = False


def ordinate_sequence(y, f, p):
    """
    Create ordinate sequences for subspace identification.

    This function automatically uses the Numba-compiled version when available
    for improved performance.

    Parameters:
    -----------
    y : ndarray
        Output data with shape (outputs, time_steps)
    f : int
        Future horizon
    p : int
        Past horizon

    Returns:
    --------
    Yf : ndarray
        Future output ordinate sequence
    Yp : ndarray
        Past output ordinate sequence
    """
    if NUMBA_AVAILABLE and ordinate_sequence_compiled is not None:
        return ordinate_sequence_compiled(y, f, p)
    else:
        # Fallback to original implementation
        l, L = y.shape
        N = L - p - f + 1
        Yp = np.zeros((l * f, N))
        Yf = np.zeros((l * f, N))

        for i in range(1, f + 1):
            Yf[l * (i - 1) : l * i] = y[:, p + i - 1 : L - f + i]
            Yp[l * (i - 1) : l * i] = y[:, i - 1 : L - f - p + i]

        return Yf, Yp


def Z_dot_PIort(z, X):
    """
    Compute the scalar product between vector z and I - X^T * pinv(X^T),
    avoiding direct computation of the full matrix.

    Parameters:
    -----------
    z : ndarray
        Vector
    X : ndarray
        Matrix

    Returns:
    --------
    result : ndarray
        Computed result
    """
    return z - np.dot(np.dot(z, X.T), np.linalg.pinv(X.T))


def Vn_mat(y, yest):
    """
    Compute the variance of the model residuals.

    This function automatically uses the enhanced Numba-compiled version when available
    for improved performance with parallel processing.

    Parameters:
    -----------
    y : ndarray
        Process output
    yest : ndarray
        Estimated model output

    Returns:
    --------
    Vn : float
        Residual variance
    """
    if NUMBA_AVAILABLE and vn_mat_parallel_compiled is not None:
        return vn_mat_parallel_compiled(y.flatten(), yest.flatten())
    elif NUMBA_AVAILABLE and Vn_mat_compiled is not None:
        return Vn_mat_compiled(y.flatten(), yest.flatten())
    else:
        # Fallback to original implementation
        y = y.flatten()
        yest = yest.flatten()
        eps = y - yest
        Vn = (eps @ eps) / (max(y.shape))
        return Vn


def impile(M1, M2):
    """
    Stack two matrices vertically with edge case handling.

    This function automatically uses the enhanced Numba-compiled version when available
    for improved performance with parallel processing and memory efficiency.

    Parameters:
    -----------
    M1, M2 : ndarray
        Matrices to stack

    Returns:
    --------
    M : ndarray
        Vertically stacked matrix
    """
    # Handle empty matrix edge cases
    # Case 1: M1 has no columns (empty in column dimension)
    if M1.shape[1] == 0 and M2.shape[1] > 0:
        return M2.copy()
    # Case 2: M1 has no rows (empty in row dimension)
    elif M1.shape[0] == 0:
        return M2.copy()
    # Case 3: M2 has no rows (empty in row dimension)
    elif M2.shape[0] == 0:
        return M1.copy()
    # Case 4: M2 has no columns (empty in column dimension)
    elif M2.shape[1] == 0 and M1.shape[1] > 0:
        return M1.copy()
    # Case 5: Column mismatch check
    elif M1.shape[1] != M2.shape[1]:
        raise ValueError(
            f"Matrix column mismatch: M1 has {M1.shape[1]} columns, "
            f"M2 has {M2.shape[1]} columns"
        )

    # Use compiled version for non-empty matrices
    if NUMBA_AVAILABLE and impile_advanced_compiled is not None:
        return impile_advanced_compiled(M1, M2)
    elif NUMBA_AVAILABLE and impile_compiled is not None:
        return impile_compiled(M1, M2)
    else:
        # Fallback to original implementation
        M = np.zeros((M1[:, 0].size + M2[:, 0].size, M1[0, :].size))
        M[0 : M1[:, 0].size] = M1
        M[M1[:, 0].size : :] = M2
        return M


def reducingOrder(U_n, S_n, V_n, threshold=0.1, max_order=10):
    """
    Reduce model order based on singular values.

    This function automatically uses the Numba-compiled version when available
    for improved performance.

    Parameters:
    -----------
    U_n, S_n, V_n : ndarray
        SVD components
    threshold : float
        Threshold for truncation
    max_order : int
        Maximum order to keep

    Returns:
    --------
    U_n, S_n, V_n : ndarray
        Truncated SVD components
    """
    if NUMBA_AVAILABLE and reducingOrder_fast_compiled is not None:
        return reducingOrder_fast_compiled(U_n, S_n, V_n, threshold, max_order)
    elif NUMBA_AVAILABLE and reducingOrder_compiled is not None:
        return reducingOrder_compiled(U_n, S_n, V_n, threshold, max_order)
    else:
        # Fallback to original implementation
        s0 = S_n[0]
        index = S_n.size
        for i in range(S_n.size):
            if S_n[i] < threshold * s0 or i >= max_order:
                index = i
                break
        return U_n[:, 0:index], S_n[0:index], V_n[0:index, :]


def check_types(threshold, max_order, fixed_order, f, p=20):
    """
    Validate parameter types and values.

    Parameters:
    -----------
    threshold : float
        Threshold value
    max_order : float or int
        Maximum model order
    fixed_order : float or int
        Fixed model order
    f : int
        Future horizon
    p : int
        Past horizon

    Returns:
    --------
    valid : bool
        Whether parameters are valid
    """
    if threshold < 0.0 or threshold >= 1.0:
        print("Error! The threshold value must be >=0. and <1.")
        return False
    if not np.isnan(max_order):
        if not isinstance(max_order, int):
            print("Error! The max_order value must be integer")
            return False
    if not np.isnan(fixed_order):
        if not isinstance(fixed_order, int):
            print("Error! The fixed_order value must be integer")
            return False
    if not isinstance(f, int):
        print("Error! The future horizon (f) must be integer")
        return False
    if not isinstance(p, int):
        print("Error! The past horizon (p) must be integer")
        return False
    return True


def check_inputs(threshold, max_order, fixed_order, f):
    """
    Adjust and validate input parameters.

    Parameters:
    -----------
    threshold : float
        Threshold value
    max_order : float or None
        Maximum model order
    fixed_order : float or None
        Fixed model order
    f : int
        Future horizon

    Returns:
    --------
    threshold : float
        Adjusted threshold
    max_order : int
        Adjusted maximum order
    """
    if not np.isnan(fixed_order):
        threshold = 0.0
        max_order = fixed_order
    if f < max_order:
        print(
            "Warning! The horizon must be larger than the model order, max_order setted as f"
        )
    if max_order >= f:
        max_order = f
    return threshold, max_order


def simulate_ss_system(A, B, C, D, u, x0=None):
    """
    Simulate state-space system in process form.

    This function automatically uses the Numba-compiled version when available
    for improved performance.

    Parameters:
    -----------
    A, B, C, D : ndarray
        State-space matrices
    u : ndarray
        Input signals (inputs x time_steps)
    x0 : ndarray, optional
        Initial state

    Returns:
    --------
    x : ndarray
        State trajectory
    y : ndarray
        Output signals
    """
    if NUMBA_AVAILABLE and simulate_ss_system_compiled is not None:
        return simulate_ss_system_compiled(A, B, C, D, u, x0)
    else:
        # Fallback to original implementation
        m, L = u.shape
        l, n = C.shape
        y = np.zeros((l, L))
        x = np.zeros((n, L))

        if x0 is not None:
            x[:, 0] = x0[:, 0]

        y[:, 0] = np.dot(C, x[:, 0]) + np.dot(D, u[:, 0])

        for i in range(1, L):
            x[:, i] = np.dot(A, x[:, i - 1]) + np.dot(B, u[:, i - 1])
            y[:, i] = np.dot(C, x[:, i]) + np.dot(D, u[:, i])

        return x, y


def ss_lsim_predictor_form(A_K, B_K, C, D, K, y, u, x0=None):
    """
    Simulate state-space system in predictor form.

    In predictor form, the state equation is:
        x[i+1] = A_K*x[i] + B_K*u[i] + K*y[i]
        y_hat[i] = C*x[i] + D*u[i]

    This is the form used in PARSIM-K algorithm where the Kalman gain K
    provides output feedback to the state update.

    Reference: master/sippy_unipi/functionsetSIM.py lines 122-134

    Parameters:
    -----------
    A_K : ndarray
        State transition matrix in predictor form (n x n)
    B_K : ndarray
        Input matrix in predictor form (n x m)
    C : ndarray
        Output matrix (l x n)
    D : ndarray
        Feedthrough matrix (l x m)
    K : ndarray
        Kalman gain matrix (n x l)
    y : ndarray
        Measured output data (l x L)
    u : ndarray
        Input signals (m x L)
    x0 : ndarray, optional
        Initial state (n x 1)

    Returns:
    --------
    x : ndarray
        State trajectory (n x L+1)
    y_hat : ndarray
        Predicted output signals (l x L)
    """
    if NUMBA_AVAILABLE and ss_lsim_predictor_form_compiled is not None:
        return ss_lsim_predictor_form_compiled(A_K, B_K, C, D, K, y, u, x0)
    else:
        m, L = u.shape
        l, n = C.shape
        y_hat = np.zeros((l, L))
        x = np.zeros((n, L + 1))

        if x0 is not None:
            x[:, 0] = x0[:, 0]

        for i in range(0, L):
            x[:, i + 1] = (
                np.dot(A_K, x[:, i]) + np.dot(B_K, u[:, i]) + np.dot(K, y[:, i])
            )
            y_hat[:, i] = np.dot(C, x[:, i]) + np.dot(D, u[:, i])

        return x, y_hat


def ssmatrix(data, axis=1):
    """
    Convert argument to a (possibly empty) state space matrix.

    Parameters:
    -----------
    data : array, list, or string
        Input data defining the contents of the 2D array
    axis : 0 or 1
        If input data is 1D, which axis to use for return object

    Returns:
    -------
    arr : ndarray
        2D array, with shape (0, 0) if empty
    """
    arr = np.array(data, dtype=float)
    ndim = arr.ndim
    shape = arr.shape

    # Change the shape of the array into a 2D array
    if ndim > 2:
        raise ValueError("state-space matrix must be 2-dimensional")
    elif (ndim == 2 and shape == (1, 0)) or (ndim == 1 and shape == (0,)):
        # Passed an empty matrix or empty vector; change shape to (0, 0)
        shape = (0, 0)
    elif ndim == 1:
        # Passed a row or column vector
        shape = (1, shape[0]) if axis == 1 else (shape[0], 1)
    elif ndim == 0:
        # Passed a constant; turn into a matrix
        shape = (1, 1)

    return arr.reshape(shape)


def K_calc(A, C, Q, R, S):
    """
    Calculate Kalman filter gain.

    This function automatically uses the enhanced Numba-compiled version when available
    for improved performance with custom Riccati solver.

    Parameters:
    -----------
    A, C, Q, R, S : ndarray
        State-space and noise covariance matrices

    Returns:
    --------
    K : ndarray
        Kalman gain
    Calculated : bool
        Whether calculation was successful
    """
    if NUMBA_AVAILABLE and kalc_riccati_compiled is not None:
        K, Calculated, P = kalc_riccati_compiled(A, C, Q, R, S)
        return K, Calculated
    else:
        # Fallback to original scipy-based implementation
        try:
            X = solve_discrete_are(A.T, C.T, Q, R)
            P = ssmatrix(X)
            K = np.dot(np.dot(A, P), C.T) + S
            K = np.dot(K, np.linalg.inv(np.dot(np.dot(C, P), C.T) + R))
            Calculated = True
        except (ValueError, np.linalg.LinAlgError, IndexError):
            K = []
            warnings.warn("Kalman filter cannot be calculated")
            Calculated = False
        return K, Calculated


def get_frequency_response_uncertainty(
    u,
    y,
    model,
    *,
    dt=1.0,
    nperseg=None,
    window="hann",
    noverlap=0,
    smoothing_bins=5,
    confidence_levels=(0.68, 0.95),
):
    """Estimate empirical frequency-response uncertainty for a model or FIR."""
    if hasattr(model, "get_frequency_response_uncertainty"):
        return model.get_frequency_response_uncertainty(
            u,
            y,
            nperseg=nperseg,
            window=window,
            noverlap=noverlap,
            smoothing_bins=smoothing_bins,
            confidence_levels=confidence_levels,
        )
    if hasattr(model, "get_model_uncertainty"):
        return model.get_model_uncertainty(
            u,
            y,
            nperseg=nperseg,
            window=window,
            noverlap=noverlap,
            smoothing_bins=smoothing_bins,
            confidence_levels=confidence_levels,
        )

    from ..identification.uncertainty import estimate_frequency_response_uncertainty

    result = estimate_frequency_response_uncertainty(
        u,
        y,
        dt=dt,
        nperseg=nperseg,
        window=window,
        noverlap=noverlap,
        smoothing_bins=smoothing_bins,
        confidence_levels=confidence_levels,
    )
    coefficients = np.asarray(model, dtype=float)
    if coefficients.ndim == 1:
        coefficients = coefficients[None, None, :]
    if coefficients.ndim != 3:
        raise ValueError(
            "FIR coefficients must have shape (lags,) or (outputs, inputs, lags)"
        )
    if coefficients.shape[0:2] != result.empirical_response.shape[1:3]:
        raise ValueError("FIR channel dimensions must match the input/output data")

    normalized_omega = result.omega * dt
    delays = np.arange(coefficients.shape[2])
    basis = np.exp(-1j * normalized_omega[:, None] * delays[None, :])
    response = np.einsum("oil,fl->foi", coefficients, basis)
    return result.with_model_response(response)


def get_model_uncertainty(
    u,
    y,
    model,
    *,
    dt=1.0,
    nperseg=None,
    window="hann",
    noverlap=0,
    smoothing_bins=5,
    confidence_levels=(0.68, 0.95),
):
    """Compatibility alias for :func:`get_frequency_response_uncertainty`."""
    return get_frequency_response_uncertainty(
        u,
        y,
        model,
        dt=dt,
        nperseg=nperseg,
        window=window,
        noverlap=noverlap,
        smoothing_bins=smoothing_bins,
        confidence_levels=confidence_levels,
    )


def get_fir_coef(model, inds, deps, sampling, tss):
    """
    Returns a nested dictionary of numpy arrays containing FIR coefficients.

    Parameters:
    -----------
    model : control.InputOutputSystem or StateSpaceModel
        Control-system model or SIPPY model wrapper.
    inds : list
        List of independent variables
    deps : list
        List of dependent variables
    sampling : int
        Model sampling rate in seconds
    tss : int
        Time to steady state in minutes

    Returns:
    --------
    fir_model : dict
        Nested dictionary of FIR coefficients
    """
    system = getattr(model, "G", model)
    if system is None or not isinstance(system, control.InputOutputSystem):
        raise TypeError("model must be a SIPPY system or a model with G")
    if len(inds) != system.ninputs or len(deps) != system.noutputs:
        raise ValueError("input and output names must match the model dimensions")
    if sampling <= 0 or tss <= 0:
        raise ValueError("sampling and tss must be positive")

    t = np.arange(0, tss * 60, sampling)
    response = control.impulse_response(system, T=t, squeeze=False)
    sample_time = 1.0 if system.dt is True else float(system.dt)
    impulse = np.asarray(response.outputs, dtype=float) * sample_time

    fir_model = {}
    for depidx, dep in enumerate(deps):
        fir_model[dep] = {}
        for indidx, ind in enumerate(inds):
            fir_model[dep][ind] = impulse[depidx, indidx, :]

    return fir_model


def get_step_response(fir_model):
    """
    Returns a nested dictionary of numpy arrays containing step response.

    Parameters:
    -----------
    fir_model : dict
        Nested dictionary of FIR coefficients

    Returns:
    --------
    step_response : dict
        Nested dictionary of step responses
    """
    step_response = dict()
    for dep in fir_model.keys():
        step_response[dep] = dict()
        for ind in fir_model[dep].keys():
            step_response[dep][ind] = np.cumsum(fir_model[dep][ind])
    return step_response


def get_deadtime(step_response, isramp=False):
    """
    Returns the estimated deadtime based on a predefined minimum response tolerance.
    Current tolerance is the 4% of steady state gain or overshoot.

    Parameters:
    -----------
    step_response : ndarray
        Step response of the model
    isramp : bool
        Ramp type flag

    Returns:
    --------
    deadtime : int
        Deadtime in terms of number of samples
    """
    if isramp:
        gain = step_response[-1] - step_response[-2]
        abs_gain = abs(gain)
        tol = abs_gain / 25
    else:
        gain = step_response[-1]
        abs_gain = abs(gain)
        overshoot = np.abs(step_response).max()
        tol = abs_gain / 25 if overshoot <= abs_gain else overshoot / 25

    deadtime = 0
    for coef in step_response:
        if abs(coef) <= tol:
            deadtime += 1
        else:
            break

    deadtime = deadtime if deadtime >= 2 else 0
    return deadtime


def simulate_fir(fir_model, data):
    """
    Returns a pandas dataframe of dependent variable predictions.

    Parameters:
    -----------
    fir_model : dict
        Nested dictionary of FIR coefficients
    data : pandas.DataFrame
        DataFrame containing independent and dependent variables data

    Returns:
    --------
    predictions : pandas.DataFrame
        DataFrame containing dependent variable predictions
    """
    N = len(data)
    deps = list(fir_model.keys())
    inds = list(fir_model[deps[0]].keys())
    predictions = data[deps].copy(deep=True)

    for dep in deps:
        predictions[dep].values[:] = 0.0
        for ind in inds:
            predictions[dep] += signal.convolve(
                data[ind], fir_model[dep][ind], mode="full"
            )[:N]
        tss = len(fir_model[dep][inds[0]])
        predictions[dep][:tss] = predictions[dep].values[tss + 1]
        predictions[dep] = predictions[dep] - predictions[dep].mean() + data[dep].mean()

    return predictions


# Uppercase alias for backward compatibility with master branch naming
SS_lsim_predictor_form = ss_lsim_predictor_form
