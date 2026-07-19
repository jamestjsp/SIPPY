"""
ARX (AutoRegressive with eXogenous inputs) identification algorithm.
"""

from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.linalg import lstsq

from sippy import systems as control

from ..base import (
    IdentificationAlgorithm,
    StateSpaceModel,
    identity_transfer_function,
    realize_transfer_function,
)

if TYPE_CHECKING:
    from ..iddata import IDData

# Import compiled utilities for performance
try:
    from ...utils.compiled_utils import (
        NUMBA_AVAILABLE,
        create_regression_matrix_arx_compiled,
        create_regression_matrix_arx_mimo_compiled,
    )
except ImportError:
    create_regression_matrix_arx_compiled = None
    create_regression_matrix_arx_mimo_compiled = None
    NUMBA_AVAILABLE = False


class ARXAlgorithm(IdentificationAlgorithm):
    """
    ARX (AutoRegressive with eXogenous inputs) identification algorithm.

    The ARX model structure is:
    A(q) y(k) = B(q) u(k - nk) + e(k)

    where:
    - A(q) = 1 + a1*q^-1 + ... + ana*q^-na (auto-regressive part)
    - B(q) = b1 + b2*q^-1 + ... + bnb*q^-(nb-1) (exogenous input part)
    - nk is the input delay (number of samples)
    - e(k) is white noise

    The algorithm uses least-squares regression to estimate the ARX parameters.
    """

    def __init__(self):
        """Initialize ARX algorithm."""
        super().__init__()

    def get_algorithm_name(self) -> str:
        """Return algorithm name."""
        return "ARX"

    def validate_parameters(self, **kwargs) -> bool:
        """
        Validate ARX-specific parameters.

        Parameters:
        -----------
        **kwargs : dict
            Parameters to validate including na, nb, nk

        Returns:
        --------
        bool
            True if parameters are valid
        """
        na = kwargs.get("na", 1)
        nb = kwargs.get("nb", 1)
        nk = kwargs.get("nk", 1)

        if na <= 0:
            raise ValueError("AR order (na) must be positive")
        if nb <= 0:
            raise ValueError("X order (nb) must be positive")
        if nk < 0:
            raise ValueError("Input delay (nk) must be non-negative")

        return True

    def identify(
        self,
        y: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
        iddata: Optional["IDData"] = None,
        **kwargs,
    ) -> StateSpaceModel:
        """
        Identify ARX model from input-output data.

        Parameters:
        -----------
        y : np.ndarray, optional
            Output data (outputs x time_steps)
        u : np.ndarray, optional
            Input data (inputs x time_steps)
        iddata : IDData, optional
            Input-output data container
        **kwargs : dict
            Configuration parameters including na, nb, nk, tsample

        Returns:
        --------
        model : StateSpaceModel
            Identified state-space model
        """
        # Validate input arguments
        if iddata is not None and (y is not None or u is not None):
            raise ValueError("Provide either iddata or (y, u), but not both")
        if iddata is None and (y is None or u is None):
            raise ValueError("Must provide either iddata or both y and u")

        # Extract data if IDData is provided
        if iddata is not None:
            u = iddata.get_input_array()
            y = iddata.get_output_array()
            sample_time = iddata.sample_time
        else:
            # Ensure arrays are 2D
            y = np.atleast_2d(y)
            u = np.atleast_2d(u)
            sample_time = kwargs.get("tsample", 1.0)

        # Extract configuration parameters (ARX specific)
        na = kwargs.get("na", 1)
        nb = kwargs.get("nb", 1)
        nk = kwargs.get("nk", 1)

        # Validate parameters
        self.validate_parameters(na=na, nb=nb, nk=nk)

        # Get data dimensions
        ny, N = y.shape
        nu, _ = u.shape

        # Calculate effective data length
        max_lag = max(na, nb + nk - 1)
        N_eff = N - max_lag

        if N_eff <= 0:
            raise ValueError(
                f"Not enough data points. Need at least {max_lag + 1} samples, got {N}"
            )

        # Estimate parameters using least squares
        max_lag = max(na, nb + nk - 1)
        N_eff = N - max_lag
        if N_eff <= 0:
            raise ValueError(
                f"Not enough data points. Need at least {max_lag + 1} samples, got {N}"
            )

        if ny == 1:
            # SISO case leverages shared compiled builder
            Phi, y_matrix = self._create_regression_matrix(u, y, na, nb, nk, ny, nu, N)
            theta, residuals, rank, s = lstsq(Phi, y_matrix.T.flatten(), rcond=None)
            A_coeffs = theta[:na].reshape(1, na)
            B_coeffs = theta[na:].reshape(1, nb)
        else:
            use_compiled_mimo = NUMBA_AVAILABLE and (
                create_regression_matrix_arx_mimo_compiled is not None
            )

            if use_compiled_mimo:
                Phi_batches, y_targets = create_regression_matrix_arx_mimo_compiled(
                    np.ascontiguousarray(u),
                    np.ascontiguousarray(y),
                    na,
                    nb,
                    nk,
                    ny,
                    nu,
                    N,
                )
            else:
                Phi, y_matrix = self._create_regression_matrix(
                    u, y, na, nb, nk, ny, nu, N
                )

            A_coeffs = np.zeros((ny, na, ny))
            B_coeffs = np.zeros((ny, nb * nu))
            for i in range(ny):
                if use_compiled_mimo:
                    Phi_i = np.ascontiguousarray(Phi_batches[i, :, :])
                    y_target = y_targets[i, :]
                else:
                    n_params_i = na * ny + nb * nu
                    Phi_i = np.zeros((N_eff, n_params_i))
                    col = 0

                    for lag in range(na):
                        for j in range(ny):
                            Phi_i[:, col] = y[
                                j, max_lag - 1 - lag : max_lag - 1 - lag + N_eff
                            ]
                            col += 1

                    for lag in range(nb):
                        for j in range(nu):
                            delay_idx = max_lag - 1 - (lag + nk - 1)
                            if delay_idx >= 0 and delay_idx + N_eff <= N:
                                Phi_i[:, col] = u[j, delay_idx : delay_idx + N_eff]
                            col += 1

                    y_target = y_matrix[i, :]

                theta_i, residuals_i, rank_i, s_i = lstsq(Phi_i, y_target, rcond=None)

                A_coeffs[i, :, :] = theta_i[: na * ny].reshape(na, ny)

                B_coeffs[i, :] = theta_i[na * ny :]

        # Compute one-step-ahead predictions (Yid) for identification data
        Yid = np.zeros_like(y)
        Yid[:, :max_lag] = y[:, :max_lag]  # Copy initial values
        if ny == 1:
            Yid[0, max_lag:] = np.dot(Phi, theta)
        else:
            # For MIMO, reconstruct predictions for each output
            for i in range(ny):
                if use_compiled_mimo:
                    Phi_i = np.ascontiguousarray(Phi_batches[i, :, :])
                    theta_i = np.zeros(na * ny + nb * nu)
                    theta_i[: na * ny] = A_coeffs[i].reshape(-1)
                    theta_i[na * ny :] = B_coeffs[i, :]
                    Yid[i, max_lag:] = np.dot(Phi_i, theta_i)
                else:
                    # Use the same Phi construction as before
                    n_params_i = na * ny + nb * nu
                    Phi_i = np.zeros((N_eff, n_params_i))
                    col = 0
                    for lag in range(na):
                        for j in range(ny):
                            Phi_i[:, col] = y[
                                j, max_lag - 1 - lag : max_lag - 1 - lag + N_eff
                            ]
                            col += 1
                    for lag in range(nb):
                        for j in range(nu):
                            delay_idx = max_lag - 1 - (lag + nk - 1)
                            if delay_idx >= 0 and delay_idx + N_eff <= N:
                                Phi_i[:, col] = u[j, delay_idx : delay_idx + N_eff]
                            col += 1

                    theta_i = np.zeros(n_params_i)
                    theta_i[: na * ny] = A_coeffs[i].reshape(-1)
                    theta_i[na * ny :] = B_coeffs[i, :]
                    Yid[i, max_lag:] = np.dot(Phi_i, theta_i)

        # Create G_tf and H_tf transfer functions
        G_tf, H_tf = self._create_transfer_functions_arx(
            A_coeffs, B_coeffs, na, nb, nk, ny, nu, sample_time
        )

        model = self._create_transfer_function(
            A_coeffs, B_coeffs, na, nb, nk, ny, nu, sample_time
        )

        # Attach transfer functions and predictions to model
        model.G_tf = G_tf
        model.H_tf = H_tf
        model.Yid = Yid

        return model

    def _create_regression_matrix(self, u, y, na, nb, nk, ny, nu, N):
        """
        Create regression matrix Phi and output matrix y for least squares.

        This function automatically uses the Numba-compiled version when available
        for improved performance.

        Parameters:
        -----------
        u, y : ndarray
            Input and output data
        na, nb, nk : int
            Model orders and delay
        ny, nu : int
            Number of outputs and inputs
        N : int
            Number of data points

        Returns:
        --------
        Phi : ndarray
            Regression matrix
        y_matrix : ndarray
            Output matrix
        """
        if NUMBA_AVAILABLE and create_regression_matrix_arx_compiled is not None:
            return create_regression_matrix_arx_compiled(u, y, na, nb, nk, ny, nu, N)
        else:
            # Fallback to original implementation
            # Determine effective data length
            max_lag = max(na, nb + nk - 1)
            N_eff = N - max_lag

            if N_eff <= 0:
                raise ValueError(
                    f"Not enough data points. Need at least {max_lag + 1} samples, got {N}"
                )

            # Output matrix - trimmed for effective length
            y_matrix = y[:, max_lag:N]
            # Return dummy Phi since we construct per-output matrices in identify()
            Phi = np.zeros((N_eff, 1))  # Not used in MIMO case

            return Phi, y_matrix

    def _create_transfer_functions_arx(
        self, A_coeffs, B_coeffs, na, nb, nk, ny, nu, Ts
    ):
        """
        Create G_tf and H_tf transfer functions for ARX.

        For ARX: H_tf = 1 (unity, since ARX has no noise model).

        Parameters:
        -----------
        A_coeffs, B_coeffs : ndarray
            AR and exogenous coefficients
        na, nb, nk : int
            Model orders and delay
        ny, nu : int
            Number of outputs and inputs
        Ts : float
            Sampling time

        Returns:
        --------
        G_tf, H_tf : control.TransferFunction
            Deterministic and noise transfer functions.
        """
        if ny == 1 and nu == 1:
            polynomial_length = max(na, nb + nk - 1) + 1
            numerator = np.zeros(polynomial_length)
            numerator[nk : nk + nb] = B_coeffs[0, :]
            denominator = np.zeros(polynomial_length)
            denominator[0] = 1.0
            denominator[1 : na + 1] = -A_coeffs[0, :]
            G_tf = control.tf(numerator, denominator, dt=Ts)
        else:
            A, B, C, D = self._realize_mimo_arx(A_coeffs, B_coeffs, na, nb, nk, ny, nu)
            G_tf = control.ss2tf(control.ss(A, B, C, D, dt=Ts))

        H_tf = identity_transfer_function(ny, Ts)
        return G_tf, H_tf

    def _create_transfer_function(self, A_coeffs, B_coeffs, na, nb, nk, ny, nu, Ts):
        """
        Create a state-space model from ARX parameters.

        Parameters:
        -----------
        A_coeffs, B_coeffs : ndarray
            AR and exogenous coefficients
        na, nb, nk : int
            Model orders and delay
        ny, nu : int
            Number of outputs and inputs
        Ts : float
            Sampling time

        Returns:
        --------
        model : StateSpaceModel
            State-space model representation
        """
        # For simple SISO case, create a transfer function
        if ny == 1 and nu == 1:
            transfer_function, _ = self._create_transfer_functions_arx(
                A_coeffs, B_coeffs, na, nb, nk, ny, nu, Ts
            )
            A, B, C, D = realize_transfer_function(transfer_function)
        else:
            A, B, C, D = self._realize_mimo_arx(A_coeffs, B_coeffs, na, nb, nk, ny, nu)

        return StateSpaceModel(
            A=A,
            B=B,
            C=C,
            D=D,
            K=np.zeros((A.shape[0], C.shape[0])),
            Q=np.eye(A.shape[0]),
            R=np.eye(C.shape[0]),
            S=np.zeros((A.shape[0], C.shape[0])),
            ts=Ts,
            Vn=0.01,
        )

    def _realize_mimo_arx(self, A_coeffs, B_coeffs, na, nb, nk, ny, nu):
        input_history = max(0, nk + nb - 1)
        output_states = na * ny
        input_states = input_history * nu
        n_states = max(1, output_states + input_states)

        A = np.zeros((n_states, n_states))
        B = np.zeros((n_states, nu))
        C = np.zeros((ny, n_states))
        D = np.zeros((ny, nu))

        for lag in range(na):
            block = slice(lag * ny, (lag + 1) * ny)
            C[:, block] = A_coeffs[:, lag, :]

        input_offset = output_states
        coefficient_blocks = B_coeffs.reshape(ny, nb, nu)
        for lag in range(nb):
            delay = nk + lag
            if delay == 0:
                D += coefficient_blocks[:, lag, :]
            else:
                block_start = input_offset + (delay - 1) * nu
                C[:, block_start : block_start + nu] += coefficient_blocks[:, lag, :]

        if na > 0:
            A[:ny, :] = C
            B[:ny, :] = D
            for lag in range(1, na):
                destination = slice(lag * ny, (lag + 1) * ny)
                source = slice((lag - 1) * ny, lag * ny)
                A[destination, source] = np.eye(ny)

        if input_history > 0:
            B[input_offset : input_offset + nu, :] = np.eye(nu)
            for lag in range(1, input_history):
                destination_start = input_offset + lag * nu
                source_start = input_offset + (lag - 1) * nu
                A[
                    destination_start : destination_start + nu,
                    source_start : source_start + nu,
                ] = np.eye(nu)

        return A, B, C, D
