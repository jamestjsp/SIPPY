"""
FIR (Finite Impulse Response) identification algorithm.
"""

from typing import TYPE_CHECKING, Optional

import control
import numpy as np
from numpy.linalg import lstsq

from ..base import IdentificationAlgorithm, StateSpaceModel, identity_transfer_function

if TYPE_CHECKING:
    from ..iddata import IDData

# Import compiled utilities for performance
try:
    from ...utils.compiled_utils import (
        NUMBA_AVAILABLE,
        create_regression_matrix_fir_compiled,
    )
except ImportError:
    create_regression_matrix_fir_compiled = None
    NUMBA_AVAILABLE = False


class FIRAlgorithm(IdentificationAlgorithm):
    """
    FIR (Finite Impulse Response) identification algorithm.

    The FIR model structure is:
    y(k) = b1*u(k-nk) + b2*u(k-nk-1) + ... + bnb*u(k-nk-nb+1) + e(k)

    where:
    - b1, b2, ..., bnb are the FIR coefficients
    - nb is the number of FIR coefficients
    - nk is the input delay (number of samples)
    - e(k) is white noise

    The algorithm uses least-squares regression to estimate the FIR coefficients.
    """

    def __init__(self):
        """Initialize FIR algorithm."""
        super().__init__()

    def get_algorithm_name(self) -> str:
        """Return algorithm name."""
        return "FIR"

    def validate_parameters(self, **kwargs) -> bool:
        """
        Validate FIR-specific parameters.

        Parameters:
        -----------
        **kwargs : dict
            Parameters to validate including nb, nk

        Returns:
        --------
        bool
            True if parameters are valid
        """
        nb = kwargs.get("nb", 1)
        nk = kwargs.get("nk", 1)

        if nb <= 0:
            raise ValueError("Number of FIR coefficients must be positive")
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
        Identify FIR model from input-output data.

        Parameters:
        -----------
        y : np.ndarray, optional
            Output data (outputs x time_steps)
        u : np.ndarray, optional
            Input data (inputs x time_steps)
        iddata : IDData, optional
            Input-output data container
        **kwargs : dict
            Configuration parameters including nb, nk, tsample

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

        # Extract configuration parameters (FIR specific: only nb and nk, no na)
        nb = kwargs.get("nb", 1)
        nk = kwargs.get("nk", 1)

        # Validate parameters
        self.validate_parameters(nb=nb, nk=nk)

        # Get data dimensions
        ny, N = y.shape
        nu, _ = u.shape

        # Calculate effective data length
        N_eff = N - nb - nk + 1

        if N_eff <= 0:
            raise ValueError(
                f"Not enough data points. Need at least {nb + nk} samples, got {N}"
            )

        # Prefer compiled regression builder when available
        if NUMBA_AVAILABLE and create_regression_matrix_fir_compiled is not None:
            Phi, y_matrix = create_regression_matrix_fir_compiled(
                np.ascontiguousarray(u), np.ascontiguousarray(y), nb, nk, ny, nu, N
            )
            fir_coeffs = np.zeros((ny, nb * nu))
            for i in range(ny):
                theta_i, _, _, _ = lstsq(Phi, y_matrix[i, :], rcond=None)
                fir_coeffs[i, :] = theta_i
        else:
            # Fallback: construct per-output regression matrices in Python
            fir_coeffs = np.zeros((ny, nb * nu))
            for i in range(ny):
                Phi_i = np.zeros((N_eff, nb * nu))
                col = 0
                for lag in range(nb):
                    for j in range(nu):
                        delay_idx = nb - 1 - lag
                        if delay_idx >= 0 and delay_idx + N_eff <= N:
                            start = delay_idx
                            Phi_i[:, col] = u[j, start : start + N_eff]
                        col += 1
                theta_i, _, _, _ = lstsq(
                    Phi_i, y[i, nk + nb - 1 : nk + nb - 1 + N_eff], rcond=None
                )
                fir_coeffs[i, :] = theta_i

        # Compute one-step-ahead predictions (Yid) for identification data
        N_eff_yid = N - nb - nk + 1
        Yid = np.zeros_like(y)
        Yid[:, : nk + nb - 1] = y[:, : nk + nb - 1]  # Copy initial values

        if NUMBA_AVAILABLE and create_regression_matrix_fir_compiled is not None:
            Phi, _ = create_regression_matrix_fir_compiled(
                np.ascontiguousarray(u), np.ascontiguousarray(y), nb, nk, ny, nu, N
            )
            for i in range(ny):
                Yid[i, nk + nb - 1 :] = (Phi @ fir_coeffs[i, :]).flatten()
        else:
            # Fallback: rebuild Phi per-output
            Phi_yid_all = np.zeros((ny, N_eff_yid, nb * nu))
            for i in range(ny):
                Phi_i = Phi_yid_all[i, :, :]
                col = 0
                for lag in range(nb):
                    for j in range(nu):
                        delay_idx = nb - 1 - lag
                        if delay_idx >= 0 and delay_idx + N_eff_yid <= N:
                            start = delay_idx
                            Phi_i[:, col] = u[j, start : start + N_eff_yid]
                        col += 1
                Yid[i, nk + nb - 1 :] = np.dot(Phi_i, fir_coeffs[i, :]).flatten()

        # Create G_tf and H_tf transfer functions
        G_tf, H_tf = self._create_transfer_functions_fir(
            fir_coeffs, nb, nk, ny, nu, sample_time
        )

        model = self._create_state_space_from_fir(
            fir_coeffs, nb, nk, ny, nu, sample_time
        )

        # Attach transfer functions and predictions to model
        model.G_tf = G_tf
        model.H_tf = H_tf
        model.Yid = Yid

        return model

    def _create_regression_matrix(self, u, y, nb, nk, ny, nu, N):
        """
        Create regression matrix Phi and output matrix y for least squares.

        This function automatically uses the Numba-compiled version when available
        for improved performance.

        Parameters:
        -----------
        u, y : ndarray
            Input and output data
        nb, nk : int
            Model coefficients count and delay
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
        if NUMBA_AVAILABLE and create_regression_matrix_fir_compiled is not None:
            return create_regression_matrix_fir_compiled(u, y, nb, nk, ny, nu, N)
        else:
            # Fallback to original implementation
            # Determine effective data length
            max_lag = nb + nk - 1
            N_eff = N - max_lag

            if N_eff <= 0:
                raise ValueError(
                    f"Not enough data points. Need at least {max_lag + 1} samples, got {N}"
                )

            # Initialize regression matrix
            n_params = nb * ny * nu
            Phi = np.zeros((N_eff, n_params))

            # Fill FIR part (lagged inputs)
            for k in range(nb):
                for i in range(nu):
                    # For MIMO, each input affects all outputs
                    for j in range(ny):
                        col_idx = k * ny * nu + i * ny + j
                        delay_idx = max_lag - 1 - k
                        if delay_idx >= 0 and delay_idx + N_eff <= N:
                            Phi[:, col_idx] = u[i, delay_idx : delay_idx + N_eff]

            # Output matrix
            y_matrix = y[:, max_lag:N]

            return Phi, y_matrix

    def _create_transfer_functions_fir(self, fir_coeffs, nb, nk, ny, nu, Ts):
        """
        Create G_tf and H_tf transfer functions for FIR.

        For FIR: G_tf = B(q) (FIR polynomial), H_tf = 1 (white noise only).

        Parameters:
        -----------
        fir_coeffs : ndarray
            FIR coefficient array (ny x nb*nu)
        nb, nk : int
            Number of FIR coefficients and delay
        ny, nu : int
            Number of outputs and inputs
        Ts : float
            Sampling time

        Returns:
        --------
        G_tf, H_tf : control.TransferFunction
            Deterministic and noise transfer functions.
        """
        polynomial_length = nb + nk
        coefficient_blocks = fir_coeffs.reshape(ny, nb, nu)
        numerators = []
        denominators = []
        for output in range(ny):
            numerator_row = []
            denominator_row = []
            for input_ in range(nu):
                numerator = np.zeros(polynomial_length)
                numerator[nk : nk + nb] = coefficient_blocks[output, :, input_]
                denominator = np.zeros(polynomial_length)
                denominator[0] = 1.0
                numerator_row.append(numerator)
                denominator_row.append(denominator)
            numerators.append(numerator_row)
            denominators.append(denominator_row)

        G_tf = control.tf(numerators, denominators, dt=Ts)
        H_tf = identity_transfer_function(ny, Ts)
        return G_tf, H_tf

    def _create_state_space_from_fir(self, fir_coeffs, nb, nk, ny, nu, Ts):
        """
        Create a delay-chain state-space model from FIR coefficients.

        Parameters:
        -----------
        fir_coeffs : ndarray
            FIR coefficients array
        nb, nk : int
            Number of coefficients and input delay
        ny, nu : int
            Number of outputs and inputs
        Ts : float
            Sampling time

        Returns:
        --------
        model : StateSpaceModel
            State-space model representation
        """
        input_history = max(0, nk + nb - 1)
        n_states = max(1, input_history * nu)
        A = np.zeros((n_states, n_states))
        B = np.zeros((n_states, nu))
        C = np.zeros((ny, n_states))
        D = np.zeros((ny, nu))

        coefficient_blocks = fir_coeffs.reshape(ny, nb, nu)
        for lag in range(nb):
            delay = nk + lag
            if delay == 0:
                D += coefficient_blocks[:, lag, :]
            else:
                block_start = (delay - 1) * nu
                C[:, block_start : block_start + nu] += coefficient_blocks[:, lag, :]

        if input_history > 0:
            B[:nu, :] = np.eye(nu)
            for lag in range(1, input_history):
                destination = slice(lag * nu, (lag + 1) * nu)
                source = slice((lag - 1) * nu, lag * nu)
                A[destination, source] = np.eye(nu)

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
