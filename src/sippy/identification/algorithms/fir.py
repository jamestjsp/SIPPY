"""
FIR (Finite Impulse Response) identification algorithm.
"""

from typing import TYPE_CHECKING, Optional

import numpy as np
import scipy
from numpy.linalg import lstsq

from sippy import systems as control

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


def _kernel_matrix(
    kernel,
    coefficient_count,
    decay,
    correlation=0.0,
    input_count=1,
):
    indices = np.arange(1, coefficient_count + 1, dtype=float)
    if kernel == "tc":
        base = decay ** np.maximum(indices[:, None], indices[None, :])
    elif kernel == "dc":
        base = decay ** ((indices[:, None] + indices[None, :]) / 2.0)
        base *= correlation ** np.abs(indices[:, None] - indices[None, :])
    else:
        raise ValueError(f"Unknown FIR regularization kernel: {kernel}")
    if input_count == 1:
        return base
    return np.kron(base, np.eye(input_count))


def _decode_kernel_parameters(parameters, kernel):
    gamma = np.exp(parameters[0])
    decay = scipy.special.expit(parameters[1])
    correlation = np.tanh(parameters[2]) if kernel == "dc" else None
    return gamma, decay, correlation


def _kernel_posterior(
    parameters,
    kernel,
    gram,
    response_product,
    response_norm,
    sample_count,
    coefficient_count,
    input_count,
    return_coefficients=False,
):
    gamma, decay, correlation = _decode_kernel_parameters(parameters, kernel)
    kernel_matrix = _kernel_matrix(
        kernel,
        coefficient_count,
        decay,
        correlation=correlation or 0.0,
        input_count=input_count,
    )
    try:
        prior_factor = scipy.linalg.cholesky(
            gamma * kernel_matrix,
            lower=True,
            check_finite=False,
        )
        system = np.eye(gram.shape[0]) + prior_factor.T @ gram @ prior_factor
        system_factor = scipy.linalg.cho_factor(
            system,
            lower=True,
            check_finite=False,
        )
        transformed_response = prior_factor.T @ response_product
        solved_response = scipy.linalg.cho_solve(
            system_factor,
            transformed_response,
            check_finite=False,
        )
    except (np.linalg.LinAlgError, ValueError):
        if return_coefficients:
            raise
        return np.inf

    quadratic = response_norm - float(transformed_response @ solved_response)
    floor = np.finfo(np.float64).eps * max(response_norm, 1.0)
    if not np.isfinite(quadratic) or quadratic <= floor:
        if return_coefficients:
            quadratic = floor
        else:
            return np.inf
    noise_variance = quadratic / sample_count
    log_determinant = 2.0 * np.sum(np.log(np.diag(system_factor[0])))
    objective = sample_count * np.log(noise_variance) + log_determinant
    if return_coefficients:
        coefficients = prior_factor @ solved_response
        return coefficients, noise_variance, gamma, decay, correlation, objective
    return objective


def _fit_kernel_regularized_fir(Phi, target, kernel, coefficient_count, input_count):
    gram = Phi.T @ Phi
    response_product = Phi.T @ target
    response_norm = float(target @ target)
    sample_count = target.size
    least_squares_coefficients = lstsq(Phi, target, rcond=None)[0]
    residual = target - Phi @ least_squares_coefficients
    noise_estimate = max(
        float(residual @ residual) / sample_count,
        np.finfo(np.float64).eps * max(response_norm / sample_count, 1.0),
    )

    best_parameters = None
    best_objective = np.inf
    for decay_start, correlation_start in (
        (0.5, 0.0),
        (0.8, 0.75),
        (0.95, -0.25),
    ):
        initial_kernel = _kernel_matrix(
            kernel,
            coefficient_count,
            decay_start,
            correlation=correlation_start,
            input_count=input_count,
        )
        scale_estimate = max(
            float(least_squares_coefficients @ least_squares_coefficients)
            / np.trace(initial_kernel),
            np.finfo(np.float64).eps,
        )
        initial = [
            np.log(scale_estimate / noise_estimate),
            scipy.special.logit(decay_start),
        ]
        bounds = [(-20.0, 20.0), (-8.0, 8.0)]
        if kernel == "dc":
            initial.append(np.arctanh(correlation_start))
            bounds.append((-4.0, 4.0))
        result = scipy.optimize.minimize(
            _kernel_posterior,
            np.asarray(initial),
            args=(
                kernel,
                gram,
                response_product,
                response_norm,
                sample_count,
                coefficient_count,
                input_count,
            ),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 100, "ftol": 1e-9},
        )
        if np.isfinite(result.fun) and result.fun < best_objective:
            best_parameters = result.x
            best_objective = float(result.fun)

    if best_parameters is None:
        raise np.linalg.LinAlgError("FIR kernel hyperparameter optimization failed")

    coefficients, noise_variance, gamma, decay, correlation, objective = (
        _kernel_posterior(
            best_parameters,
            kernel,
            gram,
            response_product,
            response_norm,
            sample_count,
            coefficient_count,
            input_count,
            return_coefficients=True,
        )
    )
    hyperparameters = {
        "kernel": kernel,
        "decay": float(decay),
        "scale": float(gamma * noise_variance),
        "noise_variance": float(noise_variance),
        "marginal_likelihood_objective": float(objective),
    }
    if correlation is not None:
        hyperparameters["correlation"] = float(correlation)
    return coefficients, hyperparameters


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

    The algorithm supports plain least squares and empirical-Bayes TC/DC kernel
    regularization for smooth, exponentially decaying impulse responses.
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
            Parameters to validate including nb, nk, and regularization

        Returns:
        --------
        bool
            True if parameters are valid
        """
        nb = kwargs.get("nb", 1)
        nk = kwargs.get("nk", 1)
        regularization = str(kwargs.get("regularization", "none")).lower()

        if nb <= 0:
            raise ValueError("Number of FIR coefficients must be positive")
        if nk < 0:
            raise ValueError("Input delay (nk) must be non-negative")
        if regularization not in {"none", "tc", "dc"}:
            raise ValueError("FIR regularization must be one of 'none', 'tc', or 'dc'")

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
            Configuration parameters including nb, nk, regularization, and tsample

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
        regularization = str(kwargs.get("regularization", "none")).lower()

        # Validate parameters
        self.validate_parameters(nb=nb, nk=nk, regularization=regularization)

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
        else:
            Phi = np.zeros((N_eff, nb * nu))
            for lag in range(nb):
                for input_ in range(nu):
                    column = lag * nu + input_
                    delay_index = nb - 1 - lag
                    Phi[:, column] = u[input_, delay_index : delay_index + N_eff]
            y_matrix = y[:, nk + nb - 1 : nk + nb - 1 + N_eff]

        fir_coeffs = np.zeros((ny, nb * nu))
        kernel_hyperparameters = []
        for output in range(ny):
            if regularization == "none":
                fir_coeffs[output] = lstsq(
                    Phi,
                    y_matrix[output],
                    rcond=None,
                )[0]
            else:
                fir_coeffs[output], hyperparameters = _fit_kernel_regularized_fir(
                    Phi,
                    y_matrix[output],
                    regularization,
                    nb,
                    nu,
                )
                kernel_hyperparameters.append(hyperparameters)

        # Compute one-step-ahead predictions (Yid) for identification data
        Yid = np.zeros_like(y)
        Yid[:, : nk + nb - 1] = y[:, : nk + nb - 1]  # Copy initial values

        for output in range(ny):
            Yid[output, nk + nb - 1 :] = Phi @ fir_coeffs[output]

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
        model.identification_info["fit_start"] = nk + nb - 1
        model.identification_info["fir_coefficients"] = fir_coeffs.copy()
        model.identification_info["regularization"] = regularization
        model.identification_info["kernel_hyperparameters"] = kernel_hyperparameters

        return model

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
            K=None,
            Q=None,
            R=None,
            S=None,
            ts=Ts,
            Vn=None,
        )
