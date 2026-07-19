"""
Base classes for system identification algorithms.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

from sippy import systems as control

if TYPE_CHECKING:
    from .iddata import IDData


def resolve_identification_data(
    y: Optional[np.ndarray],
    u: Optional[np.ndarray],
    iddata: Optional["IDData"],
    *,
    tsample: float = 1.0,
    input_required: bool = True,
) -> tuple[np.ndarray, Optional[np.ndarray], float]:
    if iddata is not None and (y is not None or u is not None):
        raise ValueError("Provide either iddata or (y, u), but not both")

    if iddata is not None:
        y = iddata.get_output_array()
        u = iddata.get_input_array()
        tsample = iddata.sample_time
    elif y is None or (input_required and u is None):
        requirement = "y" if not input_required else "both y and u"
        raise ValueError(f"Provide either iddata or {requirement}")

    y_array = np.atleast_2d(np.asarray(y, dtype=float))
    u_array = None if u is None else np.atleast_2d(np.asarray(u, dtype=float))
    if u_array is not None and u_array.shape[1] != y_array.shape[1]:
        raise ValueError("Input and output must share the same number of samples")
    return y_array, u_array, float(tsample)


def realize_transfer_function(transfer_function: object) -> tuple[np.ndarray, ...]:
    realization = control.tf2ss(transfer_function)
    return realization.A, realization.B, realization.C, realization.D


def identity_transfer_function(
    size: int, sample_time: float
) -> control.TransferFunction:
    numerator = [
        [[1.0] if output == input_ else [0.0] for input_ in range(size)]
        for output in range(size)
    ]
    denominator = [[[1.0] for _ in range(size)] for _ in range(size)]
    return control.tf(numerator, denominator, dt=sample_time)


class IdentificationAlgorithm(ABC):
    """Abstract base class for system identification algorithms."""

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def identify(
        self,
        y: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
        iddata: Optional["IDData"] = None,
        **kwargs,
    ) -> "StateSpaceModel":
        """
        Perform system identification.

        Args:
            y: Output data (outputs x time_steps) - alternative to iddata
            u: Input data (inputs x time_steps) - alternative to iddata
            iddata: IDData object containing input and output data
            **kwargs: Algorithm-specific parameters

        Returns:
            StateSpaceModel: Identified model

        Note:
            Either (y, u) or iddata should be provided, but not both.
        """
        pass

    @abstractmethod
    def validate_parameters(self, **kwargs) -> bool:
        """Validate algorithm-specific parameters."""
        pass


class StateSpaceModel:
    """Enhanced state-space model container."""

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        K: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        S: np.ndarray,
        ts: float,
        Vn: Union[float, np.ndarray],
        G_tf: Optional[object] = None,
        H_tf: Optional[object] = None,
        Yid: Optional[np.ndarray] = None,
        identification_info: Optional[dict] = None,
        is_parametric: bool = True,
    ):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.K = K
        self.Q = Q
        self.R = R
        self.S = S
        self.ts = ts
        self.Vn = Vn
        self.n = A.shape[0]  # State dimension

        # Transfer functions and identification metadata
        self.G_tf = G_tf  # Deterministic transfer function G(q) = B/A
        self.H_tf = H_tf  # Noise transfer function H(q) = C/A
        self.Yid = Yid  # One-step-ahead predictions from identification
        self.identification_info = identification_info or {}
        self.is_parametric = bool(is_parametric)

        if not self.is_parametric or B.size == 0 or B.shape[1] == 0:
            self.G = None
        else:
            self.G = control.ss(A, B, C, D, dt=ts)

        self.x0 = np.zeros((self.n, 1))

        # Calculate observer matrices if possible
        try:
            self.A_K = A - np.dot(K, C)
            self.B_K = B - np.dot(K, D)
        except (ValueError, IndexError, TypeError):
            self.A_K = np.array([])
            self.B_K = np.array([])

    def is_stable(self) -> bool:
        """Check if the system matrix A is stable."""
        if not self.is_parametric:
            raise NotImplementedError(
                "Stability is not defined for a non-parametric frequency response"
            )
        try:
            eigenvals = np.linalg.eigvals(self.A)
            return np.all(np.abs(eigenvals) < 1.0)
        except (ValueError, np.linalg.LinAlgError):
            return False

    def get_natural_frequencies(self) -> np.ndarray:
        """Get natural frequencies of the system."""
        if not self.is_parametric:
            raise NotImplementedError(
                "Natural frequencies require a parametric model realization"
            )
        try:
            eigenvals = np.linalg.eigvals(self.A)
            return np.abs(np.angle(eigenvals) / (2 * np.pi * self.ts))
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
            return np.array([])

    def get_damping_ratios(self) -> np.ndarray:
        """Get damping ratios of the system."""
        if not self.is_parametric:
            raise NotImplementedError(
                "Damping ratios require a parametric model realization"
            )
        try:
            eigenvals = np.linalg.eigvals(self.A)
            return -np.real(eigenvals) / np.abs(eigenvals)
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
            return np.array([])

    def get_fir_coefficients(
        self, inputs: list, outputs: list, sampling: float, tss: float
    ) -> dict:
        """
        Get FIR coefficients for the model.

        Parameters:
        -----------
        inputs : list
            List of input variable names
        outputs : list
            List of output variable names
        sampling : float
            Sampling rate in seconds
        tss : float
            Time to steady state in minutes

        Returns:
        --------
        fir_model : dict
            Nested dictionary of FIR coefficients
        """
        from ..utils.simulation_utils import get_fir_coef

        if not self.is_parametric:
            raise NotImplementedError(
                "FIR conversion requires a parametric model; fit the frequency "
                "response first"
            )
        return get_fir_coef(self, inputs, outputs, sampling, tss)

    def get_step_response(self, inputs: list, outputs: list) -> dict:
        """
        Get step response for the model.

        Parameters:
        -----------
        inputs : list
            List of input variable names
        outputs : list
            List of output variable names

        Returns:
        --------
        step_response : dict
            Nested dictionary of step responses
        """
        from ..utils.simulation_utils import get_step_response

        fir_model = self.get_fir_coefficients(inputs, outputs, 1.0, 60)
        return get_step_response(fir_model)

    def frequency_response(
        self, omega: Optional[np.ndarray] = None
    ) -> control.FrequencyResponseData:
        """Evaluate the identified response on an angular-frequency grid."""
        if self.is_parametric:
            system = self.G_tf or self.G or self.H_tf
            if system is None:
                raise NotImplementedError(
                    "This identification result has no frequency-response model"
                )
            if omega is None:
                if not np.isfinite(self.ts) or self.ts <= 0:
                    raise ValueError("A positive sample time is required")
                omega = np.linspace(0.0, np.pi / self.ts, 512)
            return control.frequency_response(system, omega)

        info = self.identification_info
        if info.get("method") != "FD" or "frequency_response" not in info:
            raise NotImplementedError(
                "This non-parametric result has no stored frequency response"
            )
        stored = info["frequency_response"]
        if info.get("estimator") == "correlation":
            stored_omega = np.asarray(stored["omega_real"], dtype=float)
            stored_response = np.asarray(stored["G_smooth"], dtype=complex)[
                :, None, None
            ]
            nyquist_response = np.conj(stored_response[0:1])
            positive = stored_omega >= 0
            stored_omega = stored_omega[positive]
            stored_response = stored_response[positive]
            nyquist = np.pi / self.ts
            if stored_omega[-1] < nyquist:
                stored_omega = np.concatenate([stored_omega, [nyquist]])
                stored_response = np.concatenate(
                    [stored_response, nyquist_response], axis=0
                )
        else:
            stored_omega = np.asarray(stored["omega_real"], dtype=float)
            stored_response = np.asarray(stored["G"], dtype=complex)

        if omega is None:
            frequencies = stored_omega
            response = stored_response
        else:
            frequencies = np.asarray(omega, dtype=float)
            if frequencies.ndim != 1 or not np.all(np.isfinite(frequencies)):
                raise ValueError("Frequencies must be a finite one-dimensional array")
            tolerance = 1e-12 * max(1.0, float(np.max(np.abs(stored_omega))))
            if np.any(frequencies < stored_omega[0] - tolerance) or np.any(
                frequencies > stored_omega[-1] + tolerance
            ):
                raise ValueError(
                    "Requested frequencies are outside the identified range"
                )
            response = np.empty(
                (len(frequencies), stored_response.shape[1], stored_response.shape[2]),
                dtype=complex,
            )
            for output in range(stored_response.shape[1]):
                for input_ in range(stored_response.shape[2]):
                    channel = stored_response[:, output, input_]
                    response[:, output, input_] = np.interp(
                        frequencies, stored_omega, channel.real
                    ) + 1j * np.interp(frequencies, stored_omega, channel.imag)

        return control.FrequencyResponseData(
            frequencies.copy(), np.transpose(response, (1, 2, 0))
        )

    def get_model_uncertainty(
        self,
        input_data: np.ndarray,
        output_data: np.ndarray,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
        *,
        nperseg: Optional[int] = None,
        window: str = "hann",
        noverlap: int = 0,
        smoothing_bins: int = 5,
        confidence_levels: tuple[float, ...] = (0.68, 0.95),
    ):
        """
        Estimate empirical frequency-response uncertainty from validation data.

        Parameters:
        -----------
        input_data : np.ndarray
            Input signal data
        output_data : np.ndarray
            Output signal data
        input_name, output_name : str, optional
            Retained for compatibility; uncertainty is returned for every channel.

        Returns:
        --------
        FrequencyResponseUncertainty
            Model and empirical responses, coherence, residual spectrum, SNR,
            and jackknife confidence intervals for magnitude and phase. These
            intervals quantify non-parametric FRF sampling uncertainty; they do
            not claim algorithm-specific parameter covariance.
        """
        from .uncertainty import estimate_frequency_response_uncertainty

        del input_name, output_name
        uncertainty = estimate_frequency_response_uncertainty(
            input_data,
            output_data,
            dt=self.ts,
            nperseg=nperseg,
            window=window,
            noverlap=noverlap,
            smoothing_bins=smoothing_bins,
            confidence_levels=confidence_levels,
        )
        response = self.frequency_response(uncertainty.omega).frdata
        model_response = np.transpose(response, (2, 0, 1))
        return uncertainty.with_model_response(model_response)

    def simulate(self, u: np.ndarray, x0: np.ndarray = None) -> tuple:
        """
        Simulate the state-space model.

        Parameters:
        -----------
        u : np.ndarray
            Input signals (inputs x time_steps)
        x0 : np.ndarray, optional
            Initial state

        Returns:
        --------
        x : np.ndarray
            State trajectory
        y : np.ndarray
            Output signals
        """
        from ..utils.simulation_utils import simulate_ss_system

        if not self.is_parametric:
            raise NotImplementedError(
                "Simulation is not defined for a non-parametric frequency response; "
                "fit a parametric model first"
            )
        return simulate_ss_system(self.A, self.B, self.C, self.D, u, x0)

    def supports_optimization_methods(self) -> bool:
        """
        Check if the model supports various optimization methods.

        Returns:
        --------
        bool : True if optimization methods are supported
        """
        return self.is_parametric


class SystemIdentificationConfig:
    """Configuration container for system identification."""

    def __init__(
        self,
        method: str = "N4SID",
        centering: str = "None",
        ic: str = "None",
        tsample: float = 1.0,
        ss_f: int = 20,
        ss_threshold: float = 0.1,
        ss_max_order: Optional[int] = None,
        ss_fixed_order: Optional[int] = 1,  # Default to 1 to avoid issues
        ss_orders: List[int] = [1, 10],
        ss_d_required: bool = False,
        ss_a_stability: bool = False,
        # AR* algorithm parameters (for compatibility with master branch)
        na: Optional[Union[int, List[int]]] = None,
        nb: Optional[Union[int, List[int]]] = None,
        nc: Optional[Union[int, List[int]]] = None,
        nd: Optional[Union[int, List[int]]] = None,
        nf: Optional[Union[int, List[int]]] = None,
        nk: Optional[Union[int, List[int]]] = None,
        # Master branch style parameters
        arx_orders: Optional[List] = None,
        armax_orders: Optional[List] = None,
        ararx_orders: Optional[List] = None,
        ararmax_orders: Optional[List] = None,
        bj_orders: Optional[List] = None,
        # Additional parameters
        max_iterations: int = 200,
        stab_marg: float = 1.0,
        stab_cons: bool = False,
    ):
        # Subspace method parameters
        self.method = method
        self.centering = centering
        self.ic = ic
        self.tsample = tsample
        self.ss_f = ss_f
        self.ss_threshold = ss_threshold
        self.ss_max_order = ss_max_order
        self.ss_fixed_order = ss_fixed_order
        self.ss_orders = ss_orders
        self.ss_d_required = ss_d_required
        self.ss_a_stability = ss_a_stability

        # AR* algorithm parameters (individual)
        self.na = na
        self.nb = nb
        self.nc = nc
        self.nd = nd
        self.nf = nf
        self.nk = nk

        # Master branch style parameters
        self.arx_orders = arx_orders
        self.armax_orders = armax_orders
        self.ararx_orders = ararx_orders
        self.ararmax_orders = ararmax_orders
        self.bj_orders = bj_orders

        # Additional parameters
        self.max_iterations = max_iterations
        self.stab_marg = stab_marg
        self.stab_cons = stab_cons
