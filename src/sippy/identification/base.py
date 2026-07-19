"""
Base classes for system identification algorithms.
"""

from abc import ABC, abstractmethod
from functools import wraps
from typing import TYPE_CHECKING, Any, List, Optional, Union

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

    covariance_source: Optional[str] = None
    kalman_gain_source: Optional[str] = None

    def __init_subclass__(cls, **kwargs):
        """Wrap concrete estimators so result normalization cannot be skipped."""
        super().__init_subclass__(**kwargs)
        identify = cls.__dict__.get("identify")
        if identify is None or getattr(identify, "_sippy_result_finalizer", False):
            return

        @wraps(identify)
        def finalized_identify(self, y=None, u=None, iddata=None, **options):
            model = identify(self, y=y, u=u, iddata=iddata, **options)
            result_y = y
            result_u = u
            result_data = iddata
            if result_data is None and hasattr(y, "get_output_array"):
                result_data = y
            if result_data is not None:
                result_y = result_data.get_output_array()
                result_u = result_data.get_input_array()
            if result_y is None:
                return model
            if model.B.shape[1] == 0 and model.G_tf is None:
                result_u = None
            return self.finalize_result(
                model,
                result_y,
                result_u,
                options=options,
            )

        finalized_identify._sippy_result_finalizer = True
        cls.identify = finalized_identify

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

    def finalize_result(
        self,
        model: "StateSpaceModel",
        y: np.ndarray,
        u: Optional[np.ndarray],
        *,
        covariance_source: Optional[str] = None,
        kalman_gain_source: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> "StateSpaceModel":
        """Finalize an algorithm result under the shared result contract."""
        method_getter = getattr(self, "get_algorithm_name", None)
        method = (
            method_getter() if method_getter is not None else self.__class__.__name__
        )
        if covariance_source is None:
            covariance_source = self.covariance_source
        if kalman_gain_source is None:
            kalman_gain_source = self.kalman_gain_source
        return model.finalize_identification(
            method=method,
            input_data=u,
            output_data=y,
            covariance_source=covariance_source,
            kalman_gain_source=kalman_gain_source,
            options=options,
        )


class StateSpaceModel:
    """Enhanced state-space model container."""

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        K: Optional[np.ndarray],
        Q: Optional[np.ndarray],
        R: Optional[Union[float, np.ndarray]],
        S: Optional[np.ndarray],
        ts: float,
        Vn: Optional[Union[float, np.ndarray]],
        G_tf: Optional[object] = None,
        H_tf: Optional[object] = None,
        Yid: Optional[np.ndarray] = None,
        identification_info: Optional[dict] = None,
        is_parametric: bool = True,
        x0: Optional[np.ndarray] = None,
    ):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.K = K
        self.Q = Q
        self.R = R
        self.S = S
        self.ts = float(ts)
        self.Vn = Vn
        self.n = A.shape[0]  # State dimension

        # Transfer functions and identification metadata
        self.G_tf = G_tf  # Deterministic transfer function G(q) = B/A
        self.H_tf = H_tf  # Noise transfer function H(q) = C/A
        self.Yid = Yid  # One-step-ahead predictions from identification
        self.identification_info = identification_info or {}
        self.is_parametric = bool(is_parametric)
        self.method = self.identification_info.get("method")
        self.ninputs = int(self.identification_info.get("n_inputs", B.shape[1]))
        self.noutputs = int(self.identification_info.get("n_outputs", C.shape[0]))
        self.residual_covariance: Optional[np.ndarray] = None
        self.fit_start = int(self.identification_info.get("fit_start", 0))
        self._identification_input: Optional[np.ndarray] = None
        self._identification_output: Optional[np.ndarray] = None
        self._residuals: Optional[np.ndarray] = None
        self.capabilities: dict[str, bool] = {}

        if not self.is_parametric or B.size == 0 or B.shape[1] == 0:
            self.G = None
        else:
            self.G = control.ss(A, B, C, D, dt=self.ts)

        self.x0 = (
            np.zeros((self.n, 1))
            if x0 is None
            else np.asarray(x0, dtype=float).reshape(self.n, 1)
        )

        # Calculate observer matrices if possible
        try:
            self.A_K = A - np.dot(K, C)
            self.B_K = B - np.dot(K, D)
        except (ValueError, IndexError, TypeError):
            self.A_K = np.array([])
            self.B_K = np.array([])

        self._update_capabilities()

    @property
    def deterministic_model(self) -> Optional[object]:
        """Canonical input-to-output model, when one was identified."""
        return self.G_tf or self.G

    @property
    def innovations_model(self) -> Optional[object]:
        """Canonical innovations-to-output model, when one was identified."""
        return self.H_tf

    def _update_capabilities(self) -> None:
        frequency_response = (
            self.deterministic_model is not None
            or self.innovations_model is not None
            or (
                not self.is_parametric
                and "frequency_response" in self.identification_info
            )
        )
        simulation = self.is_parametric and self.deterministic_model is not None
        dynamic_model = self.deterministic_model or self.innovations_model
        self.capabilities = {
            "frequency_response": frequency_response,
            "uncertainty": frequency_response and self.ninputs > 0,
            "simulation": simulation,
            "prediction": simulation or self.Yid is not None,
            "one_step_prediction": self.K is not None
            or self._has_invertible_innovations_model(),
            "residuals": self._residuals is not None,
            "fit": self._residuals is not None,
            "stability": self.is_parametric and dynamic_model is not None,
            "modal_properties": self.is_parametric and dynamic_model is not None,
            "time_response": simulation,
            "innovations_response": self.innovations_model is not None,
            "stochastic_state_space": all(
                value is not None for value in (self.K, self.Q, self.R, self.S)
            ),
        }

    def _has_invertible_innovations_model(self) -> bool:
        system = self.innovations_model
        if not isinstance(system, control.TransferFunction):
            return False
        if system.ninputs != self.noutputs or system.noutputs != self.noutputs:
            return False
        for output in range(self.noutputs):
            numerator = np.asarray(system.num[output][output], dtype=float)
            if numerator.size == 0 or np.isclose(numerator[0], 0.0):
                return False
            for input_ in range(self.noutputs):
                if input_ != output and np.any(np.abs(system.num[output][input_]) > 0):
                    return False
        return True

    def supports(self, operation: str) -> bool:
        """Return whether this result can perform an operation truthfully."""
        normalized = operation.lower().replace("-", "_").replace(" ", "_")
        if normalized not in self.capabilities:
            raise ValueError(f"Unknown identification-result operation: {operation}")
        return self.capabilities[normalized]

    def finalize_identification(
        self,
        *,
        method: str,
        input_data: Optional[np.ndarray],
        output_data: np.ndarray,
        covariance_source: Optional[str],
        kalman_gain_source: Optional[str],
        options: Optional[dict[str, Any]] = None,
    ) -> "StateSpaceModel":
        """Normalize provenance, fit statistics, and behavior after identification."""
        y = np.atleast_2d(np.asarray(output_data, dtype=float))
        u = (
            None
            if input_data is None
            else np.atleast_2d(np.asarray(input_data, dtype=float))
        )
        if u is not None and u.shape[1] != y.shape[1]:
            raise ValueError("Identification input and output sample counts differ")

        self.method = str(method).upper()
        self.noutputs = y.shape[0]
        self.ninputs = 0 if u is None else u.shape[0]
        self._identification_input = None if u is None else u.copy()
        self._identification_output = y.copy()
        requested_fit_start = int(self.identification_info.get("fit_start", 0))
        if requested_fit_start < 0:
            raise ValueError("fit_start must be non-negative")
        self.fit_start = min(requested_fit_start, y.shape[1])
        if requested_fit_start > y.shape[1]:
            self.identification_info["requested_fit_start"] = requested_fit_start

        if self.is_parametric and self.G_tf is not None:
            A, B, C, D = realize_transfer_function(self.G_tf)
            self.A, self.B, self.C, self.D = A, B, C, D
            self.G = control.ss(A, B, C, D, dt=self.ts)
            self.n = A.shape[0]
            self.x0 = np.zeros((self.n, 1))

        if covariance_source is None:
            self.Q = None
            self.R = None
            self.S = None
        if kalman_gain_source is None:
            self.K = None

        if self.K is None:
            self.A_K = np.array([])
            self.B_K = np.array([])
        else:
            self.A_K = self.A - self.K @ self.C
            self.B_K = self.B - self.K @ self.D

        if self.Yid is None and self.is_parametric:
            if self.K is not None:
                try:
                    self.Yid = self.predict(u=u, y=y)
                except (ValueError, NotImplementedError):
                    self.Yid = None
            elif u is not None and self.deterministic_model is not None:
                _, self.Yid = self.simulate(u)

        if self.Yid is not None:
            fitted = np.atleast_2d(np.asarray(self.Yid, dtype=float))
            if fitted.shape == y.shape:
                self.Yid = fitted
                if self.fit_start < y.shape[1]:
                    self._residuals = y - fitted
                    effective_residuals = self._residuals[:, self.fit_start :]
                    centered = effective_residuals - np.mean(
                        effective_residuals, axis=1, keepdims=True
                    )
                    denominator = max(effective_residuals.shape[1] - 1, 1)
                    self.residual_covariance = centered @ centered.T / denominator
                    self.Vn = float(np.mean(effective_residuals**2))
                else:
                    self.Vn = None

        provenance = {
            "kalman_gain": kalman_gain_source or "unavailable",
            "state_covariances": covariance_source or "unavailable",
            "residual_covariance": (
                "empirical" if self.residual_covariance is not None else "unavailable"
            ),
        }
        self.identification_info.update(
            {
                "method": self.method,
                "model_type": "parametric" if self.is_parametric else "nonparametric",
                "n_inputs": self.ninputs,
                "n_outputs": self.noutputs,
                "sample_time": self.ts,
                "fit_start": self.fit_start,
                "provenance": provenance,
                "options": dict(options or {}),
            }
        )
        if self._residuals is not None:
            self.identification_info["fit"] = self.fit()
        self._update_capabilities()
        return self

    def is_stable(self) -> bool:
        """Check if the system matrix A is stable."""
        if not self.is_parametric:
            raise NotImplementedError(
                "Stability is not defined for a non-parametric frequency response"
            )
        try:
            eigenvals = self.poles()
            return np.all(np.abs(eigenvals) < 1.0)
        except (ValueError, np.linalg.LinAlgError):
            return False

    def get_natural_frequencies(self) -> np.ndarray:
        """Return undamped natural frequencies in cycles per unit time."""
        if not self.is_parametric:
            raise NotImplementedError(
                "Natural frequencies require a parametric model realization"
            )
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                continuous_poles = np.log(self.poles().astype(complex)) / self.ts
            return np.abs(continuous_poles) / (2 * np.pi)
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
            return np.array([])

    def get_damping_ratios(self) -> np.ndarray:
        """Return damping ratios from the continuous-equivalent poles."""
        if not self.is_parametric:
            raise NotImplementedError(
                "Damping ratios require a parametric model realization"
            )
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                continuous_poles = np.log(self.poles().astype(complex)) / self.ts
            magnitudes = np.abs(continuous_poles)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.divide(
                    -continuous_poles.real,
                    magnitudes,
                    out=np.full_like(magnitudes, np.nan),
                    where=magnitudes > 0,
                )
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
            return np.array([])

    def poles(self) -> np.ndarray:
        """Return poles of the canonical dynamic model."""
        if not self.is_parametric:
            raise NotImplementedError("Poles require a parametric model realization")
        system = self.deterministic_model or self.innovations_model
        if system is None:
            raise NotImplementedError("This result has no dynamic model")
        return control.poles(system)

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
        self, omega: Optional[np.ndarray] = None, *, response: str = "process"
    ) -> control.FrequencyResponseData:
        """Evaluate the identified response on an angular-frequency grid."""
        if self.is_parametric:
            if response not in {"process", "innovations"}:
                raise ValueError("response must be 'process' or 'innovations'")
            system = (
                self.innovations_model
                if response == "innovations"
                else self.deterministic_model
            )
            if system is None and response == "process" and self.ninputs == 0:
                system = self.innovations_model
            if system is None:
                raise NotImplementedError(
                    f"This identification result has no {response} response model"
                )
            if omega is None:
                if not np.isfinite(self.ts) or self.ts <= 0:
                    raise ValueError("A positive sample time is required")
                omega = np.linspace(0.0, np.pi / self.ts, 512)
            return control.frequency_response(system, omega)

        if response != "process":
            raise NotImplementedError(
                "A non-parametric frequency-domain result has no innovations model"
            )
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

    def innovations_frequency_response(
        self, omega: Optional[np.ndarray] = None
    ) -> control.FrequencyResponseData:
        """Evaluate the identified innovations-to-output response."""
        return self.frequency_response(omega, response="innovations")

    def get_frequency_response_uncertainty(
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

        if not self.supports("uncertainty"):
            raise NotImplementedError(
                "Empirical FRF uncertainty requires an identified input-output response"
            )
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
        """Compatibility alias for empirical frequency-response uncertainty."""
        return self.get_frequency_response_uncertainty(
            input_data,
            output_data,
            input_name,
            output_name,
            nperseg=nperseg,
            window=window,
            noverlap=noverlap,
            smoothing_bins=smoothing_bins,
            confidence_levels=confidence_levels,
        )

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

        if not self.supports("simulation"):
            if not self.is_parametric:
                raise NotImplementedError(
                    "Simulation is not defined for a non-parametric frequency response"
                )
            raise NotImplementedError(
                "Simulation requires a parametric input-to-output model"
            )
        system = self.deterministic_model
        if isinstance(system, control.TransferFunction):
            realized = control.tf2ss(system)
            return simulate_ss_system(
                realized.A, realized.B, realized.C, realized.D, u, x0
            )
        return simulate_ss_system(system.A, system.B, system.C, system.D, u, x0)

    def impulse_response(self, n_samples: int = 100) -> control.TimeResponseData:
        """Return the deterministic impulse response for every input channel."""
        if not self.supports("time_response"):
            raise NotImplementedError(
                "Impulse response requires a parametric input-to-output model"
            )
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        time = np.arange(n_samples, dtype=float) * self.ts
        return control.impulse_response(self.deterministic_model, time, squeeze=False)

    def step_response(self, n_samples: int = 100) -> control.TimeResponseData:
        """Return the deterministic step response for every input channel."""
        if not self.supports("time_response"):
            raise NotImplementedError(
                "Step response requires a parametric input-to-output model"
            )
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        system = self.deterministic_model
        realized = (
            control.tf2ss(system)
            if isinstance(system, control.TransferFunction)
            else system
        )
        time = np.arange(n_samples, dtype=float) * self.ts
        outputs = np.empty((self.noutputs, self.ninputs, n_samples), dtype=float)
        final_states = np.empty((realized.nstates, self.ninputs), dtype=float)
        for input_ in range(self.ninputs):
            inputs = np.zeros((self.ninputs, n_samples), dtype=float)
            inputs[input_, :] = 1.0
            response_data = control.forced_response(
                realized, T=time, U=inputs, squeeze=False
            )
            outputs[:, input_, :] = response_data.outputs
            final_states[:, input_] = response_data.states
        return control.TimeResponseData(time, outputs, final_states)

    def predict(
        self,
        u: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict outputs deterministically or in one-step-ahead form."""
        if u is None and y is None:
            if self.Yid is None:
                raise ValueError("Provide validation data; no fitted output is stored")
            return self.Yid.copy()
        if y is None:
            if u is None:
                raise ValueError("Input data is required for deterministic prediction")
            return self.simulate(np.atleast_2d(u), x0=x0)[1]
        outputs = np.atleast_2d(np.asarray(y, dtype=float))
        if u is None:
            inputs = np.empty((0, outputs.shape[1]))
        else:
            inputs = np.atleast_2d(np.asarray(u, dtype=float))
        if inputs.shape[1] != outputs.shape[1]:
            raise ValueError("Input and output sample counts differ")
        if self.K is not None:
            from ..utils.simulation_utils import ss_lsim_predictor_form

            _, prediction = ss_lsim_predictor_form(
                self.A_K, self.B_K, self.C, self.D, self.K, outputs, inputs, x0
            )
            return prediction
        if not self._has_invertible_innovations_model():
            raise NotImplementedError(
                "One-step prediction requires an identified Kalman gain or a "
                "causally invertible diagonal innovations model"
            )

        from scipy.signal import lfilter

        if self.deterministic_model is None:
            deterministic = np.zeros_like(outputs)
        else:
            if u is None:
                raise ValueError("Input data is required for this process model")
            deterministic = self.simulate(inputs, x0=x0)[1]
        unexplained = outputs - deterministic
        innovations = np.empty_like(unexplained)
        noise_model = self.innovations_model
        for output in range(self.noutputs):
            innovations[output] = lfilter(
                noise_model.den[output][output],
                noise_model.num[output][output],
                unexplained[output],
            )
        return outputs - innovations

    def residuals(
        self,
        y: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
        *,
        prediction: bool = False,
    ) -> np.ndarray:
        """Return stored identification residuals or residuals on new data."""
        if y is None:
            if self._residuals is None:
                raise ValueError("No identification residuals are stored")
            return self._residuals.copy()
        outputs = np.atleast_2d(np.asarray(y, dtype=float))
        estimate = self.predict(u=u, y=outputs if prediction else None)
        if estimate.shape != outputs.shape:
            raise ValueError("Predicted and measured output shapes differ")
        return outputs - estimate

    def fit(
        self,
        y: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
        *,
        prediction: bool = False,
    ) -> dict[str, Union[float, np.ndarray]]:
        """Return per-output normalized-RMSE fit and its aggregate score."""
        if y is None:
            if self._identification_output is None:
                raise ValueError("No identification output is stored")
            outputs = self._identification_output
            outputs = outputs[:, self.fit_start :]
            errors = self.residuals()[:, self.fit_start :]
        else:
            outputs = np.atleast_2d(np.asarray(y, dtype=float))
            errors = self.residuals(outputs, u, prediction=prediction)
        rmse = np.sqrt(np.mean(errors**2, axis=1))
        centered = outputs - np.mean(outputs, axis=1, keepdims=True)
        scale = np.sqrt(np.mean(centered**2, axis=1))
        nrmse = 1.0 - np.divide(
            rmse,
            scale,
            out=np.full_like(rmse, np.nan),
            where=scale > 0,
        )
        finite_nrmse = nrmse[np.isfinite(nrmse)]
        score = float(np.mean(finite_nrmse)) if finite_nrmse.size else np.nan
        return {"nrmse": nrmse, "score": score}

    def supports_optimization_methods(self) -> bool:
        """
        Check if the model supports various optimization methods.

        Returns:
        --------
        bool : True if optimization methods are supported
        """
        return self.supports("simulation")


IdentificationResult = StateSpaceModel


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
