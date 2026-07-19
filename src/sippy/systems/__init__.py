import numpy as np

from ._backend import (
    CtrlSysError,
    state_frequency_response,
    state_to_transfer,
    transfer_to_state,
)
from ._models import (
    FrequencyResponseData,
    InputOutputSystem,
    StateSpace,
    TransferFunction,
)

__all__ = [
    "CtrlSysError",
    "FrequencyResponseData",
    "InputOutputSystem",
    "StateSpace",
    "TransferFunction",
    "frequency_response",
    "ss",
    "ss2tf",
    "tf",
    "tf2ss",
]


def tf(
    numerator: object,
    denominator: object,
    *,
    dt: float | bool | None = None,
) -> TransferFunction:
    return TransferFunction(numerator, denominator, dt=dt)


def ss(
    A: object,
    B: object,
    C: object,
    D: object,
    *,
    dt: float | bool | None = None,
) -> StateSpace:
    return StateSpace(A, B, C, D, dt=dt)


def tf2ss(system: TransferFunction) -> StateSpace:
    if not isinstance(system, TransferFunction):
        raise TypeError("tf2ss expects a TransferFunction")
    A, B, C, D = transfer_to_state(system)
    return StateSpace(A, B, C, D, dt=system.dt)


def ss2tf(system: StateSpace) -> TransferFunction:
    if not isinstance(system, StateSpace):
        raise TypeError("ss2tf expects a StateSpace")
    numerator, denominator = state_to_transfer(system)
    return TransferFunction(numerator, denominator, dt=system.dt)


def frequency_response(
    system: InputOutputSystem, omega: object
) -> FrequencyResponseData:
    frequencies = np.atleast_1d(np.asarray(omega, dtype=float))
    if frequencies.ndim != 1:
        raise ValueError("Frequencies must be one-dimensional")
    if isinstance(system, StateSpace):
        response = state_frequency_response(system, frequencies)
    elif isinstance(system, TransferFunction):
        response = np.empty(
            (system.noutputs, system.ninputs, frequencies.size), dtype=complex
        )
        for index, frequency in enumerate(frequencies):
            if system.dt is None or system.dt == 0:
                evaluation_point = 1j * frequency
            else:
                sample_time = 1.0 if system.dt is True else float(system.dt)
                evaluation_point = np.exp(1j * frequency * sample_time)
            for output in range(system.noutputs):
                for input_ in range(system.ninputs):
                    response[output, input_, index] = np.polyval(
                        system.num[output][input_], evaluation_point
                    ) / np.polyval(system.den[output][input_], evaluation_point)
    else:
        raise TypeError("frequency_response expects a SIPPY input/output system")
    return FrequencyResponseData(frequencies.copy(), response)
