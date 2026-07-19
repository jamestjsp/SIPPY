import numpy as np

from ._backend import (
    CtrlSysError,
    discrete_time_response,
    state_frequency_response,
    state_to_transfer,
    transfer_frequency_response,
    transfer_to_state,
)
from ._models import (
    FrequencyResponseData,
    InputOutputSystem,
    StateSpace,
    TimeResponseData,
    TransferFunction,
)

__all__ = [
    "CtrlSysError",
    "FrequencyResponseData",
    "InputOutputSystem",
    "StateSpace",
    "TimeResponseData",
    "TransferFunction",
    "frequency_response",
    "forced_response",
    "freqresp",
    "impulse_response",
    "lsim",
    "poles",
    "ss",
    "ss2tf",
    "tf",
    "tf2ss",
    "tfdata",
]


def tf(
    numerator: object,
    denominator: object,
    dt: float | bool | None = None,
) -> TransferFunction:
    return TransferFunction(numerator, denominator, dt=dt)


def ss(
    A: object,
    B: object | None = None,
    C: object | None = None,
    D: object | None = None,
    dt: float | bool | None = None,
) -> StateSpace:
    if isinstance(A, StateSpace) and B is None and C is None and D is None:
        return StateSpace(A.A, A.B, A.C, A.D, dt=A.dt if dt is None else dt)
    if isinstance(A, TransferFunction) and B is None and C is None and D is None:
        realized = tf2ss(A)
        if dt is not None:
            realized.dt = dt
        return realized
    if B is None or C is None or D is None:
        raise TypeError("ss expects a system or A, B, C, and D matrices")
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
        if system.dt is None or system.dt == 0:
            evaluation_points = 1j * frequencies
        else:
            sample_time = 1.0 if system.dt is True else float(system.dt)
            evaluation_points = np.exp(1j * frequencies * sample_time)
        response = transfer_frequency_response(system, evaluation_points)
    else:
        raise TypeError("frequency_response expects a SIPPY input/output system")
    return FrequencyResponseData(frequencies.copy(), response)


freqresp = frequency_response


def _response_inputs(system: StateSpace, U: object) -> np.ndarray:
    inputs = np.asarray(U, dtype=float)
    if inputs.ndim == 0:
        inputs = inputs.reshape(1, 1)
    elif inputs.ndim == 1:
        if system.ninputs != 1:
            raise ValueError("One-dimensional input is only valid for a SISO system")
        inputs = inputs.reshape(1, -1)
    elif inputs.ndim != 2:
        raise ValueError("Input data must be one- or two-dimensional")
    if inputs.shape[0] != system.ninputs:
        if inputs.shape[1] == system.ninputs:
            inputs = inputs.T
        else:
            raise ValueError("Input channel count does not match the system")
    return np.array(inputs, dtype=float, order="F", copy=True)


def forced_response(
    system: InputOutputSystem,
    T: object | None = None,
    U: object = 0.0,
    X0: object | None = None,
    *,
    squeeze: bool | None = None,
) -> TimeResponseData:
    realized = tf2ss(system) if isinstance(system, TransferFunction) else system
    if not isinstance(realized, StateSpace):
        raise TypeError("forced_response expects a SIPPY input/output system")
    if realized.dt is None or realized.dt == 0:
        raise ValueError("forced_response requires a discrete-time system")
    inputs = _response_inputs(realized, U)
    sample_time = 1.0 if realized.dt is True else float(realized.dt)
    if T is None:
        time = np.arange(inputs.shape[1], dtype=float) * sample_time
    else:
        time = np.asarray(T, dtype=float)
        if time.ndim != 1 or time.size != inputs.shape[1]:
            raise ValueError("Time and input arrays must have the same sample count")
    initial_state = (
        np.zeros(realized.nstates, dtype=float)
        if X0 is None
        else np.asarray(X0, dtype=float).reshape(-1)
    )
    if initial_state.size != realized.nstates:
        raise ValueError("Initial state dimension does not match the system")
    outputs, final_state = discrete_time_response(realized, inputs, initial_state)
    if squeeze is not False and realized.noutputs == 1:
        outputs = outputs[0]
    return TimeResponseData(time.copy(), outputs, final_state)


def impulse_response(
    system: InputOutputSystem,
    T: object,
    *,
    squeeze: bool | None = None,
) -> TimeResponseData:
    realized = tf2ss(system) if isinstance(system, TransferFunction) else system
    if not isinstance(realized, StateSpace):
        raise TypeError("impulse_response expects a SIPPY input/output system")
    time = np.asarray(T, dtype=float)
    if time.ndim != 1:
        raise ValueError("Time must be one-dimensional")
    sample_time = 1.0 if realized.dt is True else float(realized.dt)
    outputs = np.empty((realized.noutputs, realized.ninputs, time.size), dtype=float)
    final_states = np.empty((realized.nstates, realized.ninputs), dtype=float)
    for input_ in range(realized.ninputs):
        impulse = np.zeros((realized.ninputs, time.size), dtype=float, order="F")
        if time.size:
            impulse[input_, 0] = 1.0 / sample_time
        response = forced_response(realized, T=time, U=impulse, squeeze=False)
        outputs[:, input_, :] = response.outputs
        final_states[:, input_] = response.states
    if squeeze is not False:
        outputs = np.squeeze(outputs)
    return TimeResponseData(time.copy(), outputs, final_states)


def lsim(
    system: InputOutputSystem, U: object, T: object, X0: object | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    response = forced_response(system, T=T, U=U, X0=X0)
    return response.outputs, response.time, response.states


def poles(system: InputOutputSystem) -> np.ndarray:
    realized = tf2ss(system) if isinstance(system, TransferFunction) else system
    if not isinstance(realized, StateSpace):
        raise TypeError("poles expects a SIPPY input/output system")
    return np.linalg.eigvals(realized.A)


def tfdata(
    system: TransferFunction,
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    if not isinstance(system, TransferFunction):
        raise TypeError("tfdata expects a TransferFunction")
    return system.num, system.den
