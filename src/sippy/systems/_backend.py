from typing import TYPE_CHECKING

import ctrlsys
import numpy as np

if TYPE_CHECKING:
    from ._models import StateSpace, TransferFunction


class CtrlSysError(RuntimeError):
    def __init__(self, routine: str, info: int):
        self.routine = routine
        self.info = int(info)
        super().__init__(f"{routine} failed with info={info}")


def _check_info(routine: str, info: int) -> None:
    if info != 0:
        raise CtrlSysError(routine, info)


def _fortran_copy(matrix: np.ndarray) -> np.ndarray:
    return np.array(matrix, dtype=float, order="F", copy=True)


def _siso_transfer_to_state(
    numerator: np.ndarray, denominator: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    degree = len(denominator) - 1
    if len(numerator) - 1 > degree:
        raise ValueError("Improper transfer functions cannot be realized")
    if degree == 0:
        return (
            np.empty((0, 0), dtype=float, order="F"),
            np.empty((0, 1), dtype=float, order="F"),
            np.empty((1, 0), dtype=float, order="F"),
            np.array([[numerator[0] / denominator[0]]], order="F"),
        )

    coefficient_count = degree + 1
    pcoeff = np.zeros((1, 1, coefficient_count), dtype=float, order="F")
    qcoeff = np.zeros((1, 1, coefficient_count), dtype=float, order="F")
    pcoeff[0, 0, :] = denominator
    qcoeff[0, 0, coefficient_count - len(numerator) :] = numerator
    index = np.array([degree], dtype=np.int32)
    state_count, _, A, B, C, D, info = ctrlsys.tc04ad("L", 1, 1, index, pcoeff, qcoeff)
    _check_info("tc04ad", info)
    if state_count != A.shape[0]:
        raise RuntimeError("tc04ad returned an inconsistent state dimension")
    return A, B, C, D


def transfer_to_state(
    system: "TransferFunction",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    blocks: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []
    D = np.zeros((system.noutputs, system.ninputs), dtype=float, order="F")
    state_count = 0
    for output in range(system.noutputs):
        for input_ in range(system.ninputs):
            numerator = system.num[output][input_]
            denominator = system.den[output][input_]
            if not np.any(numerator):
                continue
            A_channel, B_channel, C_channel, D_channel = _siso_transfer_to_state(
                numerator, denominator
            )
            D[output, input_] = D_channel[0, 0]
            if A_channel.shape[0] > 0:
                blocks.append((output, input_, A_channel, B_channel, C_channel))
                state_count += A_channel.shape[0]

    A = np.zeros((state_count, state_count), dtype=float, order="F")
    B = np.zeros((state_count, system.ninputs), dtype=float, order="F")
    C = np.zeros((system.noutputs, state_count), dtype=float, order="F")
    offset = 0
    for output, input_, A_channel, B_channel, C_channel in blocks:
        next_offset = offset + A_channel.shape[0]
        A[offset:next_offset, offset:next_offset] = A_channel
        B[offset:next_offset, input_] = B_channel[:, 0]
        C[output, offset:next_offset] = C_channel[0, :]
        offset = next_offset
    return A, B, C, D


def state_to_transfer(
    system: "StateSpace",
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    if system.nstates == 0:
        numerator = [
            [np.array([system.D[output, input_]]) for input_ in range(system.ninputs)]
            for output in range(system.noutputs)
        ]
        denominator = [
            [np.array([1.0]) for _ in range(system.ninputs)]
            for _ in range(system.noutputs)
        ]
        return numerator, denominator

    _, _, _, _, _, index, dcoeff, ucoeff, info = ctrlsys.tb04ad(
        "R",
        _fortran_copy(system.A),
        _fortran_copy(system.B),
        _fortran_copy(system.C),
        _fortran_copy(system.D),
        0.0,
        0.0,
    )
    _check_info("tb04ad", info)
    numerator = []
    denominator = []
    for output in range(system.noutputs):
        coefficient_count = int(index[output]) + 1
        numerator.append(
            [
                np.array(ucoeff[output, input_, :coefficient_count], copy=True)
                for input_ in range(system.ninputs)
            ]
        )
        denominator.append(
            [
                np.array(dcoeff[output, :coefficient_count], copy=True)
                for _ in range(system.ninputs)
            ]
        )
    return numerator, denominator


def state_frequency_response(system: "StateSpace", omega: np.ndarray) -> np.ndarray:
    response = np.empty((system.noutputs, system.ninputs, omega.size), dtype=complex)
    for index, frequency in enumerate(omega):
        if system.dt is None or system.dt == 0:
            evaluation_point = 1j * frequency
        else:
            sample_time = 1.0 if system.dt is True else float(system.dt)
            evaluation_point = np.exp(1j * frequency * sample_time)
        if system.nstates == 0:
            response[:, :, index] = system.D
            continue
        dynamic, _, _, _, _, info = ctrlsys.tb05ad(
            "N",
            "G",
            _fortran_copy(system.A),
            _fortran_copy(system.B),
            _fortran_copy(system.C),
            evaluation_point,
        )
        _check_info("tb05ad", info)
        response[:, :, index] = dynamic + system.D
    return response


def transfer_frequency_response(
    system: "TransferFunction", evaluation_points: np.ndarray
) -> np.ndarray:
    if system.noutputs == 1 and system.ninputs == 1:
        return (
            np.polyval(system.num[0][0], evaluation_points)
            / np.polyval(system.den[0][0], evaluation_points)
        ).reshape(1, 1, -1)

    numerator_count = max(
        len(coefficients) for row in system.num for coefficients in row
    )
    denominator_count = max(
        len(coefficients) for row in system.den for coefficients in row
    )
    numerators = np.zeros(
        (numerator_count, system.noutputs, system.ninputs), dtype=float
    )
    denominators = np.zeros(
        (denominator_count, system.noutputs, system.ninputs), dtype=float
    )
    for output in range(system.noutputs):
        for input_ in range(system.ninputs):
            numerator = system.num[output][input_]
            denominator = system.den[output][input_]
            numerators[-len(numerator) :, output, input_] = numerator
            denominators[-len(denominator) :, output, input_] = denominator

    numerator_values = np.polynomial.polynomial.polyval(
        evaluation_points, numerators[::-1]
    )
    denominator_values = np.polynomial.polynomial.polyval(
        evaluation_points, denominators[::-1]
    )
    return numerator_values / denominator_values


def discrete_time_response(
    system: "StateSpace", inputs: np.ndarray, initial_state: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    outputs, final_state, info = ctrlsys.tf01md(
        _fortran_copy(system.A),
        _fortran_copy(system.B),
        _fortran_copy(system.C),
        _fortran_copy(system.D),
        _fortran_copy(inputs),
        np.array(initial_state, dtype=float, copy=True),
    )
    _check_info("tf01md", info)
    return outputs, final_state
