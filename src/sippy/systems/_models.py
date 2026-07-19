from dataclasses import dataclass

import numpy as np


def _normalize_sample_time(dt: float | bool | None) -> float | bool | None:
    if dt is None or dt is True:
        return dt
    sample_time = float(dt)
    if sample_time < 0:
        raise ValueError("Sample time must be nonnegative")
    return sample_time


def _coefficient_matrix(value: object, name: str) -> list[list[np.ndarray]]:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        array = None

    if array is not None:
        if array.ndim == 0:
            return [[np.array([float(array)])]]
        if array.ndim == 1:
            return [[array.copy()]]
        if array.ndim == 2:
            return [
                [np.array([array[output, input_]]) for input_ in range(array.shape[1])]
                for output in range(array.shape[0])
            ]
        if array.ndim == 3:
            return [
                [array[output, input_, :].copy() for input_ in range(array.shape[1])]
                for output in range(array.shape[0])
            ]

    try:
        rows = list(value)
        matrix = [
            [np.atleast_1d(np.asarray(cell, dtype=float)).copy() for cell in row]
            for row in rows
        ]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a polynomial or polynomial matrix") from error

    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError(f"{name} must be a nonempty rectangular polynomial matrix")
    return matrix


def _trim_leading_zeros(coefficients: np.ndarray) -> np.ndarray:
    nonzero = np.flatnonzero(coefficients)
    if nonzero.size == 0:
        return np.array([0.0])
    return coefficients[nonzero[0] :]


class InputOutputSystem:
    dt: float | bool | None
    ninputs: int
    noutputs: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.noutputs, self.ninputs


class TransferFunction(InputOutputSystem):
    def __init__(
        self,
        numerator: object,
        denominator: object,
        *,
        dt: float | bool | None = None,
    ):
        numerators = _coefficient_matrix(numerator, "Numerator")
        denominators = _coefficient_matrix(denominator, "Denominator")
        numerator_shape = (len(numerators), len(numerators[0]))
        denominator_shape = (len(denominators), len(denominators[0]))
        if numerator_shape != denominator_shape:
            raise ValueError(
                "Numerator and denominator matrices must have the same shape"
            )

        self.num: list[list[np.ndarray]] = []
        self.den: list[list[np.ndarray]] = []
        for numerator_row, denominator_row in zip(numerators, denominators):
            normalized_numerators = []
            normalized_denominators = []
            for numerator_cell, denominator_cell in zip(numerator_row, denominator_row):
                denominator_cell = _trim_leading_zeros(denominator_cell)
                if not np.any(denominator_cell):
                    raise ValueError("Transfer-function denominator cannot be zero")
                scale = denominator_cell[0]
                numerator_cell = _trim_leading_zeros(numerator_cell / scale)
                denominator_cell = denominator_cell / scale
                if not np.any(numerator_cell):
                    denominator_cell = np.array([1.0])
                normalized_numerators.append(numerator_cell)
                normalized_denominators.append(denominator_cell)
            self.num.append(normalized_numerators)
            self.den.append(normalized_denominators)

        self.noutputs, self.ninputs = numerator_shape
        self.dt = _normalize_sample_time(dt)

    def __getitem__(self, key: tuple[int, int]) -> "TransferFunction":
        output, input_ = key
        return TransferFunction(
            self.num[output][input_], self.den[output][input_], dt=self.dt
        )

    def __mul__(self, other: object) -> "TransferFunction":
        if np.isscalar(other):
            numerator = [[cell * float(other) for cell in row] for row in self.num]
            return TransferFunction(numerator, self.den, dt=self.dt)
        if isinstance(other, TransferFunction):
            if self.shape != (1, 1) or other.shape != (1, 1):
                raise NotImplementedError(
                    "Transfer-function multiplication currently supports SISO systems"
                )
            if self.dt != other.dt:
                raise ValueError("Transfer functions must have the same sample time")
            return TransferFunction(
                np.polymul(self.num[0][0], other.num[0][0]),
                np.polymul(self.den[0][0], other.den[0][0]),
                dt=self.dt,
            )
        return NotImplemented

    __rmul__ = __mul__


class StateSpace(InputOutputSystem):
    def __init__(
        self,
        A: object,
        B: object,
        C: object,
        D: object,
        *,
        dt: float | bool | None = None,
    ):
        self.A = np.array(A, dtype=float, order="F", copy=True)
        self.B = np.array(B, dtype=float, order="F", copy=True)
        self.C = np.array(C, dtype=float, order="F", copy=True)
        self.D = np.array(D, dtype=float, order="F", copy=True)
        if any(matrix.ndim != 2 for matrix in (self.A, self.B, self.C, self.D)):
            raise ValueError("State-space matrices must be two-dimensional")
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be square")
        if self.B.shape[0] != self.A.shape[0]:
            raise ValueError("B row count must match the state dimension")
        if self.C.shape[1] != self.A.shape[0]:
            raise ValueError("C column count must match the state dimension")
        if self.D.shape != (self.C.shape[0], self.B.shape[1]):
            raise ValueError("D shape must match the output and input dimensions")

        self.nstates = self.A.shape[0]
        self.ninputs = self.B.shape[1]
        self.noutputs = self.C.shape[0]
        self.dt = _normalize_sample_time(dt)


@dataclass(frozen=True)
class FrequencyResponseData:
    omega: np.ndarray
    frdata: np.ndarray

    def __iter__(self):
        yield np.abs(self.frdata)
        yield np.angle(self.frdata)
        yield self.omega


@dataclass(frozen=True)
class TimeResponseData:
    time: np.ndarray
    outputs: np.ndarray
    states: np.ndarray
