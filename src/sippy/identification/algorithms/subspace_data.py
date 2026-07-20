from dataclasses import dataclass

import numpy as np

from ...utils.simulation_utils import ordinate_sequence


@dataclass(frozen=True)
class SubspaceRankDiagnostics:
    input_rank: int
    input_rows: int
    past_rank: int
    past_rows: int
    reference_rank: int
    reference_rows: int

    @property
    def input_persistently_exciting(self) -> bool:
        return self.input_rows > 0 and self.input_rank == self.input_rows

    @property
    def reference_informative(self) -> bool:
        return self.reference_rows > 0 and self.reference_rank == self.reference_rows


@dataclass(frozen=True)
class SubspaceData:
    outputs: np.ndarray
    inputs: np.ndarray
    references: np.ndarray | None
    output_scale: np.ndarray
    input_scale: np.ndarray
    reference_scale: np.ndarray | None
    future_outputs: np.ndarray
    past_outputs: np.ndarray
    future_inputs: np.ndarray
    past_inputs: np.ndarray
    future_references: np.ndarray | None
    past_references: np.ndarray | None
    past_data: np.ndarray
    future_horizon: int
    past_offset: int
    past_block_rows: int
    sample_count: int
    usable_columns: int
    ranks: SubspaceRankDiagnostics


def _signal_matrix(value: object, name: str) -> np.ndarray:
    signal = np.asarray(value, dtype=float)
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    if signal.ndim != 2 or signal.shape[0] == 0 or signal.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty channel-by-sample matrix")
    if not np.all(np.isfinite(signal)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(signal, dtype=float, order="F", copy=True)


def _positive_horizon(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    horizon = int(value)
    if horizon < 1:
        raise ValueError(f"{name} must be a positive integer")
    return horizon


def _scale_channels(signal: np.ndarray, enabled: bool) -> tuple[np.ndarray, np.ndarray]:
    if not enabled:
        return np.ones(signal.shape[0]), signal.copy(order="F")
    scales = np.std(signal, axis=1)
    scales = np.where(scales < np.finfo(np.float64).eps, 1.0, scales)
    return scales, np.array(signal / scales[:, None], order="F", copy=True)


def _numerical_rank(matrix: np.ndarray) -> int:
    if matrix.size == 0:
        return 0
    gram = matrix @ matrix.T if matrix.shape[0] <= matrix.shape[1] else matrix.T @ matrix
    gram = 0.5 * (gram + gram.T)
    eigenvalues = np.linalg.eigvalsh(gram)
    largest_eigenvalue = max(float(eigenvalues[-1]), 0.0)
    uncertainty = (
        max(matrix.shape) * np.finfo(np.float64).eps * largest_eigenvalue
    )
    if largest_eigenvalue > 0.0 and float(eigenvalues[0]) > uncertainty:
        return min(matrix.shape)

    singular_values = np.linalg.svd(matrix, compute_uv=False, full_matrices=False)
    largest = float(singular_values[0])
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * largest
    return int(np.count_nonzero(singular_values > tolerance))


def prepare_subspace_data(
    y: object,
    u: object,
    *,
    future_horizon: int,
    past_offset: int,
    past_block_rows: int | None = None,
    reference: object | None = None,
    scale: bool = True,
    require_persistent_excitation: bool = False,
) -> SubspaceData:
    outputs = _signal_matrix(y, "outputs")
    inputs = _signal_matrix(u, "inputs")
    if outputs.shape[1] != inputs.shape[1]:
        raise ValueError("inputs and outputs must have the same sample count")

    references = None if reference is None else _signal_matrix(reference, "reference")
    if references is not None and references.shape[1] != outputs.shape[1]:
        raise ValueError(
            "reference must have the same sample count as inputs and outputs"
        )

    future = _positive_horizon(future_horizon, "future_horizon")
    past = _positive_horizon(past_offset, "past_offset")
    retained_past_rows = (
        future
        if past_block_rows is None
        else _positive_horizon(past_block_rows, "past_block_rows")
    )
    if past_block_rows is not None and retained_past_rows > past:
        raise ValueError("past_block_rows cannot exceed past_offset")
    sample_count = outputs.shape[1]
    usable_columns = sample_count - past - future + 1
    if usable_columns <= 0:
        minimum = past + future
        raise ValueError(
            f"Not enough data points. Need at least {minimum} points, got {sample_count}"
        )

    output_scale, scaled_outputs = _scale_channels(outputs, scale)
    input_scale, scaled_inputs = _scale_channels(inputs, scale)
    if references is None:
        reference_scale = None
        scaled_references = None
    else:
        reference_scale, scaled_references = _scale_channels(references, scale)

    future_outputs, legacy_past_outputs = ordinate_sequence(
        scaled_outputs, future, past
    )
    future_inputs, legacy_past_inputs = ordinate_sequence(scaled_inputs, future, past)
    if retained_past_rows == future:
        past_outputs = legacy_past_outputs
        past_inputs = legacy_past_inputs
    else:
        past_outputs = np.vstack(
            [
                scaled_outputs[:, offset : offset + usable_columns]
                for offset in range(retained_past_rows)
            ]
        )
        past_inputs = np.vstack(
            [
                scaled_inputs[:, offset : offset + usable_columns]
                for offset in range(retained_past_rows)
            ]
        )
    past_data = np.vstack((past_inputs, past_outputs))
    input_hankel = np.vstack((past_inputs, future_inputs))

    if scaled_references is None:
        future_references = None
        past_references = None
        reference_rank = 0
        reference_rows = 0
    else:
        future_references, legacy_past_references = ordinate_sequence(
            scaled_references, future, past
        )
        if retained_past_rows == future:
            past_references = legacy_past_references
        else:
            past_references = np.vstack(
                [
                    scaled_references[:, offset : offset + usable_columns]
                    for offset in range(retained_past_rows)
                ]
            )
        reference_hankel = np.vstack((past_references, future_references))
        reference_rank = _numerical_rank(reference_hankel)
        reference_rows = reference_hankel.shape[0]

    ranks = SubspaceRankDiagnostics(
        input_rank=_numerical_rank(input_hankel),
        input_rows=input_hankel.shape[0],
        past_rank=_numerical_rank(past_data),
        past_rows=past_data.shape[0],
        reference_rank=reference_rank,
        reference_rows=reference_rows,
    )
    if require_persistent_excitation and not ranks.input_persistently_exciting:
        raise ValueError(
            "input block Hankel matrix is not persistently exciting; "
            f"rank {ranks.input_rank}, need {ranks.input_rows}"
        )

    return SubspaceData(
        outputs=scaled_outputs,
        inputs=scaled_inputs,
        references=scaled_references,
        output_scale=output_scale,
        input_scale=input_scale,
        reference_scale=reference_scale,
        future_outputs=future_outputs,
        past_outputs=past_outputs,
        future_inputs=future_inputs,
        past_inputs=past_inputs,
        future_references=future_references,
        past_references=past_references,
        past_data=past_data,
        future_horizon=future,
        past_offset=past,
        past_block_rows=retained_past_rows,
        sample_count=sample_count,
        usable_columns=usable_columns,
        ranks=ranks,
    )
