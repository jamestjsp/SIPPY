"""Closed-loop-consistent Subspace ARX identification."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..base import IdentificationAlgorithm, StateSpaceModel, resolve_identification_data
from .subspace_data import prepare_subspace_data

if TYPE_CHECKING:
    from ..iddata import IDData


@dataclass(frozen=True)
class VARXPredictorEstimate:
    direct: np.ndarray
    input_blocks: np.ndarray
    output_blocks: np.ndarray
    residuals: np.ndarray
    regressor_rank: int
    regressor_rows: int


@dataclass(frozen=True)
class SSARXEstimate:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    K: np.ndarray
    initial_state: np.ndarray
    variance: float
    singular_values: np.ndarray
    selected_order: int
    future_horizon: int
    varx_order: int
    varx_rank: int
    varx_rows: int
    past_rank: int
    past_rows: int
    future_residual_rank: int
    future_residual_rows: int


def _right_least_squares(
    target: np.ndarray,
    regressor: np.ndarray,
    *,
    name: str,
    require_full_row_rank: bool = True,
) -> tuple[np.ndarray, int]:
    coefficients, _, rank, _ = np.linalg.lstsq(
        regressor.T,
        target.T,
        rcond=None,
    )
    row_rank = int(rank)
    if require_full_row_rank and row_rank < regressor.shape[0]:
        raise ValueError(
            f"{name} is rank deficient: rank {row_rank}, need {regressor.shape[0]}"
        )
    return coefficients.T, row_rank


def _estimate_varx_predictor(
    y: object,
    u: object,
    *,
    order: int,
    direct_feedthrough: bool,
) -> VARXPredictorEstimate:
    outputs = np.atleast_2d(np.asarray(y, dtype=float))
    inputs = np.atleast_2d(np.asarray(u, dtype=float))
    if outputs.ndim != 2 or inputs.ndim != 2:
        raise ValueError("VARX inputs and outputs must be matrices")
    if outputs.shape[1] != inputs.shape[1]:
        raise ValueError("VARX inputs and outputs must have equal sample counts")
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("VARX order must be a positive integer")
    if outputs.shape[1] <= order:
        raise ValueError(f"VARX order {order} requires more than {order} samples")

    output_count = outputs.shape[0]
    input_count = inputs.shape[0]
    target = outputs[:, order:]
    regressors = []
    if direct_feedthrough:
        regressors.append(inputs[:, order:])
    regressors.extend(
        inputs[:, order - lag : inputs.shape[1] - lag] for lag in range(1, order + 1)
    )
    regressors.extend(
        outputs[:, order - lag : outputs.shape[1] - lag] for lag in range(1, order + 1)
    )
    regressor = np.vstack(regressors)
    coefficients, rank = _right_least_squares(
        target,
        regressor,
        name="VARX regressor",
    )

    offset = 0
    if direct_feedthrough:
        direct = coefficients[:, :input_count].copy()
        offset = input_count
    else:
        direct = np.zeros((output_count, input_count))
    input_width = order * input_count
    input_blocks = coefficients[:, offset : offset + input_width].reshape(
        output_count,
        order,
        input_count,
    )
    input_blocks = np.transpose(input_blocks, (1, 0, 2)).copy()
    offset += input_width
    output_blocks = coefficients[:, offset:].reshape(
        output_count,
        order,
        output_count,
    )
    output_blocks = np.transpose(output_blocks, (1, 0, 2)).copy()
    residuals = target - coefficients @ regressor
    return VARXPredictorEstimate(
        direct=direct,
        input_blocks=input_blocks,
        output_blocks=output_blocks,
        residuals=residuals,
        regressor_rank=rank,
        regressor_rows=regressor.shape[0],
    )


def _predictor_toeplitz(
    estimate: VARXPredictorEstimate,
    *,
    future_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    output_count, input_count = estimate.direct.shape
    if estimate.input_blocks.shape[0] < future_horizon - 1:
        raise ValueError("VARX order must be at least future_horizon - 1 for SSARX")
    input_toeplitz = np.zeros(
        (future_horizon * output_count, future_horizon * input_count)
    )
    output_toeplitz = np.zeros(
        (future_horizon * output_count, future_horizon * output_count)
    )
    for row in range(future_horizon):
        output_slice = slice(row * output_count, (row + 1) * output_count)
        input_slice = slice(row * input_count, (row + 1) * input_count)
        input_toeplitz[output_slice, input_slice] = estimate.direct
        for column in range(row):
            lag = row - column - 1
            input_column = slice(
                column * input_count,
                (column + 1) * input_count,
            )
            output_column = slice(
                column * output_count,
                (column + 1) * output_count,
            )
            input_toeplitz[output_slice, input_column] = estimate.input_blocks[lag]
            output_toeplitz[output_slice, output_column] = estimate.output_blocks[lag]
    return input_toeplitz, output_toeplitz


def _inverse_covariance_square_root(
    signal: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, int]:
    covariance = signal @ signal.T / signal.shape[1]
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest = max(float(eigenvalues[-1]), 0.0)
    tolerance = max(covariance.shape) * np.finfo(float).eps * largest
    retained = eigenvalues > tolerance
    rank = int(np.count_nonzero(retained))
    if rank < signal.shape[0]:
        raise ValueError(
            f"{name} covariance is rank deficient: rank {rank}, need {signal.shape[0]}"
        )
    inverse = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
    return inverse, rank


def _selected_order(
    singular_values: np.ndarray,
    *,
    threshold: float,
    fixed_order: int | None,
    future_horizon: int,
) -> int:
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        raise ValueError("SSARX canonical correlation has no nonzero singular values")
    if fixed_order is not None:
        order = fixed_order
    else:
        order = int(np.count_nonzero(singular_values >= threshold * singular_values[0]))
    order = min(order, singular_values.size)
    if order < 1:
        raise ValueError("SSARX order selection produced an empty state space")
    if order >= future_horizon:
        raise ValueError(
            f"SSARX future horizon {future_horizon} must exceed selected order {order}"
        )
    return order


def identify_ssarx(
    y: object,
    u: object,
    *,
    future_horizon: int,
    varx_order: int,
    threshold: float,
    fixed_order: int | None,
    direct_feedthrough: bool,
) -> SSARXEstimate:
    data = prepare_subspace_data(
        y,
        u,
        future_horizon=future_horizon,
        past_offset=varx_order,
        past_block_rows=varx_order,
    )
    predictor = _estimate_varx_predictor(
        data.outputs,
        data.inputs,
        order=varx_order,
        direct_feedthrough=direct_feedthrough,
    )
    input_toeplitz, output_toeplitz = _predictor_toeplitz(
        predictor,
        future_horizon=future_horizon,
    )
    future_residuals = (
        data.future_outputs
        - input_toeplitz @ data.future_inputs
        - output_toeplitz @ data.future_outputs
    )
    past = data.past_data
    future_inverse, future_rank = _inverse_covariance_square_root(
        future_residuals,
        name="SSARX future residual",
    )
    past_inverse, past_rank = _inverse_covariance_square_root(
        past,
        name="SSARX past regressor",
    )
    cross_covariance = future_residuals @ past.T / past.shape[1]
    canonical_matrix = future_inverse @ cross_covariance @ past_inverse
    _, singular_values, right_vectors = np.linalg.svd(
        canonical_matrix,
        full_matrices=False,
    )
    order = _selected_order(
        singular_values,
        threshold=threshold,
        fixed_order=fixed_order,
        future_horizon=future_horizon,
    )
    state_map = right_vectors[:order] @ past_inverse
    states = state_map @ past

    output_count = data.outputs.shape[0]
    input_count = data.inputs.shape[0]
    current_outputs = data.future_outputs[:output_count]
    current_inputs = data.future_inputs[:input_count]
    output_regressor = (
        np.vstack((states, current_inputs)) if direct_feedthrough else states
    )
    output_parameters, _ = _right_least_squares(
        current_outputs,
        output_regressor,
        name="SSARX output regressor",
    )
    C = output_parameters[:, :order]
    if direct_feedthrough:
        D = output_parameters[:, order:].copy()
    else:
        D = np.zeros((output_count, input_count))
    innovations = current_outputs - C @ states - D @ current_inputs

    state_regressor = np.vstack(
        (states[:, :-1], current_inputs[:, :-1], innovations[:, :-1])
    )
    state_parameters, _ = _right_least_squares(
        states[:, 1:],
        state_regressor,
        name="SSARX state regressor",
    )
    A = state_parameters[:, :order]
    B = state_parameters[:, order : order + input_count]
    K = state_parameters[:, order + input_count :]

    for channel in range(input_count):
        B[:, channel] /= data.input_scale[channel]
        D[:, channel] /= data.input_scale[channel]
    original_innovations = innovations.copy()
    for channel in range(output_count):
        C[channel] *= data.output_scale[channel]
        D[channel] *= data.output_scale[channel]
        K[:, channel] /= data.output_scale[channel]
        original_innovations[channel] *= data.output_scale[channel]

    variance = float(np.mean(original_innovations**2))
    return SSARXEstimate(
        A=A,
        B=B,
        C=C,
        D=D,
        K=K,
        initial_state=np.zeros(order),
        variance=variance,
        singular_values=singular_values,
        selected_order=order,
        future_horizon=future_horizon,
        varx_order=varx_order,
        varx_rank=predictor.regressor_rank,
        varx_rows=predictor.regressor_rows,
        past_rank=past_rank,
        past_rows=past.shape[0],
        future_residual_rank=future_rank,
        future_residual_rows=future_residuals.shape[0],
    )


class SSARXAlgorithm(IdentificationAlgorithm):
    """Jansson's high-order ARX and CCA closed-loop subspace estimator."""

    covariance_source = "ssarx_innovations"
    kalman_gain_source = "ssarx_state_regression"

    def get_algorithm_name(self) -> str:
        return "SSARX"

    def identify(
        self,
        y: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
        iddata: Optional["IDData"] = None,
        **kwargs,
    ) -> StateSpaceModel:
        y, u, sample_time = resolve_identification_data(
            y,
            u,
            iddata,
            tsample=kwargs.get("tsample", 1.0),
        )
        self.validate_parameters(**kwargs)
        future_horizon = kwargs.get("ss_f", 10)
        varx_order = kwargs.get("ss_p", max(2 * future_horizon, future_horizon - 1))
        fixed_order = kwargs.get("ss_fixed_order")
        threshold = kwargs.get("ss_threshold", 0.1)
        direct_feedthrough = kwargs.get("ss_d_required", False)
        try:
            estimate = identify_ssarx(
                y,
                u,
                future_horizon=future_horizon,
                varx_order=varx_order,
                threshold=threshold,
                fixed_order=fixed_order,
                direct_feedthrough=direct_feedthrough,
            )
        except Exception as exc:
            raise RuntimeError(f"SSARX identification failed: {exc}") from exc

        identification_info = {
            "estimator_route": "ssarx",
            "selected_order": estimate.selected_order,
            "future_horizon": estimate.future_horizon,
            "varx_order": estimate.varx_order,
            "singular_values": estimate.singular_values.copy(),
            "numerical_ranks": {
                "varx_regressor": estimate.varx_rank,
                "varx_regressor_rows": estimate.varx_rows,
                "past_covariance": estimate.past_rank,
                "past_covariance_rows": estimate.past_rows,
                "future_residual_covariance": estimate.future_residual_rank,
                "future_residual_covariance_rows": estimate.future_residual_rows,
            },
            "fit_start": estimate.varx_order,
            "direct_feedthrough": direct_feedthrough,
        }
        return StateSpaceModel(
            estimate.A,
            estimate.B,
            estimate.C,
            estimate.D,
            estimate.K,
            None,
            None,
            None,
            sample_time,
            estimate.variance,
            identification_info=identification_info,
            x0=estimate.initial_state.reshape(-1, 1),
        )

    def validate_parameters(self, **kwargs) -> bool:
        future_horizon = kwargs.get("ss_f", 10)
        varx_order = kwargs.get("ss_p", max(2 * future_horizon, future_horizon - 1))
        fixed_order = kwargs.get("ss_fixed_order")
        threshold = kwargs.get("ss_threshold", 0.1)
        for name, value in (("ss_f", future_horizon), ("ss_p", varx_order)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if varx_order < future_horizon - 1:
            raise ValueError("ss_p must be at least ss_f - 1")
        if fixed_order is not None:
            if (
                isinstance(fixed_order, bool)
                or not isinstance(fixed_order, (int, np.integer))
                or fixed_order < 1
            ):
                raise ValueError("ss_fixed_order must be a positive integer or None")
            if future_horizon <= fixed_order:
                raise ValueError("ss_f must exceed ss_fixed_order")
        if not isinstance(threshold, (int, float, np.number)) or not 0 <= threshold < 1:
            raise ValueError("ss_threshold must be in [0, 1)")
        if not isinstance(kwargs.get("ss_d_required", False), (bool, np.bool_)):
            raise ValueError("ss_d_required must be boolean")
        return True
