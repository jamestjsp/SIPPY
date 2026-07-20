import warnings
from dataclasses import dataclass

import numpy as np

from .parsim_core import (
    _prepare_predictor_subspace,
    _realize_predictor_subspace,
)
from .subspace_core import (
    DimensionCandidate,
    DimensionSelection,
    _candidate_orders_from_singular_values,
    _default_horizon_candidates,
    _prepare_ort_subspace,
    _realize_ort_dimension_candidate,
    _select_dimension_candidate,
)
from .subspace_data import prepare_subspace_data
from .subspace_weighting import SubspaceWeightingDiagnostics


@dataclass(frozen=True)
class AutomaticDimensionEstimate:
    selection: DimensionSelection
    route: str
    horizon_candidates: tuple[int, ...]
    reference_projection_reason: str | None
    weighting: SubspaceWeightingDiagnostics


def _singular_gap(values, order):
    if order >= values.size:
        return np.inf
    denominator = max(float(values[order]), np.finfo(np.float64).tiny)
    return float(values[order - 1] / denominator)


def _predictor_candidate(prepared, order):
    realization = _realize_predictor_subspace(
        prepared,
        threshold=0.0,
        max_order=order,
    )
    _, C, _, D, K, A, B, x0, _ = realization
    _, effective_rank = _candidate_orders_from_singular_values(
        prepared.singular_values,
        prepared.future_horizon,
    )
    parameter_count = order * (prepared.input_count + 2 * prepared.output_count)
    parameter_count += prepared.output_count * (prepared.output_count + 1) // 2
    if prepared.direct_feedthrough:
        parameter_count += prepared.output_count * prepared.input_count
    return DimensionCandidate(
        horizon=prepared.future_horizon,
        order=order,
        singular_values=prepared.singular_values.copy(),
        effective_rank=effective_rank,
        singular_gap=_singular_gap(prepared.singular_values, order),
        A=A,
        B=B,
        C=C,
        D=D,
        K=K,
        parameter_count=parameter_count,
        initial_state=x0.reshape(-1),
    )


def _build_ort_candidates(
    y,
    u,
    reference,
    horizons,
    *,
    explicit_order,
    weighting,
):
    candidates = []
    weighting_by_dimension = {}
    reasons = []
    for horizon in horizons:
        data = prepare_subspace_data(
            y,
            u,
            future_horizon=horizon,
            past_offset=horizon,
            reference=reference,
        )
        prepared, diagnostics = _prepare_ort_subspace(
            data,
            weights=weighting,
            warn_on_fallback=False,
        )
        if prepared is None:
            reasons.append(diagnostics.reason)
            continue
        orders, _ = _candidate_orders_from_singular_values(
            prepared.singular_values,
            horizon,
            explicit_order=explicit_order,
        )
        for order in orders:
            candidate = _realize_ort_dimension_candidate(prepared, order)
            candidates.append(candidate)
            weighting_by_dimension[(horizon, order)] = prepared.ort.weighting
    return candidates, weighting_by_dimension, reasons


def _build_predictor_candidates(
    y,
    u,
    horizons,
    *,
    explicit_order,
    weighting,
    direct_feedthrough,
):
    candidates = []
    weighting_by_dimension = {}
    for horizon in horizons:
        prepared = _prepare_predictor_subspace(
            y,
            u,
            future_horizon=horizon,
            past_horizon=horizon,
            direct_feedthrough=direct_feedthrough,
            strict_identifiability=True,
            weighting=weighting,
        )
        orders, _ = _candidate_orders_from_singular_values(
            prepared.singular_values,
            horizon,
            explicit_order=explicit_order,
        )
        for order in orders:
            try:
                candidate = _predictor_candidate(prepared, order)
            except (ValueError, np.linalg.LinAlgError, OverflowError):
                continue
            if not all(
                np.all(np.isfinite(matrix))
                for matrix in (
                    candidate.A,
                    candidate.B,
                    candidate.C,
                    candidate.D,
                    candidate.K,
                )
            ):
                continue
            candidates.append(candidate)
            weighting_by_dimension[(horizon, order)] = prepared.weighting_diagnostics
    return candidates, weighting_by_dimension


def select_automatic_dimensions(
    y,
    u,
    *,
    reference=None,
    explicit_horizon=None,
    explicit_order=None,
    weighting="CVA",
    criterion="BIC",
    validation_fraction=0.2,
    direct_feedthrough=False,
):
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    outputs = np.atleast_2d(np.asarray(y, dtype=float))
    inputs = np.atleast_2d(np.asarray(u, dtype=float))
    references = (
        None if reference is None else np.atleast_2d(np.asarray(reference, dtype=float))
    )
    if outputs.shape[1] != inputs.shape[1] or (
        references is not None and references.shape[1] != outputs.shape[1]
    ):
        raise ValueError("all identification signals must have equal sample counts")

    validation_count = max(1, int(np.ceil(validation_fraction * outputs.shape[1])))
    training_count = outputs.shape[1] - validation_count
    training_y = outputs[:, :training_count]
    training_u = inputs[:, :training_count]
    training_reference = None if references is None else references[:, :training_count]
    horizons = _default_horizon_candidates(
        training_count,
        inputs.shape[0],
        outputs.shape[0],
        reference_count=0 if references is None else references.shape[0],
        explicit_horizon=explicit_horizon,
    )

    route = "predictor"
    projection_reason = "reference_missing"
    candidates = []
    weighting_by_dimension = {}
    if training_reference is not None:
        candidates, weighting_by_dimension, reasons = _build_ort_candidates(
            training_y,
            training_u,
            training_reference,
            horizons,
            explicit_order=explicit_order,
            weighting=weighting,
        )
        if candidates:
            route = "two-stage-ort"
            projection_reason = None
        else:
            projection_reason = reasons[0] if reasons else "reference_unusable"
            warnings.warn(
                "Measured exogenous reference is unusable for two-stage ORT "
                f"({projection_reason}); using the predictor-form estimator",
                UserWarning,
                stacklevel=2,
            )

    if route == "predictor":
        candidates, weighting_by_dimension = _build_predictor_candidates(
            training_y,
            training_u,
            horizons,
            explicit_order=explicit_order,
            weighting=weighting,
            direct_feedthrough=direct_feedthrough,
        )
    selection = _select_dimension_candidate(
        candidates,
        outputs,
        inputs,
        method=criterion,
        validation_fraction=validation_fraction,
    )
    key = (selection.candidate.horizon, selection.candidate.order)
    return AutomaticDimensionEstimate(
        selection=selection,
        route=route,
        horizon_candidates=horizons,
        reference_projection_reason=projection_reason,
        weighting=weighting_by_dimension[key],
    )
