from dataclasses import asdict
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..base import IdentificationAlgorithm, StateSpaceModel, resolve_identification_data
from .automatic_subspace import select_automatic_dimensions
from .subspace_core import _causal_prediction_errors

if TYPE_CHECKING:
    from ..iddata import IDData


class SUBSPACEAlgorithm(IdentificationAlgorithm):
    """Canonical subspace estimator for open- and closed-loop records."""

    kalman_gain_source = "consistent_subspace_prediction"

    def get_algorithm_name(self) -> str:
        return "SUBSPACE"

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
        reference = kwargs.get("reference")
        estimate = select_automatic_dimensions(
            y,
            u,
            reference=reference,
            explicit_horizon=kwargs.get("ss_f"),
            explicit_order=kwargs.get("ss_fixed_order"),
            weighting=kwargs.get("ss_weighting", "CVA"),
            criterion=kwargs.get("criterion", "BIC"),
            validation_fraction=kwargs.get("validation_fraction", 0.2),
            direct_feedthrough=kwargs.get("ss_d_required", False),
        )
        candidate = estimate.selection.candidate
        fit_start = 2 * candidate.horizon
        errors = _causal_prediction_errors(
            candidate.A,
            candidate.B,
            candidate.C,
            candidate.D,
            candidate.K,
            y,
            u,
            candidate.initial_state,
        )
        variance = float(np.mean(errors[:, fit_start:] ** 2))
        reference_status = "not-provided"
        if estimate.route == "two-stage-ort":
            reference_status = "used"
        elif reference is not None:
            reference_status = "fallback"

        ranks = {
            "effective_subspace_rank": candidate.effective_rank,
            "conditional_covariance_rank": estimate.weighting.covariance_rank,
            "conditional_covariance_rows": estimate.weighting.covariance_rows,
        }
        if estimate.reference_diagnostics is not None:
            diagnostics = estimate.reference_diagnostics
            ranks.update(
                {
                    "reference_rank": diagnostics.reference_rank,
                    "reference_rows": diagnostics.reference_rows,
                    "projected_input_rank": diagnostics.projected_input_rank,
                    "projected_input_rows": diagnostics.projected_input_rows,
                    "deterministic_regressor_rank": (
                        diagnostics.deterministic_regressor_rank
                    ),
                    "deterministic_regressor_rows": (
                        diagnostics.deterministic_regressor_rows
                    ),
                }
            )

        identification_info = {
            "estimator_route": estimate.route,
            "reference_projection": {
                "status": reference_status,
                "reason": estimate.reference_projection_reason,
            },
            "selected_horizon": candidate.horizon,
            "selected_order": candidate.order,
            "horizon_candidates": estimate.horizon_candidates,
            "order_scores": estimate.selection.scores,
            "order_criterion": estimate.selection.criterion,
            "singular_values": candidate.singular_values.copy(),
            "singular_gap": candidate.singular_gap,
            "numerical_ranks": ranks,
            "weighting": asdict(estimate.weighting),
            "fit_start": fit_start,
        }
        return StateSpaceModel(
            candidate.A,
            candidate.B,
            candidate.C,
            candidate.D,
            candidate.K,
            None,
            None,
            None,
            sample_time,
            variance,
            identification_info=identification_info,
            x0=candidate.initial_state.reshape(-1, 1),
        )

    def validate_parameters(self, **kwargs) -> bool:
        for name in ("ss_f", "ss_fixed_order"):
            value = kwargs.get(name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, np.integer))
            ):
                raise ValueError(f"{name} must be a positive integer or None")
            if value is not None and value < 1:
                raise ValueError(f"{name} must be a positive integer or None")
        if kwargs.get("ss_weighting", "CVA") not in {"CVA", "N4SID"}:
            raise ValueError("ss_weighting must be 'CVA' or 'N4SID'")
        if kwargs.get("criterion", "BIC") not in {"AIC", "AICc", "BIC"}:
            raise ValueError("criterion must be 'AIC', 'AICc', or 'BIC'")
        return True
