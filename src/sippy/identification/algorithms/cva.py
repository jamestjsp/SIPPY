"""
CVA algorithm implementation.
"""

from typing import TYPE_CHECKING, Optional

import numpy as np

from ..base import (
    IdentificationAlgorithm,
    StateSpaceModel,
    resolve_identification_data,
)
from .subspace_core import SubspaceCoreAlgorithm

if TYPE_CHECKING:
    from ..iddata import IDData


class CVAAlgorithm(IdentificationAlgorithm):
    """CVA (Canonical Variate Analysis) algorithm."""

    def get_algorithm_name(self) -> str:
        """Return algorithm name."""
        return "CVA"

    def identify(
        self,
        y: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
        iddata: Optional["IDData"] = None,
        **kwargs,
    ) -> StateSpaceModel:
        """
        Perform CVA system identification.

        Args:
            y: Output data (outputs x time_steps) - alternative to iddata
            u: Input data (inputs x time_steps) - alternative to iddata
            iddata: IDData object containing input and output data
            **kwargs: Algorithm parameters

        Returns:
            StateSpaceModel: Identified model

        Note:
            Either (y, u) or iddata should be provided, but not both.
        """
        y, u, tsample = resolve_identification_data(
            y, u, iddata, tsample=kwargs.get("tsample", 1.0)
        )
        self.validate_parameters(**kwargs)

        # Extract parameters with defaults
        f = kwargs.get("ss_f", 20)
        threshold = kwargs.get("ss_threshold", 0.1)
        fixed_order = kwargs.get("ss_fixed_order", np.nan)
        d_required = kwargs.get("ss_d_required", False)
        a_stability = kwargs.get("ss_a_stability", False)

        # Call the core CVA implementation
        try:
            A, B, C, D, Vn, Q, R, S, K = SubspaceCoreAlgorithm.olsims(
                y,
                u,
                f,
                "CVA",
                threshold,
                np.nan,
                fixed_order,  # max_order, fixed_order
                d_required,
                a_stability,
            )
        except Exception as exc:
            raise RuntimeError(f"CVA identification failed: {exc}") from exc

        return StateSpaceModel(A, B, C, D, K, Q, R, S, tsample, Vn)

    def validate_parameters(self, **kwargs) -> bool:
        """Validate CVA-specific parameters."""
        f = kwargs.get("ss_f", 20)
        if not isinstance(f, int) or f <= 0:
            raise ValueError("ss_f must be a positive integer")

        return True
