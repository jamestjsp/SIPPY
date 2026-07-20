"""
Main system identification interface.
"""

import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np

from .base import StateSpaceModel, SystemIdentificationConfig
from .factory import create_algorithm
from .parameters import normalize_identification_options, normalize_method

try:
    from sysidbox import functionset as fs
except ImportError:
    fs = None

if TYPE_CHECKING:
    from .iddata import IDData


class SystemIdentification:
    """Main class for system identification using factory pattern."""

    def __init__(self, config: Optional[SystemIdentificationConfig] = None):
        """
        Initialize system identification.

        Args:
            config: Configuration object. If None, default config is used.
        """
        self.config = config or SystemIdentificationConfig()

    def identify(
        self,
        y: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
        iddata: Optional["IDData"] = None,
        reference: Optional[np.ndarray] = None,
        **kwargs,
    ) -> StateSpaceModel:
        """
        Perform system identification.

        Args:
            y: Output data (outputs x time_steps) - alternative to iddata
            u: Input data (inputs x time_steps) - alternative to iddata (can be None for ARMA)
            iddata: IDData object containing input and output data
            **kwargs: Override config parameters

        Returns:
            StateSpaceModel: Identified model

        Note:
            Either (y, u) or iddata should be provided, but not both.
            For ARMA (time series model), u can be None.
        """
        # Validate input arguments
        if iddata is not None and (y is not None or u is not None):
            raise ValueError("Provide either iddata or (y, u), but not both")
        if iddata is not None and reference is not None:
            raise ValueError(
                "Provide references through IDData or reference=, not both"
            )

        method_override = kwargs.pop("method", None)
        id_method = kwargs.pop("id_method", None)
        if (
            method_override is not None
            and id_method is not None
            and normalize_method(method_override) != normalize_method(id_method)
        ):
            raise ValueError("method and id_method select different algorithms")
        method = normalize_method(method_override or id_method or self.config.method)
        is_time_series_method = method == "ARMA"

        if iddata is None and y is None:
            raise ValueError("Must provide either iddata or y")
        if iddata is None and u is None and not is_time_series_method:
            raise ValueError(
                "Must provide either iddata or both y and u (unless using ARMA)"
            )

        # Extract data if IDData is provided
        if iddata is not None:
            y = iddata.get_output_array()
            u = iddata.get_input_array()
            reference = iddata.get_reference_array()
            kwargs["tsample"] = iddata.sample_time

        if reference is not None:
            reference = np.atleast_2d(np.asarray(reference, dtype=float)).copy()
            if reference.shape[1] != np.atleast_2d(y).shape[1]:
                raise ValueError(
                    "Reference must share the input and output sample count"
                )

        config_dict = self.config.__dict__.copy()
        config_dict.pop("method", None)
        centering = config_dict.pop("centering", "None")
        criterion = config_dict.pop("ic", None)
        if method == "SUBSPACE" and criterion not in (None, "None"):
            config_dict["criterion"] = criterion
        config_dict = normalize_identification_options(
            method,
            config_dict,
            warn_unknown=False,
            warn_deprecated=False,
        )
        config_dict.update(normalize_identification_options(method, kwargs))

        # Create algorithm instance
        algorithm = create_algorithm(method)

        # Apply data centering if specified
        y_centered, u_centered = self._apply_centering(y, u, centering)
        if reference is not None:
            if centering == "InitVal":
                reference -= reference[:, :1]
            elif centering == "MeanVal":
                reference -= np.mean(reference, axis=1, keepdims=True)
            config_dict["reference"] = reference

        # Perform identification
        model = algorithm.identify(y_centered, u_centered, **config_dict)

        return model

    def _apply_centering(
        self, y: np.ndarray, u: Optional[np.ndarray], centering: str
    ) -> tuple:
        """Apply data centering preprocessing."""
        y = np.atleast_2d(np.asarray(y, dtype=float)).copy()
        ylength = y.shape[1]

        # Handle case where u is None (e.g., ARMA time series model)
        if u is not None:
            u = np.atleast_2d(np.asarray(u, dtype=float)).copy()
            ulength = u.shape[1]

            # Checking data consistency
            if ulength != ylength:
                raise ValueError(
                    "Input and output must share the same number of samples"
                )

        if centering == "InitVal":
            y -= y[:, :1]
            if u is not None:
                u -= u[:, :1]
        elif centering == "MeanVal":
            y -= np.mean(y, axis=1, keepdims=True)
            if u is not None:
                u -= np.mean(u, axis=1, keepdims=True)
        elif centering != "None":
            raise ValueError("centering must be 'None', 'InitVal', or 'MeanVal'")

        return y, u


def identify(
    y: Optional[np.ndarray] = None,
    u: Optional[np.ndarray] = None,
    *,
    data: Optional["IDData"] = None,
    iddata: Optional["IDData"] = None,
    reference: Optional[np.ndarray] = None,
    method: str = "SUBSPACE",
    centering: str = "None",
    **options,
) -> StateSpaceModel:
    """Identify a model through SIPPY's canonical functional API."""
    if data is not None and iddata is not None:
        raise ValueError("Provide only one of data or iddata")
    identifier = SystemIdentification(
        SystemIdentificationConfig(method=method, centering=centering)
    )
    source = data if data is not None else iddata
    return identifier.identify(
        y=y,
        u=u,
        iddata=source,
        reference=reference,
        **options,
    )


# Convenience function for backward compatibility
def system_identification(
    y: Optional[np.ndarray] = None,
    u: Optional[np.ndarray] = None,
    iddata: Optional["IDData"] = None,
    id_method: str = "SUBSPACE",
    **kwargs,
) -> StateSpaceModel:
    """
    Backward compatibility function that mimics the original API.

    This function provides the same interface as the original system_identification
    function but uses the new class-based architecture internally.
    """
    warnings.warn(
        "system_identification() is deprecated; use identify(..., method=...) instead",
        DeprecationWarning,
        stacklevel=2,
    )

    # Map old parameter names to new ones
    param_mapping = {
        "SS_fixed_order": "ss_fixed_order",
        "SS_max_order": "ss_max_order",
        "SS_orders": "ss_orders",
        "SS_threshold": "ss_threshold",
        "SS_f": "ss_f",
        "SS_D_required": "ss_d_required",
        "SS_A_stability": "ss_a_stability",
        "IC": "ic",
    }

    # Convert parameter names
    mapped_kwargs = {}
    for key, value in kwargs.items():
        mapped_key = param_mapping.get(key, key)
        # Don't override method if it's already set via id_method
        if mapped_key != "method":
            mapped_kwargs[mapped_key] = value

    return identify(y, u, iddata=iddata, method=id_method, **mapped_kwargs)
