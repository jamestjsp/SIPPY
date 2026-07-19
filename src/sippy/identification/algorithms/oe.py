"""
OE (Output Error) identification algorithm.
"""

from typing import TYPE_CHECKING, Optional

import numpy as np

from ..base import (
    IdentificationAlgorithm,
    StateSpaceModel,
)
from .ararx import _state_space_from_results, _state_space_from_single_result
from .opt_support import (
    gen_mimo_id,
    gen_miso_id,
    nk_to_theta,
)

if TYPE_CHECKING:
    from ..iddata import IDData

# Check for CasADi availability for NLP-based identification
try:
    import casadi  # noqa: F401

    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False


class OEAlgorithm(IdentificationAlgorithm):
    """
    OE (Output Error) identification algorithm.

    Implements two identification methods:

    1. **NLP Method** (CasADi + IPOPT) - DEFAULT when available:
       - Uses predicted outputs (Yid) in regressor (iterative, nonlinear)
       - Matches master branch reference implementation
       - Decision variables: [b, f, Yid] coefficients + auxiliary time series
       - Objective: minimize ||Y - Yid||^2
       - Equality constraint: symbolic Yid - optimization Yid = 0
       - Optional stability constraints via companion matrix norms
       - Exact maximum likelihood estimates

    Model Structure:
    ----------------
    The OE model structure is:
    y(k) = B(q)/F(q) * u(k-nk) + e(k)

    where:
    - B(q) = b1 + b2*q^-1 + ... + bnb*q^-(nb-1) (numerator polynomial)
    - F(q) = 1 + f1*q^-1 + ... + fnf*q^-nf (denominator polynomial)
    - nk is the input delay (number of samples)
    - e(k) is white noise

    The OE algorithm estimates parameters using output error prediction error methods,
    which is nonlinear due to the noise-free output feedback through F(q).
    """

    def __init__(self):
        """Initialize OE algorithm."""
        super().__init__()

    def get_algorithm_name(self) -> str:
        """Return algorithm name."""
        return "OE"

    def validate_parameters(self, **kwargs) -> bool:
        """
        Validate OE-specific parameters.

        Parameters:
        -----------
        **kwargs : dict
            Parameters to validate including nb, nf, nk

        Returns:
        --------
        bool
            True if parameters are valid
        """
        nb = kwargs.get("nb", 2)
        nf = kwargs.get("nf", 2)
        nk = kwargs.get("nk", 1)

        if nb <= 0:
            raise ValueError("Numerator order (nb) must be positive")
        if nf <= 0:
            raise ValueError("Denominator order (nf) must be positive")
        if nk < 0:
            raise ValueError("Input delay (nk) must be non-negative")

        return True

    def identify(
        self,
        y: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
        iddata: Optional["IDData"] = None,
        **kwargs,
    ) -> StateSpaceModel:
        """
        Identify OE model from input-output data.

        Automatically selects NLP method (CasADi + IPOPT) if available,
        otherwise falls back to simplified direct least squares method.

        Parameters:
        -----------
        y : np.ndarray, optional
            Output data (outputs x time_steps)
        u : np.ndarray, optional
            Input data (inputs x time_steps)
        iddata : IDData, optional
            Input-output data container
        **kwargs : dict
            Configuration parameters including:
            - nb: int, numerator order
            - nf: int, denominator order
            - nk: int, input delay
            - tsample: float, sampling time
            - max_iterations: int, IPOPT max iterations (default 200)
            - stability_constraint: bool, enable stability constraints (default False)
            - stability_margin: float, stability margin for constraints (default 1.0)

        Returns:
        --------
        model : StateSpaceModel
            Identified state-space model with G_tf, H_tf, Yid attributes
        """
        # Backward compatibility: detect old API (data, config) vs new API (y, u, **kwargs)
        from ..base import SystemIdentificationConfig
        from ..iddata import IDData as IDDataClass

        if (
            y is not None
            and isinstance(y, IDDataClass)
            and u is not None
            and isinstance(u, SystemIdentificationConfig)
        ):
            # Old API: identify(data, config)
            iddata = y
            config = u
            y = None
            u = None
            # Extract parameters from config
            kwargs = {
                "nb": getattr(config, "nb", 2),
                "nf": getattr(config, "nf", 2),
                "nk": getattr(config, "nk", 1),
                "max_iterations": getattr(config, "max_iterations", 200),
            }

        # Validate input arguments
        if iddata is not None and (y is not None or u is not None):
            raise ValueError("Provide either iddata or (y, u), but not both")
        if iddata is None and (y is None or u is None):
            raise ValueError("Must provide either iddata or both y and u")

        # Extract data if IDData is provided
        if iddata is not None:
            u = iddata.get_input_array()
            y = iddata.get_output_array()
            sample_time = iddata.sample_time
        else:
            # Ensure arrays are 2D
            y = np.atleast_2d(y)
            u = np.atleast_2d(u)
            sample_time = kwargs.get("tsample", 1.0)

        # Extract configuration parameters (OE specific)
        nb = kwargs.get("nb", 2)
        nf = kwargs.get("nf", 2)
        nk = kwargs.get("nk", 1)

        # Validate parameters
        self.validate_parameters(nb=nb, nf=nf, nk=nk)

        # Route to appropriate implementation
        # Remove nb, nf, nk from kwargs to avoid duplicate argument errors
        kwargs_filtered = {
            k: v for k, v in kwargs.items() if k not in ["nb", "nf", "nk", "tsample"]
        }

        if CASADI_AVAILABLE:
            try:
                if y.shape[0] == 1:
                    result = gen_miso_id(
                        id_method="OE",
                        y=y[0],
                        u=u,
                        na=0,
                        nb=np.full(u.shape[0], nb, dtype=int),
                        nc=0,
                        nd=0,
                        nf=int(nf),
                        theta=np.full(u.shape[0], nk_to_theta(nk), dtype=int),
                        max_iterations=kwargs_filtered.get("max_iterations", 200),
                        stability_margin=kwargs_filtered.get(
                            "stability_margin", kwargs_filtered.get("stab_marg", 1.0)
                        ),
                        enforce_stability=kwargs_filtered.get(
                            "stability_constraint",
                            kwargs_filtered.get("stab_cons", False),
                        ),
                    )
                    return _state_space_from_single_result(
                        result, u.shape[0], sample_time
                    )
                results, _ = gen_mimo_id(
                    id_method="OE",
                    y=y,
                    u=u,
                    na=[0] * y.shape[0],
                    nb=np.full((y.shape[0], u.shape[0]), nb, dtype=int),
                    nc=[0] * y.shape[0],
                    nd=[0] * y.shape[0],
                    nf=[int(nf)] * y.shape[0],
                    theta=np.full((y.shape[0], u.shape[0]), nk_to_theta(nk), dtype=int),
                    sample_time=sample_time,
                    max_iterations=kwargs_filtered.get("max_iterations", 200),
                    stability_margin=kwargs_filtered.get(
                        "stability_margin", kwargs_filtered.get("stab_marg", 1.0)
                    ),
                    enforce_stability=kwargs_filtered.get(
                        "stability_constraint", kwargs_filtered.get("stab_cons", False)
                    ),
                )
                return _state_space_from_results(results, u.shape[0], sample_time)
            except Exception as e:
                raise RuntimeError("OE prediction-error optimization failed") from e
        else:
            raise RuntimeError(
                "CasADi is required for OE prediction-error identification"
            )
