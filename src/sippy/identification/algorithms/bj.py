"""
Box-Jenkins (BJ) identification algorithm.
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

# Import compiled utilities for performance
try:
    from ...utils.compiled_utils import (
        NUMBA_AVAILABLE,
        create_regression_matrix_bj_compiled,
    )
except ImportError:
    create_regression_matrix_bj_compiled = None
    NUMBA_AVAILABLE = False

# Check for CasADi availability for NLP-based identification
try:
    import casadi  # noqa: F401

    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False


class BJAlgorithm(IdentificationAlgorithm):
    """
    Box-Jenkins (BJ) identification algorithm.

    Implements two identification methods:

    1. **NLP Method** (CasADi + IPOPT) - DEFAULT when available:
       - Dual-path structure: separate input (B/F) and noise (C/D) optimization
       - Auxiliary variables: W (input path), V (noise path)
       - Matches master branch reference implementation
       - Decision variables: [b, f, c, d, Yidw, Ww, Vw]
       - Objective: minimize ||Y - Yidw||^2
       - Equality constraints: W - Ww = 0, V - Vw = 0, Yid - Yidw = 0
       - Optional stability constraints for F and D polynomials
       - Exact maximum likelihood estimates

    Model Structure:
    ----------------
    The BJ model structure is:
    y(k) = B(q)/F(q) u(k-nk) + C(q)/D(q) e(k)

    where:
    - B(q) = b1 + b2*q^-1 + ... + bnb*q^-(nb-1) (input numerator)
    - F(q) = 1 + f1*q^-1 + ... + fnf*q^-nf (input denominator)
    - C(q) = 1 + c1*q^-1 + ... + cnc*q^-nc (noise numerator)
    - D(q) = 1 + d1*q^-1 + ... + dnd*q^-nd (noise denominator)
    - nk is the input delay
    - e(k) is white noise

    Unlike ARMA, BJ separates input dynamics from noise dynamics
    using different polynomial structures for each path.
    """

    def __init__(self):
        """Initialize BJ algorithm."""
        super().__init__()

    def get_algorithm_name(self) -> str:
        """Return algorithm name."""
        return "BJ"

    def validate_parameters(self, **kwargs) -> bool:
        """
        Validate BJ-specific parameters.

        Parameters:
        -----------
        **kwargs : dict
            Parameters to validate including nb, nc, nd, nf

        Returns:
        --------
        bool
            True if parameters are valid
        """
        nb = kwargs.get("nb", 1)
        nc = kwargs.get("nc", 1)
        nd = kwargs.get("nd", 1)
        nf = kwargs.get("nf", 1)

        if nb <= 0:
            raise ValueError("Input order (nb) must be positive")
        if nc <= 0:
            raise ValueError("Noise AR order (nc) must be positive")
        if nd <= 0:
            raise ValueError("Noise MA orders must be positive")
        if nf <= 0:
            raise ValueError("Noise MA orders must be positive")

        return True

    def identify(
        self,
        y: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
        iddata: Optional["IDData"] = None,
        **kwargs,
    ) -> StateSpaceModel:
        """
        Identify BJ model from input-output data.

        Parameters:
        -----------
        y : np.ndarray, optional
            Output data (outputs x time_steps)
        u : np.ndarray, optional
            Input data (inputs x time_steps)
        iddata : IDData, optional
            Input-output data container
        **kwargs : dict
            Configuration parameters including nb, nc, nd, nf, nk, tsample

        Returns:
        --------
        model : StateSpaceModel
            Identified state-space model
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
                "nb": getattr(config, "nb", 1),
                "nc": getattr(config, "nc", 1),
                "nd": getattr(config, "nd", 1),
                "nf": getattr(config, "nf", 1),
                "nk": getattr(config, "nk", 1) or 1,
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

        # Extract configuration parameters (BJ specific)
        nb = kwargs.get("nb", 1)
        nc = kwargs.get("nc", 1)
        nd = kwargs.get("nd", 1)
        nf = kwargs.get("nf", 1)
        nk = kwargs.get("nk", 1) or 1  # Input delay (handle None case)

        # Validate parameters
        self.validate_parameters(nb=nb, nc=nc, nd=nd, nf=nf)

        # Remove duplicate parameters from kwargs
        kwargs_filtered = {
            k: v
            for k, v in kwargs.items()
            if k not in ["nb", "nc", "nd", "nf", "nk", "tsample"]
        }

        # Route to appropriate implementation
        if CASADI_AVAILABLE:
            try:
                if y.shape[0] == 1:
                    result = gen_miso_id(
                        id_method="BJ",
                        y=y[0],
                        u=u,
                        na=0,
                        nb=np.full(u.shape[0], nb, dtype=int),
                        nc=int(nc),
                        nd=int(nd),
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
                    id_method="BJ",
                    y=y,
                    u=u,
                    na=[0] * y.shape[0],
                    nb=np.full((y.shape[0], u.shape[0]), nb, dtype=int),
                    nc=[int(nc)] * y.shape[0],
                    nd=[int(nd)] * y.shape[0],
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
                raise RuntimeError("BJ prediction-error optimization failed") from e
        else:
            raise RuntimeError(
                "CasADi is required for BJ prediction-error identification"
            )
