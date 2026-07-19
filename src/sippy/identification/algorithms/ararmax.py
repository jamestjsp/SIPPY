"""
ARARMAX (Auto-Regressive ARMAX) identification algorithm.
"""

import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.linalg import lstsq

from sippy import systems as control

from ..base import IdentificationAlgorithm, StateSpaceModel, realize_transfer_function
from ..parameters import normalize_identification_options
from .ararx import _state_space_from_results, _state_space_from_single_result
from .opt_support import (
    apply_platform_ipopt_options,
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
        create_regression_matrix_ararmax_compiled,
    )
except ImportError:
    create_regression_matrix_ararmax_compiled = None
    NUMBA_AVAILABLE = False

# Check for CasADi availability for NLP-based identification
try:
    import casadi  # noqa: F401

    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False


class ARARMAXAlgorithm(IdentificationAlgorithm):
    """
    ARARMAX (Auto-Regressive ARMAX) identification algorithm.

    Implements two identification methods:

    1. **NLP Method** (CasADi + IPOPT) - DEFAULT when available:
       - Simultaneous optimization of all parameters [a, b, c, d]
       - Auxiliary variables: W (input dynamics), V (residual after AR and input dynamics)
       - Matches master branch reference implementation
       - Decision variables: [a, b, c, d, Yidw, Ww, Vw]
       - Objective: minimize ||Y - Yidw||^2
       - Equality constraints: W - Ww = 0, V - Vw = 0, Yid - Yidw = 0
       - Optional stability constraints for A and D polynomials
       - True prediction error refinement (no hardcoded approximations)

    2. **Simplified Method** (Direct LS) - Fallback when CasADi unavailable:
       - Single-pass least squares
       - Approximated noise with heuristics (hardcoded 0.1 scaling)
       - 50-200x faster but may produce suboptimal parameters

    Model Structure:
    ----------------
    The ARARMAX model structure is:
    A(q) y(k) = B(q)/F(q) u(k-nk) + C(q)/D(q) e(k)

    where:
    - A(q) = 1 + a1*q^-1 + ... + ana*q^-na (auto-regressive part for y)
    - B(q)/F(q) = input transfer function polynomials
    - C(q) = 1 + c1*q^-1 + ... + cnc*q^-nc (noise AR polynomial)
    - D(q) = 1 + d1*q^-1 + ... + dnd*q^-nd (noise MA polynomial)
    - e(k) is white noise
    - nk is the input delay
    """

    def __init__(self):
        """Initialize ARARMAX algorithm."""
        super().__init__()

    def get_algorithm_name(self) -> str:
        """Return the algorithm name."""
        return "ARARMAX"

    def validate_parameters(self, **kwargs) -> bool:
        """Validate ARARMAX orders."""
        values = {
            "na": kwargs.get("na", 1),
            "nb": kwargs.get("nb", 1),
            "nc": kwargs.get("nc", 1),
            "nd": kwargs.get("nd", 1),
            "nf": kwargs.get("nf", 0),
            "nk": kwargs.get("nk", 1),
        }
        labels = {
            "na": "AR order (na)",
            "nb": "Input order (nb)",
            "nc": "Noise AR order (nc)",
            "nd": "Noise MA order (nd)",
            "nf": "Input TF order (nf)",
            "nk": "Input delay (nk)",
        }
        for name, value in values.items():
            if isinstance(value, (str, bytes)):
                raise ValueError(f"{labels[name]} must contain integers")
            try:
                entries = np.asarray(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{labels[name]} must contain integers") from error
            if entries.size == 0:
                raise ValueError(f"{labels[name]} must not be empty")
            if not np.issubdtype(entries.dtype, np.integer):
                raise ValueError(f"{labels[name]} must contain integers")
            if name in {"na", "nb"}:
                if np.any(entries < 0) or not np.any(entries > 0):
                    raise ValueError(f"{labels[name]} must be positive")
            elif np.any(entries < 0):
                raise ValueError(f"{labels[name]} must be non-negative")
        return True

    def validate_config(self, config):
        """Validate a configuration through the shared option normalizer."""
        options = normalize_identification_options(
            "ARARMAX",
            vars(config),
            warn_unknown=False,
            warn_deprecated=False,
        )
        return self.validate_parameters(**options)

    def identify(
        self,
        y: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
        iddata: Optional["IDData"] = None,
        **kwargs,
    ) -> "StateSpaceModel":
        """
        Identify ARARMAX model from data.

        Args:
            y: Output data (deprecated, use iddata)
            u: Input data (deprecated, use iddata)
            iddata: IDData instance containing input/output data
            **kwargs: Configuration parameters including config

        Returns:
            StateSpaceModel: Identified ARARMAX model

        Raises:
            ValueError: If parameters are invalid or data insufficient
        """
        if y is None or u is None:
            raise ValueError("ARARMAX requires both input and output data")
        y = np.atleast_2d(y)
        u = np.atleast_2d(u)
        sample_time = kwargs.get("tsample", 1.0)

        # Check data sufficiency
        if y.shape[1] < 2 or u.shape[1] < 2:
            raise ValueError("Insufficient data: need at least 2 samples")

        na = kwargs.get("na", 1)
        nb = kwargs.get("nb", 1)
        nc = kwargs.get("nc", 1)
        nd = kwargs.get("nd", 1)
        nf = kwargs.get("nf", 0)
        nk = kwargs.get("nk", 1)

        # Handle None values with defaults
        if na is None:
            na = 1
        if nb is None:
            nb = 1
        if nc is None:
            nc = 1
        if nd is None:
            nd = 1
        if nf is None:
            nf = 0
        if nk is None:
            nk = 1

        self.validate_parameters(na=na, nb=nb, nc=nc, nd=nd, nf=nf, nk=nk)

        # Flatten nested lists and extract max values for MIMO compatibility
        def flatten_and_max(param):
            flat_param = np.asarray(param, dtype=int).ravel().tolist()
            return max(flat_param) if flat_param else 0, flat_param

        max_na, na = flatten_and_max(na)
        max_nb, nb = flatten_and_max(nb)
        max_nc, nc = flatten_and_max(nc)
        max_nd, nd = flatten_and_max(nd)
        max_nf, nf = flatten_and_max(nf)
        max_nk, nk = flatten_and_max(nk)

        # Get dimensions (arrays are already 2D from atleast_2d)
        ny, n_samples = y.shape
        nu, _ = u.shape

        max_order = max(max_na, max_nb + max_nk, max_nc, max_nd, max_nf)

        # Check data sufficiency
        if n_samples <= max_order + 10:  # Need enough data for estimation
            raise ValueError(
                f"Insufficient data: need at least {max_order + 10} samples, got {n_samples}"
            )

        # Route to appropriate implementation
        # Filter kwargs to avoid duplicate argument errors
        kwargs_filtered = {
            k: v
            for k, v in kwargs.items()
            if k not in ["na", "nb", "nc", "nd", "nf", "nk", "tsample"]
        }

        if CASADI_AVAILABLE:
            try:
                if ny == 1:
                    result = gen_miso_id(
                        id_method="ARARMAX",
                        y=y[0],
                        u=u,
                        na=max_na,
                        nb=np.full(nu, max_nb, dtype=int),
                        nc=max_nc,
                        nd=max_nd,
                        nf=0,
                        theta=np.full(nu, nk_to_theta(max_nk), dtype=int),
                        max_iterations=kwargs_filtered.get("max_iterations", 200),
                        stability_margin=kwargs_filtered.get("stability_margin", 1.0),
                        enforce_stability=kwargs_filtered.get(
                            "stability_constraint", False
                        ),
                    )
                    return _state_space_from_single_result(
                        result, u.shape[0], sample_time
                    )
                results, _ = gen_mimo_id(
                    id_method="ARARMAX",
                    y=y,
                    u=u,
                    na=[max_na] * ny,
                    nb=np.full((ny, nu), max_nb, dtype=int),
                    nc=[max_nc] * ny,
                    nd=[max_nd] * ny,
                    nf=[0] * ny,
                    theta=np.full((ny, nu), nk_to_theta(max_nk), dtype=int),
                    sample_time=sample_time,
                    max_iterations=kwargs_filtered.get("max_iterations", 200),
                    stability_margin=kwargs_filtered.get("stability_margin", 1.0),
                    enforce_stability=kwargs_filtered.get(
                        "stability_constraint", False
                    ),
                )
                return _state_space_from_results(results, nu, sample_time)
            except Exception as e:
                warnings.warn(
                    f"NLP identification failed: {e}. Falling back to simplified LS method."
                )
                # Fall through to existing implementation
        else:
            warnings.warn(
                "CasADi not available. Using simplified LS method (may be less accurate than master branch)."
            )
            # Fall through to existing implementation

        # EXISTING SIMPLIFIED IMPLEMENTATION (FALLBACK)
        # Convert to format expected by regression builder (samples x channels)
        u_values = u.T  # Transpose to (samples, inputs)
        y_values = y.T  # Transpose to (samples, outputs)

        # Build regression matrices for ARARMAX
        # Combine ARX structure with ARMA noise modeling
        phi, target = self._build_regression_matrices_ararmax(
            u_values, y_values, na, nb, nc, nd, nf, nk
        )

        if phi.shape[0] < phi.shape[1]:
            raise ValueError("Insufficient data for reliable parameter estimation")

        # Solve least squares problem
        theta, residuals, rank, s = lstsq(phi, target, rcond=None)

        # Extract system matrices from estimated parameters
        A, B, C, D, x0 = self._extract_state_space_matrices_ararmax(
            theta, na, nb, nc, nd, nf, nk, sample_time, nu, ny
        )

        # Create G_tf and H_tf transfer functions
        G_tf, H_tf = self._create_transfer_functions_ararmax(
            theta, na, nb, nc, nd, nf, nk, sample_time
        )

        # Compute one-step-ahead predictions (Yid) for identification data
        Yid = self._compute_yid_ararmax(
            u_values, y_values, theta, na, nb, nc, nd, nf, nk
        )

        return StateSpaceModel(
            A=A,
            B=B,
            C=C,
            D=D,
            K=None,
            Q=None,
            R=None,
            S=None,
            ts=sample_time,
            Vn=None,
            G_tf=G_tf,
            H_tf=H_tf,
            Yid=Yid,
            identification_info={"fit_start": max_order},
        )

    def _identify_nlp(self, u, y, na, nb, nc, nd, nf, nk, sample_time, **kwargs):
        """
        ARARMAX identification using NLP (CasADi + IPOPT).

        Matches master branch implementation with auxiliary variables W and V.
        """
        import casadi as ca  # noqa: F401 - used extensively in this function

        # Get dimensions (u and y are 2D: channels x samples)
        ny, N = y.shape
        nu, _ = u.shape

        # Check SISO constraint
        if ny > 1 or nu > 1:
            raise ValueError("NLP method currently supports SISO only")

        # Flatten for SISO
        y_flat = y.flatten()
        u_flat = u.flatten()

        # Extract parameters
        max_iterations = kwargs.get("max_iterations", 200)
        stability_constraint = kwargs.get("stability_constraint", False)
        stability_margin = kwargs.get("stability_margin", 1.0)

        # Build and solve NLP
        solution = self._build_ararmax_nlp(
            u_flat,
            y_flat,
            na,
            nb,
            nc,
            nd,
            nf,
            nk,
            N,
            max_iterations,
            stability_constraint,
            stability_margin,
        )

        # Create model from solution
        # Extract coefficients
        theta = np.concatenate(
            [solution["a"], solution["b"], solution["c"], solution["d"]]
        )

        # Build state-space matrices (reuse existing helper methods)
        A, B, C, D, x0 = self._extract_state_space_matrices_ararmax(
            theta, [na], [nb], [nc], [nd], [nf], [nk], sample_time, nu, ny
        )

        # Create G_tf and H_tf
        G_tf, H_tf = self._create_transfer_functions_ararmax(
            theta, [na], [nb], [nc], [nd], [nf], [nk], sample_time
        )
        if G_tf is not None:
            A, B, C, D = realize_transfer_function(G_tf)

        # Reshape Yid for output (outputs x samples)
        Yid = solution["Yid"].reshape(1, -1) if ny == 1 else solution["Yid"]

        # Create StateSpaceModel
        from ..base import StateSpaceModel

        model = StateSpaceModel(
            A=A,
            B=B,
            C=C,
            D=D,
            K=None,
            Q=None,
            R=None,
            S=None,
            ts=sample_time,
            Vn=solution["Vn"],
        )

        # Attach results
        model.G_tf = G_tf
        model.H_tf = H_tf
        model.Yid = Yid

        return model

    def _build_ararmax_nlp(
        self,
        u,
        y,
        na,
        nb,
        nc,
        nd,
        nf,
        nk,
        N,
        max_iterations,
        stability_constraint,
        stability_margin,
    ):
        """
        Build and solve ARARMAX NLP problem using CasADi + IPOPT.

        ARARMAX structure from master branch:
        - Decision variables: [a, b, c, d, Yidw, Ww, Vw]
        - W[k] = B*u (input dynamics, simplified without F for now)
        - V[k] = A*y - W (residual after AR and input dynamics)
        - Regressor: [-vecY, vecU, vecE, -vecV]
        - Constraints: W - Ww = 0, V - Vw = 0, Yid - Yidw = 0
        """
        import casadi as ca

        # Number of coefficients
        n_coeff = na + nb + nc + nd

        # Decision variables: [a (na), b (nb), c (nc), d (nd), Yidw (N), Ww (N), Vw (N)]
        n_opt = n_coeff + 3 * N  # 3*N for Yidw, Ww, Vw
        w_opt = ca.SX.sym("w", n_opt)

        # Extract coefficient variables
        a = w_opt[0:na]
        b = w_opt[na : na + nb]
        c = w_opt[na + nb : na + nb + nc]
        d = w_opt[na + nb + nc : na + nb + nc + nd]

        # Extract auxiliary variables (following master branch pattern)
        Yidw = w_opt[n_coeff + 2 * N : n_coeff + 3 * N]  # Last N elements
        Ww = w_opt[n_coeff : n_coeff + N]  # First auxiliary N
        Vw = w_opt[n_coeff + N : n_coeff + 2 * N]  # Second auxiliary N

        # Initialize symbolic variables
        Yid = y * ca.SX.ones(1)
        W = y * ca.SX.ones(1)  # w = B*u
        V = y * ca.SX.ones(1)  # v = A*y - W
        Epsi = ca.SX.zeros(N)  # Prediction error

        # Maximum lag
        n_tr = max(na, nb + nk, nc, nd, nf)

        # Build symbolic loop (follow master branch ARARMAX structure!)
        for k in range(n_tr, N):
            # === Build regressor parts ===

            # AR terms: -vecY (lagged outputs)
            vecY = []
            for i in range(na):
                idx = k - 1 - i
                if idx >= 0:
                    vecY.append(-y[idx])
                else:
                    vecY.append(0.0)

            # Input terms: vecU (lagged inputs)
            vecU = []
            for i in range(nb):
                idx = k - nk - i
                if idx >= 0:
                    vecU.append(u[idx])
                else:
                    vecU.append(0.0)

            # Prediction error terms
            Epsi[k] = y[k] - Yidw[k]

            # Noise AR terms: vecE (lagged prediction errors)
            vecE = []
            for i in range(nc):
                idx = k - 1 - i
                if idx >= 0:
                    vecE.append(Epsi[idx])
                else:
                    vecE.append(0.0)

            # Auxiliary variable V: lagged residuals
            vecV = []
            for i in range(nd):
                idx = k - 1 - i
                if idx >= 0:
                    vecV.append(-Vw[idx])
                else:
                    vecV.append(0.0)

            # === Build full regressor: [-vecY, vecU, vecE, -vecV] ===
            phi_parts = vecY + vecU + vecE + vecV
            if phi_parts:
                phi = ca.vertcat(*phi_parts)
                coeff = ca.vertcat(a, b, c, d)
                Yid[k] = ca.mtimes(phi.T, coeff)

            # === Compute auxiliary variable W[k] = B*u ===
            vecU_w = []
            for i in range(nb):
                idx = k - nk - i
                if idx >= 0:
                    vecU_w.append(u[idx])
                else:
                    vecU_w.append(0.0)

            if vecU_w:
                phiw = ca.vertcat(*vecU_w)
                W[k] = ca.mtimes(phiw.T, b)

            # === Compute auxiliary variable V[k] = A*y - W ===
            # V[k] = Y[k] + sum(a[i] * Y[k-1-i]) - Ww[k]
            V_k = y[k]
            for i in range(na):
                idx = k - 1 - i
                if idx >= 0:
                    V_k += a[i] * y[idx]
            V[k] = V_k - Ww[k]

        # Objective: minimize ||Y - Yidw||^2
        DY = y - Yidw
        f_obj = (1.0 / N) * ca.mtimes(DY.T, DY)

        # Constraints
        g = []
        g_lb = []
        g_ub = []

        # Equality constraints
        g.append(Yid - Yidw)
        g_lb.extend([-1e-7] * N)
        g_ub.extend([1e-7] * N)

        g.append(W - Ww)
        g_lb.extend([-1e-7] * N)
        g_ub.extend([1e-7] * N)

        g.append(V - Vw)
        g_lb.extend([-1e-7] * N)
        g_ub.extend([1e-7] * N)

        # Optional stability constraints
        if stability_constraint:
            if na > 0:
                compA = ca.SX.zeros(na, na)
                if na > 1:
                    diagA = ca.SX.eye(na - 1)
                    compA[:-1, 1:] = diagA
                compA[-1, :] = -a[::-1]
                norm_CompA = ca.norm_inf(compA)
                g.append(norm_CompA)
                g_lb.append(-1e-7)
                g_ub.append(stability_margin)

            if nd > 0:
                compD = ca.SX.zeros(nd, nd)
                if nd > 1:
                    diagD = ca.SX.eye(nd - 1)
                    compD[:-1, 1:] = diagD
                compD[-1, :] = -d[::-1]
                norm_CompD = ca.norm_inf(compD)
                g.append(norm_CompD)
                g_lb.append(-1e-7)
                g_ub.append(stability_margin)

        # Stack constraints
        g_vec = ca.vertcat(*g)

        # Bounds
        w_lb = -1e2 * ca.DM.ones(n_opt)
        w_ub = 1e2 * ca.DM.ones(n_opt)

        # NLP problem
        nlp = {"x": w_opt, "f": f_obj, "g": g_vec}

        # Solver options
        opts = {
            "ipopt.max_iter": max_iterations,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
        }
        apply_platform_ipopt_options(opts)

        # Create solver
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        # Initial guess
        w_0 = np.zeros(n_opt)
        w_0[n_coeff + 2 * N : n_coeff + 3 * N] = y  # Initialize Yidw
        # Ww and Vw start at zero

        # Solve NLP
        sol = solver(lbx=w_lb, ubx=w_ub, x0=w_0, lbg=g_lb, ubg=g_ub)

        # Extract solution
        x_opt = sol["x"].full().flatten()

        a_opt = x_opt[0:na]
        b_opt = x_opt[na : na + nb]
        c_opt = x_opt[na + nb : na + nb + nc]
        d_opt = x_opt[na + nb + nc : na + nb + nc + nd]
        Yid_opt = x_opt[n_coeff + 2 * N : n_coeff + 3 * N]

        # Compute noise variance
        Vn = np.linalg.norm(y - Yid_opt, 2) ** 2 / (2 * N)

        return {
            "a": a_opt,
            "b": b_opt,
            "c": c_opt,
            "d": d_opt,
            "Yid": Yid_opt,
            "Vn": Vn,
        }

    def _build_regression_matrices_ararmax(self, u, y, na, nb, nc, nd, nf, nk):
        """
        Build regression matrices for ARARMAX parameter estimation.

        This function automatically uses the Numba-compiled version when available
        for improved performance.
        """
        n_samples = len(u)
        n_outputs = y.shape[1] if len(y.shape) > 1 else 1
        n_inputs = u.shape[1] if len(u.shape) > 1 else 1

        # Use scalar orders (already flattened by caller)
        na_val = max(na) if isinstance(na, (list, tuple)) else na
        nb_val = max(nb) if isinstance(nb, (list, tuple)) else nb
        nc_val = max(nc) if isinstance(nc, (list, tuple)) else nc
        nd_val = max(nd) if isinstance(nd, (list, tuple)) else nd

        # Try to use compiled version for simplified case
        if NUMBA_AVAILABLE and create_regression_matrix_ararmax_compiled is not None:
            # Transpose data to match expected format (outputs/inputs x time)
            if n_outputs > 1:
                y_t = y.T
            else:
                y_t = y.reshape(1, -1)
            if n_inputs > 1:
                u_t = u.T
            else:
                u_t = u.reshape(1, -1)

            try:
                phi_compiled, target_compiled = (
                    create_regression_matrix_ararmax_compiled(
                        u_t,
                        y_t,
                        na_val,
                        nb_val,
                        nc_val,
                        nd_val,
                        max(nf) if isinstance(nf, (list, tuple)) else nf,
                        max(nk) if isinstance(nk, (list, tuple)) else nk,
                        n_outputs,
                        n_inputs,
                        n_samples,
                    )
                )
                # The compiled version returns flattened target, so we need to extract single output
                if phi_compiled.shape[0] > 0 and phi_compiled.shape[1] > 1:
                    return phi_compiled[:, :-1], phi_compiled[
                        :, -1
                    ]  # Use last column as output
            except Exception:
                # Fall back to original implementation if compilation fails
                pass

        # Fallback to original implementation
        # Calculate total number of parameters
        n_params_ar = na_val * n_outputs
        n_params_input = nb_val * n_inputs
        n_params_noise_ar = nc_val
        n_params_noise_ma = nd_val
        n_params_total = (
            n_params_ar + n_params_input + n_params_noise_ar + n_params_noise_ma
        )

        # Initialize regression matrices
        phi = np.zeros(
            (
                n_samples
                - max(
                    na_val,
                    nb_val + max(nk) if isinstance(nk, (list, tuple)) else nk,
                    nc_val,
                    nd_val,
                    max(nf) if isinstance(nf, (list, tuple)) else nf,
                ),
                n_params_total,
            )
        )
        target = np.zeros(
            n_samples
            - max(
                na_val,
                nb_val + max(nk) if isinstance(nk, (list, tuple)) else nk,
                nc_val,
                nd_val,
                max(nf) if isinstance(nf, (list, tuple)) else nf,
            )
        )

        max_order = max(
            na_val,
            nb_val + max(nk) if isinstance(nk, (list, tuple)) else nk,
            nc_val,
            nd_val,
            max(nf) if isinstance(nf, (list, tuple)) else nf,
        )

        for k in range(max_order, n_samples):
            row_idx = 0
            # AR terms (for y) - handle SISO properly
            for i in range(1, min(na_val + 1, k + 1)):
                for j in range(n_outputs):
                    if row_idx < phi.shape[1]:
                        phi[k - max_order, row_idx] = float(
                            y[k - i][j] if n_outputs > 1 else y[k - i][0]
                        )
                        row_idx += 1
                    else:
                        break

            # Input terms - handle SISO properly
            for i in range(
                max(nk) if isinstance(nk, (list, tuple)) else nk,
                min(
                    nb_val + (max(nk) if isinstance(nk, (list, tuple)) else nk) + 1,
                    k + 1,
                ),
            ):
                for j in range(n_inputs):
                    if row_idx < phi.shape[1]:
                        phi[k - max_order, row_idx] = float(
                            u[k - i][j] if n_inputs > 1 else u[k - i][0]
                        )
                        row_idx += 1
                    else:
                        break

            # Noise AR terms (approximated with residuals)
            for i in range(1, min(nc_val + 1, k + 1)):
                # Use approximate residual y[k] - predicted y[k]
                if k >= na_val + nb_val:
                    pred = sum(
                        (y[k - j - 1][0] if n_outputs > 1 else y[k - j - 1])
                        * (0.1 if j < na_val else 0)
                        for j in range(max(na_val, 1))
                    )
                    resid = (y[k][0] if n_outputs > 1 else y[k]) - pred
                else:
                    resid = (
                        y[k][0] if n_outputs > 1 else y[k]
                    ) * 0.1  # Initial approximation
                if row_idx < phi.shape[1]:
                    resid_val = resid.item() if hasattr(resid, "item") else float(resid)
                    phi[k - max_order, row_idx] = resid_val
                    row_idx += 1
                else:
                    break

            # Noise MA terms (approximated)
            for i in range(1, min(nd_val + 1, k + 1)):
                if k >= i:
                    # Use difference of residuals
                    if k >= na_val + nb_val:
                        pred1 = sum(
                            (y[k - j - 1][0] if n_outputs > 1 else y[k - j - 1])
                            * (0.1 if j < na_val else 0)
                            for j in range(max(na_val, 1))
                        )
                        pred2 = sum(
                            (y[k - i - j - 1][0] if n_outputs > 1 else y[k - i - j - 1])
                            * (0.1 if j < na_val else 0)
                            for j in range(max(na_val, 1))
                        )
                        resid_diff = ((y[k][0] if n_outputs > 1 else y[k]) - pred1) - (
                            (y[k - i][0] if n_outputs > 1 else y[k - i]) - pred2
                        )
                    else:
                        resid_diff = (
                            (y[k][0] if n_outputs > 1 else y[k])
                            - (y[k - i][0] if n_outputs > 1 else y[k - i])
                        ) * 0.1
                    if row_idx < phi.shape[1]:
                        phi[k - max_order, row_idx] = float(resid_diff)
                        row_idx += 1
                    else:
                        # Skip if we exceed the allocated columns
                        break

            target[k - max_order] = y[k, 0] if n_outputs > 1 else y[k]

        return phi, target

    def _extract_state_space_matrices_ararmax(
        self, theta, na, nb, nc, nd, nf, nk, sample_time, n_inputs, n_outputs
    ):
        """Extract state space matrices from ARARMAX parameters."""
        # For ARARMAX, we need to handle both AR and MA components
        # This is a simplified implementation
        na_val = max(na)
        nb_val = max(nb)
        nk_val = max(nk) if isinstance(nk, (list, tuple)) else nk

        # Main system parameters (first na_val + nb_val coefficients)
        n_main = max(na_val, nb_val + nk_val)
        if n_main == 0:
            n_main = 1

        A = np.eye(n_main)
        B = np.zeros((n_main, n_inputs))
        C = np.zeros((n_outputs, n_main))
        D = np.zeros((n_outputs, n_inputs))

        # Build companion form for main system
        if na_val > 0:
            # AR polynomial coefficients
            for i in range(min(na_val, len(theta))):
                A[-1, i] = -theta[i]
            C[0, -1] = 1.0

        if nb_val > 0 and na_val + nb_val <= len(theta):
            # Input coefficients - handle MIMO
            for i in range(min(nb_val, len(theta) - na_val)):
                for j in range(n_inputs):
                    coeff_idx = na_val + i * n_inputs + j
                    if coeff_idx < len(theta):
                        B[max(0, na_val - 1 - i), j] = theta[coeff_idx]

        x0 = np.zeros((n_main, 1))

        return A, B, C, D, x0

    def _companion_matrix_main(self, theta, na, nb, nk):
        """Build companion matrix for main system part."""
        na_val = max(na) if hasattr(na, "__iter__") else na
        nb_val = max(nb) if hasattr(nb, "__iter__") else nb
        nk_val = max(nk) if hasattr(nk, "__iter__") else nk

        n_main = max(na_val, nb_val + nk_val)
        if n_main == 0:
            return np.array([[1.0]])

        A_main = np.eye(n_main)
        A_main[-1, :] = 0.0  # Last row for AR coefficients

        # Fill AR coefficients
        for i in range(min(na_val, len(theta))):
            A_main[-1, i] = -theta[i]

        # Add companion structure
        for i in range(n_main - 1):
            A_main[i, i + 1] = 1.0

        return A_main

    def _build_B_matrix(self, theta, na, nb, nk, n_inputs):
        """Build B matrix from input coefficients."""
        na_val = max(na) if hasattr(na, "__iter__") else na
        nb_val = max(nb) if hasattr(nb, "__iter__") else nb
        nk_val = max(nk) if hasattr(nk, "__iter__") else nk

        n_main = max(na_val, nb_val + nk_val)
        B = np.zeros((n_main, n_inputs))

        if nb_val > 0 and na_val + nb_val <= len(theta):
            for i in range(min(nb_val, len(theta) - na_val)):
                # For MIMO, distribute coefficients across inputs
                for j in range(n_inputs):
                    coeff_idx = na_val + i * n_inputs + j
                    if coeff_idx < len(theta):
                        B[max(0, na_val - 1 - i), j] = theta[coeff_idx]

        return B

    def _create_transfer_functions_ararmax(self, theta, na, nb, nc, nd, nf, nk, Ts):
        """
        Create G_tf and H_tf transfer functions for ARARMAX.

        For ARARMAX: G_tf = B(q)/F(q), H_tf = C(q)/D(q).

        Parameters:
        -----------
        theta : ndarray
            Estimated parameters [AR; Input; Noise_AR; Noise_MA]
        na, nb, nc, nd, nf, nk : int or list
            ARARMAX polynomial orders and delay
        Ts : float
            Sampling time

        Returns:
        --------
        G_tf, H_tf : control.TransferFunction
            Deterministic and noise transfer functions.
        """
        na_val = max(na) if isinstance(na, (list, tuple)) else na
        nb_val = max(nb) if isinstance(nb, (list, tuple)) else nb
        nc_val = max(nc) if isinstance(nc, (list, tuple)) else nc
        nd_val = max(nd) if isinstance(nd, (list, tuple)) else nd
        nk_val = max(nk) if isinstance(nk, (list, tuple)) else nk

        polynomial_length_g = max(nb_val + nk_val, na_val + 1)

        numerator_g = np.zeros(polynomial_length_g)
        if len(theta) >= na_val + nb_val:
            numerator_g[nk_val : nk_val + nb_val] = theta[na_val : na_val + nb_val]

        denominator_g = np.zeros(polynomial_length_g)
        denominator_g[0] = 1.0
        if na_val > 0:
            denominator_g[1 : na_val + 1] = theta[:na_val]

        max_order_h = max(nc_val, nd_val)
        numerator_h = np.zeros(max_order_h + 1)
        numerator_h[0] = 1.0
        if len(theta) >= na_val + nb_val + nc_val:
            numerator_h[1 : nc_val + 1] = theta[
                na_val + nb_val : na_val + nb_val + nc_val
            ]

        denominator_h = np.zeros(max_order_h + 1)
        denominator_h[0] = 1.0
        if len(theta) >= na_val + nb_val + nc_val + nd_val:
            denominator_h[1 : nd_val + 1] = theta[
                na_val + nb_val + nc_val : na_val + nb_val + nc_val + nd_val
            ]

        return (
            control.tf(numerator_g, denominator_g, dt=Ts),
            control.tf(numerator_h, denominator_h, dt=Ts),
        )

    def _compute_yid_ararmax(self, u, y, theta, na, nb, nc, nd, nf, nk):
        """
        Compute one-step-ahead predictions (Yid) for ARARMAX model.

        Parameters:
        -----------
        u, y : ndarray
            Input and output data
        theta : ndarray
            Estimated parameters
        na, nb, nc, nd, nf, nk : int or list
            ARARMAX polynomial orders and delay

        Returns:
        --------
        Yid : ndarray
            One-step-ahead predictions
        """
        # Extract scalar orders
        na_val = max(na) if isinstance(na, (list, tuple)) else na
        nb_val = max(nb) if isinstance(nb, (list, tuple)) else nb
        nc_val = max(nc) if isinstance(nc, (list, tuple)) else nc
        nd_val = max(nd) if isinstance(nd, (list, tuple)) else nd
        nf_val = max(nf) if isinstance(nf, (list, tuple)) else nf
        nk_val = max(nk) if isinstance(nk, (list, tuple)) else nk

        n_samples = len(u)
        n_outputs = y.shape[1] if len(y.shape) > 1 else 1
        n_inputs = u.shape[1] if len(u.shape) > 1 else 1

        # Initialize Yid with actual outputs
        Yid = np.zeros_like(y)
        max_order = max(na_val, nb_val + nk_val, nc_val, nd_val, nf_val)

        # Copy initial values (can't predict without history)
        Yid[:max_order] = y[:max_order]

        # Extract coefficients from theta
        # theta structure: [AR (na); Input (nb); Noise_AR (nc); Noise_MA (nd)]
        ar_coeffs = theta[:na_val] if len(theta) >= na_val else np.zeros(na_val)
        input_coeffs = (
            theta[na_val : na_val + nb_val]
            if len(theta) >= na_val + nb_val
            else np.zeros(nb_val)
        )

        # Compute predictions for each time step
        for k in range(max_order, n_samples):
            y_pred = 0.0

            # AR part: sum of past outputs
            for i in range(na_val):
                if k - i - 1 >= 0 and i < len(ar_coeffs):
                    y_past = y[k - i - 1][0] if n_outputs > 1 else y[k - i - 1]
                    y_pred += ar_coeffs[i] * y_past

            # Input part: sum of delayed inputs
            for i in range(nb_val):
                if k - nk_val - i >= 0 and i < len(input_coeffs):
                    u_past = u[k - nk_val - i][0] if n_inputs > 1 else u[k - nk_val - i]
                    y_pred += input_coeffs[i] * u_past

            # Store prediction
            if n_outputs > 1:
                Yid[k, 0] = y_pred
            else:
                Yid[k] = y_pred

        return Yid

    def _create_fallback_armax_model(self, u, y, config):
        """Create fallback ARARMAX model using ARX as base."""
        # Use ARX as fallback with extended noise handling
        from .arx import ARXAlgorithm

        arx_algo = ARXAlgorithm()
        arx_config = config.copy()
        arx_config.method = "ARX"
        # Handle both int and list parameters for fallback
        nb_val = config.nb
        nk_val = config.nk
        na_val = config.na
        arx_config.na = na_val

        # Convert to single integers if they're lists
        if hasattr(nb_val, "__len__"):
            nb_val = nb_val[0]
        if hasattr(nk_val, "__len__"):
            nk_val = nk_val[0]
        if hasattr(na_val, "__len__"):
            na_val = na_val[0]

        arx_config.nb = nb_val
        arx_config.nk = nk_val
        arx_config.na = na_val

        # Use original u, y arrays for fallback
        arx_result = arx_algo.identify(y=y, u=u, config=arx_config)
        return arx_result
