"""
Fit a parametric transfer function to a non-parametric FRF estimate.

Closes the loop from the frequency-domain (FD) identification back to a
parametric model: the complex frequency response estimated by the FD
algorithm is fitted with a rational discrete-time transfer function

    G(z) = z^-nk * (b_0 + b_1 z^-1 + ... + b_{nb-1} z^-(nb-1))
           / (1 + a_1 z^-1 + ... + a_na z^-na)

using coherence-weighted least squares with Sanathanan-Koerner iterations
(iteratively reweighting by 1/|A(e^jw)| removes the bias of the plain Levy
linearization). Bins are weighted by sqrt(coh / (1 - coh)), the inverse
standard deviation of the FRF estimate, so unreliable frequencies barely
influence the fit.

References:
    - Sanathanan & Koerner (1963). Transfer function synthesis as a ratio
      of two complex polynomials.
    - Pintelon & Schoukens (2012). System Identification: A Frequency
      Domain Approach, ch. 9.
"""

import warnings
from typing import Any

import numpy as np

from sippy import systems as control

from .base import StateSpaceModel, realize_transfer_function

# nk follows the repository-wide convention: delay of the first B
# coefficient, so nk=1 means the numerator starts at u[k-1].


def fit_rational_frf(
    omega: np.ndarray,
    G: np.ndarray,
    na: int,
    nb: int,
    nk: int = 1,
    weights: np.ndarray | None = None,
    n_iter: int = 30,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Fit a SISO rational transfer function to complex FRF samples.

    Parameters
    ----------
    omega : ndarray
        Normalized frequencies (rad/sample), in (0, pi]
    G : ndarray
        Complex frequency response samples at omega
    na : int
        Denominator order (>= 0)
    nb : int
        Number of numerator coefficients (>= 1)
    nk : int
        Input delay of the first numerator coefficient (>= 0)
    weights : ndarray, optional
        Non-negative per-bin weights (default: uniform)
    n_iter : int
        Maximum Sanathanan-Koerner iterations
    tol : float
        Stop when the coefficient update is below this norm

    Returns
    -------
    b : ndarray
        Numerator coefficients [b_0, ..., b_{nb-1}] (before the z^-nk delay)
    a : ndarray
        Denominator coefficients [1, a_1, ..., a_na]
    info : dict
        n_iter, converged, weighted_rms_error, relative_error (per bin)
    """
    omega = np.asarray(omega, dtype=float)
    G = np.asarray(G, dtype=complex)
    if omega.ndim != 1 or G.ndim != 1:
        raise ValueError("omega and G must be one-dimensional")
    if not all(isinstance(order, (int, np.integer)) for order in (na, nb, nk)):
        raise ValueError("na, nb, and nk must be integers")
    if na < 0 or nb < 1 or nk < 0:
        raise ValueError("Orders must satisfy na >= 0, nb >= 1, nk >= 0")
    if len(omega) != len(G):
        raise ValueError("omega and G must have the same length")
    if not np.all(np.isfinite(omega)) or not np.all(np.isfinite(G)):
        raise ValueError("omega and G must contain only finite values")
    if len(omega) < na + nb:
        raise ValueError(
            f"Need at least na + nb = {na + nb} frequency bins to fit, got {len(omega)}"
        )
    if not isinstance(n_iter, (int, np.integer)) or n_iter < 1:
        raise ValueError("n_iter must be a positive integer")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")

    w0 = np.ones(len(omega)) if weights is None else np.asarray(weights, float)
    if w0.ndim != 1 or w0.shape != omega.shape:
        raise ValueError(
            "weights must be one-dimensional with the same length as omega"
        )
    if not np.all(np.isfinite(w0)) or np.any(w0 < 0):
        raise ValueError("weights must contain only finite, non-negative values")
    if np.count_nonzero(w0) < na + nb:
        raise ValueError(f"Need at least na + nb = {na + nb} positive weights to fit")
    w0 = w0 / np.max(w0)

    E = np.exp(-1j * omega)  # z^-1 on the unit circle
    # Design matrix: G_k = sum_i b_i E^(nk+i) - sum_i a_i E^i G_k
    numerator_terms = E[:, None] ** np.arange(nk, nk + nb)
    denominator_terms = -(E[:, None] ** np.arange(1, na + 1)) * G[:, None]
    M = np.concatenate([numerator_terms, denominator_terms], axis=1)

    x_prev = np.zeros(nb + na)
    sk_weight = np.ones(len(omega))
    converged = False
    for iteration in range(1, n_iter + 1):
        w = w0 * sk_weight
        Mw = M * w[:, None]
        gw = G * w
        M_ri = np.vstack([Mw.real, Mw.imag])
        g_ri = np.concatenate([gw.real, gw.imag])
        x, *_ = np.linalg.lstsq(M_ri, g_ri, rcond=None)

        if np.linalg.norm(x - x_prev) < tol * max(np.linalg.norm(x), 1.0):
            converged = True
            x_prev = x
            break
        x_prev = x

        a_poly = np.concatenate([[1.0], x[nb:]])
        # |1 + a_1 E + ... + a_na E^na| evaluated on the unit circle
        A_eval = np.abs(np.polyval(a_poly[::-1], E))
        sk_weight = 1.0 / np.maximum(A_eval, 1e-12)

    b = x_prev[:nb]
    a = np.concatenate([[1.0], x_prev[nb:]])

    A_fit = np.polyval(a[::-1], E)
    B_fit = E**nk * np.polyval(b[::-1], E)
    G_fit = B_fit / A_fit
    residual = G_fit - G
    scale = np.maximum(np.abs(G), 1e-12)
    squared_weights = w0**2
    info = {
        "n_iter": iteration,
        "converged": converged,
        "weighted_rms_error": float(
            np.sqrt(
                np.sum(squared_weights * np.abs(residual) ** 2)
                / np.sum(squared_weights)
            )
        ),
        "relative_error": np.abs(residual) / scale,
    }
    return b, a, info


def _channel_polynomials(
    b: np.ndarray, a: np.ndarray, nk: int
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-length z-domain polynomial pair for one fitted channel."""
    na = len(a) - 1
    nb = len(b)
    length = max(na, nb + nk - 1) + 1
    den = np.zeros(length)
    den[: na + 1] = a
    num = np.zeros(length)
    num[nk : nk + nb] = b
    return num, den


def fit_frf_model(
    model: StateSpaceModel,
    na: int,
    nb: int,
    nk: int = 1,
    min_coherence: float = 0.1,
    n_iter: int = 30,
) -> StateSpaceModel:
    """
    Fit a parametric state-space model to an FD identification result.

    Each output-input channel of the non-parametric frequency response in
    ``model.identification_info`` is fitted with a rational transfer
    function (see :func:`fit_rational_frf`); the resulting transfer matrix
    is realized as a state-space model via python-control and Slycot.

    Parameters
    ----------
    model : StateSpaceModel
        Result of the FD algorithm (correlation or welch estimator)
    na : int
        Denominator order for every channel
    nb : int
        Numerator length for every channel
    nk : int
        Input delay of the first numerator coefficient (repo convention:
        nk=1 means the response starts at u[k-1])
    min_coherence : float
        Bins with coherence below this are excluded from the fit
    n_iter : int
        Maximum Sanathanan-Koerner iterations per channel

    Returns
    -------
    StateSpaceModel
        Parametric model with real A, B, C, D matrices, ``G_tf`` set to the
        fitted control transfer function, and per-channel fit diagnostics in
        ``identification_info["frf_fit"]``.
    """
    if not np.isfinite(min_coherence) or not 0 <= min_coherence <= 1:
        raise ValueError("min_coherence must be in [0, 1]")

    info = getattr(model, "identification_info", None) or {}
    if info.get("method") != "FD" or "frequency_response" not in info:
        raise ValueError(
            "fit_frf_model expects the result of the FD (frequency-domain) "
            "algorithm; got a model without frequency response data"
        )

    fr = info["frequency_response"]
    omega = np.asarray(fr["omega"])
    n_outputs = info["n_outputs"]
    n_inputs = info["n_inputs"]

    if info["estimator"] == "correlation":
        G = fr["G_smooth"][:, None, None]
        coherence = fr["coherence"][:, None]
    else:
        G = fr["G"]
        coherence = fr["coherence"]

    positive = (omega > 0) & (omega < np.pi)
    fits: list[list[dict[str, Any]]] = []
    num_rows: list[list[np.ndarray]] = []
    den_rows: list[list[np.ndarray]] = []
    residuals = []
    for out in range(n_outputs):
        coh = coherence[positive, out]
        usable = coh >= min_coherence
        if usable.sum() < na + nb:
            raise ValueError(
                f"Only {int(usable.sum())} bins of output {out} have "
                f"coherence >= {min_coherence}; not enough to fit "
                f"na + nb = {na + nb} coefficients. Lower min_coherence or "
                "collect better-excited data."
            )
        om = omega[positive][usable]
        weight = np.sqrt(coh[usable] / (1.0 - coh[usable] + 1e-6))

        fit_row = []
        num_row = []
        den_row = []
        for inp in range(n_inputs):
            b, a, fit_info = fit_rational_frf(
                om,
                G[positive, out, inp][usable],
                na=na,
                nb=nb,
                nk=nk,
                weights=weight,
                n_iter=n_iter,
            )
            poles = np.roots(a) if len(a) > 1 else np.array([])
            if len(poles) and np.max(np.abs(poles)) >= 1.0:
                warnings.warn(
                    f"Fitted channel ({out}, {inp}) is unstable "
                    f"(max |pole| = {np.max(np.abs(poles)):.3f}); consider "
                    "different orders or a higher min_coherence."
                )
            fit_info = dict(fit_info)
            fit_info["relative_error"] = float(np.median(fit_info["relative_error"]))
            fit_info.update({"b": b, "a": a, "poles": poles})
            fit_row.append(fit_info)
            residuals.append(fit_info["weighted_rms_error"] ** 2)

            num, den = _channel_polynomials(b, a, nk)
            num_trim = np.trim_zeros(num, "f")
            num_row.append(num_trim if len(num_trim) else np.array([0.0]))
            den_row.append(den)
        fits.append(fit_row)
        num_rows.append(num_row)
        den_rows.append(den_row)

    if n_outputs == 1 and n_inputs == 1:
        G_tf = control.tf(num_rows[0][0], den_rows[0][0], dt=model.ts)
    else:
        G_tf = control.tf(num_rows, den_rows, dt=model.ts)

    A, B, C, D = realize_transfer_function(G_tf)
    A = np.atleast_2d(A)
    n_states = A.shape[0]
    B = np.asarray(B).reshape(n_states, n_inputs)
    C = np.asarray(C).reshape(n_outputs, n_states)
    D = np.asarray(D).reshape(n_outputs, n_inputs)

    return StateSpaceModel(
        A=A,
        B=B,
        C=C,
        D=D,
        K=np.zeros((n_states, n_outputs)),
        Q=np.eye(n_states),
        R=np.eye(n_outputs),
        S=np.zeros((n_states, n_outputs)),
        ts=model.ts,
        Vn=float(np.mean(residuals)),
        G_tf=G_tf,
        identification_info={
            "method": "FD-FIT",
            "source_estimator": info["estimator"],
            "orders": {"na": na, "nb": nb, "nk": nk},
            "min_coherence": min_coherence,
            "frf_fit": fits,
        },
    )
