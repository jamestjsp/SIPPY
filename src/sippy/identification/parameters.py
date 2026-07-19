"""Shared identification method and option vocabulary."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np

METHOD_ALIASES = {"FREQUENCY_DOMAIN": "FD", "FREQ_DOMAIN": "FD"}

COMMON_OPTIONS = {"tsample"}
ALGORITHM_OPTIONS = {
    "N4SID": {
        "ss_f",
        "ss_threshold",
        "ss_fixed_order",
        "ss_d_required",
        "ss_a_stability",
    },
    "MOESP": {
        "ss_f",
        "ss_threshold",
        "ss_fixed_order",
        "ss_d_required",
        "ss_a_stability",
    },
    "CVA": {
        "ss_f",
        "ss_threshold",
        "ss_fixed_order",
        "ss_d_required",
        "ss_a_stability",
    },
    "PARSIM-K": {
        "ss_f",
        "ss_p",
        "ss_threshold",
        "ss_fixed_order",
        "ss_d_required",
        "ss_pk_b_reval",
    },
    "PARSIM-S": {"ss_f", "ss_p", "ss_threshold", "ss_fixed_order", "ss_d_required"},
    "PARSIM-P": {"ss_f", "ss_p", "ss_threshold", "ss_fixed_order", "ss_d_required"},
    "ARX": {"na", "nb", "nk"},
    "FIR": {"nb", "nk"},
    "ARMAX": {
        "na",
        "nb",
        "nc",
        "nk",
        "max_iterations",
        "mode",
        "stability_margin",
        "stability_constraint",
    },
    "ARMA": {"na", "nc", "max_iterations", "stability_margin", "stability_constraint"},
    "ARARX": {
        "na",
        "nb",
        "nd",
        "nk",
        "max_iterations",
        "stability_margin",
        "stability_constraint",
    },
    "ARARMAX": {
        "na",
        "nb",
        "nc",
        "nd",
        "nf",
        "nk",
        "max_iterations",
        "stability_margin",
        "stability_constraint",
    },
    "OE": {
        "nb",
        "nf",
        "nk",
        "max_iterations",
        "stability_margin",
        "stability_constraint",
    },
    "BJ": {
        "nb",
        "nc",
        "nd",
        "nf",
        "nk",
        "max_iterations",
        "stability_margin",
        "stability_constraint",
    },
    "GEN": {
        "na",
        "nb",
        "nc",
        "nd",
        "nf",
        "nk",
        "max_iterations",
        "stability_margin",
        "stability_constraint",
    },
    "FD": {
        "fd_method",
        "smoothing_window",
        "coherence_threshold",
        "window_type",
        "lag_window",
        "max_lag",
        "nperseg",
        "noverlap",
        "welch_window",
        "remove_mean",
    },
}

OPTION_ALIASES = {
    "dt": "tsample",
    "stab_marg": "stability_margin",
    "stab_cons": "stability_constraint",
    "algorithm": "mode",
    "theta": "nk",
}


def normalize_method(method: str) -> str:
    if not isinstance(method, str) or not method.strip():
        raise ValueError("method must be a non-empty string")
    normalized = method.strip().upper()
    return METHOD_ALIASES.get(normalized, normalized)


def _alias_value(alias: str, value: Any) -> Any:
    if alias != "theta":
        return value
    delay = np.asarray(value)
    if np.any(delay < 0):
        raise ValueError("Input delay theta must be non-negative")
    converted = delay + 1
    if converted.ndim == 0:
        return converted.item()
    return converted.tolist() if isinstance(value, list) else converted


def _values_equal(first: Any, second: Any) -> bool:
    try:
        return bool(np.array_equal(first, second))
    except (TypeError, ValueError):
        return first == second


def normalize_identification_options(
    method: str,
    options: Mapping[str, Any],
    *,
    warn_unknown: bool = True,
    warn_deprecated: bool = True,
) -> dict[str, Any]:
    """Translate deprecated aliases and retain options declared by a method."""
    normalized_method = normalize_method(method)
    accepted = ALGORITHM_OPTIONS.get(normalized_method)
    if accepted is None:
        return dict(options)
    accepted = accepted | COMMON_OPTIONS
    normalized: dict[str, Any] = {}
    unknown: list[str] = []
    for name, value in options.items():
        canonical = OPTION_ALIASES.get(name, name)
        if name in OPTION_ALIASES:
            if warn_deprecated:
                warnings.warn(
                    f"'{name}' is deprecated; use '{canonical}'",
                    DeprecationWarning,
                    stacklevel=3,
                )
            value = _alias_value(name, value)
        if canonical not in accepted:
            unknown.append(name)
            continue
        if value is None:
            continue
        if canonical in normalized and not _values_equal(normalized[canonical], value):
            raise ValueError(
                f"Conflicting values were provided for identification option '{canonical}'"
            )
        normalized[canonical] = value

    if warn_unknown and unknown:
        names = ", ".join(sorted(unknown))
        warnings.warn(
            f"Unknown {normalized_method} identification option(s): {names}",
            UserWarning,
            stacklevel=3,
        )

    mode = normalized.get("mode")
    if isinstance(mode, str):
        mode = mode.upper()
        mode_aliases = {"RLLS": "ILLS", "OPT": "NLP"}
        if mode in mode_aliases:
            replacement = mode_aliases[mode]
            warnings.warn(
                f"ARMAX mode '{mode}' is deprecated; use '{replacement}'",
                DeprecationWarning,
                stacklevel=3,
            )
            mode = replacement
        normalized["mode"] = mode
    return normalized
