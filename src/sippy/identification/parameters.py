"""Shared identification method and option vocabulary."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np

METHOD_ALIASES = {"FREQUENCY_DOMAIN": "FD", "FREQ_DOMAIN": "FD"}

COMMON_OPTIONS = {"tsample"}
ALGORITHM_OPTIONS = {
    "SUBSPACE": {
        "reference",
        "ss_f",
        "ss_fixed_order",
        "ss_d_required",
        "ss_weighting",
        "criterion",
        "validation_fraction",
    },
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
    "FIR": {"nb", "nk", "regularization"},
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

ORDER_BUNDLES = {
    "ARX": ("arx_orders", ("na", "nb", "theta")),
    "ARMAX": ("armax_orders", ("na", "nb", "nc", "theta")),
    "ARARX": ("ararx_orders", ("na", "nb", "nd", "theta")),
    "ARARMAX": (
        "ararmax_orders",
        ("na", "nb", "nc", "nd", "theta"),
    ),
    "BJ": ("bj_orders", ("nb", "nc", "nd", "nf", "theta")),
}

ORDER_BUNDLE_ALIASES = {
    "ARX_orders": "arx_orders",
    "ARMAX_orders": "armax_orders",
    "ARARX_orders": "ararx_orders",
    "ARARMAX_orders": "ararmax_orders",
    "BJ_orders": "bj_orders",
}

_NUMBER_WORDS = {3: "three", 4: "four", 5: "five"}


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


def _set_option(
    normalized: dict[str, Any], canonical: str, value: Any, *, source: str
) -> None:
    if canonical in normalized and not _values_equal(normalized[canonical], value):
        raise ValueError(
            f"Conflicting values were provided for identification option "
            f"'{canonical}' through {source}"
        )
    normalized[canonical] = value


def _expand_order_bundle(
    method: str,
    name: str,
    value: Any,
    *,
    warn_deprecated: bool,
) -> dict[str, Any] | None:
    specification = ORDER_BUNDLES.get(method)
    canonical_name = ORDER_BUNDLE_ALIASES.get(name, name)
    if specification is None or canonical_name != specification[0]:
        return None
    fields = specification[1]
    if isinstance(value, (str, bytes, Mapping)) or not hasattr(value, "__len__"):
        length = None
    else:
        length = len(value)
    if length != len(fields):
        count = _NUMBER_WORDS.get(len(fields), str(len(fields)))
        replacements = ", ".join(
            "nk" if field == "theta" else field for field in fields
        )
        raise ValueError(
            f"{canonical_name} must contain {count} values "
            f"[{', '.join(fields)}]; pass {replacements} as named options instead"
        )
    if warn_deprecated:
        warnings.warn(
            f"'{name}' is deprecated; pass named identification options instead",
            DeprecationWarning,
            stacklevel=4,
        )
    expanded = {}
    for field, field_value in zip(fields, value):
        canonical = "nk" if field == "theta" else field
        expanded[canonical] = _alias_value(field, field_value)
    return expanded


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
        if value is None:
            continue
        bundle = _expand_order_bundle(
            normalized_method,
            name,
            value,
            warn_deprecated=warn_deprecated,
        )
        if bundle is not None:
            for canonical, bundle_value in bundle.items():
                _set_option(
                    normalized,
                    canonical,
                    bundle_value,
                    source=name,
                )
            continue
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
        _set_option(normalized, canonical, value, source=name)

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
