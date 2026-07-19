"""
SIPPY - Systems Identification Package for Python

New modular architecture with object-oriented identification algorithms.
"""

from .identification import (
    FrequencyResponseUncertainty,
    GBN_seq,
    IdentificationResult,
    StateSpaceModel,
    SystemIdentification,
    SystemIdentificationConfig,
    get_fir_coef,
    get_frequency_response_uncertainty,
    get_model_uncertainty,
    get_step_response,
    simulate_ss_system,
    system_identification,
    white_noise_var,
)

__all__ = [
    "SystemIdentification",
    "FrequencyResponseUncertainty",
    "system_identification",
    "SystemIdentificationConfig",
    "IdentificationResult",
    "StateSpaceModel",
    "GBN_seq",
    "white_noise_var",
    "get_fir_coef",
    "get_step_response",
    "get_frequency_response_uncertainty",
    "get_model_uncertainty",
    "simulate_ss_system",
]


def hello() -> str:
    return "Hello from sippy!"
