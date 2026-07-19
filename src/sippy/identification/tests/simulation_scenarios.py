from dataclasses import dataclass
from typing import Literal

import harold
import numpy as np

ExcitationKind = Literal["white", "binary", "multisine"]


@dataclass(frozen=True)
class IdentificationScenario:
    plant: harold.State
    sample_time: float
    u_train: np.ndarray
    y_train_clean: np.ndarray
    noise_train: np.ndarray
    y_train: np.ndarray
    u_validation: np.ndarray
    y_validation_clean: np.ndarray
    noise_validation: np.ndarray
    y_validation: np.ndarray
    snr_db: float


def stable_siso_plant(
    dt: float = 1.0, direct_feedthrough: float = 0.08
) -> harold.State:
    return harold.State(
        np.array([[0.72, 0.12], [-0.08, 0.84]]),
        np.array([[0.35], [0.18]]),
        np.array([[1.0, -0.25]]),
        np.array([[direct_feedthrough]]),
        dt=dt,
    )


def stable_mimo_plant(dt: float = 1.0, direct_feedthrough: bool = True) -> harold.State:
    return harold.State(
        np.array(
            [
                [0.72, 0.08, 0.0],
                [-0.04, 0.81, 0.06],
                [0.02, -0.05, 0.66],
            ]
        ),
        np.array([[0.35, 0.08], [0.12, 0.28], [0.18, -0.15]]),
        np.array([[1.0, 0.2, -0.1], [0.15, -0.25, 0.9]]),
        (
            np.array([[0.05, 0.0], [0.0, -0.03]])
            if direct_feedthrough
            else np.zeros((2, 2))
        ),
        dt=dt,
    )


def unstable_siso_plant(dt: float = 1.0) -> harold.State:
    return harold.State(
        np.array([[1.015, 0.0], [0.0, 0.93]]),
        np.array([[0.06], [0.2]]),
        np.array([[0.7, 0.3]]),
        np.array([[0.0]]),
        dt=dt,
    )


def delayed_siso_plant(delay: int = 12, dt: float = 1.0) -> harold.State:
    if delay < 1:
        raise ValueError("delay must be at least one sample")

    order = delay + 1
    a = np.zeros((order, order))
    a[0, 0] = 0.78
    a[0, -1] = 0.4
    if delay > 1:
        a[2:, 1:-1] = np.eye(delay - 1)
    b = np.zeros((order, 1))
    b[1, 0] = 1.0
    c = np.zeros((1, order))
    c[0, 0] = 1.0
    d = np.zeros((1, 1))
    return harold.State(a, b, c, d, dt=dt)


def stable_arma_noise_filter(dt: float = 1.0) -> harold.Transfer:
    return harold.Transfer(
        np.array([1.0, 0.35]),
        np.array([1.0, -1.15, 0.32]),
        dt=dt,
    )


def simulate_noise_process(
    noise_filter: harold.Transfer,
    *,
    n_samples: int,
    seed: int,
    burn_in: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    innovations = rng.standard_normal(n_samples + burn_in)
    output, _ = harold.simulate_linear_system(noise_filter, innovations[:, np.newaxis])
    return innovations[burn_in:], np.asarray(output).reshape(-1)[burn_in:]


def _correlation_factor(size: int, correlation: float) -> np.ndarray:
    if size < 1:
        raise ValueError("channel count must be positive")
    if size == 1:
        if correlation != 0:
            raise ValueError("correlation requires at least two channels")
        return np.ones((1, 1))

    lower_bound = -1.0 / (size - 1)
    if not lower_bound < correlation < 1.0:
        raise ValueError(
            f"correlation must be between {lower_bound:g} and 1 for {size} channels"
        )
    matrix = np.full((size, size), correlation)
    np.fill_diagonal(matrix, 1.0)
    return np.linalg.cholesky(matrix)


def generate_excitation(
    n_inputs: int,
    n_samples: int,
    *,
    kind: ExcitationKind = "white",
    correlation: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    if n_samples < 2:
        raise ValueError("n_samples must be at least two")

    rng = np.random.default_rng(seed)
    if kind == "white":
        independent = rng.standard_normal((n_inputs, n_samples))
    elif kind == "binary":
        independent = np.where(
            rng.standard_normal((n_inputs, n_samples)) >= 0, 1.0, -1.0
        )
    elif kind == "multisine":
        time = np.arange(n_samples)
        independent = np.zeros((n_inputs, n_samples))
        max_frequency = max(3, min(24, n_samples // 8))
        for channel in range(n_inputs):
            frequencies = rng.choice(
                np.arange(1, max_frequency + 1),
                size=min(8, max_frequency),
                replace=False,
            )
            phases = rng.uniform(0, 2 * np.pi, size=frequencies.size)
            for frequency, phase in zip(frequencies, phases, strict=True):
                independent[channel] += np.sin(
                    2 * np.pi * frequency * time / n_samples + phase
                )
    else:
        raise ValueError(f"unsupported excitation kind: {kind}")

    excitation = _correlation_factor(n_inputs, correlation) @ independent
    excitation -= np.mean(excitation, axis=1, keepdims=True)
    scale = np.std(excitation, axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("excitation has zero variance")
    return excitation / scale


def _simulate(plant: harold.State, inputs: np.ndarray) -> np.ndarray:
    output, _ = harold.simulate_linear_system(plant, inputs.T)
    output = np.asarray(output, dtype=float)
    if output.ndim == 1:
        output = output[:, np.newaxis]
    return output.T


def _noise_for_snr(
    clean_output: np.ndarray,
    snr_db: float,
    *,
    temporal_correlation: float,
    channel_correlation: float,
    seed: int,
) -> np.ndarray:
    if not -1 < temporal_correlation < 1:
        raise ValueError("noise_color must be between -1 and 1")

    n_outputs, n_samples = clean_output.shape
    rng = np.random.default_rng(seed)
    innovations = _correlation_factor(
        n_outputs, channel_correlation
    ) @ rng.standard_normal((n_outputs, n_samples))
    noise = np.empty_like(innovations)
    noise[:, 0] = innovations[:, 0]
    innovation_scale = np.sqrt(1 - temporal_correlation**2)
    for sample in range(1, n_samples):
        noise[:, sample] = (
            temporal_correlation * noise[:, sample - 1]
            + innovation_scale * innovations[:, sample]
        )

    noise -= np.mean(noise, axis=1, keepdims=True)
    noise_power = np.mean(noise**2, axis=1, keepdims=True)
    signal_power = np.mean(clean_output**2, axis=1, keepdims=True)
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    return noise * np.sqrt(target_noise_power / noise_power)


def simulate_scenario(
    plant: harold.State,
    *,
    n_train: int,
    n_validation: int,
    input_kind: ExcitationKind,
    snr_db: float,
    input_correlation: float = 0.0,
    noise_correlation: float = 0.0,
    noise_color: float = 0.0,
    seed: int = 0,
) -> IdentificationScenario:
    n_inputs = plant.shape[1]
    u_train = generate_excitation(
        n_inputs,
        n_train,
        kind=input_kind,
        correlation=input_correlation,
        seed=seed,
    )
    u_validation = generate_excitation(
        n_inputs,
        n_validation,
        kind=input_kind,
        correlation=input_correlation,
        seed=seed + 1,
    )
    y_train_clean = _simulate(plant, u_train)
    y_validation_clean = _simulate(plant, u_validation)
    noise_train = _noise_for_snr(
        y_train_clean,
        snr_db,
        temporal_correlation=noise_color,
        channel_correlation=noise_correlation,
        seed=seed + 2,
    )
    noise_validation = _noise_for_snr(
        y_validation_clean,
        snr_db,
        temporal_correlation=noise_color,
        channel_correlation=noise_correlation,
        seed=seed + 3,
    )
    return IdentificationScenario(
        plant=plant,
        sample_time=float(plant.SamplingPeriod),
        u_train=u_train,
        y_train_clean=y_train_clean,
        noise_train=noise_train,
        y_train=y_train_clean + noise_train,
        u_validation=u_validation,
        y_validation_clean=y_validation_clean,
        noise_validation=noise_validation,
        y_validation=y_validation_clean + noise_validation,
        snr_db=snr_db,
    )


def normalized_rmse(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    actual = np.atleast_2d(np.asarray(actual, dtype=float))
    predicted = np.atleast_2d(np.asarray(predicted, dtype=float))
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted outputs must have matching shapes")

    error_norm = np.linalg.norm(actual - predicted, axis=1)
    centered_norm = np.linalg.norm(
        actual - np.mean(actual, axis=1, keepdims=True), axis=1
    )
    return np.divide(
        error_norm,
        centered_norm,
        out=np.full_like(error_norm, np.inf),
        where=centered_norm > np.finfo(float).eps,
    )


def simulate_identified_model(model: object, inputs: np.ndarray) -> np.ndarray:
    _, output = model.simulate(inputs)
    return np.atleast_2d(np.asarray(output, dtype=float))
