from dataclasses import dataclass
from typing import Literal

import numpy as np

from sippy import systems as control

ExcitationKind = Literal["white", "binary", "multisine"]


@dataclass(frozen=True)
class IdentificationScenario:
    plant: control.StateSpace
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


@dataclass(frozen=True)
class ClosedLoopTrajectory:
    plant: control.StateSpace
    controller: control.StateSpace
    sample_time: float
    reference: np.ndarray
    dither: np.ndarray
    disturbance: np.ndarray
    plant_input: np.ndarray
    plant_output: np.ndarray
    output: np.ndarray
    plant_states: np.ndarray
    controller_states: np.ndarray


@dataclass(frozen=True)
class ClosedLoopIdentificationScenario(ClosedLoopTrajectory):
    innovations: np.ndarray
    u_validation: np.ndarray
    y_validation_clean: np.ndarray
    excitation_order: int
    excitation_rank: int


def stable_siso_plant(
    dt: float = 1.0, direct_feedthrough: float = 0.08
) -> control.StateSpace:
    return control.ss(
        np.array([[0.72, 0.12], [-0.08, 0.84]]),
        np.array([[0.35], [0.18]]),
        np.array([[1.0, -0.25]]),
        np.array([[direct_feedthrough]]),
        dt=dt,
    )


def stable_mimo_plant(
    dt: float = 1.0, direct_feedthrough: bool = True
) -> control.StateSpace:
    return control.ss(
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


def unstable_siso_plant(dt: float = 1.0) -> control.StateSpace:
    return control.ss(
        np.array([[1.015, 0.0], [0.0, 0.93]]),
        np.array([[0.06], [0.2]]),
        np.array([[0.7, 0.3]]),
        np.array([[0.0]]),
        dt=dt,
    )


def delayed_siso_plant(delay: int = 12, dt: float = 1.0) -> control.StateSpace:
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
    return control.ss(a, b, c, d, dt=dt)


def stable_arma_noise_filter(dt: float = 1.0) -> control.TransferFunction:
    # Biproper H = (1 + 0.35 q^-1) / (1 - 1.15 q^-1 + 0.32 q^-2): the ARMA
    # innovation model includes the contemporaneous e(k) term, so the truth
    # must too (a strictly proper H is outside the ARMA model class).
    return control.tf(
        np.array([1.0, 0.35, 0.0]),
        np.array([1.0, -1.15, 0.32]),
        dt=dt,
    )


def simulate_noise_process(
    noise_filter: control.TransferFunction,
    *,
    n_samples: int,
    seed: int,
    burn_in: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    innovations = rng.standard_normal(n_samples + burn_in)
    output = control.forced_response(noise_filter, U=innovations).outputs
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


def static_output_feedback_controller(
    gain: object, *, dt: float = 1.0
) -> control.StateSpace:
    matrix = np.atleast_2d(np.asarray(gain, dtype=float))
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("controller gain must be a nonempty matrix")
    n_inputs = matrix.shape[1]
    n_outputs = matrix.shape[0]
    return control.ss(
        np.empty((0, 0)),
        np.empty((0, n_inputs)),
        np.empty((n_outputs, 0)),
        matrix,
        dt=dt,
    )


def _discrete_sample_time(system: control.StateSpace, name: str) -> float:
    if system.dt is None or system.dt == 0:
        raise ValueError(f"{name} must be discrete-time")
    return 1.0 if system.dt is True else float(system.dt)


def _signal_matrix(
    value: object,
    *,
    channels: int,
    samples: int | None,
    name: str,
) -> np.ndarray:
    signal = np.asarray(value, dtype=float)
    if signal.ndim == 1:
        if channels != 1:
            raise ValueError(f"{name} must have {channels} channels")
        signal = signal.reshape(1, -1)
    elif signal.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional")
    if signal.shape[0] != channels:
        if signal.shape[1] == channels and (
            samples is None or signal.shape[0] == samples
        ):
            signal = signal.T
        else:
            raise ValueError(f"{name} must have {channels} channels")
    if samples is not None and signal.shape[1] != samples:
        raise ValueError(f"{name} must have {samples} samples")
    if not np.all(np.isfinite(signal)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(signal, dtype=float, order="F", copy=True)


def _initial_state(value: object | None, size: int, name: str) -> np.ndarray:
    if value is None:
        return np.zeros(size)
    state = np.asarray(value, dtype=float).reshape(-1)
    if state.size != size:
        raise ValueError(f"{name} dimension does not match the system")
    if not np.all(np.isfinite(state)):
        raise ValueError(f"{name} must contain only finite values")
    return state.copy()


def simulate_closed_loop(
    plant: control.StateSpace,
    controller: control.StateSpace,
    reference: object,
    *,
    dither: object | None = None,
    disturbance: object | None = None,
    initial_plant_state: object | None = None,
    initial_controller_state: object | None = None,
) -> ClosedLoopTrajectory:
    if controller.ninputs != plant.noutputs:
        raise ValueError("controller input count must match plant outputs")
    if controller.noutputs != plant.ninputs:
        raise ValueError("controller output count must match plant inputs")

    sample_time = _discrete_sample_time(plant, "plant")
    controller_sample_time = _discrete_sample_time(controller, "controller")
    if not np.isclose(sample_time, controller_sample_time):
        raise ValueError("plant and controller sample times must match")

    reference_array = _signal_matrix(
        reference,
        channels=plant.noutputs,
        samples=None,
        name="reference",
    )
    n_samples = reference_array.shape[1]
    dither_array = (
        np.zeros((plant.ninputs, n_samples), order="F")
        if dither is None
        else _signal_matrix(
            dither,
            channels=plant.ninputs,
            samples=n_samples,
            name="dither",
        )
    )
    disturbance_array = (
        np.zeros((plant.noutputs, n_samples), order="F")
        if disturbance is None
        else _signal_matrix(
            disturbance,
            channels=plant.noutputs,
            samples=n_samples,
            name="disturbance",
        )
    )

    loop_matrix = np.eye(plant.noutputs) + plant.D @ controller.D
    singular_values = np.linalg.svd(loop_matrix, compute_uv=False)
    tolerance = (
        max(loop_matrix.shape)
        * np.finfo(float).eps
        * max(float(singular_values[0]), 1.0)
    )
    if singular_values[-1] <= tolerance:
        raise ValueError(
            "plant and controller direct feedthrough form an algebraic loop"
        )

    plant_states = np.empty((plant.nstates, n_samples + 1), order="F")
    controller_states = np.empty((controller.nstates, n_samples + 1), order="F")
    plant_states[:, 0] = _initial_state(
        initial_plant_state, plant.nstates, "initial plant state"
    )
    controller_states[:, 0] = _initial_state(
        initial_controller_state, controller.nstates, "initial controller state"
    )
    plant_input = np.empty((plant.ninputs, n_samples), order="F")
    plant_output = np.empty((plant.noutputs, n_samples), order="F")
    output = np.empty((plant.noutputs, n_samples), order="F")

    for sample in range(n_samples):
        controller_feedforward = (
            controller.C @ controller_states[:, sample]
            + controller.D @ reference_array[:, sample]
            + dither_array[:, sample]
        )
        right_hand_side = (
            plant.C @ plant_states[:, sample]
            + plant.D @ controller_feedforward
            + disturbance_array[:, sample]
        )
        output[:, sample] = np.linalg.solve(loop_matrix, right_hand_side)
        error = reference_array[:, sample] - output[:, sample]
        plant_input[:, sample] = (
            controller.C @ controller_states[:, sample]
            + controller.D @ error
            + dither_array[:, sample]
        )
        plant_output[:, sample] = (
            plant.C @ plant_states[:, sample] + plant.D @ plant_input[:, sample]
        )
        plant_states[:, sample + 1] = (
            plant.A @ plant_states[:, sample] + plant.B @ plant_input[:, sample]
        )
        controller_states[:, sample + 1] = (
            controller.A @ controller_states[:, sample] + controller.B @ error
        )

    return ClosedLoopTrajectory(
        plant=plant,
        controller=controller,
        sample_time=sample_time,
        reference=reference_array,
        dither=dither_array,
        disturbance=disturbance_array,
        plant_input=plant_input,
        plant_output=plant_output,
        output=output,
        plant_states=plant_states,
        controller_states=controller_states,
    )


def _closed_loop_noise(
    n_outputs: int,
    n_samples: int,
    *,
    scale: float,
    temporal_correlation: float,
    channel_correlation: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if scale < 0:
        raise ValueError("noise_scale must be nonnegative")
    if not -1 < temporal_correlation < 1:
        raise ValueError("noise_color must be between -1 and 1")
    rng = np.random.default_rng(seed)
    innovations = scale * (
        _correlation_factor(n_outputs, channel_correlation)
        @ rng.standard_normal((n_outputs, n_samples))
    )
    disturbance = np.empty_like(innovations)
    if n_samples == 0:
        return innovations, disturbance
    disturbance[:, 0] = innovations[:, 0]
    innovation_scale = np.sqrt(1 - temporal_correlation**2)
    for sample in range(1, n_samples):
        disturbance[:, sample] = (
            temporal_correlation * disturbance[:, sample - 1]
            + innovation_scale * innovations[:, sample]
        )
    return innovations, disturbance


def _block_hankel_rank(signal: np.ndarray, block_rows: int) -> int:
    if block_rows < 1:
        raise ValueError("excitation order must be positive")
    column_count = signal.shape[1] - block_rows + 1
    if column_count < 1:
        return 0
    hankel = np.vstack(
        [signal[:, offset : offset + column_count] for offset in range(block_rows)]
    )
    singular_values = np.linalg.svd(hankel, compute_uv=False)
    if singular_values.size == 0:
        return 0
    tolerance = (
        max(hankel.shape) * np.finfo(float).eps * max(float(singular_values[0]), 1.0)
    )
    return int(np.count_nonzero(singular_values > tolerance))


def simulate_closed_loop_scenario(
    plant: control.StateSpace,
    controller: control.StateSpace,
    *,
    n_train: int,
    n_validation: int,
    reference_kind: ExcitationKind = "white",
    reference_correlation: float = 0.0,
    reference_scale: float = 1.0,
    noise_scale: float = 0.05,
    noise_correlation: float = 0.0,
    noise_color: float = 0.0,
    dither_scale: float = 0.0,
    dither_correlation: float = 0.0,
    excitation_order: int | None = None,
    seed: int = 0,
) -> ClosedLoopIdentificationScenario:
    if n_train < 2 or n_validation < 2:
        raise ValueError("training and validation records need at least two samples")
    if reference_scale < 0 or dither_scale < 0:
        raise ValueError("excitation scales must be nonnegative")

    reference = reference_scale * generate_excitation(
        plant.noutputs,
        n_train,
        kind=reference_kind,
        correlation=reference_correlation,
        seed=seed,
    )
    dither = (
        np.zeros((plant.ninputs, n_train))
        if dither_scale == 0
        else dither_scale
        * generate_excitation(
            plant.ninputs,
            n_train,
            kind=reference_kind,
            correlation=dither_correlation,
            seed=seed + 1,
        )
    )
    innovations, disturbance = _closed_loop_noise(
        plant.noutputs,
        n_train,
        scale=noise_scale,
        temporal_correlation=noise_color,
        channel_correlation=noise_correlation,
        seed=seed + 2,
    )
    trajectory = simulate_closed_loop(
        plant,
        controller,
        reference,
        dither=dither,
        disturbance=disturbance,
    )

    centered_external = np.vstack((reference, dither))
    centered_external -= np.mean(centered_external, axis=1, keepdims=True)
    if np.linalg.norm(centered_external) <= np.finfo(float).eps:
        raise ValueError("closed-loop experiment is not persistently exciting")

    selected_order = (
        min(max(2, plant.nstates + 1), 8)
        if excitation_order is None
        else int(excitation_order)
    )
    excitation_rank = _block_hankel_rank(trajectory.plant_input, selected_order)
    required_rank = plant.ninputs * selected_order
    if excitation_rank < required_rank:
        raise ValueError(
            "closed-loop plant input is not persistently exciting; "
            f"rank {excitation_rank}, need {required_rank}"
        )

    u_validation = generate_excitation(
        plant.ninputs,
        n_validation,
        kind="white",
        seed=seed + 3,
    )
    y_validation_clean = _simulate(plant, u_validation)
    return ClosedLoopIdentificationScenario(
        plant=trajectory.plant,
        controller=trajectory.controller,
        sample_time=trajectory.sample_time,
        reference=trajectory.reference,
        dither=trajectory.dither,
        disturbance=trajectory.disturbance,
        plant_input=trajectory.plant_input,
        plant_output=trajectory.plant_output,
        output=trajectory.output,
        plant_states=trajectory.plant_states,
        controller_states=trajectory.controller_states,
        innovations=innovations,
        u_validation=u_validation,
        y_validation_clean=y_validation_clean,
        excitation_order=selected_order,
        excitation_rank=excitation_rank,
    )


def frequency_response_error(
    reference: control.InputOutputSystem,
    candidate: control.InputOutputSystem,
    frequencies: object | None = None,
) -> float:
    if reference.shape != candidate.shape:
        raise ValueError("frequency-response models must have matching dimensions")
    if frequencies is None:
        sample_time = 1.0 if reference.dt is True else float(reference.dt)
        frequencies = np.linspace(0.0, 0.95 * np.pi / sample_time, 128)
    reference_response = control.frequency_response(reference, frequencies).frdata
    candidate_response = control.frequency_response(candidate, frequencies).frdata
    denominator = max(np.linalg.norm(reference_response), np.finfo(float).tiny)
    return float(np.linalg.norm(reference_response - candidate_response) / denominator)


def _simulate(plant: control.StateSpace, inputs: np.ndarray) -> np.ndarray:
    output = control.forced_response(plant, U=inputs, squeeze=False).outputs
    return np.asarray(output, dtype=float)


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
    plant: control.StateSpace,
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
    n_inputs = plant.ninputs
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
        sample_time=float(plant.dt),
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
