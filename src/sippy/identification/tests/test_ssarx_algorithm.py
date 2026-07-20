from itertools import permutations

import numpy as np
import pandas as pd
import pytest

import sippy
from sippy import systems as control
from sippy.identification.algorithms.ssarx import (
    SSARXAlgorithm,
    _estimate_varx_predictor,
)
from sippy.identification.factory import AlgorithmFactory
from sippy.identification.iddata import IDData

from .simulation_scenarios import (
    delayed_siso_plant,
    frequency_response_error,
    normalized_rmse,
    simulate_closed_loop_scenario,
    simulate_scenario,
    stable_mimo_plant,
    stable_siso_plant,
    static_output_feedback_controller,
    unstable_siso_plant,
)


def _simulate_innovations_model(A, B, C, D, K, inputs, innovations):
    state = np.zeros(A.shape[0])
    outputs = np.empty((C.shape[0], inputs.shape[1]))
    for sample in range(inputs.shape[1]):
        outputs[:, sample] = C @ state + D @ inputs[:, sample] + innovations[:, sample]
        state = A @ state + B @ inputs[:, sample] + K @ innovations[:, sample]
    return outputs


def _simulate_process(system, inputs):
    return control.forced_response(system, U=inputs, squeeze=False).outputs


def _pole_error(reference, candidate):
    expected = np.asarray(control.poles(reference), dtype=complex)
    actual = np.asarray(control.poles(candidate), dtype=complex)
    errors = [
        np.linalg.norm(expected - actual[list(order)])
        for order in permutations(range(actual.size))
    ]
    return float(min(errors) / max(np.linalg.norm(expected), np.finfo(float).tiny))


def _markov_error(reference, candidate, count=16):
    def parameters(system):
        blocks = [np.asarray(system.D, dtype=float)]
        state_power = np.eye(system.nstates)
        for _ in range(1, count):
            blocks.append(system.C @ state_power @ system.B)
            state_power = state_power @ system.A
        return np.stack(blocks)

    expected = parameters(reference)
    actual = parameters(candidate)
    return float(
        np.linalg.norm(expected - actual)
        / max(np.linalg.norm(expected), np.finfo(float).tiny)
    )


def _identified_system(scenario, *, order, horizon, varx_order, direct=False):
    model = sippy.identify(
        scenario.output,
        scenario.plant_input,
        method="SSARX",
        ss_f=horizon,
        ss_p=varx_order,
        ss_fixed_order=order,
        ss_d_required=direct,
        tsample=scenario.sample_time,
    )
    system = control.ss(model.A, model.B, model.C, model.D, dt=model.ts)
    return model, system


def test_varx_predictor_blocks_match_innovations_model_oracle():
    rng = np.random.default_rng(741)
    A = np.array([[0.72, 0.08], [-0.04, 0.81]])
    B = np.array([[0.35], [0.12]])
    C = np.array([[1.0, -0.2]])
    D = np.array([[0.06]])
    K = np.array([[0.16], [0.08]])
    inputs = rng.normal(size=(1, 20000))
    innovations = 0.2 * rng.normal(size=(1, inputs.shape[1]))
    outputs = _simulate_innovations_model(A, B, C, D, K, inputs, innovations)

    estimate = _estimate_varx_predictor(
        outputs[:, 300:],
        inputs[:, 300:],
        order=30,
        direct_feedthrough=True,
    )

    A_predictor = A - K @ C
    B_predictor = B - K @ D
    expected_input = []
    expected_output = []
    state_power = np.eye(A.shape[0])
    for _ in range(5):
        expected_input.append(C @ state_power @ B_predictor)
        expected_output.append(C @ state_power @ K)
        state_power = state_power @ A_predictor

    np.testing.assert_allclose(estimate.direct, D, atol=0.004)
    np.testing.assert_allclose(
        estimate.input_blocks[:5], np.stack(expected_input), atol=0.012
    )
    np.testing.assert_allclose(
        estimate.output_blocks[:5], np.stack(expected_output), atol=0.012
    )
    assert estimate.regressor_rank == estimate.regressor_rows


def test_ssarx_reconstructs_canonical_closed_loop_example():
    rng = np.random.default_rng(742)
    A = np.array([[1.5, -0.7], [1.0, 0.0]])
    B = np.array([[1.0], [0.0]])
    C = np.array([[1.0, 0.5]])
    D = np.zeros((1, 1))
    plant = control.ss(A, B, C, D, dt=1.0)
    sample_count = 6000
    reference = rng.normal(size=(1, sample_count))
    innovations = 0.25 * rng.normal(size=(1, sample_count))
    state = np.zeros(2)
    outputs = np.empty((1, sample_count))
    inputs = np.empty((1, sample_count))
    for sample in range(sample_count):
        outputs[:, sample] = C @ state + innovations[:, sample]
        inputs[:, sample] = reference[:, sample] - 0.5 * outputs[:, sample]
        state = A @ state + B @ inputs[:, sample]

    model = sippy.identify(
        outputs,
        inputs,
        method="SSARX",
        ss_f=12,
        ss_p=24,
        ss_fixed_order=2,
    )
    identified = control.ss(model.A, model.B, model.C, model.D, dt=model.ts)
    validation_input = rng.normal(size=(1, 800))
    expected = _simulate_process(plant, validation_input)
    actual = _simulate_process(identified, validation_input)

    assert model.method == "SSARX"
    assert model.identification_info["estimator_route"] == "ssarx"
    assert model.identification_info["selected_order"] == 2
    assert model.identification_info["future_horizon"] == 12
    assert model.identification_info["varx_order"] == 24
    assert float(np.max(normalized_rmse(expected, actual))) < 0.12
    assert frequency_response_error(plant, identified) < 0.12
    assert np.max(np.abs(control.poles(identified))) < 1.0


def test_ssarx_factory_iddata_and_failure_contract():
    assert AlgorithmFactory.is_registered("SSARX")
    rng = np.random.default_rng(743)
    inputs = rng.normal(size=1200)
    outputs = np.zeros_like(inputs)
    for sample in range(2, inputs.size):
        outputs[sample] = (
            1.25 * outputs[sample - 1]
            - 0.36 * outputs[sample - 2]
            + 0.3 * inputs[sample - 1]
            + 0.02 * rng.normal()
        )
    data = IDData(
        pd.DataFrame({"u": inputs, "y": outputs}),
        inputs=["u"],
        outputs=["y"],
        tsample=0.2,
    )

    model = sippy.identify(
        data=data,
        method="SSARX",
        ss_f=8,
        ss_p=16,
        ss_fixed_order=2,
    )

    assert model.ts == pytest.approx(0.2)
    assert model.K.shape == (2, 1)
    assert model.x0.shape == (2, 1)
    assert model.identification_info["numerical_ranks"]["varx_regressor"] > 0

    with pytest.raises(ValueError, match="ss_f must exceed ss_fixed_order"):
        sippy.identify(
            outputs,
            inputs,
            method="SSARX",
            ss_f=2,
            ss_p=8,
            ss_fixed_order=2,
        )

    with pytest.raises(RuntimeError, match="VARX regressor is rank deficient"):
        sippy.identify(
            np.ones((1, 80)),
            np.ones((1, 80)),
            method="SSARX",
            ss_f=4,
            ss_p=8,
            ss_fixed_order=1,
        )


def test_ssarx_zero_threshold_requires_an_explicit_order():
    with pytest.raises(
        ValueError,
        match="ss_threshold must be positive when ss_fixed_order is not provided",
    ):
        SSARXAlgorithm().identify(
            np.zeros(32),
            np.zeros(32),
            ss_threshold=0,
        )

    assert SSARXAlgorithm().validate_parameters(
        ss_threshold=0,
        ss_fixed_order=1,
    )


@pytest.mark.parametrize(
    "noise_scale,maximum_frequency_error",
    [(0.015, 0.15), (0.12, 0.35)],
)
def test_ssarx_closed_loop_siso_reconstruction_across_noise_levels(
    noise_scale,
    maximum_frequency_error,
):
    plant = stable_siso_plant(direct_feedthrough=0.0)
    scenario = simulate_closed_loop_scenario(
        plant,
        static_output_feedback_controller([[1.1]], dt=plant.dt),
        n_train=5000,
        n_validation=500,
        noise_scale=noise_scale,
        noise_color=0.65,
        dither_scale=0.05,
        seed=750 + int(1000 * noise_scale),
    )

    model, identified = _identified_system(
        scenario,
        order=plant.nstates,
        horizon=12,
        varx_order=24,
    )
    prediction = _simulate_process(identified, scenario.u_validation)

    assert model.identification_info["estimator_route"] == "ssarx"
    assert float(np.max(normalized_rmse(scenario.y_validation_clean, prediction))) < 0.3
    assert frequency_response_error(plant, identified) < maximum_frequency_error
    assert _pole_error(plant, identified) < 0.25
    assert _markov_error(plant, identified) < maximum_frequency_error


def test_ssarx_closed_loop_mimo_reconstructs_correlated_colored_record():
    plant = stable_mimo_plant(direct_feedthrough=False)
    controller = control.ss(
        np.diag([0.4, 0.55]),
        0.2 * np.eye(2),
        0.15 * np.eye(2),
        np.array([[0.45, 0.04], [-0.03, 0.38]]),
        dt=plant.dt,
    )
    scenario = simulate_closed_loop_scenario(
        plant,
        controller,
        n_train=7000,
        n_validation=600,
        reference_correlation=0.7,
        noise_scale=0.05,
        noise_correlation=0.5,
        noise_color=0.65,
        dither_scale=0.08,
        dither_correlation=0.3,
        seed=760,
    )

    _, identified = _identified_system(
        scenario,
        order=plant.nstates,
        horizon=12,
        varx_order=24,
    )
    prediction = _simulate_process(identified, scenario.u_validation)

    assert (
        float(np.max(normalized_rmse(scenario.y_validation_clean, prediction))) < 0.35
    )
    assert frequency_response_error(plant, identified) < 0.35
    assert _pole_error(plant, identified) < 0.3
    assert _markov_error(plant, identified) < 0.35


def test_ssarx_recovers_feedback_stabilized_unstable_plant():
    plant = unstable_siso_plant()
    scenario = simulate_closed_loop_scenario(
        plant,
        static_output_feedback_controller([[2.0]], dt=plant.dt),
        n_train=9000,
        n_validation=700,
        reference_kind="binary",
        noise_scale=0.02,
        noise_color=0.35,
        dither_scale=0.04,
        seed=770,
    )

    _, identified = _identified_system(
        scenario,
        order=plant.nstates,
        horizon=14,
        varx_order=30,
    )

    assert np.max(np.abs(control.poles(plant))) > 1.0
    assert np.max(np.abs(control.poles(identified))) > 1.0
    assert frequency_response_error(plant, identified) < 0.35
    assert _pole_error(plant, identified) < 0.2
    assert _markov_error(plant, identified) < 0.35


@pytest.mark.parametrize(
    "plant,order,horizon,varx_order,direct,maximum_error",
    [
        (stable_mimo_plant(direct_feedthrough=True), 3, 12, 24, True, 0.3),
        (delayed_siso_plant(delay=3), 4, 10, 24, False, 0.35),
    ],
    ids=["mimo-direct-feedthrough", "siso-input-delay"],
)
def test_ssarx_open_loop_structural_cases(
    plant,
    order,
    horizon,
    varx_order,
    direct,
    maximum_error,
):
    scenario = simulate_scenario(
        plant,
        n_train=6000,
        n_validation=600,
        input_kind="white",
        snr_db=32,
        input_correlation=0.35 if plant.ninputs > 1 else 0.0,
        noise_correlation=0.35 if plant.noutputs > 1 else 0.0,
        noise_color=0.45,
        seed=780 + order,
    )
    model = sippy.identify(
        scenario.y_train,
        scenario.u_train,
        method="SSARX",
        ss_f=horizon,
        ss_p=varx_order,
        ss_fixed_order=order,
        ss_d_required=direct,
        tsample=scenario.sample_time,
    )
    identified = control.ss(model.A, model.B, model.C, model.D, dt=model.ts)
    prediction = _simulate_process(identified, scenario.u_validation)

    assert (
        float(np.max(normalized_rmse(scenario.y_validation_clean, prediction)))
        < maximum_error
    )
    assert frequency_response_error(plant, identified) < maximum_error
    assert _pole_error(plant, identified) < maximum_error
    assert _markov_error(plant, identified) < maximum_error
