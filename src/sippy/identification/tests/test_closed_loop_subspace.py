import numpy as np
import pandas as pd
import pytest

import sippy
from sippy import systems as control
from sippy.identification.algorithms.automatic_subspace import (
    select_automatic_dimensions,
)
from sippy.identification.algorithms.parsim_k import PARSIMKAlgorithm
from sippy.identification.algorithms.subspace_core import SubspaceCoreAlgorithm
from sippy.identification.algorithms.subspace_data import prepare_subspace_data
from sippy.identification.iddata import IDData

from .simulation_scenarios import (
    frequency_response_error,
    normalized_rmse,
    simulate_closed_loop,
    simulate_closed_loop_scenario,
    simulate_scenario,
    stable_mimo_plant,
    stable_siso_plant,
    static_output_feedback_controller,
    unstable_siso_plant,
)


def _identify_with_parsim_k(y, u, *, order, future_horizon, past_horizon, dt):
    model = PARSIMKAlgorithm().identify(
        y=y,
        u=u,
        ss_f=future_horizon,
        ss_p=past_horizon,
        ss_fixed_order=order,
        ss_d_required=False,
        tsample=dt,
    )
    return model, control.ss(model.A, model.B, model.C, model.D, dt=model.ts)


def _assert_plant_recovery(
    plant,
    identified,
    u_validation,
    y_validation,
    *,
    maximum_nrmse,
    maximum_frequency_error,
):
    prediction = control.forced_response(
        identified,
        U=u_validation,
        squeeze=False,
    ).outputs
    error = float(np.max(normalized_rmse(y_validation, prediction)))
    frequency_error = frequency_response_error(plant, identified)
    assert error < maximum_nrmse, f"validation NRMSE was {error:.4g}"
    assert frequency_error < maximum_frequency_error, (
        f"frequency-response error was {frequency_error:.4g}"
    )


def test_closed_loop_simulator_satisfies_plant_and_controller_equations():
    plant = stable_siso_plant(direct_feedthrough=0.08)
    controller = static_output_feedback_controller([[0.7]], dt=plant.dt)
    reference = np.array([[0.2, -0.4, 0.6, 0.1, -0.3]])
    dither = np.array([[0.05, 0.0, -0.02, 0.03, 0.01]])
    disturbance = np.array([[0.01, -0.02, 0.03, -0.01, 0.02]])

    trajectory = simulate_closed_loop(
        plant,
        controller,
        reference,
        dither=dither,
        disturbance=disturbance,
    )

    for sample in range(reference.shape[1]):
        plant_state = trajectory.plant_states[:, sample]
        controller_state = trajectory.controller_states[:, sample]
        error = reference[:, sample] - trajectory.output[:, sample]

        np.testing.assert_allclose(
            trajectory.plant_input[:, sample],
            controller.C @ controller_state + controller.D @ error + dither[:, sample],
        )
        np.testing.assert_allclose(
            trajectory.plant_output[:, sample],
            plant.C @ plant_state + plant.D @ trajectory.plant_input[:, sample],
        )
        np.testing.assert_allclose(
            trajectory.output[:, sample],
            trajectory.plant_output[:, sample] + disturbance[:, sample],
        )
        np.testing.assert_allclose(
            trajectory.plant_states[:, sample + 1],
            plant.A @ plant_state + plant.B @ trajectory.plant_input[:, sample],
        )


def test_closed_loop_simulator_rejects_non_well_posed_feedthrough():
    plant = control.ss([[0.5]], [[1.0]], [[1.0]], [[1.0]], dt=1.0)
    controller = static_output_feedback_controller([[-1.0]], dt=1.0)

    with pytest.raises(ValueError, match="algebraic loop"):
        simulate_closed_loop(plant, controller, np.ones((1, 20)))


def test_closed_loop_scenario_rejects_insufficient_excitation():
    plant = stable_siso_plant(direct_feedthrough=0.0)
    controller = static_output_feedback_controller([[0.5]], dt=plant.dt)

    with pytest.raises(ValueError, match="persistently exciting"):
        simulate_closed_loop_scenario(
            plant,
            controller,
            n_train=120,
            n_validation=40,
            reference_scale=0.0,
            noise_scale=0.0,
            dither_scale=0.0,
            seed=4,
        )


def test_feedback_stabilizes_unstable_siso_training_record():
    plant = unstable_siso_plant()
    controller = static_output_feedback_controller([[2.0]], dt=plant.dt)

    scenario = simulate_closed_loop_scenario(
        plant,
        controller,
        n_train=1000,
        n_validation=120,
        reference_kind="binary",
        noise_scale=0.03,
        seed=10,
    )

    closed_loop_poles = np.linalg.eigvals(plant.A - plant.B @ controller.D @ plant.C)
    assert np.max(np.abs(np.linalg.eigvals(plant.A))) > 1.0
    assert np.max(np.abs(closed_loop_poles)) < 1.0
    assert np.all(np.isfinite(scenario.output))
    assert np.max(np.abs(scenario.output)) < 20.0
    np.testing.assert_allclose(
        scenario.y_validation_clean,
        control.forced_response(
            plant,
            U=scenario.u_validation,
            squeeze=False,
        ).outputs,
    )


def test_mimo_dynamic_feedback_supports_correlated_colored_data():
    plant = stable_mimo_plant(direct_feedthrough=True)
    controller = control.ss(
        np.diag([0.45, 0.6]),
        0.25 * np.eye(2),
        0.2 * np.eye(2),
        np.array([[0.35, 0.04], [-0.03, 0.3]]),
        dt=plant.dt,
    )

    scenario = simulate_closed_loop_scenario(
        plant,
        controller,
        n_train=3000,
        n_validation=300,
        reference_correlation=0.75,
        noise_correlation=0.55,
        noise_color=0.7,
        dither_scale=0.08,
        seed=22,
    )

    assert scenario.reference.shape == (2, 3000)
    assert scenario.plant_input.shape == (2, 3000)
    assert scenario.output.shape == (2, 3000)
    assert scenario.plant_states.shape == (3, 3001)
    assert scenario.controller_states.shape == (2, 3001)
    assert scenario.excitation_rank == 2 * scenario.excitation_order
    assert np.corrcoef(scenario.reference)[0, 1] == pytest.approx(0.75, abs=0.05)
    assert np.corrcoef(scenario.innovations)[0, 1] == pytest.approx(0.55, abs=0.06)
    lag_one = np.corrcoef(scenario.disturbance[0, :-1], scenario.disturbance[0, 1:])[
        0, 1
    ]
    assert lag_one == pytest.approx(0.7, abs=0.05)
    assert np.linalg.norm(scenario.controller_states) > 0.0
    error = scenario.reference - scenario.output
    np.testing.assert_allclose(
        scenario.controller_states[:, 1:],
        controller.A @ scenario.controller_states[:, :-1] + controller.B @ error,
    )
    np.testing.assert_allclose(
        scenario.plant_states[:, 1:],
        plant.A @ scenario.plant_states[:, :-1] + plant.B @ scenario.plant_input,
    )


def test_closed_loop_scenario_is_reproducible_and_exposes_plant_metrics():
    plant = stable_siso_plant()
    controller = static_output_feedback_controller([[0.8]], dt=plant.dt)
    first = simulate_closed_loop_scenario(
        plant,
        controller,
        n_train=500,
        n_validation=120,
        noise_color=0.4,
        dither_scale=0.05,
        seed=31,
    )
    second = simulate_closed_loop_scenario(
        plant,
        controller,
        n_train=500,
        n_validation=120,
        noise_color=0.4,
        dither_scale=0.05,
        seed=31,
    )

    np.testing.assert_array_equal(first.reference, second.reference)
    np.testing.assert_array_equal(first.plant_input, second.plant_input)
    np.testing.assert_array_equal(first.output, second.output)
    np.testing.assert_array_equal(first.innovations, second.innovations)
    assert frequency_response_error(plant, plant) == pytest.approx(0.0)

    perturbed = control.ss(
        plant.A,
        0.8 * plant.B,
        plant.C,
        plant.D,
        dt=plant.dt,
    )
    assert frequency_response_error(plant, perturbed) > 0.05


def test_parsim_k_recovers_the_same_stable_plant_from_open_and_closed_loop_data():
    plant = stable_siso_plant(direct_feedthrough=0.0)
    open_loop = simulate_scenario(
        plant,
        n_train=3000,
        n_validation=300,
        input_kind="white",
        snr_db=28.0,
        noise_color=0.45,
        seed=51,
    )
    closed_loop = simulate_closed_loop_scenario(
        plant,
        static_output_feedback_controller([[1.1]], dt=plant.dt),
        n_train=3000,
        n_validation=300,
        noise_scale=0.035,
        noise_color=0.45,
        dither_scale=0.04,
        seed=52,
    )

    open_model, open_system = _identify_with_parsim_k(
        open_loop.y_train,
        open_loop.u_train,
        order=2,
        future_horizon=12,
        past_horizon=24,
        dt=plant.dt,
    )
    closed_model, closed_system = _identify_with_parsim_k(
        closed_loop.output,
        closed_loop.plant_input,
        order=2,
        future_horizon=12,
        past_horizon=24,
        dt=plant.dt,
    )

    assert open_model.ts == pytest.approx(plant.dt)
    assert closed_model.ts == pytest.approx(plant.dt)
    _assert_plant_recovery(
        plant,
        open_system,
        open_loop.u_validation,
        open_loop.y_validation_clean,
        maximum_nrmse=0.2,
        maximum_frequency_error=0.2,
    )
    _assert_plant_recovery(
        plant,
        closed_system,
        closed_loop.u_validation,
        closed_loop.y_validation_clean,
        maximum_nrmse=0.25,
        maximum_frequency_error=0.25,
    )


def test_parsim_k_recovers_a_feedback_stabilized_unstable_plant():
    plant = unstable_siso_plant()
    scenario = simulate_closed_loop_scenario(
        plant,
        static_output_feedback_controller([[2.0]], dt=plant.dt),
        n_train=4500,
        n_validation=160,
        reference_kind="binary",
        noise_scale=0.02,
        dither_scale=0.03,
        seed=61,
    )

    model, identified = _identify_with_parsim_k(
        scenario.output,
        scenario.plant_input,
        order=2,
        future_horizon=16,
        past_horizon=30,
        dt=plant.dt,
    )

    assert np.max(np.abs(np.linalg.eigvals(model.A))) > 1.0
    _assert_plant_recovery(
        plant,
        identified,
        scenario.u_validation,
        scenario.y_validation_clean,
        maximum_nrmse=0.35,
        maximum_frequency_error=0.3,
    )


def test_parsim_k_recovers_mimo_plant_under_dynamic_feedback():
    plant = stable_mimo_plant(direct_feedthrough=False)
    controller = control.ss(
        np.diag([0.45, 0.6]),
        0.25 * np.eye(2),
        0.2 * np.eye(2),
        np.array([[0.35, 0.04], [-0.03, 0.3]]),
        dt=plant.dt,
    )
    scenario = simulate_closed_loop_scenario(
        plant,
        controller,
        n_train=5000,
        n_validation=300,
        reference_correlation=0.45,
        noise_scale=0.025,
        noise_correlation=0.35,
        noise_color=0.4,
        dither_scale=0.05,
        seed=71,
    )

    _, identified = _identify_with_parsim_k(
        scenario.output,
        scenario.plant_input,
        order=3,
        future_horizon=12,
        past_horizon=24,
        dt=plant.dt,
    )

    _assert_plant_recovery(
        plant,
        identified,
        scenario.u_validation,
        scenario.y_validation_clean,
        maximum_nrmse=0.35,
        maximum_frequency_error=0.3,
    )


def test_two_stage_ort_removes_reference_uncorrelated_feedback_disturbance():
    plant = stable_siso_plant(direct_feedthrough=0.0)
    scenario = simulate_closed_loop_scenario(
        plant,
        static_output_feedback_controller([[1.1]], dt=plant.dt),
        n_train=4000,
        n_validation=300,
        noise_scale=0.12,
        noise_color=0.75,
        dither_scale=0.12,
        seed=81,
    )
    references = np.vstack((scenario.reference, scenario.dither))
    noisy_data = prepare_subspace_data(
        scenario.output,
        scenario.plant_input,
        future_horizon=12,
        past_offset=12,
        reference=references,
        scale=False,
    )
    clean_data = prepare_subspace_data(
        scenario.plant_output,
        scenario.plant_input,
        future_horizon=12,
        past_offset=12,
        reference=references,
        scale=False,
    )

    _, ort = SubspaceCoreAlgorithm.svd_weighted_ort(noisy_data)
    assert ort.diagnostics.usable
    deterministic = ort.reference_projection.materialize()
    output_start = noisy_data.future_inputs.shape[0] + noisy_data.past_data.shape[0]
    projected_noisy_output = deterministic[output_start:]
    reference_hankel = ort.reference_projection.reference_hankel
    projected_clean_output = (
        clean_data.future_outputs @ np.linalg.pinv(reference_hankel) @ reference_hankel
    )

    raw_error = np.linalg.norm(noisy_data.future_outputs - clean_data.future_outputs)
    projected_error = np.linalg.norm(projected_noisy_output - projected_clean_output)
    assert projected_error < 0.35 * raw_error


def test_two_stage_ort_recovers_closed_loop_plant_from_measured_references():
    plant = stable_siso_plant(direct_feedthrough=0.0)
    scenario = simulate_closed_loop_scenario(
        plant,
        static_output_feedback_controller([[1.1]], dt=plant.dt),
        n_train=4000,
        n_validation=300,
        noise_scale=0.06,
        noise_color=0.7,
        dither_scale=0.1,
        seed=82,
    )

    result, diagnostics = SubspaceCoreAlgorithm.olsims_ort(
        scenario.output,
        scenario.plant_input,
        np.vstack((scenario.reference, scenario.dither)),
        f=12,
        fixed_order=2,
    )

    assert diagnostics.usable
    A, B, C, D, *_ = result
    identified = control.ss(A, B, C, D, dt=plant.dt)
    _assert_plant_recovery(
        plant,
        identified,
        scenario.u_validation,
        scenario.y_validation_clean,
        maximum_nrmse=0.3,
        maximum_frequency_error=0.3,
    )


def test_unusable_measured_reference_records_predictor_fallback_reason():
    rng = np.random.default_rng(83)
    y = rng.normal(size=(1, 300))
    u = rng.normal(size=(1, 300))
    duplicate = np.vstack((u, u))
    data = prepare_subspace_data(
        y,
        u,
        future_horizon=8,
        past_offset=8,
        reference=duplicate,
    )

    with pytest.warns(UserWarning, match="using the predictor-form estimator"):
        decomposition, ort = SubspaceCoreAlgorithm.svd_weighted_ort(data)

    assert decomposition is None
    assert ort.diagnostics.reason == "reference_rank_deficient"


def test_automatic_dimensions_select_consistent_predictor_model_without_loop_flag():
    plant = stable_siso_plant(direct_feedthrough=0.0)
    scenario = simulate_closed_loop_scenario(
        plant,
        static_output_feedback_controller([[1.1]], dt=plant.dt),
        n_train=3000,
        n_validation=300,
        noise_scale=0.035,
        noise_color=0.45,
        dither_scale=0.04,
        seed=91,
    )

    estimate = select_automatic_dimensions(
        scenario.output,
        scenario.plant_input,
        explicit_horizon=12,
    )

    assert estimate.route == "predictor"
    assert estimate.selection.candidate.order == 2
    candidate = estimate.selection.candidate
    identified = control.ss(
        candidate.A,
        candidate.B,
        candidate.C,
        candidate.D,
        dt=plant.dt,
    )
    _assert_plant_recovery(
        plant,
        identified,
        scenario.u_validation,
        scenario.y_validation_clean,
        maximum_nrmse=0.3,
        maximum_frequency_error=0.3,
    )


def test_canonical_subspace_automatically_uses_predictor_route_for_raw_data():
    plant = stable_siso_plant(direct_feedthrough=0.0)
    scenario = simulate_closed_loop_scenario(
        plant,
        static_output_feedback_controller([[1.1]], dt=plant.dt),
        n_train=3000,
        n_validation=300,
        noise_scale=0.035,
        noise_color=0.45,
        dither_scale=0.04,
        seed=93,
    )

    model = sippy.identify(scenario.output, scenario.plant_input)

    assert model.method == "SUBSPACE"
    assert model.identification_info["estimator_route"] == "predictor"
    assert model.identification_info["reference_projection"] == {
        "status": "not-provided",
        "reason": "reference_missing",
    }
    assert model.identification_info["selected_order"] == 2
    identified = control.ss(model.A, model.B, model.C, model.D, dt=model.ts)
    _assert_plant_recovery(
        plant,
        identified,
        scenario.u_validation,
        scenario.y_validation_clean,
        maximum_nrmse=0.3,
        maximum_frequency_error=0.3,
    )


def test_automatic_dimensions_use_valid_reference_and_honor_expert_overrides():
    plant = stable_siso_plant(direct_feedthrough=0.0)
    scenario = simulate_closed_loop_scenario(
        plant,
        static_output_feedback_controller([[1.1]], dt=plant.dt),
        n_train=3000,
        n_validation=300,
        noise_scale=0.06,
        noise_color=0.7,
        dither_scale=0.1,
        seed=92,
    )

    estimate = select_automatic_dimensions(
        scenario.output,
        scenario.plant_input,
        reference=np.vstack((scenario.reference, scenario.dither)),
        explicit_horizon=12,
        explicit_order=2,
    )

    assert estimate.route == "two-stage-ort"
    assert estimate.horizon_candidates == (12,)
    assert estimate.selection.candidate.order == 2
    assert estimate.reference_projection_reason is None
    assert estimate.weighting.requested == "CVA"

    with pytest.warns(UserWarning, match="using the predictor-form estimator"):
        fallback = select_automatic_dimensions(
            scenario.output,
            scenario.plant_input,
            reference=np.vstack((scenario.reference, scenario.reference)),
            explicit_horizon=12,
            explicit_order=2,
        )
    assert fallback.route == "predictor"
    assert fallback.reference_projection_reason == "reference_rank_deficient"


def test_canonical_subspace_uses_iddata_references_without_algorithm_selection():
    plant = stable_siso_plant(direct_feedthrough=0.0)
    scenario = simulate_closed_loop_scenario(
        plant,
        static_output_feedback_controller([[1.1]], dt=plant.dt),
        n_train=3000,
        n_validation=300,
        noise_scale=0.06,
        noise_color=0.7,
        dither_scale=0.1,
        seed=94,
    )
    frame = pd.DataFrame(
        {
            "u": scenario.plant_input[0],
            "y": scenario.output[0],
            "reference": scenario.reference[0],
            "dither": scenario.dither[0],
        }
    )
    data = IDData(
        frame,
        inputs=["u"],
        outputs=["y"],
        references=["reference", "dither"],
        tsample=plant.dt,
    )

    model = sippy.identify(data=data, ss_f=12, ss_fixed_order=2)

    assert model.method == "SUBSPACE"
    assert model.ts == pytest.approx(plant.dt)
    assert model.identification_info["estimator_route"] == "two-stage-ort"
    assert model.identification_info["reference_projection"] == {
        "status": "used",
        "reason": None,
    }
    assert model.identification_info["options"]["reference_channels"] == 2
    assert "reference" not in model.identification_info["options"]
    numerical_ranks = model.identification_info["numerical_ranks"]
    assert numerical_ranks["reference_rank"] == numerical_ranks["reference_rows"]


def test_canonical_subspace_records_invalid_reference_predictor_fallback():
    rng = np.random.default_rng(95)
    u = rng.normal(size=(1, 500))
    y = np.zeros_like(u)
    for sample in range(1, u.shape[1]):
        y[0, sample] = (
            0.7 * y[0, sample - 1] + 0.25 * u[0, sample - 1] + 0.01 * rng.normal()
        )
    duplicate_reference = np.vstack((u, u))

    with pytest.warns(UserWarning, match="using the predictor-form estimator"):
        model = sippy.identify(
            y,
            u,
            reference=duplicate_reference,
            ss_f=8,
            ss_fixed_order=1,
        )

    assert model.identification_info["estimator_route"] == "predictor"
    assert model.identification_info["reference_projection"] == {
        "status": "fallback",
        "reason": "reference_rank_deficient",
    }


def test_projected_cva_preserves_variance_and_closed_loop_bias_trend():
    plant = stable_siso_plant(direct_feedthrough=0.0)

    def identification_error(sample_count, seed, weighting):
        scenario = simulate_closed_loop_scenario(
            plant,
            static_output_feedback_controller([[1.1]], dt=plant.dt),
            n_train=sample_count,
            n_validation=120,
            noise_scale=0.07,
            noise_color=0.65,
            dither_scale=0.05,
            seed=seed,
        )
        estimate = select_automatic_dimensions(
            scenario.output,
            scenario.plant_input,
            explicit_horizon=10,
            explicit_order=2,
            weighting=weighting,
        )
        candidate = estimate.selection.candidate
        identified = control.ss(
            candidate.A,
            candidate.B,
            candidate.C,
            candidate.D,
            dt=plant.dt,
        )
        return frequency_response_error(plant, identified)

    unweighted = np.array(
        [identification_error(1200, seed, "N4SID") for seed in range(101, 106)]
    )
    cva = np.array(
        [identification_error(1200, seed, "CVA") for seed in range(101, 106)]
    )
    assert np.mean(cva) <= 1.15 * np.mean(unweighted)
    assert np.var(cva) <= 1.25 * np.var(unweighted)

    sample_counts = (700, 1400, 2800)
    unweighted_trend = np.array(
        [identification_error(count, 111, "N4SID") for count in sample_counts]
    )
    cva_trend = np.array(
        [identification_error(count, 111, "CVA") for count in sample_counts]
    )
    assert unweighted_trend[-1] < unweighted_trend[0]
    assert cva_trend[-1] < cva_trend[0]
    assert cva_trend[-1] <= 1.2 * unweighted_trend[-1]
