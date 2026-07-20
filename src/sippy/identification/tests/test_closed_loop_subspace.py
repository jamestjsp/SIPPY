import numpy as np
import pytest

from sippy import systems as control

from .simulation_scenarios import (
    frequency_response_error,
    simulate_closed_loop,
    simulate_closed_loop_scenario,
    stable_mimo_plant,
    stable_siso_plant,
    static_output_feedback_controller,
    unstable_siso_plant,
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
