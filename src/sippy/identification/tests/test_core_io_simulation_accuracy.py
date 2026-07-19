import numpy as np
import pytest

from sippy.identification.factory import create_algorithm

from .simulation_scenarios import (
    delayed_siso_plant,
    normalized_rmse,
    simulate_identified_model,
    simulate_scenario,
    stable_mimo_plant,
    stable_siso_plant,
)


@pytest.mark.parametrize(
    "plant,orders,tolerance,seed",
    [
        (stable_siso_plant(), {"na": 2, "nb": 3, "nk": 0}, 0.12, 301),
        (stable_mimo_plant(), {"na": 3, "nb": 4, "nk": 0}, 0.2, 302),
    ],
    ids=["siso", "mimo"],
)
def test_arx_predicts_held_out_control_simulation(plant, orders, tolerance, seed):
    scenario = simulate_scenario(
        plant,
        n_train=1400,
        n_validation=400,
        input_kind="white",
        snr_db=35,
        input_correlation=0.45 if plant.shape[1] > 1 else 0.0,
        seed=seed,
    )
    model = create_algorithm("ARX").identify(
        y=scenario.y_train,
        u=scenario.u_train,
        tsample=scenario.sample_time,
        **orders,
    )

    predicted = simulate_identified_model(model, scenario.u_validation)
    error = normalized_rmse(scenario.y_validation_clean[:, 30:], predicted[:, 30:])

    assert np.all(error < tolerance), f"ARX held-out NRMSE: {error}"


def test_fir_predicts_long_delay_control_simulation():
    delay = 16
    scenario = simulate_scenario(
        delayed_siso_plant(delay),
        n_train=1600,
        n_validation=500,
        input_kind="binary",
        snr_db=40,
        seed=303,
    )
    model = create_algorithm("FIR").identify(
        y=scenario.y_train,
        u=scenario.u_train,
        nb=36,
        nk=0,
        tsample=scenario.sample_time,
    )

    predicted = simulate_identified_model(model, scenario.u_validation)
    error = normalized_rmse(scenario.y_validation_clean[:, 50:], predicted[:, 50:])

    assert np.all(error < 0.08), f"FIR long-delay held-out NRMSE: {error}"


def test_fir_predicts_correlated_input_mimo_control_simulation():
    scenario = simulate_scenario(
        stable_mimo_plant(),
        n_train=1800,
        n_validation=450,
        input_kind="white",
        snr_db=30,
        input_correlation=0.7,
        seed=304,
    )
    model = create_algorithm("FIR").identify(
        y=scenario.y_train,
        u=scenario.u_train,
        nb=32,
        nk=0,
    )

    predicted = simulate_identified_model(model, scenario.u_validation)
    error = normalized_rmse(scenario.y_validation_clean[:, 50:], predicted[:, 50:])

    assert np.all(error < 0.12), f"FIR MIMO held-out NRMSE: {error}"
