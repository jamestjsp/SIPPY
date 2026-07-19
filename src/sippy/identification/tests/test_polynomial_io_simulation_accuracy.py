import numpy as np
import pytest

from sippy.identification.factory import create_algorithm

from .simulation_scenarios import (
    normalized_rmse,
    simulate_identified_model,
    simulate_scenario,
    stable_mimo_plant,
    stable_siso_plant,
)

POLYNOMIAL_METHODS = [
    ("ARMAX", {"na": 2, "nb": 3, "nc": 1, "nk": 1, "mode": "ILLS"}),
    ("OE", {"nb": 3, "nf": 2, "nk": 1}),
    ("ARARX", {"na": 2, "nb": 3, "nd": 1, "nk": 1}),
    (
        "ARARMAX",
        {"na": 2, "nb": 3, "nc": 1, "nd": 1, "nf": 0, "nk": 1},
    ),
    ("BJ", {"nb": 2, "nc": 1, "nd": 1, "nf": 2, "nk": 1}),
    ("GEN", {"na": 2, "nb": 3, "nc": 1, "nd": 1, "nf": 1, "nk": 1}),
]

MIMO_POLYNOMIAL_METHODS = [
    ("OE", {"nb": 3, "nf": 3, "nk": 1}),
    ("BJ", {"nb": 3, "nc": 1, "nd": 1, "nf": 3, "nk": 1}),
    (
        "ARARX",
        {
            "na": [3, 3],
            "nb": [[3, 3], [3, 3]],
            "nd": [1, 1],
            "nk": [[1, 1], [1, 1]],
        },
    ),
    (
        "ARARMAX",
        {"na": 3, "nb": 3, "nc": 1, "nd": 1, "nf": 0, "nk": 1},
    ),
    ("GEN", {"na": 2, "nb": 3, "nc": 1, "nd": 1, "nf": 1, "nk": 1}),
]


@pytest.fixture(scope="module")
def colored_noise_scenario():
    return simulate_scenario(
        stable_siso_plant(direct_feedthrough=0.0),
        n_train=500,
        n_validation=250,
        input_kind="white",
        snr_db=30,
        noise_color=0.65,
        seed=401,
    )


@pytest.mark.parametrize(
    "method,orders", POLYNOMIAL_METHODS, ids=lambda value: str(value)
)
def test_polynomial_method_predicts_held_out_harold_simulation(
    method, orders, colored_noise_scenario
):
    scenario = colored_noise_scenario
    model = create_algorithm(method).identify(
        y=scenario.y_train,
        u=scenario.u_train,
        tsample=scenario.sample_time,
        max_iterations=50,
        **orders,
    )

    predicted = simulate_identified_model(model, scenario.u_validation)
    error = normalized_rmse(scenario.y_validation_clean[:, 30:], predicted[:, 30:])

    assert np.all(error < 0.35), f"{method} held-out NRMSE: {error}"


def test_armax_predicts_correlated_input_mimo_harold_simulation():
    scenario = simulate_scenario(
        stable_mimo_plant(direct_feedthrough=False),
        n_train=1400,
        n_validation=400,
        input_kind="white",
        snr_db=25,
        input_correlation=0.65,
        noise_correlation=0.5,
        noise_color=0.5,
        seed=402,
    )
    model = create_algorithm("ARMAX").identify(
        y=scenario.y_train,
        u=scenario.u_train,
        na=[3, 3],
        nb=[[3, 3], [3, 3]],
        nc=[1, 1],
        nk=[[1, 1], [1, 1]],
        mode="ILLS",
        max_iterations=50,
    )

    predicted = simulate_identified_model(model, scenario.u_validation)
    error = normalized_rmse(scenario.y_validation_clean[:, 30:], predicted[:, 30:])

    assert np.all(error < 0.3), f"ARMAX MIMO held-out NRMSE: {error}"


@pytest.fixture(scope="module")
def correlated_mimo_scenario():
    return simulate_scenario(
        stable_mimo_plant(direct_feedthrough=False),
        n_train=900,
        n_validation=300,
        input_kind="white",
        snr_db=28,
        input_correlation=0.6,
        noise_correlation=0.45,
        noise_color=0.55,
        seed=403,
    )


@pytest.mark.parametrize(
    "method,orders", MIMO_POLYNOMIAL_METHODS, ids=lambda value: str(value)
)
def test_polynomial_method_predicts_correlated_input_mimo_harold_simulation(
    method, orders, correlated_mimo_scenario
):
    scenario = correlated_mimo_scenario
    model = create_algorithm(method).identify(
        y=scenario.y_train,
        u=scenario.u_train,
        tsample=scenario.sample_time,
        max_iterations=60,
        **orders,
    )

    predicted = simulate_identified_model(model, scenario.u_validation)
    error = normalized_rmse(scenario.y_validation_clean[:, 40:], predicted[:, 40:])

    assert np.all(error < 0.4), f"{method} MIMO held-out NRMSE: {error}"
