import numpy as np
import pytest

from sippy.identification.factory import create_algorithm

from .simulation_scenarios import (
    normalized_rmse,
    simulate_identified_model,
    simulate_scenario,
    stable_mimo_plant,
    stable_siso_plant,
    unstable_siso_plant,
)

SUBSPACE_METHODS = ["N4SID", "MOESP", "CVA", "PARSIM-K", "PARSIM-S", "PARSIM-P"]


@pytest.fixture(scope="module")
def siso_scenario():
    return simulate_scenario(
        stable_siso_plant(),
        n_train=800,
        n_validation=300,
        input_kind="white",
        snr_db=30,
        seed=201,
    )


@pytest.fixture(scope="module")
def mimo_scenario():
    return simulate_scenario(
        stable_mimo_plant(),
        n_train=1000,
        n_validation=350,
        input_kind="white",
        snr_db=30,
        input_correlation=0.5,
        noise_correlation=0.4,
        seed=202,
    )


@pytest.mark.parametrize("method", SUBSPACE_METHODS)
@pytest.mark.parametrize("scenario_name", ["siso_scenario", "mimo_scenario"])
def test_subspace_method_predicts_held_out_harold_simulation(
    method, scenario_name, request
):
    scenario = request.getfixturevalue(scenario_name)
    model = create_algorithm(method).identify(
        y=scenario.y_train,
        u=scenario.u_train,
        ss_f=12,
        ss_p=12,
        ss_fixed_order=scenario.plant.a.shape[0],
        ss_d_required=True,
        tsample=scenario.sample_time,
    )

    predicted = simulate_identified_model(model, scenario.u_validation)
    error = normalized_rmse(scenario.y_validation_clean[:, 20:], predicted[:, 20:])

    assert np.all(error < 0.35), f"{method} held-out NRMSE: {error}"


@pytest.mark.parametrize("method", SUBSPACE_METHODS)
def test_subspace_method_identifies_unstable_harold_system(method):
    scenario = simulate_scenario(
        unstable_siso_plant(),
        n_train=360,
        n_validation=180,
        input_kind="binary",
        snr_db=40,
        seed=203,
    )
    model = create_algorithm(method).identify(
        y=scenario.y_train,
        u=scenario.u_train,
        ss_f=12,
        ss_p=12,
        ss_fixed_order=2,
        ss_a_stability=False,
        tsample=scenario.sample_time,
    )

    predicted = simulate_identified_model(model, scenario.u_validation)
    short_horizon_error = normalized_rmse(
        scenario.y_validation_clean[:, :40], predicted[:, :40]
    )
    dominant_pole = np.max(np.abs(np.linalg.eigvals(model.A)))

    assert not model.is_stable(), (
        f"{method} incorrectly forced an unstable model stable"
    )
    assert dominant_pole == pytest.approx(1.015, abs=3e-4)
    assert np.all(short_horizon_error < 0.9), (
        f"{method} unstable-system short-horizon NRMSE: {short_horizon_error}"
    )


@pytest.mark.parametrize(
    "snr_db,input_kind,tolerance",
    [(5, "white", 0.5), (20, "binary", 0.25), (40, "multisine", 0.15)],
)
def test_n4sid_accuracy_across_snr_and_excitation(snr_db, input_kind, tolerance):
    scenario = simulate_scenario(
        stable_siso_plant(),
        n_train=900,
        n_validation=300,
        input_kind=input_kind,
        snr_db=snr_db,
        seed=210 + snr_db,
    )
    model = create_algorithm("N4SID").identify(
        y=scenario.y_train,
        u=scenario.u_train,
        ss_f=12,
        ss_fixed_order=2,
        ss_d_required=True,
    )

    predicted = simulate_identified_model(model, scenario.u_validation)
    error = normalized_rmse(scenario.y_validation_clean[:, 20:], predicted[:, 20:])

    assert np.all(error < tolerance), f"N4SID at {snr_db} dB NRMSE: {error}"
