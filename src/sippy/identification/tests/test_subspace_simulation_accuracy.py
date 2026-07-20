import numpy as np
import pytest

from sippy.identification.factory import create_algorithm

from .simulation_scenarios import (
    delayed_siso_plant,
    frequency_response_error,
    normalized_rmse,
    simulate_identified_model,
    simulate_scenario,
    stable_mimo_plant,
    stable_siso_plant,
    unstable_siso_plant,
)

SUBSPACE_METHODS = [
    "SUBSPACE",
    "N4SID",
    "MOESP",
    "CVA",
    "PARSIM-K",
    "PARSIM-S",
    "PARSIM-P",
]
PARSIM_METHODS = {"PARSIM-K", "PARSIM-S", "PARSIM-P"}
CLASSICAL_SUBSPACE_METHODS = {"N4SID", "MOESP", "CVA"}


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
def test_subspace_method_predicts_held_out_control_simulation(
    method, scenario_name, request
):
    scenario = request.getfixturevalue(scenario_name)
    options = {
        "ss_f": 12,
        "ss_fixed_order": scenario.plant.A.shape[0],
        "ss_d_required": True,
        "tsample": scenario.sample_time,
    }
    if method in PARSIM_METHODS:
        options["ss_p"] = 12
    model = create_algorithm(method).identify(
        y=scenario.y_train,
        u=scenario.u_train,
        **options,
    )

    predicted = simulate_identified_model(model, scenario.u_validation)
    error = normalized_rmse(scenario.y_validation_clean[:, 20:], predicted[:, 20:])

    assert np.all(error < 0.35), f"{method} held-out NRMSE: {error}"


@pytest.mark.parametrize("method", SUBSPACE_METHODS)
def test_subspace_method_identifies_unstable_control_system(method):
    scenario = simulate_scenario(
        unstable_siso_plant(),
        n_train=360,
        n_validation=180,
        input_kind="binary",
        snr_db=40,
        seed=203,
    )
    options = {
        "ss_f": 12,
        "ss_fixed_order": 2,
        "tsample": scenario.sample_time,
    }
    if method in PARSIM_METHODS:
        options["ss_p"] = 12
    if method in CLASSICAL_SUBSPACE_METHODS:
        options["ss_a_stability"] = False
    model = create_algorithm(method).identify(
        y=scenario.y_train,
        u=scenario.u_train,
        **options,
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


@pytest.mark.parametrize(
    "plant,n_train,snr_db,noise_color,tolerance",
    [
        (stable_siso_plant(), 1200, 5.0, 0.7, 0.55),
        (delayed_siso_plant(delay=5), 1800, 35.0, 0.2, 0.3),
    ],
    ids=["low-snr-colored", "delayed"],
)
def test_canonical_subspace_preserves_open_loop_plant_behavior(
    plant,
    n_train,
    snr_db,
    noise_color,
    tolerance,
):
    scenario = simulate_scenario(
        plant,
        n_train=n_train,
        n_validation=300,
        input_kind="white",
        snr_db=snr_db,
        noise_color=noise_color,
        seed=301 + plant.nstates,
    )
    model = create_algorithm("SUBSPACE").identify(
        y=scenario.y_train,
        u=scenario.u_train,
        ss_f=max(12, plant.nstates + 2),
        ss_fixed_order=plant.nstates,
        ss_d_required=bool(np.any(plant.D)),
        tsample=scenario.sample_time,
    )

    predicted = simulate_identified_model(model, scenario.u_validation)
    start = max(20, plant.nstates + 2)
    error = normalized_rmse(
        scenario.y_validation_clean[:, start:],
        predicted[:, start:],
    )
    frf_error = frequency_response_error(plant, model.deterministic_model)

    assert np.all(error < tolerance), f"held-out NRMSE: {error}"
    assert frf_error < tolerance, f"FRF error: {frf_error:.4g}"


def test_canonical_subspace_handles_correlated_colored_open_loop_mimo_data():
    plant = stable_mimo_plant(direct_feedthrough=True)
    scenario = simulate_scenario(
        plant,
        n_train=1600,
        n_validation=350,
        input_kind="white",
        snr_db=15.0,
        input_correlation=0.65,
        noise_correlation=0.5,
        noise_color=0.6,
        seed=312,
    )

    model = create_algorithm("SUBSPACE").identify(
        y=scenario.y_train,
        u=scenario.u_train,
        ss_f=12,
        ss_fixed_order=plant.nstates,
        ss_d_required=True,
        tsample=scenario.sample_time,
    )
    predicted = simulate_identified_model(model, scenario.u_validation)
    error = normalized_rmse(scenario.y_validation_clean[:, 20:], predicted[:, 20:])
    frf_error = frequency_response_error(plant, model.deterministic_model)

    assert np.all(error < 0.35), f"held-out MIMO NRMSE: {error}"
    assert frf_error < 0.35, f"MIMO FRF error: {frf_error:.4g}"
