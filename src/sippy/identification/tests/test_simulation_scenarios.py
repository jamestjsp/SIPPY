import harold
import numpy as np
import pytest

from .simulation_scenarios import (
    delayed_siso_plant,
    generate_excitation,
    normalized_rmse,
    simulate_scenario,
    stable_mimo_plant,
    stable_siso_plant,
    unstable_siso_plant,
)


@pytest.mark.parametrize("input_kind", ["white", "binary", "multisine"])
def test_siso_scenarios_are_reproducible(input_kind):
    first = simulate_scenario(
        stable_siso_plant(),
        n_train=256,
        n_validation=128,
        input_kind=input_kind,
        snr_db=20,
        seed=42,
    )
    second = simulate_scenario(
        stable_siso_plant(),
        n_train=256,
        n_validation=128,
        input_kind=input_kind,
        snr_db=20,
        seed=42,
    )

    assert first.u_train.shape == (1, 256)
    assert first.y_train.shape == (1, 256)
    assert first.u_validation.shape == (1, 128)
    assert first.y_validation_clean.shape == (1, 128)
    np.testing.assert_array_equal(first.u_train, second.u_train)
    np.testing.assert_array_equal(first.y_train, second.y_train)
    assert not np.array_equal(first.u_train[:, :128], first.u_validation)


def test_mimo_scenario_supports_correlated_inputs_and_noise():
    scenario = simulate_scenario(
        stable_mimo_plant(),
        n_train=4000,
        n_validation=500,
        input_kind="white",
        snr_db=15,
        input_correlation=0.8,
        noise_correlation=0.6,
        seed=7,
    )

    assert scenario.u_train.shape == (2, 4000)
    assert scenario.y_train.shape == (2, 4000)
    assert np.corrcoef(scenario.u_train)[0, 1] == pytest.approx(0.8, abs=0.04)
    assert np.corrcoef(scenario.noise_train)[0, 1] == pytest.approx(0.6, abs=0.06)


@pytest.mark.parametrize("snr_db", [5, 20, 40])
def test_scenario_noise_has_requested_snr(snr_db):
    scenario = simulate_scenario(
        stable_mimo_plant(),
        n_train=3000,
        n_validation=200,
        input_kind="white",
        snr_db=snr_db,
        seed=11,
    )

    signal_power = np.mean(scenario.y_train_clean**2, axis=1)
    noise_power = np.mean(scenario.noise_train**2, axis=1)
    measured_snr = 10 * np.log10(signal_power / noise_power)
    np.testing.assert_allclose(measured_snr, snr_db, atol=0.15)


def test_colored_noise_has_temporal_correlation():
    scenario = simulate_scenario(
        stable_siso_plant(),
        n_train=3000,
        n_validation=200,
        input_kind="white",
        snr_db=20,
        noise_color=0.75,
        seed=13,
    )

    lag_one = np.corrcoef(scenario.noise_train[0, :-1], scenario.noise_train[0, 1:])[
        0, 1
    ]
    assert lag_one > 0.65


def test_unstable_scenario_remains_finite_over_bounded_horizon():
    scenario = simulate_scenario(
        unstable_siso_plant(),
        n_train=240,
        n_validation=120,
        input_kind="binary",
        snr_db=30,
        seed=21,
    )

    assert np.all(np.isfinite(scenario.y_train))
    assert np.max(np.abs(scenario.y_train_clean)) > 1.0


def test_long_delay_plant_has_no_response_before_delay():
    delay = 16
    plant = delayed_siso_plant(delay)
    impulse = np.zeros((40, 1))
    impulse[0, 0] = 1.0

    output, _ = harold.simulate_linear_system(plant, impulse)
    output = np.asarray(output).ravel()

    np.testing.assert_allclose(output[: delay + 1], 0.0)
    assert output[delay + 1] != 0.0


def test_excitation_rejects_invalid_correlation():
    with pytest.raises(ValueError, match="correlation"):
        generate_excitation(2, 100, kind="white", correlation=1.0, seed=1)


def test_normalized_rmse_handles_exact_and_degraded_predictions():
    actual = np.array([[1.0, -1.0, 2.0, -2.0]])

    np.testing.assert_allclose(normalized_rmse(actual, actual), 0.0)
    assert normalized_rmse(actual, np.zeros_like(actual))[0] > 0.9
