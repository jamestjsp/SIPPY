import numpy as np

from sippy import systems as control
from sippy.identification.factory import create_algorithm

from .simulation_scenarios import (
    normalized_rmse,
    simulate_noise_process,
    stable_arma_noise_filter,
)


def test_arma_recovers_control_noise_filter_on_held_out_innovations():
    process = stable_arma_noise_filter()
    _, y_train = simulate_noise_process(process, n_samples=2400, seed=501)
    innovations_validation, y_validation = simulate_noise_process(
        process, n_samples=600, seed=502
    )

    model = create_algorithm("ARMA").identify(
        y=y_train[np.newaxis, :],
        na=2,
        nc=1,
        max_iterations=80,
    )

    assert model.H_tf is not None
    predicted = control.forced_response(model.H_tf, U=innovations_validation).outputs
    error = normalized_rmse(
        y_validation[np.newaxis, 30:],
        np.asarray(predicted).reshape(1, -1)[:, 30:],
    )

    assert np.all(error < 0.2), f"ARMA noise-filter held-out NRMSE: {error}"
