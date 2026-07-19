import numpy as np
import pytest

from sippy.systems import (
    CtrlSysError,
    StateSpace,
    TransferFunction,
    forced_response,
    frequency_response,
    impulse_response,
    poles,
    ss,
    ss2tf,
    tf,
    tf2ss,
    tfdata,
)


def test_transfer_function_normalizes_non_monic_coefficients():
    system = tf([2.0, 4.0], [2.0, 6.0, 4.0], dt=0.1)

    np.testing.assert_allclose(system.num[0][0], [1.0, 2.0])
    np.testing.assert_allclose(system.den[0][0], [1.0, 3.0, 2.0])
    assert system.shape == (1, 1)
    assert system.dt == pytest.approx(0.1)


def test_siso_transfer_to_state_preserves_response_and_inputs():
    numerator = np.array([0.2, 0.1])
    denominator = np.array([1.0, -0.7, 0.12])
    numerator_before = numerator.copy()
    denominator_before = denominator.copy()
    transfer = tf(numerator, denominator, dt=0.2)

    realized = tf2ss(transfer)

    assert isinstance(realized, StateSpace)
    assert realized.nstates == 2
    assert realized.dt == pytest.approx(0.2)
    np.testing.assert_array_equal(numerator, numerator_before)
    np.testing.assert_array_equal(denominator, denominator_before)
    np.testing.assert_allclose(
        frequency_response(realized, [0.1, 1.0]).frdata,
        frequency_response(transfer, [0.1, 1.0]).frdata,
        rtol=1e-12,
        atol=1e-12,
    )


def test_mimo_transfer_to_state_preserves_channel_responses():
    transfer = tf(
        [[[0.2], [0.1]], [[0.3], [0.4]]],
        [
            [[1.0, -0.5], [1.0, -0.4]],
            [[1.0, -0.3], [1.0, -0.2]],
        ],
        dt=0.1,
    )

    realized = tf2ss(transfer)

    assert realized.shape == (2, 2)
    assert realized.nstates == 4
    np.testing.assert_allclose(
        frequency_response(realized, [0.2, 0.8]).frdata,
        frequency_response(transfer, [0.2, 0.8]).frdata,
        rtol=1e-11,
        atol=1e-11,
    )


def test_mimo_transfer_to_state_compacts_shared_dynamics():
    denominator = np.poly(np.linspace(0.2, 0.8, 6)).real
    transfer = tf(
        [
            [[0.01 * (1 + output + input_), 0.02 * (1 + output)] for input_ in range(4)]
            for output in range(4)
        ],
        [[denominator for _ in range(4)] for _ in range(4)],
        dt=0.1,
    )

    realized = tf2ss(transfer)
    frequencies = np.geomspace(0.01, 20.0, 257)

    assert realized.nstates == 12
    np.testing.assert_allclose(
        frequency_response(realized, frequencies).frdata,
        frequency_response(transfer, frequencies).frdata,
        rtol=2e-11,
        atol=2e-11,
    )


@pytest.mark.parametrize("sample_time", [None, 0.2])
def test_mimo_transfer_to_state_handles_distinct_channel_denominators(sample_time):
    transfer = tf(
        [
            [[0.2, 0.1], [0.1]],
            [[0.3], [0.4, 0.2]],
        ],
        [
            [[1.0, -0.5, 0.1], [1.0, -0.4]],
            [[1.0, -0.3], [1.0, -0.2, 0.05]],
        ],
        dt=sample_time,
    )

    realized = tf2ss(transfer)
    frequencies = np.geomspace(0.01, 10.0, 129)

    assert realized.nstates == 6
    np.testing.assert_allclose(
        frequency_response(realized, frequencies).frdata,
        frequency_response(transfer, frequencies).frdata,
        rtol=1e-11,
        atol=1e-11,
    )


def test_state_to_transfer_round_trip_preserves_mimo_response():
    system = ss(
        np.diag([0.8, 0.6]),
        np.eye(2),
        np.array([[1.0, 0.5], [-0.25, 1.0]]),
        np.array([[0.1, 0.0], [0.0, -0.2]]),
        dt=0.25,
    )

    transfer = ss2tf(system)

    assert isinstance(transfer, TransferFunction)
    assert transfer.shape == (2, 2)
    assert transfer.dt == pytest.approx(0.25)
    np.testing.assert_allclose(
        frequency_response(transfer, [0.1, 0.7]).frdata,
        frequency_response(system, [0.1, 0.7]).frdata,
        rtol=1e-11,
        atol=1e-11,
    )


@pytest.mark.parametrize("sample_time", [None, 0.1])
def test_transfer_frequency_response_vectorizes_unequal_mimo_orders(sample_time):
    system = tf(
        [
            [[0.2, 0.1, 0.05], [0.1]],
            [[0.3, -0.04], [0.4, 0.2]],
        ],
        [
            [[1.0, -1.2, 0.55, -0.08], [1.0, -0.4]],
            [[1.0, -0.3, 0.02], [1.0, -0.2, 0.05]],
        ],
        dt=sample_time,
    )
    frequencies = np.geomspace(0.01, 20.0, 257)
    response = frequency_response(system, frequencies).frdata
    evaluation_points = (
        1j * frequencies
        if sample_time is None
        else np.exp(1j * frequencies * sample_time)
    )
    expected = np.empty((2, 2, frequencies.size), dtype=complex)
    for output in range(2):
        for input_ in range(2):
            expected[output, input_] = np.polyval(
                system.num[output][input_], evaluation_points
            ) / np.polyval(system.den[output][input_], evaluation_points)

    np.testing.assert_allclose(response, expected, rtol=1e-14, atol=1e-14)


def test_transfer_frequency_response_preserves_empty_frequency_shape():
    system = tf(
        [[[1.0], [2.0]], [[3.0], [4.0]]],
        [[[1.0, -0.5], [1.0]], [[1.0, -0.2], [1.0, -0.1]]],
        dt=0.1,
    )

    response = frequency_response(system, []).frdata

    assert response.shape == (2, 2, 0)


def test_state_frequency_response_does_not_mutate_caller_arrays():
    matrices = [
        np.array([[0.75]]),
        np.array([[0.25]]),
        np.array([[2.0]]),
        np.array([[0.1]]),
    ]
    originals = [matrix.copy() for matrix in matrices]
    system = ss(*matrices, dt=0.5)

    frequency_response(system, [0.1, 0.5])

    for matrix, original in zip(matrices, originals):
        np.testing.assert_array_equal(matrix, original)


@pytest.mark.parametrize(
    ("state_count", "frequency_count", "sample_time"),
    [(4, 1, None), (4, 8, 0.1), (4, 32, 0.1), (24, 8, None), (24, 128, 0.1)],
)
def test_state_frequency_response_optimized_paths_match_dense_solve(
    state_count, frequency_count, sample_time
):
    rng = np.random.default_rng(state_count * 1000 + frequency_count)
    A = rng.normal(scale=0.02, size=(state_count, state_count))
    A += np.diag(np.linspace(0.2, 0.85, state_count))
    A *= 0.9 / np.max(np.abs(np.linalg.eigvals(A)))
    B = rng.normal(scale=0.1, size=(state_count, 2))
    C = rng.normal(scale=0.1, size=(2, state_count))
    D = rng.normal(scale=0.01, size=(2, 2))
    system = ss(A, B, C, D, dt=sample_time)
    frequencies = np.geomspace(0.01, 20.0, frequency_count)

    response = frequency_response(system, frequencies).frdata
    evaluation_points = (
        1j * frequencies
        if sample_time is None
        else np.exp(1j * frequencies * sample_time)
    )
    expected = np.empty_like(response)
    identity = np.eye(state_count)
    for index, point in enumerate(evaluation_points):
        expected[:, :, index] = C @ np.linalg.solve(point * identity - A, B) + D

    np.testing.assert_allclose(response, expected, rtol=2e-11, atol=2e-12)


def test_ctrlsys_info_code_becomes_python_exception(monkeypatch):
    import sippy.systems._backend as backend

    def failing_tc04ad(*args, **kwargs):
        return (
            0,
            0.0,
            np.empty((0, 0)),
            np.empty((0, 1)),
            np.empty((1, 0)),
            np.zeros((1, 1)),
            2,
        )

    monkeypatch.setattr(backend.ctrlsys, "tc04ad", failing_tc04ad)

    with pytest.raises(CtrlSysError, match=r"tc04ad failed with info=2"):
        tf2ss(tf([1.0], [1.0, -0.5], dt=1.0))


def test_batched_frequency_response_preserves_ctrlsys_singularity_error():
    system = ss([[1.0]], [[1.0]], [[1.0]], [[0.0]], dt=1.0)

    with pytest.raises(CtrlSysError, match=r"tb05ad failed with info=2"):
        frequency_response(system, np.zeros(16))


def test_forced_response_uses_discrete_state_equations():
    system = ss([[0.5]], [[1.0]], [[2.0]], [[0.25]], dt=0.2)

    response = forced_response(system, U=[1.0, 2.0, 0.0], X0=[0.5], squeeze=False)

    np.testing.assert_allclose(response.outputs, [[1.25, 3.0, 5.25]])
    np.testing.assert_allclose(response.states, [1.3125])


def test_forced_response_does_not_mutate_model_or_inputs():
    system = ss(
        [[0.5, 0.1], [0.0, 0.75]], [[1.0], [0.5]], [[2.0, -0.5]], [[0.25]], dt=0.2
    )
    inputs = np.asfortranarray([[1.0, 2.0, 0.0]])
    initial_state = np.array([0.5, -0.25])
    matrices_before = [
        matrix.copy() for matrix in (system.A, system.B, system.C, system.D)
    ]
    inputs_before = inputs.copy()
    initial_state_before = initial_state.copy()

    forced_response(system, U=inputs, X0=initial_state, squeeze=False)

    for matrix, before in zip(
        (system.A, system.B, system.C, system.D), matrices_before
    ):
        np.testing.assert_array_equal(matrix, before)
    np.testing.assert_array_equal(inputs, inputs_before)
    np.testing.assert_array_equal(initial_state, initial_state_before)


def test_impulse_response_matches_unit_area_convention():
    system = ss([[0.5]], [[1.0]], [[1.0]], [[0.0]], dt=0.2)

    response = impulse_response(system, T=np.arange(4) * 0.2, squeeze=False)

    np.testing.assert_allclose(response.outputs, [[[0.0, 5.0, 2.5, 1.25]]])


def test_transfer_function_reference_operations():
    first = tf([1.0, 0.5], [1.0, -0.4], dt=1.0)
    second = tf([2.0], [1.0, -0.2], dt=1.0)

    product = first * second
    numerator, denominator = tfdata(product)

    np.testing.assert_allclose(numerator[0][0], [2.0, 1.0])
    np.testing.assert_allclose(denominator[0][0], [1.0, -0.6, 0.08])
    np.testing.assert_allclose(np.sort(poles(product)), [0.2, 0.4])
