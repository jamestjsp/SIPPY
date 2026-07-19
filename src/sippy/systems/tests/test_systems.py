import numpy as np
import pytest

from sippy.systems import (
    CtrlSysError,
    StateSpace,
    TransferFunction,
    forced_response,
    frequency_response,
    impulse_response,
    ss,
    ss2tf,
    tf,
    tf2ss,
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


def test_forced_response_uses_discrete_state_equations():
    system = ss([[0.5]], [[1.0]], [[2.0]], [[0.25]], dt=0.2)

    response = forced_response(system, U=[1.0, 2.0, 0.0], X0=[0.5], squeeze=False)

    np.testing.assert_allclose(response.outputs, [[1.25, 3.0, 5.25]])
    np.testing.assert_allclose(response.states, [1.3125])


def test_impulse_response_matches_unit_area_convention():
    system = ss([[0.5]], [[1.0]], [[1.0]], [[0.0]], dt=0.2)

    response = impulse_response(system, T=np.arange(4) * 0.2, squeeze=False)

    np.testing.assert_allclose(response.outputs, [[[0.0, 5.0, 2.5, 1.25]]])
