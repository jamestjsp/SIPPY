import numpy as np
import pytest

from sippy import systems as control
from sippy.identification.algorithms.armax import ARMAXAlgorithm
from sippy.identification.algorithms.armax_modes import (
    ILLSHandler,
    OPTHandler,
    RLLSHandler,
)
from sippy.identification.base import StateSpaceModel


def _simulate_armax(na, nb, nc, theta, n_samples=400):
    rng = np.random.default_rng(7)
    nu = nb.shape[1]
    ny = nb.shape[0]
    u = rng.normal(scale=0.3, size=(nu, n_samples))
    noise = rng.normal(scale=0.05, size=(ny, n_samples))
    y = np.zeros((ny, n_samples))

    warmup = max(int(na.max()), int((nb + theta).max()), int(nc.max()), 1) + 1
    for k in range(warmup, n_samples):
        for i in range(ny):
            ar = 0.0
            if na[i]:
                idx = k - np.arange(1, na[i] + 1)
                ar = np.dot(y[i, idx], 0.2 * np.ones(na[i]))

            ma = 0.0
            if nc[i]:
                idx = k - np.arange(1, nc[i] + 1)
                ma = np.dot(noise[i, idx], 0.4 * np.ones(nc[i]))

            reg = 0.0
            for j in range(nu):
                order = nb[i, j]
                if order == 0:
                    continue
                idx = k - theta[i, j] - np.arange(1, order + 1)
                reg += 0.5 * np.dot(u[j, idx], np.ones(order))

            y[i, k] = -ar + reg + ma + noise[i, k]

    return y, u


def test_armax_default_nlp_runs():
    na = np.array([2])
    nb = np.array([[2]])
    nc = np.array([1])
    theta = np.array([[1]])
    y, u = _simulate_armax(na, nb, nc, theta)

    algo = ARMAXAlgorithm()
    model = algo.identify(
        y=y,
        u=u,
        na=2,
        nb=2,
        nc=1,
        theta=1,
        tsample=1.0,
        max_iterations=10,
    )

    assert isinstance(model, StateSpaceModel)
    assert isinstance(model.G, control.StateSpace)
    assert isinstance(model.G_tf, control.TransferFunction)
    assert isinstance(model.H_tf, control.TransferFunction)
    assert model.G.dt == pytest.approx(1.0)
    assert model.B.shape[1] == u.shape[0]
    assert model.Yid.shape == y.shape


def test_armax_ills_mode():
    na = np.array([2])
    nb = np.array([[2]])
    nc = np.array([1])
    theta = np.array([[1]])
    y, u = _simulate_armax(na, nb, nc, theta)

    algo = ARMAXAlgorithm()
    model = algo.identify(
        y=y,
        u=u,
        na=2,
        nb=2,
        nc=1,
        theta=1,
        mode="ILLS",
        tsample=1.0,
        max_iterations=5,
    )

    assert isinstance(model, StateSpaceModel)
    assert isinstance(model.G, control.StateSpace)
    assert isinstance(model.G_tf, control.TransferFunction)
    assert isinstance(model.H_tf, control.TransferFunction)
    assert model.G_tf.dt == pytest.approx(1.0)
    assert model.H_tf.dt == pytest.approx(1.0)
    assert model.A.shape[0] == model.A.shape[1]
    assert model.C.shape[0] == 1


@pytest.mark.parametrize("nu", [1, 2])
def test_armax_mimo_support(nu):
    na = np.array([2, 2])
    nb = np.full((2, nu), 1)
    nc = np.array([1, 1])
    theta = np.zeros_like(nb)
    y, u = _simulate_armax(na, nb, nc, theta, n_samples=300)

    algo = ARMAXAlgorithm()
    model = algo.identify(
        y=y,
        u=u,
        na=na.tolist(),
        nb=nb.tolist(),
        nc=nc.tolist(),
        theta=theta.tolist(),
        mode="ILLS",
        tsample=1.0,
        max_iterations=5,
    )

    assert isinstance(model, StateSpaceModel)
    assert model.C.shape[0] == y.shape[0]
    assert model.B.shape[1] == u.shape[0]


@pytest.mark.parametrize(
    ("handler", "creator_name"),
    [
        (ILLSHandler(), "_create_state_space_model"),
        (RLLSHandler(), "_create_state_space_model_rlls"),
        (OPTHandler(), "_create_state_space_model_opt"),
    ],
)
def test_legacy_armax_mode_handlers_build_control_models(handler, creator_name):
    creator = getattr(handler, creator_name)
    model = creator(
        np.array([0.2, 0.5, 0.1]),
        na=1,
        nb=1,
        nc=1,
        nk=1,
        ny=1,
        nu=1,
        Ts=0.25,
    )

    assert isinstance(model, StateSpaceModel)
    assert isinstance(model.G, control.StateSpace)
    assert isinstance(model.G_tf, control.TransferFunction)
    assert isinstance(model.H_tf, control.TransferFunction)
    assert model.G_tf.dt == pytest.approx(0.25)
    assert model.H_tf.dt == pytest.approx(0.25)
    np.testing.assert_allclose(model.G_tf.num[0][0], [0.5, 0.0])
    np.testing.assert_allclose(model.G_tf.den[0][0], [1.0, 0.2, 0.0])
