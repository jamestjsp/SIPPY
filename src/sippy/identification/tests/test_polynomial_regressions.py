import numpy as np

from sippy import systems as control
from sippy.identification.algorithms.armax_modes import ILLSHandler, RLLSHandler
from sippy.identification.algorithms.arx import ARXAlgorithm
from sippy.identification.algorithms.opt_support import MISOResult


def _simulate_arx(u, ar_coeffs, input_gain, nk):
    y = np.zeros_like(u)
    for sample in range(max(len(ar_coeffs), nk), u.size):
        history = y[sample - len(ar_coeffs) : sample][::-1]
        y[sample] = np.dot(ar_coeffs, history) + input_gain * u[sample - nk]
    return y


def test_miso_transfer_function_preserves_requested_input_delay():
    result = MISOResult(
        a_coeffs=np.array([-0.6, 0.1]),
        b_coeffs=[np.array([1.0])],
        c_coeffs=np.array([]),
        d_coeffs=np.array([]),
        f_coeffs=np.array([]),
        delay=np.array([0]),
        y_hat=np.zeros(8),
        fit_start=2,
        noise_variance=0.0,
        reached_max=False,
        y_std=1.0,
        u_std=np.ones(1),
    )

    process, _ = result.build_transfer_function(sample_time=1.0)
    impulse = np.r_[1.0, np.zeros(7)]
    response = np.asarray(control.forced_response(process, U=impulse).outputs)

    np.testing.assert_allclose(response[:4], [0.0, 1.0, 0.6, 0.26])


def test_ills_uses_nk_as_the_first_input_coefficient_delay():
    rng = np.random.default_rng(914)
    u = rng.normal(size=800)
    y = _simulate_arx(u, np.array([0.6, -0.1]), input_gain=0.8, nk=1)

    model, info = ILLSHandler().identify(
        u=u,
        y=y,
        na=2,
        nb=1,
        nc=0,
        nk=1,
        max_iterations=10,
    )

    assert model is not None, info
    np.testing.assert_allclose(model.G_tf.den[0][0], [1.0, -0.6, 0.1], atol=1e-10)
    np.testing.assert_allclose(model.G_tf.num[0][0], [0.8, 0.0], atol=1e-10)
    np.testing.assert_allclose(model.Yid[0, 2:], y[2:], atol=1e-10)


def test_rlls_updates_parameters_and_uses_requested_delay():
    rng = np.random.default_rng(122)
    u = rng.normal(size=1200)
    y = _simulate_arx(u, np.array([0.55]), input_gain=0.75, nk=1)

    model, info = RLLSHandler().identify(
        u=u,
        y=y,
        na=1,
        nb=1,
        nc=1,
        nk=1,
        max_iterations=1,
    )

    assert model is not None, info
    np.testing.assert_allclose(info["final_parameters"][:2], [-0.55, 0.75], atol=1e-5)
    np.testing.assert_allclose(model.Yid[0, -100:], y[-100:], atol=1e-6)


def test_arx_noise_model_is_inverse_ar_polynomial():
    algorithm = ARXAlgorithm()
    _, noise = algorithm._create_transfer_functions_arx(
        A_coeffs=np.array([[0.6, -0.1]]),
        B_coeffs=np.array([[0.8]]),
        na=2,
        nb=1,
        nk=1,
        ny=1,
        nu=1,
        Ts=1.0,
    )

    impulse = np.r_[1.0, np.zeros(5)]
    response = np.asarray(control.forced_response(noise, U=impulse).outputs)

    np.testing.assert_allclose(response[:4], [1.0, 0.6, 0.26, 0.096])


def test_mimo_arx_noise_model_filters_cross_coupled_innovations():
    algorithm = ARXAlgorithm()
    ar_coeffs = np.array([[[0.5, 0.2]], [[-0.1, 0.4]]])
    _, noise = algorithm._create_transfer_functions_arx(
        A_coeffs=ar_coeffs,
        B_coeffs=np.zeros((2, 2)),
        na=1,
        nb=1,
        nk=1,
        ny=2,
        nu=2,
        Ts=1.0,
    )

    innovations = np.zeros((2, 4))
    innovations[0, 0] = 1.0
    response = np.asarray(control.forced_response(noise, U=innovations).outputs)

    expected = np.zeros((2, 4))
    expected[:, 0] = [1.0, 0.0]
    for sample in range(1, expected.shape[1]):
        expected[:, sample] = ar_coeffs[:, 0, :] @ expected[:, sample - 1]
    np.testing.assert_allclose(response, expected)
