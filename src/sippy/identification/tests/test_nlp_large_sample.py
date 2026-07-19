"""Large-sample robustness of the CasADi/IPOPT-based algorithms.

Regression coverage for a hard interpreter crash: CasADi's bundled MUMPS
selects the METIS fill-in ordering once the KKT system is large enough
(N >~ 1100 samples for structures with a D polynomial, whose NLP carries 3N
auxiliary variables), and that METIS build segfaults on macOS. opt_id now
pins the AMD ordering there, so these identifications must complete.
"""

import numpy as np
import pytest
from scipy import signal

from sippy.identification.factory import create_algorithm

N_LARGE = 1500


@pytest.fixture(scope="module")
def large_sample_data():
    rng = np.random.default_rng(7)
    u = np.sign(rng.normal(size=N_LARGE))
    y = signal.lfilter([0.0, 0.5, 0.3], [1.0, -1.2, 0.35], u)
    y += 0.01 * y.std() * rng.normal(size=N_LARGE)
    return y[None, :], u[None, :]


@pytest.mark.parametrize(
    "method,orders",
    [
        ("ARARX", {"na": 2, "nb": 2, "nd": 1}),
        ("BJ", {"nb": 2, "nc": 1, "nd": 1, "nf": 2}),
    ],
)
def test_d_polynomial_nlp_survives_large_samples(method, orders, large_sample_data):
    y, u = large_sample_data
    model = create_algorithm(method).identify(
        y=y, u=u, nk=1, max_iterations=100, **orders
    )
    assert np.all(np.isfinite(model.A))
    assert np.all(np.abs(np.linalg.eigvals(model.A)) < 1.0)
