# SIPPY

SIPPY (Systems Identification Package for Python) identifies discrete-time
linear models from measured input-output data. The maintained `main` branch
provides a typed, factory-based API for transfer-function and state-space
models.

SIPPY was originally developed by Giuseppe Armenise at the Department of Civil
and Industrial Engineering of the University of Pisa under the supervision of
[Prof. Gabriele Pannocchia](https://people.unipi.it/gabriele_pannocchia/). It is
distributed under the LGPL license.

## Algorithms

- State space: N4SID, MOESP, CVA, PARSIM-K, PARSIM-S, PARSIM-P
- Input/output: ARX, ARMAX, ARARX, ARARMAX, FIR, OE, ARMA, BJ
- Generalized polynomial model: GEN
- Nonparametric frequency response: FD

All 16 algorithms share the same parameterless factory and
`identify(y=None, u=None, iddata=None, **options)` algorithm interface. The
primary `sippy.identify(...)` function accepts raw arrays or `IDData` and
returns an `IdentificationResult` (`StateSpaceModel` remains its compatibility
name).
Every result exposes the same analysis methods and reports support through
`model.supports(operation)`. Unsupported operations raise `NotImplementedError`
instead of returning fabricated state or covariance values.

The test suite covers SISO and MIMO systems, multiple excitation types and SNR
levels, correlated inputs, colored and cross-correlated noise, unstable
dynamics, long delays, API consistency, and numerical parity.

## Installation

SIPPY requires Python 3.13 or newer and uses
[UV](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

[ctrlsys](https://pypi.org/project/ctrlsys/) supplies the C11-backed SLICOT
routines used for transfer-function realization, frequency response, and
discrete simulation. SIPPY exposes these routines through its `sippy.systems`
model API, so callers do not handle Fortran-order work arrays or routine status
codes directly. CasADi and IPOPT support nonlinear polynomial-model estimation.
NumPy, SciPy, Pandas, and Numba provide the numerical runtime.

## Quick start

```python
import numpy as np

import sippy

rng = np.random.default_rng(42)
u = rng.standard_normal((2, 1000))
y = rng.standard_normal((1, 1000))

model = sippy.identify(
    y,
    u,
    method="n4sid",
    ss_f=20,
    ss_fixed_order=2,
)

print(model.n)
print(model.is_stable())
x, y_simulated = model.simulate(u)
print(model.fit())
print(model.residual_covariance)
print(model.capabilities)
```

The deterministic process model is available as `model.deterministic_model`;
the innovations model, when identified, is `model.innovations_model`.
`predict(u=...)` performs deterministic prediction, while
`predict(u=..., y=...)` performs one-step-ahead prediction when an estimated
Kalman gain or causally invertible innovations model is available. `Vn` is the
mean squared identification residual after the model warm-up interval;
`residual_covariance` retains the full output covariance matrix. `K`, `Q`, `R`,
and `S` are `None` unless the selected estimator actually identifies them.

Algorithms can also be created directly:

```python
from sippy.identification.factory import create_algorithm

model = create_algorithm("ARX").identify(
    y=y,
    u=u,
    na=2,
    nb=3,
    nk=1,
    tsample=1.0,
)
```

Each method declares the options it accepts. Unknown options warn and are
discarded instead of being silently ignored. Deprecated spellings such as
`dt`, `stab_marg`, `stab_cons`, `theta`, and ARMAX's `algorithm` are translated
to the canonical vocabulary with `DeprecationWarning`.

## Architecture

```text
src/sippy/
├── identification/
│   ├── __main__.py        # Functional API and compatibility facades
│   ├── base.py            # Algorithm and model abstractions
│   ├── factory.py         # Algorithm registry
│   ├── parameters.py      # Shared method and option vocabulary
│   ├── iddata.py          # Input/output data container
│   ├── algorithms/        # Identification implementations
│   └── tests/             # Unit, parity, and simulation tests
├── filters/               # Signal preprocessing
└── utils/                 # Numerical, simulation, and signal helpers
```

## Development

```bash
uv run ruff check src/
uv run ruff format src/
uv run pytest
uv run python Examples/example_new_architecture.py
```

Development branches from and targets `main`. See [AGENTS.md](AGENTS.md) for
repository conventions and [USER_GUIDE.md](USER_GUIDE.md) for detailed usage.
