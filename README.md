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

- State space: canonical SUBSPACE, SSARX, N4SID, MOESP, CVA, PARSIM-K,
  PARSIM-S, PARSIM-P
- Input/output: ARX, ARMAX, ARARX, ARARMAX, FIR, OE, ARMA, BJ
- Generalized polynomial model: GEN
- Nonparametric frequency response: FD

All 18 registered algorithm entries share the same parameterless factory and
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

model = sippy.identify(y, u)

print(model.n)
print(model.is_stable())
x, y_simulated = model.simulate(u)
print(model.fit())
print(model.residual_covariance)
print(model.capabilities)
```

`sippy.identify(y, u)` is the canonical state-space workflow for both open-
and closed-loop records. It chooses feasible block horizons and model order,
uses a predictor-form estimator that remains consistent when feedback
correlates plant inputs with output innovations, and applies CVA weighting to
the resulting subspace. The caller does not classify the experiment or select
a closed-loop algorithm:

```python
options = {}
open_loop_model = sippy.identify(y_open, u_open, **options)
closed_loop_model = sippy.identify(y_closed, u_closed, **options)
```

When measured exogenous excitation is available, declare it as reference data
and the same call automatically uses compact two-stage ORT projection:

```python
from sippy.identification.iddata import IDData

data = IDData(
    frame,
    inputs=["plant_input"],
    outputs=["plant_output"],
    references=["setpoint", "input_dither"],
    tsample=0.1,
)
model = sippy.identify(data=data)
print(model.identification_info["estimator_route"])
```

Feedback correlation and MIMO input collinearity are different problems.
Feedback makes plant inputs statistically dependent on innovations and can
bias ordinary open-loop projections. Collinearity is correlation among input
channels and primarily reduces numerical rank and excitation; neither ORT nor
CVA can restore directions that were never excited. Informative measured
references can reduce closed-loop variance, while absent or rank-deficient
references safely leave the canonical estimator on its predictor route. See
the predictor-form [PARSIM analysis](https://skoge.folk.ntnu.no/prost/proceedings/dycops-2010/Papers_DYCOPS2010/MoAT4-03.pdf)
and Katayama and Tanaka's
[two-stage ORT method](https://doi.org/10.1016/j.automatica.2007.02.011).

Jansson's SSARX is also available as an explicit closed-loop estimator. It
fits one high-order VARX predictor, uses its Markov parameters to remove future
input and output terms, estimates the state sequence by canonical correlation,
and recovers the innovations model by linear regression:

```python
model = sippy.identify(
    y_closed,
    u_closed,
    method="SSARX",
    ss_f=12,
    ss_p=24,
    ss_fixed_order=2,
)
```

`ss_p` is the high-order VARX/past window and must be at least `ss_f - 1`.
SSARX defaults to `D = 0`, the consistency condition used by Jansson for
closed-loop data with instantaneous feedback. Set `ss_d_required=True` only
when direct feedthrough is part of a well-posed experiment without an
instantaneous feedback algebraic loop.

The deterministic process model is available as `model.deterministic_model`;
the innovations model, when identified, is `model.innovations_model`.
`predict(u=...)` performs deterministic prediction, while
`predict(u=..., y=...)` performs one-step-ahead prediction when an estimated
Kalman gain or causally invertible innovations model is available. `Vn` is the
mean squared identification residual after the model warm-up interval;
`residual_covariance` retains the full output covariance matrix. `K`, `Q`, `R`,
and `S` are `None` unless the selected estimator actually identifies them.

Named algorithms remain available as advanced compatibility entry points and
can also be created directly:

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
