# SIPPY contributor guide

SIPPY is a Python system-identification library with a factory-based API. The
`main` branch is the maintained implementation and source of truth.

## Commands

```bash
uv sync
uv run ruff check src/
uv run ruff format src/
uv run pytest
uv run pytest src/sippy/identification/tests/test_factory.py
uv run python Examples/example_new_architecture.py
```

Use UV for dependencies and commands; do not install project dependencies with
`pip`.

## Architecture

- `src/sippy/identification/base.py`: algorithm interface and model types
- `src/sippy/identification/factory.py`: algorithm registry
- `src/sippy/identification/iddata.py`: input/output data container
- `src/sippy/identification/algorithms/`: identification implementations
- `src/sippy/filters/`: preprocessing filters
- `src/sippy/utils/`: numerical, signal, and simulation helpers
- `src/sippy/identification/tests/`: unit, parity, and simulation tests

All identification algorithms implement:

```python
def identify(
    self,
    y: Optional[np.ndarray] = None,
    u: Optional[np.ndarray] = None,
    iddata: Optional["IDData"] = None,
    **kwargs,
) -> StateSpaceModel:
```

Accept either `(y, u)` or `iddata`, never both. Preserve the `IDData` sample
time in returned models.

## Development rules

- Write persistent tests before changing algorithm behavior.
- Preserve numerical accuracy and cover regressions with behavioral assertions.
- Test applicable SISO and MIMO cases, delays, low and high SNR, correlated
  inputs, colored noise, and unstable dynamics.
- Use `sippy.systems` for transfer functions, state-space models, frequency
  response, and simulation. Pass discrete sample time as `dt=...`; state
  matrices use uppercase attributes. Keep raw `ctrlsys` calls isolated in the
  systems backend, pass Fortran-order copies, and check every `info` code.
- Put algorithm implementations in `identification/algorithms/` and register
  them in `algorithms/__init__.py`.
- Reuse `compiled_utils.py` only when it preserves correctness and robustness.
- Do not add inline comments unless the logic is important and counterintuitive.
- Keep changes focused and commits atomic.

Before committing, run Ruff and the relevant tests. Run the full test suite for
algorithm, API, or shared numerical changes.

## GitHub

- Develop against `main`; new work branches from `main` and targets `main`.
- Use `origin` to select the GitHub account and authenticate with `gh`.
- `jamestjsp` is the personal account; `jamestjst` is the work account.
