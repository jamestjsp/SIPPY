# Establish the ctrlsys-backed model boundary
## Goal
- Replace SIPPY's direct python-control/Slycot model boundary with small SIPPY-owned state-space and transfer-function types backed by `ctrlsys` routines.
- Add `ctrlsys` as a required dependency while preserving sample time, SISO/MIMO shapes, and public model metadata.

## Context
- `ctrlsys` is a low-level C11/SLICOT binding rather than a high-level modeling package, so SIPPY must own the narrow model API it exposes.
- Use `tc04ad` for transfer-to-state conversion, `tb04ad` for state-to-transfer conversion, and `tb05ad` for state-space frequency response where their documented domains apply.
- Pass Fortran-order copies and reject every nonzero `info` result with a useful Python exception.

## Acceptance Criteria
- Persistent tests cover SISO/MIMO construction, conversion, timebase preservation, frequency response, and non-mutation of caller arrays.
- `StateSpaceModel.G`, `G_tf`, and `H_tf` use SIPPY-owned types instead of python-control objects.
- The new adapter contains the ctrlsys-specific array ordering and `info` handling in one place.
- The project dependency set includes `ctrlsys` and does not require callers to interact with raw SLICOT routine signatures.

## Validation Gates
- `/opt/homebrew/bin/uv run pytest src/sippy/identification/tests/test_base.py`
- `/opt/homebrew/bin/uv run ruff check src/sippy/`
---
# Migrate polynomial identification algorithms
## Goal
- Move ARX, FIR, OE, GEN, BJ, ARMAX, ARARMAX, and ARARX from python-control factories to the SIPPY ctrlsys-backed model API.

## Acceptance Criteria
- Tests assert transfer coefficients, delays, sample times, matrix dimensions, and held-out simulation behavior for SISO and MIMO paths.
- Algorithms no longer import python-control or rely on Slycot realization options.
- Deterministic and noise models retain the prior coefficient ordering and delay conventions.
- Conversion failures expose ctrlsys routine/context details rather than returning fake or partial models.

## Validation Gates
- `/opt/homebrew/bin/uv run pytest src/sippy/identification/tests/test_arx_algorithm.py src/sippy/identification/tests/test_fir_algorithm.py src/sippy/identification/tests/test_oe_algorithm.py src/sippy/identification/tests/test_gen.py src/sippy/identification/tests/test_bj_algorithm.py`
- `/opt/homebrew/bin/uv run pytest src/sippy/identification/tests/test_armax_algorithm.py src/sippy/identification/tests/test_ararmax_algorithm.py src/sippy/identification/tests/test_ararx_algorithm.py`
- `/opt/homebrew/bin/uv run ruff check src/sippy/identification/algorithms/`
---
# Migrate FRF and simulation infrastructure
## Goal
- Replace python-control frequency-response and time-response calls in FRF realization, SIPPY simulation helpers, and behavioral fixtures.

## Acceptance Criteria
- State-space frequency response uses the ctrlsys-backed boundary and includes feedthrough consistently.
- Forced, impulse, and state-space simulations preserve output-by-sample orientation, initial-state behavior, and discrete sample times.
- Stable, unstable, delayed, colored-noise, correlated-input, low-SNR, high-SNR, SISO, and MIMO regressions remain runnable.
- No runtime helper silently falls back to random or shape-only output.

## Validation Gates
- `/opt/homebrew/bin/uv run pytest src/sippy/identification/tests/test_frf_fit.py src/sippy/identification/tests/test_simulation_scenarios.py src/sippy/identification/tests/test_arma_simulation_accuracy.py src/sippy/identification/tests/test_subspace_simulation_accuracy.py`
- `/opt/homebrew/bin/uv run ruff check src/sippy/identification/frf_fit.py src/sippy/utils/simulation_utils.py`
---
# Update tests, examples, and user guidance
## Goal
- Make repository-facing examples, tests, and documentation demonstrate the SIPPY ctrlsys-backed API instead of python-control/Slycot.

## Acceptance Criteria
- Maintained tests use independent numerical assertions or SIPPY model types without importing python-control.
- Examples execute with the new model API and retain their documented outputs.
- README, contributor guidance, docstrings, and messages accurately describe ctrlsys as the C11 backend.
- The repository contains no stale python-control or Slycot usage outside historical `.ergo` records.

## Validation Gates
- `/opt/homebrew/bin/uv run python Examples/example_new_architecture.py`
- `/opt/homebrew/bin/uv run ruff check src/ Examples/`
- Repository search excluding `.git`, `.venv`, and historical `.ergo` records finds no runtime or documentation references to python-control or Slycot.
---
# Remove superseded dependencies and validate the migration
## Goal
- Remove python-control and Slycot from packaging and prove the complete ctrlsys migration on `main`.

## Acceptance Criteria
- `pyproject.toml`, legacy packaging metadata, Docker setup, and `uv.lock` require `ctrlsys` but not python-control or Slycot.
- Ruff check and formatting pass for source and maintained examples.
- The complete test suite passes with no masked skips or xfails introduced by the migration.
- Runtime smoke checks exercise SISO/MIMO conversion, simulation, frequency response, identification, and `IDData.sample_time` preservation.
- Local `HEAD`, `origin/main`, and the remote main ref match after publication.

## Validation Gates
- `/opt/homebrew/bin/uv sync`
- `/opt/homebrew/bin/uv run ruff check src/ Examples/`
- `/opt/homebrew/bin/uv run ruff format --check src/ Examples/`
- `/opt/homebrew/bin/uv run pytest`
- `/opt/homebrew/bin/uv run python Examples/example_new_architecture.py`
