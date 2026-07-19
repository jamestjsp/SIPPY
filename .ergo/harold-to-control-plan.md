# Establish the python-control and Slycot foundation
## Goal
- Make python-control and Slycot the required model backend and remove Harold from the dependency contract.
- Define the control-native model and conversion boundary used by every identification algorithm.

## Context
- Keep SIPPY's `StateSpaceModel` wrapper and factory API.
- `StateSpaceModel.G` must become a `control.StateSpace`; `G_tf` and `H_tf` must be `control.TransferFunction` objects when the corresponding model exists.
- Use `control>=0.10.2` and `slycot>=0.7.0`; do not retain an optional-dependency or mock-model path for Harold.

## Acceptance Criteria
- Persistent tests are written before implementation changes and assert control-native object types, uppercase state matrices, sample-time preservation, and SISO/MIMO dimensions.
- `pyproject.toml`, `setup.py`, `uv.lock`, and the Docker image install control and Slycot without Harold.
- Shared transfer-to-state realization uses python-control with the Slycot method where needed and returns real NumPy matrices with stable shape conventions.
- `StateSpaceModel.G` is constructed with `control.ss(..., dt=ts)` whenever the model has inputs.

## Validation Gates
- `uv run pytest src/sippy/identification/tests/test_base.py`
- `uv run ruff check src/sippy/identification/base.py src/sippy/identification/tests/test_base.py`

## Completion Note
- Replaced the shared Harold realization and `StateSpaceModel.G` construction with python-control.
- Required control 0.10.2 and Slycot 0.7.0 across project, legacy, Docker, and locked dependency metadata.
- Added SISO/MIMO realization, timebase, dimensions, and no-input behavioral regressions; 11 base tests and Ruff pass.
---
# Migrate direct polynomial identification algorithms
## Goal
- Replace Harold construction and availability branches in ARX, FIR, OE, GEN, and BJ with python-control.

## Acceptance Criteria
- Tests are updated first to assert behavioral transfer-function coefficients, delays, sample time, and state-space simulation rather than mocked Harold availability.
- ARX, FIR, OE, GEN, and BJ return control-native deterministic and noise transfer functions for supported SISO/MIMO structures.
- Harold polynomial multiplication is replaced by NumPy operations with identical coefficient ordering.
- No Harold imports, flags, warnings, fallback names, or lowercase matrix access remain in these modules.
- Existing low/high SNR, delays, colored noise, and SISO/MIMO behavioral accuracy remains within current tolerances.

## Validation Gates
- `uv run pytest src/sippy/identification/tests/test_arx_algorithm.py src/sippy/identification/tests/test_fir_algorithm.py src/sippy/identification/tests/test_oe_algorithm.py src/sippy/identification/tests/test_gen.py src/sippy/identification/tests/test_bj_algorithm.py`
- `uv run pytest src/sippy/identification/tests/test_core_io_simulation_accuracy.py src/sippy/identification/tests/test_polynomial_io_simulation_accuracy.py`
- `uv run ruff check src/sippy/identification/algorithms/`

## Completion Note
- Replaced Harold construction and mock fallbacks in ARX, FIR, OE, GEN, BJ, and the shared MISO realization boundary with python-control.
- Added control-native SISO/MIMO transfer-function, delay, coefficient, timebase, and solver-failure regressions.
- Corrected the zero-output-AR optimization path and passed the combined 122-test gate plus Ruff.
---
# Migrate optimization-backed polynomial algorithms
## Goal
- Replace Harold in the shared optimization support and ARMAX, ARARMAX, and ARARX realization paths.

## Acceptance Criteria
- Persistent tests are updated first for control-native transfer functions and state realizations.
- `opt_support.py`, `armax_modes.py`, `ararmax.py`, and `ararx.py` use control factories, NumPy polynomial multiplication, and uppercase matrices.
- MISO and MIMO transfer matrices preserve input delays and per-channel numerator/denominator structure.
- Noise-model realization preserves Kalman-gain/state dimensions and sample time.
- No Harold availability fallback remains.

## Validation Gates
- `uv run pytest src/sippy/identification/tests/test_armax_algorithm.py src/sippy/identification/tests/test_ararmax_algorithm.py src/sippy/identification/tests/test_ararx_algorithm.py src/sippy/identification/tests/test_ararx_mimo.py`
- `uv run pytest src/sippy/identification/tests/test_master_comparison.py`
- `uv run ruff check src/sippy/identification/algorithms/`

## Completion Note
- Replaced Harold in all legacy ARMAX mode handlers and the ARARMAX simplified path with python-control transfer functions and StateSpaceModel construction.
- Removed availability branches and the random fake-model fallback, preserved delay polynomials, and updated reference tests for control coefficient indexing.
- Passed the combined 76-test optimization and master-reference gate plus Ruff.
---
# Migrate FRF realization and simulation infrastructure
## Goal
- Replace Harold in parametric FRF realization, FIR/step-response utilities, and independent simulation scenarios.

## Acceptance Criteria
- Tests are updated first to cover `control.tf2ss`, `control.forced_response`, and `control.impulse_response` behavior with explicit shape assertions.
- SISO and MIMO FRF fits produce control transfer functions and Slycot-backed state-space models with the original sample time.
- Simulation fixtures use control systems as the independent plant/noise source and preserve output-by-sample orientation.
- FIR coefficient and step-response utilities no longer produce random fallback results and work for SISO and MIMO control systems.
- Stable, unstable, delayed, correlated-input, colored-noise, and low/high-SNR scenarios keep their behavioral coverage.

## Validation Gates
- `uv run pytest src/sippy/identification/tests/test_frf_fit.py src/sippy/identification/tests/test_simulation_scenarios.py src/sippy/identification/tests/test_arma_simulation_accuracy.py src/sippy/identification/tests/test_subspace_simulation_accuracy.py`
- `uv run ruff check src/sippy/identification/frf_fit.py src/sippy/utils/simulation_utils.py src/sippy/identification/tests/simulation_scenarios.py`

## Completion Note
- Migrated FRF transfer realization, independent plant/noise scenarios, and held-out simulations to python-control.
- Replaced random FIR fallback behavior with control impulse responses supporting SIPPY wrappers, SISO/MIMO systems, and non-unit sample times.
- Passed the combined 77-test behavioral gate for FRF, scenario, ARMA, subspace, core, and polynomial accuracy plus Ruff.
---
# Update examples, documentation, and terminology
## Goal
- Present python-control and Slycot as SIPPY's supported control-system integration everywhere users and contributors see it.

## Acceptance Criteria
- README, USER_GUIDE, AGENTS.md, Dockerfile, examples, test names, local variables, messages, and docstrings contain no stale Harold API guidance.
- Examples use python-control matrices, time-response APIs, and transfer-function data shapes correctly.
- Installation guidance explains that Slycot wheels are available from PyPI and that Slycot is required.
- `Examples/example_new_architecture.py` runs successfully.

## Validation Gates
- `uv run python Examples/example_new_architecture.py`
- `uv run ruff check src/ Examples/`
- Repository search excluding `.git`, `.venv`, and `.ergo` finds no `harold` references.

## Completion Note
- Updated contributor guidance, installation documentation, user-guide integration examples, maintained examples, test names, and comparison variables for python-control and Slycot.
- Applied and resolved Ruff's maintained-example lint findings so the full source/example lint gate passes.
- Verified zero Harold references, ran the architecture example successfully, and passed 47 affected comparison and behavioral tests.
---
# Validate and audit the complete migration
## Goal
- Prove the maintained implementation is fully control-native and numerically sound.

## Acceptance Criteria
- Ruff check and formatting pass for source and maintained examples.
- The complete test suite passes from the synchronized UV environment.
- The dependency graph contains control and Slycot but not Harold.
- Runtime smoke checks import Slycot, create SISO and MIMO control models, run identification, and preserve `IDData.sample_time`.
- A final source, tests, packaging, examples, docs, and lockfile audit finds no remaining Harold imports, names, fallbacks, or API assumptions.

## Validation Gates
- `uv sync`
- `uv run ruff check src/ Examples/`
- `uv run ruff format --check src/ Examples/`
- `uv run pytest`
- `uv run python Examples/example_new_architecture.py`

## Completion Note
- Synchronized the UV environment, which removed Harold and retained control 0.10.2 plus Slycot 0.7.0 in the dependency graph.
- Passed Ruff lint and formatting, all 523 tests, the architecture example, and explicit Slycot SISO/MIMO plus IDData sample-time smokes.
- Completed a zero-reference audit across source, tests, packaging, examples, documentation, and the lockfile.
