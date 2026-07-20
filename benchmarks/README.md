# SIPPY benchmarks

## Closed-loop identification

Run the closed-loop capability benchmark:

```bash
uv run python benchmarks/benchmark_closed_loop.py
```

The benchmark compares explicit `SSARX` with canonical `SUBSPACE`, both without
a measured reference (predictor route) and with one (two-stage ORT route). It
uses two complementary records:

- the ground-truth closed-loop example from the
  [MathWorks `n4sid` SSARX documentation](https://www.mathworks.com/help/ident/ref/n4sid.html),
  plus a two-independent-excitation variant that can exercise SIPPY's
  two-stage ORT route, with held-out open-loop simulation, frequency-response,
  pole, stability, route-selection, and timing metrics;
- the CC BY 4.0 OpenMCT DC-motor controller-validation data published by Von
  Chong and Cardenas, using measured reference, PWM actuation, and motor speed.

The experimental archive is downloaded from Mendeley Data and checksum-checked.
To reuse a local copy, pass `--archive path/to/archive.zip`; to run only the
network-independent ground-truth workload, pass `--skip-motor`.

Dataset citation: A. Von Chong and D. Cardenas, *Dataset for an end-to-end
open-source DC motor control workflow: current calibration, system
identification, and controller validation*, Mendeley Data, version 1, 2026,
[doi:10.17632/5xvg43r9r8.1](https://doi.org/10.17632/5xvg43r9r8.1).

A reference arm64 macOS run used 3,000 synthetic training samples and three
post-warmup repetitions. NRMSE is measured on held-out data; FRF and pole
columns are relative errors against known ground truth or, for the motor, the
published open-loop fit.

| Dataset | Method | Reference | Route | Median | NRMSE | FRF error | Pole error |
|---|---|---|---|---:|---:|---:|---:|
| MathWorks SSARX | SUBSPACE | unavailable | predictor | 13.7 ms | 1.588 | 1.367 | 1.194 |
| MathWorks SSARX | SUBSPACE | measured | predictor fallback | 25.6 ms | 1.588 | 1.367 | 1.194 |
| MathWorks SSARX | SSARX | not used | SSARX | 4.61 ms | 0.0161 | 0.0159 | 0.00263 |
| Two excitations | SUBSPACE | unavailable | predictor | 13.8 ms | 0.0188 | 0.0187 | 0.107 |
| Two excitations | SUBSPACE | measured | two-stage ORT | 14.9 ms | 0.0334 | 0.0332 | 0.395 |
| Two excitations | SSARX | not used | SSARX | 4.42 ms | 0.0224 | 0.0223 | 0.189 |
| OpenMCT motor | SUBSPACE | unavailable | predictor | 7.09 ms | 0.0312 | 0.151 | 0.0938 |
| OpenMCT motor | SUBSPACE | measured | predictor fallback | 9.84 ms | 0.0312 | 0.151 | 0.0938 |
| OpenMCT motor | SSARX | not used | SSARX | 2.30 ms | 0.0315 | 0.140 | 0.0875 |

The single-excitation MathWorks record and the motor's step-reference record
both trigger `reference_deterministic_regressor_rank_deficient`, so measured
references fall back to the predictor estimator. The two-independent-
excitation case proves that the ORT path is reachable and accurately recovers
the plant response. SSARX directly reconstructs the difficult single-excitation
MathWorks plant without a measured-reference channel or fallback. The motor
NRMSE is evaluated on a held-out closed-loop segment; its FRF and pole errors
against the separately published open-loop fit are the stronger plant-recovery
checks. The `MathWorks SSARX` dataset name records provenance and does not imply
that a row used the SSARX method; the `Method` column states the estimator.

## Subspace identification

Run the loop-agnostic estimator comparison:

```bash
uv run python benchmarks/benchmark_subspace.py
```

This compares ordinary option-free `SUBSPACE` with compact N4SID and
QR-reusing PARSIM-K on the same 2-state SISO and 3-state MIMO records. The
named algorithms receive their known order and a 15-row horizon; canonical
`SUBSPACE` owns validation, horizon candidates, order selection, CVA weighting,
and realization, so its timing includes the automatic-selection work avoided
by expert named calls.

A quick arm64 macOS run on Python 3.13.5 and NumPy 2.3.5 used 2,500 samples and
two post-warmup repetitions:

| Dataset | Method | Median | Traced peak | Order |
|---|---|---:|---:|---:|
| SISO | SUBSPACE | 370.2 ms | 11.30 MiB | 2 |
| SISO | N4SID | 400.2 ms | 4.06 MiB | 2 |
| SISO | PARSIM-K | 8.23 ms | 3.93 MiB | 2 |
| MIMO | SUBSPACE | 496.6 ms | 26.88 MiB | 3 |
| MIMO | N4SID | 424.7 ms | 8.19 MiB | 3 |
| MIMO | PARSIM-K | 20.05 ms | 7.89 MiB | 3 |

Peak values are Python-visible allocations reported by `tracemalloc`, not
process RSS. The same run reports structural bounds alongside timing: for
2,471 usable Hankel columns, compact LQ factors have maximum dimensions 60
(SISO) and 120 (MIMO), while reusable predictor QR factors have at most 32 and
64 rows. Thus the factorization dimensions depend on channel count and horizon,
not a 2,471-by-2,471 sample-space projector. Persistent structural gates in
`test_subspace_lq_compression.py` and the PARSIM reimplementation tests enforce
these bounds and factorization reuse.

Run the seven-method Monte Carlo reconstruction grid separately:

```bash
uv run python benchmarks/benchmark_subspace.py \
  --scenario-grid --samples 2500 --seeds 5
```

Each seed creates independent training and validation records. All methods use
the known plant order, a 12-row future horizon, and, where applicable, a
24-row past horizon. The reported errors are medians across seeds and compare
held-out output simulation, frequency response, poles, and Markov parameters,
so they do not depend on the identified state coordinates.

An arm64 macOS run on Python 3.13.5 and NumPy 2.3.5 completed all five seeds for
all 28 method/scenario combinations. Median held-out NRMSE was:

| Scenario | SSARX | N4SID | MOESP | CVA | PARSIM-K | PARSIM-S | PARSIM-P |
|---|---:|---:|---:|---:|---:|---:|---:|
| Open-loop SISO | 0.00100 | 0.00102 | 0.00102 | 0.00102 | 0.00105 | 0.109 | 0.0982 |
| Open-loop MIMO | 0.00124 | 0.00118 | 0.00118 | 0.00118 | 0.00125 | 0.0203 | 0.0148 |
| Closed-loop MathWorks | 0.0181 | 0.364 | 0.346 | 0.314 | 0.0217 | 0.699 | 0.555 |
| Closed-loop two excitation | 0.0230 | 0.0212 | 0.0212 | 0.0212 | 0.0206 | 0.358 | 0.0533 |

Median fit times on the open-loop records were:

| Method | SISO | MIMO |
|---|---:|---:|
| SSARX | 4.77 ms | 12.63 ms |
| N4SID | 446.86 ms | 409.43 ms |
| MOESP | 443.03 ms | 407.52 ms |
| CVA | 444.36 ms | 406.97 ms |
| PARSIM-K | 11.88 ms | 31.15 ms |
| PARSIM-S | 13.30 ms | 23.21 ms |
| PARSIM-P | 8.51 ms | 18.32 ms |

SSARX is the strongest estimator on the single-excitation MathWorks loop and
matches the classical methods on the open-loop records while fitting much
faster. It is not universally best: N4SID, MOESP, CVA, and PARSIM-K have a
small accuracy advantage on the two-excitation closed-loop record, and the
three classical methods have a small NRMSE advantage on the open-loop MIMO
record. PARSIM-S and PARSIM-P are fast but substantially less reliable across
this scenario set.

## Model operations

Run the current ctrlsys-backed implementation:

```bash
uv run python benchmarks/benchmark_systems.py
```

Compare it with python-control 0.10.2 and Slycot 0.7.0 in the same NumPy and
SciPy environment:

```bash
uv run --with control==0.10.2 --with slycot==0.7.0 \
  python benchmarks/benchmark_systems.py --compare-control
```

Set BLAS thread counts when comparing runs across commits:

```bash
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  uv run --with control==0.10.2 --with slycot==0.7.0 \
  python benchmarks/benchmark_systems.py --compare-control
```

## Reference result

The reference run used Python 3.13.5 and NumPy 2.3.5 on arm64 macOS. Times are
medians after warmup. A speedup above one means SIPPY is faster than the
python-control/Slycot baseline.

| Workload | SIPPY | Baseline | Speedup |
|---|---:|---:|---:|
| SISO transfer to state space | 0.0087 ms | 0.2992 ms | 34.45x |
| MIMO transfer to state space | 0.0322 ms | 0.8573 ms | 26.64x |
| Shared-dynamics 4x4 transfer to state space | 0.3050 ms | 5.9205 ms | 19.41x |
| 40-state state space to transfer | 0.2064 ms | 0.4406 ms | 2.13x |
| 512-point SISO transfer response | 0.0215 ms | 0.0474 ms | 2.20x |
| 512-point MIMO transfer response | 0.0373 ms | 0.0729 ms | 1.95x |
| 512-point, 40-state response | 2.4237 ms | 3.6232 ms | 1.49x |
| Short forced response | 0.0088 ms | 0.1235 ms | 14.01x |
| 20,000-sample forced response | 12.2633 ms | 61.6775 ms | 5.03x |

The shared-dynamics realization has 12 states in both implementations. The
maximum frequency-response difference was `1.63e-14`; the maximum simulation
difference was `1.39e-16`.

## Routine selection

- Transfer-function frequency response uses vectorized NumPy polynomial
  evaluation. The ctrlsys `td05ad` binding accepts only one real frequency per
  call, so using it would retain the dominant Python loop.
- MIMO transfers with repeated exact denominators use full polynomial-matrix
  `tc04ad` followed by `tb01pd`. Independent channel dynamics retain the faster
  per-channel `tc04ad` path.
- State-space sweeps choose among direct `tb05ad`, batched NumPy solves,
  `tb01wd` plus Hessenberg-mode `tb05ad`, and SciPy Hessenberg reduction plus
  Hessenberg-mode `tb05ad` based on measured state and frequency crossovers.
- Simulation stays on `tf01md`. A per-call `tb01wd` plus `tf01nd` path did not
  recover its transformation cost, and direct `tf01nd` regressed at several
  state dimensions. SIPPY passes existing Fortran arrays directly because the
  binding treats the matrices and input sequence as read-only.
