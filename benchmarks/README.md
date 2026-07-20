# SIPPY benchmarks

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
