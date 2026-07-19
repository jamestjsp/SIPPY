# Control backend benchmarks

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
