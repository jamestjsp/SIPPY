---
name: ctrlsys-control
description: This skill should be used when the user asks about ctrlsys routines, control theory functions, LQR design, Riccati solvers, Kalman filtering, system identification, or needs to find the right ctrlsys routine for a control systems task.
---

# ctrlsys Control Theory Library

C11 control systems library with Python bindings, published as ctrlsys on PyPI.

## Naming Convention

Routines follow `XX##YY` pattern:
- `XX` - 2-letter category prefix (AB, SB, MB, etc.)
- `##` - 2-digit subcategory number
- `YY` - variant suffix (MD=real, MZ=complex, ND=multi-input, etc.)

## Looking Up Routine Details

To get detailed documentation for any ctrlsys routine, use Python's help function:
```python
import ctrlsys
help(ctrlsys.sb02md)  # Full docstring with params, returns, examples
```

## Category Quick Reference

| Prefix | Domain | Key Functions |
|--------|--------|---------------|
| AB | Analysis | Controllability, observability, zeros, model reduction |
| AG | Analysis (Generalized) | Descriptor system analysis |
| SB | Synthesis | Riccati, Lyapunov, pole placement, H-inf, LQR/LQG |
| SG | Synthesis (Generalized) | Descriptor Riccati, Lyapunov |
| MB | Matrix | QR, SVD, eigenvalues, Hessenberg, Schur |
| MC | Matrix Construction | Polynomial operations |
| MA | Matrix Auxiliary | Helper matrix operations |
| MD | Matrix Decomposition | Advanced decompositions |
| TB | Transform (State-space) | Similarity, minimal realization, balancing |
| TC | Transform (Polynomial) | Coprime factorization, TF↔SS |
| TD | Transform (Data) | Transfer function manipulation |
| TF | Transform (Frequency) | Frequency response |
| TG | Transform (Descriptor) | Descriptor systems, staircase forms |
| IB | Identification | MOESP, N4SID subspace methods |
| NF | Nonlinear Filter | Nonlinear estimation |
| FB | Filter | Kalman filtering |
| FD | Filter Design | IIR/digital filter design |
| DE | Data Exchange | FFT, data conversion |
| BB, BD, DF, DG, DK | Utility | Basic ops, block diagonal, validation |
| UD, UE | Utility | Triangular, update/extract |

## Task-to-Routine Lookup

### Controller Design
| Task | Routine | Description |
|------|---------|-------------|
| LQR (continuous) | `sb02md` | Solve continuous ARE for optimal gain |
| LQR (discrete) | `sb02md` | Solve discrete ARE (DICO='D') |
| LQG | `sb02md` + `fb01qd` | ARE + Kalman filter |
| Pole placement | `sb01bd` | Multi-input eigenvalue assignment |
| Eigenstructure | `sb01dd` | Eigenvector assignment |
| H-infinity | `sb10ad` | Continuous H-inf controller |
| H-infinity (discrete) | `sb10dd` | Discrete H-inf controller |
| Deadbeat | `sb06nd` | Finite settling time |

### System Analysis
| Task | Routine | Description |
|------|---------|-------------|
| Controllability | `ab01nd` | Multi-input controllable realization |
| Observability | `ab01od` | Staircase form reduction |
| System zeros | `ab08nd` | Transmission zeros |
| Stability check | `ab09jx` | Stable/antistable decomposition |
| H-infinity norm | `ab13dd` | Compute ‖G‖∞ (L-infinity norm) |
| H-infinity norm (continuous, stable) | `ab13cd` | Compute ‖G‖∞ via bisection |
| H2 norm | `ab13bd` | Compute ‖G‖₂ |
| Hankel norm | `ab13ad` | Compute Hankel singular values |
| Structured singular value (mu) | `ab13md` | Upper bound on mu |
| Stability radius | `ab13ed` | Distance to instability |

### Model Reduction
| Task | Routine | Description |
|------|---------|-------------|
| Balanced truncation | `ab09ad` | Hankel-norm approximation |
| Balanced stochastic | `ab09bd` | Stochastic balancing |
| Frequency-weighted | `ab09cd` | Weighted balanced reduction |
| Singular perturbation | `ab09md` | Frequency-weighted SPA |
| Coprime factorization | `ab09fd` | Normalized coprime factors |
| Hankel MDA | `ab09hd` | Hankel minimum degree |

### Matrix Equations
| Equation | Routine | Description |
|----------|---------|-------------|
| Riccati (ARE) | `sb02md` | A'X + XA - XGX + Q = 0 |
| Riccati (generalized) | `sg02ad` | Generalized ARE for descriptor systems |
| Optimal gain | `sb02od` | Optimal state feedback via Riccati |
| Lyapunov | `sb03md` | A'X + XA + Q = 0 |
| Sylvester | `sb04md` | AX + XB = C |
| Generalized Lyapunov | `sg03ad` | A'XE + E'XA + Q = 0 |

### System Transformation
| Task | Routine | Description |
|------|---------|-------------|
| Continuous ↔ Discrete | `ab04md` | Bilinear transformation |
| State-space → Transfer fn | `tb04ad` | SS to transfer function |
| Transfer fn → State-space | `tc04ad` | Transfer function to SS |
| Minimal realization | `tb01pd` | Remove uncontrollable/unobservable |
| Frequency response (SS) | `tb05ad` | G(jw) from state-space |
| Frequency response (TF) | `td05ad` | G(jw) from transfer function |

### System Identification
| Task | Routine | Description |
|------|---------|-------------|
| Order estimation | `ib01ad` | MOESP/N4SID preprocessing |
| Matrix estimation | `ib01bd` | A, B, C, D from data |
| Kalman gain | `ib01bd` | With JOBCK='K' |

## Common Patterns

### Array Order
All arrays use **Fortran column-major order**:
```python
import numpy as np
A = np.array([[1, 2], [3, 4]], dtype=float, order='F')
```

### In-Place Array Modification Warning
**IMPORTANT:** ctrlsys functions may modify input arrays in-place. Always pass copies if you need to preserve originals:
```python
# BAD - A, B, C, D may be corrupted after call
A_d, B_d, C_d, D_d, info = ab04md('C', A, B, C, D, alpha=1.0, beta=2.0/dt)

# GOOD - originals preserved
A_d, B_d, C_d, D_d, info = ab04md('C', A.copy(), B.copy(), C.copy(), D.copy(), alpha=1.0, beta=2.0/dt)
```
This applies to most ctrlsys routines including `ab04md`, `sb02md`, `tb05ad`, etc.

### Info Codes
- `info = 0` → Success
- `info < 0` → Parameter `-info` is invalid
- `info > 0` → Algorithm-specific warning/error

### DICO Parameter
- `'C'` → Continuous-time system
- `'D'` → Discrete-time system

### JOB Parameter
- `'A'` → All computations
- `'B'` → B-related only
- `'C'` → C-related only
- `'N'` → None/minimal

### UPLO Parameter (Symmetric Matrices)
- `'U'` → Upper triangle stored
- `'L'` → Lower triangle stored

## Type Aliases

| Type | Size | C Equivalent |
|------|------|----|
| `i32` | 32-bit int | `int32_t` |
| `i64` | 64-bit int | `int64_t` |
| `f64` | 64-bit float | `double` |
| `c128` | 128-bit complex | `double _Complex` |

## Subcategory Numbers

Common patterns in the 2-digit number:
- `01-03` → Realization, reduction, canonical forms
- `04-05` → Transformation, interconnection
- `06-07` → Feedback, inverse
- `08` → Zeros, poles analysis
- `09` → Model reduction
- `10` → H-infinity, robust control
- `13` → Norms (H2, H-inf, Hankel)

## Example: Finding the Right Routine

**Task:** Design LQR for continuous system with state Q and input R weights

1. Need Riccati solver → SB prefix (synthesis)
2. Algebraic equation → sb02 subcategory
3. Real matrices → MD suffix
4. **Answer:** `sb02md` with DICO='C'

**Task:** Identify system from I/O data

1. System identification → IB prefix
2. Subspace method → ib01 subcategory
3. First call `ib01ad` (preprocessing + order)
4. Then call `ib01bd` (matrix estimation)

## Resources

- **Routine Details:** Use `help(ctrlsys.routine_name)` in Python for full docstring
- **Install:** `uv add ctrlsys` or `pip install ctrlsys`

See `references/` for detailed category info, workflows, and quick reference tables.
