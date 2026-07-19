# ctrlsys Quick Reference

## Task-to-Routine Tables

### Equations
| Equation | Routine | Notes |
|----------|---------|-------|
| Riccati (ARE) | `sb02md` | Continuous/discrete |
| Riccati (generalized) | `sg02ad` | Descriptor systems |
| Optimal gain | `sb02od` | State feedback via Riccati |
| Lyapunov | `sb03md` | A'X + XA + Q = 0 |
| Sylvester | `sb04md` | AX + XB = C |
| Stein (discrete Lyapunov) | `sb03md` | DICO='D' |
| Generalized Lyapunov | `sg03ad` | Descriptor systems |

### System Operations
| Operation | Routine |
|-----------|---------|
| Series/cascade | `ab05md` |
| Parallel | `ab05nd` |
| Feedback | `ab05pd` |
| Inverse | `ab07nd` |

### Transformations
| From | To | Routine |
|------|-----|---------|
| Continuous | Discrete | `ab04md` (type='C') |
| Discrete | Continuous | `ab04md` (type='D') |
| State-space | Transfer fn | `tb04ad` |
| Transfer fn | State-space | `tc04ad` |

### Norms
| Norm | Routine |
|------|---------|
| H-infinity (general) | `ab13dd` |
| H-infinity (continuous, stable) | `ab13cd` |
| H2 | `ab13bd` |
| Hankel | `ab13ad` |
| Structured singular value (mu) | `ab13md` |

## Parameter Conventions

### DICO - System Type
```
'C' = Continuous-time
'D' = Discrete-time
```

### JOB - Operation Mode
```
'A' = All computations
'B' = Partial (B-related)
'C' = Partial (C-related)
'N' = None/minimal
```

### UPLO - Triangle Storage
```
'U' = Upper triangle stored
'L' = Lower triangle stored
```

### FACT - Factorization State
```
'N' = Not factored (compute)
'F' = Already factored
```

### TRANS - Transpose Mode
```
'N' = No transpose
'T' = Transpose
'C' = Conjugate transpose
```

### METH - Identification Method (IB01xx)
```
'M' = MOESP algorithm
'N' = N4SID algorithm
'C' = Combined (MOESP for A,C; N4SID for B,D)
```

### ALG - Algorithm Variant (IB01xx)
```
'C' = Cholesky (fast, less stable)
'F' = Fast QR
'Q' = QR (slower, more stable)
```

### BATCH - Data Processing Mode
```
'F' = First batch (initialize)
'I' = Intermediate batch
'L' = Last batch (finalize)
'O' = One batch (all data at once)
```

### ORDSEL - Order Selection
```
'F' = Fixed order (use NR)
'A' = Automatic (use tolerance)
```

### EQUIL - Equilibration
```
'S' = Scale system
'N' = No scaling
```

### JOBCF - Coprime Factorization
```
'L' = Left coprime factorization
'R' = Right coprime factorization
```

### WEIGHT - Frequency Weighting
```
'N' = None
'L' = Left weighting
'R' = Right weighting
'B' = Both
```

## Info Codes

### Universal
| Code | Meaning |
|------|---------|
| 0 | Success |
| -i | Parameter i invalid |

### sb02md (Riccati)
| Code | Meaning |
|------|---------|
| 1 | A singular (discrete) |
| 2 | Schur reduction failed |
| 3 | Eigenvalue reordering failed |
| 4 | Less than n stable eigenvalues |
| 5 | Singular solution matrix |

### ab09ad (Model Reduction)
| Code | Meaning |
|------|---------|
| 1 | System unstable |
| 2 | Lyapunov solver failed |
| 3 | SVD did not converge |

### ib01ad (Identification)
| Code | Meaning |
|------|---------|
| 1 | Matrix rank-deficient |
| 2 | SVD did not converge |

### sb01bd (Pole Placement)
| Code | Meaning |
|------|---------|
| 1 | Schur form reduction failed |
| 2 | Eigenvalue reordering failed |
| 3 | Not enough eigenvalues provided |
| 4 | Incompatible poles |

### sb03md (Lyapunov)
| Code | Meaning |
|------|---------|
| 1 | A is not stable |
| 2 | A is not in Schur form |
| 3 | Singular matrix (discrete) |

### sb10ad (H-infinity)
| Code | Meaning |
|------|---------|
| 1 | D12 not full column rank |
| 2 | D21 not full row rank |
| 3 | Singular I-D11'D11 |
| 4 | X-Riccati has no solution |
| 5 | Y-Riccati has no solution |
| 6 | Spectral radius ≥ gamma²

## Routine Suffixes

### Data Type
| Suffix | Type |
|--------|------|
| MD | Real double |
| MZ | Complex double |
| SD | Real single |

### Variant
| Suffix | Description |
|--------|-------------|
| AD | Basic/primary version |
| BD | Alternative algorithm |
| CD | Coprime/factored form |
| DD | Discrete-time specific |
| ED | Extended version |
| FD | Frequency domain |
| HD | H-infinity related |
| ID | Identification related |
| JD | Jordan form |
| KD | Kalman related |
| ND | Multi-input/output |
| OD | Observability-related |
| PD | Polynomial form |
| RD | Reduced version |
| TD | Transformation |
| XD, YD | Auxiliary routines |

## Common Pitfalls

### 1. Array Order
Always use Fortran order:
```python
A = np.array(..., order='F')
# or
A = np.asfortranarray(A)
```

### 2. In-Place Modification
Many routines modify arrays in-place. Pass copies:
```python
result = routine(A.copy(), B.copy())
```

### 3. Workspace Sizing
Let wrappers handle workspace. For C API:
```c
ldwork = -1;  // Query optimal size
routine(..., &ldwork);
dwork = malloc(ldwork * sizeof(double));
routine(..., dwork, ldwork);
```

### 4. Leading Dimensions
C API requires explicit LDA parameters:
```c
lda = n;  // Usually max(1, n)
```

### 5. Complex Conjugate Pairs
For pole placement, complex eigenvalues must appear as consecutive conjugate pairs in wr/wi arrays.

## Type Mapping (Python ↔ C)

| NumPy | C Type | Size |
|-------|--------|------|
| `float64` | `f64` (double) | 8 bytes |
| `int32` | `i32` | 4 bytes |
| `complex128` | `c128` | 16 bytes |

## Build/Test Commands

```bash
# Install dev
uv sync --extra test

# Run tests
uv run pytest tests/python/ -v

# Single test
uv run pytest tests/python/test_sb02md.py::test_basic -v

# Parallel
uv run pytest tests/python/ -n auto
```
