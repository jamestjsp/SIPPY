# ctrlsys Categories Reference

## Analysis (AB, AG)

### AB - State-Space Analysis
| Routine | Description |
|---------|-------------|
| `ab01md` | Controllable realization (single-input) |
| `ab01nd` | Controllable realization (multi-input) |
| `ab01od` | Staircase form reduction |
| `ab04md` | Bilinear transformation (c2d/d2c) |
| `ab05md` | Cascade connection |
| `ab05nd` | Parallel connection |
| `ab05pd` | Feedback connection |
| `ab07nd` | System inverse |
| `ab08nd` | Transmission zeros |
| `ab09ad` | Balanced truncation |
| `ab09bd` | Balanced stochastic truncation |
| `ab09cd` | Frequency-weighted balanced reduction |
| `ab09fd` | Coprime factorization reduction |
| `ab09md` | Singular perturbation approximation |
| `ab13ad` | Hankel singular values / Hankel norm |
| `ab13bd` | H2/L2 norm of transfer function |
| `ab13cd` | H-infinity norm (continuous, stable) |
| `ab13dd` | L-infinity norm (continuous/discrete) |
| `ab13ed` | Distance to instability (stability radius) |
| `ab13fd` | Complex stability radius (SVD-based) |
| `ab13hd` | L-infinity norm (descriptor systems) |
| `ab13id` | Check properness of descriptor transfer function |
| `ab13md` | Structured singular value (mu) upper bound |

### AG - Generalized State-Space
| Routine | Description |
|---------|-------------|
| `ag07bd` | Inverse of descriptor system |
| `ag08by` | Rank of transfer matrix (descriptor) |
| `ag08bz` | Complex version of ag08by |

## Synthesis (SB, SG)

### SB - Control Synthesis
| Routine | Description |
|---------|-------------|
| `sb01bd` | Pole placement (multi-input) |
| `sb01dd` | Eigenstructure assignment |
| `sb02md` | Algebraic Riccati equation (ARE) |
| `sb02mu` | Hamiltonian/symplectic matrix construction |
| `sb02od` | Optimal state feedback |
| `sb03md` | Lyapunov equation |
| `sb03od` | Cholesky factor of Lyapunov solution |
| `sb04md` | Sylvester equation |
| `sb06nd` | Deadbeat control |
| `sb08cd` | Spectral factorization |
| `sb10ad` | H-infinity controller (continuous) |
| `sb10dd` | H-infinity controller (discrete) |
| `sb10fd` | H-infinity optimal controller |
| `sb10hd` | Discrete H-inf (gamma iteration) |

### SB16 - Controller Reduction
| Routine | Description |
|---------|-------------|
| `sb16ad` | Frequency-weighted controller reduction via balancing |
| `sb16bd` | Coprime factorization state feedback controller reduction |
| `sb16cd` | Frequency-weighted coprime factorization controller reduction |

### SG - Generalized Synthesis
Descriptor system synthesis routines.
| Routine | Description |
|---------|-------------|
| `sg02ad` | Generalized algebraic Riccati equation (descriptor) |
| `sg02nd` | Optimal gain for generalized Riccati problems |
| `sg03ad` | Generalized Lyapunov equation |
| `sg03bd` | Cholesky factor of generalized Lyapunov |

## Transform (TG, TB, TC, TD, TF)

### TG - Descriptor Transform
| Routine | Description |
|---------|-------------|
| `tg01ad` | Balance descriptor pencil |
| `tg01bd` | Reduce to Hessenberg-triangular |
| `tg01cd` | Orthogonal reduction |
| `tg01fd` | Irreducible form |
| `tg01hd` | Observability staircase |
| `tg01id` | Controllability staircase |
| `tg01jd` | Descriptor to standard form |

### TB - State-Space Transform
| Routine | Description |
|---------|-------------|
| `tb01id` | Similarity transformation |
| `tb01pd` | Minimal realization |
| `tb01td` | Balance state-space system |
| `tb01ud` | Schur form reduction |
| `tb01wd` | Hessenberg form |
| `tb05ad` | Frequency response matrix from state-space |

### TC - Polynomial/Transfer
| Routine | Description |
|---------|-------------|
| `tc01od` | Left coprime factorization |
| `tc04ad` | Transfer function to state-space |

### TD - Transfer Domain
| Routine | Description |
|---------|-------------|
| `td03ad` | Polynomial to state-space (row form) |
| `td03ay` | State-space to polynomial |
| `td04ad` | Polynomial to state-space (column form) |
| `td05ad` | Frequency response from transfer fn |

### TF - Frequency Response / Markov Parameters
| Routine | Description |
|---------|-------------|
| `tf01md` | Output sequence from state-space |
| `tf01nd` | Output response (Hessenberg state matrix) |
| `tf01od` | Block Hankel expansion |
| `tf01pd` | Block Toeplitz expansion |
| `tf01qd` | Markov parameters from transfer function |
| `tf01rd` | Markov parameters from state-space |

## Matrix (MB, MC, MA, MD)

### MB - Matrix Computations
| Routine | Description |
|---------|-------------|
| `mb01rd` | Matrix scaling |
| `mb02ed` | Solve linear systems (Hessenberg) |
| `mb02md` | Total least squares |
| `mb03ad` | Real Schur form |
| `mb03bd` | Eigenvalues of Hessenberg |
| `mb03rd` | Schur form reordering |
| `mb03ud` | SVD of triangular matrix |
| `mb03vd` | Product SVD |
| `mb04dd` | Symplectic URV decomposition |
| `mb05md` | Matrix exponential |
| `mb05nd` | Matrix exponential and integral |

### MC - Matrix Construction
| Routine | Description |
|---------|-------------|
| `mc01md` | Polynomial evaluation |
| `mc01pd` | Polynomial roots |
| `mc01sw` | Polynomial value (Horner) |

### MA - Matrix Auxiliary
| Routine | Description |
|---------|-------------|
| `ma01ad` | Scale matrix to reduce norm |
| `ma01bd` | Scale matrix with given factors |
| `ma01bz` | Complex version of ma01bd |
| `ma01cd` | Infinity norm of matrix |
| `ma02ad` | Transpose matrix in-place |
| `ma02bd` | Reverse rows/columns |
| `ma02ed` | Skew-symmetric matrix operations |

### MD - Matrix Decomposition / Optimization
| Routine | Description |
|---------|-------------|
| `md03ba` | QR factorization with column pivoting (Levenberg-Marquardt) |
| `md03bb` | Levenberg-Marquardt parameter (compressed Jacobian) |
| `md03bd` | Levenberg-Marquardt nonlinear least squares optimizer |
| `md03bf` | Kowalik-Osborne test function |
| `md03by` | Levenberg-Marquardt parameter (trust region) |

## Identification (IB, IB03, NF)

### IB - Subspace Identification
| Routine | Description |
|---------|-------------|
| `ib01ad` | MOESP/N4SID preprocessing + order estimation |
| `ib01bd` | System matrices (A, B, C, D) estimation |
| `ib01cd` | Initial state estimation |
| `ib01md` | QR factorization of block Hankel |
| `ib01nd` | SVD for order determination |
| `ib01od` | Order estimation from singular values |
| `ib01pd` | Compute system matrices |
| `ib01rd` | Residuals and covariances |

### IB03 - Wiener System Identification
| Routine | Description |
|---------|-------------|
| `ib03ad` | Wiener system identification (algorithm selection) |
| `ib03bd` | Wiener system identification (Levenberg-Marquardt) |

### NF - Nonlinear/Neural Network
| Routine | Description |
|---------|-------------|
| `nf01ay` | Neural network output computation |
| `nf01br` | Solve linear equations in least squares sense |
| `nf01bs` | QR factorization of Jacobian (compressed form) |
| `nf01by` | Jacobian of neural network error function |

## Filter (FB, FD)

### FB - Kalman Filtering
| Routine | Description |
|---------|-------------|
| `fb01qd` | Square root covariance Kalman filter (one step) |
| `fb01rd` | Time update (conventional) |
| `fb01sd` | Measurement update |
| `fb01td` | Square root information Kalman filter (controller Hessenberg) |
| `fb01vd` | Conventional Kalman filter (one recursion) |

### FD - Filter Design
| Routine | Description |
|---------|-------------|
| `fd01ad` | IIR filter design |

## Utility (BB, BD, DE, DF, DG, DK, UD, UE)

### BB - Basic
| Routine | Description |
|---------|-------------|
| `bb01ad` | Benchmark model A (double integrator) |
| `bb02ad` | Benchmark model B (van der Pol) |
| `bb03ad` | Benchmark model C (flexible beam) |
| `bb04ad` | Benchmark model D (four-disk system) |

### BD - Block Diagonal
| Routine | Description |
|---------|-------------|
| `bd01ad` | Block diagonal system to full form |
| `bd02ad` | Full form to block diagonal |

### DE - Data Exchange
| Routine | Description |
|---------|-------------|
| `de01pd` | FFT of real sequences |
| `de01od` | Inverse FFT |

### DF - Data Format
| Routine | Description |
|---------|-------------|
| `df01md` | Sine/cosine transforms |

### DG - Data Generation
| Routine | Description |
|---------|-------------|
| `dg01md` | Generate test polynomial |
| `dg01nd` | Generate random orthogonal matrix |

### DK - Data Check
| Routine | Description |
|---------|-------------|
| `dk01md` | Anti-aliasing window (Kaiser) |

### UD - Upper/Lower Decomposition
| Routine | Description |
|---------|-------------|
| `ud01bd` | Print matrix in MATLAB format |
| `ud01cd` | Print complex matrix |
| `ud01dd` | Print integer matrix |

### UE - Update/Extract
| Routine | Description |
|---------|-------------|
| `ue01md` | Update factor after rank-1 mod |
