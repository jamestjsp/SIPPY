---
name: control-theory
description: Work with control theory problems, LTI systems, controller design, and system dynamics using ctrlsys
---

# Control Theory with ctrlsys

This skill provides practical guidance for implementing and analyzing linear time-invariant (LTI) control systems using the ctrlsys library, grounded in IFAC computer control principles.

## Purpose

Control theory is fundamental to engineering systems from simple regulators to complex autonomous systems. This skill helps translate theoretical control concepts into practical implementations using ctrlsys, covering system representations, analysis, design, and discretization.

**Key ctrlsys routines used:**
- `sb02md` - Riccati equation solver for LQR design
- `ab04md` - Bilinear (Tustin) discretization
- `tf01md` - Discrete-time state-space simulation
- `tb05ad` - Frequency response evaluation

## When to Use This Skill

To apply this skill, first determine the control theory task:

- **System Representation**: Creating and manipulating transfer functions or state-space models (SISO/MIMO)
- **System Analysis**: Computing poles, zeros, stability, frequency response, norms, controllability, observability
- **Control Design**: Pole placement, LQR optimization, observer design, Kalman filtering
- **Discretization**: Converting continuous systems to discrete-time (ZOH, FOH, Tustin, etc.)
- **Simulation**: Time-domain and frequency-domain responses
- **Learning**: Understanding IFAC computer control principles and best practices

This skill activates when users request help with these domains using ctrlsys, or when they need to understand the theoretical foundations of computer-controlled systems.

## How to Use This Skill

### Step 1: Identify the System Type

Determine whether the problem involves:
- **Continuous or discrete-time** systems
- **SISO** (single-input single-output) or **MIMO** (multi-input multi-output)
- **Transfer function** or **state-space** representation

### Step 2: Reference Practical Code Examples

The scripts in `scripts/` provide working code patterns organized by functionality:
- `lqr_design.py` - LQR controller design using `sb02md`
- `discretize_controller.py` - Discretization using `ab04md`
- `analyze_system.py` - System analysis with frequency response via `tb05ad`
- `pid_controller.py` - PID control simulation using ctrlsys

All arrays use Fortran column-major order (`order='F'` in NumPy) as required by ctrlsys.

### Step 3: Understand Theoretical Foundations

For deeper understanding of computer-controlled systems theory, reference `IFAC_Computer_Control.md`. This comprehensive guide covers:
- Sampling and reconstruction (Chapters 2-3)
- Mathematical models and frequency response (Chapters 4-5)
- Control design and specification methods (Chapters 6-10)
- Practical implementation issues (Chapters 11-12)
- Real-time control and performance (Chapters 13-14)

Use the IFAC reference when:
- Implementing discretization methods (section on aliasing, prewarping)
- Designing controllers for sampled systems
- Understanding practical constraints (quantization, saturation, timing)
- Selecting between design approaches (H∞, LQ, pole placement)

### Step 4: Build Incrementally

When implementing control systems:

1. Start with a simple representation (create transfer function or state-space model)
2. Verify system properties (check stability, poles/zeros, controllability)
3. Design the controller (pole placement, LQR, or observer)
4. Discretize for implementation (choose appropriate method and sampling period)
5. Validate in simulation (step response, frequency response, closed-loop performance)
6. Implement in real-time (account for quantization, saturation, numerical issues)

## Key Concepts from ctrlsys

ctrlsys works with state-space representation (A, B, C, D matrices) using raw numpy arrays:

### State-Space Models
- A, B, C, D matrix representation in Fortran column-major order
- Natural for time-domain simulation and design
- MIMO with arbitrary dimensions
- Properties: poles (A eigenvalues via `np.linalg.eigvals`)

```python
import numpy as np

# Create state-space matrices (Fortran order)
A = np.array([[0, 1], [-1, -2]], order='F', dtype=float)
B = np.array([[0], [1]], order='F', dtype=float)
C = np.array([[1, 0]], order='F', dtype=float)
D = np.array([[0]], order='F', dtype=float)

# Check stability via eigenvalues
poles = np.linalg.eigvals(A)
is_stable = np.all(poles.real < 0)  # Continuous: negative real parts
print(f"Stable: {is_stable}, Poles: {poles}")
```

### Transfer Function to State-Space Conversion
Use a helper function to convert from transfer function to controllable canonical form:

```python
def tf_to_ss(num, den):
    """Convert SISO transfer function to controllable canonical state-space."""
    num = np.atleast_1d(num).astype(float)
    den = np.atleast_1d(den).astype(float)
    den = den / den[0]
    num = num / den[0]
    n = len(den) - 1
    if n == 0:
        return (np.zeros((0, 0), order='F'),
                np.zeros((0, 1), order='F'),
                np.zeros((1, 0), order='F'),
                np.array([[num[0]]], order='F'))
    num_padded = np.zeros(n + 1)
    num_padded[n + 1 - len(num):] = num
    A = np.zeros((n, n), order='F')
    A[:-1, 1:] = np.eye(n - 1)
    A[-1, :] = -den[1:][::-1]
    B = np.zeros((n, 1), order='F')
    B[-1, 0] = 1.0
    C = np.zeros((1, n), order='F')
    d0 = num_padded[0]
    for i in range(n):
        C[0, n - 1 - i] = num_padded[i + 1] - d0 * den[i + 1]
    D = np.array([[d0]], order='F')
    return A, B, C, D
```

## Common Control Theory Workflows

### Discretize a Continuous Controller

Use `ab04md` for bilinear (Tustin) transformation:

```python
from ctrlsys import ab04md
import numpy as np

# Continuous state-space (A, B, C, D)
A = np.array([[0, 1], [-1, -2]], order='F', dtype=float)
B = np.array([[0], [1]], order='F', dtype=float)
C = np.array([[1, 0]], order='F', dtype=float)
D = np.array([[0]], order='F', dtype=float)

dt = 0.01  # Sampling period
# Tustin transformation: alpha=1, beta=2/dt
A_d, B_d, C_d, D_d, info = ab04md('C', A.copy(), B.copy(), C.copy(), D.copy(), alpha=1.0, beta=2.0/dt)
```

### Tustin with Frequency Prewarping

Standard Tustin causes frequency warping: ω_d = (2/dt)×tan(ω_c×dt/2). For PI/PID controllers, prewarp at the crossover frequency to preserve gain/phase margins:

```python
from ctrlsys import ab04md
import numpy as np

def tustin_prewarp(A, B, C, D, dt, wc):
    """Discretize with Tustin prewarping at frequency wc (rad/s).

    The prewarped transformation maps continuous frequency wc exactly
    to the corresponding discrete frequency, preserving behavior at
    the critical crossover point.
    """
    # Prewarped beta: ensures G_d(e^(j*wc*dt)) = G_c(j*wc)
    beta = wc / np.tan(wc * dt / 2)
    A_d, B_d, C_d, D_d, info = ab04md(
        'C', A.copy(), B.copy(), C.copy(), D.copy(),
        alpha=1.0, beta=beta
    )
    return A_d, B_d, C_d, D_d

# Example: PI controller discretized with prewarping at 1 rad/s crossover
Kc, Ti = 0.938, 45.0
A_pi = np.array([[0]], order='F', dtype=float)
B_pi = np.array([[1]], order='F', dtype=float)
C_pi = np.array([[Kc/Ti]], order='F', dtype=float)
D_pi = np.array([[Kc]], order='F', dtype=float)

dt = 0.1  # 100ms sampling
wc = 1.0  # Crossover frequency (rad/s)
A_d, B_d, C_d, D_d = tustin_prewarp(A_pi, B_pi, C_pi, D_pi, dt, wc)
```

**When to prewarp:** Always for PI/PID controllers. The integral action frequency response must be preserved at the crossover frequency for stability margins to match the continuous design.

### Design a Discrete-Time LQR Controller

Use `sb02md` to solve the Riccati equation:

```python
from ctrlsys import sb02md, ab04md
import numpy as np

# Continuous system
A = np.array([[0, 1], [-1, -2]], order='F', dtype=float)
B = np.array([[0], [1]], order='F', dtype=float)
C = np.eye(2, order='F', dtype=float)
D = np.zeros((2, 1), order='F', dtype=float)

# Discretize first
dt = 0.01
A_d, B_d, C_d, D_d, _ = ab04md('C', A.copy(), B.copy(), C.copy(), D.copy(), alpha=1.0, beta=2.0/dt)

# LQR design: solve discrete Riccati equation
n = A_d.shape[0]
Q = np.eye(n, order='F', dtype=float)  # State weight
R = np.array([[1.0]], order='F', dtype=float)  # Input weight
R_inv = np.linalg.inv(R)
G = (B_d @ R_inv @ B_d.T).astype(float, order='F')

X, rcond, wr, wi, S, U, info = sb02md('D', 'D', 'U', 'N', 'U', n, A_d.copy(), G.copy(), Q.copy())
K = np.linalg.inv(R + B_d.T @ X @ B_d) @ B_d.T @ X @ A_d  # Feedback gain
```

### Analyze Discrete System Stability

For discrete-time systems, stability requires poles inside unit circle (|z| < 1):

```python
import numpy as np

# Discrete system (from ab04md output)
poles = np.linalg.eigvals(A_d)
is_stable = np.all(np.abs(poles) < 1)  # Discrete: inside unit circle
if is_stable:
    print(f"System stable. Poles: {poles}")
else:
    unstable = poles[np.abs(poles) >= 1]
    print(f"Unstable poles: {unstable}")
```

### Test Controllability Before Design

Before applying pole placement or LQR, verify the system is controllable:

```python
import numpy as np

def controllability_matrix(A, B):
    """Compute controllability matrix [B, AB, A^2B, ...]."""
    n = A.shape[0]
    m = B.shape[1]
    C_mat = np.zeros((n, n * m))
    for i in range(n):
        C_mat[:, i*m:(i+1)*m] = np.linalg.matrix_power(A, i) @ B
    return C_mat

A = np.array([[0, 1], [-1, -2]], dtype=float)
B = np.array([[0], [1]], dtype=float)
C_mat = controllability_matrix(A, B)
rank = np.linalg.matrix_rank(C_mat)
if rank == A.shape[0]:
    print("System is controllable - pole placement will work")
else:
    print("Uncontrollable modes exist - design will fail")
```

### Dead Time Modeling with Padé Approximation

Pure dead time e^(-Td*s) cannot be represented exactly in state-space. Use Padé approximation to create a rational approximation suitable for analysis:

```python
import numpy as np
from math import factorial

def pade_delay(Td, order=2):
    """Create Padé approximation of dead time e^(-Td*s).

    Args:
        Td: Dead time in seconds
        order: Approximation order (1-5, higher = more accurate)

    Returns:
        (A, B, C, D): State-space matrices (Fortran order)

    The (n,n) Padé approximant has n poles and n zeros, symmetric
    about the imaginary axis. Order 2 is sufficient for most control
    applications; order 3-4 for high-fidelity simulation.
    """
    n = order
    # Padé coefficients
    num_coeffs = []
    den_coeffs = []
    for k in range(n + 1):
        coef = factorial(2*n - k) * factorial(n) / (
            factorial(2*n) * factorial(k) * factorial(n - k)
        )
        num_coeffs.append(coef * (-Td)**k)
        den_coeffs.append(coef * Td**k)

    # Convert to state-space (controllable canonical form)
    num = np.array(num_coeffs)
    den = np.array(den_coeffs)

    # Normalize
    den = den / den[0]
    num = num / den[0]

    # Build controllable canonical form
    A = np.zeros((n, n), order='F', dtype=float)
    A[:-1, 1:] = np.eye(n - 1)
    A[-1, :] = -den[1:][::-1]

    B = np.zeros((n, 1), order='F', dtype=float)
    B[-1, 0] = 1.0

    C = np.zeros((1, n), order='F', dtype=float)
    d0 = num[0]
    for i in range(n):
        C[0, n - 1 - i] = num[i + 1] - d0 * den[i + 1]

    D = np.array([[d0]], order='F', dtype=float)
    return A, B, C, D

# Example: Model FOPDT process G(s) = Kp*e^(-Td*s) / (tau*s + 1)
Kp, tau, Td = 2.0, 45.0, 8.0

# First-order lag
A_lag = np.array([[-1/tau]], order='F', dtype=float)
B_lag = np.array([[Kp/tau]], order='F', dtype=float)
C_lag = np.array([[1]], order='F', dtype=float)
D_lag = np.array([[0]], order='F', dtype=float)

# Dead time (2nd order Padé)
A_delay, B_delay, C_delay, D_delay = pade_delay(Td, order=2)

# Combine: delay * lag (series connection)
def ss_series(sys1, sys2):
    A1, B1, C1, D1 = sys1
    A2, B2, C2, D2 = sys2
    n1, n2 = A1.shape[0], A2.shape[0]
    A = np.zeros((n1 + n2, n1 + n2), order='F', dtype=float)
    A[:n1, :n1] = A1
    A[n1:, :n1] = B2 @ C1
    A[n1:, n1:] = A2
    B = np.vstack([B1, B2 @ D1]).astype(float, order='F')
    C = np.hstack([D2 @ C1, C2]).astype(float, order='F')
    D = (D2 @ D1).astype(float, order='F')
    return A, B, C, D

plant = ss_series((A_lag, B_lag, C_lag, D_lag), (A_delay, B_delay, C_delay, D_delay))
print(f"FOPDT model: {plant[0].shape[0]} states")
```

### Gain and Phase Margins with tb05ad

Use `tb05ad` to compute frequency response and extract stability margins:

```python
from ctrlsys import tb05ad
import numpy as np

def bode_margins(A, B, C, D, w_range=None):
    """Compute gain and phase margins from open-loop frequency response.

    Args:
        A, B, C, D: State-space matrices of open-loop system L(s)
        w_range: Frequency range [w_min, w_max] in rad/s (default: auto)

    Returns:
        dict with: gm (gain margin in dB), pm (phase margin in deg),
                   wcg (gain crossover freq), wcp (phase crossover freq)
    """
    if w_range is None:
        eigvals = np.linalg.eigvals(A)
        w_max = max(10, 10 * np.max(np.abs(eigvals)))
        w_min = w_max / 1000
        w_range = [w_min, w_max]

    w = np.logspace(np.log10(w_range[0]), np.log10(w_range[1]), 500)

    mag = np.zeros(len(w))
    phase = np.zeros(len(w))

    for i, wi in enumerate(w):
        freq = 1j * wi  # s = jω
        g, rcond, evre, evim, hinvb, info = tb05ad(
            'N', 'G', A.copy(), B.copy(), C.copy(), freq
        )
        if info != 0:
            continue
        H = g[0, 0] + D[0, 0]  # tb05ad returns C(sI-A)^{-1}B, must add D
        mag[i] = np.abs(H)
        phase[i] = np.angle(H) * 180 / np.pi

    # Unwrap phase
    phase = np.unwrap(phase * np.pi / 180) * 180 / np.pi

    # Find gain crossover (|L(jw)| = 1, i.e., 0 dB)
    mag_db = 20 * np.log10(mag + 1e-12)
    crossings_gain = np.where(np.diff(np.sign(mag_db)))[0]
    wcg = w[crossings_gain[0]] if len(crossings_gain) > 0 else None

    # Find phase crossover (phase = -180°)
    crossings_phase = np.where(np.diff(np.sign(phase + 180)))[0]
    wcp = w[crossings_phase[0]] if len(crossings_phase) > 0 else None

    # Calculate margins
    pm = None
    if wcg is not None:
        idx = crossings_gain[0]
        pm = 180 + phase[idx]  # Phase margin at gain crossover

    gm = None
    if wcp is not None:
        idx = crossings_phase[0]
        gm = -mag_db[idx]  # Gain margin at phase crossover (in dB)

    return {
        'gm': gm, 'pm': pm,
        'wcg': wcg, 'wcp': wcp,
        'w': w, 'mag_db': mag_db, 'phase': phase
    }

# Example: Check margins for PI-controlled heat exchanger
# Open loop L(s) = C(s) * G(s)
A_L = np.array([[0, 0, 0],
                [1, -1/45, 0],
                [0, 2/45, 0]], order='F', dtype=float)  # Simplified
B_L = np.array([[1], [0], [0]], order='F', dtype=float)
C_L = np.array([[0, 0, 1]], order='F', dtype=float)
D_L = np.array([[0.938]], order='F', dtype=float)

margins = bode_margins(A_L, B_L, C_L, D_L)
if margins['pm'] is not None:
    print(f"Phase margin: {margins['pm']:.1f}° at {margins['wcg']:.3f} rad/s")
if margins['gm'] is not None:
    print(f"Gain margin: {margins['gm']:.1f} dB at {margins['wcp']:.3f} rad/s")

# Rule of thumb: PM > 45°, GM > 6 dB for robust design
```

### Visualize Controller Performance

After designing a controller, use ctrlsys and matplotlib to analyze closed-loop performance:

```python
from ctrlsys import tf01md, ab04md, tb05ad
import matplotlib.pyplot as plt
import numpy as np

# Helper functions (see full implementations in "Key Concepts" section above)
def tf_to_ss(num, den):
    """Convert SISO transfer function to controllable canonical state-space."""
    num = np.atleast_1d(num).astype(float)
    den = np.atleast_1d(den).astype(float)
    den = den / den[0]
    num = num / den[0]
    n = len(den) - 1
    if n == 0:
        return (np.zeros((0, 0), order='F'), np.zeros((0, 1), order='F'),
                np.zeros((1, 0), order='F'), np.array([[num[0]]], order='F'))
    num_padded = np.zeros(n + 1)
    num_padded[n + 1 - len(num):] = num
    A = np.zeros((n, n), order='F')
    A[:-1, 1:] = np.eye(n - 1)
    A[-1, :] = -den[1:][::-1]
    B = np.zeros((n, 1), order='F')
    B[-1, 0] = 1.0
    C = np.zeros((1, n), order='F')
    d0 = num_padded[0]
    for i in range(n):
        C[0, n - 1 - i] = num_padded[i + 1] - d0 * den[i + 1]
    D = np.array([[d0]], order='F')
    return A, B, C, D

def ss_series(sys1, sys2):
    """Connect two state-space systems in series."""
    A1, B1, C1, D1 = sys1
    A2, B2, C2, D2 = sys2
    n1, n2 = A1.shape[0], A2.shape[0]
    A = np.zeros((n1 + n2, n1 + n2), order='F', dtype=float)
    A[:n1, :n1] = A1
    A[n1:, :n1] = B2 @ C1
    A[n1:, n1:] = A2
    B = np.vstack([B1, B2 @ D1]).astype(float, order='F')
    C = np.hstack([D2 @ C1, C2]).astype(float, order='F')
    D = (D2 @ D1).astype(float, order='F')
    return A, B, C, D

def ss_feedback(sys, k=1.0):
    """Negative unity feedback: L / (1 + k*L)."""
    A, B, C, D = sys
    inv_term = 1.0 / (1.0 + k * D[0, 0])
    Af = (A - k * inv_term * B @ C).astype(float, order='F')
    Bf = (inv_term * B).astype(float, order='F')
    Cf = (inv_term * C).astype(float, order='F')
    Df = (inv_term * D).astype(float, order='F')
    return Af, Bf, Cf, Df

# Plant and controller
G = tf_to_ss([1], [1, 2, 1])  # 1/(s^2 + 2s + 1)
C = tf_to_ss([2.0, 1.0], [1, 0])  # PI controller

# Open-loop and closed-loop
L = ss_series(G, C)
T = ss_feedback(L, 1.0)

# Simulate step response
dt = 0.01
A_d, B_d, C_d, D_d, _ = ab04md('C', T[0].copy(), T[1].copy(), T[2].copy(), T[3].copy(), alpha=1.0, beta=2.0/dt)
t = np.arange(0, 10, dt)
u = np.ones((1, len(t)), order='F', dtype=float)
y, _, _ = tf01md(A_d, B_d, C_d, D_d, u, np.zeros(A_d.shape[0]))

plt.plot(t, y.flatten())
plt.title('Closed-Loop Step Response')
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.grid()
plt.show()
```

This workflow demonstrates the complete design cycle:
1. Define plant and controller as state-space
2. Form open-loop and closed-loop systems
3. Discretize and simulate step response
4. Verify stability via eigenvalues

## References and Standards

- **IFAC Computer Control**: Comprehensive reference on computer-controlled systems, discretization, and practical implementation
- **ctrlsys routine reference**: Use `help(ctrlsys.routine_name)` for full docstrings
- **State-Space Theory**: Properties of A, B, C, D matrices and their interpretation in control design

## Tips for Success

1. **Always verify stability first**: Check eigenvalues before relying on analysis
2. **Test controllability/observability**: Design only works on controllable/observable subsystems
3. **Choose discretization carefully**: Use `ab04md` with Tustin for bandwidth-critical applications
4. **Validate in simulation**: Always simulate step response using `tf01md`
5. **Use Fortran column-major order**: All arrays must use `order='F'` for ctrlsys
6. **Account for implementation constraints**: Quantization, saturation, computational limits (see IFAC Ch. 11-12)
