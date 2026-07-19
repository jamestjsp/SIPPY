# ctrlsys Control Theory Code Examples

A practical reference of control theory code examples using ctrlsys, a C11 control systems library with Python bindings.

## 1. State-Space Representation

ctrlsys works with state-space representations. Transfer functions must be converted to state-space form.

### State-Space Arrays

```python
import numpy as np

# SISO state-space: dx/dt = Ax + Bu, y = Cx + Du
A = np.array([[0, 1], [-4, -5]], order='F', dtype=float)
B = np.array([[0], [1]], order='F', dtype=float)
C = np.array([[1, 0]], order='F', dtype=float)
D = np.array([[0]], order='F', dtype=float)

# MIMO state-space (3 states, 2 inputs, 2 outputs)
A = np.random.rand(3, 3).astype(float, order='F')
B = np.random.rand(3, 2).astype(float, order='F')
C = np.random.rand(2, 3).astype(float, order='F')
D = np.zeros((2, 2), order='F', dtype=float)
```

### Transfer Function to State-Space Conversion

```python
import numpy as np

def tf_to_ss(num, den):
    """Convert SISO transfer function to controllable canonical state-space.

    Args:
        num: Numerator polynomial coefficients [b_n, b_{n-1}, ..., b_0]
        den: Denominator polynomial coefficients [a_n, a_{n-1}, ..., a_0]

    Returns:
        A, B, C, D: State-space matrices in Fortran order
    """
    num = np.atleast_1d(num).astype(float)
    den = np.atleast_1d(den).astype(float)
    den = den / den[0]
    num = num / den[0]
    n = len(den) - 1

    if n == 0:
        return (np.zeros((0, 0), order='F', dtype=float),
                np.zeros((0, 1), order='F', dtype=float),
                np.zeros((1, 0), order='F', dtype=float),
                np.array([[num[0]]], order='F', dtype=float))

    num_padded = np.zeros(n + 1)
    num_padded[n + 1 - len(num):] = num

    A = np.zeros((n, n), order='F', dtype=float)
    A[:-1, 1:] = np.eye(n - 1)
    A[-1, :] = -den[1:][::-1]

    B = np.zeros((n, 1), order='F', dtype=float)
    B[-1, 0] = 1.0

    C = np.zeros((1, n), order='F', dtype=float)
    d0 = num_padded[0]
    for i in range(n):
        C[0, n - 1 - i] = num_padded[i + 1] - d0 * den[i + 1]

    D = np.array([[d0]], order='F', dtype=float)
    return A, B, C, D

# Example: G(s) = 1/(s^2 + 2s + 1)
A, B, C, D = tf_to_ss([1], [1, 2, 1])

# Example: G(s) = (s + 1)/(s^2 + s + 1)
A, B, C, D = tf_to_ss([1, 1], [1, 1, 1])
```

## 2. System Analysis

### Poles and Stability

```python
import numpy as np

# Poles are eigenvalues of A matrix
A = np.array([[0, 1], [-4, -5]], order='F', dtype=float)
poles = np.linalg.eigvals(A)

# Continuous-time stability: all poles must have negative real parts
is_stable_continuous = np.all(poles.real < 0)

# Discrete-time stability: all poles must be inside unit circle
is_stable_discrete = np.all(np.abs(poles) < 1)

print(f"Poles: {poles}")
print(f"Stable (continuous): {is_stable_continuous}")
```

### DC Gain

```python
import numpy as np

def dc_gain_continuous(A, B, C, D):
    """Compute DC gain G(0) = C*(-A)^{-1}*B + D for continuous system."""
    n = A.shape[0]
    if n == 0:
        return D
    return -C @ np.linalg.solve(A, B) + D

def dc_gain_discrete(A, B, C, D):
    """Compute DC gain G(1) = C*(I-A)^{-1}*B + D for discrete system."""
    n = A.shape[0]
    if n == 0:
        return D
    return C @ np.linalg.solve(np.eye(n) - A, B) + D

# For transfer function G(s), DC gain = G(0) = lim_{s->0} G(s)
A, B, C, D = tf_to_ss([100], [1, 10, 100])
gain = dc_gain_continuous(A, B, C, D)
print(f"DC Gain: {gain}")
```

### Frequency Response with tb05ad

```python
import numpy as np
from ctrlsys import tb05ad

A = np.array([[0, 1], [-4, -5]], order='F', dtype=float)
B = np.array([[0], [1]], order='F', dtype=float)
C = np.array([[1, 0]], order='F', dtype=float)
D = np.array([[0]], order='F', dtype=float)

# Compute frequency response at specific frequencies
frequencies = np.logspace(-1, 2, 100)
magnitudes = np.zeros(len(frequencies))
phases = np.zeros(len(frequencies))

for i, w in enumerate(frequencies):
    s = 1j * w
    g, _, _, _, _, info = tb05ad('N', 'G',
        np.asfortranarray(A.copy()),
        np.asfortranarray(B.copy()),
        np.asfortranarray(C.copy()), s)

    H = g[0, 0] + D[0, 0]
    magnitudes[i] = np.abs(H)
    phases[i] = np.angle(H, deg=True)

# Plot Bode diagram
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.loglog(frequencies, magnitudes)
ax1.set_ylabel('Magnitude')
ax1.grid(True, which='both', alpha=0.3)

ax2.semilogx(frequencies, phases)
ax2.set_xlabel('Frequency (rad/s)')
ax2.set_ylabel('Phase (deg)')
ax2.grid(True, which='both', alpha=0.3)

plt.tight_layout()
```

## 3. Discretization with ab04md

### Tustin (Bilinear) Transformation

```python
import numpy as np
from ctrlsys import ab04md

def discretize_tustin(A, B, C, D, dt):
    """Discretize continuous system using Tustin transformation.

    Args:
        A, B, C, D: Continuous-time state-space matrices
        dt: Sampling period

    Returns:
        A_d, B_d, C_d, D_d: Discrete-time state-space matrices
    """
    alpha = 1.0
    beta = 2.0 / dt

    A_d, B_d, C_d, D_d, info = ab04md('C',
        np.asfortranarray(A.copy()),
        np.asfortranarray(B.copy()),
        np.asfortranarray(C.copy()),
        np.asfortranarray(D.copy()),
        alpha=alpha, beta=beta)

    if info != 0:
        raise RuntimeError(f"ab04md failed with info={info}")

    return A_d, B_d, C_d, D_d

# Example: Discretize G(s) = 1/(s+1) with Ts = 0.1s
A, B, C, D = tf_to_ss([1], [1, 1])
A_d, B_d, C_d, D_d = discretize_tustin(A, B, C, D, dt=0.1)

print(f"Continuous poles: {np.linalg.eigvals(A)}")
print(f"Discrete poles: {np.linalg.eigvals(A_d)}")
```

### Tustin with Frequency Prewarping

```python
import numpy as np
from ctrlsys import ab04md

def discretize_tustin_prewarp(A, B, C, D, dt, prewarp_freq):
    """Discretize with Tustin and frequency prewarping.

    Args:
        A, B, C, D: Continuous-time state-space matrices
        dt: Sampling period
        prewarp_freq: Frequency (rad/s) at which to match continuous response

    Returns:
        A_d, B_d, C_d, D_d: Discrete-time state-space matrices
    """
    # Prewarp: beta = omega / tan(omega * dt / 2)
    beta = prewarp_freq / np.tan(prewarp_freq * dt / 2)
    alpha = 1.0

    A_d, B_d, C_d, D_d, info = ab04md('C',
        np.asfortranarray(A.copy()),
        np.asfortranarray(B.copy()),
        np.asfortranarray(C.copy()),
        np.asfortranarray(D.copy()),
        alpha=alpha, beta=beta)

    if info != 0:
        raise RuntimeError(f"ab04md failed with info={info}")

    return A_d, B_d, C_d, D_d

# Prewarp at 3 rad/s
A, B, C, D = tf_to_ss([1, 0.5, 9], [1, 5, 9])
A_d, B_d, C_d, D_d = discretize_tustin_prewarp(A, B, C, D, dt=0.5, prewarp_freq=3.0)
```

### Discrete to Continuous Conversion

```python
import numpy as np
from ctrlsys import ab04md

def undiscretize_tustin(A_d, B_d, C_d, D_d, dt):
    """Convert discrete system back to continuous using inverse Tustin."""
    alpha = 1.0
    beta = 2.0 / dt

    # 'D' mode converts discrete to continuous
    A_c, B_c, C_c, D_c, info = ab04md('D',
        np.asfortranarray(A_d.copy()),
        np.asfortranarray(B_d.copy()),
        np.asfortranarray(C_d.copy()),
        np.asfortranarray(D_d.copy()),
        alpha=alpha, beta=beta)

    if info != 0:
        raise RuntimeError(f"ab04md failed with info={info}")

    return A_c, B_c, C_c, D_c
```

## 4. Time-Domain Simulation with tf01md

### Step Response

```python
import numpy as np
from ctrlsys import tf01md, ab04md

def simulate_step_response(A, B, C, D, t_final=10.0, dt=0.01, continuous=True):
    """Simulate step response of a system.

    Args:
        A, B, C, D: State-space matrices
        t_final: Simulation duration
        dt: Time step for simulation
        continuous: True if system is continuous (will be discretized)

    Returns:
        t: Time vector
        y: Output response
    """
    n = A.shape[0]
    n_steps = int(t_final / dt)
    t = np.arange(n_steps) * dt

    if continuous and n > 0:
        A_d, B_d, C_d, D_d, _ = ab04md('C',
            np.asfortranarray(A.copy()),
            np.asfortranarray(B.copy()),
            np.asfortranarray(C.copy()),
            np.asfortranarray(D.copy()),
            alpha=1.0, beta=2.0/dt)
    else:
        A_d, B_d, C_d, D_d = A, B, C, D

    n_inputs = B.shape[1]
    u = np.ones((n_inputs, n_steps), order='F', dtype=float)
    x0 = np.zeros(n, dtype=float)

    y, x_final, info = tf01md(A_d, B_d, C_d, D_d, u, x0)

    return t, y

# Example
A, B, C, D = tf_to_ss([1], [1, 1, 1])
t, y = simulate_step_response(A, B, C, D)

import matplotlib.pyplot as plt
plt.plot(t, y.flatten())
plt.xlabel('Time (s)')
plt.ylabel('Response')
plt.title('Step Response')
plt.grid(True, alpha=0.3)
```

### Custom Input Simulation

```python
import numpy as np
from ctrlsys import tf01md

def simulate_system(A_d, B_d, C_d, D_d, u, x0=None):
    """Simulate discrete system with arbitrary input.

    Args:
        A_d, B_d, C_d, D_d: Discrete state-space matrices
        u: Input array, shape (n_inputs, n_steps)
        x0: Initial state (optional)

    Returns:
        y: Output array, shape (n_outputs, n_steps)
        x_final: Final state
    """
    n = A_d.shape[0]
    if x0 is None:
        x0 = np.zeros(n, dtype=float)

    y, x_final, info = tf01md(A_d, B_d, C_d, D_d,
                               np.asfortranarray(u),
                               np.asarray(x0, dtype=float))
    return y, x_final

# Example: MIMO simulation with initial conditions
A_d = np.eye(3, order='F', dtype=float) * 0.9
B_d = np.ones((3, 1), order='F', dtype=float) * 0.1
C_d = np.ones((1, 3), order='F', dtype=float)
D_d = np.zeros((1, 1), order='F', dtype=float)

n_steps = 100
u = np.ones((1, n_steps), order='F', dtype=float)
x0 = np.array([1.0, 0.0, 0.0])

y, x_final = simulate_system(A_d, B_d, C_d, D_d, u, x0)
```

## 5. LQR Control Design with sb02md

### Continuous-Time LQR

```python
import numpy as np
from ctrlsys import sb02md

def lqr_continuous(A, B, Q, R):
    """Solve continuous-time LQR problem.

    Minimizes: J = integral(x'Qx + u'Ru) dt
    Subject to: dx/dt = Ax + Bu

    Args:
        A: State matrix (n x n)
        B: Input matrix (n x m)
        Q: State weight matrix (n x n), positive semi-definite
        R: Input weight matrix (m x m), positive definite

    Returns:
        K: Optimal feedback gain (m x n), use u = -Kx
        P: Solution to Riccati equation
        poles: Closed-loop poles
    """
    n = A.shape[0]
    R_inv = np.linalg.inv(R)
    G = (B @ R_inv @ B.T).astype(float, order='F')

    A_f = np.asfortranarray(A.copy())
    Q_f = np.asfortranarray(Q.copy())

    X, rcond, wr, wi, S, U, info = sb02md('C', 'D', 'U', 'N', 'S', n, A_f, G, Q_f)

    if info != 0:
        raise RuntimeError(f"sb02md failed with info={info}")

    K = R_inv @ B.T @ X
    poles = wr[:n] + 1j * wi[:n]

    return K, X, poles

# Example: Double integrator
A = np.array([[0, 1], [0, 0]], order='F', dtype=float)
B = np.array([[0], [1]], order='F', dtype=float)
Q = np.eye(2, order='F', dtype=float)
R = np.array([[1]], order='F', dtype=float)

K, P, poles = lqr_continuous(A, B, Q, R)
print(f"Optimal gain K: {K}")
print(f"Closed-loop poles: {poles}")
```

### Discrete-Time LQR

```python
import numpy as np
from ctrlsys import sb02md

def lqr_discrete(A, B, Q, R):
    """Solve discrete-time LQR problem.

    Minimizes: J = sum(x'Qx + u'Ru)
    Subject to: x[k+1] = Ax[k] + Bu[k]

    Args:
        A: State matrix (n x n)
        B: Input matrix (n x m)
        Q: State weight matrix (n x n)
        R: Input weight matrix (m x m)

    Returns:
        K: Optimal feedback gain (m x n), use u = -Kx
        P: Solution to Riccati equation
        poles: Closed-loop poles
    """
    n = A.shape[0]
    R_inv = np.linalg.inv(R)
    G = (B @ R_inv @ B.T).astype(float, order='F')

    A_f = np.asfortranarray(A.copy())
    Q_f = np.asfortranarray(Q.copy())

    # 'D' for discrete, 'U' for upper Hessenberg in closed-loop
    X, rcond, wr, wi, S, U, info = sb02md('D', 'D', 'U', 'N', 'U', n, A_f, G, Q_f)

    if info != 0:
        raise RuntimeError(f"sb02md failed with info={info}")

    K = np.linalg.inv(R + B.T @ X @ B) @ B.T @ X @ A
    poles = wr[:n] + 1j * wi[:n]

    return K, X, poles

# Example
A_d = np.array([[1, 0.1], [0, 1]], order='F', dtype=float)
B_d = np.array([[0.005], [0.1]], order='F', dtype=float)
Q = np.eye(2, order='F', dtype=float)
R = np.array([[1]], order='F', dtype=float)

K, P, poles = lqr_discrete(A_d, B_d, Q, R)
print(f"Discrete LQR gain K: {K}")
print(f"Closed-loop poles: {poles}")
print(f"All poles inside unit circle: {np.all(np.abs(poles) < 1)}")
```

## 6. Controllability and Observability

### Controllability Matrix

```python
import numpy as np

def controllability_matrix(A, B):
    """Compute controllability matrix [B, AB, A²B, ..., A^(n-1)B]."""
    n = A.shape[0]
    m = B.shape[1]
    Wc = np.zeros((n, n * m), dtype=float)

    AB = B.copy()
    for i in range(n):
        Wc[:, i*m:(i+1)*m] = AB
        AB = A @ AB

    return Wc

def is_controllable(A, B):
    """Check if system is controllable."""
    Wc = controllability_matrix(A, B)
    return np.linalg.matrix_rank(Wc) == A.shape[0]

# Example
A = np.array([[0, 1], [-2, -3]], order='F', dtype=float)
B = np.array([[0], [1]], order='F', dtype=float)

Wc = controllability_matrix(A, B)
print(f"Controllability matrix:\n{Wc}")
print(f"Controllable: {is_controllable(A, B)}")
```

### Observability Matrix

```python
import numpy as np

def observability_matrix(A, C):
    """Compute observability matrix [C; CA; CA²; ...; CA^(n-1)]."""
    n = A.shape[0]
    p = C.shape[0]
    Wo = np.zeros((n * p, n), dtype=float)

    CA = C.copy()
    for i in range(n):
        Wo[i*p:(i+1)*p, :] = CA
        CA = CA @ A

    return Wo

def is_observable(A, C):
    """Check if system is observable."""
    Wo = observability_matrix(A, C)
    return np.linalg.matrix_rank(Wo) == A.shape[0]

# Example
A = np.array([[0, 1], [-2, -3]], order='F', dtype=float)
C = np.array([[1, 0]], order='F', dtype=float)

Wo = observability_matrix(A, C)
print(f"Observability matrix:\n{Wo}")
print(f"Observable: {is_observable(A, C)}")
```

## 7. Feedback and Closed-Loop Systems

### State Feedback

```python
import numpy as np

def closed_loop_state_feedback(A, B, K):
    """Compute closed-loop system matrices with state feedback u = -Kx.

    Returns:
        A_cl: Closed-loop A matrix (A - BK)
    """
    return A - B @ K

# Example with LQR
A = np.array([[0, 1], [0, 0]], order='F', dtype=float)
B = np.array([[0], [1]], order='F', dtype=float)

# Design LQR gain
K, _, _ = lqr_continuous(A, B, np.eye(2), np.array([[1]]))

# Closed-loop system
A_cl = closed_loop_state_feedback(A, B, K)
print(f"Open-loop poles: {np.linalg.eigvals(A)}")
print(f"Closed-loop poles: {np.linalg.eigvals(A_cl)}")
```

### Output Feedback (Negative Unity)

```python
import numpy as np

def closed_loop_output_feedback(A, B, C, D, K=None):
    """Compute closed-loop system with output feedback u = r - Ky.

    If K=None, uses unity feedback (K=I).

    Closed-loop: A_cl = A - BK(I + DK)^{-1}C
    """
    if K is None:
        K = np.eye(C.shape[0])

    n_out = C.shape[0]
    S = np.linalg.inv(np.eye(n_out) + D @ K)
    A_cl = A - B @ K @ S @ C
    B_cl = B @ K @ S
    C_cl = S @ C
    D_cl = S @ D

    return A_cl, B_cl, C_cl, D_cl
```

## 8. Companion Matrix Construction

```python
import numpy as np

def companion_matrix(coeffs):
    """Create companion matrix from polynomial coefficients.

    For polynomial p(s) = s^n + a_{n-1}s^{n-1} + ... + a_1*s + a_0
    Input: [1, a_{n-1}, ..., a_1, a_0] (leading coefficient first)

    Returns companion matrix with eigenvalues = roots of polynomial.
    """
    coeffs = np.atleast_1d(coeffs).astype(float)
    coeffs = coeffs / coeffs[0]  # Normalize
    n = len(coeffs) - 1

    if n == 0:
        return np.zeros((0, 0), dtype=float)

    A = np.zeros((n, n), dtype=float)
    A[:-1, 1:] = np.eye(n - 1)
    A[-1, :] = -coeffs[1:][::-1]

    return A

# Example: s^3 + 6s^2 + 5s + 1
A = companion_matrix([1, 6, 5, 1])
print(f"Companion matrix:\n{A}")
print(f"Eigenvalues (roots): {np.linalg.eigvals(A)}")
```

## 9. Complete Design Example

```python
#!/usr/bin/env python3
"""Complete LQR design example for a DC motor system."""
import numpy as np
from ctrlsys import ab04md, sb02md, tf01md, tb05ad

# DC Motor model: theta'' + 10*theta' = 100*u
# State: x = [theta, theta']
A = np.array([[0, 1], [0, -10]], order='F', dtype=float)
B = np.array([[0], [100]], order='F', dtype=float)
C = np.array([[1, 0]], order='F', dtype=float)
D = np.array([[0]], order='F', dtype=float)

# LQR design
n = A.shape[0]
Q = np.diag([100, 1]).astype(float, order='F')  # Penalize position error
R = np.array([[1]], order='F', dtype=float)      # Control effort weight

R_inv = np.linalg.inv(R)
G = (B @ R_inv @ B.T).astype(float, order='F')
X, _, wr, wi, _, _, info = sb02md('C', 'D', 'U', 'N', 'S', n,
                                   np.asfortranarray(A), G,
                                   np.asfortranarray(Q))
K = R_inv @ B.T @ X
print(f"LQR gain K = {K}")

# Closed-loop system
A_cl = A - B @ K
print(f"Closed-loop poles: {np.linalg.eigvals(A_cl)}")

# Discretize for implementation (Ts = 0.01s)
dt = 0.01
A_d, B_d, C_d, D_d, _ = ab04md('C',
    np.asfortranarray(A_cl),
    np.asfortranarray(B),
    np.asfortranarray(C),
    np.asfortranarray(D),
    alpha=1.0, beta=2.0/dt)

# Simulate step response
n_steps = 500
u = np.ones((1, n_steps), order='F', dtype=float)
x0 = np.zeros(n, dtype=float)
y, _, _ = tf01md(A_d, B_d, C_d, D_d, u, x0)

# Plot
import matplotlib.pyplot as plt
t = np.arange(n_steps) * dt
plt.figure(figsize=(10, 4))
plt.plot(t, y.flatten())
plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.title('DC Motor Position Control with LQR')
plt.grid(True, alpha=0.3)
plt.show()
```

This reference covers the main control theory operations using ctrlsys. Key patterns:

- **State-space**: Use NumPy arrays with Fortran order (`order='F'`)
- **Transfer functions**: Convert to state-space using `tf_to_ss()` helper
- **Discretization**: Use `ab04md()` with appropriate alpha/beta for Tustin
- **Simulation**: Use `tf01md()` for time-domain response
- **Frequency response**: Use `tb05ad()` for evaluating H(s) or H(z)
- **LQR**: Use `sb02md()` Riccati solver for optimal control
