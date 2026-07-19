# ctrlsys Workflows

## 1. LQR Controller Design

Solve continuous-time algebraic Riccati equation for optimal state feedback.

```python
import numpy as np
from ctrlsys import sb02md

n = 2
A = np.array([[0, 1], [-2, -3]], dtype=float, order='F')
B = np.array([[0], [1]], dtype=float, order='F')
Q = np.eye(n, dtype=float, order='F')
R = np.array([[1.0]], dtype=float, order='F')

# Form G = B @ inv(R) @ B.T
G = B @ np.linalg.solve(R, B.T)
G = np.asfortranarray(G)

# Solve ARE: A'X + XA - XGX + Q = 0
X, rcond, wr, wi, S, U, info = sb02md(
    'C',  # continuous
    'D',  # H matrix form
    'U',  # upper triangle
    'N',  # no scaling
    'S',  # stable eigenvalues first
    n, A.copy(), G, Q.copy()
)

K = np.linalg.solve(R, B.T @ X)  # Optimal gain
```

## 2. Pole Placement

Assign closed-loop eigenvalues using state feedback.

```python
import numpy as np
from ctrlsys import sb01bd

n, m = 3, 1
A = np.array([[0, 1, 0], [0, 0, 1], [-6, -11, -6]], dtype=float, order='F')
B = np.array([[0], [0], [1]], dtype=float, order='F')

# Desired poles
wr = np.array([-1.0, -2.0, -3.0], dtype=float)
wi = np.zeros(3, dtype=float)

A_schur, wr_out, wi_out, nfp, nap, nup, F, Z, iwarn, info = sb01bd(
    'C',     # continuous
    n, m,
    len(wr), # number of poles to assign
    0.0,     # alpha (stability threshold)
    A.copy(), B.copy(), wr.copy(), wi.copy(), 0.0
)

# Closed-loop: A + B @ F has eigenvalues at -1, -2, -3
```

## 3. Model Reduction (Balanced Truncation)

Reduce system order using Balance & Truncate method.

```python
import numpy as np
from ctrlsys import ab09ad

n, m, p = 6, 1, 1
# Assume A, B, C are defined (stable system, order n)
A = ...  # n x n, stable
B = ...  # n x m
C = ...  # p x n

nr = 3  # desired reduced order

ar, br, cr, hsv, nr_out, iwarn, info = ab09ad(
    'C',    # continuous
    'B',    # square-root B&T method
    'N',    # no equilibration
    'F',    # fixed order (use nr)
    n, m, p, nr,
    A.copy(), B.copy(), C.copy(), 0.0
)
# hsv contains Hankel singular values
# nr_out is actual reduced order achieved
```

## 4. System Identification (N4SID/MOESP)

Estimate state-space model from input-output data.

```python
import numpy as np
from ctrlsys import ib01ad, ib01bd

# Input-output data: u (nsmp x m), y (nsmp x l)
nsmp, m, l = 1000, 1, 1
nobr = 15  # block rows
u = ...    # (nsmp, m) Fortran order
y = ...    # (nsmp, l) Fortran order

# Step 1: Preprocessing and order estimation
n_est, r, sv, iwarn, info = ib01ad(
    'M',    # MOESP method
    'C',    # Cholesky algorithm
    'N',    # no B,D via MOESP
    'O',    # one batch
    'N',    # no connection
    'N',    # no control
    nobr, m, l, u, y, 0.0, -1.0
)

# Step 2: Estimate system matrices
n = max(n_est, 1)  # ensure n >= 1
A, C, B, D, Q, Ry, S, K, iwarn, info = ib01bd(
    'C',    # combined method
    'A',    # all matrices
    'K',    # compute Kalman gain
    nobr, n, m, l, nsmp, r, 0.0
)
```

## 5. H-infinity Controller

Design H-infinity optimal controller.

```python
import numpy as np
from ctrlsys import sb10ad

# Generalized plant P partitioned as:
#   [A  | B1  B2 ]
#   [C1 | D11 D12]
#   [C2 | D21 D22]

n = 4       # states
m1, m2 = 1, 1  # disturbance inputs, control inputs
p1, p2 = 1, 1  # performance outputs, measurements

# Define matrices (all F-order)...
A = ...
B = np.asfortranarray(np.hstack([B1, B2]))
C = np.asfortranarray(np.vstack([C1, C2]))
D = np.asfortranarray(np.block([[D11, D12], [D21, D22]]))

ncon = m2   # control inputs
nmeas = p2  # measurements
gamma = 10.0  # initial H-inf bound

ak, bk, ck, dk, ac, bc, cc, dc, gamma_opt, rcond, info = sb10ad(
    2,      # job: scan from gamma to 0
    n, m1 + m2, p1 + p2,
    ncon, nmeas,
    A.copy(), B.copy(), C.copy(), D.copy(),
    gamma, 0.0, 0.0
)
# Controller K: u = ck @ xk + dk @ y
# gamma_opt: achieved H-inf norm bound
```

## 6. Controllability Check

Verify controllability and find controllable realization.

```python
import numpy as np
from ctrlsys import ab01nd

n, m = 4, 2
A = ...  # n x n, F-order
B = ...  # n x m, F-order

a_out, b_out, ncont, indcon, nblk, z, tau, info = ab01nd(
    'I',    # form transformation matrix
    A.copy(), B.copy(), 0.0
)

if ncont == n:
    print("System is controllable")
else:
    print(f"Controllable subspace dimension: {ncont}")
```

## 7. Continuous to Discrete Conversion

Bilinear (Tustin) transformation.

```python
import numpy as np
from ctrlsys import ab04md

Ts = 0.1  # sampling time
A = np.array([[0, 1], [-2, -3]], dtype=float, order='F')
B = np.array([[0], [1]], dtype=float, order='F')
C = np.array([[1, 0]], dtype=float, order='F')
D = np.array([[0]], dtype=float, order='F')

Ad, Bd, Cd, Dd, info = ab04md(
    'C',    # continuous to discrete
    A.copy(), B.copy(), C.copy(), D.copy(),
    alpha=2.0/Ts, beta=1.0
)
```

## 8. Lyapunov Equation

Solve A'X + XA + Q = 0 (or discrete variant).

```python
import numpy as np
from ctrlsys import sb03md

n = 3
A = np.array([[-1, 0.5, 0], [0, -2, 0.5], [0, 0, -3]], dtype=float, order='F')
C = np.eye(n, dtype=float, order='F')  # Q = C'C

X, A_schur, U, wr, wi, scale, sep, ferr, info = sb03md(
    'C',    # continuous
    'X',    # solve for X
    'N',    # not factored
    'T',    # transpose form
    n,
    A.copy(),
    C.copy(),
    0.0
)
```
