#!/usr/bin/env python3
"""
Design Linear Quadratic Regulator (LQR) controllers for systems.

Designs optimal feedback gains by solving the LQR problem with user-specified
state and input weights. Computes closed-loop poles and simulation response.

Usage:
    python lqr_design.py --help
    python lqr_design.py <A> <B> <C> --Q Q_weight --R R_weight [--dt DT] [--plot]

Examples:
    # SISO system: A=[0 1; -1 -1], B=[0; 1], C=[1 0], Q=1, R=1
    python lqr_design.py "0 1 -1 -1" "0 1" "1 0" --Q 1 --R 1

    # MIMO system with matrix Q (diagonal 2x2)
    python lqr_design.py "0 1 -1 -2" "0 1 1 0" "1 0 0 1" --Q 1 10 --R 1 --plot

    # Discrete-time LQR with 0.01s sampling
    python lqr_design.py "0 1 -1 -2" "0 1 1 0" "1 0 0 1" \
        --Q 1 10 --R 1 --dt 0.01

Matrix Input Format:
    - For matrix A (state): "a11 a12 ... a1n a21 a22 ... a2n ..." (row-major)
    - For vector B (input): "b1 b2 ... bm" or matrix "b11 b12 ..." if MIMO
    - For vector C (output): "c1 c2 ... cn"
"""

import sys
import argparse
import numpy as np
from ctrlsys import sb02md, ab04md


def parse_matrix(matrix_str, rows, cols):
    """Parse space-separated matrix values in row-major order."""
    values = [float(x) for x in matrix_str.split()]
    if len(values) != rows * cols:
        raise ValueError(f"Expected {rows*cols} values, got {len(values)}")
    return np.array(values, order='F').reshape(rows, cols, order='C')


def infer_dimensions(A_str, B_str, C_str):
    """Infer matrix dimensions from inputs."""
    A_vals = [float(x) for x in A_str.split()]
    B_vals = [float(x) for x in B_str.split()]
    C_vals = [float(x) for x in C_str.split()]

    n = int(np.sqrt(len(A_vals)))
    if n * n != len(A_vals):
        raise ValueError(f"A must be square matrix, got {len(A_vals)} values")

    if len(B_vals) == n:
        n_inputs = 1
    elif len(B_vals) % n == 0:
        n_inputs = len(B_vals) // n
    else:
        raise ValueError(f"B dimension mismatch: {len(B_vals)} values for {n} states")

    if len(C_vals) == n:
        n_outputs = 1
    elif len(C_vals) % n == 0:
        n_outputs = len(C_vals) // n
    else:
        raise ValueError(f"C dimension mismatch: {len(C_vals)} values for {n} states")

    return n, n_inputs, n_outputs


def parse_Q_R(Q_str, R_str, n_states, n_inputs):
    """Parse Q and R matrices from user input."""
    Q_vals = [float(x) for x in Q_str.split()]
    R_vals = [float(x) for x in R_str.split()]

    if len(Q_vals) == 1:
        Q = Q_vals[0] * np.eye(n_states, order='F')
    elif len(Q_vals) == n_states:
        Q = np.diag(Q_vals).astype(float, order='F')
    elif len(Q_vals) == n_states * n_states:
        Q = parse_matrix(Q_str, n_states, n_states)
    else:
        raise ValueError(f"Q size mismatch: expected 1, {n_states}, or {n_states**2} values")

    if len(R_vals) == 1:
        R = R_vals[0] * np.eye(n_inputs, order='F')
    elif len(R_vals) == n_inputs:
        R = np.diag(R_vals).astype(float, order='F')
    elif len(R_vals) == n_inputs * n_inputs:
        R = parse_matrix(R_str, n_inputs, n_inputs)
    else:
        raise ValueError(f"R size mismatch: expected 1, {n_inputs}, or {n_inputs**2} values")

    return Q, R


def solve_lqr_continuous(A, B, Q, R):
    """Solve continuous-time LQR using sb02md (Riccati solver)."""
    n = A.shape[0]
    R_inv = np.linalg.inv(R)
    G = (B @ R_inv @ B.T).astype(float, order='F')
    A_f = np.asfortranarray(A.copy())
    Q_f = np.asfortranarray(Q.copy())
    X, rcond, wr, wi, S, U, info = sb02md('C', 'D', 'U', 'N', 'S', n, A_f, G, Q_f)
    if info != 0:
        raise RuntimeError(f"sb02md failed with info={info}")
    K = R_inv @ B.T @ X
    A_cl = A - B @ K
    poles = np.linalg.eigvals(A_cl)
    return K, X, poles


def solve_lqr_discrete(A, B, Q, R):
    """Solve discrete-time LQR using sb02md (Riccati solver)."""
    n = A.shape[0]
    R_inv = np.linalg.inv(R)
    G = (B @ R_inv @ B.T).astype(float, order='F')
    A_f = np.asfortranarray(A.copy())
    Q_f = np.asfortranarray(Q.copy())
    X, rcond, wr, wi, S, U, info = sb02md('D', 'D', 'U', 'N', 'U', n, A_f, G, Q_f)
    if info != 0:
        raise RuntimeError(f"sb02md failed with info={info}")
    K = np.linalg.inv(R + B.T @ X @ B) @ B.T @ X @ A
    A_cl = A - B @ K
    poles = np.linalg.eigvals(A_cl)
    return K, X, poles


def discretize_system(A, B, C, D, dt):
    """Discretize continuous system using bilinear (Tustin) transformation via ab04md."""
    alpha = 1.0
    beta = 2.0 / dt
    A_f = np.asfortranarray(A.copy())
    B_f = np.asfortranarray(B.copy())
    C_f = np.asfortranarray(C.copy())
    D_f = np.asfortranarray(D.copy())
    A_d, B_d, C_d, D_d, info = ab04md('C', A_f, B_f, C_f, D_f, alpha=alpha, beta=beta)
    if info != 0:
        raise RuntimeError(f"ab04md failed with info={info}")
    return A_d, B_d, C_d, D_d


def design_lqr(A, B, C, D, Q, R, dt=None):
    """Design LQR controller."""
    if dt is not None:
        A_d, B_d, C_d, D_d = discretize_system(A, B, C, D, dt)
        K, P, poles = solve_lqr_discrete(A_d, B_d, Q, R)
        return (A_d, B_d, C_d, D_d), K, P, poles
    else:
        K, P, poles = solve_lqr_continuous(A, B, Q, R)
        return (A, B, C, D), K, P, poles


def print_results(sys_matrices, K, P, poles, dt=None):
    """Print LQR design results."""
    A, B, C, D = sys_matrices
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]

    print("\n" + "="*70)
    print("LQR DESIGN RESULTS")
    print("="*70)

    system_type = "Discrete-time" if dt is not None else "Continuous-time"
    print(f"\nSystem Type: {system_type}")
    if dt is not None:
        print(f"Sampling Period: {dt}s")

    print(f"\nSystem Dimension: {n} states, {m} inputs, {p} outputs")

    print(f"\nA matrix:\n{A}")
    print(f"\nB matrix:\n{B}")
    print(f"\nC matrix:\n{C}")

    print(f"\n{'-'*70}")
    print("OPTIMAL FEEDBACK GAIN K")
    print(f"{'-'*70}")
    print(f"Dimensions: {K.shape[0]} x {K.shape[1]}")
    print(f"\nK matrix:\n{K}")

    print(f"\n{'-'*70}")
    print("RICCATI SOLUTION P")
    print(f"{'-'*70}")
    print(f"Dimensions: {P.shape}")
    print(f"\nP matrix:\n{P}")

    print(f"\n{'-'*70}")
    print("CLOSED-LOOP EIGENVALUES")
    print(f"{'-'*70}")
    print(f"Closed-loop poles (feedback u = -Kx):")
    for i, pole in enumerate(poles):
        if dt is None:
            print(f"  p{i+1} = {pole:.6f}")
        else:
            print(f"  z{i+1} = {pole:.6f} (|z| = {abs(pole):.6f})")

    if dt is None:
        stable = np.all(poles.real < 0)
    else:
        stable = np.all(np.abs(poles) < 1)

    stability_str = "STABLE" if stable else "UNSTABLE"
    print(f"\nClosed-loop stability: {stability_str}")


def plot_results(sys_matrices, K, dt=None):
    """Plot step response and closed-loop dynamics."""
    try:
        import matplotlib.pyplot as plt
        from ctrlsys import tf01md

        A, B, C, D = sys_matrices
        n = A.shape[0]

        A_cl = (A - B @ K).astype(float, order='F')
        B_cl = np.eye(n, order='F', dtype=float)
        C_cl = np.asfortranarray(C.copy())
        D_cl = np.zeros((C.shape[0], n), order='F', dtype=float)

        if dt is None:
            max_pole_freq = max(np.abs(np.linalg.eigvals(A_cl)))
            dt_sim = min(0.01, 1.0 / (20 * max(max_pole_freq, 0.1)))
            t_final = 5.0
        else:
            dt_sim = dt
            t_final = 5.0

        n_steps = int(t_final / dt_sim)
        t = np.arange(n_steps) * dt_sim

        if dt is None:
            alpha = 1.0
            beta = 2.0 / dt_sim
            A_d, B_d, C_d, D_d, info = ab04md('C',
                np.asfortranarray(A_cl.copy()),
                np.asfortranarray(B_cl.copy()),
                np.asfortranarray(C_cl.copy()),
                np.asfortranarray(D_cl.copy()),
                alpha=alpha, beta=beta)
        else:
            A_d, B_d, C_d, D_d = A_cl, B_cl, C_cl, D_cl

        u = np.zeros((n, n_steps), order='F', dtype=float)
        x0 = np.zeros(n, dtype=float)
        x0[0] = 1.0

        y, x_final, info = tf01md(A_d, B_d, C_d, D_d, u, x0)

        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(y.shape[0]):
            ax.plot(t, y[i, :], label=f'Output {i+1}', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Output')
        ax.set_title('Closed-loop Response with LQR (Initial Condition x0[0]=1)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("\nmatplotlib not available - skipping plots")
    except Exception as e:
        print(f"\nCould not plot results: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Design LQR controllers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('A', help='State matrix A (row-major, space-separated)')
    parser.add_argument('B', help='Input matrix B (row-major, space-separated)')
    parser.add_argument('C', help='Output matrix C (row-major, space-separated)')
    parser.add_argument('--D', default=None, help='Feedthrough matrix D')
    parser.add_argument('--Q', required=True,
                        help='State weight Q (scalar or diagonal values)')
    parser.add_argument('--R', required=True,
                        help='Input weight R (scalar or diagonal values)')
    parser.add_argument('--dt', type=float, default=None,
                        help='Sampling period for discrete-time design')
    parser.add_argument('--plot', action='store_true',
                        help='Plot step response and state trajectory')

    args = parser.parse_args()

    try:
        n, n_inputs, n_outputs = infer_dimensions(args.A, args.B, args.C)

        A = parse_matrix(args.A, n, n)
        B = parse_matrix(args.B, n, n_inputs)
        C = parse_matrix(args.C, n_outputs, n)
        D = np.zeros((n_outputs, n_inputs), order='F', dtype=float)
        if args.D:
            D = parse_matrix(args.D, n_outputs, n_inputs)

        Q, R = parse_Q_R(args.Q, args.R, n, n_inputs)

        sys_matrices, K, P, poles = design_lqr(A, B, C, D, Q, R, args.dt)

        print_results(sys_matrices, K, P, poles, args.dt)

        if args.plot:
            plot_results(sys_matrices, K, args.dt)

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
