#!/usr/bin/env python3
"""
Discretize continuous-time controllers using Tustin (bilinear) transformation.

Converts continuous state-space to discrete-time and shows
the resulting discrete transfer function and pole mapping.

Usage:
    python discretize_controller.py <num> <den> <dt> [--plot]

Examples:
    # Discretize 1/(s+1) with 0.01s sampling period
    python discretize_controller.py "1" "1 1" 0.01

    # Plot pole-zero maps and step responses
    python discretize_controller.py "1" "1 1" 0.01 --plot
"""

import sys
import argparse
import numpy as np
from ctrlsys import ab04md, tf01md, tb05ad


def parse_polynomial(poly_str):
    """Convert space-separated string to polynomial coefficients."""
    return [float(x) for x in poly_str.split()]


def tf_to_ss(num, den):
    """Convert SISO transfer function to controllable canonical state-space."""
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


def discretize_controller(num, den, dt):
    """Discretize continuous controller using Tustin transformation via ab04md."""
    A, B, C, D = tf_to_ss(num, den)
    n = A.shape[0]

    print("\n" + "="*70)
    print(f"DISCRETIZATION (Ts = {dt}s)")
    print("="*70)

    poles_c = np.linalg.eigvals(A) if n > 0 else np.array([])
    print(f"\nContinuous System:")
    print(f"  Order: {n}")
    print(f"  Poles: {poles_c}")
    print(f"  Stability: {'STABLE' if np.all(poles_c.real < 0) else 'UNSTABLE'}")

    if n > 0:
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
    else:
        A_d, B_d, C_d, D_d = A, B, C, D

    poles_d = np.linalg.eigvals(A_d) if n > 0 else np.array([])

    print(f"\n{'-'*70}")
    print("DISCRETE SYSTEM (Tustin/Bilinear)")
    print(f"{'-'*70}")
    print(f"\nDiscrete A matrix:\n{A_d}")
    print(f"\nDiscrete B matrix:\n{B_d}")
    print(f"\nDiscrete C matrix:\n{C_d}")
    print(f"\nDiscrete D matrix:\n{D_d}")

    print(f"\nDiscrete Poles: {poles_d}")
    print(f"Pole Magnitudes: {np.abs(poles_d)}")

    stability = "STABLE" if np.all(np.abs(poles_d) < 1) else "UNSTABLE"
    print(f"Stability: {stability}")

    if n > 0:
        print(f"\nPole Mapping (s -> z):")
        for i, (sc, sd) in enumerate(zip(poles_c, poles_d)):
            print(f"  p{i+1}: {sc:8.4f} -> {sd:8.4f} (|z|={abs(sd):.4f})")

    return (A, B, C, D), (A_d, B_d, C_d, D_d), dt


def plot_discretization(sys_c, sys_d, dt):
    """Plot pole-zero maps and step responses."""
    try:
        import matplotlib.pyplot as plt

        A_c, B_c, C_c, D_c = sys_c
        A_d, B_d, C_d, D_d = sys_d
        n = A_c.shape[0]

        fig = plt.figure(figsize=(14, 10))

        ax1 = plt.subplot(2, 2, 1)
        theta = np.linspace(0, 2*np.pi, 100)
        ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1)

        if n > 0:
            poles_c = np.linalg.eigvals(A_c)
            poles_d = np.linalg.eigvals(A_d)
            ax1.scatter(poles_c.real, poles_c.imag, marker='x', s=100,
                       color='red', label='Continuous poles', zorder=5)
            ax1.scatter(poles_d.real, poles_d.imag, marker='x', s=100,
                       color='blue', label='Discrete poles', zorder=5)

        ax1.set_xlabel('Real')
        ax1.set_ylabel('Imaginary')
        ax1.set_title('Pole-Zero Map')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        ax1.legend(fontsize=8)
        ax1.axhline(y=0, color='k', linewidth=0.5)
        ax1.axvline(x=0, color='k', linewidth=0.5)

        ax2 = plt.subplot(2, 2, 2)
        w = np.logspace(-2, np.log10(np.pi/dt), 200)
        mag_c = np.zeros(len(w))
        mag_d = np.zeros(len(w))

        for i, freq in enumerate(w):
            s = 1j * freq
            if n > 0:
                g, _, _, _, _, info = tb05ad('N', 'G',
                    np.asfortranarray(A_c.copy()),
                    np.asfortranarray(B_c.copy()),
                    np.asfortranarray(C_c.copy()), s)
                mag_c[i] = np.abs(g[0, 0] + D_c[0, 0])
            else:
                mag_c[i] = np.abs(D_c[0, 0])

            z = np.exp(1j * freq * dt)
            if n > 0:
                g_d, _, _, _, _, info = tb05ad('N', 'G',
                    np.asfortranarray(A_d.copy()),
                    np.asfortranarray(B_d.copy()),
                    np.asfortranarray(C_d.copy()), z)
                mag_d[i] = np.abs(g_d[0, 0] + D_d[0, 0])
            else:
                mag_d[i] = np.abs(D_d[0, 0])

        nyquist = np.pi / dt
        ax2.loglog(w, mag_c, 'r-', label='Continuous', linewidth=2)
        ax2.loglog(w, mag_d, 'b--', label='Discrete (Tustin)', linewidth=2)
        ax2.axvline(x=nyquist, color='gray', linestyle='--', alpha=0.5,
                   label=f'Nyquist ({nyquist:.2f} rad/s)')
        ax2.set_xlabel('Frequency (rad/s)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('Magnitude Response')
        ax2.grid(True, which='both', alpha=0.3)
        ax2.legend(fontsize=8)

        ax3 = plt.subplot(2, 2, 3)
        if n > 0:
            max_pole_freq = max(np.abs(np.linalg.eigvals(A_c)))
            dt_sim = min(dt/10, 1.0 / (20 * max(max_pole_freq, 0.1)))
        else:
            dt_sim = dt / 10
        t_cont = np.arange(0, 5, dt_sim)

        if n > 0:
            A_sim, B_sim, C_sim, D_sim, _ = ab04md('C',
                np.asfortranarray(A_c.copy()),
                np.asfortranarray(B_c.copy()),
                np.asfortranarray(C_c.copy()),
                np.asfortranarray(D_c.copy()),
                alpha=1.0, beta=2.0/dt_sim)
            u_cont = np.ones((1, len(t_cont)), order='F', dtype=float)
            x0 = np.zeros(n, dtype=float)
            y_cont, _, _ = tf01md(A_sim, B_sim, C_sim, D_sim, u_cont, x0)
            y_cont = y_cont.flatten()
        else:
            y_cont = D_c[0, 0] * np.ones(len(t_cont))

        t_disc = np.arange(0, 5, dt)
        if n > 0:
            u_disc = np.ones((1, len(t_disc)), order='F', dtype=float)
            x0_d = np.zeros(n, dtype=float)
            y_disc, _, _ = tf01md(A_d, B_d, C_d, D_d, u_disc, x0_d)
            y_disc = y_disc.flatten()
        else:
            y_disc = D_d[0, 0] * np.ones(len(t_disc))

        ax3.plot(t_cont, y_cont, 'r-', label='Continuous', linewidth=2)
        ax3.plot(t_disc, y_disc, 'bo-', label=f'Discrete (Ts={dt}s)',
                markersize=4, alpha=0.7)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Response')
        ax3.set_title('Step Response')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)

        ax4 = plt.subplot(2, 2, 4)
        ax4.text(0.5, 0.5, f"Sampling Period: {dt}s\n"
                          f"Nyquist Frequency: {nyquist:.2f} rad/s\n"
                          f"System Order: {n}",
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.axis('off')

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("\nmatplotlib not available - skipping plots")


def main():
    parser = argparse.ArgumentParser(
        description='Discretize continuous controllers using Tustin transformation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('numerator', help='Numerator coefficients (space-separated)')
    parser.add_argument('denominator', help='Denominator coefficients (space-separated)')
    parser.add_argument('dt', type=float, help='Sampling period (seconds)')
    parser.add_argument('--plot', action='store_true',
                        help='Plot pole-zero maps and frequency responses')

    args = parser.parse_args()

    try:
        num = parse_polynomial(args.numerator)
        den = parse_polynomial(args.denominator)

        sys_c, sys_d, dt = discretize_controller(num, den, args.dt)

        if args.plot:
            plot_discretization(sys_c, sys_d, args.dt)

    except ValueError as e:
        print(f"Error parsing polynomials: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
