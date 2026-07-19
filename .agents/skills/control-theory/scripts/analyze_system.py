#!/usr/bin/env python3
"""
Analyze control system properties: poles, zeros, stability, frequency response.

Usage:
    python analyze_system.py <num> <den> [--dt DT] [--plot]

Examples:
    # Continuous SISO: 1/(s^2 + 2s + 1)
    python analyze_system.py "1" "1 2 1"

    # Discrete system with 0.1s sampling period
    python analyze_system.py "1" "1 -0.8" --dt 0.1

    # Include frequency response plot
    python analyze_system.py "1 1" "1 2 1" --plot
"""

import sys
import argparse
import numpy as np
from ctrlsys import tb05ad


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


def compute_frequency_response(A, B, C, D, w, dt=None):
    """Compute frequency response using tb05ad."""
    n = A.shape[0]
    mag = np.zeros(len(w))
    phase = np.zeros(len(w))

    for i, freq in enumerate(w):
        if dt is None:
            s = 1j * freq
        else:
            s = np.exp(1j * freq * dt)

        if n > 0:
            g, _, _, _, _, info = tb05ad('N', 'G',
                np.asfortranarray(A.copy()),
                np.asfortranarray(B.copy()),
                np.asfortranarray(C.copy()), s)
            H = g[0, 0] + D[0, 0]
        else:
            H = D[0, 0]

        mag[i] = np.abs(H)
        phase[i] = np.angle(H)

    return mag, phase, w


def analyze_continuous_system(num, den):
    """Analyze continuous-time transfer function."""
    A, B, C, D = tf_to_ss(num, den)
    n = A.shape[0]

    print("\n" + "="*60)
    print("CONTINUOUS-TIME SYSTEM ANALYSIS")
    print("="*60)

    poles = np.linalg.eigvals(A) if n > 0 else np.array([])
    zeros = np.roots(num) if len(num) > 1 else np.array([])

    is_stable = np.all(poles.real < 0) if len(poles) > 0 else True
    print(f"\nStability: {'STABLE' if is_stable else 'UNSTABLE'}")
    print(f"Poles: {poles}")
    print(f"Zeros: {zeros}")

    if len(poles) == 2 and np.any(np.iscomplex(poles)):
        real_part = poles[0].real
        imag_part = abs(poles[0].imag)
        omega_n = abs(poles[0])
        zeta = -real_part / omega_n if omega_n > 0 else 0
        print(f"\nNatural frequency wn: {omega_n:.4f} rad/s")
        print(f"Damping ratio zeta: {zeta:.4f}")

    try:
        if n > 0:
            dc_gain = (C @ np.linalg.inv(-A) @ B + D)[0, 0]
        else:
            dc_gain = D[0, 0]
        print(f"DC Gain: {dc_gain:.6f}")
    except:
        print("DC Gain: inf (pole at origin)")

    w = np.logspace(-2, 2, 500)
    mag, phase, w = compute_frequency_response(A, B, C, D, w)

    max_mag_idx = np.argmax(mag)
    print(f"\nPeak magnitude: {mag[max_mag_idx]:.4f} at w = {w[max_mag_idx]:.4f} rad/s")
    idx_1 = np.argmin(np.abs(w - 1))
    print(f"Magnitude at 1 rad/s: {mag[idx_1]:.4f}")

    return (A, B, C, D), (mag, phase, w)


def analyze_discrete_system(num, den, dt):
    """Analyze discrete-time transfer function."""
    A, B, C, D = tf_to_ss(num, den)
    n = A.shape[0]

    print("\n" + "="*60)
    print(f"DISCRETE-TIME SYSTEM ANALYSIS (Ts = {dt}s)")
    print("="*60)

    poles = np.linalg.eigvals(A) if n > 0 else np.array([])
    zeros = np.roots(num) if len(num) > 1 else np.array([])

    is_stable = np.all(np.abs(poles) < 1) if len(poles) > 0 else True
    print(f"\nStability: {'STABLE' if is_stable else 'UNSTABLE'}")
    print(f"Poles: {poles}")
    print(f"Pole magnitudes: {np.abs(poles)}")
    unstable_poles = poles[np.abs(poles) >= 1] if len(poles) > 0 else np.array([])
    if len(unstable_poles) > 0:
        print(f"Unstable poles: {unstable_poles}")

    print(f"Zeros: {zeros}")

    try:
        if n > 0:
            dc_gain = (C @ np.linalg.inv(np.eye(n) - A) @ B + D)[0, 0]
        else:
            dc_gain = D[0, 0]
        print(f"\nDC Gain: {dc_gain:.6f}")
    except:
        print("\nDC Gain: inf (pole at z=1)")

    nyquist_freq = np.pi / dt
    print(f"Nyquist frequency: {nyquist_freq:.4f} rad/s ({nyquist_freq/(2*np.pi):.4f} Hz)")

    w = np.logspace(-2, np.log10(nyquist_freq), 500)
    mag, phase, w = compute_frequency_response(A, B, C, D, w, dt)

    max_mag_idx = np.argmax(mag)
    print(f"\nPeak magnitude: {mag[max_mag_idx]:.4f} at w = {w[max_mag_idx]:.4f} rad/s")

    return (A, B, C, D), (mag, phase, w)


def plot_frequency_response(sys, mag, phase, w, title, dt=None):
    """Plot magnitude and phase response."""
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.semilogx(w, 20*np.log10(mag + 1e-10))
        ax1.set_ylabel('Magnitude (dB)')
        ax1.set_title(f'{title} - Magnitude Response')
        ax1.grid(True, which='both', alpha=0.3)

        ax2.semilogx(w, np.degrees(phase))
        ax2.set_xlabel('Frequency (rad/s)')
        ax2.set_ylabel('Phase (degrees)')
        ax2.set_title(f'{title} - Phase Response')
        ax2.grid(True, which='both', alpha=0.3)

        if dt is not None:
            nyquist = np.pi / dt
            ax1.axvline(x=nyquist, color='r', linestyle='--', alpha=0.5, label='Nyquist')
            ax2.axvline(x=nyquist, color='r', linestyle='--', alpha=0.5, label='Nyquist')
            ax1.legend()
            ax2.legend()

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("\nmatplotlib not available - skipping plot")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze control system poles, zeros, and frequency response',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('numerator', help='Numerator coefficients (space-separated)')
    parser.add_argument('denominator', help='Denominator coefficients (space-separated)')
    parser.add_argument('--dt', type=float, default=None,
                        help='Sampling period for discrete systems')
    parser.add_argument('--plot', action='store_true',
                        help='Plot frequency response')

    args = parser.parse_args()

    try:
        num = parse_polynomial(args.numerator)
        den = parse_polynomial(args.denominator)

        if args.dt is None:
            sys_matrices, fresp = analyze_continuous_system(num, den)
            title = "Continuous-Time System"
        else:
            sys_matrices, fresp = analyze_discrete_system(num, den, args.dt)
            title = f"Discrete-Time System (Ts={args.dt}s)"

        if args.plot:
            mag, phase, w = fresp
            plot_frequency_response(sys_matrices, mag, phase, w, title, args.dt)

    except ValueError as e:
        print(f"Error parsing polynomials: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
