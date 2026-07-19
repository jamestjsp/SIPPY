#!/usr/bin/env python3
"""
Digital PID Controller with Anti-windup

Python implementation of the PID controller from the IFAC Computer Control paper
(Listing 1), converted from Java. Uses ctrlsys for control system integration.

Reference:
    Wittenmark, B., Astrom, K.J., and Arzen, K.-E., "Computer Control: An Overview,"
    IFAC Professional Brief, 2002.

Author: Converted from Java to Python
"""

import numpy as np
from typing import Optional
from ctrlsys import ab04md


class PIDController:
    """
    Digital PID controller with anti-windup and bumpless transfer.

    Implements the discrete-time PID algorithm with:
    - Proportional, Integral, and Derivative action
    - Setpoint weighting
    - Anti-windup using back-calculation method
    - Output limiting
    - Bumpless parameter changes

    The controller uses the position form:
        u(k) = P + I + D

    where:
        P = K * (b * uc - y)        # Proportional term with setpoint weighting
        I = integral state           # Integral term
        D = derivative state         # Filtered derivative term

    Parameters
    ----------
    K : float
        Proportional gain
    Ti : float
        Integral time constant (seconds)
    Td : float
        Derivative time constant (seconds)
    Tt : float, optional
        Anti-windup tracking time constant (default: 10.0)
    N : float, optional
        Maximum derivative gain / filter constant (default: 10.0)
    b : float, optional
        Setpoint weighting for proportional term (default: 1.0)
        Set b<1 to reduce overshoot
    ulow : float, optional
        Lower output limit (default: -1.0)
    uhigh : float, optional
        Upper output limit (default: 1.0)
    h : float
        Sampling period (seconds)

    Attributes
    ----------
    signals : dict
        Current signal values (uc, y, v, u)
    states : dict
        Internal controller states (I, D, yold)
    params : dict
        Controller parameters

    Examples
    --------
    >>> # Create PID controller for temperature control
    >>> pid = PIDController(K=2.0, Ti=0.5, Td=0.1, h=0.01)
    >>>
    >>> # Control loop
    >>> setpoint = 100.0  # Target temperature
    >>> measurement = 95.0  # Current temperature
    >>> control = pid.calculate_output(setpoint, measurement)
    >>> pid.update_state(control)
    """

    def __init__(
        self,
        K: float = 4.4,
        Ti: float = 0.4,
        Td: float = 0.2,
        h: float = 0.03,
        Tt: float = 10.0,
        N: float = 10.0,
        b: float = 1.0,
        ulow: float = -1.0,
        uhigh: float = 1.0
    ):
        if Ti <= 0:
            raise ValueError(f"Ti must be positive, got {Ti}")
        if h <= 0:
            raise ValueError(f"h (sampling period) must be positive, got {h}")
        if ulow >= uhigh:
            raise ValueError(f"ulow must be < uhigh, got ulow={ulow}, uhigh={uhigh}")
        if Tt <= 0:
            raise ValueError(f"Tt must be positive, got {Tt}")

        self.params = {
            'K': K,
            'Ti': Ti,
            'Td': Td,
            'Tt': Tt,
            'N': N,
            'b': b,
            'ulow': ulow,
            'uhigh': uhigh,
            'h': h
        }

        self._compute_coefficients()

        self.signals = {
            'uc': 0.0,
            'y': 0.0,
            'v': 0.0,
            'u': 0.0
        }

        self.states = {
            'I': 0.0,
            'D': 0.0,
            'yold': 0.0
        }

    def _compute_coefficients(self):
        """Compute discretization coefficients for PID implementation."""
        p = self.params
        p['bi'] = p['K'] * p['h'] / p['Ti']
        p['ar'] = p['h'] / p['Tt']
        p['ad'] = p['Td'] / (p['Td'] + p['N'] * p['h'])
        p['bd'] = p['K'] * p['N'] * p['ad']

    def calculate_output(self, uc: float, y: float) -> float:
        """
        Calculate PID controller output.

        Parameters
        ----------
        uc : float
            Setpoint (command signal)
        y : float
            Measured variable (process output)

        Returns
        -------
        float
            Limited control signal u(k)
        """
        p = self.params

        self.signals['uc'] = uc
        self.signals['y'] = y

        P = p['K'] * (p['b'] * uc - y)

        self.states['D'] = (
            p['ad'] * self.states['D']
            - p['bd'] * (y - self.states['yold'])
        )

        self.signals['v'] = P + self.states['I'] + self.states['D']

        if self.signals['v'] < p['ulow']:
            self.signals['u'] = p['ulow']
        elif self.signals['v'] > p['uhigh']:
            self.signals['u'] = p['uhigh']
        else:
            self.signals['u'] = self.signals['v']

        return self.signals['u']

    def update_state(self, u: float):
        """
        Update internal controller states (call after calculate_output).

        Parameters
        ----------
        u : float
            Actual control signal applied to the process
        """
        p = self.params

        self.states['I'] += (
            p['bi'] * (self.signals['uc'] - self.signals['y'])
            + p['ar'] * (u - self.signals['v'])
        )

        self.states['yold'] = self.signals['y']

    def reset(self):
        """Reset controller to initial state."""
        self.states['I'] = 0.0
        self.states['D'] = 0.0
        self.states['yold'] = 0.0
        self.signals = {key: 0.0 for key in self.signals}

    def set_parameters(self, **kwargs):
        """Update controller parameters with bumpless transfer."""
        p_old = self.params.copy()

        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

        self._compute_coefficients()

        if 'K' in kwargs or 'b' in kwargs:
            uc, y = self.signals['uc'], self.signals['y']
            P_old = p_old['K'] * (p_old['b'] * uc - y)
            P_new = self.params['K'] * (self.params['b'] * uc - y)
            self.states['I'] += P_old - P_new

    def __repr__(self) -> str:
        """String representation of the controller."""
        p = self.params
        return (
            f"PIDController(K={p['K']:.3f}, Ti={p['Ti']:.3f}, "
            f"Td={p['Td']:.3f}, h={p['h']:.4f})"
        )


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


def simulate_pid_control(
    pid: PIDController,
    plant_ss: tuple,
    setpoint: float,
    duration: float,
    disturbance: Optional[np.ndarray] = None
) -> tuple:
    """
    Simulate PID control of a plant using ctrlsys.

    Parameters
    ----------
    pid : PIDController
        PID controller instance
    plant_ss : tuple
        Plant model as state-space (A, B, C, D) tuple
    setpoint : float
        Desired setpoint value
    duration : float
        Simulation duration (seconds)
    disturbance : np.ndarray, optional
        Input disturbance signal (same length as time vector)

    Returns
    -------
    tuple
        (time, output, control_signal, setpoint_vector)

    Examples
    --------
    >>> # Create a first-order plant: 1/(s+1)
    >>> plant = tf_to_ss([1], [1, 1])
    >>> # Create PID controller
    >>> pid = PIDController(K=2.0, Ti=1.0, Td=0.1, h=0.01)
    >>> # Simulate
    >>> t, y, u, sp = simulate_pid_control(pid, plant, setpoint=1.0, duration=10.0)
    """
    h = pid.params['h']
    time = np.arange(0, duration, h)
    n_steps = len(time)

    A, B, C, D = plant_ss
    n = A.shape[0]

    output = np.zeros(n_steps)
    control = np.zeros(n_steps)
    setpoint_vec = np.full(n_steps, setpoint)

    x = np.zeros(n, dtype=float)

    pid.reset()

    for k in range(n_steps):
        if n > 0:
            y_k = (C @ x + D @ np.array([[0.0]]))[0, 0]
        else:
            y_k = D[0, 0] * 0.0
        output[k] = y_k

        u_k = pid.calculate_output(setpoint, y_k)

        if disturbance is not None:
            u_k += disturbance[k]

        control[k] = u_k

        pid.update_state(u_k)

        if k < n_steps - 1 and n > 0:
            x = A @ x + B.flatten() * u_k

    return time, output, control, setpoint_vec


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='PID Controller Demo')
    parser.add_argument('--save', type=str, help='Save plot to file')
    parser.add_argument('--show', action='store_true', help='Display plot interactively')
    args = parser.parse_args()

    print("=" * 70)
    print("PID CONTROLLER DEMONSTRATION")
    print("=" * 70)

    G_cont = tf_to_ss([1], [1, 2, 1])
    A_c, B_c, C_c, D_c = G_cont
    n = A_c.shape[0]

    if n > 0:
        A_d, B_d, C_d, D_d, info = ab04md('C',
            np.asfortranarray(A_c.copy()),
            np.asfortranarray(B_c.copy()),
            np.asfortranarray(C_c.copy()),
            np.asfortranarray(D_c.copy()),
            alpha=1.0, beta=2.0/0.01)
        plant = (A_d, B_d, C_d, D_d)
    else:
        plant = G_cont

    print(f"\nPlant: 1/(s^2 + 2s + 1)")
    print(f"Discrete plant (h=0.01s, Tustin)")

    pid = PIDController(K=3.0, Ti=1.0, Td=0.2, h=0.01, N=10, b=0.8,
                        ulow=-10.0, uhigh=10.0)  # Wide limits to show control action
    print(f"\nController: {pid}")

    print("\nSimulating step response...")
    t, y, u, sp = simulate_pid_control(pid, plant, setpoint=1.0, duration=10.0)

    settling_idx = np.where(np.abs(y - 1.0) < 0.02)[0]
    if len(settling_idx) > 0:
        settling_time = t[settling_idx[0]]
        print(f"\nPerformance Metrics:")
        print(f"  Settling time (2%): {settling_time:.2f} s")
        print(f"  Max overshoot: {(np.max(y) - 1.0) * 100:.1f}%")
        print(f"  Steady-state error: {abs(y[-1] - 1.0):.4f}")
        print(f"  Control range: [{u.min():.3f}, {u.max():.3f}]")
        print(f"  Control at steady-state: {u[-1]:.4f}")

    if args.save or args.show:
        try:
            import matplotlib
            if args.save and not args.show:
                matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            ax1.plot(t, sp, 'r--', label='Setpoint', linewidth=2)
            ax1.plot(t, y, 'b-', label='Output', linewidth=1.5)
            ax1.set_ylabel('Output')
            ax1.set_title('PID Control: Step Response')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(t, u, 'g-', label='Control Signal', linewidth=1.5)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Control Signal')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            if args.save:
                plt.savefig(args.save, dpi=150)
                print(f"\nPlot saved as '{args.save}'")
            if args.show:
                plt.show()
        except ImportError:
            print("\nmatplotlib not available - skipping plot")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
