"""
Compatibility shim for tf2ss module.

This module provides a wrapper around control.forced_response to maintain
backward compatibility with legacy SIPPY code that imports from tf2ss.

The original tf2ss module was likely part of an older version of python-control
or a standalone module that has since been deprecated. This shim allows the
master branch to work with modern python-control (0.10.x).

Author: Claude Code
Date: 2025-10-12
Purpose: Enable cross-branch validation tests
"""

import warnings
import numpy as np

try:
    import control
    CONTROL_AVAILABLE = True
except ImportError:
    CONTROL_AVAILABLE = False
    warnings.warn("python-control not available, tf2ss shim will not work")


def forced_response(sys, T, U=0, X0=0, transpose=False):
    """
    Compute the forced response of a system.

    This is a wrapper around control.forced_response that maintains API
    compatibility with the legacy tf2ss.forced_response function.

    Parameters
    ----------
    sys : TransferFunction or StateSpace
        LTI system to simulate
    T : array_like
        Time vector for simulation
    U : array_like, optional
        Input vector (default: 0)
    X0 : array_like, optional
        Initial state (default: 0)
    transpose : bool, optional
        If True, transpose input/output (default: False)

    Returns
    -------
    T : array
        Time vector
    yout : array
        System response

    Notes
    -----
    This function handles non-proper transfer functions (where numerator
    degree >= denominator degree) by using an algebraic approach for the
    direct feedthrough term.

    For a non-proper system like 1 - 1/H where H is proper, we decompose:
        1 - 1/H = (H - 1)/H

    This ensures the resulting transfer function is proper and can be
    simulated with control.forced_response.
    """
    if not CONTROL_AVAILABLE:
        raise ImportError("python-control package required for forced_response")

    # Convert inputs to arrays
    T = np.asarray(T)
    if np.isscalar(U):
        U = U * np.ones_like(T)
    else:
        U = np.asarray(U)

    # Handle array shape: ensure U is 1D or 2D (rows=inputs, cols=time)
    if U.ndim == 1:
        U_sim = U
    elif U.ndim == 2:
        # If U is (1, N), flatten it
        if U.shape[0] == 1:
            U_sim = U.flatten()
        # If U is (M, N) with M > 1, it's a MIMO input
        else:
            U_sim = U
    else:
        U_sim = U

    try:
        # Try standard forced_response
        T_out, yout = control.forced_response(
            sys, T, U_sim, X0=X0, transpose=transpose
        )
        return T_out, yout

    except ValueError as e:
        if "non-proper" in str(e):
            # Handle non-proper transfer function
            # This occurs for systems like: 1 - 1/H
            #
            # Strategy: If the system is non-proper, we decompose it into:
            #   sys = D + sys_proper
            # where D is the direct feedthrough (constant gain at high freq)
            # and sys_proper is a proper transfer function.
            #
            # For discrete-time: 1 - 1/H = (H - 1)/H
            # We can compute this by manipulating the numerator/denominator

            if hasattr(sys, 'num') and hasattr(sys, 'den'):
                # It's a transfer function
                num = np.atleast_2d(sys.num)
                den = np.atleast_2d(sys.den)

                # For SISO systems
                if num.shape[0] == 1:
                    num_coeffs = np.atleast_1d(np.squeeze(num))
                    den_coeffs = np.atleast_1d(np.squeeze(den))

                    # Extract direct feedthrough (high-frequency gain)
                    # For non-proper systems: numerator order >= denominator order
                    if len(num_coeffs) >= len(den_coeffs):
                        # For a system like: (a*z + b) / 1
                        # Direct feedthrough D = a*z (the highest order term)
                        # But for forced_response, we need to handle this differently

                        # Special case: constant denominator (den = [c])
                        if len(den_coeffs) == 1:
                            # System is purely polynomial: num(z) / c
                            # This is like: D + num_remaining(z) / c
                            # where D = num[0] / c (constant gain)
                            D = num_coeffs[0] / den_coeffs[0]

                            # Remaining numerator (without leading term)
                            if len(num_coeffs) > 1:
                                num_remaining = num_coeffs[1:]
                                # Create proper TF: num_remaining / z^(n-1)
                                # In z-domain: multiply denominator by z
                                den_proper = np.array([den_coeffs[0], 0.0])  # [c, 0] = c*z

                                # Create proper transfer function
                                if hasattr(sys, 'dt'):
                                    sys_proper = control.tf(num_remaining, den_proper, dt=sys.dt)
                                else:
                                    sys_proper = control.tf(num_remaining, den_proper)

                                # Simulate proper part
                                T_out, yout_proper = control.forced_response(
                                    sys_proper, T, U_sim, X0=X0, transpose=transpose
                                )

                                # Add direct feedthrough contribution
                                yout = yout_proper + D * U_sim
                            else:
                                # Pure constant gain
                                yout = D * U_sim
                                T_out = T

                            return T_out, yout
                        else:
                            # General non-proper case: pad and compute
                            max_len = max(len(num_coeffs), len(den_coeffs))
                            num_pad = np.pad(num_coeffs, (max_len - len(num_coeffs), 0), 'constant')
                            den_pad = np.pad(den_coeffs, (max_len - len(den_coeffs), 0), 'constant')

                            D = num_pad[0] / den_pad[0]
                            num_proper = num_pad - D * den_pad

                            # Create proper transfer function
                            if hasattr(sys, 'dt'):
                                sys_proper = control.tf(num_proper, den_pad, dt=sys.dt)
                            else:
                                sys_proper = control.tf(num_proper, den_pad)

                            # Simulate proper part
                            T_out, yout_proper = control.forced_response(
                                sys_proper, T, U_sim, X0=X0, transpose=transpose
                            )

                            # Add direct feedthrough contribution
                            yout = yout_proper + D * U_sim

                            return T_out, yout

            # If we can't handle it, re-raise
            raise
        else:
            # Re-raise other errors
            raise


# Maintain API compatibility - export forced_response as the main function
__all__ = ['forced_response']
