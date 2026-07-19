import sys
from types import ModuleType

import numpy as np
from scipy.linalg import eigvals, solve_discrete_are

from sippy import systems


def _dare(A, B, Q, R, S=None, E=None):
    cross_weight = np.zeros((A.shape[0], B.shape[1])) if S is None else S
    solution = solve_discrete_are(A, B, Q, R, e=E, s=cross_weight)
    gain = np.linalg.solve(B.T @ solution @ B + R, B.T @ solution @ A + cross_weight.T)
    closed_loop = eigvals(A - B @ gain, E)
    return solution, closed_loop, gain


def _matlab_impulse(system, time):
    response = systems.impulse_response(system, T=time)
    return response.outputs, response.time


def install_reference_control_compat() -> None:
    if "control.matlab" in sys.modules:
        return
    control_module = ModuleType("control")
    matlab_module = ModuleType("control.matlab")
    for name in ("lsim", "poles", "ss", "tf", "tfdata"):
        value = getattr(systems, name)
        setattr(control_module, name, value)
        setattr(matlab_module, name, value)
    matlab_module.dare = _dare
    matlab_module.impulse = _matlab_impulse
    matlab_module.matlab = matlab_module
    control_module.matlab = matlab_module
    sys.modules["control"] = control_module
    sys.modules["control.matlab"] = matlab_module
