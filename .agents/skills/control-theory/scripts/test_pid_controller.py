#!/usr/bin/env python3
"""
Test Suite for PID Controller Implementation

Tests the digital PID controller with anti-windup and bumpless transfer
from pid_controller.py.

Tests cover:
- Initialization and parameter validation
- calculate_output() method functionality
- update_state() with anti-windup
- set_parameters() for bumpless transfer
- Simulation integration
"""

import numpy as np
from pid_controller import PIDController, simulate_pid_control, tf_to_ss
from ctrlsys import ab04md


def test_pid_initialization():
    """Test PID controller initialization with default and custom parameters."""
    print("\nTest 1: PID Initialization")
    print("-" * 60)

    # Test default parameters
    pid_default = PIDController()
    assert pid_default.params['K'] == 4.4, "Default K should be 4.4"
    assert pid_default.params['Ti'] == 0.4, "Default Ti should be 0.4"
    assert pid_default.params['Td'] == 0.2, "Default Td should be 0.2"
    assert pid_default.params['h'] == 0.03, "Default h should be 0.03"
    print("  [PASS] Default parameters initialized correctly")

    # Test custom parameters
    pid_custom = PIDController(K=2.0, Ti=1.0, Td=0.5, h=0.01)
    assert pid_custom.params['K'] == 2.0, "Custom K should be 2.0"
    assert pid_custom.params['Ti'] == 1.0, "Custom Ti should be 1.0"
    assert pid_custom.params['Td'] == 0.5, "Custom Td should be 0.5"
    assert pid_custom.params['h'] == 0.01, "Custom h should be 0.01"
    print("  [PASS] Custom parameters initialized correctly")

    # Test coefficient computation
    assert 'bi' in pid_custom.params, "Integral coefficient should be computed"
    assert 'ar' in pid_custom.params, "Anti-windup coefficient should be computed"
    assert 'ad' in pid_custom.params, "Derivative filter coefficient should be computed"
    assert 'bd' in pid_custom.params, "Derivative gain coefficient should be computed"
    print("  [PASS] Discretization coefficients computed")

    # Test initial states
    assert pid_custom.states['I'] == 0.0, "Initial integral state should be 0"
    assert pid_custom.states['D'] == 0.0, "Initial derivative state should be 0"
    assert pid_custom.states['yold'] == 0.0, "Initial yold should be 0"
    print("  [PASS] Initial states are zero")

    # Test initial signals
    assert pid_custom.signals['uc'] == 0.0, "Initial setpoint should be 0"
    assert pid_custom.signals['y'] == 0.0, "Initial measurement should be 0"
    assert pid_custom.signals['v'] == 0.0, "Initial v should be 0"
    assert pid_custom.signals['u'] == 0.0, "Initial u should be 0"
    print("  [PASS] Initial signals are zero")

    print("[PASS] Test 1 passed: PID Initialization")
    return True


def test_calculate_output_basic():
    """Test basic calculate_output() functionality."""
    print("\nTest 2: Calculate Output - Basic Functionality")
    print("-" * 60)

    pid = PIDController(K=1.0, Ti=1.0, Td=0.0, h=0.1, b=1.0, ulow=-10.0, uhigh=10.0)

    # Test step response with pure proportional
    setpoint = 1.0
    measurement = 0.0
    u = pid.calculate_output(setpoint, measurement)

    # With K=1, b=1, error=1, should get P=1, I=0, D=0 -> u=1
    assert abs(u - 1.0) < 1e-6, f"Expected u=1.0, got {u}"
    print(f"  [PASS] Proportional action: u={u:.6f} (expected 1.0)")

    # Update state and calculate again
    pid.update_state(u)
    u2 = pid.calculate_output(setpoint, measurement)

    # Now I should have integrated: bi*(uc-y) = 1.0*0.1/1.0 * 1.0 = 0.1
    # u = P + I = 1.0 + 0.1 = 1.1
    expected_u2 = 1.0 + 0.1
    assert abs(u2 - expected_u2) < 1e-6, f"Expected u={expected_u2}, got {u2}"
    print(f"  [PASS] P+I action: u={u2:.6f} (expected {expected_u2:.6f})")

    # Test with measurement = setpoint (zero error)
    pid.reset()
    u3 = pid.calculate_output(1.0, 1.0)
    assert abs(u3) < 1e-6, f"Expected u=0 with zero error, got {u3}"
    print(f"  [PASS] Zero error: u={u3:.6f} (expected 0.0)")

    print("[PASS] Test 2 passed: Calculate Output - Basic")
    return True


def test_calculate_output_saturation():
    """Test output limiting/saturation."""
    print("\nTest 3: Calculate Output - Saturation")
    print("-" * 60)

    pid = PIDController(K=10.0, Ti=1.0, Td=0.0, h=0.1, ulow=-1.0, uhigh=1.0)

    # Large error should saturate
    u = pid.calculate_output(10.0, 0.0)
    assert u == 1.0, f"Expected saturation at 1.0, got {u}"
    print(f"  [PASS] Upper saturation: u={u} (limit=1.0)")

    # Check that v (before limiting) is different
    assert pid.signals['v'] > 1.0, "v should be > uhigh when saturated"
    print(f"  [PASS] Pre-limit signal v={pid.signals['v']:.2f} > 1.0")

    # Negative saturation
    pid.reset()
    u_neg = pid.calculate_output(0.0, 10.0)
    assert u_neg == -1.0, f"Expected saturation at -1.0, got {u_neg}"
    print(f"  [PASS] Lower saturation: u={u_neg} (limit=-1.0)")

    # Within limits
    pid.reset()
    u_ok = pid.calculate_output(0.05, 0.0)
    assert -1.0 <= u_ok <= 1.0, "u should be within limits"
    assert abs(u_ok - pid.signals['v']) < 1e-10, "u should equal v when not saturated"
    print(f"  [PASS] Within limits: u={u_ok:.3f}, v={pid.signals['v']:.3f}")

    print("[PASS] Test 3 passed: Output Saturation")
    return True


def test_update_state_antiwindup():
    """Test update_state() with anti-windup mechanism."""
    print("\nTest 4: Update State - Anti-windup")
    print("-" * 60)

    pid = PIDController(K=5.0, Ti=1.0, Td=0.0, h=0.1, Tt=1.0, ulow=-1.0, uhigh=1.0)

    # Create saturation condition
    u = pid.calculate_output(10.0, 0.0)  # Large error -> saturation
    assert u == 1.0, "Should saturate at 1.0"

    I_before = pid.states['I']
    v = pid.signals['v']

    # Update with actual output (saturated)
    pid.update_state(u)

    # Anti-windup term: ar*(u - v) where ar = h/Tt = 0.1/1.0 = 0.1
    # Since u < v (saturation), anti-windup reduces integral growth
    ar = pid.params['ar']
    antiwindup_term = ar * (u - v)
    assert antiwindup_term < 0, "Anti-windup term should be negative during saturation"
    print(f"  [PASS] Anti-windup active: term={antiwindup_term:.4f} < 0")

    # Verify integral state updated correctly
    # I_new = I_old + bi*(uc - y) + ar*(u - v)
    bi = pid.params['bi']
    expected_I = I_before + bi * (10.0 - 0.0) + antiwindup_term
    assert abs(pid.states['I'] - expected_I) < 1e-10, "Integral state mismatch"
    print(f"  [PASS] Integral updated: I={pid.states['I']:.4f}")

    # Test without saturation
    pid.reset()
    u2 = pid.calculate_output(0.1, 0.0)  # Small error, no saturation
    pid.update_state(u2)

    # With no saturation, u == v, so antiwindup term = 0
    assert abs(u2 - pid.signals['v']) < 1e-10, "No saturation: u should equal v"
    print(f"  [PASS] No saturation: u={u2:.4f}, v={pid.signals['v']:.4f}")

    # Verify yold updated
    assert pid.states['yold'] == 0.0, "yold should store previous measurement"
    print(f"  [PASS] Previous measurement stored: yold={pid.states['yold']}")

    print("[PASS] Test 4 passed: Anti-windup Mechanism")
    return True


def test_derivative_action():
    """Test derivative term calculation."""
    print("\nTest 5: Derivative Action")
    print("-" * 60)

    pid = PIDController(K=1.0, Ti=100.0, Td=1.0, h=0.1, N=10.0, b=1.0)

    # First call - no derivative yet (yold = 0)
    u1 = pid.calculate_output(0.0, 0.0)
    D1 = pid.states['D']
    assert abs(D1) < 1e-10, "Initial derivative should be zero"
    print(f"  [PASS] Initial D={D1:.6f}")

    pid.update_state(u1)

    # Second call - measurement changes
    u2 = pid.calculate_output(0.0, 0.5)
    D2 = pid.states['D']

    # D = ad*D_old - bd*(y - yold)
    # bd = K*N*ad, ad = Td/(Td + N*h) = 1.0/(1.0 + 10*0.1) = 0.5
    ad = pid.params['ad']
    bd = pid.params['bd']
    expected_D = ad * D1 - bd * (0.5 - 0.0)

    assert abs(D2 - expected_D) < 1e-10, f"D mismatch: {D2} vs {expected_D}"
    print(f"  [PASS] Derivative calculated: D={D2:.6f} (expected {expected_D:.6f})")
    print(f"         Coefficients: ad={ad:.4f}, bd={bd:.4f}")

    # Derivative should oppose rapid changes
    assert D2 < 0, "Derivative should be negative for increasing measurement"
    print(f"  [PASS] Derivative opposes change: D < 0")

    print("[PASS] Test 5 passed: Derivative Action")
    return True


def test_setpoint_weighting():
    """Test setpoint weighting parameter b."""
    print("\nTest 6: Setpoint Weighting")
    print("-" * 60)

    # With b=1 (full setpoint weighting)
    pid1 = PIDController(K=2.0, Ti=10.0, Td=0.0, h=0.1, b=1.0, ulow=-10.0, uhigh=10.0)
    u1 = pid1.calculate_output(1.0, 0.0)

    # P = K*(b*uc - y) = 2.0*(1.0*1.0 - 0.0) = 2.0
    expected_u1 = 2.0
    assert abs(u1 - expected_u1) < 1e-6, f"Expected u={expected_u1}, got {u1}"
    print(f"  [PASS] b=1.0: u={u1:.3f} (full proportional on setpoint)")

    # With b=0 (no setpoint weighting - only measurement)
    pid2 = PIDController(K=2.0, Ti=10.0, Td=0.0, h=0.1, b=0.0, ulow=-10.0, uhigh=10.0)
    u2 = pid2.calculate_output(1.0, 0.0)

    # P = K*(b*uc - y) = 2.0*(0.0*1.0 - 0.0) = 0.0
    expected_u2 = 0.0
    assert abs(u2 - expected_u2) < 1e-6, f"Expected u={expected_u2}, got {u2}"
    print(f"  [PASS] b=0.0: u={u2:.3f} (no proportional on setpoint)")

    # With b=0.5 (partial setpoint weighting)
    pid3 = PIDController(K=2.0, Ti=10.0, Td=0.0, h=0.1, b=0.5, ulow=-10.0, uhigh=10.0)
    u3 = pid3.calculate_output(1.0, 0.0)

    # P = K*(b*uc - y) = 2.0*(0.5*1.0 - 0.0) = 1.0
    expected_u3 = 1.0
    assert abs(u3 - expected_u3) < 1e-6, f"Expected u={expected_u3}, got {u3}"
    print(f"  [PASS] b=0.5: u={u3:.3f} (partial proportional)")

    print("[PASS] Test 6 passed: Setpoint Weighting")
    return True


def test_set_parameters_bumpless():
    """Test set_parameters() for bumpless transfer."""
    print("\nTest 7: Set Parameters - Bumpless Transfer")
    print("-" * 60)

    pid = PIDController(K=2.0, Ti=1.0, Td=0.1, h=0.1, b=1.0)

    # Establish steady state
    setpoint = 1.0
    measurement = 0.8

    u1 = pid.calculate_output(setpoint, measurement)
    pid.update_state(u1)

    # Store current output and P+I sum
    P_old = pid.params['K'] * (pid.params['b'] * setpoint - measurement)
    I_old = pid.states['I']
    PI_old = P_old + I_old

    print(f"  Before change: K={pid.params['K']:.2f}, P={P_old:.4f}, I={I_old:.4f}, P+I={PI_old:.4f}")

    # Change gain
    pid.set_parameters(K=4.0)

    # Calculate new P
    P_new = pid.params['K'] * (pid.params['b'] * setpoint - measurement)
    I_new = pid.states['I']
    PI_new = P_new + I_new

    print(f"  After change:  K={pid.params['K']:.2f}, P={P_new:.4f}, I={I_new:.4f}, P+I={PI_new:.4f}")

    # P+I should remain constant for bumpless transfer
    assert abs(PI_new - PI_old) < 1e-10, f"P+I should be constant: {PI_new} vs {PI_old}"
    print(f"  [PASS] Bumpless transfer: P+I preserved")

    # Verify coefficients recalculated
    u2 = pid.calculate_output(setpoint, measurement)
    print(f"  [PASS] Controller operational after parameter change: u={u2:.4f}")

    # Test changing b parameter
    pid2 = PIDController(K=2.0, Ti=1.0, Td=0.1, h=0.1, b=1.0)
    u_before = pid2.calculate_output(1.0, 0.5)
    pid2.update_state(u_before)

    PI_before = pid2.params['K'] * (pid2.params['b'] * 1.0 - 0.5) + pid2.states['I']

    pid2.set_parameters(b=0.5)

    PI_after = pid2.params['K'] * (pid2.params['b'] * 1.0 - 0.5) + pid2.states['I']

    assert abs(PI_after - PI_before) < 1e-10, "P+I should be preserved when changing b"
    print(f"  [PASS] Bumpless transfer with b change: P+I preserved")

    print("[PASS] Test 7 passed: Bumpless Transfer")
    return True


def test_reset():
    """Test controller reset functionality."""
    print("\nTest 8: Controller Reset")
    print("-" * 60)

    pid = PIDController(K=2.0, Ti=1.0, Td=0.5, h=0.1, ulow=-10.0, uhigh=10.0)

    # Run controller to build up states
    for _ in range(5):
        u = pid.calculate_output(1.0, 0.5)
        pid.update_state(u)

    # Verify states are non-zero
    assert abs(pid.states['I']) > 1e-6, "Integral should be non-zero"
    print(f"  Before reset: I={pid.states['I']:.4f}, D={pid.states['D']:.4f}")

    # Reset
    pid.reset()

    # Verify all states and signals are zero
    assert pid.states['I'] == 0.0, "I should be zero after reset"
    assert pid.states['D'] == 0.0, "D should be zero after reset"
    assert pid.states['yold'] == 0.0, "yold should be zero after reset"
    assert pid.signals['uc'] == 0.0, "uc should be zero after reset"
    assert pid.signals['y'] == 0.0, "y should be zero after reset"
    assert pid.signals['v'] == 0.0, "v should be zero after reset"
    assert pid.signals['u'] == 0.0, "u should be zero after reset"

    print(f"  After reset:  I={pid.states['I']:.4f}, D={pid.states['D']:.4f}")
    print(f"  [PASS] All states and signals reset to zero")

    # Verify controller still works - P = K*(b*uc - y) = 2.0*(1.0*1.0 - 0.0) = 2.0
    u_new = pid.calculate_output(1.0, 0.0)
    expected_u = 2.0  # K=2.0, b=1.0 (default), error=1.0
    assert abs(u_new - expected_u) < 1e-6, f"Controller should work normally after reset, expected {expected_u}, got {u_new}"
    print(f"  [PASS] Controller operational after reset: u={u_new:.4f}")

    print("[PASS] Test 8 passed: Controller Reset")
    return True


def test_simulation_integration():
    """Test PID controller with plant simulation."""
    print("\nTest 9: Simulation Integration")
    print("-" * 60)

    # Create a simple first-order plant: G(s) = 1/(s+1)
    A, B, C, D = tf_to_ss([1], [1, 1])

    # Discretize using Tustin
    dt = 0.01
    A_d, B_d, C_d, D_d, _ = ab04md('C', A.copy(), B.copy(), C.copy(), D.copy(),
                                    alpha=1.0, beta=2.0/dt)
    plant = (A_d, B_d, C_d, D_d)

    print(f"  Plant: G(s) = 1/(s+1)")
    print(f"  Discretized with h=0.01s (Tustin)")

    # Create PID controller
    pid = PIDController(K=2.0, Ti=0.5, Td=0.1, h=0.01, b=0.8)

    # Simulate
    duration = 5.0
    setpoint = 1.0
    t, y, u, sp = simulate_pid_control(pid, plant, setpoint, duration)

    # Verify simulation results
    assert len(t) > 0, "Time vector should not be empty"
    assert len(y) == len(t), "Output length should match time"
    assert len(u) == len(t), "Control length should match time"
    assert len(sp) == len(t), "Setpoint length should match time"

    print(f"  [PASS] Simulation completed: {len(t)} steps")
    print(f"  Duration: {t[-1]:.2f}s")

    # Check steady-state tracking
    steady_state_error = abs(y[-1] - setpoint)
    assert steady_state_error < 0.1, f"Steady-state error too large: {steady_state_error}"
    print(f"  [PASS] Steady-state tracking: error={steady_state_error:.4f}")

    # Check output is within bounds
    assert np.all(y >= -10) and np.all(y <= 10), "Output should be bounded"
    print(f"  [PASS] Output bounded: min={np.min(y):.2f}, max={np.max(y):.2f}")

    # Check control signal is within limits
    assert np.all(u >= pid.params['ulow']) and np.all(u <= pid.params['uhigh']), \
        "Control should respect limits"
    print(f"  [PASS] Control within limits: min={np.min(u):.2f}, max={np.max(u):.2f}")

    print("[PASS] Test 9 passed: Simulation Integration")
    return True


def test_numerical_stability():
    """Test numerical stability with extreme parameters."""
    print("\nTest 10: Numerical Stability")
    print("-" * 60)

    # Test with very small sampling time
    pid_fast = PIDController(K=1.0, Ti=1.0, Td=0.1, h=0.001)
    u = pid_fast.calculate_output(1.0, 0.0)
    assert np.isfinite(u), "Output should be finite with small h"
    print(f"  [PASS] Small sampling time (h=0.001): u={u:.6f}")

    # Test with large gains
    pid_highgain = PIDController(K=100.0, Ti=0.1, Td=1.0, h=0.01)
    u = pid_highgain.calculate_output(1.0, 0.99)
    assert np.isfinite(u), "Output should be finite with high gain"
    print(f"  [PASS] High gain (K=100): u={u:.6f}")

    # Test with zero integral time (pure PD)
    # Note: This would cause division by zero, so Ti should be > 0
    # Instead test with very large Ti (weak integral)
    pid_pd = PIDController(K=1.0, Ti=1000.0, Td=0.1, h=0.01)
    u = pid_pd.calculate_output(1.0, 0.0)
    assert np.isfinite(u), "Output should be finite with weak integral"
    print(f"  [PASS] Weak integral (Ti=1000): u={u:.6f}")

    # Test long simulation
    pid_long = PIDController(K=1.0, Ti=1.0, Td=0.1, h=0.01)
    for k in range(10000):
        u = pid_long.calculate_output(1.0, 0.5 + 0.1 * np.sin(0.01 * k))
        pid_long.update_state(u)
        assert np.isfinite(u), f"Output diverged at step {k}"

    print(f"  [PASS] Long simulation (10000 steps): stable")

    print("[PASS] Test 10 passed: Numerical Stability")
    return True


def test_string_representation():
    """Test __repr__ method."""
    print("\nTest 11: String Representation")
    print("-" * 60)

    pid = PIDController(K=2.5, Ti=0.8, Td=0.15, h=0.025)

    repr_str = repr(pid)
    assert 'PIDController' in repr_str, "Should contain class name"
    assert 'K=2.500' in repr_str, "Should contain K parameter"
    assert 'Ti=0.800' in repr_str, "Should contain Ti parameter"
    assert 'Td=0.150' in repr_str, "Should contain Td parameter"
    assert 'h=0.0250' in repr_str, "Should contain h parameter"

    print(f"  Representation: {repr_str}")
    print(f"  [PASS] String representation complete")

    print("[PASS] Test 11 passed: String Representation")
    return True


def run_all_tests():
    """Run all PID controller tests."""
    print("=" * 70)
    print("PID CONTROLLER TEST SUITE")
    print("=" * 70)

    tests = [
        ("PID Initialization", test_pid_initialization),
        ("Calculate Output - Basic", test_calculate_output_basic),
        ("Calculate Output - Saturation", test_calculate_output_saturation),
        ("Update State - Anti-windup", test_update_state_antiwindup),
        ("Derivative Action", test_derivative_action),
        ("Setpoint Weighting", test_setpoint_weighting),
        ("Set Parameters - Bumpless Transfer", test_set_parameters_bumpless),
        ("Controller Reset", test_reset),
        ("Simulation Integration", test_simulation_integration),
        ("Numerical Stability", test_numerical_stability),
        ("String Representation", test_string_representation),
    ]

    passed = 0
    failed = 0
    errors = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except AssertionError as e:
            failed += 1
            errors.append(f"{test_name}: {str(e)}")
            print(f"[FAIL] Test failed: {test_name}")
            print(f"       Error: {e}")
        except Exception as e:
            failed += 1
            errors.append(f"{test_name}: Unexpected error - {str(e)}")
            print(f"[ERROR] Test error: {test_name}")
            print(f"        Error: {e}")

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Total tests run: {len(tests)}")
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if errors:
        print("\nFailure Details:")
        for error in errors:
            print(f"  - {error}")

    print("=" * 70)

    if failed == 0:
        print("All tests PASSED!")
        return True
    else:
        print(f"{failed} test(s) FAILED!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
