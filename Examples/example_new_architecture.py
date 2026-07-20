#!/usr/bin/env python3
"""
Demonstration of the new refactored system identification architecture.

This script shows how the new class-based architecture with factory pattern
works compared to the original function-based approach.
"""

import numpy as np

import sippy
from sippy.identification import (
    SystemIdentification,
    SystemIdentificationConfig,
    system_identification,
)
from sippy.identification.factory import AlgorithmFactory


def generate_open_and_closed_loop_data():
    """Generate records from one plant under open- and closed-loop operation."""
    rng = np.random.default_rng(20260720)
    sample_count = 800
    plant_a = 0.72
    plant_b = 0.35

    open_input = rng.normal(size=(1, sample_count))
    open_output = np.zeros((1, sample_count))
    for sample in range(sample_count - 1):
        open_output[0, sample + 1] = (
            plant_a * open_output[0, sample]
            + plant_b * open_input[0, sample]
            + 0.02 * rng.normal()
        )

    reference = rng.normal(size=(1, sample_count))
    closed_input = np.zeros((1, sample_count))
    closed_output = np.zeros((1, sample_count))
    state = 0.0
    for sample in range(sample_count):
        closed_output[0, sample] = state + 0.03 * rng.normal()
        closed_input[0, sample] = reference[0, sample] - 0.9 * closed_output[0, sample]
        state = plant_a * state + plant_b * closed_input[0, sample]

    return (open_output, open_input), (closed_output, closed_input)


def demo_loop_agnostic_subspace():
    """Identify one plant without selecting an open- or closed-loop algorithm."""
    print("=== LOOP-AGNOSTIC SUBSPACE IDENTIFICATION ===\n")
    open_loop, closed_loop = generate_open_and_closed_loop_data()
    identification_options = {}

    open_model = sippy.identify(*open_loop, **identification_options)
    closed_model = sippy.identify(*closed_loop, **identification_options)

    print(
        "Open loop:  "
        f"{open_model.method}, route={open_model.identification_info['estimator_route']}, "
        f"order={open_model.n}"
    )
    print(
        "Closed loop: "
        f"{closed_model.method}, "
        f"route={closed_model.identification_info['estimator_route']}, "
        f"order={closed_model.n}"
    )
    return open_model, closed_model


def generate_sample_data():
    """Generate sample data for demonstration."""
    np.random.seed(42)
    n_points = 100

    # Simple 2-input, 1-output system
    u = np.random.randn(2, n_points)
    y = np.zeros((1, n_points))

    # Simulate a simple linear system with some dynamics
    for i in range(1, n_points):
        y[0, i] = (
            0.7 * y[0, i - 1]
            + 0.3 * u[0, i - 1]
            + 0.2 * u[1, i - 1]
            + 0.05 * np.random.randn()
        )

    return y, u


def demo_new_class_based_approach():
    """Demonstrate the new class-based approach."""
    print("=== NEW CLASS-BASED ARCHITECTURE DEMO ===\n")

    # Generate sample data
    y, u = generate_sample_data()
    print(f"Generated sample data: {y.shape} outputs, {u.shape} inputs")

    # Create a configuration
    print("\n1. Creating custom configuration...")
    config = SystemIdentificationConfig(
        method="N4SID", ss_f=10, ss_fixed_order=1, ss_threshold=0.1
    )
    print(f"Method: {config.method}")
    print(f"Horizon (ss_f): {config.ss_f}")
    print(f"Fixed order: {config.ss_fixed_order}")

    # Create identification system
    print("\n2. Creating SystemIdentification instance...")
    identifier = SystemIdentification(config)

    # Perform identification
    print("\n3. Performing system identification...")
    model = identifier.identify(y, u)

    print(f"✓ Identified model with {model.n} states")
    print(f"✓ A matrix shape: {model.A.shape}")
    print(f"✓ B matrix shape: {model.B.shape}")
    print(f"✓ C matrix shape: {model.C.shape}")
    print(f"✓ D matrix shape: {model.D.shape}")
    print(f"✓ System stable: {model.is_stable()}")

    # Test algorithm factory
    print("\n4. Demonstrating factory pattern...")
    print(f"Available algorithms: {AlgorithmFactory.list_algorithms()}")

    # Create algorithm directly
    from sippy.identification.factory import create_algorithm

    n4sid_algo = create_algorithm("N4SID")
    print(f"Created algorithm: {n4sid_algo.name}")

    return model


def demo_backward_compatibility():
    """Demonstrate backward compatibility with original API."""
    print("\n\n=== BACKWARD COMPATIBILITY DEMO ===\n")

    # Generate sample data
    y, u = generate_sample_data()

    print("Using original function signature...")
    model = system_identification(
        y=y, u=u, id_method="N4SID", tsample=1.0, SS_fixed_order=1, SS_f=10
    )

    print("✓ Model identified using old API")
    print(f"✓ Model states: {model.n}")
    print(f"✓ Model stable: {model.is_stable()}")

    return model


def demo_different_algorithms():
    """Demonstrate using different algorithms."""
    print("\n\n=== DIFFERENT ALGORITHMS DEMO ===\n")

    # Generate sample data
    y, u = generate_sample_data()

    algorithms = ["N4SID"]  # Start with one for demo
    models = {}

    for method in algorithms:
        print(f"\nTesting {method} algorithm...")

        config = SystemIdentificationConfig(
            method=method, ss_f=10, ss_fixed_order=1, ss_threshold=0.1
        )

        identifier = SystemIdentification(config)
        model = identifier.identify(y, u)

        models[method] = model
        print(f"✓ {method} -> {model.n} states, stable: {model.is_stable()}")

    return models


def demo_flexible_configuration():
    """Demonstrate flexible configuration options."""
    print("\n\n=== FLEXIBLE CONFIGURATION DEMO ===\n")

    # Generate sample data
    y, u = generate_sample_data()

    # Test different configurations
    configs = [
        ("Default config", SystemIdentificationConfig()),
        ("Custom horizon", SystemIdentificationConfig(ss_f=15, ss_fixed_order=1)),
        (
            "Different method",
            SystemIdentificationConfig(method="N4SID", ss_f=8, ss_fixed_order=1),
        ),
    ]

    for name, config in configs:
        print(f"\nTesting: {name}")
        try:
            identifier = SystemIdentification(config)
            model = identifier.identify(y, u)
            print(f"✓ Success: {model.n} states")
        except Exception as e:
            print(f"✗ Error: {e}")


if __name__ == "__main__":
    print("🚀 SIPPY System Identification - New Architecture Demo")
    print("=" * 60)

    try:
        # Run demonstrations
        demo_loop_agnostic_subspace()
        model1 = demo_new_class_based_approach()
        model2 = demo_backward_compatibility()
        models = demo_different_algorithms()
        demo_flexible_configuration()

        print("\n" + "=" * 60)
        print("✅ All demos completed successfully!")
        print("\n🎯 Key Benefits of New Architecture:")
        print("  • Object-oriented design with clear separation of concerns")
        print("  • Factory pattern for extensible algorithm support")
        print("  • Type safety and better error handling")
        print("  • Backward compatibility with existing code")
        print("  • Configurable and reusable components")
        print("  • Enhanced testing and maintainability")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback

        traceback.print_exc()
