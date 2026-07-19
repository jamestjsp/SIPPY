"""
Tests for base classes.
"""

import numpy as np
import pytest

from sippy import systems as control
from sippy.identification.base import (
    IdentificationAlgorithm,
    StateSpaceModel,
    SystemIdentificationConfig,
    realize_transfer_function,
)


class TestIdentificationAlgorithm:
    """Test the abstract base class."""

    def test_cannot_instantiate_abstract(self):
        """Test that IdentificationAlgorithm cannot be instantiated directly."""
        with pytest.raises(TypeError):
            IdentificationAlgorithm()

    def test_concrete_implementation(self):
        """Test that a concrete implementation works."""

        class TestAlgorithm(IdentificationAlgorithm):
            def identify(self, y, u, **kwargs):
                return StateSpaceModel(
                    A=np.eye(2),
                    B=np.zeros((2, 1)),
                    C=np.zeros((1, 2)),
                    D=np.zeros((1, 1)),
                    K=np.zeros((2, 1)),
                    Q=np.eye(2),
                    R=1.0,
                    S=np.zeros((2, 1)),
                    ts=1.0,
                    Vn=1.0,
                )

            def validate_parameters(self, **kwargs):
                return True

        algo = TestAlgorithm()
        assert algo.name == "TestAlgorithm"

        # Test with dummy data
        y = np.random.randn(2, 100)
        u = np.random.randn(1, 100)
        model = algo.identify(y, u)
        assert isinstance(model, StateSpaceModel)
        assert model.A.shape == (2, 2)


class TestStateSpaceModel:
    """Test the StateSpaceModel class."""

    def test_model_creation(self):
        """Test creating a state space model."""
        A = np.array([[0.9, 0.1], [-0.1, 0.8]])
        B = np.array([[1.0], [0.5]])
        C = np.array([[1.0, 0.0]])
        D = np.array([[0.0]])
        K = np.array([[0.1], [0.05]])
        Q = np.eye(2)
        R = 0.1
        S = np.zeros((2, 1))
        ts = 1.0
        Vn = 0.5

        model = StateSpaceModel(A, B, C, D, K, Q, R, S, ts, Vn)

        assert model.n == 2
        assert np.array_equal(model.A, A)
        assert np.array_equal(model.B, B)
        assert np.array_equal(model.C, C)
        assert np.array_equal(model.D, D)
        assert model.ts == ts
        assert model.Vn == Vn
        assert isinstance(model.G, control.StateSpace)
        assert model.G.dt == pytest.approx(ts)
        np.testing.assert_array_equal(model.G.A, A)
        np.testing.assert_array_equal(model.G.B, B)
        np.testing.assert_array_equal(model.G.C, C)
        np.testing.assert_array_equal(model.G.D, D)

    def test_model_creation_preserves_mimo_dimensions(self):
        model = StateSpaceModel(
            A=np.diag([0.8, 0.6]),
            B=np.eye(2),
            C=np.eye(2),
            D=np.zeros((2, 2)),
            K=np.zeros((2, 2)),
            Q=np.eye(2),
            R=np.eye(2),
            S=np.zeros((2, 2)),
            ts=0.25,
            Vn=np.eye(2),
        )

        assert isinstance(model.G, control.StateSpace)
        assert model.G.shape == (2, 2)
        assert model.G.nstates == 2
        assert model.G.dt == pytest.approx(0.25)

    def test_model_without_inputs_has_no_control_system(self):
        model = StateSpaceModel(
            A=np.eye(2),
            B=np.empty((2, 0)),
            C=np.ones((1, 2)),
            D=np.empty((1, 0)),
            K=np.zeros((2, 1)),
            Q=np.eye(2),
            R=np.eye(1),
            S=np.zeros((2, 1)),
            ts=0.5,
            Vn=0.0,
        )

        assert model.G is None

    def test_stability_check(self):
        """Test stability checking."""
        # Stable system
        A_stable = np.array([[0.9, 0.1], [-0.1, 0.8]])
        model_stable = StateSpaceModel(
            A_stable,
            np.zeros((2, 1)),
            np.zeros((1, 2)),
            np.zeros((1, 1)),
            np.zeros((2, 1)),
            np.eye(2),
            0.1,
            np.zeros((2, 1)),
            1.0,
            0.5,
        )
        assert model_stable.is_stable()

        # Unstable system
        A_unstable = np.array([[1.1, 0.0], [0.0, 1.2]])
        model_unstable = StateSpaceModel(
            A_unstable,
            np.zeros((2, 1)),
            np.zeros((1, 2)),
            np.zeros((1, 1)),
            np.zeros((2, 1)),
            np.eye(2),
            0.1,
            np.zeros((2, 1)),
            1.0,
            0.5,
        )
        assert not model_unstable.is_stable()

    def test_natural_frequencies(self):
        """Test natural frequency calculation."""
        A = np.array([[0.9, -0.5], [0.5, 0.9]])  # Complex conjugate pair
        model = StateSpaceModel(
            A,
            np.zeros((2, 1)),
            np.zeros((1, 2)),
            np.zeros((1, 1)),
            np.zeros((2, 1)),
            np.eye(2),
            0.1,
            np.zeros((2, 1)),
            1.0,
            0.5,
        )
        freqs = model.get_natural_frequencies()
        assert len(freqs) == 2
        assert np.all(freqs >= 0)


def test_realize_transfer_function_uses_control_state_space_conventions():
    transfer_function = control.tf([0.2, 0.1], [1.0, -0.7, 0.12], dt=0.2)

    A, B, C, D = realize_transfer_function(transfer_function)
    realized = control.ss(A, B, C, D, dt=transfer_function.dt)

    assert A.ndim == B.ndim == C.ndim == D.ndim == 2
    assert realized.nstates == 2
    np.testing.assert_allclose(
        control.frequency_response(realized, [0.1, 1.0]).frdata,
        control.frequency_response(transfer_function, [0.1, 1.0]).frdata,
        rtol=1e-12,
        atol=1e-12,
    )


def test_realize_mimo_transfer_function_with_ctrlsys():
    transfer_function = control.tf(
        [[[0.2], [0.1]], [[0.3], [0.4]]],
        [
            [[1.0, -0.5], [1.0, -0.4]],
            [[1.0, -0.3], [1.0, -0.2]],
        ],
        dt=0.1,
    )

    A, B, C, D = realize_transfer_function(transfer_function)
    realized = control.ss(A, B, C, D, dt=transfer_function.dt)

    assert realized.shape == (2, 2)
    assert realized.dt == pytest.approx(0.1)
    np.testing.assert_allclose(
        control.frequency_response(realized, [0.2, 0.8]).frdata,
        control.frequency_response(transfer_function, [0.2, 0.8]).frdata,
        rtol=1e-10,
        atol=1e-10,
    )


class TestSystemIdentificationConfig:
    """Test the configuration class."""

    def test_default_config(self):
        """Test default configuration."""
        config = SystemIdentificationConfig()
        assert config.method == "N4SID"
        assert config.centering == "None"
        assert config.ss_f == 20
        assert config.ss_threshold == 0.1
        assert not config.ss_d_required
        assert not config.ss_a_stability

    def test_custom_config(self):
        """Test custom configuration."""
        config = SystemIdentificationConfig(
            method="CVA",
            centering="MeanVal",
            ss_f=15,
            ss_threshold=0.05,
            ss_d_required=True,
            ss_a_stability=True,
        )
        assert config.method == "CVA"
        assert config.centering == "MeanVal"
        assert config.ss_f == 15
        assert config.ss_threshold == 0.05
        assert config.ss_d_required
        assert config.ss_a_stability
