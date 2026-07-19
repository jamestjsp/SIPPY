"""
Test cases for FIR identification algorithm implementation.
"""

import numpy as np
import pandas as pd
import pytest

from sippy import systems as control
from sippy.identification import IDData, SystemIdentificationConfig
from sippy.identification.algorithms.fir import FIRAlgorithm
from sippy.identification.base import IdentificationAlgorithm, StateSpaceModel


class TestFIRAlgorithm:
    """Test suite for FIR algorithm implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create simple test data
        np.random.seed(42)
        self.n_samples = 1000
        self.u = np.random.randn(self.n_samples)

        # Create a simple FIR system: y(k) = 0.8*u(k-1) + 0.3*u(k-2) + noise
        fir_coeffs = [0.8, 0.3]
        y_clean = np.zeros(self.n_samples)
        for k in range(2, self.n_samples):
            for i, coeff in enumerate(fir_coeffs):
                if k - i - 1 >= 0:
                    y_clean[k] += coeff * self.u[k - i - 1]
        self.y = y_clean + 0.1 * np.random.randn(self.n_samples)

        # Create DataFrame for IDData
        time_index = pd.date_range("2023-01-01", periods=self.n_samples, freq="1s")
        data_df = pd.DataFrame({"u": self.u, "y": self.y}, index=time_index)

        # Configure data
        self.data = IDData(data=data_df, inputs=["u"], outputs=["y"], tsample=1.0)

        self.config = SystemIdentificationConfig(method="FIR")
        # Set FIR-specific parameters
        self.config.nb = 3  # Number of FIR coefficients
        self.config.nk = 1  # Input delay

    def test_fir_algorithm_initialization(self):
        """Test FIR algorithm can be initialized."""
        algorithm = FIRAlgorithm()
        assert algorithm is not None
        assert isinstance(algorithm, IdentificationAlgorithm)

    def test_fir_algorithm_name(self):
        """Test algorithm returns correct name."""
        algorithm = FIRAlgorithm()
        assert algorithm.get_algorithm_name() == "FIR"

    def test_fir_basic_identification(self):
        """Test basic FIR identification functionality."""
        algorithm = FIRAlgorithm()

        result = algorithm.identify(
            iddata=self.data, nb=self.config.nb, nk=self.config.nk
        )

        assert isinstance(result, StateSpaceModel)
        assert isinstance(result.G, control.StateSpace)
        assert isinstance(result.G_tf, control.TransferFunction)
        assert isinstance(result.H_tf, control.TransferFunction)
        assert result.G_tf.dt == pytest.approx(self.data.sample_time)

    def test_fir_with_different_orders(self):
        """Test FIR with different number of coefficients."""
        algorithm = FIRAlgorithm()

        # Test different coefficient counts
        for nb in [2, 3, 5, 10]:
            config = SystemIdentificationConfig(method="FIR")
            config.nb = nb
            config.nk = 1

            result = algorithm.identify(iddata=self.data, nb=nb, nk=1)
            assert isinstance(result.G_tf, control.TransferFunction)

    def test_fir_mimo_system(self):
        """Test FIR with MIMO system."""
        # Create 2-input, 2-output test data
        np.random.seed(42)
        u = np.random.randn(2, self.n_samples)
        y = np.random.randn(2, self.n_samples)

        time_index = pd.date_range("2023-01-01", periods=self.n_samples, freq="1s")
        data_df = pd.DataFrame(
            {"u1": u[0, :], "u2": u[1, :], "y1": y[0, :], "y2": y[1, :]},
            index=time_index,
        )

        data = IDData(
            data=data_df, inputs=["u1", "u2"], outputs=["y1", "y2"], tsample=1.0
        )
        config = SystemIdentificationConfig(method="FIR")
        config.nb = 5
        config.nk = 1

        algorithm = FIRAlgorithm()

        result = algorithm.identify(iddata=data, nb=5, nk=1)
        assert isinstance(result.G, control.StateSpace)
        assert isinstance(result.G_tf, control.TransferFunction)
        assert result.G.shape == (2, 2)
        assert result.G_tf.shape == (2, 2)

    def test_fir_transfer_function_preserves_delay_and_coefficients(self):
        result = FIRAlgorithm().identify(iddata=self.data, nb=2, nk=1)

        numerator = result.G_tf.num[0][0]
        denominator = result.G_tf.den[0][0]
        assert numerator == pytest.approx([0.8, 0.3], abs=0.02)
        assert denominator == pytest.approx([1.0, 0.0, 0.0])

    def test_fir_invalid_parameters(self):
        """Test FIR algorithm with invalid parameters."""
        algorithm = FIRAlgorithm()

        # Test with invalid coefficient count
        invalid_config = SystemIdentificationConfig(method="FIR")
        invalid_config.nb = 0  # Invalid nb

        with pytest.raises(
            ValueError, match="Number of FIR coefficients must be positive"
        ):
            # Use new signature
            algorithm.identify(iddata=self.data, nb=0, nk=1)

    def test_fir_data_validation(self):
        """Test FIR algorithm validates input data."""
        algorithm = FIRAlgorithm()

        # Test with MIMO data
        time_index = pd.date_range("2023-01-01", periods=self.n_samples, freq="1s")
        data_df = pd.DataFrame(
            {
                "u1": np.random.randn(self.n_samples),
                "u2": np.random.randn(self.n_samples),
                "y1": np.random.randn(self.n_samples),
                "y2": np.random.randn(self.n_samples),
                "y3": np.random.randn(self.n_samples),  # Extra output
            },
            index=time_index,
        )

        data = IDData(
            data=data_df,
            inputs=["u1", "u2"],
            outputs=["y1", "y2", "y3"],  # Different number of outputs
            tsample=1.0,
        )

        result = algorithm.identify(iddata=data, nb=5, nk=1)
        assert result.G.shape == (3, 2)
        assert result.G_tf.shape == (3, 2)
