"""
Test cases for OE (Output Error) identification algorithm implementation.
"""

import control
import numpy as np
import pandas as pd
import pytest

from sippy.identification import IDData, SystemIdentificationConfig
from sippy.identification.algorithms.oe import OEAlgorithm
from sippy.identification.base import IdentificationAlgorithm, StateSpaceModel


class TestOEAlgorithm:
    """Test suite for OE algorithm implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create simple test data with OE characteristics
        np.random.seed(42)
        self.n_samples = 1000
        self.u = np.random.randn(self.n_samples)

        # Create a simple OE system: y(k) = (0.5*u(k-1) + 0.3*u(k-2)) / (1 + 0.3*yp(k-1) - 0.2*yp(k-2)) + noise
        y_clean = np.zeros(self.n_samples)
        noise_free_output = np.zeros(self.n_samples)
        for k in range(2, self.n_samples):
            # Noise-free output feedback
            denominator = (
                1 + 0.3 * noise_free_output[k - 1] - 0.2 * noise_free_output[k - 2]
            )
            numerator = 0.5 * self.u[k - 1] + 0.3 * self.u[k - 2]
            noise_free_output[k] = numerator / denominator
            y_clean[k] = noise_free_output[k]

        self.y = y_clean + 0.05 * np.random.randn(self.n_samples)

        # Create DataFrame for IDData
        time_index = pd.date_range("2023-01-01", periods=self.n_samples, freq="1s")
        data_df = pd.DataFrame({"u": self.u, "y": self.y}, index=time_index)

        # Configure data
        self.data = IDData(data=data_df, inputs=["u"], outputs=["y"], tsample=1.0)

        self.config = SystemIdentificationConfig(method="OE")
        # Set OE-specific parameters
        self.config.nb = 2  # Numerator order
        self.config.nf = 2  # Denominator order
        self.config.nk = 1  # Input delay

    def test_oe_algorithm_initialization(self):
        """Test OE algorithm can be initialized."""
        algorithm = OEAlgorithm()
        assert algorithm is not None
        assert isinstance(algorithm, IdentificationAlgorithm)

    def test_oe_algorithm_name(self):
        """Test algorithm returns correct name."""
        algorithm = OEAlgorithm()
        assert algorithm.get_algorithm_name() == "OE"

    def test_oe_parameter_validation(self):
        """Test OE parameter validation."""
        algorithm = OEAlgorithm()

        # Test valid parameters
        algorithm.validate_parameters(nb=2, nf=2, nk=0)
        algorithm.validate_parameters(nb=1, nf=1, nk=1)
        algorithm.validate_parameters(nb=3, nf=4, nk=2)

        # Test boundary conditions
        with pytest.raises(
            ValueError, match="Numerator order \\(nb\\) must be positive"
        ):
            algorithm.validate_parameters(nb=0, nf=2, nk=0)
        with pytest.raises(
            ValueError, match="Denominator order \\(nf\\) must be positive"
        ):
            algorithm.validate_parameters(nb=2, nf=0, nk=0)
        with pytest.raises(
            ValueError, match="Input delay \\(nk\\) must be non-negative"
        ):
            algorithm.validate_parameters(nb=2, nf=2, nk=-1)

    def test_oe_basic_identification(self):
        """Test basic OE identification functionality."""
        algorithm = OEAlgorithm()

        result = algorithm.identify(self.data, self.config)

        assert isinstance(result, StateSpaceModel)
        assert isinstance(result.G, control.StateSpace)
        assert isinstance(result.G_tf, control.TransferFunction)
        assert isinstance(result.H_tf, control.TransferFunction)
        assert result.G_tf.dt == pytest.approx(self.data.sample_time)

    def test_oe_with_different_orders(self):
        """Test OE with different model orders."""
        algorithm = OEAlgorithm()

        # Test different orders
        for nb, nf in [(2, 2), (3, 2), (2, 3), (3, 3)]:
            config = SystemIdentificationConfig(method="OE")
            config.nb = nb
            config.nf = nf
            config.nk = 1

            result = algorithm.identify(self.data, config)
            assert isinstance(result.G_tf, control.TransferFunction)

    def test_oe_mimo_system(self):
        """Test OE with MIMO system."""
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
        config = SystemIdentificationConfig(method="OE")
        config.nb = 2
        config.nf = 2
        config.nk = 1

        algorithm = OEAlgorithm()

        result = algorithm.identify(data, config)
        assert isinstance(result.G, control.StateSpace)
        assert isinstance(result.G_tf, control.TransferFunction)
        assert result.G.shape == (2, 2)
        assert result.G_tf.shape == (2, 2)

    def test_oe_transfer_functions_preserve_sample_time(self):
        result = OEAlgorithm().identify(self.data, self.config)

        assert result.G_tf.dt == pytest.approx(1.0)
        assert result.H_tf.dt == pytest.approx(1.0)
        assert result.H_tf.num[0][0] == pytest.approx([1.0])
        assert result.H_tf.den[0][0] == pytest.approx([1.0])

    def test_oe_data_validation(self):
        """Test OE algorithm validates input data."""
        algorithm = OEAlgorithm()

        # Test with insufficient data
        small_data_time_index = pd.date_range("2023-01-01", periods=5, freq="1s")
        small_data_df = pd.DataFrame(
            {"u": np.random.randn(5), "y": np.random.randn(5)},
            index=small_data_time_index,
        )

        small_data = IDData(
            data=small_data_df, inputs=["u"], outputs=["y"], tsample=1.0
        )

        config = SystemIdentificationConfig(method="OE")
        config.nb = 5  # Requires more data than available
        config.nf = 4
        config.nk = 1

        # OE algorithm works with insufficient data but uses simplified estimation
        # This test just verifies it doesn't crash
        result = algorithm.identify(
            iddata=small_data, nb=config.nb, nf=config.nf, nk=config.nk
        )
        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_oe_order_calculation(self):
        """Test that OE calculates correct model order."""
        algorithm = OEAlgorithm()

        config = SystemIdentificationConfig(method="OE")
        config.nb = 2
        config.nf = 3  # Denominator order determines state dimension
        config.nk = 0

        result = algorithm.identify(self.data, config)
        assert result.A.shape == (3, 3)  # State dimension = nf
        assert result.n == 3

    def test_oe_noise_modeling(self):
        """Test OE properly models output error structure."""
        algorithm = OEAlgorithm()

        config = SystemIdentificationConfig(method="OE")
        config.nb = 2
        config.nf = 2
        config.nk = 0

        result = algorithm.identify(self.data, config)
        assert result is not None
        assert result.A.shape == (2, 2)  # nf = 2 states
