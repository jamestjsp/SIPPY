"""
Test cases for FIR identification algorithm implementation.
"""

import numpy as np
import pandas as pd
import pytest

from sippy import systems as control
from sippy.identification import IDData, SystemIdentificationConfig
from sippy.identification.algorithms import fir as fir_module
from sippy.identification.algorithms.fir import FIRAlgorithm, _kernel_matrix
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

    def test_plain_least_squares_is_the_backward_compatible_default(self):
        default = FIRAlgorithm().identify(iddata=self.data, nb=8, nk=1)
        explicit = FIRAlgorithm().identify(
            iddata=self.data,
            nb=8,
            nk=1,
            regularization="none",
        )

        np.testing.assert_allclose(
            default.identification_info["fir_coefficients"],
            explicit.identification_info["fir_coefficients"],
            rtol=0.0,
            atol=0.0,
        )
        assert explicit.identification_info["regularization"] == "none"

    @pytest.mark.parametrize("kernel", ["tc", "dc"])
    def test_kernel_regularization_supports_mimo(self, kernel):
        rng = np.random.default_rng(58)
        u = rng.normal(size=(2, 180))
        y = np.vstack(
            (
                np.convolve(u[0], [0.7, 0.3], mode="full")[:180],
                np.convolve(u[1], [0.5, -0.2], mode="full")[:180],
            )
        )
        y += 0.25 * rng.normal(size=y.shape)

        result = FIRAlgorithm().identify(
            y=y,
            u=u,
            nb=16,
            nk=0,
            regularization=kernel,
        )

        assert result.G.shape == (2, 2)
        hyperparameters = result.identification_info["kernel_hyperparameters"]
        assert len(hyperparameters) == 2
        assert all(parameters["kernel"] == kernel for parameters in hyperparameters)
        assert all(0.0 < parameters["decay"] < 1.0 for parameters in hyperparameters)
        assert all(parameters["scale"] > 0.0 for parameters in hyperparameters)
        assert all(parameters["noise_variance"] > 0.0 for parameters in hyperparameters)

    def test_tc_regularization_improves_noisy_long_fir_estimation(self):
        rng = np.random.default_rng(912)
        coefficient_count = 40
        lags = np.arange(coefficient_count)
        true_coefficients = 0.9**lags * (
            0.65 * np.cos(0.22 * lags) + 0.25 * np.sin(0.08 * lags)
        )
        u = rng.normal(size=150)
        clean = np.convolve(u, true_coefficients, mode="full")[: u.size]
        y = clean + 0.8 * rng.normal(size=u.size)

        least_squares = FIRAlgorithm().identify(
            y=y,
            u=u,
            nb=coefficient_count,
            nk=0,
            regularization="none",
        )
        regularized = FIRAlgorithm().identify(
            y=y,
            u=u,
            nb=coefficient_count,
            nk=0,
            regularization="tc",
        )
        least_squares_error = np.linalg.norm(
            least_squares.identification_info["fir_coefficients"][0] - true_coefficients
        )
        regularized_error = np.linalg.norm(
            regularized.identification_info["fir_coefficients"][0] - true_coefficients
        )

        assert regularized_error < 0.75 * least_squares_error

    def test_kernel_tuning_factorizes_only_parameter_sized_matrices(self, monkeypatch):
        rng = np.random.default_rng(913)
        sample_count = 2_000
        coefficient_count = 24
        u = rng.normal(size=sample_count)
        y = np.convolve(u, 0.85 ** np.arange(coefficient_count), mode="full")[
            :sample_count
        ]
        factor_shapes = []
        original = fir_module.scipy.linalg.cho_factor

        def tracked_factor(matrix, *args, **kwargs):
            factor_shapes.append(matrix.shape)
            return original(matrix, *args, **kwargs)

        monkeypatch.setattr(fir_module.scipy.linalg, "cho_factor", tracked_factor)

        FIRAlgorithm().identify(
            y=y,
            u=u,
            nb=coefficient_count,
            nk=0,
            regularization="tc",
        )

        assert factor_shapes
        assert all(max(shape) <= coefficient_count for shape in factor_shapes)

    def test_tc_and_dc_kernel_definitions(self):
        tc = _kernel_matrix("tc", coefficient_count=3, decay=0.8)
        expected_tc = np.array(
            [
                [0.8, 0.8**2, 0.8**3],
                [0.8**2, 0.8**2, 0.8**3],
                [0.8**3, 0.8**3, 0.8**3],
            ]
        )
        np.testing.assert_allclose(tc, expected_tc)

        dc = _kernel_matrix(
            "dc",
            coefficient_count=3,
            decay=0.8,
            correlation=0.6,
        )
        indices = np.arange(1, 4)
        expected_dc = 0.8 ** (
            (indices[:, None] + indices[None, :]) / 2
        ) * 0.6 ** np.abs(indices[:, None] - indices[None, :])
        np.testing.assert_allclose(dc, expected_dc)
        assert np.all(np.linalg.eigvalsh(dc) > 0.0)

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

        with pytest.raises(ValueError, match="regularization"):
            algorithm.identify(
                iddata=self.data,
                nb=3,
                nk=1,
                regularization="stable-spline-3",
            )

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
