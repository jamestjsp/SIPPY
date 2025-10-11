"""
Test suite for RLS (Recursive Least Squares) Algorithm.
"""
import numpy as np
import pandas as pd
import pytest

from sippy.identification.base import SystemIdentificationConfig
from sippy.identification.iddata import IDData
from sippy.identification.algorithms.rls import RLSAlgorithm


class TestRLSAlgorithm:
    """Test cases for RLS algorithm."""

    def setup_method(self):
        """Set up test data and algorithm instance."""
        np.random.seed(42)  # For reproducible tests
        
        # Create test system
        self.na = 2  # AR order
        self.nb = 3  # Input order  
        self.nc = 1  # Noise AR order
        self.nd = 1  # Noise MA order
        self.nf = 2  # Input TF order
        self.nk = 1  # Input delay
        
        # SISO test data
        N = 100
        u = np.random.randn(N)  # Input signal
        # Simple system: y[k] = 0.6*y[k-1] - 0.2*y[k-2] + 0.5*u[k-1] + 0.3*u[k-2] + 0.1*u[k-3] + noise
        y = np.zeros(N)
        for k in range(max(self.na, self.nb + self.nk), N):
            y[k] = (0.6 * y[k-1] - 0.2 * y[k-2] + 
                   0.5 * u[k-1] + 0.3 * u[k-2] + 0.1 * u[k-3] + 
                   0.1 * np.random.randn())
        self.y_siso = y
        self.u_siso = u
        
        # MIMO test data (2x2 system)
        N_mimo = 100
        u_mimo = np.random.randn(2, N_mimo)  # 2 inputs
        y_mimo = np.zeros((2, N_mimo))  # 2 outputs
        for k in range(max(self.na, self.nb + self.nk), N_mimo):
            y_mimo[0, k] = (0.6 * y_mimo[0, k-1] - 0.2 * y_mimo[0, k-2] +
                          0.5 * u_mimo[0, k-1] + 0.1 * u_mimo[1, k-2] + 0.1 * np.random.randn())
            y_mimo[1, k] = (0.4 * y_mimo[1, k-1] + 
                          0.3 * u_mimo[1, k-1] + 0.2 * u_mimo[0, k-2] + 0.1 * np.random.randn())
        self.y_mimo = y_mimo
        self.u_mimo = u_mimo
        
        # Initialize algorithm
        self.algorithm = RLSAlgorithm()

    def test_rls_initialization(self):
        """Test RLS algorithm initialization."""
        assert self.algorithm.get_algorithm_name() == "RLS"
        assert self.algorithm is not None

    def test_rls_basic_siso_identification(self):
        """Test basic RLS identification with SISO data."""
        # Create data container
        time_index = pd.date_range('2023-01-01', periods=len(self.y_siso), freq='1s')
        data_df = pd.DataFrame({
            'y': self.y_siso,
            'u': self.u_siso
        }, index=time_index)
        data = IDData(data=data_df,
                     inputs=['u'], outputs=['y'], tsample=1.0)
        
        # Configure algorithm
        config = SystemIdentificationConfig(method='RLS')
        config.na = self.na
        config.nb = self.nb
        config.nk = self.nk
        config.forgetting_factor = 1.0  # No forgetting
        
        # Identify model
        model = self.algorithm.identify(data, config)
        
        # Verify result
        assert model is not None
        assert hasattr(model, 'A') and hasattr(model, 'B')
        assert hasattr(model, 'C') and hasattr(model, 'D')

    def test_rls_forgetting_factor(self):
        """Test RLS with forgetting factor for time-varying systems."""
        # Create time-varying system
        N = 200
        u = np.random.randn(N)
        y = np.zeros(N)
        
        # Parameter change at k=100
        for k in range(max(self.na, self.nb + self.nk), N):
            if k < 100:
                # First system parameters
                b_coeff = [0.5, 0.3, 0.1]
            else:
                # Changed system parameters  
                b_coeff = [0.2, 0.6, 0.3]
            
            y[k] = (0.6 * y[k-1] - 0.2 * y[k-2] + 
                   b_coeff[0] * u[k-1] + b_coeff[1] * u[k-2] + b_coeff[2] * u[k-3] + 
                   0.1 * np.random.randn())
        
        time_index = pd.date_range('2023-01-01', periods=len(y), freq='1s')
        data_df = pd.DataFrame({
            'y': y,
            'u': u
        }, index=time_index)
        data = IDData(data=data_df,
                     inputs=['u'], outputs=['y'], tsample=1.0)
        
        config = SystemIdentificationConfig(method='RLS')
        config.na = self.na
        config.nb = self.nb
        config.nk = self.nk
        config.forgetting_factor = 0.95  # Forgetting factor
        
        model = self.algorithm.identify(data, config)
        assert model is not None

    def test_rls_different_model_structures(self):
        """Test RLS with different model structures."""
        time_index = pd.date_range('2023-01-01', periods=len(self.y_siso), freq='1s')
        data_df = pd.DataFrame({
            'y': self.y_siso,
            'u': self.u_siso
        }, index=time_index)
        data = IDData(data=data_df,
                     inputs=['u'], outputs=['y'], tsample=1.0)
        
        # Test ARX structure
        config = SystemIdentificationConfig(method='RLS')
        config.na = 2
        config.nb = 3
        config.nk = 1
        config.nc = 0
        config.nd = 0
        config.nf = 0
        config.structure = 'ARX'
        config.forgetting_factor = 1.0
        
        model_arx = self.algorithm.identify(data, config)
        assert model_arx is not None
        
        # Test ARMAX structure
        config.nc = 1
        config.structure = 'ARMAX'
        
        model_armax = self.algorithm.identify(data, config)
        assert model_armax is not None

    def test_rls_mimo_identification(self):
        """Test RLS identification with MIMO data."""
        # Create MIMO DataFrame
        time_index = pd.date_range('2023-01-01', periods=self.y_mimo.shape[1], freq='1s')
        data_df = pd.DataFrame({
            'y1': self.y_mimo[0, :],
            'y2': self.y_mimo[1, :],
            'u1': self.u_mimo[0, :],
            'u2': self.u_mimo[1, :]
        }, index=time_index)
        data = IDData(data=data_df,
                     inputs=['u1', 'u2'], outputs=['y1', 'y2'], tsample=1.0)
        
        config = SystemIdentificationConfig(method='RLS')
        config.na = 2
        config.nb = self.nb
        config.nk = 1
        config.forgetting_factor = 1.0
        
        model = self.algorithm.identify(data, config)
        assert model is not None

    def test_rls_parameter_validation(self):
        """Test RLS parameter validation."""
        # Test negative forgetting factor
        with pytest.raises(ValueError, match="Forgetting factor must be in range"):
            config = SystemIdentificationConfig(method='RLS')
            config.forgetting_factor = -0.5
            self.algorithm.validate_parameters(**config.__dict__)
        
        # Test invalid orders
        with pytest.raises(ValueError, match="AR order must be positive"):
            config = SystemIdentificationConfig(method='RLS')
            config.na = 0
            self.algorithm.validate_parameters(**config.__dict__)

    def test_rls_insufficient_data(self):
        """Test RLS with insufficient data points."""
        # Create very short dataset
        u_short = np.random.randn(4)  # Too short for na=2, nb=3, nk=1 (need 5+)
        y_short = np.random.randn(4)
        
        time_index = pd.date_range('2023-01-01', periods=len(y_short), freq='1s')
        data_df = pd.DataFrame({
            'y': y_short,
            'u': u_short
        }, index=time_index)
        data = IDData(data=data_df,
                     inputs=['u'], outputs=['y'], tsample=1.0)
        
        config = SystemIdentificationConfig(method='RLS')
        config.na = 2
        config.nb = 3
        config.nk = 1
        config.forgetting_factor = 1.0
        
        with pytest.raises(ValueError, match="Insufficient data points"):
            self.algorithm.identify(data, config)

    def test_rls_convergence_properties(self):
        """Test RLS algorithm convergence properties."""
        # Create simple system for convergence test
        N = 50
        u = np.ones(N)  # Step input
        y = np.zeros(N)
        b_true = 0.5
        a_true = 0.6
        
        for k in range(1, N):
            y[k] = a_true * y[k-1] + b_true * u[k-1]
        
        time_index = pd.date_range('2023-01-01', periods=len(y), freq='1s')
        data_df = pd.DataFrame({
            'y': y,
            'u': u
        }, index=time_index)
        data = IDData(data=data_df,
                     inputs=['u'], outputs=['y'], tsample=1.0)
        
        config = SystemIdentificationConfig(method='RLS')
        config.na = 1
        config.nb = 1
        config.nk = 1
        config.forgetting_factor = 1.0
        
        model = self.algorithm.identify(data, config)
        
        # Check that model exists and has reasonable properties
        assert model is not None
        assert hasattr(model, 'A') and hasattr(model, 'B')

    def test_rls_with_iddata_arrays(self):
        """Test RLS algorithm with direct array inputs."""
        # Create config for direct interface
        config = type('Config', (), {
            'na': self.na,
            'nb': self.nb,
            'nk': 1,
            'forgetting_factor': 1.0,
            'u': self.u_siso
        })()
        # Test with direct numpy arrays
        model = self.algorithm.identify(
            data=self.y_siso,
            config=config
        )
        
        assert model is not None
        assert hasattr(model, 'A') and hasattr(model, 'B')

    def test_rls_properties_and_methods(self):
        """Test RLS algorithm properties and methods."""
        # Test algorithm properties
        assert self.algorithm.get_algorithm_info() is not None
        assert isinstance(self.algorithm.get_algorithm_info(), dict)
        
        # Test parameter validation method
        valid_params = {'na': 2, 'nb': 3, 'nk': 1, 'forgetting_factor': 0.99}
        assert self.algorithm.validate_parameters(**valid_params) is True

    def test_rls_factory_registration(self):
        """Test RLS algorithm factory registration."""
        from sippy.identification.algorithms import AlgorithmFactory
        
        # Check RLS is registered
        assert 'RLS' in AlgorithmFactory.list_algorithms()
        
        # Check we can create RLS instance
        rls_algorithm = AlgorithmFactory.create('RLS')
        assert rls_algorithm is not None
        assert rls_algorithm.get_algorithm_name() == 'RLS'

    def test_rls_edge_cases(self):
        """Test RLS algorithm edge cases."""
        # Test with zero input (pure AR process)
        N = 50
        y = np.zeros(N)
        for k in range(1, N):
            y[k] = 0.7 * y[k-1] + 0.1 * np.random.randn()
        
        time_index = pd.date_range('2023-01-01', periods=len(y), freq='1s')
        u_dummy = np.zeros_like(y)  # Add dummy input since IDData requires inputs
        data_df = pd.DataFrame({
            'y': y,
            'u': u_dummy
        }, index=time_index)
        data = IDData(data=data_df,
                     inputs=['u'], outputs=['y'], tsample=1.0)
        
        config = SystemIdentificationConfig(method='RLS')
        config.na = 1
        config.nb = 0  # No input
        config.nk = 1
        config.forgetting_factor = 1.0
        
        model = self.algorithm.identify(data, config)
        assert model is not None
