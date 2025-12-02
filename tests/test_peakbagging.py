"""
Tests for the Bayesian Peakbagging module.
"""

import numpy as np
import pytest
from src.linewidths.peakbagging import BayesianPeakbagger


class TestBayesianPeakbagger:
    """Test suite for BayesianPeakbagger class."""
    
    @pytest.fixture
    def synthetic_spectrum(self):
        """Generate a synthetic power spectrum with known modes."""
        np.random.seed(42)
        
        # Frequency array
        frequency = np.linspace(1800, 2200, 4000)
        
        # Background
        background = 50 * np.ones_like(frequency)
        
        # Add oscillation modes
        mode_frequencies = [1900, 2000, 2100]
        mode_heights = [200, 300, 250]
        mode_linewidths = [0.5, 0.6, 0.55]  # FWHM
        
        power = background.copy()
        for nu, h, gamma in zip(mode_frequencies, mode_heights, mode_linewidths):
            # Add Lorentzian mode
            power += h / (1 + ((frequency - nu) / (gamma / 2))**2)
        
        # Add noise (chi-squared distributed)
        power = power * np.random.exponential(size=len(frequency))
        
        return {
            "frequency": frequency,
            "power": power,
            "background": background,
            "mode_frequencies": mode_frequencies,
            "mode_heights": mode_heights,
            "mode_linewidths": mode_linewidths
        }
    
    def test_initialization(self, synthetic_spectrum):
        """Test that the class initializes correctly."""
        pb = BayesianPeakbagger(
            frequency=synthetic_spectrum["frequency"],
            power=synthetic_spectrum["power"]
        )
        
        assert len(pb.frequency) == len(synthetic_spectrum["frequency"])
        assert len(pb.power) == len(synthetic_spectrum["power"])
        assert pb.background is not None
    
    def test_initialization_with_background(self, synthetic_spectrum):
        """Test initialization with provided background."""
        pb = BayesianPeakbagger(
            frequency=synthetic_spectrum["frequency"],
            power=synthetic_spectrum["power"],
            background=synthetic_spectrum["background"]
        )
        
        np.testing.assert_array_equal(pb.background, synthetic_spectrum["background"])
    
    def test_lorentzian(self):
        """Test Lorentzian profile calculation."""
        frequency = np.linspace(0, 10, 100)
        height = 100
        nu_0 = 5
        gamma = 1
        
        profile = BayesianPeakbagger.lorentzian(frequency, height, nu_0, gamma)
        
        # Check peak value
        peak_idx = np.argmax(profile)
        assert np.isclose(frequency[peak_idx], nu_0, atol=0.1)
        assert np.isclose(profile[peak_idx], height, rtol=0.1)
    
    def test_model_spectrum(self, synthetic_spectrum):
        """Test model spectrum generation."""
        pb = BayesianPeakbagger(
            frequency=synthetic_spectrum["frequency"],
            power=synthetic_spectrum["power"],
            background=synthetic_spectrum["background"]
        )
        
        params = {
            "heights": np.array([100, 200]),
            "frequencies": np.array([1950, 2050]),
            "linewidths": np.array([0.5, 0.5])
        }
        
        model = pb.model_spectrum(params)
        
        assert len(model) == len(pb.frequency)
        # Model should be greater than background due to added modes
        assert np.mean(model) > np.mean(pb.background)
    
    def test_log_likelihood(self, synthetic_spectrum):
        """Test log-likelihood calculation."""
        pb = BayesianPeakbagger(
            frequency=synthetic_spectrum["frequency"],
            power=synthetic_spectrum["power"],
            background=synthetic_spectrum["background"]
        )
        
        params = {
            "heights": np.array([200]),
            "frequencies": np.array([2000]),
            "linewidths": np.array([0.5])
        }
        
        log_like = pb.log_likelihood(params)
        
        assert np.isfinite(log_like)
    
    def test_log_prior(self, synthetic_spectrum):
        """Test log-prior calculation."""
        pb = BayesianPeakbagger(
            frequency=synthetic_spectrum["frequency"],
            power=synthetic_spectrum["power"]
        )
        
        # Valid parameters
        valid_params = {
            "heights": np.array([100]),
            "frequencies": np.array([2000]),
            "linewidths": np.array([0.5])
        }
        assert pb.log_prior(valid_params) == 0.0
        
        # Invalid parameters (negative linewidth)
        invalid_params = {
            "heights": np.array([100]),
            "frequencies": np.array([2000]),
            "linewidths": np.array([-0.5])
        }
        assert pb.log_prior(invalid_params) == -np.inf
    
    def test_find_peaks(self, synthetic_spectrum):
        """Test peak finding."""
        pb = BayesianPeakbagger(
            frequency=synthetic_spectrum["frequency"],
            power=synthetic_spectrum["power"],
            background=synthetic_spectrum["background"]
        )
        
        modes = pb.find_peaks(threshold_sigma=3.0)
        
        assert len(modes) > 0
        
        for mode in modes:
            assert "frequency" in mode
            assert "height" in mode
            assert "linewidth" in mode
    
    def test_fit_single_mode(self, synthetic_spectrum):
        """Test single mode fitting."""
        pb = BayesianPeakbagger(
            frequency=synthetic_spectrum["frequency"],
            power=synthetic_spectrum["power"],
            background=synthetic_spectrum["background"]
        )
        
        # Fit mode at 2000 microHz
        result = pb.fit_single_mode(2000, n_samples=200)
        
        assert "frequency" in result
        assert "frequency_err" in result
        assert "height" in result
        assert "linewidth" in result
        
        # Check that fitted frequency is close to true value
        assert np.isclose(result["frequency"], 2000, atol=5)
    
    def test_extract_linewidths_by_degree(self, synthetic_spectrum):
        """Test linewidth extraction by degree."""
        pb = BayesianPeakbagger(
            frequency=synthetic_spectrum["frequency"],
            power=synthetic_spectrum["power"],
            background=synthetic_spectrum["background"]
        )
        
        # Set up some modes
        pb.modes = [
            {"frequency": 1900, "linewidth": 0.5},
            {"frequency": 2000, "linewidth": 0.6},
            {"frequency": 2100, "linewidth": 0.55}
        ]
        
        # Mode identification
        mode_id = {
            0: {"n": 15, "l": 0},
            1: {"n": 15, "l": 1},
            2: {"n": 15, "l": 2}
        }
        
        result = pb.extract_linewidths_by_degree(mode_id)
        
        assert 0 in result
        assert 1 in result
        assert 2 in result
        assert result[0] == [0.5]
        assert result[1] == [0.6]
        assert result[2] == [0.55]
