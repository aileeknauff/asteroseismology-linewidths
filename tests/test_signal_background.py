"""
Tests for the Signal to Background module.
"""

import numpy as np
import pytest
from src.linewidths.signal_background import (
    calculate_sbr,
    estimate_background,
    identify_oscillation_envelope
)


class TestEstimateBackground:
    """Test suite for background estimation functions."""
    
    @pytest.fixture
    def synthetic_spectrum(self):
        """Generate a synthetic power spectrum with known background."""
        np.random.seed(42)
        
        # Frequency array
        frequency = np.linspace(100, 4000, 10000)
        
        # Harvey-like background
        background = 1e4 / (1 + (frequency / 200)**4) + 100
        
        # Add oscillations around nu_max = 2000 microHz
        nu_max = 2000
        delta_nu = 100
        envelope = 500 * np.exp(-0.5 * ((frequency - nu_max) / 300)**2)
        
        # Add individual modes
        modes = np.zeros_like(frequency)
        for n in range(-5, 6):
            for l in range(3):
                nu = nu_max + n * delta_nu + l * 0.5  # Simplified mode pattern
                gamma = 0.5  # Linewidth
                modes += 200 * np.exp(-0.5 * ((frequency - nu) / gamma)**2)
        
        power = background + modes + 50 * np.random.exponential(size=len(frequency))
        
        return {
            "frequency": frequency,
            "power": power,
            "true_background": background,
            "nu_max": nu_max,
            "delta_nu": delta_nu
        }
    
    def test_smooth_background(self, synthetic_spectrum):
        """Test smooth background estimation."""
        bg = estimate_background(
            synthetic_spectrum["frequency"],
            synthetic_spectrum["power"],
            method="smooth"
        )
        
        assert len(bg) == len(synthetic_spectrum["frequency"])
        assert np.all(bg > 0)
    
    def test_harvey_background(self, synthetic_spectrum):
        """Test Harvey model background estimation."""
        bg = estimate_background(
            synthetic_spectrum["frequency"],
            synthetic_spectrum["power"],
            method="harvey"
        )
        
        assert len(bg) == len(synthetic_spectrum["frequency"])
        assert np.all(bg > 0)
    
    def test_invalid_method(self, synthetic_spectrum):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError):
            estimate_background(
                synthetic_spectrum["frequency"],
                synthetic_spectrum["power"],
                method="invalid"
            )


class TestCalculateSBR:
    """Test suite for SBR calculation."""
    
    @pytest.fixture
    def simple_spectrum(self):
        """Generate a simple spectrum for testing."""
        np.random.seed(42)
        
        frequency = np.linspace(100, 4000, 1000)
        background = 100 * np.ones_like(frequency)
        signal = 50 * np.ones_like(frequency)  # Flat signal for simplicity
        
        power = background + signal
        
        return {
            "frequency": frequency,
            "power": power,
            "background": background,
            "expected_sbr": 0.5  # signal / background
        }
    
    def test_calculate_sbr_with_background(self, simple_spectrum):
        """Test SBR calculation with provided background."""
        result = calculate_sbr(
            simple_spectrum["frequency"],
            simple_spectrum["power"],
            background=simple_spectrum["background"]
        )
        
        assert "sbr" in result
        assert "signal_power" in result
        assert "background_power" in result
        assert "snr" in result
        
        # Check approximate value
        assert np.isclose(result["sbr"], simple_spectrum["expected_sbr"], rtol=0.1)
    
    def test_calculate_sbr_without_background(self, simple_spectrum):
        """Test SBR calculation without provided background."""
        result = calculate_sbr(
            simple_spectrum["frequency"],
            simple_spectrum["power"]
        )
        
        assert "sbr" in result
        assert "background" in result
    
    def test_calculate_sbr_with_numax(self, simple_spectrum):
        """Test SBR calculation with nu_max and delta_nu."""
        result = calculate_sbr(
            simple_spectrum["frequency"],
            simple_spectrum["power"],
            background=simple_spectrum["background"],
            nu_max=2000,
            delta_nu=100
        )
        
        assert "sbr" in result


class TestIdentifyOscillationEnvelope:
    """Test suite for envelope identification."""
    
    def test_identify_envelope(self):
        """Test oscillation envelope identification."""
        np.random.seed(42)
        
        frequency = np.linspace(500, 3500, 1000)
        nu_max_true = 2000
        
        # Create Gaussian envelope with clear signal above background
        envelope = 5000 * np.exp(-0.5 * ((frequency - nu_max_true) / 300)**2)
        background = 100 + 1e3 / (1 + (frequency / 200)**4)
        power = background + envelope + 10 * np.abs(np.random.randn(len(frequency)))
        
        nu_max, width, amplitude = identify_oscillation_envelope(frequency, power)
        
        # Check that envelope parameters are reasonable
        assert width > 0
        assert amplitude > 0
        # nu_max should be within the frequency range
        assert frequency.min() <= nu_max <= frequency.max()
        assert width > 0
        assert amplitude > 0
