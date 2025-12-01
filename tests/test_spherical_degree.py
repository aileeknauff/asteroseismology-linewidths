"""
Tests for the Spherical Degree Analysis module.
"""

import numpy as np
import pytest
from src.linewidths.spherical_degree import SphericalDegreeAnalysis


class TestSphericalDegreeAnalysis:
    """Test suite for SphericalDegreeAnalysis class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample oscillation mode data."""
        np.random.seed(42)
        
        # Generate synthetic data for l=0, 1, 2 modes
        n_modes = 30
        
        # Frequencies spanning a typical oscillation range
        base_frequencies = np.linspace(1000, 2000, n_modes)
        
        # Spherical degrees (cyclic pattern like real stars)
        degrees = np.tile([0, 1, 2], n_modes // 3 + 1)[:n_modes]
        
        # Linewidths with slight l-dependence
        base_linewidth = 0.5
        l_effect = np.array([0, 0.1, 0.15])[degrees.astype(int)]
        linewidths = base_linewidth + l_effect + 0.05 * np.random.randn(n_modes)
        
        uncertainties = 0.1 * np.ones(n_modes)
        
        return {
            "frequencies": base_frequencies,
            "linewidths": linewidths,
            "degrees": degrees,
            "uncertainties": uncertainties
        }
    
    def test_initialization(self, sample_data):
        """Test that the class initializes correctly."""
        analysis = SphericalDegreeAnalysis(
            frequencies=sample_data["frequencies"],
            linewidths=sample_data["linewidths"],
            degrees=sample_data["degrees"]
        )
        
        assert len(analysis.frequencies) == len(sample_data["frequencies"])
        assert len(analysis.linewidths) == len(sample_data["linewidths"])
        assert len(analysis.degrees) == len(sample_data["degrees"])
        assert analysis.uncertainties is None
    
    def test_initialization_with_uncertainties(self, sample_data):
        """Test initialization with uncertainties."""
        analysis = SphericalDegreeAnalysis(
            frequencies=sample_data["frequencies"],
            linewidths=sample_data["linewidths"],
            degrees=sample_data["degrees"],
            uncertainties=sample_data["uncertainties"]
        )
        
        assert analysis.uncertainties is not None
        assert len(analysis.uncertainties) == len(sample_data["uncertainties"])
    
    def test_validation_mismatched_lengths(self):
        """Test that mismatched array lengths raise ValueError."""
        with pytest.raises(ValueError):
            SphericalDegreeAnalysis(
                frequencies=np.array([1000, 1100, 1200]),
                linewidths=np.array([0.5, 0.6]),  # Wrong length
                degrees=np.array([0, 1, 2])
            )
    
    def test_group_by_degree(self, sample_data):
        """Test grouping data by spherical degree."""
        analysis = SphericalDegreeAnalysis(
            frequencies=sample_data["frequencies"],
            linewidths=sample_data["linewidths"],
            degrees=sample_data["degrees"]
        )
        
        grouped = analysis.group_by_degree()
        
        assert 0 in grouped
        assert 1 in grouped
        assert 2 in grouped
        
        # Check that all modes are accounted for
        total_modes = sum(len(g["frequencies"]) for g in grouped.values())
        assert total_modes == len(sample_data["frequencies"])
    
    def test_calculate_mean_linewidth_by_degree(self, sample_data):
        """Test mean linewidth calculation."""
        analysis = SphericalDegreeAnalysis(
            frequencies=sample_data["frequencies"],
            linewidths=sample_data["linewidths"],
            degrees=sample_data["degrees"]
        )
        
        means = analysis.calculate_mean_linewidth_by_degree()
        
        assert 0 in means
        assert 1 in means
        assert 2 in means
        
        # Each result should be (mean, std_error) tuple
        for l, (mean, err) in means.items():
            assert isinstance(mean, float)
            assert isinstance(err, float)
            assert err >= 0
    
    def test_test_degree_dependence(self, sample_data):
        """Test ANOVA for degree dependence."""
        analysis = SphericalDegreeAnalysis(
            frequencies=sample_data["frequencies"],
            linewidths=sample_data["linewidths"],
            degrees=sample_data["degrees"]
        )
        
        result = analysis.test_degree_dependence()
        
        assert "f_statistic" in result
        assert "p_value" in result
        assert "significant" in result
        # Check boolean value (numpy bool or Python bool)
        assert result["significant"] in (True, False)
    
    def test_fit_linewidth_frequency_relation(self, sample_data):
        """Test power-law fit."""
        analysis = SphericalDegreeAnalysis(
            frequencies=sample_data["frequencies"],
            linewidths=sample_data["linewidths"],
            degrees=sample_data["degrees"]
        )
        
        result = analysis.fit_linewidth_frequency_relation()
        
        assert "amplitude" in result
        assert "exponent" in result
    
    def test_fit_linewidth_frequency_relation_by_degree(self, sample_data):
        """Test power-law fit for specific degree."""
        analysis = SphericalDegreeAnalysis(
            frequencies=sample_data["frequencies"],
            linewidths=sample_data["linewidths"],
            degrees=sample_data["degrees"]
        )
        
        result = analysis.fit_linewidth_frequency_relation(degree=0)
        
        assert "amplitude" in result
        assert "exponent" in result
    
    def test_compare_degrees(self, sample_data):
        """Test comparison between two degrees."""
        analysis = SphericalDegreeAnalysis(
            frequencies=sample_data["frequencies"],
            linewidths=sample_data["linewidths"],
            degrees=sample_data["degrees"]
        )
        
        result = analysis.compare_degrees(0, 1)
        
        assert "t_statistic" in result
        assert "p_value" in result
        assert "mean_difference" in result
        assert "significant" in result
    
    def test_compare_degrees_invalid(self, sample_data):
        """Test that comparing invalid degrees raises ValueError."""
        analysis = SphericalDegreeAnalysis(
            frequencies=sample_data["frequencies"],
            linewidths=sample_data["linewidths"],
            degrees=sample_data["degrees"]
        )
        
        with pytest.raises(ValueError):
            analysis.compare_degrees(0, 5)  # l=5 doesn't exist
    
    def test_summary(self, sample_data):
        """Test summary generation."""
        analysis = SphericalDegreeAnalysis(
            frequencies=sample_data["frequencies"],
            linewidths=sample_data["linewidths"],
            degrees=sample_data["degrees"]
        )
        
        summary = analysis.summary()
        
        assert isinstance(summary, str)
        assert "Spherical Degree" in summary
        assert "l=0" in summary
        assert "ANOVA" in summary


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_mode_per_degree(self):
        """Test with only one mode per degree."""
        analysis = SphericalDegreeAnalysis(
            frequencies=np.array([1000, 1100, 1200]),
            linewidths=np.array([0.5, 0.6, 0.7]),
            degrees=np.array([0, 1, 2])
        )
        
        means = analysis.calculate_mean_linewidth_by_degree()
        
        # With single mode, std_error should be 0
        for l, (mean, err) in means.items():
            assert err == 0
    
    def test_single_degree(self):
        """Test with all modes having same degree."""
        analysis = SphericalDegreeAnalysis(
            frequencies=np.array([1000, 1100, 1200, 1300]),
            linewidths=np.array([0.5, 0.6, 0.7, 0.55]),
            degrees=np.array([0, 0, 0, 0])
        )
        
        result = analysis.test_degree_dependence()
        
        # With single degree, ANOVA should return NaN
        assert np.isnan(result["f_statistic"])
