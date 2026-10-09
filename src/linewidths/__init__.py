"""
Asteroseismology Linewidths Analysis Package

This package provides tools for analyzing stellar oscillation mode linewidths,
including Signal to Background Noise ratio calculations and Bayesian peakbagging
for determining spherical degree dependence.
"""

from .spherical_degree import SphericalDegreeAnalysis
from .signal_background import calculate_sbr, estimate_background
from .peakbagging import BayesianPeakbagger

__version__ = "0.1.0"
__all__ = [
    "SphericalDegreeAnalysis",
    "calculate_sbr",
    "estimate_background",
    "BayesianPeakbagger",
]
