# Asteroseismology Linewidths

Analysis tools for determining if there is a dependence of oscillation mode linewidths on spherical degree (l) in solar-like oscillators.

## Overview

This package provides tools for:
- **Signal to Background Noise (SBR)** calculations for stellar power spectra
- **Bayesian peakbagging** to extract oscillation mode parameters
- **Spherical degree analysis** to investigate the l-dependence of linewidths

## Installation

```bash
# Clone the repository
git clone https://github.com/aileeknauff/asteroseismology-linewidths.git
cd asteroseismology-linewidths

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Spherical Degree Analysis

```python
import numpy as np
from src.linewidths import SphericalDegreeAnalysis

# Example data
frequencies = np.array([1800, 1900, 2000, 2100, 2200, 2300])  # microHz
linewidths = np.array([0.45, 0.52, 0.48, 0.55, 0.50, 0.58])  # microHz
degrees = np.array([0, 1, 2, 0, 1, 2])  # spherical degree l

# Create analysis object
analysis = SphericalDegreeAnalysis(frequencies, linewidths, degrees)

# Get summary statistics
print(analysis.summary())

# Test for degree dependence using ANOVA
result = analysis.test_degree_dependence()
print(f"p-value: {result['p_value']:.4f}")
print(f"Significant: {result['significant']}")
```

### Signal to Background Ratio

```python
from src.linewidths import calculate_sbr, estimate_background

# Calculate SBR from power spectrum
sbr_result = calculate_sbr(frequency, power, nu_max=2000, delta_nu=100)
print(f"Signal to Background Ratio: {sbr_result['sbr']:.2f}")
```

### Bayesian Peakbagging

```python
from src.linewidths import BayesianPeakbagger

# Initialize peakbagger
pb = BayesianPeakbagger(frequency, power)

# Find peaks automatically
modes = pb.find_peaks(threshold_sigma=4.0)

# Fit individual modes
fitted_params = pb.fit_single_mode(nu_0=2000, n_samples=1000)
print(f"Linewidth: {fitted_params['linewidth']:.3f} ± {fitted_params['linewidth_err']:.3f} µHz")
```

## Project Structure

```
asteroseismology-linewidths/
├── src/
│   └── linewidths/
│       ├── __init__.py          # Package initialization
│       ├── spherical_degree.py  # Spherical degree analysis
│       ├── signal_background.py # SBR calculations
│       └── peakbagging.py       # Bayesian peakbagging
├── tests/
│   ├── test_spherical_degree.py
│   ├── test_signal_background.py
│   └── test_peakbagging.py
├── data/                        # Place your data files here
├── requirements.txt
└── README.md
```

## Running Tests

```bash
pytest tests/ -v
```

## Scientific Background

In asteroseismology, oscillation modes are characterized by their spherical degree (l), which describes the number of nodal lines on the stellar surface. The linewidth (Γ) of an oscillation mode is related to the mode lifetime (τ) by:

Γ = 1 / (π τ)

Understanding whether linewidths depend on spherical degree can provide insights into the physical processes that damp stellar oscillations.

## References

- Chaplin, W. J. & Miglio, A. (2013). "Asteroseismology of Solar-Type and Red-Giant Stars". Annual Review of Astronomy and Astrophysics.
- Handberg, R. & Campante, T. L. (2011). "Bayesian peak-bagging of solar-like oscillators using MCMC". Astronomy & Astrophysics.
