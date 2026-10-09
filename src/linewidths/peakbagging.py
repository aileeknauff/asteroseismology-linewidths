"""
Bayesian Peakbagging Module

This module provides tools for Bayesian inference of oscillation mode
parameters, including frequencies, linewidths, and amplitudes.
"""

import numpy as np
from scipy.signal import find_peaks as scipy_find_peaks
from typing import Optional, Dict, List, Tuple, Union


class BayesianPeakbagger:
    """
    Bayesian peakbagging for stellar oscillation modes.
    
    Uses Markov Chain Monte Carlo (MCMC) to infer mode parameters
    from power spectra.
    
    Parameters
    ----------
    frequency : array-like
        Frequency array in microHz
    power : array-like
        Power spectral density
    background : array-like, optional
        Background power spectrum. If None, will be estimated.
    
    Attributes
    ----------
    modes : list
        List of identified oscillation modes with parameters
    samples : dict
        MCMC samples for each parameter
    """
    
    def __init__(
        self,
        frequency: np.ndarray,
        power: np.ndarray,
        background: Optional[np.ndarray] = None
    ):
        self.frequency = np.asarray(frequency)
        self.power = np.asarray(power)
        
        if background is not None:
            self.background = np.asarray(background)
        else:
            from .signal_background import estimate_background
            self.background = estimate_background(self.frequency, self.power)
        
        self.modes = []
        self.samples = {}
    
    @staticmethod
    def lorentzian(
        frequency: np.ndarray,
        height: float,
        nu_0: float,
        gamma: float
    ) -> np.ndarray:
        """
        Lorentzian profile for an oscillation mode.
        
        Parameters
        ----------
        frequency : np.ndarray
            Frequency array
        height : float
            Mode height (power at peak)
        nu_0 : float
            Central frequency
        gamma : float
            Half-width at half maximum (HWHM)
        
        Returns
        -------
        np.ndarray
            Lorentzian profile
        """
        return height / (1 + ((frequency - nu_0) / gamma)**2)
    
    def model_spectrum(
        self,
        params: Dict[str, np.ndarray],
        mode_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Generate model power spectrum from mode parameters.
        
        Parameters
        ----------
        params : dict
            Dictionary with arrays of heights, frequencies, and linewidths
        mode_indices : list, optional
            Indices of modes to include. If None, include all.
        
        Returns
        -------
        np.ndarray
            Model power spectrum
        """
        model = self.background.copy()
        
        heights = params["heights"]
        frequencies = params["frequencies"]
        linewidths = params["linewidths"]
        
        if mode_indices is None:
            mode_indices = range(len(heights))
        
        for i in mode_indices:
            # Linewidth is FWHM, convert to HWHM (gamma)
            gamma = linewidths[i] / 2
            model += self.lorentzian(
                self.frequency,
                heights[i],
                frequencies[i],
                gamma
            )
        
        return model
    
    def log_likelihood(
        self,
        params: Dict[str, np.ndarray]
    ) -> float:
        """
        Calculate log-likelihood for given parameters.
        
        Assumes chi-squared distributed power spectrum
        (exponential distribution for each frequency bin).
        
        Parameters
        ----------
        params : dict
            Mode parameters
        
        Returns
        -------
        float
            Log-likelihood value
        """
        model = self.model_spectrum(params)
        
        # Avoid log of zero or negative values
        model = np.maximum(model, 1e-10)
        
        # Chi-squared likelihood for power spectrum
        # P(data|model) = prod_i (1/model_i) * exp(-data_i/model_i)
        log_like = -np.sum(np.log(model) + self.power / model)
        
        return log_like
    
    def log_prior(
        self,
        params: Dict[str, np.ndarray],
        prior_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> float:
        """
        Calculate log-prior probability.
        
        Uses uniform priors within specified ranges.
        
        Parameters
        ----------
        params : dict
            Mode parameters
        prior_ranges : dict, optional
            Prior ranges for each parameter type
        
        Returns
        -------
        float
            Log-prior value (0 if valid, -inf if outside range)
        """
        if prior_ranges is None:
            prior_ranges = {
                "heights": (0, np.max(self.power) * 10),
                "frequencies": (self.frequency.min(), self.frequency.max()),
                "linewidths": (0.01, 10)  # microHz
            }
        
        for param_type, values in params.items():
            low, high = prior_ranges.get(param_type, (-np.inf, np.inf))
            if np.any(values < low) or np.any(values > high):
                return -np.inf
        
        return 0.0
    
    def find_peaks(
        self,
        threshold_sigma: float = 4.0,
        min_distance: float = 1.0
    ) -> List[Dict[str, float]]:
        """
        Find initial peak locations in the power spectrum.
        
        Parameters
        ----------
        threshold_sigma : float
            Detection threshold in units of standard deviation
            above background
        min_distance : float
            Minimum separation between peaks in microHz
        
        Returns
        -------
        list
            List of dictionaries with initial mode parameters
        """
        # Calculate signal-to-noise ratio
        snr = (self.power - self.background) / self.background
        
        # Convert min_distance to number of bins
        freq_resolution = np.median(np.diff(self.frequency))
        distance_bins = max(1, int(min_distance / freq_resolution))
        
        # Find peaks
        peak_indices, properties = scipy_find_peaks(
            snr,
            height=threshold_sigma,
            distance=distance_bins,
            prominence=threshold_sigma / 2
        )
        
        modes = []
        for idx in peak_indices:
            mode = {
                "frequency": self.frequency[idx],
                "height": self.power[idx] - self.background[idx],
                "linewidth": 0.5,  # Initial guess in microHz
                "snr": snr[idx]
            }
            modes.append(mode)
        
        self.modes = modes
        return modes
    
    def fit_single_mode(
        self,
        nu_0_init: float,
        freq_range: Optional[Tuple[float, float]] = None,
        n_samples: int = 1000
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Fit a single oscillation mode using MCMC.
        
        Parameters
        ----------
        nu_0_init : float
            Initial guess for mode frequency
        freq_range : tuple, optional
            Frequency range to fit (min, max). If None, uses ±5 microHz
        n_samples : int
            Number of MCMC samples
        
        Returns
        -------
        dict
            Fitted parameters with uncertainties and samples
        """
        if freq_range is None:
            freq_range = (nu_0_init - 5, nu_0_init + 5)
        
        # Select data in frequency range
        mask = (self.frequency >= freq_range[0]) & (self.frequency <= freq_range[1])
        freq_subset = self.frequency[mask]
        power_subset = self.power[mask]
        bg_subset = self.background[mask]
        
        # Initial parameters: [height, frequency, linewidth]
        height_init = np.max(power_subset) - np.mean(bg_subset)
        params_init = np.array([height_init, nu_0_init, 0.5])
        
        # Simple Metropolis-Hastings MCMC
        samples = self._run_mcmc(
            freq_subset, power_subset, bg_subset,
            params_init, n_samples
        )
        
        # Calculate statistics
        height_samples = samples[:, 0]
        freq_samples = samples[:, 1]
        lw_samples = samples[:, 2]
        
        return {
            "frequency": np.median(freq_samples),
            "frequency_err": np.std(freq_samples),
            "height": np.median(height_samples),
            "height_err": np.std(height_samples),
            "linewidth": np.median(lw_samples),
            "linewidth_err": np.std(lw_samples),
            "samples": samples
        }
    
    def _run_mcmc(
        self,
        frequency: np.ndarray,
        power: np.ndarray,
        background: np.ndarray,
        params_init: np.ndarray,
        n_samples: int,
        step_sizes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Run Metropolis-Hastings MCMC.
        
        Parameters
        ----------
        frequency : np.ndarray
            Frequency array
        power : np.ndarray
            Power spectrum
        background : np.ndarray
            Background spectrum
        params_init : np.ndarray
            Initial parameters [height, freq, linewidth]
        n_samples : int
            Number of samples
        step_sizes : np.ndarray, optional
            Step sizes for proposals
        
        Returns
        -------
        np.ndarray
            MCMC samples (n_samples x 3)
        """
        if step_sizes is None:
            step_sizes = np.array([params_init[0] * 0.1, 0.1, 0.05])
        
        def log_prob(params):
            height, nu_0, gamma = params
            
            # Prior constraints
            if height <= 0 or gamma <= 0 or gamma > 5:
                return -np.inf
            if nu_0 < frequency.min() or nu_0 > frequency.max():
                return -np.inf
            
            # Model spectrum
            model = background + self.lorentzian(frequency, height, nu_0, gamma)
            model = np.maximum(model, 1e-10)
            
            # Chi-squared likelihood
            log_like = -np.sum(np.log(model) + power / model)
            
            return log_like
        
        samples = np.zeros((n_samples, 3))
        current_params = params_init.copy()
        current_log_prob = log_prob(current_params)
        
        n_accepted = 0
        
        for i in range(n_samples):
            # Propose new parameters
            proposed = current_params + step_sizes * np.random.randn(3)
            proposed_log_prob = log_prob(proposed)
            
            # Accept/reject
            if np.log(np.random.rand()) < proposed_log_prob - current_log_prob:
                current_params = proposed
                current_log_prob = proposed_log_prob
                n_accepted += 1
            
            samples[i] = current_params
        
        # Discard burn-in
        burn_in = n_samples // 4
        return samples[burn_in:]
    
    def fit_all_modes(
        self,
        n_samples: int = 1000
    ) -> List[Dict[str, float]]:
        """
        Fit all detected modes.
        
        Parameters
        ----------
        n_samples : int
            Number of MCMC samples per mode
        
        Returns
        -------
        list
            List of fitted mode parameters
        """
        if not self.modes:
            self.find_peaks()
        
        fitted_modes = []
        
        for mode in self.modes:
            result = self.fit_single_mode(
                mode["frequency"],
                n_samples=n_samples
            )
            
            fitted_modes.append({
                "frequency": result["frequency"],
                "frequency_err": result["frequency_err"],
                "height": result["height"],
                "height_err": result["height_err"],
                "linewidth": result["linewidth"],
                "linewidth_err": result["linewidth_err"]
            })
        
        return fitted_modes
    
    def extract_linewidths_by_degree(
        self,
        mode_identification: Dict[int, Dict[str, int]]
    ) -> Dict[int, List[float]]:
        """
        Extract linewidths grouped by spherical degree.
        
        Parameters
        ----------
        mode_identification : dict
            Dictionary mapping mode index to (n, l) values:
            {mode_index: {"n": radial_order, "l": spherical_degree}}
        
        Returns
        -------
        dict
            Linewidths grouped by spherical degree
        """
        linewidths_by_l = {}
        
        for idx, identification in mode_identification.items():
            l = identification["l"]
            
            if idx < len(self.modes):
                lw = self.modes[idx].get("linewidth", np.nan)
                
                if l not in linewidths_by_l:
                    linewidths_by_l[l] = []
                linewidths_by_l[l].append(lw)
        
        return linewidths_by_l
