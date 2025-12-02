"""
Signal to Background Ratio (SBR) Module

This module provides tools for calculating the Signal to Background Noise
ratio in stellar power spectra, essential for asteroseismology analysis.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit
from typing import Optional, Tuple


def estimate_background(
    frequency: np.ndarray,
    power: np.ndarray,
    method: str = "harvey"
) -> np.ndarray:
    """
    Estimate the background power spectrum.
    
    Parameters
    ----------
    frequency : array-like
        Frequency array in microHz
    power : array-like
        Power spectral density
    method : str
        Method for background estimation. Options:
        - "harvey": Harvey model for granulation background
        - "smooth": Smoothed background using moving average
    
    Returns
    -------
    np.ndarray
        Estimated background power spectrum
    """
    frequency = np.asarray(frequency)
    power = np.asarray(power)
    
    if method == "smooth":
        return _smooth_background(frequency, power)
    elif method == "harvey":
        return _harvey_background(frequency, power)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'harvey' or 'smooth'.")


def _smooth_background(
    frequency: np.ndarray,
    power: np.ndarray,
    window_size: int = 50
) -> np.ndarray:
    """
    Estimate background using moving average smoothing.
    
    Parameters
    ----------
    frequency : np.ndarray
        Frequency array
    power : np.ndarray
        Power spectral density
    window_size : int
        Window size for moving average
    
    Returns
    -------
    np.ndarray
        Smoothed background estimate
    """
    # Work in log space for better handling of dynamic range
    log_power = np.log10(power + 1e-10)
    smoothed = uniform_filter1d(log_power, size=window_size, mode="reflect")
    
    return 10**smoothed


def _harvey_background(
    frequency: np.ndarray,
    power: np.ndarray
) -> np.ndarray:
    """
    Estimate background using Harvey model fit.
    
    The Harvey model represents stellar granulation as:
    B(nu) = sum_i (a_i / (1 + (nu/b_i)^c_i)) + white_noise
    
    Parameters
    ----------
    frequency : np.ndarray
        Frequency array
    power : np.ndarray
        Power spectral density
    
    Returns
    -------
    np.ndarray
        Harvey model background estimate
    """
    def harvey_model(nu, a1, b1, a2, b2, white):
        """Two-component Harvey model with white noise."""
        # Avoid division by zero
        nu = np.maximum(nu, 1e-10)
        
        component1 = a1 / (1 + (nu / b1)**4)
        component2 = a2 / (1 + (nu / b2)**4)
        
        return component1 + component2 + white
    
    # Initial parameter guess
    max_power = np.max(power)
    p0 = [max_power * 0.5, 100, max_power * 0.1, 1000, np.min(power)]
    
    try:
        # Fit in log space for numerical stability
        popt, _ = curve_fit(
            harvey_model,
            frequency,
            power,
            p0=p0,
            bounds=([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf]),
            maxfev=5000
        )
        background = harvey_model(frequency, *popt)
    except (RuntimeError, ValueError):
        # Fall back to smooth background if fit fails
        background = _smooth_background(frequency, power)
    
    return background


def calculate_sbr(
    frequency: np.ndarray,
    power: np.ndarray,
    background: Optional[np.ndarray] = None,
    nu_max: Optional[float] = None,
    delta_nu: Optional[float] = None
) -> dict:
    """
    Calculate Signal to Background Noise Ratio.
    
    Parameters
    ----------
    frequency : array-like
        Frequency array in microHz
    power : array-like
        Power spectral density
    background : array-like, optional
        Pre-computed background. If None, will be estimated.
    nu_max : float, optional
        Frequency of maximum oscillation power in microHz.
        If provided, SBR is calculated at this frequency.
    delta_nu : float, optional
        Large frequency separation in microHz.
        If provided along with nu_max, defines the oscillation range.
    
    Returns
    -------
    dict
        Dictionary containing:
        - sbr: Signal to background ratio
        - signal_power: Mean signal power
        - background_power: Mean background power
        - snr: Signal to noise ratio per mode
    """
    frequency = np.asarray(frequency)
    power = np.asarray(power)
    
    # Estimate background if not provided
    if background is None:
        background = estimate_background(frequency, power)
    
    # Signal is power minus background
    signal = power - background
    
    # If nu_max and delta_nu are provided, focus on oscillation range
    if nu_max is not None and delta_nu is not None:
        # Typical oscillation range is ±5 delta_nu around nu_max
        range_low = nu_max - 5 * delta_nu
        range_high = nu_max + 5 * delta_nu
        
        mask = (frequency >= range_low) & (frequency <= range_high)
        signal_region = signal[mask]
        background_region = background[mask]
    else:
        signal_region = signal
        background_region = background
    
    # Calculate mean values
    mean_signal = np.mean(np.maximum(signal_region, 0))
    mean_background = np.mean(background_region)
    
    # Signal to Background Ratio
    sbr = mean_signal / mean_background if mean_background > 0 else np.inf
    
    # Estimate mode height (approximately 2/3 of peak signal)
    mode_height = np.percentile(signal_region[signal_region > 0], 90) if np.any(signal_region > 0) else 0
    
    # Signal to Noise Ratio per mode
    # Assuming Gaussian noise with variance equal to background
    snr = mode_height / np.sqrt(mean_background) if mean_background > 0 else np.inf
    
    return {
        "sbr": sbr,
        "signal_power": mean_signal,
        "background_power": mean_background,
        "snr": snr,
        "background": background
    }


def identify_oscillation_envelope(
    frequency: np.ndarray,
    power: np.ndarray
) -> Tuple[float, float, float]:
    """
    Identify the oscillation envelope parameters.
    
    Uses Gaussian envelope fitting to find nu_max, envelope width,
    and amplitude.
    
    Parameters
    ----------
    frequency : array-like
        Frequency array in microHz
    power : array-like
        Power spectral density
    
    Returns
    -------
    tuple
        (nu_max, envelope_width, amplitude)
    """
    frequency = np.asarray(frequency)
    power = np.asarray(power)
    
    # Estimate background and subtract
    background = estimate_background(frequency, power, method="smooth")
    signal = np.maximum(power - background, 0)
    
    # Smooth the signal
    smoothed = uniform_filter1d(signal, size=20)
    
    # Initial guess for nu_max from maximum power
    nu_max_guess = frequency[np.argmax(smoothed)]
    
    def gaussian_envelope(nu, a, nu_max, sigma):
        """Gaussian envelope model."""
        return a * np.exp(-0.5 * ((nu - nu_max) / sigma)**2)
    
    try:
        popt, _ = curve_fit(
            gaussian_envelope,
            frequency,
            smoothed,
            p0=[np.max(smoothed), nu_max_guess, 0.1 * nu_max_guess],
            bounds=([0, frequency.min(), 0], [np.inf, frequency.max(), frequency.max()])
        )
        amplitude, nu_max, envelope_width = popt
    except (RuntimeError, ValueError):
        amplitude = np.max(smoothed)
        nu_max = nu_max_guess
        envelope_width = 0.1 * nu_max
    
    return nu_max, envelope_width, amplitude
