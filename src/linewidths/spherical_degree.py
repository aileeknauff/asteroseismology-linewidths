"""
Spherical Degree Analysis Module

This module provides tools for analyzing the dependence of oscillation mode
linewidths on spherical degree (l) in asteroseismology data.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple


class SphericalDegreeAnalysis:
    """
    Analyze the relationship between oscillation mode linewidths
    and spherical degree.
    
    Parameters
    ----------
    frequencies : array-like
        Array of oscillation frequencies in microHz
    linewidths : array-like
        Array of measured linewidths in microHz
    degrees : array-like
        Array of spherical degree (l) values for each mode
    uncertainties : array-like, optional
        Uncertainties on linewidth measurements
    """
    
    def __init__(
        self,
        frequencies: np.ndarray,
        linewidths: np.ndarray,
        degrees: np.ndarray,
        uncertainties: Optional[np.ndarray] = None
    ):
        self.frequencies = np.asarray(frequencies)
        self.linewidths = np.asarray(linewidths)
        self.degrees = np.asarray(degrees)
        self.uncertainties = np.asarray(uncertainties) if uncertainties is not None else None
        
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        """Validate input array shapes and values."""
        if len(self.frequencies) != len(self.linewidths):
            raise ValueError("Frequencies and linewidths must have the same length")
        if len(self.frequencies) != len(self.degrees):
            raise ValueError("Frequencies and degrees must have the same length")
        if self.uncertainties is not None and len(self.uncertainties) != len(self.frequencies):
            raise ValueError("Uncertainties must have the same length as frequencies")
    
    def group_by_degree(self) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Group data by spherical degree.
        
        Returns
        -------
        dict
            Dictionary with degree as key, containing arrays of frequencies,
            linewidths, and uncertainties for each degree.
        """
        unique_degrees = np.unique(self.degrees)
        grouped = {}
        
        for l in unique_degrees:
            mask = self.degrees == l
            grouped[int(l)] = {
                "frequencies": self.frequencies[mask],
                "linewidths": self.linewidths[mask],
                "uncertainties": self.uncertainties[mask] if self.uncertainties is not None else None
            }
        
        return grouped
    
    def calculate_mean_linewidth_by_degree(self) -> Dict[int, Tuple[float, float]]:
        """
        Calculate mean linewidth and standard error for each spherical degree.
        
        Returns
        -------
        dict
            Dictionary with degree as key and (mean, std_error) tuple as value.
        """
        grouped = self.group_by_degree()
        results = {}
        
        for l, data in grouped.items():
            lw = data["linewidths"]
            mean = np.mean(lw)
            std_error = np.std(lw) / np.sqrt(len(lw)) if len(lw) > 1 else 0.0
            results[l] = (mean, std_error)
        
        return results
    
    def test_degree_dependence(self) -> Dict[str, float]:
        """
        Perform statistical test for spherical degree dependence.
        
        Uses ANOVA (Analysis of Variance) to test whether there are
        statistically significant differences in linewidths between
        different spherical degrees.
        
        Returns
        -------
        dict
            Dictionary containing F-statistic and p-value.
        """
        from scipy import stats
        
        grouped = self.group_by_degree()
        groups = [data["linewidths"] for data in grouped.values()]
        
        if len(groups) < 2:
            return {"f_statistic": np.nan, "p_value": np.nan, "significant": False}
        
        # Perform one-way ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        
        return {
            "f_statistic": f_stat,
            "p_value": p_value,
            "significant": p_value < 0.05
        }
    
    def fit_linewidth_frequency_relation(
        self,
        degree: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Fit a power-law relation between linewidth and frequency.
        
        The relation is: Gamma = a * nu^b
        
        Parameters
        ----------
        degree : int, optional
            If provided, fit only modes of this spherical degree.
            If None, fit all modes.
        
        Returns
        -------
        dict
            Fitted parameters a (amplitude) and b (exponent).
        """
        from scipy import optimize
        
        if degree is not None:
            mask = self.degrees == degree
            freq = self.frequencies[mask]
            lw = self.linewidths[mask]
        else:
            freq = self.frequencies
            lw = self.linewidths
        
        if len(freq) < 2:
            return {"amplitude": np.nan, "exponent": np.nan}
        
        # Fit in log-log space
        def model(log_nu, log_a, b):
            return log_a + b * log_nu
        
        log_freq = np.log10(freq)
        log_lw = np.log10(lw)
        
        try:
            popt, _ = optimize.curve_fit(model, log_freq, log_lw)
            amplitude = 10**popt[0]
            exponent = popt[1]
        except (RuntimeError, ValueError):
            amplitude = np.nan
            exponent = np.nan
        
        return {"amplitude": amplitude, "exponent": exponent}
    
    def compare_degrees(
        self,
        degree1: int,
        degree2: int
    ) -> Dict[str, float]:
        """
        Compare linewidths between two specific spherical degrees.
        
        Parameters
        ----------
        degree1 : int
            First spherical degree
        degree2 : int
            Second spherical degree
        
        Returns
        -------
        dict
            Dictionary containing t-statistic, p-value, and mean difference.
        """
        from scipy import stats
        
        grouped = self.group_by_degree()
        
        if degree1 not in grouped or degree2 not in grouped:
            raise ValueError(f"Degrees {degree1} and/or {degree2} not found in data")
        
        lw1 = grouped[degree1]["linewidths"]
        lw2 = grouped[degree2]["linewidths"]
        
        t_stat, p_value = stats.ttest_ind(lw1, lw2)
        mean_diff = np.mean(lw1) - np.mean(lw2)
        
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "mean_difference": mean_diff,
            "significant": p_value < 0.05
        }
    
    def summary(self) -> str:
        """
        Generate a summary of the spherical degree analysis.
        
        Returns
        -------
        str
            Formatted summary string.
        """
        mean_by_degree = self.calculate_mean_linewidth_by_degree()
        dependence_test = self.test_degree_dependence()
        
        lines = [
            "=" * 50,
            "Spherical Degree Linewidth Analysis Summary",
            "=" * 50,
            f"Total number of modes: {len(self.frequencies)}",
            f"Spherical degrees present: {sorted(mean_by_degree.keys())}",
            "",
            "Mean linewidth by degree:",
        ]
        
        for l, (mean, err) in sorted(mean_by_degree.items()):
            lines.append(f"  l={l}: {mean:.4f} ± {err:.4f} µHz")
        
        lines.extend([
            "",
            "ANOVA test for degree dependence:",
            f"  F-statistic: {dependence_test['f_statistic']:.4f}",
            f"  p-value: {dependence_test['p_value']:.4e}",
            f"  Significant (p < 0.05): {dependence_test['significant']}",
            "=" * 50,
        ])
        
        return "\n".join(lines)
