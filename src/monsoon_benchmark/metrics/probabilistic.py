"""
Probabilistic evaluation metrics for monsoon onset forecasts.

Implements:
- Fair Brier Score (with Ferro et al. adjustment for ensemble size)
- Fair Ranked Probability Score
- Area Under ROC Curve (AUC)
- Reliability diagrams

Reference:
- Ferro, C. (2014). Fair scores for ensemble forecasts.
  Q. J. Royal Meteorol. Soc. 140, 1917-1923.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple
import logging

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


def bin_ensemble_onsets(
    ensemble_onsets: List[Optional[datetime]],
    init_date: datetime,
    bin_size_days: int = 5,
    n_bins: int = 7,
) -> np.ndarray:
    """
    Convert ensemble onset dates to probability distribution over bins.

    Bins:
    - 0: days 1-5 after init
    - 1: days 6-10
    - 2: days 11-15
    - ...
    - n_bins-1: after day (n_bins-1)*bin_size or no onset

    Parameters
    ----------
    ensemble_onsets : list
        List of onset dates (None for no onset)
    init_date : datetime
        Forecast initialization date
    bin_size_days : int
        Size of each bin in days
    n_bins : int
        Number of bins

    Returns
    -------
    np.ndarray
        Probability for each bin (sums to 1)
    """
    n_members = len(ensemble_onsets)
    counts = np.zeros(n_bins)

    for onset in ensemble_onsets:
        if onset is None:
            counts[-1] += 1  # Last bin for "no onset" or "after window"
        else:
            lead_days = (onset - init_date).days
            if lead_days < 1:
                counts[0] += 1
            else:
                bin_idx = (lead_days - 1) // bin_size_days
                bin_idx = min(bin_idx, n_bins - 1)
                counts[bin_idx] += 1

    # Convert to probabilities
    probs = counts / n_members
    return probs


def compute_fair_brier_score(
    forecast_probs: np.ndarray,
    observed: np.ndarray,
    ensemble_size: int,
) -> float:
    """
    Compute Fair Brier Score with Ferro et al. adjustment.

    The fair Brier score accounts for sampling uncertainty in ensemble forecasts:

    BS_fair = (1/n*m) * Σ_i Σ_j [(Y_ij - p_ij)² - p_ij(1-p_ij)/(w-1)]

    where w is the ensemble size.

    Parameters
    ----------
    forecast_probs : np.ndarray
        2D array of forecast probabilities (n_forecasts, n_bins)
    observed : np.ndarray
        2D array of observations (n_forecasts, n_bins), binary 0/1
    ensemble_size : int
        Number of ensemble members

    Returns
    -------
    float
        Fair Brier score (lower is better)
    """
    n_forecasts, n_bins = forecast_probs.shape

    # Standard Brier score component
    squared_error = (observed - forecast_probs) ** 2

    # Ferro adjustment for finite ensemble size
    if ensemble_size > 1:
        adjustment = forecast_probs * (1 - forecast_probs) / (ensemble_size - 1)
    else:
        adjustment = 0

    fair_bs = np.mean(squared_error - adjustment)

    return float(fair_bs)


def compute_fair_rps(
    forecast_probs: np.ndarray,
    observed: np.ndarray,
    ensemble_size: int,
) -> float:
    """
    Compute Fair Ranked Probability Score.

    RPS accounts for the ordering of categories (bins), penalizing
    forecasts that are further from the observed category.

    RPS = (1/n*m) * Σ_i Σ_k [(Σ_{j=1}^k (Y_ij - p_ij))² - adjustment]

    Parameters
    ----------
    forecast_probs : np.ndarray
        2D array of forecast probabilities (n_forecasts, n_bins)
    observed : np.ndarray
        2D array of observations (n_forecasts, n_bins), binary 0/1
    ensemble_size : int
        Number of ensemble members

    Returns
    -------
    float
        Fair RPS (lower is better)
    """
    n_forecasts, n_bins = forecast_probs.shape

    rps_sum = 0.0

    for i in range(n_forecasts):
        for k in range(n_bins):
            # Cumulative probabilities up to bin k
            cum_fcst = np.sum(forecast_probs[i, : k + 1])
            cum_obs = np.sum(observed[i, : k + 1])

            # Squared error of cumulative
            sq_error = (cum_obs - cum_fcst) ** 2

            # Ferro adjustment
            if ensemble_size > 1:
                adjustment = cum_fcst * (1 - cum_fcst) / (ensemble_size - 1)
            else:
                adjustment = 0

            rps_sum += sq_error - adjustment

    fair_rps = rps_sum / (n_forecasts * n_bins)

    return float(fair_rps)


def compute_auc(
    forecast_probs: np.ndarray,
    observed: np.ndarray,
) -> float:
    """
    Compute Area Under ROC Curve.

    The AUC measures the probability that a randomly selected positive
    example (onset in bin) is ranked higher than a randomly selected
    negative example.

    Parameters
    ----------
    forecast_probs : np.ndarray
        2D array of forecast probabilities (n_forecasts, n_bins)
    observed : np.ndarray
        2D array of observations (n_forecasts, n_bins), binary 0/1

    Returns
    -------
    float
        AUC (higher is better, 0.5 = random, 1.0 = perfect)
    """
    # Flatten arrays
    probs_flat = forecast_probs.flatten()
    obs_flat = observed.flatten()

    # Handle edge cases
    if len(np.unique(obs_flat)) < 2:
        logger.warning("Cannot compute AUC: only one class present")
        return 0.5

    try:
        auc = roc_auc_score(obs_flat, probs_flat)
    except ValueError as e:
        logger.warning(f"AUC computation failed: {e}")
        auc = 0.5

    return float(auc)


@dataclass
class ReliabilityBin:
    """Single bin in reliability diagram."""

    bin_center: float
    forecast_prob_mean: float
    observed_frequency: float
    n_samples: int
    std_error: float


def compute_reliability_diagram(
    forecast_probs: np.ndarray,
    observed: np.ndarray,
    n_bins: int = 10,
) -> List[ReliabilityBin]:
    """
    Compute reliability diagram data.

    The reliability diagram plots forecast probability against
    observed frequency. A perfectly calibrated forecast lies on
    the diagonal.

    Parameters
    ----------
    forecast_probs : np.ndarray
        2D array of forecast probabilities (n_forecasts, n_onset_bins)
    observed : np.ndarray
        2D array of observations (n_forecasts, n_onset_bins)
    n_bins : int
        Number of bins for probability axis

    Returns
    -------
    list of ReliabilityBin
        Reliability diagram data
    """
    # Flatten arrays
    probs_flat = forecast_probs.flatten()
    obs_flat = observed.flatten()

    # Define probability bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    results = []

    for i in range(n_bins):
        # Select forecasts in this probability bin
        mask = (probs_flat >= bin_edges[i]) & (probs_flat < bin_edges[i + 1])

        if i == n_bins - 1:  # Include right edge for last bin
            mask = (probs_flat >= bin_edges[i]) & (probs_flat <= bin_edges[i + 1])

        n_samples = np.sum(mask)

        if n_samples > 0:
            probs_in_bin = probs_flat[mask]
            obs_in_bin = obs_flat[mask]

            forecast_prob_mean = np.mean(probs_in_bin)
            observed_freq = np.mean(obs_in_bin)

            # Standard error of observed frequency
            if n_samples > 1:
                std_error = np.sqrt(observed_freq * (1 - observed_freq) / n_samples)
            else:
                std_error = 0.5
        else:
            forecast_prob_mean = bin_centers[i]
            observed_freq = np.nan
            std_error = np.nan

        results.append(
            ReliabilityBin(
                bin_center=bin_centers[i],
                forecast_prob_mean=forecast_prob_mean,
                observed_frequency=observed_freq,
                n_samples=n_samples,
                std_error=std_error,
            )
        )

    return results


def compute_probabilistic_metrics(
    ensemble_forecasts: List[List[Optional[datetime]]],
    observed_onsets: List[Optional[datetime]],
    init_dates: List[datetime],
    bin_size_days: int = 5,
    n_bins: int = 7,
) -> dict:
    """
    Compute all probabilistic metrics for a set of ensemble forecasts.

    Parameters
    ----------
    ensemble_forecasts : list of list
        Ensemble onset forecasts for each initialization
    observed_onsets : list
        Observed onset dates
    init_dates : list
        Initialization dates
    bin_size_days : int
        Size of probability bins
    n_bins : int
        Number of bins

    Returns
    -------
    dict
        Dictionary with BS, RPS, AUC, and reliability data
    """
    n_forecasts = len(ensemble_forecasts)
    ensemble_size = len(ensemble_forecasts[0]) if ensemble_forecasts else 1

    # Build probability and observation arrays
    forecast_probs = np.zeros((n_forecasts, n_bins))
    observed = np.zeros((n_forecasts, n_bins))

    for i, (ens_fcst, obs, init) in enumerate(
        zip(ensemble_forecasts, observed_onsets, init_dates)
    ):
        # Convert ensemble to probabilities
        forecast_probs[i] = bin_ensemble_onsets(
            ens_fcst, init, bin_size_days, n_bins
        )

        # Convert observation to one-hot
        if obs is None:
            observed[i, -1] = 1  # Last bin for no onset
        else:
            lead_days = (obs - init).days
            if lead_days < 1:
                observed[i, 0] = 1
            else:
                bin_idx = min((lead_days - 1) // bin_size_days, n_bins - 1)
                observed[i, bin_idx] = 1

    # Compute metrics
    bs = compute_fair_brier_score(forecast_probs, observed, ensemble_size)
    rps = compute_fair_rps(forecast_probs, observed, ensemble_size)
    auc = compute_auc(forecast_probs, observed)
    reliability = compute_reliability_diagram(forecast_probs, observed)

    return {
        "brier_score": bs,
        "rps": rps,
        "auc": auc,
        "reliability": reliability,
        "n_forecasts": n_forecasts,
        "ensemble_size": ensemble_size,
    }


def compute_climatology_probabilistic(
    historical_onsets: List[Optional[datetime]],
    init_date: datetime,
    bin_size_days: int = 5,
    n_bins: int = 7,
) -> np.ndarray:
    """
    Compute climatological probability distribution.

    Uses historical onset dates to create a probability distribution
    over bins, treating each historical year as an ensemble member.

    Parameters
    ----------
    historical_onsets : list
        Historical onset dates (from climatology record)
    init_date : datetime
        Initialization date (to compute lead times)
    bin_size_days : int
        Size of bins
    n_bins : int
        Number of bins

    Returns
    -------
    np.ndarray
        Climatological probability distribution
    """
    return bin_ensemble_onsets(
        historical_onsets, init_date, bin_size_days, n_bins
    )
