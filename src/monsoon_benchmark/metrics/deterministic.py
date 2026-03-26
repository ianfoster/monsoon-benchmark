"""
Deterministic evaluation metrics for monsoon onset forecasts.

Implements:
- Mean Absolute Error (MAE)
- False Alarm Rate (FAR)
- Miss Rate (MR)

These metrics are computed following the methodology in the paper,
with tolerance windows for True Positive classification.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple, Union
import logging

import numpy as np

logger = logging.getLogger(__name__)


class ForecastOutcome(Enum):
    """Classification of a single forecast outcome."""

    TRUE_POSITIVE = "TP"  # Correct onset prediction within tolerance
    FALSE_POSITIVE = "FP"  # Predicted onset but wrong timing or no observed onset
    TRUE_NEGATIVE = "TN"  # Correctly predicted no onset
    FALSE_NEGATIVE = "FN"  # Missed observed onset


@dataclass
class ForecastClassification:
    """Result of classifying a forecast against observation."""

    outcome: ForecastOutcome
    forecast_onset: Optional[datetime]
    observed_onset: Optional[datetime]
    error_days: Optional[float]
    within_tolerance: bool
    tolerance_days: int


def classify_forecast(
    forecast_onset: Optional[datetime],
    observed_onset: Optional[datetime],
    forecast_window: Tuple[int, int],
    init_date: datetime,
    tolerance_days: int = 3,
) -> ForecastClassification:
    """
    Classify a forecast as TP, FP, TN, or FN.

    Classification rules (from paper):
    - TP: Forecast onset within tolerance_days of observed onset
    - FP: Forecast onset but error > tolerance, or no observed onset in window
    - TN: No forecast onset and no observed onset in window
    - FN: No forecast onset but observed onset occurred in window

    Parameters
    ----------
    forecast_onset : datetime or None
        Predicted onset date
    observed_onset : datetime or None
        Actual onset date (None if no onset occurred)
    forecast_window : tuple
        (start_day, end_day) of forecast window
    init_date : datetime
        Forecast initialization date
    tolerance_days : int
        Tolerance for true positive classification

    Returns
    -------
    ForecastClassification
        Classification result with metadata
    """
    # Compute forecast window boundaries
    window_start = init_date + np.timedelta64(forecast_window[0], "D")
    window_end = init_date + np.timedelta64(forecast_window[1], "D")

    # Check if observed onset is within window
    obs_in_window = False
    if observed_onset is not None:
        obs_in_window = window_start <= observed_onset <= window_end

    # Compute error if both exist
    error_days = None
    if forecast_onset is not None and observed_onset is not None:
        error_days = (forecast_onset - observed_onset).total_seconds() / 86400

    within_tolerance = False
    if error_days is not None:
        within_tolerance = abs(error_days) <= tolerance_days

    # Classification logic
    if forecast_onset is not None:
        if observed_onset is not None and within_tolerance:
            outcome = ForecastOutcome.TRUE_POSITIVE
        else:
            outcome = ForecastOutcome.FALSE_POSITIVE
    else:
        if obs_in_window:
            outcome = ForecastOutcome.FALSE_NEGATIVE
        else:
            outcome = ForecastOutcome.TRUE_NEGATIVE

    return ForecastClassification(
        outcome=outcome,
        forecast_onset=forecast_onset,
        observed_onset=observed_onset,
        error_days=error_days,
        within_tolerance=within_tolerance,
        tolerance_days=tolerance_days,
    )


def compute_mae(
    forecast_onsets: List[Optional[datetime]],
    observed_onsets: List[Optional[datetime]],
    exclude_none: bool = True,
) -> float:
    """
    Compute Mean Absolute Error in days.

    MAE = (1/n) * Σ|forecast_onset - observed_onset|

    Parameters
    ----------
    forecast_onsets : list
        List of forecast onset dates (None for no forecast)
    observed_onsets : list
        List of observed onset dates (None for no onset)
    exclude_none : bool
        If True, exclude cases where either is None

    Returns
    -------
    float
        MAE in days
    """
    errors = []

    for f, o in zip(forecast_onsets, observed_onsets):
        if f is not None and o is not None:
            error = abs((f - o).total_seconds() / 86400)
            errors.append(error)
        elif not exclude_none:
            # Handle missing forecasts/observations
            # Could assign a large penalty error
            pass

    if not errors:
        return np.nan

    return np.mean(errors)


def compute_far(
    classifications: List[ForecastClassification],
) -> float:
    """
    Compute False Alarm Rate.

    FAR = FP / (FP + TN)

    This measures how often the model incorrectly predicts onset
    when there is no actual onset in the window.

    Parameters
    ----------
    classifications : list
        List of ForecastClassification results

    Returns
    -------
    float
        FAR as proportion (0 to 1)
    """
    fp = sum(1 for c in classifications if c.outcome == ForecastOutcome.FALSE_POSITIVE)
    tn = sum(1 for c in classifications if c.outcome == ForecastOutcome.TRUE_NEGATIVE)

    denominator = fp + tn
    if denominator == 0:
        return 0.0

    return fp / denominator


def compute_mr(
    classifications: List[ForecastClassification],
) -> float:
    """
    Compute Miss Rate.

    MR = FN / (total onsets in forecast window)

    This measures how often the model fails to predict onset
    when onset actually occurred.

    Parameters
    ----------
    classifications : list
        List of ForecastClassification results

    Returns
    -------
    float
        MR as proportion (0 to 1)
    """
    fn = sum(1 for c in classifications if c.outcome == ForecastOutcome.FALSE_NEGATIVE)

    # Total true onsets = TP + FN + FP (where FP had wrong timing)
    # But simpler: count cases where observed onset was in window
    total_onsets = sum(
        1
        for c in classifications
        if c.observed_onset is not None
        and c.outcome in [ForecastOutcome.TRUE_POSITIVE, ForecastOutcome.FALSE_NEGATIVE]
    )

    # Also count FP cases where there was an observed onset (wrong timing)
    total_onsets += sum(
        1
        for c in classifications
        if c.outcome == ForecastOutcome.FALSE_POSITIVE and c.observed_onset is not None
    )

    if total_onsets == 0:
        return 0.0

    return fn / total_onsets


def compute_metrics_for_period(
    forecast_results: List[dict],
    tolerance_days: int = 3,
) -> dict:
    """
    Compute all deterministic metrics for a set of forecasts.

    Parameters
    ----------
    forecast_results : list
        List of dicts with keys: forecast_onset, observed_onset, init_date, window
    tolerance_days : int
        Tolerance for TP classification

    Returns
    -------
    dict
        Dictionary with MAE, FAR, MR, and counts
    """
    classifications = []
    forecast_onsets = []
    observed_onsets = []

    for result in forecast_results:
        clf = classify_forecast(
            forecast_onset=result["forecast_onset"],
            observed_onset=result["observed_onset"],
            forecast_window=result["window"],
            init_date=result["init_date"],
            tolerance_days=tolerance_days,
        )
        classifications.append(clf)
        forecast_onsets.append(result["forecast_onset"])
        observed_onsets.append(result["observed_onset"])

    # Count outcomes
    tp = sum(1 for c in classifications if c.outcome == ForecastOutcome.TRUE_POSITIVE)
    fp = sum(1 for c in classifications if c.outcome == ForecastOutcome.FALSE_POSITIVE)
    tn = sum(1 for c in classifications if c.outcome == ForecastOutcome.TRUE_NEGATIVE)
    fn = sum(1 for c in classifications if c.outcome == ForecastOutcome.FALSE_NEGATIVE)

    return {
        "mae": compute_mae(forecast_onsets, observed_onsets),
        "far": compute_far(classifications),
        "mr": compute_mr(classifications),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_forecasts": len(classifications),
    }


def compute_total_miss_rate(
    yearly_results: dict,
    cmz_cells: List[Tuple[float, float]],
) -> float:
    """
    Compute total miss rate across CMZ for years with no onset detected.

    This is the percentage of years where no onset was detected for any
    initialization from May 2 until the true onset date.

    Parameters
    ----------
    yearly_results : dict
        Dictionary of {year: {(lat, lon): has_onset}}
    cmz_cells : list
        List of (lat, lon) tuples for CMZ grid cells

    Returns
    -------
    float
        Total miss rate as percentage
    """
    n_cells = len(cmz_cells)
    n_years = len(yearly_results)

    total_possible = n_cells * n_years
    total_misses = 0

    for year, cell_results in yearly_results.items():
        for cell in cmz_cells:
            if cell not in cell_results or not cell_results[cell]:
                total_misses += 1

    return 100 * total_misses / total_possible if total_possible > 0 else 0.0


def aggregate_metrics_cmz(
    cell_metrics: dict,
    cmz_cells: List[Tuple[float, float]],
) -> dict:
    """
    Aggregate metrics across CMZ grid cells.

    Parameters
    ----------
    cell_metrics : dict
        Dictionary of {(lat, lon): metrics_dict}
    cmz_cells : list
        List of CMZ cell coordinates

    Returns
    -------
    dict
        Aggregated metrics with mean and standard error
    """
    maes = []
    fars = []
    mrs = []

    for cell in cmz_cells:
        if cell in cell_metrics:
            m = cell_metrics[cell]
            if not np.isnan(m["mae"]):
                maes.append(m["mae"])
            fars.append(m["far"])
            mrs.append(m["mr"])

    n = len(maes)
    mae_mean = np.mean(maes) if maes else np.nan
    mae_se = np.std(maes) / np.sqrt(n) if n > 0 else np.nan

    return {
        "mae_mean": mae_mean,
        "mae_se": mae_se,
        "far_mean": np.mean(fars) * 100 if fars else np.nan,  # as percentage
        "mr_mean": np.mean(mrs) * 100 if mrs else np.nan,  # as percentage
        "n_cells": n,
    }
