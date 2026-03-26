"""
Skill score computations for monsoon onset forecasts.

Skill scores measure improvement relative to a reference forecast
(typically climatology).

Skill = 1 - (metric_forecast / metric_reference)
"""

from typing import Optional
import numpy as np


def compute_skill_score(
    metric_forecast: float,
    metric_reference: float,
    perfect_score: float = 0.0,
) -> float:
    """
    Compute generic skill score.

    Skill = 1 - (metric_forecast / metric_reference)

    For metrics where lower is better (MAE, Brier, RPS):
    - Skill > 0 means better than reference
    - Skill = 0 means same as reference
    - Skill < 0 means worse than reference
    - Skill = 1 means perfect

    Parameters
    ----------
    metric_forecast : float
        Metric value for forecast
    metric_reference : float
        Metric value for reference (climatology)
    perfect_score : float
        Perfect score for the metric (default 0 for error metrics)

    Returns
    -------
    float
        Skill score
    """
    if np.isclose(metric_reference, perfect_score):
        # Reference is already perfect
        return 0.0 if np.isclose(metric_forecast, perfect_score) else -np.inf

    skill = 1 - (metric_forecast - perfect_score) / (metric_reference - perfect_score)

    return float(skill)


def compute_brier_skill_score(
    brier_forecast: float,
    brier_climatology: float,
) -> float:
    """
    Compute Brier Skill Score (BSS).

    BSS = 1 - BS_forecast / BS_climatology

    Parameters
    ----------
    brier_forecast : float
        Brier score of forecast
    brier_climatology : float
        Brier score of climatological forecast

    Returns
    -------
    float
        BSS (positive = better than climatology)
    """
    if brier_climatology == 0:
        return 0.0 if brier_forecast == 0 else -np.inf

    return 1 - brier_forecast / brier_climatology


def compute_rps_skill_score(
    rps_forecast: float,
    rps_climatology: float,
) -> float:
    """
    Compute Ranked Probability Skill Score (RPSS).

    RPSS = 1 - RPS_forecast / RPS_climatology

    Parameters
    ----------
    rps_forecast : float
        RPS of forecast
    rps_climatology : float
        RPS of climatological forecast

    Returns
    -------
    float
        RPSS (positive = better than climatology)
    """
    if rps_climatology == 0:
        return 0.0 if rps_forecast == 0 else -np.inf

    return 1 - rps_forecast / rps_climatology


def compute_mae_skill_score(
    mae_forecast: float,
    mae_climatology: float,
) -> float:
    """
    Compute MAE-based skill score.

    Parameters
    ----------
    mae_forecast : float
        MAE of forecast (days)
    mae_climatology : float
        MAE of climatological forecast (days)

    Returns
    -------
    float
        Skill score (positive = better than climatology)
    """
    if mae_climatology == 0:
        return 0.0 if mae_forecast == 0 else -np.inf

    return 1 - mae_forecast / mae_climatology


def compute_all_skill_scores(
    forecast_metrics: dict,
    climatology_metrics: dict,
) -> dict:
    """
    Compute all skill scores given forecast and climatology metrics.

    Parameters
    ----------
    forecast_metrics : dict
        Dictionary with mae, brier_score, rps, auc
    climatology_metrics : dict
        Dictionary with same keys for climatology

    Returns
    -------
    dict
        Dictionary of skill scores
    """
    skill_scores = {}

    # MAE skill score
    if "mae" in forecast_metrics and "mae" in climatology_metrics:
        skill_scores["mae_skill"] = compute_mae_skill_score(
            forecast_metrics["mae"],
            climatology_metrics["mae"],
        )

    # Brier Skill Score
    if "brier_score" in forecast_metrics and "brier_score" in climatology_metrics:
        skill_scores["bss"] = compute_brier_skill_score(
            forecast_metrics["brier_score"],
            climatology_metrics["brier_score"],
        )

    # Ranked Probability Skill Score
    if "rps" in forecast_metrics and "rps" in climatology_metrics:
        skill_scores["rpss"] = compute_rps_skill_score(
            forecast_metrics["rps"],
            climatology_metrics["rps"],
        )

    # AUC (not a skill score, but include difference from climatology)
    if "auc" in forecast_metrics and "auc" in climatology_metrics:
        skill_scores["auc"] = forecast_metrics["auc"]
        skill_scores["auc_clim"] = climatology_metrics["auc"]
        skill_scores["auc_diff"] = (
            forecast_metrics["auc"] - climatology_metrics["auc"]
        )

    return skill_scores


def skill_score_significance(
    skill_score: float,
    n_samples: int,
    confidence_level: float = 0.95,
) -> dict:
    """
    Estimate significance of skill score using bootstrap-like approach.

    This is a simplified approach. Full bootstrap would require
    recomputing metrics over resampled data.

    Parameters
    ----------
    skill_score : float
        Computed skill score
    n_samples : int
        Number of samples used to compute skill
    confidence_level : float
        Confidence level for interval

    Returns
    -------
    dict
        Dictionary with standard error and confidence bounds
    """
    from scipy import stats

    # Approximate standard error (simplified)
    # This assumes skill scores have approximately normal sampling distribution
    se = np.sqrt((1 - skill_score**2) / n_samples) if n_samples > 1 else np.inf

    # Confidence interval
    z = stats.norm.ppf((1 + confidence_level) / 2)
    ci_lower = skill_score - z * se
    ci_upper = skill_score + z * se

    # Check if significantly different from zero
    significant = ci_lower > 0 or ci_upper < 0

    return {
        "skill_score": skill_score,
        "standard_error": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": significant,
        "confidence_level": confidence_level,
    }
