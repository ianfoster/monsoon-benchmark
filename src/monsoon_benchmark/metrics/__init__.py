"""Evaluation metrics for monsoon onset forecasts."""

from monsoon_benchmark.metrics.deterministic import (
    compute_mae,
    compute_far,
    compute_mr,
    classify_forecast,
    ForecastClassification,
)
from monsoon_benchmark.metrics.probabilistic import (
    compute_fair_brier_score,
    compute_fair_rps,
    compute_auc,
    compute_reliability_diagram,
    bin_ensemble_onsets,
)
from monsoon_benchmark.metrics.skill_scores import (
    compute_skill_score,
    compute_brier_skill_score,
    compute_rps_skill_score,
)

__all__ = [
    # Deterministic
    "compute_mae",
    "compute_far",
    "compute_mr",
    "classify_forecast",
    "ForecastClassification",
    # Probabilistic
    "compute_fair_brier_score",
    "compute_fair_rps",
    "compute_auc",
    "compute_reliability_diagram",
    "bin_ensemble_onsets",
    # Skill scores
    "compute_skill_score",
    "compute_brier_skill_score",
    "compute_rps_skill_score",
]
