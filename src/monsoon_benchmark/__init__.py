"""
Monsoon Benchmark: Decision-oriented benchmarking for AI weather prediction models.

This package implements the framework described in:
Masiwal et al., "Decision-oriented benchmarking to transform AI weather forecast access:
Application to the Indian monsoon"
https://arxiv.org/abs/2602.03767

Main components:
- data: Data loading and preprocessing (IMD, ERA5, regridding)
- indices: Monsoon onset indices (Moron-Robertson, Webster-Yang)
- metrics: Evaluation metrics (deterministic and probabilistic)
- evaluation: Benchmarking framework
- models: AIWP model wrappers
- visualization: Plotting utilities
"""

__version__ = "0.1.0"

from monsoon_benchmark.indices.onset import compute_local_onset
from monsoon_benchmark.indices.wyi import compute_wyi, compute_wyi_onset
from monsoon_benchmark.metrics.deterministic import compute_mae, compute_far, compute_mr
from monsoon_benchmark.metrics.probabilistic import (
    compute_fair_brier_score,
    compute_fair_rps,
    compute_auc,
)

__all__ = [
    "compute_local_onset",
    "compute_wyi",
    "compute_wyi_onset",
    "compute_mae",
    "compute_far",
    "compute_mr",
    "compute_fair_brier_score",
    "compute_fair_rps",
    "compute_auc",
]
