"""Monsoon onset index computations."""

from monsoon_benchmark.indices.onset import (
    compute_local_onset,
    compute_onset_from_forecast,
    OnsetResult,
)
from monsoon_benchmark.indices.wyi import (
    compute_wyi,
    compute_wyi_onset,
    compute_wyi_climatology,
)
from monsoon_benchmark.indices.climatology import (
    compute_wet_spell_threshold,
    compute_climatological_onset,
    get_climatological_onset_date,
)

__all__ = [
    "compute_local_onset",
    "compute_onset_from_forecast",
    "OnsetResult",
    "compute_wyi",
    "compute_wyi_onset",
    "compute_wyi_climatology",
    "compute_wet_spell_threshold",
    "compute_climatological_onset",
    "get_climatological_onset_date",
]
