"""
Analysis period definitions for monsoon onset benchmarking.

The paper evaluates models over multiple periods to address small sample sizes:
- Recent test period (2019-2024): Out-of-sample for most AIWP models
- Extended period (1965-1978 + 2019-2024): Pre-satellite + recent OOS
- Common period (2004-2021): Shared period for all models
- All available (1965-2024): Full historical period
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AnalysisPeriod:
    """Definition of an analysis period."""

    name: str
    year_ranges: List[Tuple[int, int]]  # List of (start_year, end_year) tuples
    description: str
    is_out_of_sample: bool = True

    def get_years(self) -> List[int]:
        """Get list of all years in this period."""
        years = []
        for start, end in self.year_ranges:
            years.extend(range(start, end + 1))
        return sorted(years)

    def contains_year(self, year: int) -> bool:
        """Check if year is in this period."""
        for start, end in self.year_ranges:
            if start <= year <= end:
                return True
        return False


# Standard analysis periods from the paper
ANALYSIS_PERIODS = {
    "recent_test": AnalysisPeriod(
        name="Recent Test Period",
        year_ranges=[(2019, 2024)],
        description="Out-of-sample period for most AIWP models",
        is_out_of_sample=True,
    ),
    "extended": AnalysisPeriod(
        name="Extended Period",
        year_ranges=[(1965, 1978), (2019, 2024)],
        description="Pre-satellite + recent out-of-sample years",
        is_out_of_sample=True,
    ),
    "common": AnalysisPeriod(
        name="Common Period",
        year_ranges=[(2004, 2021)],
        description="Shared period for IFS and AIWP models",
        is_out_of_sample=False,  # Overlaps with training for some models
    ),
    "all_available": AnalysisPeriod(
        name="All Available Years",
        year_ranges=[(1965, 2024)],
        description="Full historical period (includes training years)",
        is_out_of_sample=False,
    ),
}


def get_period_years(period_name: str) -> List[int]:
    """
    Get list of years for a named analysis period.

    Parameters
    ----------
    period_name : str
        Name of period (recent_test, extended, common, all_available)

    Returns
    -------
    list of int
        Years in the period
    """
    if period_name not in ANALYSIS_PERIODS:
        raise ValueError(
            f"Unknown period: {period_name}. "
            f"Available: {list(ANALYSIS_PERIODS.keys())}"
        )

    return ANALYSIS_PERIODS[period_name].get_years()


def generate_init_dates(
    year: int,
    frequency: str = "twice_weekly",
    start_date: str = "05-02",
    end_date: Optional[str] = None,
) -> List[datetime]:
    """
    Generate forecast initialization dates for a year.

    The paper uses twice-weekly initialization (Monday and Thursday)
    starting from May 2nd.

    Parameters
    ----------
    year : int
        Year to generate dates for
    frequency : str
        Initialization frequency: "twice_weekly", "weekly", "daily"
    start_date : str
        Start date in MM-DD format (default "05-02")
    end_date : str, optional
        End date in MM-DD format (default: end of September)

    Returns
    -------
    list of datetime
        Initialization dates
    """
    start_month, start_day = map(int, start_date.split("-"))
    start = datetime(year, start_month, start_day)

    if end_date is not None:
        end_month, end_day = map(int, end_date.split("-"))
        end = datetime(year, end_month, end_day)
    else:
        end = datetime(year, 9, 30)  # End of September

    dates = []

    if frequency == "twice_weekly":
        # Monday (0) and Thursday (3)
        target_weekdays = [0, 3]
        current = start
        while current <= end:
            if current.weekday() in target_weekdays:
                dates.append(current)
            current += timedelta(days=1)

    elif frequency == "weekly":
        # Every Monday
        current = start
        while current <= end:
            if current.weekday() == 0:
                dates.append(current)
            current += timedelta(days=1)

    elif frequency == "daily":
        current = start
        while current <= end:
            dates.append(current)
            current += timedelta(days=1)

    else:
        raise ValueError(f"Unknown frequency: {frequency}")

    return dates


def get_init_dates_before_onset(
    init_dates: List[datetime],
    observed_onset: datetime,
) -> List[datetime]:
    """
    Filter initialization dates to those before observed onset.

    For evaluation, we only consider forecasts initialized before
    the actual onset occurred.

    Parameters
    ----------
    init_dates : list of datetime
        All possible initialization dates
    observed_onset : datetime
        Observed onset date

    Returns
    -------
    list of datetime
        Initialization dates before onset
    """
    return [d for d in init_dates if d < observed_onset]


def compute_lead_time(
    init_date: datetime,
    target_date: datetime,
) -> int:
    """
    Compute lead time in days.

    Parameters
    ----------
    init_date : datetime
        Forecast initialization date
    target_date : datetime
        Target/verification date

    Returns
    -------
    int
        Lead time in days
    """
    return (target_date - init_date).days


def categorize_forecast_window(
    lead_time: int,
) -> str:
    """
    Categorize lead time into forecast window.

    Parameters
    ----------
    lead_time : int
        Lead time in days

    Returns
    -------
    str
        Window category: "medium_range" (1-15) or "subseasonal" (16-30)
    """
    if 1 <= lead_time <= 15:
        return "medium_range"
    elif 16 <= lead_time <= 30:
        return "subseasonal"
    elif lead_time > 30:
        return "extended"
    else:
        return "nowcast"


# Model-specific out-of-sample periods
MODEL_OOS_PERIODS = {
    "ifs": {
        "training_end": None,  # NWP, not ML-trained
        "hindcast_available": (2004, 2023),
    },
    "aifs": {
        "training_end": 2022,
        "finetuning_end": 2022,
        "out_of_sample_start": 2023,
    },
    "fuxi": {
        "training_end": 2017,
        "out_of_sample_start": 2018,
    },
    "graphcast": {
        "training_end": 2017,
        "out_of_sample_start": 2018,
    },
    "gencast": {
        "training_end": 2018,
        "out_of_sample_start": 2019,
    },
    "fuxi_s2s": {
        "training_end": 2016,
        "out_of_sample_start": 2017,
        "hindcast_available": (2002, 2021),
    },
    "neuralgcm": {
        "training_end": 2018,
        "out_of_sample_start": 2019,
    },
}


def is_out_of_sample(model: str, year: int) -> bool:
    """
    Check if a year is out-of-sample for a specific model.

    Parameters
    ----------
    model : str
        Model name
    year : int
        Year to check

    Returns
    -------
    bool
        True if year is out of training sample
    """
    if model not in MODEL_OOS_PERIODS:
        logger.warning(f"Unknown model: {model}")
        return True

    info = MODEL_OOS_PERIODS[model]

    if info.get("training_end") is None:
        return True  # NWP model

    oos_start = info.get("out_of_sample_start", info["training_end"] + 1)

    # Also consider years before training as OOS (pre-satellite era)
    # Most models trained from 1979 onwards
    training_start = 1979

    return year >= oos_start or year < training_start
