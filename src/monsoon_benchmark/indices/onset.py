"""
Modified Moron-Robertson monsoon onset index.

This implements the agriculturally relevant local onset definition from the paper.

Key modifications from original Moron & Robertson (2014):
1. Uses median MOK date (June 2) as filter instead of 30-day dry spell check
2. Local onset = first wet spell after June 2
3. Evaluated on 4° × 4° grid

Reference:
- Moron, V. & Robertson, A. W. (2014). Interannual variability of Indian summer
  monsoon rainfall onset date at local scale. Int. J. Climatol. 34.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple, Union, List
import logging

import numpy as np
import xarray as xr
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OnsetResult:
    """Result of onset detection for a single location/year."""

    onset_date: Optional[datetime]
    onset_doy: Optional[int]  # Day of year
    wet_spell_start: Optional[datetime]
    wet_spell_total_mm: Optional[float]
    wet_spell_threshold_mm: float
    is_valid: bool
    message: str = ""

    def days_from_reference(self, ref_date: datetime) -> Optional[int]:
        """Compute days from reference date (e.g., June 2)."""
        if self.onset_date is None:
            return None
        return (self.onset_date - ref_date).days


def compute_local_onset(
    precip: xr.DataArray,
    wet_spell_threshold: Union[float, xr.DataArray],
    wet_day_threshold_mm: float = 1.0,
    wet_spell_days: int = 5,
    mok_date: str = "06-02",
    search_start: str = "04-01",
    year: Optional[int] = None,
) -> OnsetResult:
    """
    Compute local monsoon onset date using modified Moron-Robertson index.

    The onset is defined as the first day of the first 5-day wet spell that:
    1. Starts on or after the median MOK date (June 2)
    2. Has all days with precipitation >= wet_day_threshold
    3. Has total accumulation >= local climatological wet spell threshold

    Parameters
    ----------
    precip : xr.DataArray
        Daily precipitation time series (mm/day) for a single location
    wet_spell_threshold : float or xr.DataArray
        Minimum total precipitation for wet spell (mm)
        Can be a scalar or spatially varying array
    wet_day_threshold_mm : float
        Minimum precipitation for a wet day (default 1.0 mm)
    wet_spell_days : int
        Duration of wet spell to detect (default 5 days)
    mok_date : str
        Median MOK date in MM-DD format (default "06-02")
    search_start : str
        Start of search window in MM-DD format (default "04-01")
    year : int, optional
        Year for onset detection (inferred from data if not provided)

    Returns
    -------
    OnsetResult
        Onset detection result with date and metadata
    """
    # Infer year if not provided
    if year is None:
        years = np.unique(precip.time.dt.year.values)
        if len(years) != 1:
            raise ValueError(
                f"Data contains multiple years: {years}. Please specify year."
            )
        year = int(years[0])

    # Parse MOK date
    mok_month, mok_day = map(int, mok_date.split("-"))
    mok_datetime = datetime(year, mok_month, mok_day)

    # Get scalar threshold if needed
    if isinstance(wet_spell_threshold, xr.DataArray):
        threshold = float(wet_spell_threshold.values)
    else:
        threshold = float(wet_spell_threshold)

    # Select time range starting from MOK date
    time_mask = precip.time >= np.datetime64(mok_datetime)
    precip_after_mok = precip.sel(time=time_mask)

    if len(precip_after_mok.time) < wet_spell_days:
        return OnsetResult(
            onset_date=None,
            onset_doy=None,
            wet_spell_start=None,
            wet_spell_total_mm=None,
            wet_spell_threshold_mm=threshold,
            is_valid=False,
            message=f"Insufficient data after {mok_date}",
        )

    # Search for wet spell
    precip_values = precip_after_mok.values
    times = pd.to_datetime(precip_after_mok.time.values)

    for i in range(len(precip_values) - wet_spell_days + 1):
        window = precip_values[i : i + wet_spell_days]

        # Check if all days are wet
        all_wet = np.all(window >= wet_day_threshold_mm)

        # Check if total exceeds threshold
        total = np.sum(window)
        exceeds_threshold = total >= threshold

        if all_wet and exceeds_threshold:
            onset_datetime = times[i].to_pydatetime()
            return OnsetResult(
                onset_date=onset_datetime,
                onset_doy=onset_datetime.timetuple().tm_yday,
                wet_spell_start=onset_datetime,
                wet_spell_total_mm=float(total),
                wet_spell_threshold_mm=threshold,
                is_valid=True,
                message="Onset detected",
            )

    # No onset found
    return OnsetResult(
        onset_date=None,
        onset_doy=None,
        wet_spell_start=None,
        wet_spell_total_mm=None,
        wet_spell_threshold_mm=threshold,
        is_valid=False,
        message=f"No valid wet spell found after {mok_date}",
    )


def compute_onset_from_forecast(
    forecast_precip: xr.DataArray,
    wet_spell_threshold: Union[float, xr.DataArray],
    init_date: datetime,
    forecast_window: Tuple[int, int] = (1, 15),
    wet_day_threshold_mm: float = 1.0,
    wet_spell_days: int = 5,
    mok_date: str = "06-02",
) -> Optional[datetime]:
    """
    Compute onset date from forecast precipitation.

    Parameters
    ----------
    forecast_precip : xr.DataArray
        Forecast precipitation time series
    wet_spell_threshold : float or xr.DataArray
        Wet spell accumulation threshold
    init_date : datetime
        Forecast initialization date
    forecast_window : tuple
        (start_day, end_day) of forecast window to check
    wet_day_threshold_mm : float
        Wet day threshold
    wet_spell_days : int
        Wet spell duration
    mok_date : str
        MOK filter date

    Returns
    -------
    datetime or None
        Forecast onset date, or None if no onset in window
    """
    year = init_date.year
    mok_month, mok_day = map(int, mok_date.split("-"))
    mok_datetime = datetime(year, mok_month, mok_day)

    # Define forecast time range
    start_date = init_date + timedelta(days=forecast_window[0])
    end_date = init_date + timedelta(days=forecast_window[1] + wet_spell_days)

    # Ensure we're after MOK date
    search_start = max(start_date, mok_datetime)

    # Select forecast period
    try:
        forecast = forecast_precip.sel(
            time=slice(
                np.datetime64(search_start),
                np.datetime64(end_date),
            )
        )
    except Exception as e:
        logger.warning(f"Could not select forecast period: {e}")
        return None

    # Compute onset
    result = compute_local_onset(
        forecast,
        wet_spell_threshold=wet_spell_threshold,
        wet_day_threshold_mm=wet_day_threshold_mm,
        wet_spell_days=wet_spell_days,
        mok_date=mok_date,
        year=year,
    )

    if result.is_valid and result.onset_date is not None:
        # Check if onset is within forecast window
        onset_lead = (result.onset_date - init_date).days
        if forecast_window[0] <= onset_lead <= forecast_window[1]:
            return result.onset_date

    return None


def compute_onset_grid(
    precip: xr.DataArray,
    wet_spell_thresholds: xr.DataArray,
    year: int,
    wet_day_threshold_mm: float = 1.0,
    wet_spell_days: int = 5,
    mok_date: str = "06-02",
) -> xr.DataArray:
    """
    Compute onset dates for entire grid.

    Parameters
    ----------
    precip : xr.DataArray
        3D precipitation array (time, lat, lon)
    wet_spell_thresholds : xr.DataArray
        2D threshold array (lat, lon)
    year : int
        Year for onset detection
    wet_day_threshold_mm : float
        Wet day threshold
    wet_spell_days : int
        Wet spell duration
    mok_date : str
        MOK filter date

    Returns
    -------
    xr.DataArray
        2D array of onset day-of-year (lat, lon), NaN where no onset
    """
    lat = precip.lat.values
    lon = precip.lon.values

    onset_doy = np.full((len(lat), len(lon)), np.nan)

    for i, lat_val in enumerate(lat):
        for j, lon_val in enumerate(lon):
            # Extract time series for this location
            ts = precip.sel(lat=lat_val, lon=lon_val)

            # Get threshold for this location
            if "lat" in wet_spell_thresholds.dims:
                threshold = wet_spell_thresholds.sel(lat=lat_val, lon=lon_val)
            else:
                threshold = wet_spell_thresholds

            # Compute onset
            result = compute_local_onset(
                ts,
                wet_spell_threshold=threshold,
                wet_day_threshold_mm=wet_day_threshold_mm,
                wet_spell_days=wet_spell_days,
                mok_date=mok_date,
                year=year,
            )

            if result.is_valid and result.onset_doy is not None:
                onset_doy[i, j] = result.onset_doy

    # Create DataArray
    return xr.DataArray(
        onset_doy,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        attrs={
            "long_name": "Monsoon onset day of year",
            "units": "day of year",
            "year": year,
            "mok_date": mok_date,
        },
    )


def compute_ensemble_onset(
    ensemble_precip: xr.DataArray,
    wet_spell_threshold: Union[float, xr.DataArray],
    init_date: datetime,
    forecast_window: Tuple[int, int] = (1, 15),
    wet_day_threshold_mm: float = 1.0,
    wet_spell_days: int = 5,
    mok_date: str = "06-02",
    member_dim: str = "member",
) -> List[Optional[datetime]]:
    """
    Compute onset dates from ensemble forecast.

    Parameters
    ----------
    ensemble_precip : xr.DataArray
        Ensemble forecast precipitation (member, time, ...)
    wet_spell_threshold : float or xr.DataArray
        Wet spell threshold
    init_date : datetime
        Initialization date
    forecast_window : tuple
        Forecast window
    wet_day_threshold_mm : float
        Wet day threshold
    wet_spell_days : int
        Wet spell duration
    mok_date : str
        MOK filter date
    member_dim : str
        Name of ensemble member dimension

    Returns
    -------
    list of datetime or None
        Onset date for each ensemble member
    """
    n_members = len(ensemble_precip[member_dim])
    onsets = []

    for m in range(n_members):
        member_precip = ensemble_precip.isel({member_dim: m})

        onset = compute_onset_from_forecast(
            member_precip,
            wet_spell_threshold=wet_spell_threshold,
            init_date=init_date,
            forecast_window=forecast_window,
            wet_day_threshold_mm=wet_day_threshold_mm,
            wet_spell_days=wet_spell_days,
            mok_date=mok_date,
        )
        onsets.append(onset)

    return onsets


def onset_to_bin(
    onset_date: Optional[datetime],
    init_date: datetime,
    bin_size_days: int = 5,
    n_bins: int = 7,
) -> int:
    """
    Convert onset date to probability bin index.

    Bins are:
    - 0: days 1-5
    - 1: days 6-10
    - 2: days 11-15
    - ...
    - n_bins-1: after day (n_bins-1)*bin_size

    Parameters
    ----------
    onset_date : datetime or None
        Detected onset date (None = no onset)
    init_date : datetime
        Forecast initialization date
    bin_size_days : int
        Size of each bin in days
    n_bins : int
        Number of bins

    Returns
    -------
    int
        Bin index (0 to n_bins-1)
    """
    if onset_date is None:
        return n_bins - 1  # Last bin = "after window"

    lead_days = (onset_date - init_date).days

    if lead_days < 1:
        return 0  # Before window starts

    bin_idx = (lead_days - 1) // bin_size_days

    return min(bin_idx, n_bins - 1)
