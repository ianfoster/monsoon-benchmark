"""
Climatological baseline computations for monsoon onset.

Computes:
1. Wet spell thresholds (local climatological 5-day wet spell accumulation)
2. Climatological onset dates (historical mean onset date per location)
"""

from typing import Optional, Tuple
import logging

import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


def compute_wet_spell_threshold(
    precip: xr.DataArray,
    wet_spell_days: int = 5,
    wet_day_threshold_mm: float = 1.0,
    start_month: int = 6,
    end_month: int = 9,
    percentile: float = 50.0,
) -> xr.DataArray:
    """
    Compute climatological wet spell threshold for each grid cell.

    The threshold is the median (or specified percentile) of 5-day wet spell
    accumulations during the monsoon season across all years.

    Parameters
    ----------
    precip : xr.DataArray
        Multi-year precipitation data (time, lat, lon)
    wet_spell_days : int
        Duration of wet spell (default 5)
    wet_day_threshold_mm : float
        Minimum precipitation for wet day (default 1.0 mm)
    start_month : int
        Start month for monsoon season (default 6 = June)
    end_month : int
        End month for monsoon season (default 9 = September)
    percentile : float
        Percentile for threshold (default 50 = median)

    Returns
    -------
    xr.DataArray
        2D array of wet spell thresholds (lat, lon) in mm
    """
    # Select monsoon season months
    month_mask = (precip.time.dt.month >= start_month) & (
        precip.time.dt.month <= end_month
    )
    precip_monsoon = precip.sel(time=month_mask)

    lat = precip.lat.values
    lon = precip.lon.values

    thresholds = np.zeros((len(lat), len(lon)))

    for i, lat_val in enumerate(lat):
        for j, lon_val in enumerate(lon):
            ts = precip_monsoon.sel(lat=lat_val, lon=lon_val).values

            # Find all wet spells
            wet_spell_totals = []
            k = 0
            while k <= len(ts) - wet_spell_days:
                window = ts[k : k + wet_spell_days]

                # Check if valid wet spell
                if np.all(window >= wet_day_threshold_mm):
                    wet_spell_totals.append(np.sum(window))
                    k += wet_spell_days  # Skip to after this spell
                else:
                    k += 1

            if wet_spell_totals:
                thresholds[i, j] = np.percentile(wet_spell_totals, percentile)
            else:
                # Fallback: use a reasonable default
                thresholds[i, j] = wet_spell_days * wet_day_threshold_mm

    return xr.DataArray(
        thresholds,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        attrs={
            "long_name": f"Climatological {wet_spell_days}-day wet spell threshold",
            "units": "mm",
            "percentile": percentile,
            "wet_day_threshold_mm": wet_day_threshold_mm,
        },
    )


def compute_climatological_onset(
    precip: xr.DataArray,
    wet_spell_threshold: xr.DataArray,
    years: Optional[list] = None,
    wet_day_threshold_mm: float = 1.0,
    wet_spell_days: int = 5,
    mok_date: str = "06-02",
) -> xr.DataArray:
    """
    Compute climatological mean onset date for each grid cell.

    Parameters
    ----------
    precip : xr.DataArray
        Multi-year precipitation data (time, lat, lon)
    wet_spell_threshold : xr.DataArray
        Wet spell thresholds (lat, lon)
    years : list, optional
        Years to use for climatology (default: all available)
    wet_day_threshold_mm : float
        Wet day threshold
    wet_spell_days : int
        Wet spell duration
    mok_date : str
        MOK filter date

    Returns
    -------
    xr.DataArray
        Mean onset day-of-year for each grid cell
    """
    from monsoon_benchmark.indices.onset import compute_local_onset

    if years is None:
        years = np.unique(precip.time.dt.year.values)

    lat = precip.lat.values
    lon = precip.lon.values

    # Collect onset dates for each cell
    onset_doys = np.zeros((len(years), len(lat), len(lon)))
    onset_doys[:] = np.nan

    for y_idx, year in enumerate(years):
        # Select year
        year_mask = precip.time.dt.year == year
        precip_year = precip.sel(time=year_mask)

        for i, lat_val in enumerate(lat):
            for j, lon_val in enumerate(lon):
                ts = precip_year.sel(lat=lat_val, lon=lon_val)

                if "lat" in wet_spell_threshold.dims:
                    threshold = wet_spell_threshold.sel(lat=lat_val, lon=lon_val)
                else:
                    threshold = wet_spell_threshold

                result = compute_local_onset(
                    ts,
                    wet_spell_threshold=threshold,
                    wet_day_threshold_mm=wet_day_threshold_mm,
                    wet_spell_days=wet_spell_days,
                    mok_date=mok_date,
                    year=int(year),
                )

                if result.is_valid and result.onset_doy is not None:
                    onset_doys[y_idx, i, j] = result.onset_doy

    # Compute mean, ignoring NaN
    mean_onset = np.nanmean(onset_doys, axis=0)

    return xr.DataArray(
        mean_onset,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        attrs={
            "long_name": "Climatological mean onset day of year",
            "units": "day of year",
            "climatology_years": f"{min(years)}-{max(years)}",
            "n_years": len(years),
        },
    )


def get_climatological_onset_date(
    clim_onset_doy: xr.DataArray,
    lat: float,
    lon: float,
    year: int,
) -> datetime:
    """
    Get climatological onset date for specific location and year.

    Parameters
    ----------
    clim_onset_doy : xr.DataArray
        Climatological onset day-of-year (lat, lon)
    lat : float
        Latitude
    lon : float
        Longitude
    year : int
        Year (to convert DOY to date)

    Returns
    -------
    datetime
        Climatological onset date
    """
    doy = float(clim_onset_doy.sel(lat=lat, lon=lon, method="nearest").values)

    if np.isnan(doy):
        # Fallback to regional mean
        doy = float(clim_onset_doy.mean().values)

    # Convert DOY to date
    return datetime(year, 1, 1) + pd.Timedelta(days=int(doy) - 1)


def compute_onset_anomaly(
    onset_doy: xr.DataArray,
    clim_onset_doy: xr.DataArray,
) -> xr.DataArray:
    """
    Compute onset anomaly relative to climatology.

    Parameters
    ----------
    onset_doy : xr.DataArray
        Observed onset day-of-year
    clim_onset_doy : xr.DataArray
        Climatological onset day-of-year

    Returns
    -------
    xr.DataArray
        Onset anomaly in days (negative = early, positive = late)
    """
    anomaly = onset_doy - clim_onset_doy

    anomaly.attrs = {
        "long_name": "Onset anomaly relative to climatology",
        "units": "days",
        "description": "Negative = early onset, Positive = late onset",
    }

    return anomaly


def compute_mok_statistics(years: list = None) -> dict:
    """
    Compute statistics of Monsoon Onset over Kerala (MOK) dates.

    Based on IMD declared MOK dates from 1901-2024.

    Returns
    -------
    dict
        MOK statistics including median, mean, std
    """
    # Historical MOK dates (from IMD records)
    # This is a simplified version - full data would come from IMD
    # Median MOK date based on 124-year record is June 2 (DOY 153/154)

    return {
        "median_date": "06-02",  # June 2
        "median_doy": 153,
        "mean_doy": 153.5,
        "std_doy": 7.5,
        "earliest_doy": 135,  # May 15
        "latest_doy": 178,  # June 27
        "reference_period": "1901-2024",
    }


class ClimatologyBaseline:
    """
    Climatological baseline for benchmark comparison.

    The climatological forecast simply predicts the historical mean
    onset date for each location.
    """

    def __init__(
        self,
        clim_onset_doy: xr.DataArray,
        historical_onsets: Optional[xr.DataArray] = None,
    ):
        """
        Initialize climatological baseline.

        Parameters
        ----------
        clim_onset_doy : xr.DataArray
            Mean onset day-of-year (lat, lon)
        historical_onsets : xr.DataArray, optional
            Historical onset DOYs (year, lat, lon) for probabilistic baseline
        """
        self.clim_onset_doy = clim_onset_doy
        self.historical_onsets = historical_onsets

    def predict_deterministic(self, lat: float, lon: float, year: int) -> datetime:
        """
        Get deterministic climatological prediction.

        Parameters
        ----------
        lat : float
            Latitude
        lon : float
            Longitude
        year : int
            Year

        Returns
        -------
        datetime
            Predicted onset date
        """
        return get_climatological_onset_date(self.clim_onset_doy, lat, lon, year)

    def predict_probabilistic(
        self,
        lat: float,
        lon: float,
        init_date: datetime,
        bin_size_days: int = 5,
        n_bins: int = 7,
    ) -> np.ndarray:
        """
        Get probabilistic climatological prediction.

        Uses historical distribution of onset dates to create
        probability distribution over bins.

        Parameters
        ----------
        lat : float
            Latitude
        lon : float
            Longitude
        init_date : datetime
            Initialization date (for computing lead times)
        bin_size_days : int
            Size of probability bins
        n_bins : int
            Number of bins

        Returns
        -------
        np.ndarray
            Probability for each bin
        """
        if self.historical_onsets is None:
            # Fall back to uniform distribution around climatology
            probs = np.zeros(n_bins)
            clim_doy = float(
                self.clim_onset_doy.sel(lat=lat, lon=lon, method="nearest").values
            )
            clim_date = datetime(init_date.year, 1, 1) + pd.Timedelta(days=int(clim_doy) - 1)
            lead_days = (clim_date - init_date).days

            # Put most probability in bin containing climatology
            bin_idx = max(0, min((lead_days - 1) // bin_size_days, n_bins - 1))
            probs[bin_idx] = 0.6
            if bin_idx > 0:
                probs[bin_idx - 1] = 0.2
            if bin_idx < n_bins - 1:
                probs[bin_idx + 1] = 0.2
            return probs / probs.sum()

        # Use historical distribution
        hist = self.historical_onsets.sel(lat=lat, lon=lon, method="nearest").values

        probs = np.zeros(n_bins)
        for doy in hist:
            if np.isnan(doy):
                probs[-1] += 1  # No onset -> last bin
            else:
                onset_date = datetime(init_date.year, 1, 1) + pd.Timedelta(days=int(doy) - 1)
                lead_days = (onset_date - init_date).days
                bin_idx = max(0, min((lead_days - 1) // bin_size_days, n_bins - 1))
                probs[bin_idx] += 1

        # Normalize
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.ones(n_bins) / n_bins

        return probs
