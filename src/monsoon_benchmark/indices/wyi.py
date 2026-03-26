"""
Webster-Yang Index (WYI) for large-scale monsoon circulation onset.

The WYI captures the change in large-scale vertical wind shear during
monsoon onset. It is defined as the difference between upper-level (200 hPa)
and lower-level (850 hPa) zonal wind averaged over the monsoon region.

Reference:
- Webster, P. J. & Yang, S. (1992). Monsoon and ENSO: Selectively interactive
  systems. Q. J. Royal Meteorol. Soc. 118, 877-926.
"""

from datetime import datetime
from typing import Optional, Tuple
import logging

import numpy as np
import xarray as xr
import pandas as pd

logger = logging.getLogger(__name__)

# Default WYI averaging region
WYI_LON_RANGE = (40, 110)  # degrees East
WYI_LAT_RANGE = (0, 20)  # degrees North


def compute_wyi(
    u200: xr.DataArray,
    u850: xr.DataArray,
    lon_range: Tuple[float, float] = WYI_LON_RANGE,
    lat_range: Tuple[float, float] = WYI_LAT_RANGE,
) -> xr.DataArray:
    """
    Compute Webster-Yang Index time series.

    WYI = U_200hPa - U_850hPa

    averaged over the specified region.

    Parameters
    ----------
    u200 : xr.DataArray
        Zonal wind at 200 hPa (time, lat, lon)
    u850 : xr.DataArray
        Zonal wind at 850 hPa (time, lat, lon)
    lon_range : tuple
        Longitude range for averaging (min, max)
    lat_range : tuple
        Latitude range for averaging (min, max)

    Returns
    -------
    xr.DataArray
        WYI time series
    """
    # Select region
    u200_region = u200.sel(
        lat=slice(lat_range[0], lat_range[1]),
        lon=slice(lon_range[0], lon_range[1]),
    )
    u850_region = u850.sel(
        lat=slice(lat_range[0], lat_range[1]),
        lon=slice(lon_range[0], lon_range[1]),
    )

    # Compute area-weighted mean
    weights = np.cos(np.deg2rad(u200_region.lat))

    u200_mean = u200_region.weighted(weights).mean(dim=["lat", "lon"])
    u850_mean = u850_region.weighted(weights).mean(dim=["lat", "lon"])

    # WYI = U200 - U850
    wyi = u200_mean - u850_mean

    wyi.attrs = {
        "long_name": "Webster-Yang Index",
        "units": "m/s",
        "formula": "U_200hPa - U_850hPa",
        "averaging_region": f"lat={lat_range}, lon={lon_range}",
    }

    return wyi


def compute_wyi_from_dataset(
    ds: xr.Dataset,
    upper_level: int = 200,
    lower_level: int = 850,
    lon_range: Tuple[float, float] = WYI_LON_RANGE,
    lat_range: Tuple[float, float] = WYI_LAT_RANGE,
) -> xr.DataArray:
    """
    Compute WYI from dataset with level dimension.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with zonal wind 'u' variable and 'level' dimension
    upper_level : int
        Upper pressure level in hPa (default 200)
    lower_level : int
        Lower pressure level in hPa (default 850)
    lon_range : tuple
        Longitude range for averaging
    lat_range : tuple
        Latitude range for averaging

    Returns
    -------
    xr.DataArray
        WYI time series
    """
    # Extract u-wind at specified levels
    u200 = ds["u"].sel(level=upper_level)
    u850 = ds["u"].sel(level=lower_level)

    return compute_wyi(u200, u850, lon_range, lat_range)


def compute_wyi_climatology(
    wyi: xr.DataArray,
    groupby: str = "dayofyear",
) -> xr.DataArray:
    """
    Compute climatological WYI.

    Parameters
    ----------
    wyi : xr.DataArray
        Multi-year WYI time series
    groupby : str
        Grouping for climatology ("dayofyear" or "month")

    Returns
    -------
    xr.DataArray
        Climatological WYI
    """
    if groupby == "dayofyear":
        clim = wyi.groupby("time.dayofyear").mean()
    elif groupby == "month":
        clim = wyi.groupby("time.month").mean()
    else:
        raise ValueError(f"Invalid groupby: {groupby}")

    clim.attrs["long_name"] = f"Climatological WYI ({groupby})"
    return clim


def compute_wyi_onset(
    wyi: xr.DataArray,
    threshold: float,
    smoothing_window: int = 7,
    year: Optional[int] = None,
    search_start_doy: int = 120,  # ~May 1
    search_end_doy: int = 200,  # ~July 19
) -> Optional[datetime]:
    """
    Detect large-scale circulation onset using WYI.

    Onset is defined as the date when the 7-day moving average of WYI
    crosses the climatological June 2 value (threshold).

    Parameters
    ----------
    wyi : xr.DataArray
        WYI time series for a single year
    threshold : float
        WYI threshold value (typically climatological June 2 value)
    smoothing_window : int
        Window for moving average smoothing (default 7 days)
    year : int, optional
        Year (inferred from data if not provided)
    search_start_doy : int
        Day of year to start searching (default 120 = ~May 1)
    search_end_doy : int
        Day of year to stop searching (default 200 = ~July 19)

    Returns
    -------
    datetime or None
        Circulation onset date, or None if not detected
    """
    # Infer year if not provided
    if year is None:
        years = np.unique(wyi.time.dt.year.values)
        if len(years) != 1:
            raise ValueError("Data contains multiple years. Please specify year.")
        year = int(years[0])

    # Apply smoothing
    wyi_smooth = wyi.rolling(time=smoothing_window, center=True).mean()

    # Select search window
    start_date = datetime(year, 1, 1) + pd.Timedelta(days=search_start_doy - 1)
    end_date = datetime(year, 1, 1) + pd.Timedelta(days=search_end_doy - 1)

    wyi_search = wyi_smooth.sel(
        time=slice(np.datetime64(start_date), np.datetime64(end_date))
    )

    # Find crossing point
    values = wyi_search.values
    times = pd.to_datetime(wyi_search.time.values)

    for i in range(len(values) - 1):
        # Check for upward crossing of threshold
        if values[i] < threshold and values[i + 1] >= threshold:
            # Linear interpolation to find exact crossing
            frac = (threshold - values[i]) / (values[i + 1] - values[i])
            onset_time = times[i] + frac * (times[i + 1] - times[i])
            return onset_time.to_pydatetime()

    return None


def compute_wyi_onset_from_climatology(
    wyi: xr.DataArray,
    wyi_clim: xr.DataArray,
    mok_doy: int = 153,  # June 2
    smoothing_window: int = 7,
    year: Optional[int] = None,
) -> Optional[datetime]:
    """
    Detect WYI onset using climatological June 2 threshold.

    Parameters
    ----------
    wyi : xr.DataArray
        WYI time series for a single year
    wyi_clim : xr.DataArray
        Climatological WYI (indexed by dayofyear)
    mok_doy : int
        Day of year for MOK threshold (default 153 = June 2)
    smoothing_window : int
        Smoothing window
    year : int, optional
        Year

    Returns
    -------
    datetime or None
        Circulation onset date
    """
    # Get threshold from climatology
    threshold = float(wyi_clim.sel(dayofyear=mok_doy).values)

    return compute_wyi_onset(
        wyi,
        threshold=threshold,
        smoothing_window=smoothing_window,
        year=year,
    )


def compute_somali_jet_index(
    u850: xr.DataArray,
    lon_range: Tuple[float, float] = (45, 55),
    lat_range: Tuple[float, float] = (5, 15),
) -> xr.DataArray:
    """
    Compute Somali Jet intensity index.

    The Somali Jet is the low-level cross-equatorial flow that feeds
    moisture into the Indian monsoon.

    Parameters
    ----------
    u850 : xr.DataArray
        Zonal wind at 850 hPa
    lon_range : tuple
        Longitude range for jet core
    lat_range : tuple
        Latitude range for jet core

    Returns
    -------
    xr.DataArray
        Somali Jet index (mean zonal wind in jet region)
    """
    jet = u850.sel(
        lat=slice(lat_range[0], lat_range[1]),
        lon=slice(lon_range[0], lon_range[1]),
    )

    weights = np.cos(np.deg2rad(jet.lat))
    jet_index = jet.weighted(weights).mean(dim=["lat", "lon"])

    jet_index.attrs = {
        "long_name": "Somali Jet Index",
        "units": "m/s",
        "averaging_region": f"lat={lat_range}, lon={lon_range}",
    }

    return jet_index
