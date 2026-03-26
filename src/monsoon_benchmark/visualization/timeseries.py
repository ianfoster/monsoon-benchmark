"""
Time series visualizations for monsoon onset benchmarking.

These plots show temporal evolution of:
- Rainfall accumulation toward onset
- Webster-Yang circulation index
- Somali Jet strength
"""

from typing import Optional, Tuple, List, Dict
from datetime import datetime
import logging

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)


def plot_rainfall_evolution(
    precip: xr.DataArray,
    onset_date: Optional[datetime] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    year: Optional[int] = None,
    wet_day_threshold: float = 1.0,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot rainfall evolution leading up to monsoon onset.

    Parameters
    ----------
    precip : xr.DataArray
        Daily precipitation time series (mm/day)
    onset_date : datetime, optional
        Detected onset date to mark
    lat, lon : float, optional
        Grid cell coordinates for title
    year : int, optional
        Year for title
    wet_day_threshold : float
        Threshold for wet days (mm)
    title : str, optional
        Custom title
    figsize : tuple
        Figure size
    ax : plt.Axes, optional
        Existing axes

    Returns
    -------
    plt.Figure
        Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Convert time coordinate to datetime if needed
    if isinstance(precip.time.values[0], np.datetime64):
        times = pd.to_datetime(precip.time.values)
    else:
        times = precip.time.values

    values = precip.values

    # Plot daily rainfall
    ax.bar(times, values, width=1, alpha=0.7, color="steelblue", label="Daily rainfall")

    # Add 5-day running mean
    if len(values) >= 5:
        running_mean = pd.Series(values).rolling(5, center=True).mean()
        ax.plot(
            times,
            running_mean,
            "r-",
            linewidth=2,
            label="5-day mean",
        )

    # Mark wet day threshold
    ax.axhline(
        wet_day_threshold,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=f"Wet day threshold ({wet_day_threshold} mm)",
    )

    # Mark onset date if provided
    if onset_date is not None:
        ax.axvline(
            onset_date,
            color="green",
            linestyle="-",
            linewidth=2,
            label=f"Onset: {onset_date.strftime('%b %d')}",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Precipitation (mm/day)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Title
    if title is None:
        parts = ["Rainfall Evolution"]
        if year:
            parts.append(f"({year})")
        if lat is not None and lon is not None:
            parts.append(f"at ({lat:.1f}°N, {lon:.1f}°E)")
        title = " ".join(parts)
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_wyi_evolution(
    wyi: xr.DataArray,
    climatology: Optional[xr.DataArray] = None,
    onset_date: Optional[datetime] = None,
    mok_date: Optional[datetime] = None,
    year: Optional[int] = None,
    smoothing_window: int = 7,
    figsize: Tuple[int, int] = (12, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot Webster-Yang Index evolution.

    Parameters
    ----------
    wyi : xr.DataArray
        Daily WYI time series
    climatology : xr.DataArray, optional
        Climatological WYI for comparison
    onset_date : datetime, optional
        WYI-based onset date
    mok_date : datetime, optional
        Monsoon onset over Kerala date
    year : int, optional
        Year for title
    smoothing_window : int
        Days for running mean
    figsize : tuple
        Figure size
    ax : plt.Axes, optional
        Existing axes

    Returns
    -------
    plt.Figure
        Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Convert time coordinate
    if isinstance(wyi.time.values[0], np.datetime64):
        times = pd.to_datetime(wyi.time.values)
    else:
        times = wyi.time.values

    # Plot raw WYI
    ax.plot(times, wyi.values, "b-", alpha=0.3, linewidth=0.5, label="Daily WYI")

    # Plot smoothed WYI
    if len(wyi.values) >= smoothing_window:
        smoothed = pd.Series(wyi.values).rolling(smoothing_window, center=True).mean()
        ax.plot(
            times,
            smoothed,
            "b-",
            linewidth=2,
            label=f"{smoothing_window}-day MA",
        )

    # Plot climatology if provided
    if climatology is not None:
        if isinstance(climatology.time.values[0], np.datetime64):
            clim_times = pd.to_datetime(climatology.time.values)
        else:
            clim_times = climatology.time.values
        ax.plot(
            clim_times,
            climatology.values,
            "k--",
            linewidth=1.5,
            alpha=0.7,
            label="Climatology",
        )

    # Mark onset date
    if onset_date is not None:
        ax.axvline(
            onset_date,
            color="green",
            linestyle="-",
            linewidth=2,
            label=f"WYI Onset: {onset_date.strftime('%b %d')}",
        )

    # Mark MOK date
    if mok_date is not None:
        ax.axvline(
            mok_date,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            label=f"MOK: {mok_date.strftime('%b %d')}",
        )

    ax.axhline(0, color="gray", linestyle="-", alpha=0.5)

    ax.set_xlabel("Date")
    ax.set_ylabel("Webster-Yang Index (m/s)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    title = "Webster-Yang Index Evolution"
    if year:
        title += f" ({year})"
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_somali_jet(
    u850: xr.DataArray,
    year: Optional[int] = None,
    climatology: Optional[xr.DataArray] = None,
    jet_lat: float = 10.0,
    jet_lon_range: Tuple[float, float] = (45, 60),
    figsize: Tuple[int, int] = (12, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot Somali Jet (low-level cross-equatorial flow) evolution.

    The Somali Jet is a key driver of Indian monsoon moisture transport.

    Parameters
    ----------
    u850 : xr.DataArray
        850 hPa zonal wind (time, lat, lon)
    year : int, optional
        Year for title
    climatology : xr.DataArray, optional
        Climatological jet strength
    jet_lat : float
        Latitude for jet strength extraction
    jet_lon_range : tuple
        Longitude range for averaging
    figsize : tuple
        Figure size
    ax : plt.Axes, optional
        Existing axes

    Returns
    -------
    plt.Figure
        Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Extract jet strength (area average of u850)
    if "lat" in u850.dims and "lon" in u850.dims:
        jet = u850.sel(
            lat=jet_lat,
            lon=slice(jet_lon_range[0], jet_lon_range[1]),
            method="nearest",
        ).mean(dim="lon")
    else:
        jet = u850  # Assume already extracted

    # Convert time coordinate
    if isinstance(jet.time.values[0], np.datetime64):
        times = pd.to_datetime(jet.time.values)
    else:
        times = jet.time.values

    # Plot jet strength
    ax.plot(times, jet.values, "purple", linewidth=2, label="Somali Jet (850 hPa)")

    # Plot climatology if provided
    if climatology is not None:
        if isinstance(climatology.time.values[0], np.datetime64):
            clim_times = pd.to_datetime(climatology.time.values)
        else:
            clim_times = climatology.time.values
        ax.plot(
            clim_times,
            climatology.values,
            "k--",
            linewidth=1.5,
            alpha=0.7,
            label="Climatology",
        )

    ax.axhline(0, color="gray", linestyle="-", alpha=0.5)

    ax.set_xlabel("Date")
    ax.set_ylabel("Zonal Wind at 850 hPa (m/s)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    title = f"Somali Jet Evolution ({jet_lat}°N, {jet_lon_range[0]}-{jet_lon_range[1]}°E)"
    if year:
        title += f" ({year})"
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_forecast_evolution(
    forecast_precip: xr.DataArray,
    observed_precip: xr.DataArray,
    init_date: datetime,
    model_name: str = "Model",
    onset_forecast: Optional[datetime] = None,
    onset_observed: Optional[datetime] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot forecast vs observed rainfall evolution.

    Parameters
    ----------
    forecast_precip : xr.DataArray
        Forecasted precipitation
    observed_precip : xr.DataArray
        Observed precipitation
    init_date : datetime
        Forecast initialization date
    model_name : str
        Model name for legend
    onset_forecast : datetime, optional
        Forecasted onset date
    onset_observed : datetime, optional
        Observed onset date
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure
        Figure object
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Top panel: Forecast
    ax1 = axes[0]
    if isinstance(forecast_precip.time.values[0], np.datetime64):
        fcst_times = pd.to_datetime(forecast_precip.time.values)
    else:
        fcst_times = forecast_precip.time.values

    ax1.bar(
        fcst_times,
        forecast_precip.values,
        width=1,
        alpha=0.7,
        color="steelblue",
        label=f"{model_name} Forecast",
    )

    if onset_forecast is not None:
        ax1.axvline(
            onset_forecast,
            color="blue",
            linestyle="-",
            linewidth=2,
            label=f"Fcst Onset: {onset_forecast.strftime('%b %d')}",
        )

    ax1.axvline(init_date, color="gray", linestyle="--", label="Init Date")
    ax1.set_ylabel("Precip (mm/day)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Forecast vs Observed - Init: {init_date.strftime('%Y-%m-%d')}")

    # Bottom panel: Observed
    ax2 = axes[1]
    if isinstance(observed_precip.time.values[0], np.datetime64):
        obs_times = pd.to_datetime(observed_precip.time.values)
    else:
        obs_times = observed_precip.time.values

    ax2.bar(
        obs_times,
        observed_precip.values,
        width=1,
        alpha=0.7,
        color="forestgreen",
        label="Observed (IMD)",
    )

    if onset_observed is not None:
        ax2.axvline(
            onset_observed,
            color="green",
            linestyle="-",
            linewidth=2,
            label=f"Obs Onset: {onset_observed.strftime('%b %d')}",
        )

    ax2.axvline(init_date, color="gray", linestyle="--", label="Init Date")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Precip (mm/day)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    return fig


def plot_ensemble_onset_spread(
    ensemble_onsets: List[Optional[datetime]],
    observed_onset: Optional[datetime],
    init_date: datetime,
    model_name: str = "Model",
    bin_size_days: int = 5,
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Plot ensemble spread of onset predictions.

    Parameters
    ----------
    ensemble_onsets : list
        List of onset dates from each ensemble member
    observed_onset : datetime
        Observed onset date
    init_date : datetime
        Forecast initialization date
    model_name : str
        Model name for title
    bin_size_days : int
        Days per bin for histogram
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure
        Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Convert onsets to day-of-year
    valid_onsets = [o for o in ensemble_onsets if o is not None]

    if len(valid_onsets) == 0:
        ax.text(
            0.5,
            0.5,
            "No valid onset predictions",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    # Convert to days after init
    days_after_init = [(o - init_date).days for o in valid_onsets]

    # Create histogram
    max_days = max(days_after_init) + bin_size_days
    bins = np.arange(0, max_days + bin_size_days, bin_size_days)

    ax.hist(
        days_after_init,
        bins=bins,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        label=f"Ensemble ({len(valid_onsets)} members)",
    )

    # Mark observed onset
    if observed_onset is not None:
        obs_days = (observed_onset - init_date).days
        ax.axvline(
            obs_days,
            color="red",
            linestyle="-",
            linewidth=2,
            label=f"Observed: Day {obs_days}",
        )

    # Mark ensemble median
    median_days = np.median(days_after_init)
    ax.axvline(
        median_days,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Ensemble Median: Day {median_days:.0f}",
    )

    # Count no-onset members
    n_no_onset = len(ensemble_onsets) - len(valid_onsets)
    if n_no_onset > 0:
        ax.text(
            0.98,
            0.95,
            f"No onset: {n_no_onset} members",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax.set_xlabel(f"Days after initialization ({init_date.strftime('%Y-%m-%d')})")
    ax.set_ylabel("Number of ensemble members")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{model_name} Ensemble Onset Spread")

    plt.tight_layout()
    return fig


def plot_model_comparison_timeseries(
    model_forecasts: Dict[str, xr.DataArray],
    observed: xr.DataArray,
    init_date: datetime,
    variable: str = "precipitation",
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Compare multiple model forecasts against observations.

    Parameters
    ----------
    model_forecasts : dict
        Dictionary of {model_name: forecast_dataarray}
    observed : xr.DataArray
        Observed values
    init_date : datetime
        Initialization date
    variable : str
        Variable name for y-label
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure
        Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Color cycle
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_forecasts)))

    # Plot each model
    for (model_name, forecast), color in zip(model_forecasts.items(), colors):
        if isinstance(forecast.time.values[0], np.datetime64):
            times = pd.to_datetime(forecast.time.values)
        else:
            times = forecast.time.values

        ax.plot(times, forecast.values, color=color, linewidth=1.5, label=model_name)

    # Plot observations
    if isinstance(observed.time.values[0], np.datetime64):
        obs_times = pd.to_datetime(observed.time.values)
    else:
        obs_times = observed.time.values

    ax.plot(
        obs_times,
        observed.values,
        "k-",
        linewidth=2.5,
        label="Observed",
    )

    # Mark init date
    ax.axvline(init_date, color="gray", linestyle="--", alpha=0.7, label="Init Date")

    ax.set_xlabel("Date")
    ax.set_ylabel(variable.capitalize())
    ax.legend(loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax.set_title(f"Model Comparison - Init: {init_date.strftime('%Y-%m-%d')}")

    plt.tight_layout()
    return fig
