"""
Spatial map visualizations for monsoon onset benchmarking.

Creates maps showing:
- MAE at each grid cell
- Skill difference relative to climatology
- Onset date climatology
"""

from typing import Optional, Tuple, List
import logging

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

logger = logging.getLogger(__name__)

# Try to import cartopy for geographic plotting
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    logger.warning("cartopy not installed. Maps will use simple plotting.")

from monsoon_benchmark.data.regions import CMZ_BOUNDS, INDIA_BOUNDS


def plot_skill_map(
    skill_data: xr.DataArray,
    title: str = "Forecast Skill",
    metric: str = "MAE",
    cmap: str = "RdBu_r",
    vmin: float = -10,
    vmax: float = 10,
    show_cmz: bool = True,
    figsize: Tuple[int, int] = (8, 8),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot spatial map of forecast skill.

    Parameters
    ----------
    skill_data : xr.DataArray
        2D array of skill values (lat, lon)
    title : str
        Plot title
    metric : str
        Metric name for colorbar label
    cmap : str
        Colormap name
    vmin, vmax : float
        Colorbar limits
    show_cmz : bool
        Whether to outline CMZ region
    figsize : tuple
        Figure size
    ax : plt.Axes, optional
        Existing axes to plot on

    Returns
    -------
    plt.Figure
        Figure object
    """
    if ax is None:
        if HAS_CARTOPY:
            fig, ax = plt.subplots(
                1, 1, figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()}
            )
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Get coordinates
    lat = skill_data.lat.values
    lon = skill_data.lon.values

    # Plot data
    if HAS_CARTOPY:
        im = ax.pcolormesh(
            lon,
            lat,
            skill_data.values,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.coastlines(resolution="50m")
        ax.add_feature(cfeature.BORDERS, linestyle=":")

        # Set extent
        ax.set_extent(
            [INDIA_BOUNDS.lon_min, INDIA_BOUNDS.lon_max,
             INDIA_BOUNDS.lat_min, INDIA_BOUNDS.lat_max],
            crs=ccrs.PlateCarree(),
        )
    else:
        im = ax.pcolormesh(lon, lat, skill_data.values, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlim(INDIA_BOUNDS.lon_min, INDIA_BOUNDS.lon_max)
        ax.set_ylim(INDIA_BOUNDS.lat_min, INDIA_BOUNDS.lat_max)

    # Add CMZ outline
    if show_cmz:
        cmz_lon = [
            CMZ_BOUNDS.lon_min,
            CMZ_BOUNDS.lon_max,
            CMZ_BOUNDS.lon_max,
            CMZ_BOUNDS.lon_min,
            CMZ_BOUNDS.lon_min,
        ]
        cmz_lat = [
            CMZ_BOUNDS.lat_min,
            CMZ_BOUNDS.lat_min,
            CMZ_BOUNDS.lat_max,
            CMZ_BOUNDS.lat_max,
            CMZ_BOUNDS.lat_min,
        ]
        if HAS_CARTOPY:
            ax.plot(cmz_lon, cmz_lat, "k-", linewidth=2, transform=ccrs.PlateCarree())
        else:
            ax.plot(cmz_lon, cmz_lat, "k-", linewidth=2)

    # Add cell values
    for i, lat_val in enumerate(lat):
        for j, lon_val in enumerate(lon):
            val = skill_data.values[i, j]
            if not np.isnan(val):
                ax.text(
                    lon_val,
                    lat_val,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    transform=ccrs.PlateCarree() if HAS_CARTOPY else ax.transData,
                )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(metric)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    return fig


def plot_mae_difference(
    model_mae: xr.DataArray,
    clim_mae: xr.DataArray,
    model_name: str,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """
    Plot MAE difference between model and climatology.

    Blue = model better than climatology (negative difference)
    Red = model worse than climatology (positive difference)

    Parameters
    ----------
    model_mae : xr.DataArray
        Model MAE (lat, lon)
    clim_mae : xr.DataArray
        Climatology MAE (lat, lon)
    model_name : str
        Model name for title
    title : str, optional
        Custom title
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure
        Figure object
    """
    diff = model_mae - clim_mae

    if title is None:
        title = f"{model_name} - Climatology MAE (days)"

    return plot_skill_map(
        diff,
        title=title,
        metric="MAE difference (days)",
        cmap="RdBu_r",
        vmin=-8,
        vmax=8,
        figsize=figsize,
    )


def plot_onset_climatology(
    onset_clim: xr.DataArray,
    title: str = "Climatological Onset Date (DOY)",
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """
    Plot climatological onset dates.

    Parameters
    ----------
    onset_clim : xr.DataArray
        Climatological onset day-of-year (lat, lon)
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure
        Figure object
    """
    return plot_skill_map(
        onset_clim,
        title=title,
        metric="Day of Year",
        cmap="YlOrRd",
        vmin=140,  # ~May 20
        vmax=200,  # ~July 19
        show_cmz=True,
        figsize=figsize,
    )


def plot_model_comparison(
    model_results: dict,
    metric: str = "mae",
    forecast_window: str = "medium_range",
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """
    Create multi-panel figure comparing models.

    Parameters
    ----------
    model_results : dict
        Dictionary of {model_name: xr.DataArray} with metric values
    metric : str
        Metric name
    forecast_window : str
        Forecast window for title
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure
        Figure with multiple subplots
    """
    n_models = len(model_results)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    if HAS_CARTOPY:
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    axes = np.atleast_2d(axes)

    for idx, (model_name, data) in enumerate(model_results.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        plot_skill_map(
            data,
            title=model_name,
            metric=metric.upper(),
            ax=ax,
        )

    # Hide empty subplots
    for idx in range(n_models, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(f"{metric.upper()} for {forecast_window} forecasts", fontsize=14)
    plt.tight_layout()

    return fig
