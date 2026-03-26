"""
Regridding utilities for standardizing data to common grid.

The benchmarking framework uses a 4° × 4° grid for evaluation.
This module provides conservative regridding to preserve area-weighted means.
"""

import logging
from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# Try to import xesmf for regridding
try:
    import xesmf as xe
    HAS_XESMF = True
except ImportError:
    HAS_XESMF = False
    logger.warning(
        "xesmf not installed. Using simple interpolation instead of conservative regridding."
    )


def create_target_grid(
    lat_range: Tuple[float, float] = (6, 38),
    lon_range: Tuple[float, float] = (66, 98),
    resolution: float = 4.0,
) -> xr.Dataset:
    """
    Create target grid dataset for regridding.

    Parameters
    ----------
    lat_range : tuple
        (min_lat, max_lat) in degrees
    lon_range : tuple
        (min_lon, max_lon) in degrees
    resolution : float
        Grid resolution in degrees

    Returns
    -------
    xr.Dataset
        Dataset with lat/lon coordinates for target grid
    """
    lat = np.arange(
        lat_range[0] + resolution / 2,
        lat_range[1],
        resolution,
    )
    lon = np.arange(
        lon_range[0] + resolution / 2,
        lon_range[1],
        resolution,
    )

    return xr.Dataset(
        {
            "lat": (["lat"], lat),
            "lon": (["lon"], lon),
        }
    )


def conservative_regrid(
    ds: xr.Dataset,
    target_resolution: float = 4.0,
    lat_range: Tuple[float, float] = (6, 38),
    lon_range: Tuple[float, float] = (66, 98),
    method: str = "conservative",
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> xr.Dataset:
    """
    Regrid dataset to target resolution using conservative remapping.

    Conservative remapping preserves the area-weighted integral of the field,
    which is important for precipitation data.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to regrid
    target_resolution : float
        Target grid resolution in degrees (default 4.0)
    lat_range : tuple
        Target latitude range
    lon_range : tuple
        Target longitude range
    method : str
        Regridding method: "conservative", "bilinear", "nearest_s2d"
    lat_dim : str
        Name of latitude dimension
    lon_dim : str
        Name of longitude dimension

    Returns
    -------
    xr.Dataset
        Regridded dataset
    """
    # Create target grid
    target = create_target_grid(lat_range, lon_range, target_resolution)

    if HAS_XESMF:
        return _regrid_xesmf(ds, target, method, lat_dim, lon_dim)
    else:
        logger.warning("Using simple interpolation. Install xesmf for conservative regridding.")
        return _regrid_simple(ds, target, lat_dim, lon_dim)


def _regrid_xesmf(
    ds: xr.Dataset,
    target: xr.Dataset,
    method: str,
    lat_dim: str,
    lon_dim: str,
) -> xr.Dataset:
    """Regrid using xesmf."""
    # Ensure coordinates are named correctly for xesmf
    if lat_dim != "lat" or lon_dim != "lon":
        ds = ds.rename({lat_dim: "lat", lon_dim: "lon"})

    # Create regridder
    regridder = xe.Regridder(
        ds,
        target,
        method,
        periodic=False,
        ignore_degenerate=True,
    )

    # Apply regridding
    ds_out = regridder(ds, keep_attrs=True)

    # Add regridding info to attributes
    for var in ds_out.data_vars:
        ds_out[var].attrs["regrid_method"] = method
        ds_out[var].attrs["regrid_resolution"] = f"{target['lat'].values[1] - target['lat'].values[0]:.1f}deg"

    return ds_out


def _regrid_simple(
    ds: xr.Dataset,
    target: xr.Dataset,
    lat_dim: str,
    lon_dim: str,
) -> xr.Dataset:
    """Simple interpolation fallback when xesmf is not available."""
    # Use xarray's built-in interpolation
    return ds.interp(
        {lat_dim: target["lat"], lon_dim: target["lon"]},
        method="linear",
    )


def apply_land_mask(
    ds: xr.Dataset,
    mask: Optional[xr.DataArray] = None,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> xr.Dataset:
    """
    Apply land-sea mask to dataset.

    For models with 0.25° resolution, apply land mask before regridding
    to prevent ocean values from influencing land averages.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    mask : xr.DataArray, optional
        Land mask (1 = land, 0 = ocean). If None, creates simple mask.
    lat_dim : str
        Latitude dimension name
    lon_dim : str
        Longitude dimension name

    Returns
    -------
    xr.Dataset
        Dataset with ocean values masked
    """
    if mask is None:
        # Create simple land mask for India region
        mask = create_india_land_mask(
            ds[lat_dim].values,
            ds[lon_dim].values,
        )
        mask = xr.DataArray(mask, dims=[lat_dim, lon_dim])

    # Apply mask to all data variables
    masked = ds.where(mask > 0)

    return masked


def create_india_land_mask(
    lat: np.ndarray,
    lon: np.ndarray,
    simplified: bool = True,
) -> np.ndarray:
    """
    Create a simplified land mask for India.

    Parameters
    ----------
    lat : np.ndarray
        Latitude coordinates
    lon : np.ndarray
        Longitude coordinates
    simplified : bool
        If True, use simplified rectangular approximation

    Returns
    -------
    np.ndarray
        2D boolean mask (lat, lon) where True = land
    """
    lat_2d, lon_2d = np.meshgrid(lat, lon, indexing="ij")

    if simplified:
        # Simplified mask: rectangular approximation of India
        # This is a rough approximation - for production use a proper mask
        mask = (
            (lat_2d >= 8) & (lat_2d <= 37)
            & (lon_2d >= 68) & (lon_2d <= 97)
        )

        # Exclude Bay of Bengal (roughly)
        bay_of_bengal = (
            (lat_2d < 16) & (lon_2d > 80)
            & ~((lat_2d > 10) & (lon_2d < 85))  # Keep eastern coast
        )
        mask = mask & ~bay_of_bengal

        # Exclude Arabian Sea (roughly)
        arabian_sea = (
            (lat_2d < 20) & (lon_2d < 72)
        )
        mask = mask & ~arabian_sea

    else:
        # Would use shapefile or natural earth data for accurate mask
        raise NotImplementedError(
            "Detailed land mask requires additional data. Use simplified=True."
        )

    return mask.astype(np.float32)


def regrid_to_common_grid(
    datasets: dict,
    target_resolution: float = 4.0,
    apply_mask: bool = True,
) -> dict:
    """
    Regrid multiple datasets to common grid for comparison.

    Parameters
    ----------
    datasets : dict
        Dictionary of {model_name: xr.Dataset}
    target_resolution : float
        Target resolution in degrees
    apply_mask : bool
        Whether to apply land mask before regridding fine-resolution data

    Returns
    -------
    dict
        Dictionary of regridded datasets
    """
    regridded = {}

    for name, ds in datasets.items():
        # Get native resolution
        if "lat" in ds.dims:
            native_res = abs(ds["lat"].values[1] - ds["lat"].values[0])
        else:
            native_res = None

        logger.info(f"Regridding {name} from {native_res}° to {target_resolution}°")

        # Apply land mask for fine resolution data
        if apply_mask and native_res is not None and native_res <= 0.5:
            ds = apply_land_mask(ds)

        # Regrid
        regridded[name] = conservative_regrid(ds, target_resolution)

    return regridded


def compute_grid_cell_areas(
    lat: np.ndarray,
    lon: np.ndarray,
    earth_radius_km: float = 6371.0,
) -> np.ndarray:
    """
    Compute area of each grid cell in km².

    Parameters
    ----------
    lat : np.ndarray
        Latitude coordinates (cell centers)
    lon : np.ndarray
        Longitude coordinates (cell centers)
    earth_radius_km : float
        Earth radius in km

    Returns
    -------
    np.ndarray
        2D array of cell areas in km²
    """
    # Compute cell boundaries
    dlat = lat[1] - lat[0] if len(lat) > 1 else 1.0
    dlon = lon[1] - lon[0] if len(lon) > 1 else 1.0

    lat_bounds = np.zeros(len(lat) + 1)
    lat_bounds[0] = lat[0] - dlat / 2
    lat_bounds[1:] = lat + dlat / 2

    # Convert to radians
    lat_bounds_rad = np.deg2rad(lat_bounds)
    dlon_rad = np.deg2rad(dlon)

    # Area of each latitude band
    areas = np.zeros((len(lat), len(lon)))
    for i in range(len(lat)):
        # Area = R² * |sin(lat1) - sin(lat2)| * dlon
        area = (
            earth_radius_km**2
            * abs(np.sin(lat_bounds_rad[i + 1]) - np.sin(lat_bounds_rad[i]))
            * dlon_rad
        )
        areas[i, :] = area

    return areas
