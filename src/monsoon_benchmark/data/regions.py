"""
Region definitions for Indian monsoon analysis.

Defines boundaries for:
- Core Monsoon Zone (CMZ)
- India land area
- 4°×4° evaluation grid
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import xarray as xr


@dataclass
class BoundingBox:
    """Geographic bounding box."""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def contains(self, lat: float, lon: float) -> bool:
        """Check if point is within bounding box."""
        return (
            self.lat_min <= lat <= self.lat_max
            and self.lon_min <= lon <= self.lon_max
        )

    def to_slice(self) -> Tuple[slice, slice]:
        """Return as (lat_slice, lon_slice) for xarray selection."""
        return (
            slice(self.lat_min, self.lat_max),
            slice(self.lon_min, self.lon_max),
        )


# Core Monsoon Zone (CMZ) - region of coherent intraseasonal variability
# Includes states with >200 million agricultural workers
CMZ_BOUNDS = BoundingBox(
    lat_min=18.0,
    lat_max=28.0,
    lon_min=74.0,
    lon_max=86.0,
)

# Full India domain for analysis
INDIA_BOUNDS = BoundingBox(
    lat_min=6.0,
    lat_max=38.0,
    lon_min=66.0,
    lon_max=98.0,
)

# Webster-Yang Index averaging region
WYI_BOUNDS = BoundingBox(
    lat_min=0.0,
    lat_max=20.0,
    lon_min=40.0,
    lon_max=110.0,
)


def create_target_grid(
    resolution_deg: float = 4.0,
    bounds: BoundingBox = INDIA_BOUNDS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create target grid for regridding.

    Parameters
    ----------
    resolution_deg : float
        Grid resolution in degrees (default 4.0)
    bounds : BoundingBox
        Geographic bounds for grid

    Returns
    -------
    lat : np.ndarray
        Latitude coordinates (cell centers)
    lon : np.ndarray
        Longitude coordinates (cell centers)
    """
    # Create grid with cell centers at half-resolution offsets
    lat = np.arange(
        bounds.lat_min + resolution_deg / 2,
        bounds.lat_max,
        resolution_deg,
    )
    lon = np.arange(
        bounds.lon_min + resolution_deg / 2,
        bounds.lon_max,
        resolution_deg,
    )
    return lat, lon


def get_cmz_mask(
    lat: np.ndarray,
    lon: np.ndarray,
    bounds: BoundingBox = CMZ_BOUNDS,
) -> np.ndarray:
    """
    Create boolean mask for Core Monsoon Zone.

    Parameters
    ----------
    lat : np.ndarray
        Latitude coordinates
    lon : np.ndarray
        Longitude coordinates
    bounds : BoundingBox
        CMZ boundaries

    Returns
    -------
    mask : np.ndarray
        2D boolean array (lat, lon) where True = inside CMZ
    """
    lat_2d, lon_2d = np.meshgrid(lat, lon, indexing="ij")

    mask = (
        (lat_2d >= bounds.lat_min)
        & (lat_2d <= bounds.lat_max)
        & (lon_2d >= bounds.lon_min)
        & (lon_2d <= bounds.lon_max)
    )
    return mask


def get_cmz_grid_cells(resolution_deg: float = 4.0) -> list:
    """
    Get list of (lat, lon) tuples for CMZ grid cell centers.

    Parameters
    ----------
    resolution_deg : float
        Grid resolution in degrees

    Returns
    -------
    cells : list of tuples
        List of (lat_center, lon_center) for each CMZ grid cell
    """
    lat, lon = create_target_grid(resolution_deg, INDIA_BOUNDS)
    mask = get_cmz_mask(lat, lon, CMZ_BOUNDS)

    cells = []
    for i, lat_val in enumerate(lat):
        for j, lon_val in enumerate(lon):
            if mask[i, j]:
                cells.append((float(lat_val), float(lon_val)))

    return cells


def select_region(
    ds: xr.Dataset,
    bounds: BoundingBox,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> xr.Dataset:
    """
    Select geographic region from xarray Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    bounds : BoundingBox
        Geographic bounds
    lat_dim : str
        Name of latitude dimension
    lon_dim : str
        Name of longitude dimension

    Returns
    -------
    xr.Dataset
        Subset of dataset within bounds
    """
    return ds.sel({
        lat_dim: slice(bounds.lat_min, bounds.lat_max),
        lon_dim: slice(bounds.lon_min, bounds.lon_max),
    })


def compute_regional_mean(
    da: xr.DataArray,
    bounds: BoundingBox,
    weights: Optional[xr.DataArray] = None,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> xr.DataArray:
    """
    Compute area-weighted regional mean.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    bounds : BoundingBox
        Geographic bounds for averaging
    weights : xr.DataArray, optional
        Area weights (if None, uses cos(lat) weighting)
    lat_dim : str
        Name of latitude dimension
    lon_dim : str
        Name of longitude dimension

    Returns
    -------
    xr.DataArray
        Regional mean values
    """
    # Select region
    da_region = da.sel({
        lat_dim: slice(bounds.lat_min, bounds.lat_max),
        lon_dim: slice(bounds.lon_min, bounds.lon_max),
    })

    # Compute weights if not provided
    if weights is None:
        lat = da_region[lat_dim]
        weights = np.cos(np.deg2rad(lat))

    # Weighted mean
    return da_region.weighted(weights).mean(dim=[lat_dim, lon_dim])


# India state boundaries for reference (approximate centers)
INDIA_STATES = {
    "Kerala": (10.0, 76.5),
    "Maharashtra": (19.0, 75.0),
    "Madhya_Pradesh": (23.5, 78.0),
    "Uttar_Pradesh": (27.0, 81.0),
    "Rajasthan": (27.0, 74.0),
    "Gujarat": (22.5, 71.5),
    "Bihar": (25.5, 85.5),
    "West_Bengal": (23.0, 87.5),
    "Odisha": (20.5, 85.0),
    "Andhra_Pradesh": (16.0, 80.0),
    "Tamil_Nadu": (11.0, 78.5),
    "Karnataka": (15.0, 76.0),
}
