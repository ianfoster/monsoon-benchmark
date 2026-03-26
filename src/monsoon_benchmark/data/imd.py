"""
IMD (India Meteorological Department) rainfall data handling.

Data source: https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_1_NetCDF.html
Resolution: 1° × 1° gridded daily rainfall
Period: 1901-2024
"""

import logging
from pathlib import Path
from typing import Optional, Union
from datetime import datetime

import numpy as np
import xarray as xr
import pandas as pd
import requests
from tqdm import tqdm

from monsoon_benchmark.data.regions import INDIA_BOUNDS, BoundingBox

logger = logging.getLogger(__name__)


class IMDDataLoader:
    """
    Loader for IMD gridded rainfall data.

    The IMD dataset provides daily rainfall observations from a network of
    rain-gauge stations, gridded to 1° × 1° resolution. This is the ground
    truth dataset used for monsoon onset validation.

    Parameters
    ----------
    data_dir : str or Path
        Directory to store downloaded data
    resolution : float
        Grid resolution (1.0 for 1° data)
    """

    # IMD data URL template
    BASE_URL = "https://www.imdpune.gov.in/cmpg/Griddata"
    RAINFALL_FILE = "Rainfall_ind{year}_rfp25.nc"  # 0.25° resolution
    RAINFALL_FILE_1DEG = "RF{year}.nc"  # 1° resolution

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/imd",
        resolution: float = 1.0,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.resolution = resolution
        self._cache = {}

    def download_year(self, year: int, force: bool = False) -> Path:
        """
        Download IMD rainfall data for a specific year.

        Parameters
        ----------
        year : int
            Year to download
        force : bool
            If True, re-download even if file exists

        Returns
        -------
        Path
            Path to downloaded file
        """
        filename = f"imd_rainfall_{year}_{self.resolution}deg.nc"
        filepath = self.data_dir / filename

        if filepath.exists() and not force:
            logger.debug(f"Using cached file: {filepath}")
            return filepath

        # Note: In practice, IMD data may require manual download or API access
        # This is a placeholder for the download logic
        logger.warning(
            f"IMD data for {year} not found at {filepath}. "
            "Please download manually from: "
            "https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_1_NetCDF.html"
        )

        return filepath

    def load_year(self, year: int) -> xr.Dataset:
        """
        Load IMD rainfall data for a specific year.

        Parameters
        ----------
        year : int
            Year to load

        Returns
        -------
        xr.Dataset
            Dataset with daily rainfall on 1° grid
        """
        if year in self._cache:
            return self._cache[year]

        filepath = self.download_year(year)

        if not filepath.exists():
            raise FileNotFoundError(
                f"IMD data file not found: {filepath}. "
                "Please download from IMD website."
            )

        ds = xr.open_dataset(filepath)

        # Standardize dimension names
        ds = self._standardize_dimensions(ds)

        # Standardize variable names
        ds = self._standardize_variables(ds)

        self._cache[year] = ds
        return ds

    def load_period(
        self,
        start_year: int,
        end_year: int,
        months: Optional[list] = None,
    ) -> xr.Dataset:
        """
        Load IMD rainfall data for a period of years.

        Parameters
        ----------
        start_year : int
            First year to load
        end_year : int
            Last year to load (inclusive)
        months : list, optional
            List of months to include (1-12). If None, load all months.

        Returns
        -------
        xr.Dataset
            Combined dataset for all years
        """
        datasets = []

        for year in tqdm(range(start_year, end_year + 1), desc="Loading IMD data"):
            try:
                ds = self.load_year(year)
                if months is not None:
                    ds = ds.sel(time=ds.time.dt.month.isin(months))
                datasets.append(ds)
            except FileNotFoundError as e:
                logger.warning(f"Skipping year {year}: {e}")

        if not datasets:
            raise ValueError(f"No data found for period {start_year}-{end_year}")

        return xr.concat(datasets, dim="time")

    def load_monsoon_season(
        self,
        year: int,
        start_month: int = 4,
        end_month: int = 9,
    ) -> xr.Dataset:
        """
        Load data for monsoon season (Apr-Sep by default).

        Parameters
        ----------
        year : int
            Year to load
        start_month : int
            Start month (default 4 = April)
        end_month : int
            End month (default 9 = September)

        Returns
        -------
        xr.Dataset
            Dataset for monsoon season
        """
        ds = self.load_year(year)

        # Select monsoon months
        mask = (ds.time.dt.month >= start_month) & (ds.time.dt.month <= end_month)
        return ds.sel(time=mask)

    def _standardize_dimensions(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize dimension names to lat, lon, time."""
        rename_map = {}

        # Common latitude names
        for lat_name in ["LATITUDE", "latitude", "LAT", "lat", "y"]:
            if lat_name in ds.dims and lat_name != "lat":
                rename_map[lat_name] = "lat"
                break

        # Common longitude names
        for lon_name in ["LONGITUDE", "longitude", "LON", "lon", "x"]:
            if lon_name in ds.dims and lon_name != "lon":
                rename_map[lon_name] = "lon"
                break

        # Common time names
        for time_name in ["TIME", "Time", "time", "t"]:
            if time_name in ds.dims and time_name != "time":
                rename_map[time_name] = "time"
                break

        if rename_map:
            ds = ds.rename(rename_map)

        return ds

    def _standardize_variables(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize variable names."""
        # Common rainfall variable names
        rainfall_names = ["rf", "RF", "rainfall", "RAINFALL", "precip", "PRECIP", "pr"]

        for var_name in rainfall_names:
            if var_name in ds.data_vars:
                if var_name != "rainfall":
                    ds = ds.rename({var_name: "rainfall"})
                break

        # Ensure rainfall units are mm/day
        if "rainfall" in ds.data_vars:
            ds["rainfall"].attrs["units"] = "mm/day"
            ds["rainfall"].attrs["long_name"] = "Daily rainfall"

        return ds

    def get_climatology(
        self,
        start_year: int = 1901,
        end_year: int = 2024,
        groupby: str = "dayofyear",
    ) -> xr.DataArray:
        """
        Compute climatological mean rainfall.

        Parameters
        ----------
        start_year : int
            Start year for climatology
        end_year : int
            End year for climatology
        groupby : str
            Grouping for climatology ("dayofyear" or "month")

        Returns
        -------
        xr.DataArray
            Climatological mean rainfall
        """
        ds = self.load_period(start_year, end_year)

        if groupby == "dayofyear":
            clim = ds["rainfall"].groupby("time.dayofyear").mean(dim="time")
        elif groupby == "month":
            clim = ds["rainfall"].groupby("time.month").mean(dim="time")
        else:
            raise ValueError(f"Invalid groupby: {groupby}")

        clim.attrs["climatology_period"] = f"{start_year}-{end_year}"
        return clim


def create_synthetic_imd_data(
    year: int,
    output_path: Path,
    resolution: float = 1.0,
) -> xr.Dataset:
    """
    Create synthetic IMD-like data for testing.

    This generates realistic-looking rainfall data with monsoon seasonality
    for testing the benchmarking framework.

    Parameters
    ----------
    year : int
        Year to generate
    output_path : Path
        Path to save the NetCDF file
    resolution : float
        Grid resolution in degrees

    Returns
    -------
    xr.Dataset
        Synthetic rainfall dataset
    """
    # Create coordinates
    lat = np.arange(6.5, 38.5, resolution)
    lon = np.arange(66.5, 98.5, resolution)
    time = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")

    # Create synthetic rainfall with monsoon pattern
    lat_2d, lon_2d = np.meshgrid(lat, lon, indexing="ij")

    # Base rainfall (increases towards south and east)
    base = 2.0 + (38 - lat_2d) * 0.1 + (lon_2d - 66) * 0.05

    # Monsoon seasonality (peak in July-August)
    doy = np.arange(1, len(time) + 1)
    monsoon_factor = np.exp(-((doy - 200) ** 2) / (2 * 50**2))

    # Random variability
    np.random.seed(year)
    noise = np.random.exponential(scale=1.0, size=(len(time), len(lat), len(lon)))

    # Combine
    rainfall = base[np.newaxis, :, :] * monsoon_factor[:, np.newaxis, np.newaxis] * 5 * noise

    # Create dataset
    ds = xr.Dataset(
        {
            "rainfall": (["time", "lat", "lon"], rainfall.astype(np.float32)),
        },
        coords={
            "time": time,
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "title": f"Synthetic IMD-like rainfall data for {year}",
            "source": "Generated for testing",
            "resolution": f"{resolution} degrees",
        },
    )

    ds["rainfall"].attrs = {
        "units": "mm/day",
        "long_name": "Daily rainfall (synthetic)",
    }

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)

    return ds
