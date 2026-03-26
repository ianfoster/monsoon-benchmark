"""
ERA5 reanalysis data handling.

Data source: Copernicus Climate Data Store (CDS)
Used for:
- Model initialization
- Webster-Yang Index computation (u-wind at 200/850 hPa)
- Verification of circulation patterns
"""

import logging
from pathlib import Path
from typing import Optional, Union, List
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import cdsapi, but don't fail if not available
try:
    import cdsapi
    HAS_CDSAPI = True
except ImportError:
    HAS_CDSAPI = False
    logger.warning("cdsapi not installed. ERA5 download functionality unavailable.")


class ERA5DataLoader:
    """
    Loader for ERA5 reanalysis data from Copernicus CDS.

    Parameters
    ----------
    data_dir : str or Path
        Directory to store downloaded data
    """

    def __init__(self, data_dir: Union[str, Path] = "data/era5"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._cache = {}

        if HAS_CDSAPI:
            try:
                self.client = cdsapi.Client()
            except Exception as e:
                logger.warning(f"Could not initialize CDS client: {e}")
                self.client = None
        else:
            self.client = None

    def download_pressure_levels(
        self,
        year: int,
        month: int,
        variables: List[str] = ["u_component_of_wind"],
        pressure_levels: List[int] = [200, 850],
        area: List[float] = [40, 30, -10, 120],  # N, W, S, E
        force: bool = False,
    ) -> Path:
        """
        Download ERA5 pressure level data.

        Parameters
        ----------
        year : int
            Year to download
        month : int
            Month to download
        variables : list
            Variables to download
        pressure_levels : list
            Pressure levels in hPa
        area : list
            Geographic area [North, West, South, East]
        force : bool
            If True, re-download even if file exists

        Returns
        -------
        Path
            Path to downloaded file
        """
        if self.client is None:
            raise RuntimeError(
                "CDS client not available. Install cdsapi and configure credentials."
            )

        filename = f"era5_pl_{year}{month:02d}_{'_'.join(map(str, pressure_levels))}hPa.nc"
        filepath = self.data_dir / filename

        if filepath.exists() and not force:
            logger.debug(f"Using cached file: {filepath}")
            return filepath

        logger.info(f"Downloading ERA5 data for {year}-{month:02d}")

        self.client.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "variable": variables,
                "pressure_level": [str(p) for p in pressure_levels],
                "year": str(year),
                "month": f"{month:02d}",
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "area": area,
                "format": "netcdf",
            },
            str(filepath),
        )

        return filepath

    def download_for_wyi(
        self,
        year: int,
        months: List[int] = [4, 5, 6, 7, 8, 9],
        force: bool = False,
    ) -> List[Path]:
        """
        Download ERA5 data needed for Webster-Yang Index computation.

        Parameters
        ----------
        year : int
            Year to download
        months : list
            Months to download (default Apr-Sep for monsoon)
        force : bool
            If True, re-download even if files exist

        Returns
        -------
        list of Path
            Paths to downloaded files
        """
        filepaths = []
        for month in months:
            try:
                fp = self.download_pressure_levels(
                    year=year,
                    month=month,
                    variables=["u_component_of_wind"],
                    pressure_levels=[200, 850],
                    area=[40, 30, -10, 120],  # Covers WYI region
                    force=force,
                )
                filepaths.append(fp)
            except Exception as e:
                logger.error(f"Failed to download ERA5 for {year}-{month:02d}: {e}")

        return filepaths

    def download_precipitation(
        self,
        year: int,
        month: int,
        area: List[float] = [40, 60, 5, 100],  # N, W, S, E - covers India
        force: bool = False,
    ) -> Path:
        """
        Download ERA5 daily precipitation data.

        Parameters
        ----------
        year : int
            Year to download
        month : int
            Month to download
        area : list
            Geographic area [North, West, South, East]
        force : bool
            If True, re-download even if file exists

        Returns
        -------
        Path
            Path to downloaded file
        """
        if self.client is None:
            raise RuntimeError(
                "CDS client not available. Install cdsapi and configure credentials."
            )

        filename = f"era5_precip_{year}{month:02d}.nc"
        filepath = self.data_dir / filename

        if filepath.exists() and not force:
            logger.debug(f"Using cached file: {filepath}")
            return filepath

        logger.info(f"Downloading ERA5 precipitation for {year}-{month:02d}")

        # Download total precipitation from single levels
        self.client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": "total_precipitation",
                "year": str(year),
                "month": f"{month:02d}",
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "area": area,
                "format": "netcdf",
            },
            str(filepath),
        )

        return filepath

    def download_precipitation_years(
        self,
        years: List[int],
        months: List[int] = [4, 5, 6, 7, 8, 9],
        force: bool = False,
    ) -> List[Path]:
        """
        Download ERA5 precipitation for multiple years.

        Parameters
        ----------
        years : list of int
            Years to download
        months : list of int
            Months to download
        force : bool
            If True, re-download even if files exist

        Returns
        -------
        list of Path
            Paths to downloaded files
        """
        filepaths = []
        for year in years:
            for month in months:
                try:
                    fp = self.download_precipitation(
                        year=year,
                        month=month,
                        force=force,
                    )
                    filepaths.append(fp)
                except Exception as e:
                    logger.error(f"Failed to download precip for {year}-{month:02d}: {e}")
        return filepaths

    def load_precipitation(
        self,
        years: List[int],
        months: List[int] = [4, 5, 6, 7, 8, 9],
    ) -> xr.DataArray:
        """
        Load ERA5 precipitation and convert to daily mm/day.

        ERA5 total_precipitation is in meters, accumulated over each hour.
        We sum the 4 6-hourly values and convert to mm/day.

        Parameters
        ----------
        years : list of int
            Years to load
        months : list of int
            Months to load

        Returns
        -------
        xr.DataArray
            Daily precipitation in mm/day
        """
        datasets = []

        for year in years:
            for month in months:
                filename = f"era5_precip_{year}{month:02d}.nc"
                filepath = self.data_dir / filename

                if not filepath.exists():
                    logger.warning(f"File not found: {filepath}")
                    continue

                ds = xr.open_dataset(filepath)
                ds = self._standardize_dimensions(ds)

                # Rename variable if needed
                for name in ["tp", "total_precipitation"]:
                    if name in ds.data_vars:
                        ds = ds.rename({name: "precipitation"})
                        break

                datasets.append(ds)

        if not datasets:
            raise FileNotFoundError("No ERA5 precipitation files found")

        combined = xr.concat(datasets, dim="time")

        # Convert from m (accumulated per hour) to mm/day
        # Sum 6-hourly values to daily and convert m -> mm
        daily = combined["precipitation"].resample(time="1D").sum() * 1000

        daily.attrs["units"] = "mm/day"
        daily.attrs["long_name"] = "Daily precipitation"

        return daily

    def load_wyi_data(
        self,
        year: int,
        months: List[int] = [4, 5, 6, 7, 8, 9],
    ) -> xr.Dataset:
        """
        Load ERA5 data for Webster-Yang Index computation.

        Parameters
        ----------
        year : int
            Year to load
        months : list
            Months to load

        Returns
        -------
        xr.Dataset
            Dataset with u-wind at 200 and 850 hPa
        """
        cache_key = (year, tuple(months))
        if cache_key in self._cache:
            return self._cache[cache_key]

        datasets = []
        for month in months:
            filename = f"era5_pl_{year}{month:02d}_200_850hPa.nc"
            filepath = self.data_dir / filename

            if not filepath.exists():
                logger.warning(f"File not found: {filepath}. Attempting download...")
                try:
                    self.download_pressure_levels(
                        year=year,
                        month=month,
                        variables=["u_component_of_wind"],
                        pressure_levels=[200, 850],
                    )
                except Exception as e:
                    logger.error(f"Could not download data: {e}")
                    continue

            if filepath.exists():
                ds = xr.open_dataset(filepath)
                ds = self._standardize_dimensions(ds)
                datasets.append(ds)

        if not datasets:
            raise FileNotFoundError(
                f"No ERA5 data found for {year}. Please download or generate data."
            )

        combined = xr.concat(datasets, dim="time")
        self._cache[cache_key] = combined
        return combined

    def _standardize_dimensions(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize dimension and variable names."""
        rename_map = {}

        # Latitude
        for name in ["latitude", "LATITUDE", "LAT"]:
            if name in ds.dims:
                rename_map[name] = "lat"
                break

        # Longitude
        for name in ["longitude", "LONGITUDE", "LON"]:
            if name in ds.dims:
                rename_map[name] = "lon"
                break

        # Pressure level
        for name in ["level", "pressure_level", "plev", "isobaricInhPa"]:
            if name in ds.dims:
                rename_map[name] = "level"
                break

        if rename_map:
            ds = ds.rename(rename_map)

        # Standardize variable names
        var_rename = {}
        for name in ["u", "U", "u_component_of_wind"]:
            if name in ds.data_vars and name != "u":
                var_rename[name] = "u"
                break

        if var_rename:
            ds = ds.rename(var_rename)

        return ds

    def get_daily_mean(self, ds: xr.Dataset) -> xr.Dataset:
        """Compute daily mean from sub-daily data."""
        return ds.resample(time="1D").mean()


def create_synthetic_era5_data(
    year: int,
    output_dir: Path,
    months: List[int] = [4, 5, 6, 7, 8, 9],
) -> List[Path]:
    """
    Create synthetic ERA5-like data for testing.

    Parameters
    ----------
    year : int
        Year to generate
    output_dir : Path
        Directory to save files
    months : list
        Months to generate

    Returns
    -------
    list of Path
        Paths to generated files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepaths = []

    for month in months:
        # Create coordinates
        lat = np.arange(-10, 41, 0.25)
        lon = np.arange(30, 121, 0.25)
        level = np.array([200, 850])

        # Days in month
        if month in [4, 6, 9]:
            n_days = 30
        elif month == 2:
            n_days = 29 if year % 4 == 0 else 28
        else:
            n_days = 31

        time = pd.date_range(
            f"{year}-{month:02d}-01",
            periods=n_days * 4,  # 4 times per day
            freq="6H",
        )

        # Create synthetic u-wind with monsoon pattern
        np.random.seed(year * 100 + month)

        # Upper level westerlies (stronger in summer over India)
        lat_2d, lon_2d = np.meshgrid(lat, lon, indexing="ij")
        u200_base = 10 + 5 * np.sin(np.deg2rad(lat_2d - 10))

        # Lower level easterlies (monsoon flow)
        u850_base = -5 - 3 * np.sin(np.deg2rad(lat_2d))

        # Time variation (monsoon onset strengthening)
        doy = np.array([(t.dayofyear - 120) for t in time.to_pydatetime()])
        monsoon_strength = 1 + 0.5 * np.tanh((doy - 30) / 20)

        # Random variability
        noise_200 = np.random.normal(0, 2, (len(time), len(lat), len(lon)))
        noise_850 = np.random.normal(0, 1.5, (len(time), len(lat), len(lon)))

        u200 = (
            u200_base[np.newaxis, :, :]
            * monsoon_strength[:, np.newaxis, np.newaxis]
            + noise_200
        )
        u850 = (
            u850_base[np.newaxis, :, :]
            * monsoon_strength[:, np.newaxis, np.newaxis]
            + noise_850
        )

        # Stack levels
        u = np.stack([u200, u850], axis=1)

        # Create dataset
        ds = xr.Dataset(
            {"u": (["time", "level", "lat", "lon"], u.astype(np.float32))},
            coords={
                "time": time,
                "level": level,
                "lat": lat,
                "lon": lon,
            },
            attrs={
                "title": f"Synthetic ERA5-like wind data for {year}-{month:02d}",
                "source": "Generated for testing",
            },
        )

        ds["u"].attrs = {
            "units": "m/s",
            "long_name": "U component of wind",
        }
        ds["level"].attrs = {"units": "hPa"}

        # Save
        filename = f"era5_pl_{year}{month:02d}_200_850hPa.nc"
        filepath = output_dir / filename
        ds.to_netcdf(filepath)
        filepaths.append(filepath)

        logger.info(f"Created synthetic ERA5 data: {filepath}")

    return filepaths


def download_era5_for_wyi(
    years: List[int],
    data_dir: Union[str, Path] = "data/era5",
    months: List[int] = [4, 5, 6, 7, 8, 9],
) -> None:
    """
    Download ERA5 u-wind data needed for Webster-Yang Index.

    Requires cdsapi to be installed and configured with CDS credentials.
    Get credentials at: https://cds.climate.copernicus.eu/

    Parameters
    ----------
    years : list of int
        Years to download (e.g., [2019, 2020, 2021, 2022, 2023, 2024])
    data_dir : str or Path
        Directory to save downloaded files
    months : list of int
        Months to download (default: Apr-Sep for monsoon season)

    Example
    -------
    >>> from monsoon_benchmark.data.era5 import download_era5_for_wyi
    >>> download_era5_for_wyi([2019, 2020, 2021, 2022, 2023, 2024])
    """
    if not HAS_CDSAPI:
        raise ImportError(
            "cdsapi is required for downloading ERA5 data.\n"
            "Install with: pip install cdsapi\n"
            "Then configure credentials: https://cds.climate.copernicus.eu/api-how-to"
        )

    loader = ERA5DataLoader(data_dir=data_dir)

    for year in years:
        print(f"Downloading ERA5 data for {year}...")
        try:
            loader.download_for_wyi(year=year, months=months)
            print(f"  Completed {year}")
        except Exception as e:
            print(f"  Failed {year}: {e}")


def download_era5_precipitation(
    years: List[int],
    data_dir: Union[str, Path] = "data/era5",
    months: List[int] = [4, 5, 6, 7, 8, 9],
) -> None:
    """
    Download ERA5 precipitation data over India.

    This provides an alternative to IMD rainfall data for testing
    the benchmarking framework. ERA5 precipitation is reanalysis-based,
    not ground-truth observations like IMD.

    Parameters
    ----------
    years : list of int
        Years to download
    data_dir : str or Path
        Directory to save downloaded files
    months : list of int
        Months to download (default: Apr-Sep for monsoon season)

    Example
    -------
    >>> from monsoon_benchmark.data.era5 import download_era5_precipitation
    >>> download_era5_precipitation([2019, 2020, 2021, 2022, 2023, 2024])
    """
    if not HAS_CDSAPI:
        raise ImportError(
            "cdsapi is required for downloading ERA5 data.\n"
            "Install with: pip install cdsapi\n"
            "Then configure credentials: https://cds.climate.copernicus.eu/api-how-to"
        )

    loader = ERA5DataLoader(data_dir=data_dir)

    total = len(years) * len(months)
    count = 0

    for year in years:
        for month in months:
            count += 1
            print(f"[{count}/{total}] Downloading ERA5 precip {year}-{month:02d}...")
            try:
                loader.download_precipitation(year=year, month=month)
                print(f"  Done")
            except Exception as e:
                print(f"  Failed: {e}")
