"""
WeatherBench2 hindcast loader.

Loads pre-computed hindcasts from Google WeatherBench2 dataset.
Data source: gs://weatherbench2/datasets

Reference:
    Masiwal et al., "Decision-oriented benchmarking to transform AI weather forecast access:
    Application to the Indian monsoon"
    https://arxiv.org/abs/2602.03767

Available models:
- GraphCast (DeepMind): 10-day forecasts, 0.25°
- FuXi (Fudan): 15-day forecasts, 0.25°
- Pangu-Weather (Huawei): Multiple lead times, 0.25°
- GenCast (DeepMind): Ensemble forecasts
- NeuralGCM (Google): Deterministic and ensemble
- ECMWF IFS HRES/ENS

Reference: https://weatherbench2.readthedocs.io/en/latest/data-guide.html
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import xarray as xr

from .base import ModelConfig, HindcastLoader

logger = logging.getLogger(__name__)

# WeatherBench2 bucket
WB2_BUCKET = "weatherbench2/datasets"

# Available models with their zarr paths and metadata
WB2_MODELS = {
    "graphcast": {
        "zarr_path": "graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours.zarr",
        "resolution": 0.25,
        "lead_time_hours": 240,  # 10 days
        "type": "deterministic",
        "time_step_hours": 12,
    },
    "fuxi": {
        "zarr_path": "fuxi/2020-1440x721.zarr",
        "resolution": 0.25,
        "lead_time_hours": 360,  # 15 days
        "type": "deterministic",
        "time_step_hours": 6,
    },
    "pangu": {
        "zarr_path": "pangu/2018-2022_0012_combined.zarr",
        "resolution": 0.25,
        "lead_time_hours": 168,  # 7 days
        "type": "deterministic",
        "time_step_hours": 6,
    },
    "gencast": {
        "zarr_path": "gencast/2020-1440x721.zarr",
        "resolution": 0.25,
        "lead_time_hours": 360,  # 15 days
        "type": "probabilistic",
        "ensemble_size": 50,
        "time_step_hours": 12,
    },
    "neuralgcm_deterministic": {
        "zarr_path": "neuralgcm_deterministic/2020-1440x721.zarr",
        "resolution": 0.25,
        "lead_time_hours": 240,
        "type": "deterministic",
        "time_step_hours": 6,
    },
    "neuralgcm_ens": {
        "zarr_path": "neuralgcm_ens/2020-1440x721.zarr",
        "resolution": 0.25,
        "lead_time_hours": 240,
        "type": "probabilistic",
        "ensemble_size": 50,
        "time_step_hours": 6,
    },
    "ifs_ens": {
        "zarr_path": "ens/2020-128x64_equiangular_with_poles_conservative.zarr",
        "resolution": 2.8,  # Regridded
        "lead_time_hours": 360,
        "type": "probabilistic",
        "ensemble_size": 51,
        "time_step_hours": 6,
    },
    "hres": {
        "zarr_path": "hres/2020-1440x721.zarr",
        "resolution": 0.25,
        "lead_time_hours": 240,
        "type": "deterministic",
        "time_step_hours": 6,
    },
}


class WeatherBench2Loader(HindcastLoader):
    """
    Loader for WeatherBench2 hindcasts from Google Cloud Storage.

    WeatherBench2 provides hindcasts from multiple AIWP models including
    GraphCast, FuXi, Pangu-Weather, GenCast, NeuralGCM, and ECMWF IFS.

    The data is organized as zarr stores with dimensions:
    - time: initialization times (typically every 12 hours in 2020)
    - prediction_timedelta: lead times (6 or 12 hourly)
    - latitude, longitude: spatial coordinates

    Parameters
    ----------
    model_name : str
        Model to load. Available: graphcast, fuxi, pangu, gencast,
        neuralgcm_deterministic, neuralgcm_ens, ifs_ens, hres
    data_dir : Path or str
        Local directory to cache downloaded hindcasts
    use_gcs_direct : bool
        If True, read directly from GCS (recommended, avoids large downloads)

    Example
    -------
    >>> loader = WeatherBench2Loader(model_name="fuxi")
    >>> loader.load_weights()
    >>> ds = loader.load_hindcast(datetime(2020, 6, 1), lead_days=15)
    >>> precip = ds['total_precipitation_6hr']  # 6-hourly accumulated precip
    """

    def __init__(
        self,
        model_name: str = "fuxi",
        data_dir: Union[str, Path] = "data/hindcasts/weatherbench2",
        use_gcs_direct: bool = True,
    ):
        if model_name not in WB2_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {list(WB2_MODELS.keys())}"
            )

        model_info = WB2_MODELS[model_name]

        config = ModelConfig(
            name=f"WeatherBench2-{model_name}",
            model_type=model_info["type"],
            ensemble_size=model_info.get("ensemble_size", 1),
            resolution_deg=model_info["resolution"],
        )

        super().__init__(config, data_dir)
        self.model_name = model_name
        self.model_info = model_info
        self.use_gcs_direct = use_gcs_direct
        self._gcs_available = None
        self._full_dataset = None

    def load_weights(self) -> None:
        """Initialize loader and check GCS availability."""
        self.hindcast_dir.mkdir(parents=True, exist_ok=True)

        if self.use_gcs_direct:
            self._check_gcs_availability()

        self._build_hindcast_index()
        self._is_loaded = True
        logger.info(f"WeatherBench2 {self.model_name} loader initialized")

    def _check_gcs_availability(self) -> None:
        """Check if GCS is accessible."""
        try:
            import gcsfs

            self._gcs_available = True
            logger.info("GCS direct access available")
        except ImportError:
            self._gcs_available = False
            logger.warning(
                "gcsfs not installed. Install with: pip install gcsfs zarr\n"
                "Will use local cache only."
            )

    def _build_hindcast_index(self) -> None:
        """Build index of locally cached hindcasts."""
        self._hindcast_index = {}

        model_dir = self.hindcast_dir / self.model_name
        if model_dir.exists():
            for nc_file in model_dir.glob("*.nc"):
                try:
                    date_str = nc_file.stem.split("-")[-1]
                    if len(date_str) == 8:
                        year = int(date_str[:4])
                        month = int(date_str[4:6])
                        day = int(date_str[6:8])
                        init_date = datetime(year, month, day)
                        self._hindcast_index[init_date] = nc_file
                except Exception:
                    pass

    def _load_full_dataset(self) -> xr.Dataset:
        """Load the full dataset from GCS (lazy loading with dask)."""
        if self._full_dataset is not None:
            return self._full_dataset

        import gcsfs

        fs = gcsfs.GCSFileSystem(token="anon")
        zarr_path = f"{WB2_BUCKET}/{self.model_info['zarr_path']}"

        logger.info(f"Opening {zarr_path}")
        self._full_dataset = xr.open_zarr(fs.get_mapper(zarr_path))

        return self._full_dataset

    def generate_forecast(
        self,
        init_time: datetime,
        lead_days: int,
        initial_conditions: Optional[xr.Dataset] = None,
    ) -> xr.Dataset:
        """Load pre-computed hindcast for given initialization time."""
        return self.load_hindcast(init_time, lead_days)

    def load_hindcast(
        self,
        init_time: datetime,
        lead_days: int = 10,
        variables: Optional[List[str]] = None,
        region: Optional[Dict[str, tuple]] = None,
    ) -> xr.Dataset:
        """
        Load hindcast for a specific initialization time.

        Parameters
        ----------
        init_time : datetime
            Forecast initialization time
        lead_days : int
            Number of lead days to load
        variables : list of str, optional
            Variables to load. Precipitation options:
            - 'total_precipitation_6hr' (6-hour accumulation)
            - 'total_precipitation_24hr_from_6hr' (daily accumulation)
        region : dict, optional
            Spatial subset: {"lat": (min, max), "lon": (min, max)}
            Default: India region

        Returns
        -------
        xr.Dataset
            Hindcast data with dims (prediction_timedelta, latitude, longitude)
        """
        if region is None:
            region = {"lat": (5, 40), "lon": (60, 100)}

        # Check max lead time
        max_lead_hours = self.model_info["lead_time_hours"]
        max_lead_days = max_lead_hours // 24
        lead_days = min(lead_days, max_lead_days)

        # Try local cache first
        if init_time in self._hindcast_index:
            ds = xr.open_dataset(self._hindcast_index[init_time])
        elif self.use_gcs_direct and self._gcs_available:
            ds = self._load_from_gcs(init_time, lead_days, variables, region)
        else:
            raise FileNotFoundError(
                f"No hindcast found for {init_time}. "
                f"Install gcsfs for GCS access: pip install gcsfs zarr"
            )

        return ds

    def _load_from_gcs(
        self,
        init_time: datetime,
        lead_days: int,
        variables: Optional[List[str]] = None,
        region: Optional[Dict[str, tuple]] = None,
    ) -> xr.Dataset:
        """Load hindcast from GCS for specific init time."""
        ds = self._load_full_dataset()

        # Find matching init time
        init_time64 = np.datetime64(init_time)

        # Select by time
        if "time" in ds.dims:
            # Find closest time
            time_diff = np.abs(ds.time.values - init_time64)
            closest_idx = np.argmin(time_diff)

            # Check if within 12 hours
            if time_diff[closest_idx] > np.timedelta64(12, "h"):
                raise FileNotFoundError(
                    f"No hindcast found for {init_time}. "
                    f"Closest available: {ds.time.values[closest_idx]}"
                )

            ds = ds.isel(time=closest_idx)

        # Select lead times
        max_timedelta = np.timedelta64(lead_days, "D")
        if "prediction_timedelta" in ds.dims:
            ds = ds.sel(prediction_timedelta=ds.prediction_timedelta <= max_timedelta)

        # Select variables
        if variables:
            available_vars = [v for v in variables if v in ds.data_vars]
            if available_vars:
                ds = ds[available_vars]

        # Select region
        if region:
            lat_min, lat_max = region["lat"]
            lon_min, lon_max = region["lon"]
            lat_name = "latitude" if "latitude" in ds.dims else "lat"
            lon_name = "longitude" if "longitude" in ds.dims else "lon"
            ds = ds.sel(
                {lat_name: slice(lat_min, lat_max), lon_name: slice(lon_min, lon_max)}
            )

        # Load into memory
        return ds.compute()

    def download_hindcast(
        self,
        init_time: datetime,
        lead_days: int = 15,
        variables: Optional[List[str]] = None,
        region: Optional[Dict[str, tuple]] = None,
    ) -> Path:
        """
        Download and cache a hindcast locally.

        Parameters
        ----------
        init_time : datetime
            Forecast initialization time
        lead_days : int
            Number of lead days to download
        variables : list of str, optional
            Variables to download
        region : dict, optional
            Spatial subset to download

        Returns
        -------
        Path
            Path to downloaded file
        """
        if variables is None:
            variables = ["total_precipitation_6hr", "total_precipitation_24hr_from_6hr"]

        ds = self._load_from_gcs(init_time, lead_days, variables, region)

        # Save locally
        model_dir = self.hindcast_dir / self.model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        date_str = init_time.strftime("%Y%m%d")
        out_path = model_dir / f"{self.model_name}-{date_str}.nc"

        ds.to_netcdf(out_path)
        logger.info(f"Downloaded hindcast to {out_path}")

        # Update index
        self._hindcast_index[init_time] = out_path

        return out_path

    def list_available_dates(
        self,
        year: int = 2020,
        months: Optional[List[int]] = None,
    ) -> List[datetime]:
        """
        List available initialization dates.

        Parameters
        ----------
        year : int
            Year to check
        months : list of int, optional
            Specific months to filter

        Returns
        -------
        list of datetime
            Available initialization dates
        """
        if not self._gcs_available:
            logger.warning("GCS not available. Install gcsfs.")
            return list(self._hindcast_index.keys())

        ds = self._load_full_dataset()

        if "time" not in ds.dims:
            return []

        dates = []
        for t in ds.time.values:
            dt = t.astype("datetime64[us]").astype(datetime)
            if dt.year == year:
                if months is None or dt.month in months:
                    dates.append(dt)

        return sorted(dates)

    def get_precipitation_variable(self) -> str:
        """Get the name of the precipitation variable for this model."""
        ds = self._load_full_dataset()

        # Check for common precipitation variable names
        precip_vars = [
            "total_precipitation_6hr",
            "total_precipitation_24hr_from_6hr",
            "total_precipitation",
            "tp",
        ]

        for var in precip_vars:
            if var in ds.data_vars:
                return var

        # Return first variable containing 'precip'
        for var in ds.data_vars:
            if "precip" in var.lower():
                return var

        logger.warning(f"No precipitation variable found. Available: {list(ds.data_vars)}")
        return None


def load_weatherbench2_era5(
    years: List[int],
    months: List[int] = [4, 5, 6, 7, 8, 9],
    variables: List[str] = ["total_precipitation"],
    region: Optional[Dict[str, tuple]] = None,
) -> xr.Dataset:
    """
    Load ERA5 reanalysis from WeatherBench2 as verification data.

    Parameters
    ----------
    years : list of int
        Years to load
    months : list of int
        Months to load
    variables : list of str
        Variables to load
    region : dict, optional
        Spatial subset

    Returns
    -------
    xr.Dataset
        ERA5 data from WeatherBench2
    """
    try:
        import gcsfs
    except ImportError:
        raise ImportError("gcsfs required. Install with: pip install gcsfs zarr")

    if region is None:
        region = {"lat": (5, 40), "lon": (60, 100)}

    fs = gcsfs.GCSFileSystem(token="anon")

    # List available ERA5 files
    era5_files = fs.ls(f"{WB2_BUCKET}/era5")
    logger.info(f"Available ERA5 datasets: {[Path(f).name for f in era5_files[:5]]}")

    # Try daily data first (more manageable)
    era5_path = f"{WB2_BUCKET}/era5_daily/2020-240x121_equiangular_with_poles_conservative.zarr"

    try:
        ds = xr.open_zarr(fs.get_mapper(era5_path))
    except FileNotFoundError:
        # Fall back to 6-hourly
        era5_path = f"{WB2_BUCKET}/era5/2020-1440x721.zarr"
        ds = xr.open_zarr(fs.get_mapper(era5_path))

    # Select variables
    if variables:
        available_vars = [v for v in variables if v in ds.data_vars]
        if available_vars:
            ds = ds[available_vars]

    # Select time range
    time_mask = np.zeros(len(ds.time), dtype=bool)
    for year in years:
        for month in months:
            year_month_mask = (
                (ds.time.dt.year == year) & (ds.time.dt.month == month)
            ).values
            time_mask |= year_month_mask

    ds = ds.isel(time=time_mask)

    # Select region
    lat_name = "latitude" if "latitude" in ds.dims else "lat"
    lon_name = "longitude" if "longitude" in ds.dims else "lon"
    ds = ds.sel(
        {
            lat_name: slice(region["lat"][0], region["lat"][1]),
            lon_name: slice(region["lon"][0], region["lon"][1]),
        }
    )

    return ds
