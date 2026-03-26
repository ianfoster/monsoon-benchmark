"""
FuXi-S2S hindcast loader.

Loads pre-computed FuXi-S2S hindcasts from HuggingFace.
Dataset: https://huggingface.co/datasets/FudanFuXi/FuXi-S2S

Reference:
    Masiwal et al., "Decision-oriented benchmarking to transform AI weather forecast access:
    Application to the Indian monsoon"
    https://arxiv.org/abs/2602.03767

FuXi-S2S provides:
- Hindcasts from 2002-2021
- Twice-weekly initializations (Monday/Thursday)
- 42-day lead time
- 0.25° resolution
- Variables: tp (total precipitation), t2m, u10, v10, msl, etc.

Note: Files are distributed as .7z archives (YYYYMMDD.7z) containing NetCDF files.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import xarray as xr

from .base import ModelConfig, HindcastLoader

logger = logging.getLogger(__name__)

# FuXi-S2S dataset info
FUXI_S2S_REPO = "FudanFuXi/FuXi-S2S"
FUXI_S2S_YEARS = list(range(2002, 2022))  # 2002-2021
FUXI_S2S_LEAD_DAYS = 42
FUXI_S2S_RESOLUTION = 0.25


class FuXiS2SLoader(HindcastLoader):
    """
    Loader for FuXi-S2S pre-computed hindcasts from HuggingFace.

    FuXi-S2S provides subseasonal forecasts initialized twice weekly
    for the period 2002-2021.

    Parameters
    ----------
    data_dir : Path or str
        Directory to store downloaded hindcasts
    download : bool
        Whether to automatically download missing hindcasts

    Example
    -------
    >>> loader = FuXiS2SLoader(data_dir="data/hindcasts/fuxi_s2s")
    >>> loader.load_weights()  # Build index of available hindcasts
    >>> forecast = loader.load_hindcast(datetime(2019, 5, 6), lead_days=35)
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/hindcasts/fuxi_s2s",
        download: bool = False,
    ):
        config = ModelConfig(
            name="FuXi-S2S",
            model_type="deterministic",  # FuXi-S2S provides single forecasts
            ensemble_size=1,
            resolution_deg=FUXI_S2S_RESOLUTION,
            training_end_year=2021,
        )
        super().__init__(config, data_dir)
        self.download = download
        self._available_dates = {}

    def load_weights(self) -> None:
        """Build index of available hindcasts."""
        self.hindcast_dir.mkdir(parents=True, exist_ok=True)
        self._build_hindcast_index()
        self._is_loaded = True
        logger.info(f"FuXi-S2S loader initialized with {len(self._hindcast_index)} hindcasts")

    def _build_hindcast_index(self) -> None:
        """Scan hindcast directory and build index of available files."""
        self._hindcast_index = {}

        # Look for NetCDF files in the data directory
        for nc_file in self.hindcast_dir.glob("*.nc"):
            # Parse init date from filename
            # Expected format: fuxi_s2s_YYYYMMDD.nc or similar
            try:
                init_date = self._parse_filename_date(nc_file.name)
                if init_date:
                    self._hindcast_index[init_date] = nc_file
            except Exception as e:
                logger.debug(f"Could not parse {nc_file}: {e}")

        # Also check for zarr stores
        for zarr_dir in self.hindcast_dir.glob("*.zarr"):
            try:
                init_date = self._parse_filename_date(zarr_dir.name)
                if init_date:
                    self._hindcast_index[init_date] = zarr_dir
            except Exception:
                pass

    def _parse_filename_date(self, filename: str) -> Optional[datetime]:
        """Parse initialization date from filename."""
        import re

        # Try various patterns
        patterns = [
            r"(\d{4})(\d{2})(\d{2})",  # YYYYMMDD
            r"(\d{4})-(\d{2})-(\d{2})",  # YYYY-MM-DD
            r"(\d{4})_(\d{2})_(\d{2})",  # YYYY_MM_DD
        ]

        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
                return datetime(year, month, day)
        return None

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
        lead_days: int = 35,
        variables: Optional[List[str]] = None,
    ) -> xr.Dataset:
        """
        Load FuXi-S2S hindcast for a specific initialization time.

        Parameters
        ----------
        init_time : datetime
            Forecast initialization time
        lead_days : int
            Number of lead days to load (max 42)
        variables : list of str, optional
            Variables to load (default: ['tp'] for total precipitation)

        Returns
        -------
        xr.Dataset
            Hindcast data with dimensions (time, lat, lon)
        """
        if variables is None:
            variables = ["tp"]  # Total precipitation

        lead_days = min(lead_days, FUXI_S2S_LEAD_DAYS)

        # Check if hindcast exists locally
        hindcast_file = self._find_hindcast_file(init_time)

        if hindcast_file is None:
            if self.download:
                hindcast_file = self._download_hindcast(init_time)
            else:
                raise FileNotFoundError(
                    f"No hindcast found for {init_time}. "
                    f"Set download=True or download manually from {FUXI_S2S_REPO}"
                )

        # Load the data
        if hindcast_file.suffix == ".zarr" or hindcast_file.is_dir():
            ds = xr.open_zarr(hindcast_file)
        else:
            ds = xr.open_dataset(hindcast_file)

        # Select variables if specified
        if variables:
            available_vars = [v for v in variables if v in ds.data_vars]
            if available_vars:
                ds = ds[available_vars]

        # Select lead times
        if "lead_time" in ds.dims:
            # Select up to lead_days
            max_lead = timedelta(days=lead_days)
            ds = ds.sel(lead_time=ds.lead_time <= max_lead)
        elif "time" in ds.dims:
            # Assume time is forecast valid time
            start_time = init_time
            end_time = init_time + timedelta(days=lead_days)
            ds = ds.sel(time=slice(start_time, end_time))

        return ds

    def _find_hindcast_file(self, init_time: datetime) -> Optional[Path]:
        """Find hindcast file for given initialization time."""
        # First check index
        if init_time in self._hindcast_index:
            return self._hindcast_index[init_time]

        # Try to find by date pattern
        date_str = init_time.strftime("%Y%m%d")
        patterns = [
            f"fuxi_s2s_{date_str}.nc",
            f"fuxi_s2s_{date_str}.zarr",
            f"{date_str}.nc",
            f"{date_str}.zarr",
            f"tp_{date_str}.nc",
        ]

        for pattern in patterns:
            candidate = self.hindcast_dir / pattern
            if candidate.exists():
                return candidate

        return None

    def _download_hindcast(self, init_time: datetime) -> Path:
        """Download and extract hindcast from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for downloading FuXi-S2S hindcasts. "
                "Install with: pip install huggingface-hub"
            )

        date_str = init_time.strftime("%Y%m%d")

        # FuXi-S2S files are stored as .7z archives named YYYYMMDD.7z
        filename = f"{date_str}.7z"

        try:
            # Download the 7z archive
            archive_path = hf_hub_download(
                repo_id=FUXI_S2S_REPO,
                filename=filename,
                repo_type="dataset",
                local_dir=self.hindcast_dir,
            )
            archive_path = Path(archive_path)

            # Extract the archive
            extracted_path = self._extract_7z(archive_path, date_str)
            return extracted_path

        except Exception as e:
            raise FileNotFoundError(
                f"Could not download hindcast for {init_time} from {FUXI_S2S_REPO}: {e}"
            )

    def _extract_7z(self, archive_path: Path, date_str: str) -> Path:
        """Extract 7z archive and return path to NetCDF file."""
        import subprocess

        extract_dir = self.hindcast_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)

        # Try using 7z command line tool
        try:
            subprocess.run(
                ["7z", "x", "-y", f"-o{extract_dir}", str(archive_path)],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Try py7zr Python library
            try:
                import py7zr

                with py7zr.SevenZipFile(archive_path, mode="r") as z:
                    z.extractall(path=extract_dir)
            except ImportError:
                raise ImportError(
                    "7z extraction requires either '7z' command line tool or py7zr library. "
                    "Install with: brew install p7zip  OR  pip install py7zr"
                )

        # Find the extracted NetCDF file
        nc_files = list(extract_dir.glob(f"*{date_str}*.nc")) + list(
            extract_dir.glob("*.nc")
        )
        if nc_files:
            return nc_files[0]

        # If no .nc found, check for other formats
        zarr_dirs = list(extract_dir.glob("*.zarr"))
        if zarr_dirs:
            return zarr_dirs[0]

        raise FileNotFoundError(f"No data files found after extracting {archive_path}")

    def list_available_dates(self, year: Optional[int] = None) -> List[datetime]:
        """
        List initialization dates available in the index.

        Parameters
        ----------
        year : int, optional
            Filter by year

        Returns
        -------
        list of datetime
            Available initialization dates
        """
        dates = list(self._hindcast_index.keys())
        if year:
            dates = [d for d in dates if d.year == year]
        return sorted(dates)

    def get_monsoon_init_dates(self, year: int) -> List[datetime]:
        """
        Get initialization dates relevant for monsoon onset forecasting.

        Returns dates from May 2 through June 30 (twice weekly).

        Parameters
        ----------
        year : int
            Year to get dates for

        Returns
        -------
        list of datetime
            Initialization dates for monsoon onset period
        """
        # Generate expected twice-weekly dates (Monday and Thursday)
        start = datetime(year, 5, 2)
        end = datetime(year, 6, 30)

        init_dates = []
        current = start

        while current <= end:
            # Check if Monday (0) or Thursday (3)
            if current.weekday() in [0, 3]:
                if current in self._hindcast_index:
                    init_dates.append(current)
            current += timedelta(days=1)

        return sorted(init_dates)


def download_fuxi_s2s_hindcasts(
    years: List[int],
    data_dir: Union[str, Path] = "data/hindcasts/fuxi_s2s",
    months: List[int] = [4, 5, 6, 7],
) -> None:
    """
    Download FuXi-S2S hindcasts for specified years.

    Parameters
    ----------
    years : list of int
        Years to download (2002-2021 available)
    data_dir : Path or str
        Directory to save hindcasts
    months : list of int
        Months to download (default: Apr-Jul for monsoon)
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface-hub"
        )

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()

    # List files in the dataset
    try:
        files = api.list_repo_files(repo_id=FUXI_S2S_REPO, repo_type="dataset")
        logger.info(f"Found {len(files)} files in FuXi-S2S dataset")
    except Exception as e:
        logger.error(f"Could not list files: {e}")
        return

    # Filter for relevant files
    for year in years:
        if year not in FUXI_S2S_YEARS:
            logger.warning(f"Year {year} not available in FuXi-S2S (2002-2021)")
            continue

        # Download files for this year
        year_files = [f for f in files if str(year) in f]

        for filepath in year_files:
            try:
                local_path = hf_hub_download(
                    repo_id=FUXI_S2S_REPO,
                    filename=filepath,
                    repo_type="dataset",
                    local_dir=data_dir,
                )
                logger.info(f"Downloaded: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to download {filepath}: {e}")
