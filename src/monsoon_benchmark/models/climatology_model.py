"""
Climatological baseline model.

This "model" returns the historical climatology as its forecast,
serving as the baseline for skill score computation.
"""

from datetime import datetime, timedelta
from typing import Optional, List
import logging

import numpy as np
import xarray as xr
import pandas as pd

from monsoon_benchmark.models.base import AIWPModel, ModelConfig

logger = logging.getLogger(__name__)


class ClimatologyModel(AIWPModel):
    """
    Climatological baseline for benchmark comparison.

    This model predicts based on historical climatology, providing
    a reference for skill score computation. For deterministic
    forecasts, it returns the climatological mean. For probabilistic
    forecasts, it returns the historical distribution.
    """

    def __init__(
        self,
        climatology_precip: xr.DataArray,
        historical_precip: Optional[xr.Dataset] = None,
        ensemble_size: int = 124,  # One member per historical year
    ):
        """
        Initialize climatology model.

        Parameters
        ----------
        climatology_precip : xr.DataArray
            Climatological daily mean precipitation (dayofyear, lat, lon)
        historical_precip : xr.Dataset, optional
            Historical daily precipitation for probabilistic forecasts
        ensemble_size : int
            Number of ensemble members (historical years to use)
        """
        config = ModelConfig(
            name="Climatology",
            model_type="probabilistic" if historical_precip is not None else "deterministic",
            ensemble_size=ensemble_size if historical_precip is not None else 1,
            resolution_deg=float(
                climatology_precip.lat.values[1] - climatology_precip.lat.values[0]
            ),
        )
        super().__init__(config)

        self.climatology = climatology_precip
        self.historical = historical_precip
        self._is_loaded = True

    def load_weights(self) -> None:
        """No weights to load for climatology."""
        pass

    def generate_forecast(
        self,
        init_time: datetime,
        lead_days: int,
        initial_conditions: Optional[xr.Dataset] = None,
    ) -> xr.Dataset:
        """
        Generate climatological forecast.

        Parameters
        ----------
        init_time : datetime
            Initialization time (used to determine day of year)
        lead_days : int
            Number of days to forecast
        initial_conditions : xr.Dataset, optional
            Not used for climatology

        Returns
        -------
        xr.Dataset
            Climatological precipitation forecast
        """
        # Generate forecast dates
        forecast_times = pd.date_range(
            init_time + timedelta(days=1),
            init_time + timedelta(days=lead_days),
            freq="D",
        )

        if self.historical is not None:
            return self._generate_probabilistic_forecast(forecast_times)
        else:
            return self._generate_deterministic_forecast(forecast_times)

    def _generate_deterministic_forecast(
        self,
        forecast_times: pd.DatetimeIndex,
    ) -> xr.Dataset:
        """Generate deterministic forecast from climatology."""
        doys = forecast_times.dayofyear

        # Select climatology for each forecast day
        precip = self.climatology.sel(dayofyear=doys)
        precip = precip.rename({"dayofyear": "time"})
        precip["time"] = forecast_times

        return xr.Dataset({"precipitation": precip})

    def _generate_probabilistic_forecast(
        self,
        forecast_times: pd.DatetimeIndex,
    ) -> xr.Dataset:
        """Generate probabilistic forecast from historical data."""
        # Get unique years in historical data
        years = np.unique(self.historical.time.dt.year.values)

        # Use each year as an ensemble member
        members = []
        for year in years[: self.ensemble_size]:
            # Get historical data for this year, matching forecast DOYs
            year_data = self.historical.sel(
                time=self.historical.time.dt.year == year
            )

            # Select matching days of year
            member_precip = []
            for doy in forecast_times.dayofyear:
                day_data = year_data.sel(
                    time=year_data.time.dt.dayofyear == doy
                )
                if len(day_data.time) > 0:
                    member_precip.append(day_data["rainfall"].isel(time=0))
                else:
                    # Fallback to climatology
                    member_precip.append(self.climatology.sel(dayofyear=doy))

            member = xr.concat(member_precip, dim="time")
            member["time"] = forecast_times
            members.append(member)

        # Combine into ensemble
        ensemble = xr.concat(members, dim="member")
        ensemble["member"] = np.arange(len(members))

        return xr.Dataset({"precipitation": ensemble})

    @classmethod
    def from_imd_data(
        cls,
        imd_data: xr.Dataset,
        climatology_years: tuple = (1901, 2024),
    ) -> "ClimatologyModel":
        """
        Create climatology model from IMD dataset.

        Parameters
        ----------
        imd_data : xr.Dataset
            IMD rainfall dataset
        climatology_years : tuple
            (start_year, end_year) for climatology computation

        Returns
        -------
        ClimatologyModel
            Initialized model
        """
        # Compute daily climatology
        year_mask = (
            (imd_data.time.dt.year >= climatology_years[0])
            & (imd_data.time.dt.year <= climatology_years[1])
        )
        clim_data = imd_data.sel(time=year_mask)

        climatology = clim_data["rainfall"].groupby("time.dayofyear").mean()

        return cls(
            climatology_precip=climatology,
            historical_precip=clim_data,
            ensemble_size=climatology_years[1] - climatology_years[0] + 1,
        )
