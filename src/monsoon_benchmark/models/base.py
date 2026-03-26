"""
Base class for AIWP model wrappers.

All model implementations should inherit from AIWPModel and implement
the abstract methods for forecast generation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for an AIWP model."""

    name: str
    model_type: str  # "deterministic" or "probabilistic"
    ensemble_size: int = 1
    resolution_deg: float = 0.25
    weights_path: Optional[Path] = None
    weights_url: Optional[str] = None
    training_end_year: Optional[int] = None

    # Model-specific options
    options: Dict[str, Any] = None

    def __post_init__(self):
        if self.options is None:
            self.options = {}


class AIWPModel(ABC):
    """
    Abstract base class for AI Weather Prediction models.

    Subclasses must implement:
    - generate_forecast(): Generate precipitation forecast
    - load_weights(): Load model weights
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize model.

        Parameters
        ----------
        config : ModelConfig
            Model configuration
        """
        self.config = config
        self._is_loaded = False

    @property
    def name(self) -> str:
        """Model name."""
        return self.config.name

    @property
    def is_probabilistic(self) -> bool:
        """Whether model produces ensemble forecasts."""
        return self.config.model_type == "probabilistic"

    @property
    def ensemble_size(self) -> int:
        """Number of ensemble members."""
        return self.config.ensemble_size

    @property
    def resolution(self) -> float:
        """Native resolution in degrees."""
        return self.config.resolution_deg

    @abstractmethod
    def load_weights(self) -> None:
        """Load model weights from file or URL."""
        pass

    @abstractmethod
    def generate_forecast(
        self,
        init_time: datetime,
        lead_days: int,
        initial_conditions: Optional[xr.Dataset] = None,
    ) -> xr.Dataset:
        """
        Generate weather forecast.

        Parameters
        ----------
        init_time : datetime
            Forecast initialization time
        lead_days : int
            Number of days to forecast
        initial_conditions : xr.Dataset, optional
            Initial conditions (if not provided, will be loaded)

        Returns
        -------
        xr.Dataset
            Forecast dataset with at least "precipitation" variable
            For ensemble models, includes "member" dimension
        """
        pass

    def generate_hindcast(
        self,
        year: int,
        init_dates: List[datetime],
        lead_days: int = 35,
    ) -> Dict[datetime, xr.Dataset]:
        """
        Generate hindcasts for multiple initialization dates.

        Parameters
        ----------
        year : int
            Year for hindcasts
        init_dates : list of datetime
            Initialization dates
        lead_days : int
            Forecast length in days

        Returns
        -------
        dict
            Dictionary of {init_date: forecast_dataset}
        """
        hindcasts = {}

        for init_date in init_dates:
            try:
                forecast = self.generate_forecast(init_date, lead_days)
                hindcasts[init_date] = forecast
            except Exception as e:
                logger.warning(f"Hindcast failed for {init_date}: {e}")

        return hindcasts

    def check_initialized(self) -> bool:
        """Check if model is loaded and ready."""
        return self._is_loaded


class HindcastLoader(AIWPModel):
    """
    Loader for pre-computed hindcasts.

    Used for models where we load existing hindcast data
    rather than running inference (e.g., IFS from S2S database,
    FuXi-S2S from HuggingFace).
    """

    def __init__(
        self,
        config: ModelConfig,
        hindcast_dir: Path,
    ):
        """
        Initialize hindcast loader.

        Parameters
        ----------
        config : ModelConfig
            Model configuration
        hindcast_dir : Path
            Directory containing hindcast files
        """
        super().__init__(config)
        self.hindcast_dir = Path(hindcast_dir)
        self._hindcast_index = {}

    def load_weights(self) -> None:
        """Build index of available hindcasts."""
        self._build_hindcast_index()
        self._is_loaded = True

    def _build_hindcast_index(self) -> None:
        """Scan hindcast directory and build index."""
        # Implementation depends on file naming convention
        pass

    def generate_forecast(
        self,
        init_time: datetime,
        lead_days: int,
        initial_conditions: Optional[xr.Dataset] = None,
    ) -> xr.Dataset:
        """Load pre-computed hindcast."""
        # Find and load appropriate hindcast file
        hindcast_file = self._find_hindcast_file(init_time)

        if hindcast_file is None:
            raise FileNotFoundError(
                f"No hindcast found for {init_time} in {self.hindcast_dir}"
            )

        ds = xr.open_dataset(hindcast_file)

        # Select required lead times
        # (depends on file structure)

        return ds

    def _find_hindcast_file(self, init_time: datetime) -> Optional[Path]:
        """Find hindcast file for given initialization time."""
        # Implementation depends on file naming convention
        return None


class EnsembleWrapper(AIWPModel):
    """
    Wrapper to create ensemble from deterministic model.

    Generates ensemble by running deterministic model with
    perturbed initial conditions.
    """

    def __init__(
        self,
        base_model: AIWPModel,
        ensemble_size: int = 51,
        perturbation_std: float = 0.1,
    ):
        """
        Initialize ensemble wrapper.

        Parameters
        ----------
        base_model : AIWPModel
            Deterministic model to wrap
        ensemble_size : int
            Number of ensemble members
        perturbation_std : float
            Standard deviation for initial condition perturbations
        """
        config = ModelConfig(
            name=f"{base_model.name}_ensemble",
            model_type="probabilistic",
            ensemble_size=ensemble_size,
            resolution_deg=base_model.resolution,
        )
        super().__init__(config)
        self.base_model = base_model
        self.perturbation_std = perturbation_std

    def load_weights(self) -> None:
        """Load base model weights."""
        self.base_model.load_weights()
        self._is_loaded = True

    def generate_forecast(
        self,
        init_time: datetime,
        lead_days: int,
        initial_conditions: Optional[xr.Dataset] = None,
    ) -> xr.Dataset:
        """Generate ensemble forecast with perturbed ICs."""
        members = []

        for m in range(self.ensemble_size):
            # Perturb initial conditions
            if initial_conditions is not None:
                perturbed_ic = self._perturb_ic(initial_conditions, seed=m)
            else:
                perturbed_ic = None

            # Generate forecast
            forecast = self.base_model.generate_forecast(
                init_time, lead_days, perturbed_ic
            )
            members.append(forecast)

        # Combine into ensemble
        ensemble = xr.concat(members, dim="member")
        ensemble["member"] = np.arange(self.ensemble_size)

        return ensemble

    def _perturb_ic(self, ic: xr.Dataset, seed: int) -> xr.Dataset:
        """Add random perturbations to initial conditions."""
        np.random.seed(seed)

        perturbed = ic.copy()
        for var in ic.data_vars:
            noise = np.random.normal(0, self.perturbation_std, ic[var].shape)
            perturbed[var] = ic[var] + noise * ic[var].std()

        return perturbed
