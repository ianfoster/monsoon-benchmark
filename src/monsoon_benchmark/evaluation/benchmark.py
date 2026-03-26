"""
Main benchmarking orchestration for monsoon onset forecasts.

This module ties together all components:
- Data loading (IMD, ERA5)
- Onset index computation
- Model forecasts
- Metric computation
- Result aggregation
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import yaml

import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm

from monsoon_benchmark.data.regions import (
    CMZ_BOUNDS,
    INDIA_BOUNDS,
    get_cmz_grid_cells,
    create_target_grid,
)
from monsoon_benchmark.indices.onset import (
    compute_local_onset,
    compute_onset_from_forecast,
    compute_ensemble_onset,
    onset_to_bin,
)
from monsoon_benchmark.indices.climatology import (
    compute_wet_spell_threshold,
    compute_climatological_onset,
    ClimatologyBaseline,
)
from monsoon_benchmark.metrics.deterministic import (
    classify_forecast,
    compute_mae,
    compute_far,
    compute_mr,
    compute_metrics_for_period,
    aggregate_metrics_cmz,
)
from monsoon_benchmark.metrics.probabilistic import (
    bin_ensemble_onsets,
    compute_probabilistic_metrics,
)
from monsoon_benchmark.metrics.skill_scores import compute_all_skill_scores
from monsoon_benchmark.evaluation.periods import (
    ANALYSIS_PERIODS,
    generate_init_dates,
    get_init_dates_before_onset,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for monsoon onset benchmarking."""

    # Grid configuration
    grid_resolution: float = 4.0
    lat_range: Tuple[float, float] = (6, 38)
    lon_range: Tuple[float, float] = (66, 98)

    # Onset index parameters
    mok_date: str = "06-02"
    wet_day_threshold_mm: float = 1.0
    wet_spell_days: int = 5

    # Evaluation parameters
    forecast_windows: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {
            "medium_range": (1, 15),
            "subseasonal": (16, 30),
        }
    )
    tolerance_days: Dict[str, int] = field(
        default_factory=lambda: {
            "medium_range": 3,
            "subseasonal": 5,
        }
    )
    init_frequency: str = "twice_weekly"
    init_start: str = "05-02"

    # Probabilistic evaluation
    bin_size_days: int = 5
    n_bins: int = 7

    # Paths
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")

    @classmethod
    def from_yaml(cls, path: Path) -> "BenchmarkConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            config = yaml.safe_load(f)

        # Flatten nested config
        flat_config = {}
        for section, params in config.items():
            if isinstance(params, dict):
                for key, value in params.items():
                    flat_key = f"{section}_{key}" if section != "grid" else key
                    flat_config[flat_key] = value

        return cls(**flat_config)


@dataclass
class BenchmarkResult:
    """Results from benchmarking a single model."""

    model_name: str
    period_name: str
    forecast_window: str

    # Deterministic metrics
    mae: float
    mae_se: float
    far: float
    mr: float
    total_misses: float

    # Probabilistic metrics (optional)
    brier_score: Optional[float] = None
    rps: Optional[float] = None
    auc: Optional[float] = None

    # Skill scores relative to climatology
    mae_skill: Optional[float] = None
    bss: Optional[float] = None
    rpss: Optional[float] = None

    # Metadata
    n_years: int = 0
    n_forecasts: int = 0
    is_ensemble: bool = False


class MonsoonBenchmark:
    """
    Main class for orchestrating monsoon onset benchmarking.

    This class handles:
    1. Loading and preprocessing data
    2. Computing onset indices for observations and forecasts
    3. Evaluating models against climatological baseline
    4. Aggregating metrics across space and time
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize benchmark.

        Parameters
        ----------
        config : BenchmarkConfig
            Benchmark configuration
        """
        self.config = config
        self.cmz_cells = get_cmz_grid_cells(config.grid_resolution)

        # Placeholders for loaded data
        self.observed_onsets: Dict[int, xr.DataArray] = {}
        self.wet_spell_threshold: Optional[xr.DataArray] = None
        self.climatology_onset: Optional[xr.DataArray] = None
        self.climatology_baseline: Optional[ClimatologyBaseline] = None

    def setup(
        self,
        imd_data: xr.Dataset,
        climatology_years: Tuple[int, int] = (1901, 2024),
    ):
        """
        Set up benchmark with observational data.

        Parameters
        ----------
        imd_data : xr.Dataset
            IMD rainfall dataset
        climatology_years : tuple
            Years for computing climatology
        """
        logger.info("Computing wet spell thresholds...")
        self.wet_spell_threshold = compute_wet_spell_threshold(
            imd_data["rainfall"],
            wet_spell_days=self.config.wet_spell_days,
            wet_day_threshold_mm=self.config.wet_day_threshold_mm,
        )

        logger.info("Computing climatological onset dates...")
        self.climatology_onset = compute_climatological_onset(
            imd_data["rainfall"],
            self.wet_spell_threshold,
            years=list(range(climatology_years[0], climatology_years[1] + 1)),
            wet_day_threshold_mm=self.config.wet_day_threshold_mm,
            wet_spell_days=self.config.wet_spell_days,
            mok_date=self.config.mok_date,
        )

        # Set up climatology baseline
        self.climatology_baseline = ClimatologyBaseline(
            self.climatology_onset,
        )

        logger.info("Setup complete")

    def compute_observed_onset(
        self,
        precip: xr.DataArray,
        year: int,
    ) -> xr.DataArray:
        """
        Compute observed onset dates for a year.

        Parameters
        ----------
        precip : xr.DataArray
            Precipitation data for the year
        year : int
            Year

        Returns
        -------
        xr.DataArray
            Onset day-of-year for each grid cell
        """
        from monsoon_benchmark.indices.onset import compute_onset_grid

        onset = compute_onset_grid(
            precip,
            self.wet_spell_threshold,
            year=year,
            wet_day_threshold_mm=self.config.wet_day_threshold_mm,
            wet_spell_days=self.config.wet_spell_days,
            mok_date=self.config.mok_date,
        )

        self.observed_onsets[year] = onset
        return onset

    def evaluate_deterministic_forecast(
        self,
        forecast_precip: xr.DataArray,
        observed_onset: datetime,
        init_date: datetime,
        lat: float,
        lon: float,
        forecast_window: str = "medium_range",
    ) -> dict:
        """
        Evaluate a single deterministic forecast.

        Parameters
        ----------
        forecast_precip : xr.DataArray
            Forecast precipitation
        observed_onset : datetime
            Observed onset date
        init_date : datetime
            Initialization date
        lat : float
            Grid cell latitude
        lon : float
            Grid cell longitude
        forecast_window : str
            Forecast window name

        Returns
        -------
        dict
            Evaluation results
        """
        window = self.config.forecast_windows[forecast_window]
        tolerance = self.config.tolerance_days[forecast_window]

        # Get threshold for location
        threshold = float(
            self.wet_spell_threshold.sel(lat=lat, lon=lon, method="nearest").values
        )

        # Compute forecast onset
        forecast_onset = compute_onset_from_forecast(
            forecast_precip.sel(lat=lat, lon=lon, method="nearest"),
            wet_spell_threshold=threshold,
            init_date=init_date,
            forecast_window=window,
            wet_day_threshold_mm=self.config.wet_day_threshold_mm,
            wet_spell_days=self.config.wet_spell_days,
            mok_date=self.config.mok_date,
        )

        # Classify forecast
        classification = classify_forecast(
            forecast_onset=forecast_onset,
            observed_onset=observed_onset,
            forecast_window=window,
            init_date=init_date,
            tolerance_days=tolerance,
        )

        return {
            "forecast_onset": forecast_onset,
            "observed_onset": observed_onset,
            "init_date": init_date,
            "window": window,
            "classification": classification,
        }

    def evaluate_ensemble_forecast(
        self,
        ensemble_precip: xr.DataArray,
        observed_onset: datetime,
        init_date: datetime,
        lat: float,
        lon: float,
        forecast_window: str = "medium_range",
    ) -> dict:
        """
        Evaluate an ensemble forecast.

        Parameters
        ----------
        ensemble_precip : xr.DataArray
            Ensemble forecast precipitation (member, time, ...)
        observed_onset : datetime
            Observed onset date
        init_date : datetime
            Initialization date
        lat : float
            Grid cell latitude
        lon : float
            Grid cell longitude
        forecast_window : str
            Forecast window name

        Returns
        -------
        dict
            Evaluation results including probabilities
        """
        window = self.config.forecast_windows[forecast_window]

        # Get threshold for location
        threshold = float(
            self.wet_spell_threshold.sel(lat=lat, lon=lon, method="nearest").values
        )

        # Compute onset for each member
        member_onsets = compute_ensemble_onset(
            ensemble_precip.sel(lat=lat, lon=lon, method="nearest"),
            wet_spell_threshold=threshold,
            init_date=init_date,
            forecast_window=window,
            wet_day_threshold_mm=self.config.wet_day_threshold_mm,
            wet_spell_days=self.config.wet_spell_days,
            mok_date=self.config.mok_date,
        )

        # Convert to probability distribution
        probs = bin_ensemble_onsets(
            member_onsets,
            init_date,
            self.config.bin_size_days,
            self.config.n_bins,
        )

        # Deterministic forecast from ensemble mean
        valid_onsets = [o for o in member_onsets if o is not None]
        if len(valid_onsets) >= len(member_onsets) / 2:
            # Majority predicts onset
            mean_onset_doy = np.mean(
                [(o - datetime(init_date.year, 1, 1)).days + 1 for o in valid_onsets]
            )
            det_onset = datetime(init_date.year, 1, 1) + pd.Timedelta(
                days=int(mean_onset_doy) - 1
            )
        else:
            det_onset = None

        return {
            "member_onsets": member_onsets,
            "probabilities": probs,
            "deterministic_onset": det_onset,
            "observed_onset": observed_onset,
            "init_date": init_date,
            "window": window,
        }

    def evaluate_model(
        self,
        model,  # AIWPModel instance
        period_name: str,
        forecast_window: str = "medium_range",
        progress: bool = True,
    ) -> BenchmarkResult:
        """
        Evaluate a model over an analysis period.

        Parameters
        ----------
        model : AIWPModel
            Model to evaluate
        period_name : str
            Analysis period name
        forecast_window : str
            Forecast window to evaluate
        progress : bool
            Show progress bar

        Returns
        -------
        BenchmarkResult
            Aggregated benchmark results
        """
        period = ANALYSIS_PERIODS[period_name]
        years = period.get_years()

        all_results = []
        ensemble_results = []

        iterator = tqdm(years, desc=f"Evaluating {model.name}") if progress else years

        for year in iterator:
            # Get observed onset for this year
            if year not in self.observed_onsets:
                logger.warning(f"No observed onset for {year}, skipping")
                continue

            # Generate init dates
            init_dates = generate_init_dates(
                year,
                frequency=self.config.init_frequency,
                start_date=self.config.init_start,
            )

            for cell in self.cmz_cells:
                lat, lon = cell
                obs_doy = self.observed_onsets[year].sel(
                    lat=lat, lon=lon, method="nearest"
                ).values

                if np.isnan(obs_doy):
                    continue

                obs_onset = datetime(year, 1, 1) + pd.Timedelta(days=int(obs_doy) - 1)

                # Filter init dates before onset
                valid_inits = get_init_dates_before_onset(init_dates, obs_onset)

                for init_date in valid_inits:
                    # Generate or load forecast
                    try:
                        forecast = model.generate_forecast(
                            init_date,
                            lead_days=self.config.forecast_windows[forecast_window][1]
                            + self.config.wet_spell_days,
                        )
                    except Exception as e:
                        logger.warning(f"Forecast failed for {init_date}: {e}")
                        continue

                    # Evaluate
                    if model.is_probabilistic:
                        result = self.evaluate_ensemble_forecast(
                            forecast["precipitation"],
                            obs_onset,
                            init_date,
                            lat,
                            lon,
                            forecast_window,
                        )
                        ensemble_results.append(result)
                    else:
                        result = self.evaluate_deterministic_forecast(
                            forecast["precipitation"],
                            obs_onset,
                            init_date,
                            lat,
                            lon,
                            forecast_window,
                        )
                        all_results.append(result)

        # Aggregate metrics
        return self._aggregate_results(
            model.name,
            period_name,
            forecast_window,
            all_results,
            ensemble_results,
            model.is_probabilistic,
        )

    def _aggregate_results(
        self,
        model_name: str,
        period_name: str,
        forecast_window: str,
        det_results: List[dict],
        ens_results: List[dict],
        is_ensemble: bool,
    ) -> BenchmarkResult:
        """Aggregate evaluation results into benchmark metrics."""
        # Deterministic metrics
        if det_results:
            metrics = compute_metrics_for_period(
                [
                    {
                        "forecast_onset": r["forecast_onset"],
                        "observed_onset": r["observed_onset"],
                        "init_date": r["init_date"],
                        "window": r["window"],
                    }
                    for r in det_results
                ],
                tolerance_days=self.config.tolerance_days[forecast_window],
            )
            mae = metrics["mae"]
            far = metrics["far"]
            mr = metrics["mr"]
            n_forecasts = metrics["n_forecasts"]
        else:
            mae = np.nan
            far = np.nan
            mr = np.nan
            n_forecasts = 0

        # Probabilistic metrics
        brier = None
        rps = None
        auc = None

        if ens_results:
            prob_metrics = compute_probabilistic_metrics(
                [r["member_onsets"] for r in ens_results],
                [r["observed_onset"] for r in ens_results],
                [r["init_date"] for r in ens_results],
                self.config.bin_size_days,
                self.config.n_bins,
            )
            brier = prob_metrics["brier_score"]
            rps = prob_metrics["rps"]
            auc = prob_metrics["auc"]

        # Compute skill scores vs climatology
        # (would need climatology metrics computed similarly)
        mae_skill = None
        bss = None
        rpss = None

        return BenchmarkResult(
            model_name=model_name,
            period_name=period_name,
            forecast_window=forecast_window,
            mae=mae,
            mae_se=0.0,  # Would compute from cell-level variance
            far=far * 100,  # Convert to percentage
            mr=mr * 100,
            total_misses=0.0,  # Would compute separately
            brier_score=brier,
            rps=rps,
            auc=auc,
            mae_skill=mae_skill,
            bss=bss,
            rpss=rpss,
            n_years=len(ANALYSIS_PERIODS[period_name].get_years()),
            n_forecasts=n_forecasts,
            is_ensemble=is_ensemble,
        )

    def run_full_benchmark(
        self,
        models: List,  # List of AIWPModel
        periods: List[str] = None,
        forecast_windows: List[str] = None,
    ) -> pd.DataFrame:
        """
        Run full benchmark across models, periods, and windows.

        Parameters
        ----------
        models : list
            List of models to evaluate
        periods : list, optional
            Period names (default: all)
        forecast_windows : list, optional
            Forecast windows (default: all)

        Returns
        -------
        pd.DataFrame
            Results table
        """
        if periods is None:
            periods = list(ANALYSIS_PERIODS.keys())
        if forecast_windows is None:
            forecast_windows = list(self.config.forecast_windows.keys())

        results = []

        for model in models:
            for period in periods:
                for window in forecast_windows:
                    logger.info(f"Evaluating {model.name} / {period} / {window}")
                    result = self.evaluate_model(model, period, window)
                    results.append(result)

        # Convert to DataFrame
        return pd.DataFrame([vars(r) for r in results])
