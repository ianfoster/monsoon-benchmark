"""AIWP model wrappers for monsoon onset forecasting."""

from monsoon_benchmark.models.base import AIWPModel, ModelConfig, HindcastLoader
from monsoon_benchmark.models.climatology_model import ClimatologyModel
from monsoon_benchmark.models.fuxi_s2s import FuXiS2SLoader, download_fuxi_s2s_hindcasts
from monsoon_benchmark.models.weatherbench2 import (
    WeatherBench2Loader,
    load_weatherbench2_era5,
)

__all__ = [
    "AIWPModel",
    "ModelConfig",
    "HindcastLoader",
    "ClimatologyModel",
    # Hindcast loaders
    "FuXiS2SLoader",
    "download_fuxi_s2s_hindcasts",
    "WeatherBench2Loader",
    "load_weatherbench2_era5",
]

# Optional imports for specific models (require additional dependencies)
try:
    from monsoon_benchmark.models.aifs import AIFSModel

    __all__.append("AIFSModel")
except ImportError:
    pass

try:
    from monsoon_benchmark.models.neuralgcm import NeuralGCMModel

    __all__.append("NeuralGCMModel")
except ImportError:
    pass
