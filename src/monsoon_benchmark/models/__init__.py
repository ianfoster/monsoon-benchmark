"""AIWP model wrappers for monsoon onset forecasting."""

from monsoon_benchmark.models.base import AIWPModel, ModelConfig
from monsoon_benchmark.models.climatology_model import ClimatologyModel

__all__ = [
    "AIWPModel",
    "ModelConfig",
    "ClimatologyModel",
]

# Optional imports for specific models
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
