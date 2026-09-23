"""Independent energy and force prediction interfaces."""

from ..core.prediction_result import PredictionResult
from .dpa4c import DPA4C
from .nep import NEP

__all__ = ["DPA4C", "NEP", "PredictionResult"]
