"""C00PS-MLFF descriptor public name."""

from ..._legacy.c00ps_mlff import C00PSMlffCalculator
from ...core.legacy_adapter import adapter_class

C00PSMLFF = adapter_class("C00PSMLFF", C00PSMlffCalculator, __name__)

__all__ = ["C00PSMLFF"]
