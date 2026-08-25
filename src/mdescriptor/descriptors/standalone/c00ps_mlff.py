"""C00PS-MLFF descriptor public name."""

from ...core.adapter import adapter_class
from .._kernels.c00ps_mlff import C00PSMlffKernel

C00PSMLFF = adapter_class("C00PSMLFF", C00PSMlffKernel, __name__)

__all__ = ["C00PSMLFF"]
