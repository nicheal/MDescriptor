"""ACSF descriptor public name."""

from ...core.adapter import adapter_class
from .._kernels.core import AcsfKernel

ACSF = adapter_class("ACSF", AcsfKernel, __name__)

__all__ = ["ACSF"]
