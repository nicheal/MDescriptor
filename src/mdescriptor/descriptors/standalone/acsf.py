"""ACSF descriptor public name."""

from ..._legacy.core import AcsfCalculator
from ...core.legacy_adapter import adapter_class

ACSF = adapter_class("ACSF", AcsfCalculator, __name__)

__all__ = ["ACSF"]
