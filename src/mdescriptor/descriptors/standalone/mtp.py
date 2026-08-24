"""MTP descriptor public name."""

from ..._legacy.mtp import MtpCalculator
from ...core.legacy_adapter import adapter_class

MTP = adapter_class("MTP", MtpCalculator, __name__)

__all__ = ["MTP"]
