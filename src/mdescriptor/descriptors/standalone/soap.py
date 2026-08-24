"""SOAP descriptor public name."""

from ..._legacy.core import SoapCalculator
from ...core.legacy_adapter import adapter_class

SOAP = adapter_class("SOAP", SoapCalculator, __name__)

__all__ = ["SOAP"]
