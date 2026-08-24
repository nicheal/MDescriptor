"""SOAP-Turbo descriptor public name."""

from ..._legacy.soap_turbo import SoapTurboCalculator
from ...core.legacy_adapter import adapter_class

SOAPTurbo = adapter_class("SOAPTurbo", SoapTurboCalculator, __name__)

__all__ = ["SOAPTurbo"]
