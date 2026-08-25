"""SOAP-Turbo descriptor public name."""

from ...core.adapter import adapter_class
from .._kernels.soap_turbo import SoapTurboKernel

SOAPTurbo = adapter_class("SOAPTurbo", SoapTurboKernel, __name__)

__all__ = ["SOAPTurbo"]
