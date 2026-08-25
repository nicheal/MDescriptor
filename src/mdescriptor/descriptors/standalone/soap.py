"""SOAP descriptor public name."""

from ...core.adapter import adapter_class
from .._kernels.core import SoapKernel

SOAP = adapter_class("SOAP", SoapKernel, __name__)

__all__ = ["SOAP"]
