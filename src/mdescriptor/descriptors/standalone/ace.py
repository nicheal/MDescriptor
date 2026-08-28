"""Public ACE descriptor name."""

from ...core.adapter import adapter_class
from .._kernels.ace import AceKernel, _AceAdapterMixin

ACE = adapter_class(
    "ACE",
    AceKernel,
    __name__,
    base=_AceAdapterMixin,
    requires_species=True,
)

__all__ = ["ACE"]
