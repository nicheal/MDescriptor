"""MTP descriptor public name."""

from ...core.adapter import adapter_class
from ...core.model_adapter import ModelBackedAdapter
from .._kernels.mtp import MtpKernel

MTP = adapter_class(
    "MTP",
    MtpKernel,
    __name__,
    base=ModelBackedAdapter,
    requires_species=True,
)

__all__ = ["MTP"]
