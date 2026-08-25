"""Many-body standalone descriptor family."""

from ....core.adapter import adapter_class
from ..._kernels.extra import LMBTRKernel, MBTRKernel, ValleOganovKernel

LMBTR = adapter_class("LMBTR", LMBTRKernel, __name__)
MBTR = adapter_class("MBTR", MBTRKernel, __name__)
ValleOganov = adapter_class("ValleOganov", ValleOganovKernel, __name__)

__all__ = ["LMBTR", "MBTR", "ValleOganov"]
