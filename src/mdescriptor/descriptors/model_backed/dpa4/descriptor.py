from ....core.adapter import adapter_class
from ....core.model_adapter import ModelBackedAdapter
from ....models import DPA4_RESOURCE
from ..._kernels.dpa4 import Dpa4Kernel

DPA4 = adapter_class(
    "DPA4",
    Dpa4Kernel,
    __name__,
    base=ModelBackedAdapter,
    default_model=DPA4_RESOURCE,
)

__all__ = ["DPA4"]
