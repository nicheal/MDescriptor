from ....core.adapter import adapter_class
from ....core.model_adapter import ModelBackedAdapter
from ....models import DPA4C_RESOURCE
from ..._kernels.dpa4c import Dpa4cKernel

DPA4C = adapter_class(
    "DPA4C",
    Dpa4cKernel,
    __name__,
    base=ModelBackedAdapter,
    default_model=DPA4C_RESOURCE,
)

__all__ = ["DPA4C"]
