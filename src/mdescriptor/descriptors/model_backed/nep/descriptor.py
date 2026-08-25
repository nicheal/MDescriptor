from ....core.adapter import adapter_class
from ....core.model_adapter import ModelBackedAdapter
from ....models import NEP_RESOURCE
from ..._kernels.nep import NepKernel

NEP = adapter_class(
    "NEP",
    NepKernel,
    __name__,
    base=ModelBackedAdapter,
    default_model=NEP_RESOURCE,
)

__all__ = ["NEP"]
