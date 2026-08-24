from ...._legacy.dpa4 import Dpa4Calculator
from ....core.legacy_adapter import adapter_class
from ....core.model_adapter import TorchModelAdapter
from ....models import DPA4_RESOURCE

DPA4 = adapter_class(
    "DPA4",
    Dpa4Calculator,
    __name__,
    base=TorchModelAdapter,
    default_model=DPA4_RESOURCE,
)

__all__ = ["DPA4"]
