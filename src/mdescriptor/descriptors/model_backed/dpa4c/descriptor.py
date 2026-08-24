from ...._legacy.dpa4c import Dpa4cCalculator
from ....core.legacy_adapter import adapter_class
from ....core.model_adapter import TorchModelAdapter
from ....models import DPA4C_RESOURCE

DPA4C = adapter_class(
    "DPA4C",
    Dpa4cCalculator,
    __name__,
    base=TorchModelAdapter,
    default_model=DPA4C_RESOURCE,
)

__all__ = ["DPA4C"]
