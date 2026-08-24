from ...._legacy.nep import NepCalculator
from ....core.legacy_adapter import adapter_class
from ....core.model_adapter import ModelBackedAdapter
from ....models import NEP_RESOURCE

NEP = adapter_class(
    "NEP",
    NepCalculator,
    __name__,
    base=ModelBackedAdapter,
    default_model=NEP_RESOURCE,
)

__all__ = ["NEP"]
