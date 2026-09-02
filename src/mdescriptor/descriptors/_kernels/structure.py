"""Shared structure-level kernel protocol."""


class _StructureKernel:
    name = "descriptor"
    level = "structure"

    @property
    def feature_count(self) -> int:
        return int(getattr(self, "_feature_count", 0))
