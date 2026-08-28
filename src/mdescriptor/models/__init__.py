"""Bundled model files used by the default model-backed descriptors."""

from pathlib import Path

from ..registry.model_defaults import bundled_model_identity
from .resolver import ModelResolver, ResolvedModel
from .resource import ModelResource
from .session import (
    LoadedModel,
    ModelSession,
    clear_loaded_model_cache,
    discard_loaded_model,
    shared_loaded_model,
)

MODEL_DIR = Path(__file__).resolve().parent / "assets"

DPA4_MODEL = MODEL_DIR / "DPA4-Air-OMat24-v20260704.pt"
DPA4C_MODEL = MODEL_DIR / "DPA4C-Air-OMat24-v20260819.pt"
NEP_MODEL = MODEL_DIR / "nep89_20250409.txt"

# Checksums are part of the packaged resource identity.  A changed bundled
# artifact must therefore fail closed instead of silently changing descriptor
# output.


def _bundled_resource(name: str) -> ModelResource:
    identity = bundled_model_identity(name)
    return ModelResource(
        name=identity["name"],
        expected_sha256=identity["expected_sha256"],
        identifier=identity["identifier"],
    )


DPA4_RESOURCE = _bundled_resource("DPA4")
DPA4C_RESOURCE = _bundled_resource("DPA4C")
NEP_RESOURCE = _bundled_resource("NEP")

__all__ = [
    "DPA4_MODEL",
    "DPA4_RESOURCE",
    "DPA4C_MODEL",
    "DPA4C_RESOURCE",
    "LoadedModel",
    "MODEL_DIR",
    "ModelResource",
    "ModelResolver",
    "ModelSession",
    "ResolvedModel",
    "NEP_MODEL",
    "NEP_RESOURCE",
    "clear_loaded_model_cache",
    "discard_loaded_model",
    "shared_loaded_model",
]
