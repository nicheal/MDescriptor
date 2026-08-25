"""Bundled model files used by the default model-backed descriptors."""

from pathlib import Path

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
DPA4_RESOURCE = ModelResource(
    name=DPA4_MODEL.name,
    expected_sha256="e75a9d5bc1c9b68e4d8aa097d4ab6be690a42d658fc13f030080da6c119f6a23",
    identifier="DPA4-Air-OMat24-v20260704",
)
DPA4C_RESOURCE = ModelResource(
    name=DPA4C_MODEL.name,
    expected_sha256="ff596ce704c9b4bc7149fbb0a0f12df611924cccb045917e8abb95b6a6cd4ad8",
    identifier="DPA4C-Air-OMat24-v20260819",
)
NEP_RESOURCE = ModelResource(
    name=NEP_MODEL.name,
    expected_sha256="75168ece02e840e4a32644f982b78d43cba697f5b64b4c8134ab66c7a8c28be1",
    identifier="nep89_20250409",
)

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
