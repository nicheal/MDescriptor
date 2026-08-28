"""Implementation-independent identities for bundled model resources."""

from __future__ import annotations

from typing import Any

_BUNDLED_MODEL_IDENTITIES: dict[str, dict[str, str]] = {
    "DPA4": {
        "name": "DPA4-Air-OMat24-v20260704.pt",
        "expected_sha256": "e75a9d5bc1c9b68e4d8aa097d4ab6be690a42d658fc13f030080da6c119f6a23",
        "identifier": "DPA4-Air-OMat24-v20260704",
    },
    "DPA4C": {
        "name": "DPA4C-Air-OMat24-v20260819.pt",
        "expected_sha256": "ff596ce704c9b4bc7149fbb0a0f12df611924cccb045917e8abb95b6a6cd4ad8",
        "identifier": "DPA4C-Air-OMat24-v20260819",
    },
    "NEP": {
        "name": "nep89_20250409.txt",
        "expected_sha256": "75168ece02e840e4a32644f982b78d43cba697f5b64b4c8134ab66c7a8c28be1",
        "identifier": "nep89_20250409",
    },
}


def bundled_model_identity(name: str) -> dict[str, str]:
    """Return a copy of one bundled model's deterministic identity."""

    try:
        return dict(_BUNDLED_MODEL_IDENTITIES[name])
    except KeyError as exc:  # pragma: no cover - only internal names are used
        raise KeyError(f"unknown bundled model {name!r}") from exc


def bundled_model_default(name: str) -> dict[str, Any]:
    """Return the JSON form used by the static descriptor metadata."""

    return {"__type__": "ModelResource", **bundled_model_identity(name)}


__all__ = ["bundled_model_default", "bundled_model_identity"]
