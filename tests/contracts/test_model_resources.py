"""Resource identity, cache and per-instance session contracts."""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path

import pytest

from mdescriptor import ClosedDescriptorError, DescriptorConfigError, ModelLoadError
from mdescriptor.models import (
    ModelResolver,
    ModelResource,
    ModelSession,
    clear_loaded_model_cache,
    shared_loaded_model,
)

pytestmark = pytest.mark.model


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_named_resolution_is_cache_then_package_and_corrupt_cache_fails(tmp_path):
    cache = tmp_path / "cache"
    package = tmp_path / "package"
    cache.mkdir()
    package.mkdir()
    packaged = package / "demo.bin"
    packaged.write_bytes(b"packaged")
    digest = _digest(packaged)
    resource = ModelResource.named("demo.bin", expected_sha256=digest)
    resolver = ModelResolver(cache_dir=cache, package_dir=package)
    resolved = resolver.resolve(resource)
    assert resolved.source == "package"
    assert resolved.path == packaged.resolve()

    cached = cache / "demo.bin"
    cached.write_bytes(b"corrupt")
    with pytest.raises(ModelLoadError, match="checksum mismatch"):
        resolver.resolve(resource)

    cached.write_bytes(packaged.read_bytes())
    assert resolver.resolve(resource).source == "cache"


def test_model_resource_forms_are_mutually_exclusive_and_paths_are_strict(tmp_path):
    with pytest.raises(DescriptorConfigError):
        ModelResource(path=tmp_path / "a", name="a")
    with pytest.raises(DescriptorConfigError):
        ModelResource(name="../outside.bin")
    with pytest.raises(DescriptorConfigError):
        ModelResource.from_dict({"__type__": "ModelResource", "name": "a", "path": "b"})


def test_loaded_model_cache_shares_cpu_artifact_but_sessions_are_independent(tmp_path):
    clear_loaded_model_cache()
    path = tmp_path / "model.bin"
    path.write_bytes(b"same bytes")
    resource = ModelResource.explicit(path)
    resolved = ModelResolver().resolve(resource)
    calls = 0

    def loader(_resolved):
        nonlocal calls
        calls += 1
        return {"immutable": True}, bytearray(b"cpu-weights")

    first = shared_loaded_model(
        resolved, loader_kind="test", loader_schema=1, loader=loader
    )
    second = shared_loaded_model(
        resolved, loader_kind="test", loader_schema=1, loader=loader
    )
    assert first is second
    assert calls == 1
    session_a = ModelSession(first, runtime=object())
    session_b = ModelSession(second, runtime=object())
    assert session_a.model is session_b.model
    assert session_a.runtime is not session_b.runtime
    session_a.close()
    assert session_b.model is second
    assert session_b.closed is False
    with pytest.raises(ClosedDescriptorError):
        session_a.ensure_open()
    session_b.close()
    del first, second, session_a, session_b
    gc.collect()
    shared_loaded_model(resolved, loader_kind="test", loader_schema=1, loader=loader)
    assert calls == 2


def test_failed_model_load_is_not_cached(tmp_path):
    clear_loaded_model_cache()
    path = tmp_path / "broken.bin"
    path.write_bytes(b"broken")
    resolved = ModelResolver().resolve(ModelResource.explicit(path))
    calls = 0

    def loader(_resolved):
        nonlocal calls
        calls += 1
        raise ValueError("bad checkpoint")

    for _ in range(2):
        with pytest.raises(ModelLoadError, match="failed to load"):
            shared_loaded_model(
                resolved, loader_kind="broken", loader_schema=1, loader=loader
            )
    assert calls == 2
