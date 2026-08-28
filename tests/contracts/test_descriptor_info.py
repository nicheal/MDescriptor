"""Contract tests for the static GUI-facing descriptor metadata seam."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys

import pytest

import mdescriptor
from mdescriptor import (
    DescriptorConfigError,
    DescriptorRegistry,
    DescriptorSpec,
    describe_descriptor,
)
from mdescriptor.registry import DescriptorInfo


def test_runtime_info_exposes_independent_contract_versions():
    runtime = mdescriptor.get_runtime_info()

    assert runtime["version"] == mdescriptor.__version__
    assert runtime["api_version"] == mdescriptor.API_VERSION
    assert runtime["configuration_schema_version"] == 1
    assert runtime["descriptor_info_schema_version"] == 1
    json.dumps(runtime, allow_nan=False)


def test_every_builtin_has_json_safe_static_metadata_matching_public_signature():
    expected_fields = {
        "schema_version",
        "name",
        "display_name",
        "description",
        "category",
        "level",
        "backend",
        "capabilities",
        "parameters",
        "execution",
        "input",
        "output",
        "asset",
    }
    for spec in mdescriptor.builtin_registry:
        metadata = describe_descriptor(spec.name)
        assert set(metadata) == expected_fields
        assert metadata["schema_version"] == mdescriptor.DESCRIPTOR_INFO_SCHEMA_VERSION
        assert metadata["name"] == spec.name
        assert metadata["level"] == spec.level
        assert metadata["backend"] == spec.backend
        assert metadata["capabilities"] == sorted(spec.capabilities)
        assert metadata["asset"]["policy"] == spec.asset_policy.value
        assert metadata["execution"]["devices"] == ["cpu"]
        assert set(metadata["parameters"]) <= set(
            inspect.signature(spec.load_class()).parameters
        )
        json.dumps(metadata, allow_nan=False)


def test_static_model_description_does_not_load_model_modules_or_torch():
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import mdescriptor; mdescriptor.describe_descriptor('NEP'); "
            "assert 'torch' not in sys.modules; "
            "assert not any(name.startswith('mdescriptor.models') for name in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0


def test_custom_registry_without_info_remains_compute_only():
    registry = DescriptorRegistry(
        [
            DescriptorSpec(
                "custom",
                "mdescriptor.descriptors.standalone.soap:SOAP",
                mdescriptor.AssetPolicy.NONE,
                "cpp",
                "structure",
            )
        ]
    )

    with pytest.raises(DescriptorConfigError) as caught:
        describe_descriptor("custom", registry=registry)
    assert caught.value.code == "missing_descriptor_info"
    assert caught.value.to_dict()["path"] == ["descriptor", "custom"]


def test_descriptor_info_freezes_nested_schema_and_returns_json_copy():
    info = DescriptorInfo(
        "Example",
        "Example descriptor.",
        "local",
        {"cutoff": {"type": "number", "exclusiveMinimum": 0.0}},
    )

    value = info.to_dict()
    value["parameters"]["cutoff"]["description"] = "changed"
    assert "description" not in info.to_dict()["parameters"]["cutoff"]
    json.dumps(info.to_dict(), allow_nan=False)
