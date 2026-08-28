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
from mdescriptor.registry import DescriptorInfo, validate_descriptor_parameters


def test_runtime_info_exposes_independent_contract_versions():
    runtime = mdescriptor.get_runtime_info()

    assert runtime["version"] == mdescriptor.__version__
    assert runtime["api_version"] == mdescriptor.API_VERSION
    assert runtime["configuration_schema_version"] == 1
    assert runtime["descriptor_info_schema_version"] == 1
    assert runtime["result_schema_version"] == mdescriptor.RESULT_SCHEMA_VERSION
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
        assert all("type" in schema for schema in metadata["parameters"].values())
        assert set(metadata["parameters"]) <= set(
            inspect.signature(spec.load_class()).parameters
        )
        json.dumps(metadata, allow_nan=False)


def test_dpa4c_static_default_matches_the_runtime_default():
    assert describe_descriptor("DPA4C")["parameters"]["calibrate"]["default"] is True


def test_static_model_description_does_not_load_model_modules_or_torch():
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import mdescriptor; mdescriptor.describe_descriptor('NEP'); "
            "assert 'mdescriptor._native' not in sys.modules; "
            "assert 'torch' not in sys.modules; "
            "assert not any(name.startswith('mdescriptor.models') for name in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0


@pytest.mark.parametrize("name", ["SOAP", "ACSF", "ACE"])
def test_metadata_defaults_rebuild_through_the_public_factory(name):
    metadata = describe_descriptor(name)
    parameters = {}
    for parameter_name, schema in metadata["parameters"].items():
        if parameter_name == "species":
            parameters[parameter_name] = [1]
        elif "default" in schema:
            parameters[parameter_name] = schema["default"]

    descriptor = mdescriptor.create_descriptor(
        mdescriptor.DescriptorConfiguration(1, name, parameters)
    )
    try:
        assert descriptor.name == name
    finally:
        descriptor.close()


def test_descriptor_info_parses_full_metadata_and_rejects_unknown_versions():
    metadata = describe_descriptor("SOAP")

    parsed = DescriptorInfo.from_dict(metadata)
    assert parsed.to_dict()["parameters"] == metadata["parameters"]
    assert DescriptorInfo.from_dict(parsed.to_dict()).to_dict() == parsed.to_dict()
    json.dumps(parsed.to_dict(), allow_nan=False)

    unsupported = dict(metadata)
    unsupported["schema_version"] = 999
    with pytest.raises(DescriptorConfigError) as caught:
        DescriptorInfo.from_dict(unsupported)
    assert caught.value.code == "unsupported_descriptor_info_schema"
    assert caught.value.to_dict()["path"] == ["schema_version"]


def test_descriptor_info_does_not_leak_registry_only_optional_fields():
    info = DescriptorInfo("Example", "Example descriptor.", "local")
    spec = DescriptorSpec(
        "custom",
        "mdescriptor.descriptors.standalone.soap:SOAP",
        mdescriptor.AssetPolicy.NONE,
        "cpp",
        "structure",
        optional_extra="test-only",
        info=info,
    )
    registry = DescriptorRegistry([spec])

    assert set(mdescriptor.describe_descriptor("custom", registry=registry)) == {
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


def test_model_schema_requires_path_string_or_tagged_resource_object():
    schemas = describe_descriptor("MTP")["parameters"]

    with pytest.raises(DescriptorConfigError) as caught:
        validate_descriptor_parameters(
            "MTP",
            {"species": [13, 14], "model": {"name": "untagged"}},
            schemas,
        )
    assert caught.value.code == "invalid_parameter"
    assert caught.value.to_dict()["path"] == ["parameters", "model"]

    validate_descriptor_parameters(
        "MTP",
        {
            "species": [13, 14],
            "model": {"__type__": "ModelResource", "name": "tagged"},
        },
        schemas,
    )


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


@pytest.mark.parametrize(
    "schema",
    [
        {"required": False},
        {"type": "integer", "default": 1.5},
        {"type": "enum", "enum": [], "default": "x"},
    ],
)
def test_descriptor_info_rejects_incomplete_or_inconsistent_schemas(schema):
    with pytest.raises(DescriptorConfigError):
        DescriptorInfo("Example", "Example descriptor.", "local", {"value": schema})
