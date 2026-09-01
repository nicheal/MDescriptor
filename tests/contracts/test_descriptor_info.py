"""Contract tests for the static GUI-facing descriptor metadata seam."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys

import numpy as np
import pytest

import mdescriptor
from mdescriptor import (
    DescriptorConfigError,
    DescriptorInputError,
    DescriptorRegistry,
    DescriptorSpec,
    StructureBatch,
    describe_descriptor,
)
from mdescriptor._runtime import native_extension_available
from mdescriptor.registry import DescriptorInfo, validate_descriptor_parameters


def test_runtime_info_exposes_independent_contract_versions():
    runtime = mdescriptor.get_runtime_info()

    assert runtime["version"] == mdescriptor.__version__
    assert runtime["api_version"] == mdescriptor.API_VERSION
    assert runtime["configuration_schema_version"] == 1
    assert runtime["descriptor_info_schema_version"] == 3
    assert runtime["result_schema_version"] == mdescriptor.RESULT_SCHEMA_VERSION
    json.dumps(runtime, allow_nan=False)


def test_gui_baseline_is_available_as_a_versioned_runtime_resource():
    baseline = mdescriptor.gui_baseline()
    runtime = mdescriptor.get_runtime_info()

    assert baseline.startswith("# GUI adaptation baseline")
    assert runtime["baseline_version"] == mdescriptor.GUI_BASELINE_VERSION


def test_every_builtin_has_json_safe_static_metadata_matching_public_signature():
    expected_fields = {
        "schema_version",
        "name",
        "descriptor_version",
        "display_name",
        "description",
        "category",
        "level",
        "backend",
        "execution_engine",
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
        assert metadata["descriptor_version"] == spec.descriptor_version
        assert metadata["level"] == spec.level
        assert metadata["backend"] == spec.backend
        assert metadata["execution_engine"] == spec.execution_engine
        assert metadata["capabilities"] == sorted(spec.capabilities)
        assert metadata["asset"]["policy"] == spec.asset_policy.value
        expected_devices = (
            ["cpu", "cuda"]
            if spec.name
            in {
                "NeighborList",
                "SphericalExpansion",
                "SoapRadialSpectrum",
                "SoapPowerSpectrum",
                "NEP",
                "DPA4",
                "DPA4C",
            }
            else ["cpu"]
        )
        assert metadata["execution"]["devices"] == expected_devices
        assert metadata["input"]["mixed_periodicity"] == (
            set(metadata["input"]["periodicity"])
            == {"isolated", "fully_periodic"}
        )
        assert all("type" in schema for schema in metadata["parameters"].values())
        assert all(
            isinstance(schema.get("display_name"), str)
            and schema["display_name"].strip()
            and isinstance(schema.get("description"), str)
            and schema["description"].strip()
            for schema in metadata["parameters"].values()
        ), spec.name
        assert set(metadata["parameters"]) <= set(
            inspect.signature(spec.load_class()).parameters
        )
        json.dumps(metadata, allow_nan=False)


def test_parameter_presentation_names_do_not_change_configuration_keys():
    soap = describe_descriptor("SOAP")
    neighbor_list = describe_descriptor("NeighborList")
    so3 = describe_descriptor("SO3")

    assert set(soap["parameters"]) >= {"r_cut", "n_max"}
    assert set(neighbor_list["parameters"]) >= {"cutoff"}
    assert set(so3["parameters"]) >= {"rcut"}
    assert soap["parameters"]["r_cut"]["display_name"] == "Cutoff radius"
    assert neighbor_list["parameters"]["cutoff"]["display_name"] == "Cutoff radius"
    assert so3["parameters"]["rcut"]["display_name"] == "Cutoff radius"
    assert "r_cut" not in soap["parameters"]["r_cut"]["display_name"]
    assert "cutoff" not in neighbor_list["parameters"]["cutoff"]["display_name"]
    assert "rcut" not in so3["parameters"]["rcut"]["display_name"]


def test_dpa4c_static_default_matches_the_runtime_default():
    assert describe_descriptor("DPA4C")["parameters"]["calibrate"]["default"] is True


def test_dpa_static_metadata_separates_adapter_and_execution_engine():
    expected_engine = "cpp" if native_extension_available() else "numpy"
    for name in ("DPA4", "DPA4C"):
        metadata = describe_descriptor(name)
        assert metadata["backend"] == "numpy"
        assert metadata["execution_engine"] == expected_engine
        assert metadata["descriptor_version"] == "1"


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


def test_soapturbo_metadata_defaults_rebuild_for_multiple_species():
    metadata = describe_descriptor("SOAPTurbo")
    parameters = {"species": [1, 8]}
    parameters.update(
        {
            name: schema["default"]
            for name, schema in metadata["parameters"].items()
            if "default" in schema
        }
    )

    descriptor = mdescriptor.create_descriptor(
        mdescriptor.DescriptorConfiguration(1, "SOAPTurbo", parameters)
    )
    descriptor.close()


@pytest.mark.parametrize("name", ["MBTR", "LMBTR", "ValleOganov"])
def test_periodic_schema_only_advertises_supported_value(name):
    schema = describe_descriptor(name)["parameters"]["periodic"]

    assert schema["type"] == "boolean"
    assert schema["default"] is True
    assert schema["enum"] == [True]

    with pytest.raises(DescriptorConfigError) as caught:
        mdescriptor.create_descriptor(
            mdescriptor.DescriptorConfiguration(
                1,
                name,
                {"species": [1], "periodic": False},
            )
        )
    assert caught.value.code == "invalid_parameter"
    assert caught.value.to_dict()["path"] == ["parameters", "periodic"]


def test_per_species_broadcasts_are_written_as_canonical_arrays():
    from mdescriptor.descriptors import ACE, SOAPTurbo

    ace = ACE(species=[1, 8], maxdeg=4.0, wL=1.25)
    turbo = SOAPTurbo(species=[1, 8], alpha_max=3)
    try:
        ace_parameters = ace.configuration.to_dict()["parameters"]
        assert ace_parameters["maxdeg"] == [4.0, 4.0, 4.0]
        assert ace_parameters["wL"] == [1.25, 1.25, 1.25]

        turbo_parameters = turbo.configuration.to_dict()["parameters"]
        assert turbo_parameters["alpha_max"] == [3, 3]
        assert turbo_parameters["atom_sigma_r"] == [0.5, 0.5]
        assert turbo_parameters["atom_sigma_r_scaling"] == [0.0, 0.0]
        assert turbo_parameters["atom_sigma_t"] == [0.5, 0.5]
        assert turbo_parameters["atom_sigma_t_scaling"] == [0.0, 0.0]
        assert turbo_parameters["amplitude_scaling"] == [0.0, 0.0]
        assert turbo_parameters["central_weight"] == [1.0, 1.0]

        ace_rebuilt = mdescriptor.create_descriptor(
            mdescriptor.DescriptorConfiguration.from_dict(ace.configuration.to_dict())
        )
        turbo_rebuilt = mdescriptor.create_descriptor(
            mdescriptor.DescriptorConfiguration.from_dict(turbo.configuration.to_dict())
        )
        try:
            assert ace_rebuilt.configuration.to_dict() == ace.configuration.to_dict()
            assert turbo_rebuilt.configuration.to_dict() == turbo.configuration.to_dict()
        finally:
            ace_rebuilt.close()
            turbo_rebuilt.close()
    finally:
        ace.close()
        turbo.close()


def test_ace_degree_mapping_round_trips_configuration():
    from mdescriptor.descriptors import ACE

    descriptor = ACE(species=[1, 8], D={"type": "SparsePSHDegree"})
    try:
        configuration = mdescriptor.DescriptorConfiguration.from_dict(
            descriptor.configuration.to_dict()
        )
        rebuilt = mdescriptor.create_descriptor(configuration)
        try:
            assert rebuilt.configuration.to_dict() == descriptor.configuration.to_dict()
        finally:
            rebuilt.close()
    finally:
        descriptor.close()


@pytest.mark.parametrize("name", ["NEP", "DPA4", "DPA4C"])
@pytest.mark.model
def test_model_backed_metadata_declares_a_rebuildable_bundled_default(name):
    metadata = describe_descriptor(name)
    model = metadata["parameters"]["model"].get("default")

    assert isinstance(model, dict)
    assert model["__type__"] == "ModelResource"
    assert model["name"] in metadata["asset"]["bundled_resources"]

    parameters = {"model": model}
    for parameter_name, schema in metadata["parameters"].items():
        if parameter_name != "model" and "default" in schema:
            parameters[parameter_name] = schema["default"]
    descriptor = mdescriptor.create_descriptor(
        mdescriptor.DescriptorConfiguration(1, name, parameters)
    )
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


def test_descriptor_parameter_display_name_must_be_non_empty_string():
    with pytest.raises(DescriptorConfigError) as caught:
        DescriptorInfo(
            "Example",
            "Example descriptor.",
            "local",
            {"value": {"type": "number", "display_name": "   "}},
        )
    assert caught.value.code == "invalid_descriptor_info"
    assert caught.value.to_dict()["path"] == ["parameters", "value", "display_name"]


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
        "descriptor_version",
        "display_name",
        "description",
        "category",
        "level",
        "backend",
        "execution_engine",
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
    assert str(caught.value) == (
        "serialized model must be a path string or a ModelResource object"
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


def test_factory_binds_custom_registry_input_capabilities():
    base = describe_descriptor("SOAP")
    info = DescriptorInfo(
        base["display_name"],
        base["description"],
        base["category"],
        base["parameters"],
        base["execution"],
        {
            "periodicity": ["fully_periodic"],
            "mixed_periodicity": False,
            "spin": False,
            "charge_spin": False,
        },
        base["output"],
        base["asset"],
    )
    registry = DescriptorRegistry(
        [
            DescriptorSpec(
                "custom",
                "mdescriptor.descriptors.standalone.soap:SOAP",
                mdescriptor.AssetPolicy.NONE,
                "cpp",
                "structure",
                info=info,
            )
        ]
    )
    descriptor = mdescriptor.create_descriptor(
        mdescriptor.DescriptorConfiguration(
            1, "custom", {"species": [1], "r_cut": 3.0}
        ),
        registry=registry,
    )
    isolated = StructureBatch(
        np.array([1], dtype=np.int32),
        np.zeros((1, 3)),
        np.zeros((1, 3, 3)),
        np.zeros((1, 3), dtype=np.int32),
        np.array([0, 1], dtype=np.int64),
        ("one",),
    )
    try:
        with pytest.raises(DescriptorInputError) as caught:
            descriptor.compute(isolated)
        assert caught.value.to_dict()["path"] == ["input", "periodicity"]
    finally:
        descriptor.close()


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
