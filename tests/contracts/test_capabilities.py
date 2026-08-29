"""Registry-driven checks for declared descriptor capabilities."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import mdescriptor
from mdescriptor import (
    DescriptorConfigError,
    DescriptorConfiguration,
    DescriptorRegistry,
    DescriptorSpec,
    create_descriptor,
)

ROOT = Path(__file__).parents[2]
PACKAGE_ROOT = Path(mdescriptor.__file__).resolve().parent
GOLDEN_ROOT = ROOT / "tests" / "golden"


def _restore_paths(value):
    if isinstance(value, str):
        if value.startswith("${PACKAGE_ROOT}/"):
            return str(PACKAGE_ROOT / value.removeprefix("${PACKAGE_ROOT}/"))
        if value.startswith("${PROJECT_ROOT}/"):
            return str(ROOT / value.removeprefix("${PROJECT_ROOT}/"))
    if isinstance(value, dict):
        return {key: _restore_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_paths(item) for item in value]
    return value


def _configurations() -> dict[str, DescriptorConfiguration]:
    result: dict[str, DescriptorConfiguration] = {}
    for path in sorted(GOLDEN_ROOT.glob("*/manifest.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        name = case["descriptor"]
        if name not in result:
            result[name] = DescriptorConfiguration.from_dict(
                _restore_paths(case["configuration"]),
            )
    return result


@pytest.mark.model
def test_capability_declarations_match_public_constructor_behavior():
    configurations = _configurations()
    for spec in mdescriptor.builtin_registry:
        configuration = configurations[spec.name]
        descriptor_class = spec.load_class()
        signature = inspect.signature(descriptor_class)
        parameters = dict(configuration.parameters)

        assert "sparse" in spec.capabilities, spec.name
        assert ("model" in spec.capabilities) == ("model" in signature.parameters), spec.name
        assert ("cooperative_cancel" in spec.capabilities) == bool(
            spec.info and spec.info.execution.get("cooperative_cancel", False)
        ), spec.name
        assert ("spin" in spec.capabilities) == (spec.name in {"DPA4", "DPA4C"}), spec.name
        assert (
            "charge_spin" in spec.capabilities
        ) == (spec.name in {"DPA4", "DPA4C"}), spec.name

        sparse_parameters = dict(parameters)
        sparse_parameters["output"] = {"dtype": "float64", "sparse": True}
        sparse_configuration = DescriptorConfiguration(
            configuration.schema_version,
            configuration.descriptor,
            sparse_parameters,
        )
        sparse_descriptor = create_descriptor(sparse_configuration)
        sparse_descriptor.close()

        execution = dict(parameters.get("execution", {}))
        execution["device"] = "cpu"
        execution["num_threads"] = 1
        threaded_configuration = DescriptorConfiguration(
            configuration.schema_version,
            configuration.descriptor,
            {**parameters, "execution": execution},
        )
        if "num_threads" in spec.capabilities:
            descriptor = create_descriptor(threaded_configuration)
            descriptor.close()
        else:
            with pytest.raises(DescriptorConfigError, match="num_threads"):
                create_descriptor(threaded_configuration)


def test_registry_extension_rejects_parent_name_collisions():
    parent = DescriptorRegistry(
        [
            DescriptorSpec(
                "parent-name",
                "mdescriptor.descriptors.standalone.soap:SOAP",
                mdescriptor.AssetPolicy.NONE,
                "cpp",
                "structure",
            )
        ]
    )
    child = DescriptorRegistry(parent=parent)
    with pytest.raises(ValueError, match="already registered"):
        child.register(
            DescriptorSpec(
                "parent-name",
                "mdescriptor.descriptors.standalone.soap:SOAP",
                mdescriptor.AssetPolicy.NONE,
                "cpp",
                "structure",
            )
        )
