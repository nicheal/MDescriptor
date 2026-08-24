import hashlib
import inspect
import json

import numpy as np
import pytest

import mdescriptor
from mdescriptor.core import (
    ClosedDescriptorError,
    DescriptorConfigError,
    DescriptorResult,
    ExecutionOptions,
    ModelLoadError,
    OutputOptions,
    StructureBatch,
)
from mdescriptor.descriptors import NEP, SOAP
from mdescriptor.descriptors import AtomicComposition, CoulombMatrix
from mdescriptor.models import ModelResource, ModelResolver, NEP_MODEL


def _batch() -> StructureBatch:
    return StructureBatch(
        np.array([1, 8], dtype=np.int32),
        np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]]),
        np.eye(3, dtype=np.float64)[None] * 10.0,
        np.ones((1, 3), dtype=np.int32),
        np.array([0, 2], dtype=np.int64),
        ("contract-0",),
    )


def test_root_exposes_contracts_and_not_historical_algorithm_aliases():
    assert hasattr(mdescriptor, "Descriptor")
    assert hasattr(mdescriptor, "create_descriptor")
    assert "SoapCalculator" not in mdescriptor.__all__
    assert "DESCRIPTOR_CATALOG" not in mdescriptor.__all__
    assert "SOAP" in mdescriptor.list_descriptors()


def test_builtin_registry_is_immutable_and_children_are_extendable():
    spec = mdescriptor.DescriptorSpec(
        "custom",
        "mdescriptor.descriptors.standalone:SOAP",
        mdescriptor.AssetPolicy.NONE,
        "cpp",
        "atom",
    )
    with pytest.raises(TypeError):
        mdescriptor.BUILTIN_REGISTRY.register(spec)
    child = mdescriptor.BUILTIN_REGISTRY.child()
    child.register(spec)
    assert child.get("custom") is spec


def test_builtin_levels_describe_default_output_granularity():
    assert mdescriptor.BUILTIN_REGISTRY.get("SOAP").level == "structure"
    assert mdescriptor.BUILTIN_REGISTRY.get("AtomicComposition").level == "structure"
    assert mdescriptor.BUILTIN_REGISTRY.get("LMBTR").level == "atom"
    assert mdescriptor.BUILTIN_REGISTRY.get("NeighborList").level == "pair"
    assert all("sparse" in spec.capabilities for spec in mdescriptor.BUILTIN_REGISTRY)


def test_result_is_json_safe_and_lifecycle_is_uniform():
    descriptor = SOAP(species=[1, 8], r_cut=3.0, n_max=1, l_max=1)
    result = descriptor.compute(_batch())
    assert isinstance(result, DescriptorResult)
    assert result.feature_count == result.values.shape[1]
    json.dumps(result.metadata)
    descriptor.close()
    descriptor.close()
    assert descriptor.closed
    with pytest.raises(ClosedDescriptorError):
        descriptor.compute(_batch())


def test_model_resolver_is_local_and_checksum_aware(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"local checkpoint")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert ModelResolver().resolve(ModelResource(path, expected_sha256=digest)) == path
    with pytest.raises(ModelLoadError):
        ModelResolver().resolve(ModelResource(path, expected_sha256="0" * 64))


def test_default_model_backed_descriptor_uses_the_resource_resolver():
    descriptor = NEP()
    try:
        assert descriptor.model_resource is not None
        assert descriptor.model_resource.path == NEP_MODEL.resolve()
        assert descriptor.session is not None
        assert descriptor.session.model.path == NEP_MODEL.resolve()
    finally:
        descriptor.close()


def test_adapters_have_explicit_signatures_and_no_legacy_attribute_leak():
    signature = inspect.signature(SOAP)
    assert "*args" not in str(signature)
    assert "**kwargs" not in str(signature)
    assert "output" in signature.parameters
    assert "execution" in signature.parameters
    descriptor = SOAP(species=[1, 8], r_cut=3.0, n_max=1, l_max=1)
    try:
        assert not hasattr(descriptor, "create")
        assert not hasattr(descriptor, "_labels")
    finally:
        descriptor.close()


def test_public_adapters_reject_unknown_and_legacy_model_options():
    with pytest.raises(DescriptorConfigError, match="unexpected keyword|unsupported option"):
        SOAP(unknown_option=True)
    with pytest.raises(DescriptorConfigError, match="unsupported option"):
        SOAP(config={"unknown_option": True})
    with pytest.raises(DescriptorConfigError, match="model="):
        NEP(model_path="not-a-public-entry-point.txt")
    with pytest.raises(DescriptorConfigError, match="unsupported option"):
        NEP(config={"model_path": "not-a-public-entry-point.txt"})


def test_common_options_are_applied_or_rejected_at_the_core_boundary():
    descriptor = CoulombMatrix(output=OutputOptions(dtype="float32"))
    result = descriptor.compute(_batch())
    assert result.values.dtype == np.float32
    descriptor.close()

    with pytest.raises(DescriptorConfigError, match="does not support execution.num_threads"):
        AtomicComposition(execution=ExecutionOptions(num_threads=2))


def test_result_validates_row_offsets_and_structure_rows():
    with pytest.raises(ValueError, match="final row offset"):
        DescriptorResult(
            np.zeros((2, 3)), "atom", ("a",), np.array([0, 99]), (), {}
        )
    with pytest.raises(ValueError, match="structure_ids"):
        DescriptorResult(np.zeros((2, 3)), "structure", ("a",), None, (), {})
