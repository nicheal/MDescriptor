import hashlib
import inspect
import json
import subprocess
import sys

import numpy as np
import pytest

import mdescriptor
from mdescriptor.core import (
    ClosedDescriptorError,
    Descriptor,
    DescriptorConfigError,
    DescriptorConfiguration,
    DescriptorResult,
    ExecutionOptions,
    ModelLoadError,
    OutputOptions,
    StructureBatch,
)
from mdescriptor.descriptors import NEP, SOAP, AtomicComposition, CoulombMatrix
from mdescriptor.models import NEP_MODEL, ModelResolver, ModelResource


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
    assert "SOAP" in mdescriptor.list_descriptors()


def test_root_import_does_not_load_torch_or_model_modules():
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import mdescriptor; assert 'torch' not in sys.modules; "
            "assert not any(name.startswith('mdescriptor.models') for name in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0


def test_every_registered_descriptor_uses_the_single_compute_boundary():
    for name in mdescriptor.list_descriptors():
        descriptor_class = mdescriptor.get_descriptor(name)
        assert descriptor_class.compute is Descriptor.compute, name
        signature = inspect.signature(descriptor_class)
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for parameter in signature.parameters.values()
        ), name
        assert not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ), name


def test_builtin_registry_is_immutable_and_children_are_extendable():
    spec = mdescriptor.DescriptorSpec(
        "custom",
        "mdescriptor.descriptors.standalone.soap:SOAP",
        mdescriptor.AssetPolicy.NONE,
        "cpp",
        "atom",
    )
    with pytest.raises(TypeError):
        mdescriptor.builtin_registry.register(spec)
    child = mdescriptor.DescriptorRegistry(parent=mdescriptor.builtin_registry)
    child.register(spec)
    assert child.get("custom") is spec


def test_builtin_levels_describe_default_output_granularity():
    assert mdescriptor.builtin_registry.get("SOAP").level == "structure"
    assert mdescriptor.builtin_registry.get("AtomicComposition").level == "structure"
    assert mdescriptor.builtin_registry.get("LMBTR").level == "atom"
    assert mdescriptor.builtin_registry.get("NeighborList").level == "pair"
    assert all("sparse" in spec.capabilities for spec in mdescriptor.builtin_registry)
    for name in ("DPA4", "DPA4C"):
        spec = mdescriptor.builtin_registry.get(name)
        assert spec.backend == "numpy"
        assert "cuda" not in spec.capabilities
        assert spec.optional_extra is None


def test_result_is_json_safe_and_lifecycle_is_uniform():
    descriptor = SOAP(
        species=[1, 8], r_cut=3.0, n_max=1, l_max=1,
        output=OutputOptions(dtype="float32"), execution=ExecutionOptions(num_threads=1),
    )
    result = descriptor.compute(_batch())
    assert isinstance(result, DescriptorResult)
    assert result.feature_count == result.values.shape[1]
    json.dumps(result.metadata)
    assert descriptor.metadata == result.metadata
    descriptor.close()
    descriptor.close()
    assert descriptor.closed
    assert descriptor.metadata == result.metadata
    configuration = descriptor.configuration
    serialized = configuration.to_dict()
    json.dumps(serialized)
    rebuilt = mdescriptor.create_descriptor(
        DescriptorConfiguration.from_dict(serialized)
    )
    assert rebuilt.configuration.to_dict() == serialized
    rebuilt.close()
    with pytest.raises(ClosedDescriptorError):
        descriptor.compute(_batch())


def test_model_resolver_is_local_and_checksum_aware(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"local checkpoint")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    resolved = ModelResolver().resolve(
        ModelResource.explicit(path, expected_sha256=digest)
    )
    assert resolved.path == path.resolve()
    assert resolved.source == "explicit"
    with pytest.raises(ModelLoadError):
        ModelResolver().resolve(ModelResource.explicit(path, expected_sha256="0" * 64))


def test_default_model_backed_descriptor_uses_the_resource_resolver():
    descriptor = NEP()
    try:
        assert descriptor.model_resource is not None
        assert descriptor.resolved_model is not None
        assert descriptor.resolved_model.path == NEP_MODEL.resolve()
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
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        SOAP(unknown_option=True)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        SOAP(config={"unknown_option": True})
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        NEP(model_path="not-a-public-entry-point.txt")
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        NEP(config={"model_path": "not-a-public-entry-point.txt"})


def test_dpa_public_options_do_not_expose_torch_runtime_controls():
    from mdescriptor.descriptors import DPA4, DPA4C

    assert "use_amp" not in inspect.signature(DPA4C).parameters
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        DPA4C(use_amp=False)
    with pytest.raises(DescriptorConfigError, match="does not support execution.device"):
        DPA4(execution=ExecutionOptions(device="cuda"))


def test_common_options_are_applied_or_rejected_at_the_core_boundary():
    descriptor = CoulombMatrix(output=OutputOptions(dtype="float32"))
    result = descriptor.compute(_batch())
    assert result.values.dtype == np.float32
    descriptor.close()

    threaded = AtomicComposition(species=[1, 8], execution=ExecutionOptions(num_threads=2))
    try:
        threaded_result = threaded.compute(_batch())
        assert threaded_result.values.shape == (1, 2)
    finally:
        threaded.close()


def test_result_validates_row_offsets_and_structure_rows():
    with pytest.raises(ValueError, match="final row offset"):
        DescriptorResult(
            np.zeros((2, 3)), "atom", ("a",), np.array([0, 99]), ("a", "b", "c"), {}
        )
    with pytest.raises(ValueError, match="structure_ids"):
        DescriptorResult(np.zeros((2, 3)), "structure", ("a",), None, ("a", "b", "c"), {})
