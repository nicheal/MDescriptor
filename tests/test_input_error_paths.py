"""Input validation reports the field known at the validation boundary."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mdescriptor import Descriptor, DescriptorInputError, StructureBatch
from mdescriptor.core.adapter import DescriptorAdapter
from mdescriptor.core.descriptor import _input_error_path
from mdescriptor.core.errors import _InputValidationError


def _batch_kwargs() -> dict[str, Any]:
    return {
        "numbers": np.asarray([1], dtype=np.int32),
        "positions": np.zeros((1, 3), dtype=np.float64),
        "cells": np.eye(3, dtype=np.float64)[None],
        "pbc": np.ones((1, 3), dtype=np.int32),
        "offsets": np.asarray([0, 1], dtype=np.int64),
        "ids": ("frame",),
    }


class _Probe(Descriptor):
    name = "input-path-probe"

    def _compute_batch(self, batch: StructureBatch, *, control=None):
        del batch, control
        raise AssertionError("invalid input must stop before the kernel")


class _ErrorBackend:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def compute(self, batch: StructureBatch, control: object) -> None:
        del batch, control
        raise self.error


def _adapter_for_backend(error: BaseException) -> DescriptorAdapter:
    adapter = object.__new__(DescriptorAdapter)
    adapter._closed = False
    adapter._backend = _ErrorBackend(error)
    return adapter


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("numbers", np.asarray([0], dtype=np.int32)),
        ("positions", np.zeros((1, 2), dtype=np.float64)),
        ("offsets", np.asarray([0, 2], dtype=np.int64)),
        ("spins", np.zeros((1, 2), dtype=np.float64)),
        ("charge_spin", np.zeros((1, 1), dtype=np.float64)),
    ],
)
def test_descriptor_preserves_validator_field_path(field: str, value: Any, monkeypatch) -> None:
    kwargs = _batch_kwargs()
    kwargs[field] = value
    descriptor = _Probe()
    monkeypatch.setattr(
        descriptor,
        "_as_batch",
        lambda _value: StructureBatch(**kwargs),
    )

    with pytest.raises(DescriptorInputError) as caught:
        descriptor.compute(object())
    assert caught.value.to_dict()["path"] == ["input", field]


def test_frame_validator_carries_missing_field_without_changing_direct_value_error() -> None:
    with pytest.raises(ValueError) as caught:
        StructureBatch.from_frames(
            {
                "positions": np.zeros((1, 3)),
                "cell": np.eye(3),
                "pbc": [1, 1, 1],
                "id": "frame",
            }
        )
    assert not isinstance(caught.value, DescriptorInputError)
    assert caught.value.path == ("input", "numbers")


def test_unknown_value_error_does_not_guess_from_its_message() -> None:
    class ThirdPartyProbe(_Probe):
        def _validate_batch(self, batch: StructureBatch) -> None:
            del batch
            raise ValueError("vendor mentions positions and species")

    with pytest.raises(DescriptorInputError) as caught:
        ThirdPartyProbe().compute(StructureBatch(**_batch_kwargs()))
    assert caught.value.to_dict()["path"] == ["input"]
    assert _input_error_path(ValueError("vendor mentions positions")) == ["input"]


def test_explicit_internal_path_wins_even_when_message_has_no_field() -> None:
    class StructuredProbe(_Probe):
        def _validate_batch(self, batch: StructureBatch) -> None:
            del batch
            raise _InputValidationError("opaque validator failure", ("input", "numbers"))

    with pytest.raises(DescriptorInputError) as caught:
        StructuredProbe().compute(StructureBatch(**_batch_kwargs()))
    assert caught.value.to_dict()["path"] == ["input", "numbers"]


def test_adapter_preserves_internal_validator_path() -> None:
    adapter = _adapter_for_backend(
        _InputValidationError("opaque backend validation", ("input", "positions"))
    )

    with pytest.raises(DescriptorInputError) as caught:
        adapter.compute(StructureBatch(**_batch_kwargs()))
    assert caught.value.to_dict()["path"] == ["input", "positions"]


def test_adapter_does_not_guess_path_from_vendor_error_text() -> None:
    adapter = _adapter_for_backend(ValueError("vendor mentions positions and species"))

    with pytest.raises(DescriptorInputError) as caught:
        adapter.compute(StructureBatch(**_batch_kwargs()))
    assert caught.value.to_dict()["path"] == ["input"]
