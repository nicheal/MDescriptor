"""Default-job checks for provider-generated static numerical goldens."""

from __future__ import annotations

import pytest
from scripts.numerical_baselines import BASELINES

from tests._golden import GOLDEN_ROOT, assert_descriptor_external_static_golden

pytestmark = pytest.mark.model


def test_all_promoted_external_static_goldens() -> None:
    names = sorted(
        manifest.parent.name
        for manifest in GOLDEN_ROOT.glob("*/external_manifest.json")
        if manifest.is_file()
    )
    assert len(names) == 21
    for name in names:
        descriptor = next(
            descriptor
            for descriptor in BASELINES
            if descriptor.lower() == name
        )
        assert BASELINES[descriptor]["kind"] == "external_static"
        assert_descriptor_external_static_golden(descriptor)
