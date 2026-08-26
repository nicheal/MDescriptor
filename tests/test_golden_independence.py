"""Guard the boundary between committed tests and local benchmark state."""

from __future__ import annotations

import json

from tests._golden import GOLDEN_ROOT


def test_golden_fixtures_are_self_contained() -> None:
    manifests = sorted(GOLDEN_ROOT.glob("*/manifest.json"))
    assert len(manifests) == 27
    for manifest_path in manifests:
        fixture_dir = manifest_path.parent
        manifest_text = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(manifest_text)
        assert "benchmarks" not in manifest_text
        assert (fixture_dir / manifest["input"]).is_file()
        assert (fixture_dir / manifest["expected_output"]).is_file()
        assert manifest["dataset"]["source"] == "promoted-local-dataset"
