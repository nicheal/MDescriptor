"""Guard the boundary between committed tests and local benchmark state."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scripts.external_reference import external_c00ps_project_columns, sha256

from tests._golden import GOLDEN_ROOT, ROOT


def test_golden_fixtures_are_self_contained() -> None:
    manifests = sorted(GOLDEN_ROOT.glob("*/manifest.json"))
    assert len(manifests) == 28
    for manifest_path in manifests:
        fixture_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Fixture data stays portable and local; provenance may legitimately
        # point at the checked-in upstream-oracle adapters under benchmarks/.
        portable_manifest = {key: value for key, value in manifest.items() if key != "reference"}
        assert "benchmarks" not in json.dumps(portable_manifest)
        assert Path(manifest["input"]).parent == Path(".")
        assert Path(manifest["expected_output"]).parent == Path(".")
        assert (fixture_dir / manifest["input"]).is_file()
        assert (fixture_dir / manifest["expected_output"]).is_file()
        assert manifest["dataset"]["source"] == "promoted-local-dataset"


def test_ace_and_c00ps_use_external_source_references() -> None:
    expected = {
        "ace": ("ace1_julia_source", "source_archive_sha256"),
        "c00psmlff": ("licensed_external_mlff_source", "source_archive_sha256"),
    }
    for name, (kind, digest_key) in expected.items():
        manifest_path = GOLDEN_ROOT / name / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        reference = manifest["reference"]
        assert reference["kind"] == kind
        assert len(reference[digest_key]) == 64
        evaluator = ROOT / reference["evaluator"]
        assert evaluator.is_file()
        assert reference["evaluator_sha256"] == sha256(evaluator)

    c00_reference = json.loads(
        (GOLDEN_ROOT / "c00psmlff" / "manifest.json").read_text(encoding="utf-8")
    )["reference"]
    column_mapper = ROOT / c00_reference["column_mapper"]
    assert column_mapper.is_file()
    assert c00_reference["column_mapper_sha256"] == sha256(column_mapper)

    ace_reference = json.loads((GOLDEN_ROOT / "ace" / "manifest.json").read_text(encoding="utf-8"))[
        "reference"
    ]
    ace_generator = ROOT / ace_reference["generator"]
    assert ace_reference["generator_sha256"] == sha256(ace_generator)

    assert "local-only" in c00_reference["distribution_boundary"]


def test_dpa_goldens_use_pinned_deepmd_for_nonperiodic_rows() -> None:
    for name in ("dpa4", "dpa4c"):
        manifest = json.loads(
            (GOLDEN_ROOT / name / "manifest.json").read_text(encoding="utf-8")
        )
        reference = manifest["reference"]
        assert reference["kind"] == "deepmd_kit"
        assert reference["package"] == "deepmd-kit"
        assert reference["version"] == "3.2.0"
        model = manifest["configuration"]["parameters"]["model"]
        assert reference["model_sha256"] == model["expected_sha256"]
        assert reference["nonperiodic"] == {
            "mode": "cells_none",
            "source": "deepmd-kit.eval_descriptor / graph-native call_graph",
        }
        evaluator = ROOT / reference["evaluator"]
        assert evaluator.is_file()
        assert reference["evaluator_sha256"] == sha256(evaluator)


def test_upstream_goldens_use_pinned_independent_oracles() -> None:
    lock = json.loads(
        (ROOT / "benchmarks/_legacy_oracles/sources.lock.json").read_text(encoding="utf-8")
    )["oracles"]
    expected = {
        "soapturbo": ("soapturbo", "archive", "sha256"),
        "lbispectrum": ("lbispectrum", "lammps_archive", "lammps_sha256"),
        "mtp": ("mtp", "archive", "sha256"),
    }
    for name, (lock_name, archive_key, digest_key) in expected.items():
        manifest = json.loads(
            (GOLDEN_ROOT / name / "manifest.json").read_text(encoding="utf-8")
        )
        reference = manifest["reference"]
        assert reference["kind"] == "external_upstream"
        assert reference["source_archive"] == lock[lock_name][archive_key]
        assert reference["source_sha256"] == lock[lock_name][digest_key]
        for field in ("adapter", "generator"):
            path = ROOT / reference[field]
            assert path.is_file()
            assert reference[f"{field}_sha256"] == sha256(path)


def test_external_c00ps_mapping_follows_source_loop_order() -> None:
    # Two species and radial counts [2, 1] make the source ordering differ
    # from the historical (incorrect) project-column alignment.
    indices = external_c00ps_project_columns(
        species_count=2,
        radial_counts=(2, 1),
        include_radial=True,
    )
    expected = np.asarray(
        [
            0,
            1,
            2,
            3,  # radial C00: species then radial
            4,
            5,
            8,
            9,
            6,
            13,
            10,
            16,
            17,
            18,  # l=0 project upper triangle
            7,
            11,
            19,  # l=1 project upper triangle
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(indices, expected)
