"""Run a minimal import/compute/hash check against an installed wheel tree."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _restore_paths(value, package_root=None):
    if isinstance(value, str):
        if package_root is not None and value.startswith("${PACKAGE_ROOT}/"):
            return str(package_root / value.removeprefix("${PACKAGE_ROOT}/"))
        if value.startswith("${PROJECT_ROOT}/"):
            return str(Path(__file__).resolve().parents[1] / value.removeprefix("${PROJECT_ROOT}/"))
    if isinstance(value, dict):
        return {key: _restore_paths(item, package_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_paths(item, package_root) for item in value]
    return value


def _batch(mdescriptor, path: Path, ids: tuple[str, ...]):
    import numpy as np

    with np.load(path) as arrays:
        return mdescriptor.StructureBatch(
            np.asarray(arrays["numbers"], dtype=np.int32),
            np.asarray(arrays["positions"], dtype=np.float64),
            np.asarray(arrays["cells"], dtype=np.float64),
            np.asarray(arrays["pbc"], dtype=np.int32),
            np.asarray(arrays["offsets"], dtype=np.int64),
            ids,
        )


def _single_structure(mdescriptor, batch, index: int):
    import numpy as np

    begin = int(batch.offsets[index])
    end = int(batch.offsets[index + 1])
    return mdescriptor.StructureBatch(
        batch.numbers[begin:end],
        batch.positions[begin:end],
        batch.cells[index : index + 1],
        batch.pbc[index : index + 1],
        np.asarray([0, end - begin], dtype=np.int64),
        (batch.ids[index],),
    )


def _verify_golden_fixtures(mdescriptor, golden_dir: Path) -> None:
    import numpy as np

    package_root = Path(mdescriptor.__file__).resolve().parent
    for manifest_path in sorted(golden_dir.glob("*/manifest.json")):
        fixture_dir = manifest_path.parent
        case = json.loads(manifest_path.read_text(encoding="utf-8"))
        configuration = mdescriptor.DescriptorConfiguration.from_dict(
            _restore_paths(case["configuration"], package_root)
        )
        descriptor = mdescriptor.create_descriptor(configuration)
        try:
            batch = _batch(mdescriptor, fixture_dir / case["input"], tuple(case["input_ids"]))
            compute_batch = (
                _single_structure(mdescriptor, batch, 0)
                if case["nonperiodic"]["mode"] != "output"
                else batch
            )
            result = descriptor.compute(compute_batch)
            with np.load(fixture_dir / case["expected_output"]) as arrays:
                expected_values = arrays["values"]
                expected_samples = arrays["samples"]
            tolerance = case["tolerance"]
            np.testing.assert_allclose(
                np.asarray(result.values),
                expected_values,
                rtol=tolerance["rtol"],
                atol=tolerance["atol"],
                err_msg=case["descriptor"],
            )
            np.testing.assert_array_equal(result.samples, expected_samples)
            expected = case["result"]
            if result.level.value != expected["level"]:
                raise AssertionError(f"{case['descriptor']} level changed")
            if result.feature_count != expected["feature_count"]:
                raise AssertionError(f"{case['descriptor']} feature count changed")
            if result.labels != tuple(expected["labels"]):
                raise AssertionError(f"{case['descriptor']} labels changed")
            if result.structure_ids != tuple(expected["structure_ids"]):
                raise AssertionError(f"{case['descriptor']} structure ids changed")
            expected_offsets = expected["row_offsets"]
            if expected_offsets is None:
                if result.row_offsets is not None:
                    raise AssertionError(f"{case['descriptor']} row offsets changed")
            else:
                np.testing.assert_array_equal(result.row_offsets, expected_offsets)
        finally:
            descriptor.close()


def _verify_vendored_openblas(mdescriptor) -> None:
    """Check the native extension's private OpenBLAS closure in an install."""

    package_root = Path(mdescriptor.__file__).resolve().parent
    native_candidates = [
        path
        for path in package_root.iterdir()
        if path.name.startswith("_native") and path.suffix in {".so", ".pyd", ".dylib"}
    ]
    if len(native_candidates) != 1:
        raise SystemExit(f"expected one native extension, found {native_candidates}")
    runtime_candidates = sorted(
        path
        for path in package_root.rglob("*")
        if path.is_file()
        and "openblas" in path.name.lower()
        and any(
            suffix in path.name.lower()
            for suffix in (".so", ".dylib", ".dll")
        )
    )
    if len(runtime_candidates) != 1:
        raise SystemExit(
            f"wheel must contain exactly one OpenBLAS runtime: {runtime_candidates}"
        )
    license_file = package_root / "licenses" / "scipy-openblas32-METADATA.txt"
    if not license_file.is_file() or license_file.stat().st_size == 0:
        raise SystemExit(f"missing bundled scipy-openblas license metadata: {license_file}")
    license_text = package_root / "licenses" / "scipy-openblas32-LICENSE.txt"
    if not license_text.is_file() or license_text.stat().st_size == 0:
        raise SystemExit(f"missing bundled scipy-openblas license text: {license_text}")

    native = native_candidates[0]
    if sys.platform.startswith("linux"):
        probe = subprocess.run(["ldd", str(native)], capture_output=True, text=True, check=False)
        if probe.returncode != 0 or "not found" in probe.stdout:
            raise SystemExit(f"unresolved native dependencies for {native}:\n{probe.stdout}")
    elif sys.platform == "darwin":
        probe = subprocess.run(
            ["otool", "-L", str(native)], capture_output=True, text=True, check=False
        )
        if probe.returncode != 0 or "scipy_openblas" not in probe.stdout:
            raise SystemExit(f"native extension is not linked to bundled OpenBLAS: {probe.stdout}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path)
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests" / "golden",
    )
    args = parser.parse_args(argv)
    target = None if args.target is None else args.target.resolve()
    if target is not None:
        sys.path.insert(0, str(target))
    # An editable scikit-build finder in the calling environment can override
    # sys.path during local verification.  A wheel check must resolve only the
    # supplied target tree; normal CI environments do not have this finder.
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if not finder.__class__.__module__.startswith("_editable_")
    ]

    import numpy as np

    import mdescriptor
    from mdescriptor.descriptors import NEP, SOAP
    from mdescriptor.models import DPA4_RESOURCE, DPA4C_RESOURCE, NEP_RESOURCE, ModelResolver

    package_file = Path(mdescriptor.__file__).resolve()
    if target is None:
        target = package_file.parent.parent
    if not package_file.is_relative_to(target):
        raise SystemExit(f"wheel verification imported {package_file}, not {target}")
    _verify_vendored_openblas(mdescriptor)
    names = mdescriptor.list_descriptors()
    if len(names) != 28 or len(set(names)) != 28:
        raise SystemExit(f"unexpected descriptor registry: {names!r}")

    batch = mdescriptor.StructureBatch(
        np.asarray([8, 1, 1], dtype=np.int32),
        np.asarray([[4.0, 4.0, 4.0], [4.76, 4.58, 4.0], [3.24, 4.58, 4.0]]),
        np.eye(3, dtype=np.float64)[None] * 12.0,
        np.ones((1, 3), dtype=np.int32),
        np.asarray([0, 3], dtype=np.int64),
        ("wheel-check",),
    )
    soap = SOAP(species=[1, 8], r_cut=3.5, n_max=2, l_max=2)
    try:
        assert soap.compute(batch).values.shape[0] == 1
    finally:
        soap.close()
    nep = NEP()
    try:
        assert nep.compute(batch).values.shape[0] == 3
    finally:
        nep.close()

    _verify_golden_fixtures(mdescriptor, args.golden_dir)

    from mdescriptor.descriptors import DPA4, DPA4C

    for descriptor_type in (DPA4, DPA4C):
        descriptor = descriptor_type()
        try:
            assert descriptor.compute(batch).values.shape[0] == 3
        finally:
            descriptor.close()
    for resource in (NEP_RESOURCE, DPA4_RESOURCE, DPA4C_RESOURCE):
        resolved = ModelResolver().resolve(resource)
        if resolved.digest != resource.expected_sha256:
            raise SystemExit(
                f"asset hash mismatch for {resource.name}: {resolved.digest}"
            )
    print(f"verified {package_file} ({len(names)} descriptors, including DPA4/DPA4C)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
