"""Run a minimal import/compute/hash check against an installed wheel tree."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _restore_paths(value):
    if isinstance(value, str) and value.startswith("${PROJECT_ROOT}/"):
        return str(Path(__file__).resolve().parents[1] / value.removeprefix("${PROJECT_ROOT}/"))
    if isinstance(value, dict):
        return {key: _restore_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_paths(item) for item in value]
    return value


def _batch(mdescriptor, payload):
    import numpy as np

    return mdescriptor.StructureBatch(
        np.asarray(payload["numbers"], dtype=np.int32),
        np.asarray(payload["positions"], dtype=np.float64),
        np.asarray(payload["cells"], dtype=np.float64),
        np.asarray(payload["pbc"], dtype=np.int32),
        np.asarray(payload["offsets"], dtype=np.int64),
        tuple(payload["ids"]),
    )


def _verify_standalone_baselines(mdescriptor, baseline_dir: Path) -> None:
    import numpy as np

    manifest = json.loads((baseline_dir / "manifest.json").read_text(encoding="utf-8"))
    model_names = {"NEP", "DPA4", "DPA4C", "MTP-MLIP4"}
    for case in manifest["cases"]:
        if case["name"] in model_names:
            continue
        configuration = mdescriptor.DescriptorConfiguration.from_dict(
            _restore_paths(case["configuration"])
        )
        descriptor = mdescriptor.create_descriptor(configuration)
        try:
            result = descriptor.compute(_batch(mdescriptor, case["input"]))
            with np.load(baseline_dir / case["values"]) as arrays:
                expected_values = arrays["values"]
                expected_samples = arrays["samples"]
            tolerance = case["tolerance"]
            np.testing.assert_allclose(
                np.asarray(result.values),
                expected_values,
                rtol=tolerance["rtol"],
                atol=tolerance["atol"],
                err_msg=case["name"],
            )
            np.testing.assert_array_equal(result.samples, expected_samples)
            if result.level.value != case["level"]:
                raise AssertionError(f"{case['name']} level changed")
            if result.feature_count != case["feature_count"]:
                raise AssertionError(f"{case['name']} feature count changed")
            if result.labels != tuple(case["labels"]):
                raise AssertionError(f"{case['name']} labels changed")
            if result.structure_ids != tuple(case["structure_ids"]):
                raise AssertionError(f"{case['name']} structure ids changed")
            expected_offsets = case["row_offsets"]
            if expected_offsets is None:
                if result.row_offsets is not None:
                    raise AssertionError(f"{case['name']} row offsets changed")
            else:
                np.testing.assert_array_equal(result.row_offsets, expected_offsets)
        finally:
            descriptor.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests" / "data" / "numerical_baselines",
    )
    parser.add_argument(
        "--model",
        action="store_true",
        help="also compute the bundled DPA4 and DPA4C models",
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
    names = mdescriptor.list_descriptors()
    if len(names) != 27 or len(set(names)) != 27:
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

    _verify_standalone_baselines(mdescriptor, args.baseline_dir)

    if args.model:
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
    mode = " with model descriptors" if args.model else ""
    print(f"verified {package_file} ({len(names)} descriptors){mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
