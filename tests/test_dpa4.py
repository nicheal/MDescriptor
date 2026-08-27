import math
import subprocess
import sys

import numpy as np
import pytest

from mdescriptor.descriptors.model_backed.dpa import compute_batch, load_dpa_checkpoint, new_runtime
from mdescriptor.models import DPA4_MODEL
from tests._public import DPA4, ExecutionOptions, ModelLoadError, StructureBatch

pytestmark = pytest.mark.model

MODEL = DPA4_MODEL


def _batch() -> StructureBatch:
    return StructureBatch(
        np.array([1, 8, 1, 8], dtype=np.int32),
        np.array(
            [
                [8.0, 8.0, 8.0],
                [9.0, 8.0, 8.0],
                [10.0, 10.0, 10.0],
                [28.0, 28.0, 28.0],
            ]
        ),
        np.array([np.eye(3) * 20.0, np.eye(3) * 20.0]),
        np.ones((2, 3), dtype=np.int32),
        np.array([0, 3, 4], dtype=np.int64),
        ("first", "second"),
    )


def test_official_checkpoint_and_batch_output():
    calculator = DPA4(model=MODEL)

    result = calculator.compute(_batch())

    assert result.values.shape == (4, 64)
    assert np.isfinite(result.values).all()
    assert result.level == "atom"
    assert result.metadata["backend"] == "mdescriptor-dpa4-cpp"
    assert result.row_offsets.tolist() == [0, 3, 4]
    assert result.labels[0] == "dpa4:scalar,channel=0"


def test_geometry_rotation_and_atom_permutation_are_invariant():
    calculator = DPA4(model=MODEL)
    batch = _batch()
    reference = calculator.compute(batch).values

    angle = 0.37
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = StructureBatch(
        batch.numbers,
        batch.positions @ rotation.T,
        batch.cells @ rotation.T,
        batch.pbc,
        batch.offsets,
        batch.ids,
    )
    np.testing.assert_allclose(calculator.compute(rotated).values, reference, atol=3e-5)

    order = np.array([2, 0, 1, 3])
    permuted = StructureBatch(
        batch.numbers[order],
        batch.positions[order],
        batch.cells,
        batch.pbc,
        batch.offsets,
        batch.ids,
    )
    np.testing.assert_allclose(calculator.compute(permuted).values, reference[order], atol=2e-5)


def test_native_backend_matches_bundled_numpy_reference():
    calculator = DPA4(model=MODEL)
    _info, checkpoint = load_dpa_checkpoint(MODEL, expected_descriptor="DPA4")
    reference_runtime = new_runtime(MODEL, checkpoint)
    try:
        expected = compute_batch(reference_runtime, _batch())
        actual = calculator.compute(_batch()).values
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=4e-5)
    finally:
        calculator.close()


def test_dpa4_empty_frame_is_independent_of_other_batch_frames():
    batch = _batch()
    calculator = DPA4(model=MODEL)
    try:
        batched = calculator.compute(batch).values[3]
        isolated = StructureBatch(
            batch.numbers[3:4],
            batch.positions[3:4],
            batch.cells[1:2],
            batch.pbc[1:2],
            np.asarray([0, 1], dtype=np.int64),
            ("isolated",),
        )
        np.testing.assert_array_equal(batched, calculator.compute(isolated).values[0])
    finally:
        calculator.close()


def test_native_backend_is_thread_stable():
    batch = _batch()
    serial = DPA4(model=MODEL, execution=ExecutionOptions(num_threads=1))
    threaded = DPA4(model=MODEL, execution=ExecutionOptions(num_threads=4))
    try:
        np.testing.assert_array_equal(
            serial.compute(batch).values,
            threaded.compute(batch).values,
        )
    finally:
        serial.close()
        threaded.close()


def test_dpa4_rejects_project_native_archive(tmp_path):
    path = tmp_path / "mdescriptor.pt"
    path.write_bytes(b"not a DeepMD checkpoint")

    with pytest.raises(ModelLoadError, match="invalid DPA4 checkpoint"):
        DPA4(model=path)


def test_dpa4_and_dpa4c_compute_without_importing_torch():
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins
import sys
import numpy as np

real_import = builtins.__import__
def reject_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("DPA runtime imported Torch")
    return real_import(name, *args, **kwargs)
builtins.__import__ = reject_torch

from mdescriptor import StructureBatch
from mdescriptor.descriptors import DPA4, DPA4C

batch = StructureBatch(
    np.array([1, 8], dtype=np.int32),
    np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]]),
    np.eye(3, dtype=np.float64)[None] * 12.0,
    np.ones((1, 3), dtype=np.int32),
    np.array([0, 2], dtype=np.int64),
    ("no-torch",),
)
for descriptor_type in (DPA4, DPA4C):
    descriptor = descriptor_type()
    assert descriptor.compute(batch).values.shape[0] == 2
    descriptor.close()
assert "torch" not in sys.modules
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr


def test_dpa_sessions_share_only_loaded_model_and_close_independently():
    first = DPA4()
    second = DPA4()
    try:
        assert first.session is not None
        assert second.session is not None
        assert first.session.model is second.session.model
        assert first.session.runtime is not second.session.runtime
        first.close()
        assert second.closed is False
        assert second.compute(_batch()).values.shape == (4, 64)
    finally:
        second.close()
