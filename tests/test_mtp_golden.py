from tests._golden import assert_descriptor_golden


def test_mtp_golden():
    assert_descriptor_golden("MTP")


def test_mtp_mlip4_official_fixture_remains_covered():
    # The MLIP-4 model has a distinct species map and is intentionally kept as
    # an extra MTP mode rather than forced into the common HEA/H2O fixture.
    from pathlib import Path

    import numpy as np

    from tests._public import MTP, ExecutionOptions, StructureBatch

    potential = Path(__file__).parents[1] / "tests" / "data" / "mlip4_test_mtp.json"
    batch = StructureBatch(
        np.asarray([13, 14], dtype=np.int32),
        np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.eye(3, dtype=np.float64)[None] * 10.0,
        np.ones((1, 3), dtype=np.int32),
        np.asarray([0, 2], dtype=np.int64),
        ("mlip4",),
    )
    descriptor = MTP(species=[13, 14], model=potential, execution=ExecutionOptions(num_threads=1))
    try:
        result = descriptor.compute(batch)
    finally:
        descriptor.close()
    assert result.values.shape == (2, 5)
    assert result.labels == tuple(f"mlip4:basis={index}" for index in range(5))
    assert result.metadata["details"]["official_mlip4"] is True
