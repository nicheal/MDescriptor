"""Small precision and timing regressions for Valle--Oganov threading."""

from __future__ import annotations

import time

import numpy as np
import pytest
from ase import Atoms

from tests._public import ExecutionOptions, StructureBatch, ValleOganov, builtin_registry


def _water_structure(molecules: int, cell_size: float) -> Atoms:
    motif = np.asarray(
        [[1.00, 1.00, 1.00], [1.96, 1.00, 1.00], [0.76, 1.93, 1.00]],
        dtype=np.float64,
    )
    symbols: list[str] = []
    positions: list[np.ndarray] = []
    for molecule in range(molecules):
        shift = np.asarray([3.0 * (molecule % 2), 3.0 * (molecule // 2), 0.0])
        symbols.extend(("O", "H", "H"))
        positions.extend(motif + shift)
    return Atoms(
        symbols=symbols,
        positions=np.asarray(positions),
        cell=np.diag([cell_size] * 3),
        pbc=True,
    )


def _batch() -> StructureBatch:
    # Different atom counts and cell volumes make the Valle--Oganov factors
    # structure-local: species counts and volume must not be shared by workers.
    systems = [
        _water_structure(1, 8.0),
        _water_structure(2, 9.0),
        _water_structure(3, 10.0),
        _water_structure(4, 11.0),
        _water_structure(2, 12.0),
        _water_structure(4, 13.0),
    ]
    return StructureBatch.from_ase(systems, ids=[f"water-{index}" for index in range(len(systems))])


def _single_structure(batch: StructureBatch, index: int) -> StructureBatch:
    begin = int(batch.offsets[index])
    end = int(batch.offsets[index + 1])
    return StructureBatch(
        batch.numbers[begin:end],
        batch.positions[begin:end],
        batch.cells[index : index + 1],
        batch.pbc[index : index + 1],
        np.asarray([0, end - begin], dtype=np.int64),
        (batch.ids[index],),
    )


def _descriptor(
    function: str, num_threads: int, normalization: str | None = None
) -> ValleOganov:
    return ValleOganov(
        species=[1, 8],
        function=function,
        n=24,
        sigma=0.08,
        r_cut=3.25,
        normalization=normalization,
        execution=ExecutionOptions(num_threads=num_threads),
    )


@pytest.mark.parametrize("function", ("distance", "angle"))
def test_valleoganov_openmp_matches_serial_and_is_repeatable(function: str) -> None:
    """Both Valle geometry paths retain values and deterministic output order."""

    batch = _batch()
    serial = _descriptor(function, 1)
    threaded = _descriptor(function, 2)
    try:
        serial_result = serial.compute(batch)
        threaded_result = threaded.compute(batch)
        expected = serial_result.values
        actual = threaded_result.values
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)
        np.testing.assert_array_equal(actual, threaded.compute(batch).values)

        # This also catches normalization state accidentally shared between
        # structures when workers compute species counts or cell volumes.
        for index in range(batch.structures):
            single = _descriptor(function, 1)
            try:
                expected_row = single.compute(_single_structure(batch, index)).values
            finally:
                single.close()
            np.testing.assert_allclose(
                actual[index : index + 1], expected_row, rtol=2e-12, atol=2e-13
            )
    finally:
        serial.close()
        threaded.close()


def test_valleoganov_declares_openmp_thread_support() -> None:
    assert "num_threads" in builtin_registry.get("ValleOganov").capabilities


def _pair_channel(first: int, second: int, species_count: int) -> int:
    lower = min(first, second)
    upper = max(first, second)
    return lower * species_count - lower * (lower + 1) // 2 + upper


@pytest.mark.parametrize("function", ("distance", "angle"))
def test_valleoganov_openmp_preserves_structure_local_normalization(function: str) -> None:
    """Valle factors use the current row's species counts and cell volume."""

    batch = _batch()
    raw_descriptor = _descriptor(function, 2, normalization="none")
    normalized_descriptor = _descriptor(function, 2)
    try:
        raw = raw_descriptor.compute(batch).values
        normalized = normalized_descriptor.compute(batch).values
    finally:
        raw_descriptor.close()
        normalized_descriptor.close()

    expected = raw.copy()
    pair_count = 2 * (2 + 1) // 2
    for structure in range(batch.structures):
        begin = int(batch.offsets[structure])
        end = int(batch.offsets[structure + 1])
        counts = [
            int(np.count_nonzero(batch.numbers[begin:end] == species))
            for species in (1, 8)
        ]
        volume = abs(float(np.linalg.det(batch.cells[structure])))
        if function == "distance":
            for first in range(2):
                for second in range(first, 2):
                    count_product = (
                        0.5 * counts[first] * counts[second]
                        if first == second
                        else counts[first] * counts[second]
                    )
                    channel = _pair_channel(first, second, 2)
                    factor = volume / (count_product * 4.0 * np.pi)
                    start = channel * 24
                    expected[structure, start : start + 24] *= factor
        else:
            for first in range(2):
                for center in range(2):
                    for third in range(first, 2):
                        count_product = counts[first] * counts[center] * counts[third]
                        channel = center * pair_count + _pair_channel(first, third, 2)
                        factor = volume / count_product
                        start = channel * 24
                        expected[structure, start : start + 24] *= factor

    np.testing.assert_allclose(normalized, expected, rtol=2e-12, atol=2e-13)


def _median_compute_seconds(
    descriptor: ValleOganov, batch: StructureBatch
) -> tuple[float, np.ndarray]:
    for _ in range(2):
        result = descriptor.compute(batch).values
    elapsed: list[float] = []
    for _ in range(5):
        started = time.perf_counter()
        result = descriptor.compute(batch).values
        elapsed.append(time.perf_counter() - started)
    return float(np.median(elapsed)), result


def test_valleoganov_openmp_small_batch_speed() -> None:
    """Record a repeatable small-batch timing without a machine-specific gate."""

    batch = _batch()
    serial = _descriptor("angle", 1)
    threaded = _descriptor("angle", 2)
    try:
        serial_seconds, serial_values = _median_compute_seconds(serial, batch)
        threaded_seconds, threaded_values = _median_compute_seconds(threaded, batch)
        assert serial_seconds > 0.0
        assert threaded_seconds > 0.0
        assert np.isfinite(serial_values).all()
        assert np.isfinite(threaded_values).all()
        print(
            "ValleOganov angle small batch: "
            f"serial={serial_seconds:.6f}s, "
            f"threads=2={threaded_seconds:.6f}s, "
            f"speedup={serial_seconds / threaded_seconds:.2f}x"
        )
    finally:
        serial.close()
        threaded.close()
