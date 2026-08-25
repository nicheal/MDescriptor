#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace mdescriptor::detail {

struct StructureBatchView {
    const std::int32_t* numbers = nullptr;
    const double* positions = nullptr;
    const double* cells = nullptr;
    const std::int32_t* pbc = nullptr;
    const std::int64_t* offsets = nullptr;
    std::int64_t structures = 0;
    std::int64_t atoms = 0;
};

inline void validate_batch(const StructureBatchView& batch) {
    if (batch.structures < 0 || batch.atoms < 0 || batch.offsets == nullptr) {
        throw std::invalid_argument("invalid structure batch");
    }
    if (batch.structures == 0) {
        if (batch.atoms != 0 || batch.offsets[0] != 0) {
            throw std::invalid_argument("empty structure batch has non-empty atom data");
        }
        return;
    }
    if (batch.numbers == nullptr || batch.positions == nullptr || batch.cells == nullptr
        || batch.pbc == nullptr) {
        throw std::invalid_argument("structure batch contains null arrays");
    }
    if (batch.offsets[0] != 0 || batch.offsets[batch.structures] != batch.atoms) {
        throw std::invalid_argument("offsets do not describe the flattened arrays");
    }
    for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
        if (batch.offsets[structure] < 0
            || batch.offsets[structure] > batch.offsets[structure + 1]
            || batch.offsets[structure + 1] > batch.atoms) {
            throw std::invalid_argument("offsets must be monotonic");
        }
        bool periodic = true;
        bool isolated = true;
        for (int axis = 0; axis < 3; ++axis) {
            const std::int32_t flag = batch.pbc[structure * 3 + axis];
            if (flag != 0 && flag != 1) {
                throw std::invalid_argument("pbc must contain only 0 or 1");
            }
            periodic = periodic && flag == 1;
            isolated = isolated && flag == 0;
        }
        if (!periodic && !isolated) {
            throw std::invalid_argument(
                "mixed periodicity is not supported; use all-zero or all-one pbc");
        }
        const double* cell = batch.cells + structure * 9;
        for (int index = 0; index < 9; ++index) {
            if (!std::isfinite(cell[index])) {
                throw std::invalid_argument("cells must be finite");
            }
        }
        const double determinant =
            cell[0] * (cell[4] * cell[8] - cell[5] * cell[7])
            - cell[1] * (cell[3] * cell[8] - cell[5] * cell[6])
            + cell[2] * (cell[3] * cell[7] - cell[4] * cell[6]);
        if (periodic && (!std::isfinite(determinant) || std::abs(determinant) < 1e-14)) {
            throw std::invalid_argument("cells must be nonsingular");
        }
    }
    for (std::int64_t atom = 0; atom < batch.atoms; ++atom) {
        if (batch.numbers[atom] <= 0) {
            throw std::invalid_argument("atomic numbers must be positive");
        }
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(batch.positions[atom * 3 + axis])) {
                throw std::invalid_argument("positions must be finite");
            }
        }
    }
}

} // namespace mdescriptor::detail
