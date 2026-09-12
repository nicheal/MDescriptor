#pragma once

#include "mdescriptor/descriptor.hpp"
#include "mdescriptor/detail/math3.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mdescriptor::detail {

inline Mat3 load_cell(const StructureBatchView& batch, std::int64_t structure) {
    Mat3 result;
    const double* source = batch.cells + structure * 9;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result.a[row][column] = source[row * 3 + column];
        }
    }
    return result;
}

inline Vec3 position(const StructureBatchView& batch, std::int64_t atom) {
    const double* value = batch.positions + atom * 3;
    return {value[0], value[1], value[2]};
}

// View of one structure of a batch, with atom ranges rebased to zero. The
// two-element `offsets` buffer is caller-owned and must outlive the view.
inline StructureBatchView structure_view(
    const StructureBatchView& batch, std::int64_t structure, std::int64_t* offsets) {
    const std::int64_t begin = batch.offsets[structure];
    const std::int64_t end = batch.offsets[structure + 1];
    offsets[0] = 0;
    offsets[1] = end - begin;
    return StructureBatchView{
        batch.numbers + begin,
        batch.positions + begin * 3,
        batch.cells + structure * 9,
        batch.pbc + structure * 3,
        offsets,
        1,
        end - begin,
    };
}

} // namespace mdescriptor::detail
