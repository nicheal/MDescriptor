#pragma once

#include "extra_common.hpp"

#include <vector>

namespace mdescriptor::detail {

// The dense DeepMD input adapter wraps periodic coordinates into the primary
// cell before constructing the extended image list.  Do the same in the C++
// path so a translated periodic frame follows exactly the same geometry and
// neighbor ordering as the reference implementation.
inline std::vector<double> normalized_positions(const StructureBatchView& batch) {
    std::vector<double> positions(
        static_cast<std::size_t>(batch.atoms) * 3U,
        0.0);
    if (batch.atoms > 0) {
        std::copy(
            batch.positions,
            batch.positions + static_cast<std::size_t>(batch.atoms) * 3U,
            positions.begin());
    }
    for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
        const std::int32_t* pbc = batch.pbc + structure * 3;
        const bool periodic = pbc[0] == 1 && pbc[1] == 1 && pbc[2] == 1;
        if (!periodic) {
            continue;
        }
        const Mat3 cell = load_cell(batch, structure);
        Mat3 inverse;
        const bool diagonal = cell.a[0][1] == 0.0 && cell.a[0][2] == 0.0
            && cell.a[1][0] == 0.0 && cell.a[1][2] == 0.0
            && cell.a[2][0] == 0.0 && cell.a[2][1] == 0.0;
        if (diagonal) {
            for (int axis = 0; axis < 3; ++axis) {
                inverse.a[axis][axis] = 1.0 / cell.a[axis][axis];
            }
        } else {
            inverse = detail::inverse(cell);
        }
        const std::int64_t begin = batch.offsets[structure];
        const std::int64_t end = batch.offsets[structure + 1];
        for (std::int64_t atom = begin; atom < end; ++atom) {
            const double* point = batch.positions + atom * 3;
            const Vec3 fractional{
                point[0] * inverse.a[0][0] + point[1] * inverse.a[1][0]
                    + point[2] * inverse.a[2][0],
                point[0] * inverse.a[0][1] + point[1] * inverse.a[1][1]
                    + point[2] * inverse.a[2][1],
                point[0] * inverse.a[0][2] + point[1] * inverse.a[1][2]
                    + point[2] * inverse.a[2][2],
            };
            const Vec3 wrapped{
                fractional.x - std::floor(fractional.x),
                fractional.y - std::floor(fractional.y),
                fractional.z - std::floor(fractional.z),
            };
            const Vec3 cartesian = wrapped.x * row(cell, 0)
                + wrapped.y * row(cell, 1)
                + wrapped.z * row(cell, 2);
            positions[static_cast<std::size_t>(atom * 3 + 0)] = cartesian.x;
            positions[static_cast<std::size_t>(atom * 3 + 1)] = cartesian.y;
            positions[static_cast<std::size_t>(atom * 3 + 2)] = cartesian.z;
        }
    }
    return positions;
}

} // namespace mdescriptor::detail
