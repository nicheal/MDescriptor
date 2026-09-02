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

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kSqrt2 = 1.414213562373095048801688724209698079;

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

} // namespace mdescriptor::detail
