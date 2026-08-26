#pragma once

#include "mdescriptor/extra.hpp"
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
// Row norms that differ only by a few floating-point reduction ulps are the
// same sorted_l2 key for this project. This fixed tolerance defines the
// contiguous tie buckets used below.
constexpr double kSortedL2TieUlpFactor = 4.0;

inline double reference_row_l2_norm_squared(
    const std::vector<double>& matrix,
    std::size_t row,
    std::size_t columns) {
    const std::size_t offset = row * columns;
    double squared_norm = 0.0;
    // Keep the four-value packet reduction order used by the reference matrix
    // implementation for near-tied row norms.
    const std::size_t grouped_end = columns & ~std::size_t(3);
    for (std::size_t column = 0; column < grouped_end; column += 4) {
        squared_norm += matrix[offset + column] * matrix[offset + column]
            + matrix[offset + column + 1] * matrix[offset + column + 1]
            + matrix[offset + column + 2] * matrix[offset + column + 2]
            + matrix[offset + column + 3] * matrix[offset + column + 3];
    }
    for (std::size_t column = grouped_end; column < columns; ++column) {
        squared_norm += matrix[offset + column] * matrix[offset + column];
    }
    return squared_norm;
}

inline std::vector<std::size_t> reference_sorted_l2_order(
    const std::vector<double>& matrix,
    std::size_t count) {
    std::vector<std::size_t> order(count);
    std::iota(order.begin(), order.end(), 0);
    std::vector<double> norm_squared(count, 0.0);
    double maximum_norm_squared = 1.0;
    for (std::size_t row = 0; row < count; ++row) {
        norm_squared[row] = reference_row_l2_norm_squared(matrix, row, count);
        maximum_norm_squared = std::max(maximum_norm_squared, norm_squared[row]);
    }
    const double tie_tolerance = kSortedL2TieUlpFactor
        * std::numeric_limits<double>::epsilon() * maximum_norm_squared;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        return norm_squared[left] > norm_squared[right];
    });
    for (std::size_t group_begin = 0; group_begin < count;) {
        std::size_t group_end = group_begin + 1;
        while (group_end < count
            && norm_squared[order[group_end - 1]] - norm_squared[order[group_end]]
                <= tie_tolerance) {
            ++group_end;
        }
        if (group_end - group_begin > 1) {
            std::sort(order.begin() + group_begin, order.begin() + group_end,
                [](std::size_t left, std::size_t right) {
                    // The input atom index is an explicit, deterministic
                    // final key. It is intentionally not presented as a new
                    // permutation-invariant canonicalization.
                    return left < right;
                });
        }
        group_begin = group_end;
    }
    return order;
}

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
