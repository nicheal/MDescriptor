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

inline double reference_row_l2_norm(
    const std::vector<double>& matrix,
    std::size_t row,
    std::size_t columns) {
    const std::size_t offset = row * columns;
    double squared_norm = 0.0;
    // CoulombMatrix is calculated and sorted by reference implementation's Eigen extension.
    // Keep its four-value packet reduction order for near-tied row norms.
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
    return std::sqrt(squared_norm);
}

inline std::vector<std::size_t> reference_sorted_l2_order(
    const std::vector<double>& matrix,
    std::size_t count) {
    std::vector<std::size_t> order(count);
    std::iota(order.begin(), order.end(), 0);
    std::vector<double> norms(count, 0.0);
    for (std::size_t row = 0; row < count; ++row) {
        norms[row] = reference_row_l2_norm(matrix, row, count);
    }
    std::stable_sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        return norms[left] > norms[right];
    });
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
