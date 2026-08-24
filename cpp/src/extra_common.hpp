#pragma once

#include "mdescriptor/extra.hpp"

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

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(double scale, Vec3 value) { return {scale * value.x, scale * value.y, scale * value.z}; }
inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline double norm2(Vec3 value) { return dot(value, value); }
inline double norm(Vec3 value) { return std::sqrt(norm2(value)); }

inline double dscribe_row_l2_norm(
    const std::vector<double>& matrix,
    std::size_t row,
    std::size_t columns) {
    const std::size_t offset = row * columns;
    double squared_norm = 0.0;
    // CoulombMatrix is calculated and sorted by DScribe's Eigen extension.
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

inline std::vector<std::size_t> dscribe_sorted_l2_order(
    const std::vector<double>& matrix,
    std::size_t count) {
    std::vector<std::size_t> order(count);
    std::iota(order.begin(), order.end(), 0);
    std::vector<double> norms(count, 0.0);
    for (std::size_t row = 0; row < count; ++row) {
        norms[row] = dscribe_row_l2_norm(matrix, row, count);
    }
    std::stable_sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        return norms[left] > norms[right];
    });
    return order;
}

struct Mat3 {
    double a[3][3]{};
};

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

inline Vec3 row(const Mat3& matrix, int index) {
    return {matrix.a[index][0], matrix.a[index][1], matrix.a[index][2]};
}

inline double determinant(const Mat3& matrix) {
    return matrix.a[0][0] * (matrix.a[1][1] * matrix.a[2][2] - matrix.a[1][2] * matrix.a[2][1])
        - matrix.a[0][1] * (matrix.a[1][0] * matrix.a[2][2] - matrix.a[1][2] * matrix.a[2][0])
        + matrix.a[0][2] * (matrix.a[1][0] * matrix.a[2][1] - matrix.a[1][1] * matrix.a[2][0]);
}

inline Mat3 inverse(const Mat3& matrix) {
    const double det = determinant(matrix);
    if (!std::isfinite(det) || std::abs(det) < 1e-14) {
        throw std::invalid_argument("cell matrix is singular");
    }
    Mat3 result;
    result.a[0][0] = (matrix.a[1][1] * matrix.a[2][2] - matrix.a[1][2] * matrix.a[2][1]) / det;
    result.a[0][1] = (matrix.a[0][2] * matrix.a[2][1] - matrix.a[0][1] * matrix.a[2][2]) / det;
    result.a[0][2] = (matrix.a[0][1] * matrix.a[1][2] - matrix.a[0][2] * matrix.a[1][1]) / det;
    result.a[1][0] = (matrix.a[1][2] * matrix.a[2][0] - matrix.a[1][0] * matrix.a[2][2]) / det;
    result.a[1][1] = (matrix.a[0][0] * matrix.a[2][2] - matrix.a[0][2] * matrix.a[2][0]) / det;
    result.a[1][2] = (matrix.a[0][2] * matrix.a[1][0] - matrix.a[0][0] * matrix.a[1][2]) / det;
    result.a[2][0] = (matrix.a[1][0] * matrix.a[2][1] - matrix.a[1][1] * matrix.a[2][0]) / det;
    result.a[2][1] = (matrix.a[0][1] * matrix.a[2][0] - matrix.a[0][0] * matrix.a[2][1]) / det;
    result.a[2][2] = (matrix.a[0][0] * matrix.a[1][1] - matrix.a[0][1] * matrix.a[1][0]) / det;
    return result;
}

inline Vec3 position(const StructureBatchView& batch, std::int64_t atom) {
    const double* value = batch.positions + atom * 3;
    return {value[0], value[1], value[2]};
}

inline void validate_batch(const StructureBatchView& batch) {
    if (batch.structures < 0 || batch.atoms < 0 || batch.offsets == nullptr) {
        throw std::invalid_argument("invalid structure batch");
    }
    if (batch.structures == 0) {
        return;
    }
    if (batch.numbers == nullptr || batch.positions == nullptr || batch.cells == nullptr || batch.pbc == nullptr) {
        throw std::invalid_argument("structure batch contains null arrays");
    }
    if (batch.offsets[0] != 0 || batch.offsets[batch.structures] != batch.atoms) {
        throw std::invalid_argument("offsets do not describe the flattened arrays");
    }
    for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
        if (batch.offsets[structure] > batch.offsets[structure + 1]) {
            throw std::invalid_argument("offsets must be monotonic");
        }
        for (int axis = 0; axis < 3; ++axis) {
            if (batch.pbc[structure * 3 + axis] != 1) {
                throw std::invalid_argument("only fully periodic structures are supported");
            }
        }
        (void)inverse(load_cell(batch, structure));
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

using TypeMap = std::unordered_map<std::int32_t, std::size_t>;

inline TypeMap type_map(const std::vector<std::int32_t>& species) {
    if (species.empty()) {
        throw std::invalid_argument("species must not be empty");
    }
    TypeMap result;
    for (std::size_t index = 0; index < species.size(); ++index) {
        if (species[index] <= 0 || !result.emplace(species[index], index).second) {
            throw std::invalid_argument("species must contain unique positive atomic numbers");
        }
    }
    return result;
}

inline void validate_species(const StructureBatchView& batch, const std::vector<std::int32_t>& species) {
    const auto mapping = type_map(species);
    for (std::int64_t atom = 0; atom < batch.atoms; ++atom) {
        if (!mapping.count(batch.numbers[atom])) {
            throw std::invalid_argument("batch contains an atomic number outside calculator species");
        }
    }
}

inline void check_cancelled(const std::shared_ptr<ComputeControl>& control) {
    if (control && control->cancelled()) {
        throw CancelledError();
    }
}

inline void mark_completed(const std::shared_ptr<ComputeControl>& control) {
    if (control) {
        control->mark_completed();
    }
}

} // namespace mdescriptor::detail
