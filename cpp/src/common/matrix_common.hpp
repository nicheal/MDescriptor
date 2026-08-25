#pragma once

#include "extra_common.hpp"

#include <numeric>

namespace mdescriptor::detail {

inline std::vector<double> eigenvalues_symmetric(std::vector<double> matrix, int size) {
    if (size <= 0) {
        return {};
    }

    // Reduce the symmetric matrix to tridiagonal form with Householder
    // reflections. The implicit QL iteration below is O(n^3), matching the
    // algorithmic cost of Eigen::SelfAdjointEigenSolver used by reference implementation.
    std::vector<double> diagonal(static_cast<std::size_t>(size), 0.0);
    std::vector<double> off_diagonal(static_cast<std::size_t>(size), 0.0);
    for (int row = size - 1; row > 0; --row) {
        const int last = row - 1;
        double scale = 0.0;
        for (int column = 0; column <= last; ++column) {
            scale += std::abs(matrix[static_cast<std::size_t>(row * size + column)]);
        }
        if (scale == 0.0) {
            off_diagonal[static_cast<std::size_t>(row)] = matrix[static_cast<std::size_t>(row * size + last)];
            continue;
        }

        double squared_norm = 0.0;
        for (int column = 0; column <= last; ++column) {
            double& value = matrix[static_cast<std::size_t>(row * size + column)];
            value /= scale;
            squared_norm += value * value;
        }
        double first = matrix[static_cast<std::size_t>(row * size + last)];
        double reflector = std::sqrt(squared_norm);
        if (first > 0.0) {
            reflector = -reflector;
        }
        off_diagonal[static_cast<std::size_t>(row)] = scale * reflector;
        squared_norm -= first * reflector;
        matrix[static_cast<std::size_t>(row * size + last)] = first - reflector;

        double projection = 0.0;
        for (int column = 0; column <= last; ++column) {
            double value = 0.0;
            for (int index = 0; index <= column; ++index) {
                value += matrix[static_cast<std::size_t>(column * size + index)]
                    * matrix[static_cast<std::size_t>(row * size + index)];
            }
            for (int index = column + 1; index <= last; ++index) {
                value += matrix[static_cast<std::size_t>(index * size + column)]
                    * matrix[static_cast<std::size_t>(row * size + index)];
            }
            off_diagonal[static_cast<std::size_t>(column)] = value / squared_norm;
            projection += off_diagonal[static_cast<std::size_t>(column)]
                * matrix[static_cast<std::size_t>(row * size + column)];
        }

        const double correction = projection / (squared_norm + squared_norm);
        for (int column = 0; column <= last; ++column) {
            const double row_value = matrix[static_cast<std::size_t>(row * size + column)];
            off_diagonal[static_cast<std::size_t>(column)] -= correction * row_value;
            const double column_value = off_diagonal[static_cast<std::size_t>(column)];
            for (int index = 0; index <= column; ++index) {
                matrix[static_cast<std::size_t>(column * size + index)] -= row_value
                    * off_diagonal[static_cast<std::size_t>(index)]
                    + column_value * matrix[static_cast<std::size_t>(row * size + index)];
            }
        }
    }
    for (int index = 0; index < size; ++index) {
        diagonal[static_cast<std::size_t>(index)] = matrix[static_cast<std::size_t>(index * size + index)];
    }
    off_diagonal[0] = 0.0;
    for (int index = 1; index < size; ++index) {
        off_diagonal[static_cast<std::size_t>(index - 1)] = off_diagonal[static_cast<std::size_t>(index)];
    }
    off_diagonal[static_cast<std::size_t>(size - 1)] = 0.0;

    // Implicit-shift QL iteration for the symmetric tridiagonal matrix.
    for (int lower = 0; lower < size; ++lower) {
        int upper;
        int iteration = 0;
        do {
            for (upper = lower; upper < size - 1; ++upper) {
                const double scale = std::abs(diagonal[static_cast<std::size_t>(upper)])
                    + std::abs(diagonal[static_cast<std::size_t>(upper + 1)]);
                if (std::abs(off_diagonal[static_cast<std::size_t>(upper)]) + scale == scale) {
                    break;
                }
            }
            if (upper == lower) {
                break;
            }
            if (++iteration > 100) {
                throw std::runtime_error("symmetric eigensolver failed to converge");
            }

            double shift = (diagonal[static_cast<std::size_t>(lower + 1)]
                - diagonal[static_cast<std::size_t>(lower)])
                / (2.0 * off_diagonal[static_cast<std::size_t>(lower)]);
            double radius = std::hypot(shift, 1.0);
            shift = diagonal[static_cast<std::size_t>(upper)] - diagonal[static_cast<std::size_t>(lower)]
                + off_diagonal[static_cast<std::size_t>(lower)]
                    / (shift + std::copysign(radius, shift));

            double sine = 1.0;
            double cosine = 1.0;
            double carry = 0.0;
            for (int index = upper - 1; index >= lower; --index) {
                const double first = sine * off_diagonal[static_cast<std::size_t>(index)];
                const double second = cosine * off_diagonal[static_cast<std::size_t>(index)];
                double ratio;
                if (std::abs(first) >= std::abs(shift)) {
                    cosine = shift / first;
                    radius = std::hypot(cosine, 1.0);
                    off_diagonal[static_cast<std::size_t>(index + 1)] = first * radius;
                    sine = 1.0 / radius;
                    cosine *= sine;
                } else {
                    sine = first / shift;
                    radius = std::hypot(sine, 1.0);
                    off_diagonal[static_cast<std::size_t>(index + 1)] = shift * radius;
                    cosine = 1.0 / radius;
                    sine *= cosine;
                }
                const double gap = diagonal[static_cast<std::size_t>(index + 1)] - carry;
                ratio = (diagonal[static_cast<std::size_t>(index)] - gap) * sine
                    + 2.0 * cosine * second;
                carry = sine * ratio;
                diagonal[static_cast<std::size_t>(index + 1)] = gap + carry;
                shift = cosine * ratio - second;
            }
            diagonal[static_cast<std::size_t>(lower)] -= carry;
            off_diagonal[static_cast<std::size_t>(lower)] = shift;
            off_diagonal[static_cast<std::size_t>(upper)] = 0.0;
        } while (upper != lower);
    }

    std::vector<double> result = std::move(diagonal);
    std::sort(result.begin(), result.end(), [](double left, double right) {
        return std::abs(left) > std::abs(right);
    });
    return result;
}

inline void write_matrix(
    std::vector<double> matrix,
    int count,
    std::int64_t n_atoms_max,
    const std::string& permutation,
    double* output) {
    if (count > n_atoms_max) {
        throw std::invalid_argument("structure exceeds n_atoms_max");
    }
    if (permutation == "eigenspectrum") {
        const auto eigenvalues = eigenvalues_symmetric(std::move(matrix), count);
        for (int index = 0; index < count; ++index) {
            output[index] = eigenvalues[static_cast<std::size_t>(index)];
        }
        for (std::int64_t index = count; index < n_atoms_max; ++index) {
            output[index] = 0.0;
        }
        return;
    }
    std::vector<std::size_t> order(static_cast<std::size_t>(count));
    if (permutation == "sorted_l2") {
        order = reference_sorted_l2_order(matrix, static_cast<std::size_t>(count));
    } else if (permutation != "none") {
        throw std::invalid_argument("permutation must be 'none', 'sorted_l2', or 'eigenspectrum'");
    } else {
        std::iota(order.begin(), order.end(), 0);
    }
    for (std::int64_t row_index = 0; row_index < n_atoms_max; ++row_index) {
        for (std::int64_t column = 0; column < n_atoms_max; ++column) {
            output[row_index * n_atoms_max + column] = row_index < count && column < count
                ? matrix[static_cast<std::size_t>(order[static_cast<std::size_t>(row_index)] * count + order[static_cast<std::size_t>(column)])]
                : 0.0;
        }
    }
}

std::vector<double> sine_matrix_values(
    const StructureBatchView& batch,
    std::int64_t structure,
    double exponent);

std::vector<double> ewald_matrix_values(
    const StructureBatchView& batch,
    std::int64_t structure,
    double exponent,
    double accuracy,
    double w,
    double r_cut_option,
    double g_cut_option,
    double a_option);

std::vector<double> coulomb_matrix_values(
    const StructureBatchView& batch,
    std::int64_t structure,
    double exponent);

} // namespace mdescriptor::detail
