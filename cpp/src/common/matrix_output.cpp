#include "matrix_output.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace mdescriptor::detail {
namespace {

constexpr double kSortedL2TieUlpFactor = 4.0;

double reference_row_l2_norm_squared(
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

std::vector<std::size_t> reference_sorted_l2_order(
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

std::vector<double> eigenvalues_symmetric(std::vector<double> matrix, int size) {
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

} // namespace

MatrixPermutation matrix_permutation_from_name(const std::string& name) {
    if (name == "none") {
        return MatrixPermutation::None;
    }
    if (name == "sorted_l2") {
        return MatrixPermutation::SortedL2;
    }
    if (name == "eigenspectrum") {
        return MatrixPermutation::Eigenspectrum;
    }
    throw std::invalid_argument("permutation must be 'none', 'sorted_l2', or 'eigenspectrum'");
}

MatrixLayout make_matrix_layout(std::int64_t n_atoms_max, MatrixPermutation permutation) {
    if (permutation != MatrixPermutation::None
        && permutation != MatrixPermutation::SortedL2
        && permutation != MatrixPermutation::Eigenspectrum) {
        throw std::invalid_argument("permutation must be 'none', 'sorted_l2', or 'eigenspectrum'");
    }
    if (n_atoms_max <= 0
        || (permutation != MatrixPermutation::Eigenspectrum
            && n_atoms_max > std::numeric_limits<std::int64_t>::max() / n_atoms_max)) {
        throw std::invalid_argument("n_atoms_max must be a positive value");
    }
    return {n_atoms_max, permutation};
}

MatrixLayout make_matrix_layout(std::int64_t n_atoms_max, const std::string& permutation) {
    return make_matrix_layout(n_atoms_max, matrix_permutation_from_name(permutation));
}

void write_matrix(
    std::vector<double> matrix,
    int count,
    const MatrixLayout& layout,
    double* output) {
    if (count > layout.n_atoms_max) {
        throw std::invalid_argument("structure exceeds n_atoms_max");
    }
    if (layout.permutation == MatrixPermutation::Eigenspectrum) {
        const auto eigenvalues = eigenvalues_symmetric(std::move(matrix), count);
        for (int index = 0; index < count; ++index) {
            output[index] = eigenvalues[static_cast<std::size_t>(index)];
        }
        for (std::int64_t index = count; index < layout.n_atoms_max; ++index) {
            output[index] = 0.0;
        }
        return;
    }
    std::vector<std::size_t> order(static_cast<std::size_t>(count));
    if (layout.permutation == MatrixPermutation::SortedL2) {
        order = reference_sorted_l2_order(matrix, static_cast<std::size_t>(count));
    } else {
        std::iota(order.begin(), order.end(), 0);
    }
    for (std::int64_t row_index = 0; row_index < layout.n_atoms_max; ++row_index) {
        for (std::int64_t column = 0; column < layout.n_atoms_max; ++column) {
            output[row_index * layout.n_atoms_max + column] = row_index < count && column < count
                ? matrix[static_cast<std::size_t>(order[static_cast<std::size_t>(row_index)] * count
                    + order[static_cast<std::size_t>(column)])]
                : 0.0;
        }
    }
}

} // namespace mdescriptor::detail
