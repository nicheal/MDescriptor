#pragma once
// Private CUDA support for the matrix descriptor family.
#include "mdescriptor/matrix.hpp"
#include "mdescriptor/detail/math3.hpp"
#include "extended_descriptors_common.cuh"

namespace {

constexpr int kMatrixKindSine = static_cast<int>(::mdescriptor::MatrixKind::Sine);
constexpr int kMatrixKindEwald = static_cast<int>(::mdescriptor::MatrixKind::Ewald);
constexpr int kMatrixKindCoulomb = static_cast<int>(::mdescriptor::MatrixKind::Coulomb);
constexpr int kMatrixPermutationNone = static_cast<int>(::mdescriptor::MatrixPermutation::None);
constexpr int kMatrixPermutationSortedL2 = static_cast<int>(::mdescriptor::MatrixPermutation::SortedL2);
constexpr int kMatrixPermutationEigenspectrum = static_cast<int>(::mdescriptor::MatrixPermutation::Eigenspectrum);

__device__ bool inverse3_device(const double* matrix, double* inverse) {
    const double determinant = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
        - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
        + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
    if (!isfinite(determinant) || fabs(determinant) <= 1e-12) return false;
    const double scale = 1.0 / determinant;
    inverse[0] = (matrix[4] * matrix[8] - matrix[5] * matrix[7]) * scale;
    inverse[1] = (matrix[2] * matrix[7] - matrix[1] * matrix[8]) * scale;
    inverse[2] = (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * scale;
    inverse[3] = (matrix[5] * matrix[6] - matrix[3] * matrix[8]) * scale;
    inverse[4] = (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * scale;
    inverse[5] = (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * scale;
    inverse[6] = (matrix[3] * matrix[7] - matrix[4] * matrix[6]) * scale;
    inverse[7] = (matrix[1] * matrix[6] - matrix[0] * matrix[7]) * scale;
    inverse[8] = (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * scale;
    return true;
}

__device__ void fractional_device(
    const double* inverse, double x, double y, double z,
    double& fx, double& fy, double& fz) {
    fx = x * inverse[0] + y * inverse[3] + z * inverse[6];
    fy = x * inverse[1] + y * inverse[4] + z * inverse[7];
    fz = x * inverse[2] + y * inverse[5] + z * inverse[8];
}

__device__ void cartesian_from_fractional(
    const double* cell, double fx, double fy, double fz,
    double& x, double& y, double& z) {
    x = fx * cell[0] + fy * cell[3] + fz * cell[6];
    y = fx * cell[1] + fy * cell[4] + fz * cell[7];
    z = fx * cell[2] + fy * cell[5] + fz * cell[8];
}

__device__ double sine_matrix_off_diagonal(
    const double* cell, const double* inverse,
    double dx, double dy, double dz) {
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    fractional_device(inverse, dx, dy, dz, fx, fy, fz);
    const double sx = sin(kPi * fx);
    const double sy = sin(kPi * fy);
    const double sz = sin(kPi * fz);
    const double tx = sx * sx * cell[0] + sy * sy * cell[3] + sz * sz * cell[6];
    const double ty = sx * sx * cell[1] + sy * sy * cell[4] + sz * sz * cell[7];
    const double tz = sx * sx * cell[2] + sy * sy * cell[5] + sz * sz * cell[8];
    return sqrt(tx * tx + ty * ty + tz * tz);
}

__device__ void eigenvalues_symmetric_device(
    double* matrix,
    int size,
    int stride,
    double* output,
    int output_size) {
    if (size <= 0) {
        for (int index = 0; index < output_size; ++index) output[index] = 0.0;
        return;
    }

    // Householder reduction followed by implicit-shift QL is O(n^3).  The
    // former CUDA path used a maximum-pivot Jacobi loop whose repeated O(n^2)
    // pivot scans made the practical cost approach O(n^4) for eigenspectra.
    // Keep the matrix in the existing n_atoms_max-strided workspace so this
    // change does not add a second matrix allocation or alter the output ABI.
    double diagonal[256]{};
    double off_diagonal[256]{};
    for (int row = size - 1; row > 0; --row) {
        const int last = row - 1;
        double scale = 0.0;
        for (int column = 0; column <= last; ++column) {
            scale += fabs(matrix[row * stride + column]);
        }
        if (scale == 0.0) {
            off_diagonal[row] = matrix[row * stride + last];
            continue;
        }

        double squared_norm = 0.0;
        for (int column = 0; column <= last; ++column) {
            double& value = matrix[row * stride + column];
            value /= scale;
            squared_norm += value * value;
        }
        const double first = matrix[row * stride + last];
        double reflector = sqrt(squared_norm);
        if (first > 0.0) reflector = -reflector;
        off_diagonal[row] = scale * reflector;
        squared_norm -= first * reflector;
        matrix[row * stride + last] = first - reflector;

        double projection = 0.0;
        for (int column = 0; column <= last; ++column) {
            double value = 0.0;
            for (int index = 0; index <= column; ++index) {
                value += matrix[column * stride + index]
                    * matrix[row * stride + index];
            }
            for (int index = column + 1; index <= last; ++index) {
                value += matrix[index * stride + column]
                    * matrix[row * stride + index];
            }
            off_diagonal[column] = value / squared_norm;
            projection += off_diagonal[column] * matrix[row * stride + column];
        }

        const double correction = projection / (squared_norm + squared_norm);
        for (int column = 0; column <= last; ++column) {
            const double row_value = matrix[row * stride + column];
            off_diagonal[column] -= correction * row_value;
            const double column_value = off_diagonal[column];
            for (int index = 0; index <= column; ++index) {
                matrix[column * stride + index] -= row_value
                    * off_diagonal[index]
                    + column_value * matrix[row * stride + index];
            }
        }
    }
    for (int index = 0; index < size; ++index) {
        diagonal[index] = matrix[index * stride + index];
    }
    off_diagonal[0] = 0.0;
    for (int index = 1; index < size; ++index) {
        off_diagonal[index - 1] = off_diagonal[index];
    }
    off_diagonal[size - 1] = 0.0;

    for (int lower = 0; lower < size; ++lower) {
        int upper;
        int iteration = 0;
        do {
            for (upper = lower; upper < size - 1; ++upper) {
                const double scale = fabs(diagonal[upper]) + fabs(diagonal[upper + 1]);
                if (fabs(off_diagonal[upper]) + scale == scale) break;
            }
            if (upper == lower) break;
            if (++iteration > 100) break;

            double shift = (diagonal[lower + 1] - diagonal[lower])
                / (2.0 * off_diagonal[lower]);
            double radius = hypot(shift, 1.0);
            shift = diagonal[upper] - diagonal[lower]
                + off_diagonal[lower]
                    / (shift + copysign(radius, shift));

            double sine = 1.0;
            double cosine = 1.0;
            double carry = 0.0;
            for (int index = upper - 1; index >= lower; --index) {
                const double first = sine * off_diagonal[index];
                const double second = cosine * off_diagonal[index];
                double ratio;
                if (fabs(first) >= fabs(shift)) {
                    cosine = shift / first;
                    radius = hypot(cosine, 1.0);
                    off_diagonal[index + 1] = first * radius;
                    sine = 1.0 / radius;
                    cosine *= sine;
                } else {
                    sine = first / shift;
                    radius = hypot(sine, 1.0);
                    off_diagonal[index + 1] = shift * radius;
                    cosine = 1.0 / radius;
                    sine *= cosine;
                }
                const double gap = diagonal[index + 1] - carry;
                ratio = (diagonal[index] - gap) * sine + 2.0 * cosine * second;
                carry = sine * ratio;
                diagonal[index + 1] = gap + carry;
                shift = cosine * ratio - second;
            }
            diagonal[lower] -= carry;
            off_diagonal[lower] = shift;
            off_diagonal[upper] = 0.0;
        } while (upper != lower);
    }

    for (int index = 0; index < size; ++index) output[index] = diagonal[index];
    for (int index = 1; index < size; ++index) {
        int current = index;
        while (current > 0 && fabs(output[current]) > fabs(output[current - 1])) {
            const double saved = output[current];
            output[current] = output[current - 1];
            output[current - 1] = saved;
            --current;
        }
    }
    for (int index = size; index < output_size; ++index) output[index] = 0.0;
}

// Pair-parallel fill for the Coulomb and sine matrix families: one thread
// per (structure, i, j) element.  Every element is independent and written
// exactly once by one thread, so the fill is bit-identical to the
// per-structure serial form while spreading N*N work across the GPU.
__global__ void matrix_fill_pairs_kernel(
    const I32* numbers,
    const double* positions,
    const double* cells,
    const I64* offsets,
    I64 structures,
    int n_atoms_max,
    int kind,
    double exponent,
    double* matrices) {
    const I64 index = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    const I64 square = static_cast<I64>(n_atoms_max) * n_atoms_max;
    if (index >= structures * square) return;
    const I64 structure = index / square;
    const I64 within = index - structure * square;
    const int i = static_cast<int>(within / n_atoms_max);
    const int j = static_cast<int>(within - static_cast<I64>(i) * n_atoms_max);
    double* matrix = matrices + structure * square;
    const I64 begin = offsets[structure];
    const I64 end = offsets[structure + 1];
    const int count = static_cast<int>(end - begin);
    if (i >= count || j >= count) {
        // Rows past count are never read by the permutation tail.
        return;
    }
    const double zi = static_cast<double>(numbers[begin + i]);
    const double zj = static_cast<double>(numbers[begin + j]);
    double value = 0.0;
    if (i == j) {
        value = 0.5 * pow(zi, exponent);
    } else if (kind == kMatrixKindCoulomb) {
        const double dx = positions[(begin + i) * 3 + 0] - positions[(begin + j) * 3 + 0];
        const double dy = positions[(begin + i) * 3 + 1] - positions[(begin + j) * 3 + 1];
        const double dz = positions[(begin + i) * 3 + 2] - positions[(begin + j) * 3 + 2];
        value = zi * zj / sqrt(dx * dx + dy * dy + dz * dz);
    } else {
        const double* cell = cells + structure * 9;
        double inverse[9]{};
        const bool inverse_valid = inverse3_device(cell, inverse);
        if (inverse_valid) {
            const double dx = positions[(begin + i) * 3 + 0] - positions[(begin + j) * 3 + 0];
            const double dy = positions[(begin + i) * 3 + 1] - positions[(begin + j) * 3 + 1];
            const double dz = positions[(begin + i) * 3 + 2] - positions[(begin + j) * 3 + 2];
            const double denominator = sine_matrix_off_diagonal(
                cell, inverse, dx, dy, dz);
            value = denominator > 1e-14 ? zi * zj / denominator : 0.0;
        }
    }
    matrix[i * n_atoms_max + j] = value;
}

// Per-structure post-processing (permutation / eigenspectrum) over the
// filled raw matrix; one thread per structure, identical to the serial tail.
__global__ void matrix_post_kernel(
    I64 structures,
    int n_atoms_max,
    int permutation,
    int kind,
    const I64* offsets,
    double* matrices,
    double* output) {
    const I64 structure = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (structure >= structures) return;
    const I64 begin = offsets[structure];
    const int count = static_cast<int>(offsets[structure + 1] - begin);
    double* matrix = matrices + structure
        * (static_cast<I64>(n_atoms_max) * n_atoms_max
            + (kind == kMatrixKindEwald ? static_cast<std::size_t>(3 * n_atoms_max) : 0));
    double* row = output + structure * (permutation == kMatrixPermutationEigenspectrum
        ? n_atoms_max : n_atoms_max * n_atoms_max);
    if (count <= 0) {
        const I64 columns = permutation == kMatrixPermutationEigenspectrum
            ? n_atoms_max : static_cast<I64>(n_atoms_max) * n_atoms_max;
        for (I64 index = 0; index < columns; ++index) row[index] = 0.0;
        return;
    }
    if (permutation == kMatrixPermutationEigenspectrum) {
        eigenvalues_symmetric_device(matrix, count, n_atoms_max, row, n_atoms_max);
        return;
    }
    if (permutation == kMatrixPermutationNone) {
        for (int i = 0; i < count; ++i) {
            for (int j = 0; j < count; ++j) {
                row[i * n_atoms_max + j] = matrix[i * n_atoms_max + j];
            }
            for (int j = count; j < n_atoms_max; ++j) row[i * n_atoms_max + j] = 0.0;
        }
        for (int i = count; i < n_atoms_max; ++i) {
            for (int j = 0; j < n_atoms_max; ++j) row[i * n_atoms_max + j] = 0.0;
        }
        return;
    }
    int order[256];
    double norms[256];
    double maximum_norm_squared = 1.0;
    for (int i = 0; i < count; ++i) {
        order[i] = i;
        double norm2 = 0.0;
        const int grouped_end = count & ~3;
        for (int j = 0; j < grouped_end; j += 4) {
            norm2 += matrix[i * n_atoms_max + j] * matrix[i * n_atoms_max + j]
                + matrix[i * n_atoms_max + j + 1] * matrix[i * n_atoms_max + j + 1]
                + matrix[i * n_atoms_max + j + 2] * matrix[i * n_atoms_max + j + 2]
                + matrix[i * n_atoms_max + j + 3] * matrix[i * n_atoms_max + j + 3];
        }
        for (int j = grouped_end; j < count; ++j) {
            norm2 += matrix[i * n_atoms_max + j] * matrix[i * n_atoms_max + j];
        }
        norms[i] = norm2;
        maximum_norm_squared = max(maximum_norm_squared, norm2);
    }
    for (int i = 1; i < count; ++i) {
        int current = i;
        while (current > 0 && norms[current] > norms[current - 1]) {
            const double norm_saved = norms[current]; norms[current] = norms[current - 1]; norms[current - 1] = norm_saved;
            const int index_saved = order[current]; order[current] = order[current - 1]; order[current - 1] = index_saved;
            --current;
        }
    }
    const double tie_tolerance = 4.0 * DBL_EPSILON * maximum_norm_squared;
    for (int group_begin = 0; group_begin < count;) {
        int group_end = group_begin + 1;
        while (group_end < count
            && norms[group_end - 1] - norms[group_end] <= tie_tolerance) {
            ++group_end;
        }
        for (int i = group_begin + 1; i < group_end; ++i) {
            int current = i;
            while (current > group_begin && order[current] < order[current - 1]) {
                const int index_saved = order[current];
                order[current] = order[current - 1];
                order[current - 1] = index_saved;
                --current;
            }
        }
        group_begin = group_end;
    }
    for (int i = 0; i < n_atoms_max; ++i) {
        for (int j = 0; j < n_atoms_max; ++j) {
            row[i * n_atoms_max + j] = i < count && j < count
                ? matrix[order[i] * n_atoms_max + order[j]] : 0.0;
        }
    }
}

} // namespace
