#include "matrix_common.hpp"

namespace mdescriptor::detail {

std::vector<double> ewald_matrix_values(
    const StructureBatchView& batch,
    std::int64_t structure,
    double exponent,
    double accuracy,
    double w,
    double r_cut_option,
    double g_cut_option,
    double a_option) {
    const std::int64_t begin = batch.offsets[structure];
    const std::int64_t end = batch.offsets[structure + 1];
    const int count = static_cast<int>(end - begin);
    const Mat3 cell = load_cell(batch, structure);
    const Mat3 inverse_cell = inverse(cell);
    const double volume = std::abs(determinant(cell));
    const double alpha = a_option > 0.0
        ? a_option
        : std::pow(static_cast<double>(count) * w / (volume * volume), 1.0 / 6.0) * std::sqrt(kPi);
    double r_cut = r_cut_option;
    double g_cut = g_cut_option;
    if (r_cut <= 0.0 && g_cut <= 0.0) {
        const double factor = std::sqrt(-std::log(accuracy));
        r_cut = factor / alpha;
        g_cut = 2.0 * alpha * factor;
    }
    if (r_cut <= 0.0 || g_cut <= 0.0) {
        throw std::invalid_argument("r_cut and g_cut must be provided together");
    }
    // Match DScribe's Lattice.get_points_in_sphere: image bounds depend on
    // the center atom and the upper integer bound is exclusive. This avoids
    // double-counting images exactly on the real-space cutoff sphere.
    const double inverse_norms[3] = {
        std::sqrt(inverse_cell.a[0][0] * inverse_cell.a[0][0] + inverse_cell.a[1][0] * inverse_cell.a[1][0] + inverse_cell.a[2][0] * inverse_cell.a[2][0]),
        std::sqrt(inverse_cell.a[0][1] * inverse_cell.a[0][1] + inverse_cell.a[1][1] * inverse_cell.a[1][1] + inverse_cell.a[2][1] * inverse_cell.a[2][1]),
        std::sqrt(inverse_cell.a[0][2] * inverse_cell.a[0][2] + inverse_cell.a[1][2] * inverse_cell.a[1][2] + inverse_cell.a[2][2] * inverse_cell.a[2][2]),
    };
    const double real_nmax[3] = {
        r_cut * inverse_norms[0] + 0.01,
        r_cut * inverse_norms[1] + 0.01,
        r_cut * inverse_norms[2] + 0.01,
    };
    std::vector<Vec3> positions(static_cast<std::size_t>(count));
    std::vector<double> charges(static_cast<std::size_t>(count));
    std::vector<Vec3> wrapped_positions(static_cast<std::size_t>(count));
    for (int atom = 0; atom < count; ++atom) {
        const Vec3 cartesian = position(batch, begin + atom);
        positions[static_cast<std::size_t>(atom)] = cartesian;
        charges[static_cast<std::size_t>(atom)] = static_cast<double>(batch.numbers[begin + atom]);
        Vec3 fractional{
            cartesian.x * inverse_cell.a[0][0] + cartesian.y * inverse_cell.a[1][0] + cartesian.z * inverse_cell.a[2][0],
            cartesian.x * inverse_cell.a[0][1] + cartesian.y * inverse_cell.a[1][1] + cartesian.z * inverse_cell.a[2][1],
            cartesian.x * inverse_cell.a[0][2] + cartesian.y * inverse_cell.a[1][2] + cartesian.z * inverse_cell.a[2][2],
        };
        fractional.x -= std::floor(fractional.x);
        fractional.y -= std::floor(fractional.y);
        fractional.z -= std::floor(fractional.z);
        wrapped_positions[static_cast<std::size_t>(atom)] = fractional.x * row(cell, 0)
            + fractional.y * row(cell, 1) + fractional.z * row(cell, 2);
    }
    const Mat3 reciprocal = {
        {{2.0 * kPi * inverse_cell.a[0][0], 2.0 * kPi * inverse_cell.a[1][0], 2.0 * kPi * inverse_cell.a[2][0]},
         {2.0 * kPi * inverse_cell.a[0][1], 2.0 * kPi * inverse_cell.a[1][1], 2.0 * kPi * inverse_cell.a[2][1]},
         {2.0 * kPi * inverse_cell.a[0][2], 2.0 * kPi * inverse_cell.a[1][2], 2.0 * kPi * inverse_cell.a[2][2]}}
    };
    const int g_bounds[3] = {
        static_cast<int>(std::ceil(g_cut / norm(row(reciprocal, 0)))) + 1,
        static_cast<int>(std::ceil(g_cut / norm(row(reciprocal, 1)))) + 1,
        static_cast<int>(std::ceil(g_cut / norm(row(reciprocal, 2)))) + 1,
    };
    std::vector<Vec3> g_vectors;
    std::vector<double> g_factors;
    for (int i = -g_bounds[0]; i <= g_bounds[0]; ++i) {
        for (int j = -g_bounds[1]; j <= g_bounds[1]; ++j) {
            for (int k = -g_bounds[2]; k <= g_bounds[2]; ++k) {
                const Vec3 vector = i * row(reciprocal, 0) + j * row(reciprocal, 1) + k * row(reciprocal, 2);
                const double length2 = norm2(vector);
                if (length2 > 1e-24 && length2 <= g_cut * g_cut) {
                    g_vectors.push_back(vector);
                    g_factors.push_back(std::exp(-length2 / (4.0 * alpha * alpha)) / length2);
                }
            }
        }
    }
    std::vector<double> real_matrix(static_cast<std::size_t>(count * count), 0.0);
    std::vector<Vec3> shifts;
    for (int center = 0; center < count; ++center) {
        const Vec3 center_position = positions[static_cast<std::size_t>(center)];
        const Vec3 center_fractional{
            center_position.x * inverse_cell.a[0][0] + center_position.y * inverse_cell.a[1][0] + center_position.z * inverse_cell.a[2][0],
            center_position.x * inverse_cell.a[0][1] + center_position.y * inverse_cell.a[1][1] + center_position.z * inverse_cell.a[2][1],
            center_position.x * inverse_cell.a[0][2] + center_position.y * inverse_cell.a[1][2] + center_position.z * inverse_cell.a[2][2],
        };
        const int minima[3] = {
            static_cast<int>(std::floor(center_fractional.x - real_nmax[0])),
            static_cast<int>(std::floor(center_fractional.y - real_nmax[1])),
            static_cast<int>(std::floor(center_fractional.z - real_nmax[2])),
        };
        const int maxima[3] = {
            static_cast<int>(std::ceil(center_fractional.x + real_nmax[0])),
            static_cast<int>(std::ceil(center_fractional.y + real_nmax[1])),
            static_cast<int>(std::ceil(center_fractional.z + real_nmax[2])),
        };
        shifts.clear();
        for (int i = minima[0]; i < maxima[0]; ++i) {
            for (int j = minima[1]; j < maxima[1]; ++j) {
                for (int k = minima[2]; k < maxima[2]; ++k) {
                    shifts.push_back(i * row(cell, 0) + j * row(cell, 1) + k * row(cell, 2));
                }
            }
        }
        for (int target = 0; target < count; ++target) {
            double real = 0.0;
            for (const Vec3 shift : shifts) {
                const Vec3 displacement = wrapped_positions[static_cast<std::size_t>(target)]
                    - center_position + shift;
                const double distance2 = norm2(displacement);
                if (distance2 > 1e-16 && distance2 <= r_cut * r_cut) {
                    const double distance = std::sqrt(distance2);
                    real += std::erfc(alpha * distance) / distance;
                }
            }
            real_matrix[static_cast<std::size_t>(target * count + center)] = real
                * charges[static_cast<std::size_t>(target)] * charges[static_cast<std::size_t>(center)];
        }
    }

    // Rewrite sin(phi_j - phi_i + pi/4) into products of per-atom sine and
    // cosine values. This replaces count^2 transcendental calls per G-vector
    // with 2 * count calls; the remaining work is ordinary multiply-adds.
    const double inverse_sqrt_two = 1.0 / std::sqrt(2.0);
    std::vector<double> sine_phase(static_cast<std::size_t>(count));
    std::vector<double> cosine_phase(static_cast<std::size_t>(count));
    std::vector<double> sum_phase(static_cast<std::size_t>(count));
    std::vector<double> difference_phase(static_cast<std::size_t>(count));
    std::vector<double> reciprocal_matrix(static_cast<std::size_t>(count * count), 0.0);
    for (std::size_t g_index = 0; g_index < g_vectors.size(); ++g_index) {
        const Vec3 g = g_vectors[g_index];
        for (int atom = 0; atom < count; ++atom) {
            const double phase = dot(g, positions[static_cast<std::size_t>(atom)]);
            const double sine = std::sin(phase);
            const double cosine = std::cos(phase);
            sine_phase[static_cast<std::size_t>(atom)] = sine;
            cosine_phase[static_cast<std::size_t>(atom)] = cosine;
            sum_phase[static_cast<std::size_t>(atom)] = sine + cosine;
            difference_phase[static_cast<std::size_t>(atom)] = sine - cosine;
        }
        const double factor = g_factors[g_index];
        for (int i = 0; i < count; ++i) {
            double* row_values = reciprocal_matrix.data() + static_cast<std::size_t>(i * count);
            const double sine_i = sine_phase[static_cast<std::size_t>(i)];
            const double cosine_i = cosine_phase[static_cast<std::size_t>(i)];
            for (int j = 0; j < count; ++j) {
                row_values[j] += (cosine_i * sum_phase[static_cast<std::size_t>(j)]
                    + sine_i * difference_phase[static_cast<std::size_t>(j)])
                    * factor * inverse_sqrt_two;
            }
        }
    }

    std::vector<double> matrix(static_cast<std::size_t>(count * count), 0.0);
    const double reciprocal_scale = 4.0 * kPi / volume * std::sqrt(2.0);
    for (int i = 0; i < count; ++i) {
        const double zi = charges[static_cast<std::size_t>(i)];
        for (int j = 0; j < count; ++j) {
            const double zj = charges[static_cast<std::size_t>(j)];
            matrix[static_cast<std::size_t>(i * count + j)] = real_matrix[static_cast<std::size_t>(i * count + j)]
                + reciprocal_matrix[static_cast<std::size_t>(i * count + j)] * reciprocal_scale * zi * zj;
            if (i == j) {
                matrix[static_cast<std::size_t>(i * count + j)] = 0.5 * matrix[static_cast<std::size_t>(i * count + j)]
                    - alpha / std::sqrt(kPi) * zi * zi;
            }
            matrix[static_cast<std::size_t>(i * count + j)] += -kPi / (2.0 * volume * alpha * alpha) * 2.0 * zi * zj;
            if (i == j) {
                matrix[static_cast<std::size_t>(i * count + j)] -= -kPi / (2.0 * volume * alpha * alpha) * zi * zj;
            }
        }
    }
    return matrix;
}

} // namespace mdescriptor::detail
