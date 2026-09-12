#pragma once

// Host math for the SO3 radial basis, shared between the CPU standalone
// rotational descriptors and the CUDA extended descriptor backend so both
// paths stay bit-compatible.  The CUDA side wraps these functions to keep its
// own signatures (quadrature out-parameter and the positive-eigenvalue guard).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace mdescriptor::detail {

inline std::vector<double> inverse_symmetric_sqrt(
    const std::vector<double>& matrix, int size) {
    std::vector<double> values = matrix;
    std::vector<double> vectors(static_cast<std::size_t>(size * size), 0.0);
    for (int i = 0; i < size; ++i) {
        vectors[static_cast<std::size_t>(i * size + i)] = 1.0;
    }
    for (int iteration = 0; iteration < 100 * size * size; ++iteration) {
        int p = 0;
        int q = 1 < size ? 1 : 0;
        double largest = 0.0;
        for (int i = 0; i < size; ++i) {
            for (int j = i + 1; j < size; ++j) {
                if (std::abs(values[static_cast<std::size_t>(i * size + j)]) > largest) {
                    largest = std::abs(values[static_cast<std::size_t>(i * size + j)]);
                    p = i;
                    q = j;
                }
            }
        }
        if (largest < 1e-15) {
            break;
        }
        const double angle = 0.5 * std::atan2(
            2.0 * values[static_cast<std::size_t>(p * size + q)],
            values[static_cast<std::size_t>(q * size + q)] - values[static_cast<std::size_t>(p * size + p)]);
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        for (int k = 0; k < size; ++k) {
            const double vkp = values[static_cast<std::size_t>(k * size + p)];
            const double vkq = values[static_cast<std::size_t>(k * size + q)];
            values[static_cast<std::size_t>(k * size + p)] = c * vkp - s * vkq;
            values[static_cast<std::size_t>(k * size + q)] = s * vkp + c * vkq;
        }
        for (int k = 0; k < size; ++k) {
            const double vpk = values[static_cast<std::size_t>(p * size + k)];
            const double vqk = values[static_cast<std::size_t>(q * size + k)];
            values[static_cast<std::size_t>(p * size + k)] = c * vpk - s * vqk;
            values[static_cast<std::size_t>(q * size + k)] = s * vpk + c * vqk;
        }
        for (int k = 0; k < size; ++k) {
            const double vkp = vectors[static_cast<std::size_t>(k * size + p)];
            const double vkq = vectors[static_cast<std::size_t>(k * size + q)];
            vectors[static_cast<std::size_t>(k * size + p)] = c * vkp - s * vkq;
            vectors[static_cast<std::size_t>(k * size + q)] = s * vkp + c * vkq;
        }
    }
    std::vector<double> result(static_cast<std::size_t>(size * size), 0.0);
    for (int row = 0; row < size; ++row) {
        for (int column = 0; column < size; ++column) {
            for (int eigen = 0; eigen < size; ++eigen) {
                result[static_cast<std::size_t>(row * size + column)] += vectors[static_cast<std::size_t>(row * size + eigen)]
                    * vectors[static_cast<std::size_t>(column * size + eigen)]
                    / std::sqrt(values[static_cast<std::size_t>(eigen * size + eigen)]);
            }
        }
    }
    return result;
}

inline std::vector<double> so3_radial_basis(
    int n_max, int l_max, double cutoff, double alpha) {
    constexpr double kPi = 3.141592653589793238462643383279502884;
    std::vector<double> overlap(static_cast<std::size_t>(n_max * n_max), 0.0);
    for (int a = 1; a <= n_max; ++a) {
        for (int b = 1; b <= n_max; ++b) {
            overlap[static_cast<std::size_t>((a - 1) * n_max + b - 1)] = std::sqrt(
                (2.0 * a + 5.0) * (2.0 * a + 6.0) * (2.0 * a + 7.0)
                * (2.0 * b + 5.0) * (2.0 * b + 6.0) * (2.0 * b + 7.0))
                / ((5.0 + a + b) * (6.0 + a + b) * (7.0 + a + b));
        }
    }
    const auto w = inverse_symmetric_sqrt(overlap, n_max);
    const int quadrature_count = (n_max + l_max + 1) * 10;
    std::vector<double> basis(static_cast<std::size_t>(n_max * quadrature_count), 0.0);
    for (int q_index = 0; q_index < quadrature_count; ++q_index) {
        const double x = std::cos((2.0 * (q_index + 1) - 1.0) * kPi / (2.0 * quadrature_count));
        const double radius = cutoff * 0.5 * (x + 1.0);
        const double weight = (kPi / quadrature_count) * cutoff * 0.5;
        const double common = radius * radius * std::exp(-alpha * radius * radius)
            * std::sqrt(std::max(0.0, 1.0 - x * x)) * weight;
        for (int n = 0; n < n_max; ++n) {
            double g = 0.0;
            for (int a = 1; a <= n_max; ++a) {
                const double phi = std::pow(cutoff - radius, a + 2.0) / std::sqrt(
                    2.0 * std::pow(cutoff, 2.0 * a + 7.0)
                    / ((2.0 * a + 5.0) * (2.0 * a + 6.0) * (2.0 * a + 7.0)));
                g += w[static_cast<std::size_t>(n * n_max + a - 1)] * phi;
            }
            basis[static_cast<std::size_t>(n * quadrature_count + q_index)] = g * common;
        }
    }
    return basis;
}

} // namespace mdescriptor::detail
