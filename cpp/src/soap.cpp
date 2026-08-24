#include "mdescriptor/descriptor.hpp"
#include "mdescriptor/neighbor.hpp"
#include "descriptor_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mdescriptor {
using namespace detail;

namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kSqrt2 = 1.414213562373095048801688724209698079;

std::vector<double> solid_harmonic_normalization(int l_max) {
    const int width = l_max + 1;
    std::vector<double> result(static_cast<std::size_t>(width * width), 0.0);
    for (int l = 0; l <= l_max; ++l) {
        for (int m = 0; m <= l; ++m) {
            result[static_cast<std::size_t>(l * width + m)] = std::sqrt(
                (2.0 * l + 1.0) / (4.0 * kPi)
                * std::exp(std::lgamma(l - m + 1.0) - std::lgamma(l + m + 1.0))
            );
        }
    }
    return result;
}

void solid_harmonics_l6(Vec3 v, const std::vector<double>& normalization, double* result) {
    const double x = v.x;
    const double y = v.y;
    const double z = v.z;
    const double x2 = x * x;
    const double y2 = y * y;
    const double z2 = z * z;
    const double z4 = z2 * z2;
    const double r2 = x2 + y2 + z2;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;
    const double x3 = x2 * x;
    const double y3 = y2 * y;
    const double x4 = x2 * x2;
    const double y4 = y2 * y2;
    const double x5 = x4 * x;
    const double y5 = y4 * y;
    const double x6 = x4 * x2;
    const double y6 = y4 * y2;
    const int width = 7;
    auto write = [&](int l, int m, double cosine, double sine = 0.0) {
        const double factor = normalization[static_cast<std::size_t>(l * width + m)]
            * (m == 0 ? 1.0 : kSqrt2);
        if (m == 0) {
            result[l * l + l] = factor * cosine;
        } else {
            result[l * l + (l - m)] = factor * cosine;
            result[l * l + (l + m)] = factor * sine;
        }
    };

    write(0, 0, 1.0);
    write(1, 0, z);
    write(1, 1, -x, -y);
    write(2, 0, 0.5 * (3.0 * z2 - r2));
    write(2, 1, -3.0 * z * x, -3.0 * z * y);
    write(2, 2, 3.0 * (x2 - y2), 6.0 * x * y);
    write(3, 0, 0.5 * z * (5.0 * z2 - 3.0 * r2));
    write(3, 1, -1.5 * x * (5.0 * z2 - r2), -1.5 * y * (5.0 * z2 - r2));
    write(3, 2, 15.0 * z * (x2 - y2), 30.0 * z * x * y);
    write(3, 3, -15.0 * (x3 - 3.0 * x * y2), -15.0 * (3.0 * x2 * y - y3));
    write(4, 0, 0.125 * (35.0 * z2 * z2 - 30.0 * z2 * r2 + 3.0 * r4));
    write(4, 1, -2.5 * x * z * (7.0 * z2 - 3.0 * r2), -2.5 * y * z * (7.0 * z2 - 3.0 * r2));
    write(4, 2, 7.5 * (7.0 * z2 - r2) * (x2 - y2), 15.0 * (7.0 * z2 - r2) * x * y);
    write(4, 3, -105.0 * z * (x3 - 3.0 * x * y2), -105.0 * z * (3.0 * x2 * y - y3));
    write(4, 4, 105.0 * (x4 - 6.0 * x2 * y2 + y4), 420.0 * x * y * (x2 - y2));
    write(5, 0, 0.125 * z * (63.0 * z4 - 70.0 * z2 * r2 + 15.0 * r4));
    write(5, 1, -1.875 * x * (21.0 * z4 - 14.0 * z2 * r2 + r4), -1.875 * y * (21.0 * z4 - 14.0 * z2 * r2 + r4));
    write(5, 2, 52.5 * z * (3.0 * z2 - r2) * (x2 - y2), 105.0 * z * (3.0 * z2 - r2) * x * y);
    write(5, 3, -52.5 * (9.0 * z2 - r2) * (x3 - 3.0 * x * y2), -52.5 * (9.0 * z2 - r2) * (3.0 * x2 * y - y3));
    write(5, 4, 945.0 * z * (x4 - 6.0 * x2 * y2 + y4), 3780.0 * z * x * y * (x2 - y2));
    write(5, 5, -945.0 * (x5 - 10.0 * x3 * y2 + 5.0 * x * y4), -945.0 * (5.0 * x4 * y - 10.0 * x2 * y3 + y5));
    write(6, 0, (231.0 * z2 * z2 * z2 - 315.0 * z2 * z2 * r2 + 105.0 * z2 * r4 - 5.0 * r6) / 16.0);
    write(6, 1, -2.625 * x * z * (33.0 * z4 - 30.0 * z2 * r2 + 5.0 * r4), -2.625 * y * z * (33.0 * z4 - 30.0 * z2 * r2 + 5.0 * r4));
    write(6, 2, 13.125 * (33.0 * z4 - 18.0 * z2 * r2 + r4) * (x2 - y2), 26.25 * (33.0 * z4 - 18.0 * z2 * r2 + r4) * x * y);
    write(6, 3, -157.5 * z * (11.0 * z2 - 3.0 * r2) * (x3 - 3.0 * x * y2), -157.5 * z * (11.0 * z2 - 3.0 * r2) * (3.0 * x2 * y - y3));
    write(6, 4, 472.5 * (11.0 * z2 - r2) * (x4 - 6.0 * x2 * y2 + y4), 1890.0 * (11.0 * z2 - r2) * x * y * (x2 - y2));
    write(6, 5, -10395.0 * z * (x5 - 10.0 * x3 * y2 + 5.0 * x * y4), -10395.0 * z * (5.0 * x4 * y - 10.0 * x2 * y3 + y5));
    write(6, 6, 10395.0 * (x6 - 15.0 * x4 * y2 + 15.0 * x2 * y4 - y6), 10395.0 * (6.0 * x5 * y - 20.0 * x3 * y3 + 6.0 * x * y5));
}

void solid_harmonics(
    Vec3 v,
    int l_max,
    const std::vector<double>& normalization,
    std::vector<double>& result,
    std::vector<double>& legendre
) {
    const int width = l_max + 1;
    const double r2 = norm2(v);

    // Low-order Cartesian regular-solid-harmonic polynomials avoid
    // per-neighbor trigonometric calls and associated-Legendre work.
    if (l_max <= 4) {
        const double x = v.x;
        const double y = v.y;
        const double z = v.z;
        const double x2 = x * x;
        const double y2 = y * y;
        const double z2 = z * z;
        const double r4 = r2 * r2;
        auto write = [&](int l, int m, double cosine, double sine = 0.0) {
            const double scale = normalization[static_cast<std::size_t>(l * width + m)]
                * (m == 0 ? 1.0 : kSqrt2);
            if (m == 0) {
                result[l * l + l] = scale * cosine;
            } else {
                result[l * l + (l - m)] = scale * cosine;
                result[l * l + (l + m)] = scale * sine;
            }
        };

        write(0, 0, 1.0);
        if (l_max >= 1) {
            write(1, 0, z);
            write(1, 1, -x, -y);
        }
        if (l_max >= 2) {
            write(2, 0, 0.5 * (3.0 * z2 - r2));
            write(2, 1, -3.0 * z * x, -3.0 * z * y);
            write(2, 2, 3.0 * (x2 - y2), 6.0 * x * y);
        }
        if (l_max >= 3) {
            write(3, 0, 0.5 * z * (5.0 * z2 - 3.0 * r2));
            write(3, 1, -1.5 * x * (5.0 * z2 - r2), -1.5 * y * (5.0 * z2 - r2));
            write(3, 2, 15.0 * z * (x2 - y2), 30.0 * z * x * y);
            write(3, 3, -15.0 * (x2 * x - 3.0 * x * y2), -15.0 * (3.0 * x2 * y - y2 * y));
        }
        if (l_max >= 4) {
            write(4, 0, 0.125 * (35.0 * z2 * z2 - 30.0 * z2 * r2 + 3.0 * r4));
            write(4, 1, -2.5 * x * (7.0 * z2 * z - 3.0 * z * r2), -2.5 * y * (7.0 * z2 * z - 3.0 * z * r2));
            write(4, 2, 7.5 * (7.0 * z2 - r2) * (x2 - y2), 15.0 * (7.0 * z2 - r2) * x * y);
            write(4, 3, -105.0 * z * (x2 * x - 3.0 * x * y2), -105.0 * z * (3.0 * x2 * y - y2 * y));
            write(4, 4, 105.0 * (x2 * x2 - 6.0 * x2 * y2 + y2 * y2), 420.0 * x * y * (x2 - y2));
        }
        return;
    }

    // Keep the common l_max=6 path Cartesian as well. The generic branch
    // below is substantially more expensive because it evaluates sqrt/pow
    // and trigonometric recurrences for every neighbor.
    if (l_max == 6) {
        solid_harmonics_l6(v, normalization, result.data());
        return;
    }

    if (l_max <= 6) {
        std::array<std::array<double, 7>, 7> real{};
        std::array<std::array<double, 7>, 7> imaginary{};
        real[0][0] = 1.0;
        for (int m = 0; m <= l_max; ++m) {
            if (m > 0) {
                const double factor = -(2.0 * m - 1.0);
                const double previous_real = real[m - 1][m - 1];
                const double previous_imaginary = imaginary[m - 1][m - 1];
                real[m][m] = factor * (v.x * previous_real - v.y * previous_imaginary);
                imaginary[m][m] = factor * (v.x * previous_imaginary + v.y * previous_real);
            }
            if (m == 0) {
                if (l_max >= 1) {
                    real[1][0] = v.z;
                }
                for (int l = 2; l <= l_max; ++l) {
                    real[l][0] = (
                        (2.0 * l - 1.0) * v.z * real[l - 1][0]
                        - (l - 1.0) * r2 * real[l - 2][0]
                    ) / l;
                }
            } else {
                if (m < l_max) {
                    real[m + 1][m] = (2.0 * m + 1.0) * v.z * real[m][m];
                    imaginary[m + 1][m] = (2.0 * m + 1.0) * v.z * imaginary[m][m];
                }
                for (int l = m + 2; l <= l_max; ++l) {
                    real[l][m] = (
                        (2.0 * l - 1.0) * v.z * real[l - 1][m]
                        - (l + m - 1.0) * r2 * real[l - 2][m]
                    ) / (l - m);
                    imaginary[l][m] = (
                        (2.0 * l - 1.0) * v.z * imaginary[l - 1][m]
                        - (l + m - 1.0) * r2 * imaginary[l - 2][m]
                    ) / (l - m);
                }
            }
        }
        std::fill(result.begin(), result.end(), 0.0);
        const double scale = kSqrt2;
        for (int l = 0; l <= l_max; ++l) {
            result[l * l + l] = normalization[static_cast<std::size_t>(l * width)] * real[l][0];
            for (int m = 1; m <= l; ++m) {
                const double factor = scale * normalization[static_cast<std::size_t>(l * width + m)];
                result[l * l + (l - m)] = factor * real[l][m];
                result[l * l + (l + m)] = factor * imaginary[l][m];
            }
        }
        return;
    }

    std::fill(result.begin(), result.end(), 0.0);
    std::fill(legendre.begin(), legendre.end(), 0.0);
    if (r2 == 0.0) {
        result[0] = 1.0 / std::sqrt(4.0 * kPi);
        return;
    }
    const double r = std::sqrt(r2);
    const double x = std::max(-1.0, std::min(1.0, v.z / r));
    const double sin_theta = std::sqrt(std::max(0.0, (v.x * v.x + v.y * v.y) / r2));
    const double rho = std::sqrt(v.x * v.x + v.y * v.y);
    const double cos_phi = rho == 0.0 ? 1.0 : v.x / rho;
    const double sin_phi = rho == 0.0 ? 0.0 : v.y / rho;
    for (int m = 0; m <= l_max; ++m) {
        double pmm = 1.0;
        for (int q = 1; q <= m; ++q) {
            pmm *= -(2.0 * q - 1.0) * sin_theta;
        }
        legendre[static_cast<std::size_t>(m * width + m)] = pmm;
        if (m < l_max) {
            legendre[static_cast<std::size_t>((m + 1) * width + m)] = (2.0 * m + 1.0) * x * pmm;
        }
        for (int l = m + 2; l <= l_max; ++l) {
            legendre[static_cast<std::size_t>(l * width + m)] = (
                (2.0 * l - 1.0) * x * legendre[static_cast<std::size_t>((l - 1) * width + m)]
                - (l + m - 1.0) * legendre[static_cast<std::size_t>((l - 2) * width + m)]
            ) / (l - m);
        }
    }

    for (int l = 0; l <= l_max; ++l) {
        const double rl = l == 0 ? 1.0 : std::pow(r, l);
        double cos_m_phi = 1.0;
        double sin_m_phi = 0.0;
        for (int m = 0; m <= l; ++m) {
            const double base = normalization[static_cast<std::size_t>(l * width + m)]
                * rl * legendre[static_cast<std::size_t>(l * width + m)];
            if (m == 0) {
                result[l * l + l] = base;
            } else {
                result[l * l + (l - m)] = kSqrt2 * base * cos_m_phi;
                result[l * l + (l + m)] = kSqrt2 * base * sin_m_phi;
            }
            const double next_cos = cos_m_phi * cos_phi - sin_m_phi * sin_phi;
            const double next_sin = sin_m_phi * cos_phi + cos_m_phi * sin_phi;
            cos_m_phi = next_cos;
            sin_m_phi = next_sin;
        }
    }
}

// The l_max=3 path is common for the benchmark and avoids per-neighbor
// vector/lambda setup in the generic harmonic routine.
inline void solid_harmonics_l3_soa(
    const double* __restrict x_values,
    const double* __restrict y_values,
    const double* __restrict z_values,
    std::size_t count,
    double* __restrict result
) {
    if (count == 0) {
        return;
    }
#ifdef _OPENMP
#pragma omp simd
#endif
    for (std::size_t i = 0; i < count; ++i) {
        const double x = x_values[i];
        const double y = y_values[i];
        const double z = z_values[i];
        const double x2 = x * x;
        const double y2 = y * y;
        const double z2 = z * z;
        const double r2 = x2 + y2 + z2;
        result[i] = 0.28209479177387814;
        result[count + i] = -0.48860251190292003 * x;
        result[2 * count + i] = 0.48860251190291992 * z;
        result[3 * count + i] = -0.48860251190292003 * y;
        result[4 * count + i] = 0.54627421529603981 * (x2 - y2);
        result[5 * count + i] = -1.0925484305920792 * z * x;
        result[6 * count + i] = 0.31539156525252005 * (3.0 * z2 - r2);
        result[7 * count + i] = -1.0925484305920792 * z * y;
        result[8 * count + i] = 1.0925484305920796 * x * y;
        result[9 * count + i] = -0.5900435899266433 * (x2 * x - 3.0 * x * y2);
        result[10 * count + i] = 1.4453057213202767 * z * (x2 - y2);
        result[11 * count + i] = -0.45704579946446594 * x * (5.0 * z2 - r2);
        result[12 * count + i] = 0.3731763325901154 * z * (5.0 * z2 - 3.0 * r2);
        result[13 * count + i] = -0.45704579946446594 * y * (5.0 * z2 - r2);
        result[14 * count + i] = 2.8906114426405534 * z * x * y;
        result[15 * count + i] = -0.5900435899266433 * (3.0 * x2 * y - y2 * y);
    }
}

// DScribe's common l_max=4 path uses literal regular-solid-harmonic
// coefficients and a structure-of-arrays layout. Keeping the neighbor index
// as the innermost dimension lets the compiler vectorize this whole kernel.
inline void solid_harmonics_l4_soa(
    const double* __restrict x_values,
    const double* __restrict y_values,
    const double* __restrict z_values,
    std::size_t count,
    double* __restrict result
) {
    if (count == 0) {
        return;
    }
    const double* const x = x_values;
    const double* const y = y_values;
    const double* const z = z_values;
#ifdef _OPENMP
#pragma omp simd
#endif
    for (std::size_t i = 0; i < count; ++i) {
        const double xi = x[i];
        const double yi = y[i];
        const double zi = z[i];
        const double x2 = xi * xi;
        const double y2 = yi * yi;
        const double z2 = zi * zi;
        const double r2 = x2 + y2 + z2;
        const double r4 = r2 * r2;
        const double x4 = x2 * x2;
        const double y4 = y2 * y2;

        result[i] = 0.28209479177387814;
        result[count + i] = -0.48860251190292003 * xi;
        result[2 * count + i] = 0.48860251190291992 * zi;
        result[3 * count + i] = -0.48860251190292003 * yi;

        result[4 * count + i] = 0.54627421529603981 * (x2 - y2);
        result[5 * count + i] = -1.0925484305920792 * zi * xi;
        result[6 * count + i] = 0.31539156525252005 * (3.0 * z2 - r2);
        result[7 * count + i] = -1.0925484305920792 * zi * yi;
        result[8 * count + i] = 1.0925484305920796 * xi * yi;

        result[9 * count + i] = -0.5900435899266433 * (x2 * xi - 3.0 * xi * y2);
        result[10 * count + i] = 1.4453057213202767 * zi * (x2 - y2);
        result[11 * count + i] = -0.45704579946446594 * xi * (5.0 * z2 - r2);
        result[12 * count + i] = 0.3731763325901154 * zi * (5.0 * z2 - 3.0 * r2);
        result[13 * count + i] = -0.45704579946446594 * yi * (5.0 * z2 - r2);
        result[14 * count + i] = 2.8906114426405534 * zi * xi * yi;
        result[15 * count + i] = -0.5900435899266433 * (3.0 * x2 * yi - y2 * yi);

        result[16 * count + i] = 0.62583573544917648 * (x4 - 6.0 * x2 * y2 + y4);
        result[17 * count + i] = -1.7701307697799302 * zi * (x2 * xi - 3.0 * xi * y2);
        result[18 * count + i] = 0.47308734787877965 * (7.0 * z2 - r2) * (x2 - y2);
        result[19 * count + i] = -0.66904654355728899 * xi * (7.0 * z2 * zi - 3.0 * zi * r2);
        result[20 * count + i] = 0.10578554691520431 * (35.0 * z2 * z2 - 30.0 * z2 * r2 + 3.0 * r4);
        result[21 * count + i] = -0.66904654355728899 * yi * (7.0 * z2 * zi - 3.0 * zi * r2);
        result[22 * count + i] = 0.9461746957575593 * (7.0 * z2 - r2) * xi * yi;
        result[23 * count + i] = -1.7701307697799302 * zi * (3.0 * x2 * yi - y2 * yi);
        result[24 * count + i] = 2.5033429417967059 * xi * yi * (x2 - y2);
    }
}


std::int64_t soap_features(const SoapOptions& options) {
    const std::int64_t species = static_cast<std::int64_t>(options.species.size());
    switch (options.compression) {
    case 1: // mu2
        return static_cast<std::int64_t>(options.n_max) * (options.n_max + 1) / 2 * (options.l_max + 1);
    case 2: // mu1nu1
        return species * options.n_max * options.n_max * (options.l_max + 1);
    case 3: // crossover
        return species * options.n_max * (options.n_max + 1) / 2 * (options.l_max + 1);
    default: {
        const std::int64_t n = species * options.n_max;
        return n * (n + 1) / 2 * (options.l_max + 1);
    }
    }
}

double soap_weight(const SoapOptions& options, double distance, bool exact_self, std::size_t type) {
    double value = 1.0;
    if (options.weighting_has_function) {
        const double ratio = distance / options.weighting_r0;
        switch (options.weighting_function) {
        case 1: { // polynomial
            if (distance > options.weighting_r0) {
                value = 0.0;
            } else {
                value = options.weighting_c * std::pow(1.0 + 2.0 * ratio * ratio * ratio
                    - 3.0 * ratio * ratio, options.weighting_m);
            }
            break;
        }
        case 2: // power
            value = options.weighting_c / (options.weighting_d + std::pow(ratio, options.weighting_m));
            break;
        case 3: // exponential
            value = options.weighting_c / (options.weighting_d + std::exp(-ratio));
            break;
        default:
            throw std::invalid_argument("invalid SOAP weighting function");
        }
    }
    if (exact_self && options.weighting_has_w0) {
        value = options.weighting_w0;
    }
    if (!options.species_weights.empty()) {
        value *= options.species_weights[type];
    }
    return value;
}

struct SoapWorkspace {
    std::vector<double> harmonics;
    std::vector<double> legendre;
    std::vector<double> neighbor_harmonics;
    std::vector<double> neighbor_x;
    std::vector<double> neighbor_y;
    std::vector<double> neighbor_z;
    std::vector<double> neighbor_distance2;
    std::vector<double> neighbor_distance;
    std::vector<unsigned char> neighbor_self;
    std::vector<std::int32_t> neighbor_types;
    std::vector<double> neighbor_weights;
    std::vector<double> radial_values;
    std::vector<double> polynomial_integrand;
    std::unordered_map<std::uint64_t, std::size_t> polynomial_distance_cache;
    std::vector<double> polynomial_distance_cache_values;
    std::size_t polynomial_cache_signature = 0;
    std::unordered_map<std::uint64_t, std::size_t> gto_distance_cache;
    std::vector<double> gto_distance_cache_values;
    std::size_t gto_cache_signature = 0;
    std::vector<double> harmonic_sums;
    std::vector<std::size_t> type_offsets;
    std::vector<std::size_t> type_cursors;
    std::vector<double> local_coefficients;
    std::vector<double> summed_coefficients;
    std::vector<double> power_values;
};

const double* polynomial_flir(
    const SoapOptions& options,
    double radius,
    SoapWorkspace& workspace) {
    const std::size_t grid_size = options.radial_grid.size();
    const std::size_t flir_count = static_cast<std::size_t>(options.l_max + 1) * grid_size;
    std::uint64_t distance_key = 0;
    std::memcpy(&distance_key, &radius, sizeof(distance_key));
    const auto cached = workspace.polynomial_distance_cache.find(distance_key);
    if (cached != workspace.polynomial_distance_cache.end()) {
        return workspace.polynomial_distance_cache_values.data() + cached->second * flir_count;
    }

    const std::size_t cache_index = workspace.polynomial_distance_cache_values.size() / flir_count;
    workspace.polynomial_distance_cache_values.resize(
        workspace.polynomial_distance_cache_values.size() + flir_count);
    double* values = workspace.polynomial_distance_cache_values.data() + cache_index * flir_count;
    std::fill(values, values + flir_count, 0.0);
    const double eta = 1.0 / (2.0 * options.sigma * options.sigma);
    for (std::size_t q = 0; q < grid_size; ++q) {
        const double radial = options.radial_grid[q];
        const double radial2 = radial * radial;
        if (radius <= 1e-14) {
            values[q] = std::exp(-eta * radial2);
            continue;
        }
        const double prefactor = 0.25 / (eta * radius * radial);
        const double minus = std::exp(-eta * (radial - radius) * (radial - radius));
        const double plus = std::exp(-eta * (radial + radius) * (radial + radius));
        values[q] = prefactor * (minus - plus);
        if (options.l_max >= 1) {
            values[grid_size + q] = prefactor * (minus + plus - 2.0 * values[q]);
        }
        for (int l = 2; l <= options.l_max; ++l) {
            const double value = values[static_cast<std::size_t>(l - 2) * grid_size + q]
                - prefactor * (4.0 * l - 2.0)
                    * values[static_cast<std::size_t>(l - 1) * grid_size + q];
            values[static_cast<std::size_t>(l) * grid_size + q] = std::max(0.0, value);
        }
    }
    workspace.polynomial_distance_cache.emplace(distance_key, cache_index);
    return values;
}

struct SoapPlan {
    std::unordered_map<std::int32_t, std::int32_t> mapping;
    std::vector<double> power_prefactors;
    std::vector<double> harmonic_normalization;
    std::vector<double> radial_prefactors;
    std::vector<double> radial_exponents;
    std::size_t gto_cache_signature = 0;
    std::vector<double> polynomial_weighted_values_qn;
    std::vector<double> polynomial_quadrature;
    std::size_t polynomial_cache_signature = 0;
    double soap_scale = 0.0;
};

SoapPlan prepare_soap(const SoapOptions& options) {
    const int width = options.l_max + 1;
    const double eta = 1.0 / (2.0 * options.sigma * options.sigma);
    SoapPlan plan;
    plan.mapping = species_map(options.species);
    plan.soap_scale = kPi * std::sqrt(kPi);
    plan.power_prefactors.resize(static_cast<std::size_t>(width));
    for (int l = 0; l <= options.l_max; ++l) {
        plan.power_prefactors[static_cast<std::size_t>(l)] = kPi * std::sqrt(8.0 / (2.0 * l + 1.0));
    }
    plan.harmonic_normalization = solid_harmonic_normalization(options.l_max);
    if (options.radial_basis == 0) {
        plan.radial_prefactors.resize(static_cast<std::size_t>(width * options.n_max));
        plan.radial_exponents.resize(static_cast<std::size_t>(width * options.n_max));
        std::vector<double> eta_powers(static_cast<std::size_t>(width), 1.0);
        for (int l = 1; l <= options.l_max; ++l) {
            eta_powers[static_cast<std::size_t>(l)] = eta_powers[static_cast<std::size_t>(l - 1)] * eta;
        }
        for (int l = 0; l <= options.l_max; ++l) {
            for (int k = 0; k < options.n_max; ++k) {
                const std::size_t index = static_cast<std::size_t>(l * options.n_max + k);
                const double alpha = options.alphas[index];
                const double denominator = alpha + eta;
                plan.radial_prefactors[index] = eta_powers[static_cast<std::size_t>(l)]
                    * std::pow(denominator, -l - 1.5);
                plan.radial_exponents[index] = alpha * eta / denominator;
            }
        }
        std::size_t signature = std::hash<double>{}(options.sigma);
        signature ^= static_cast<std::size_t>(options.l_max + 0x9e3779b9)
            + (signature << 6) + (signature >> 2);
        signature ^= static_cast<std::size_t>(options.n_max + 0x9e3779b9)
            + (signature << 6) + (signature >> 2);
        for (double value : options.alphas) {
            signature ^= std::hash<double>{}(value) + (signature << 6) + (signature >> 2);
        }
        plan.gto_cache_signature = signature;
    } else {
        const std::size_t grid_size = options.radial_grid.size();
        std::size_t signature = std::hash<double>{}(options.sigma);
        signature ^= static_cast<std::size_t>(options.l_max + 0x9e3779b9)
            + (signature << 6) + (signature >> 2);
        signature ^= static_cast<std::size_t>(options.n_max + 0x9e3779b9)
            + (signature << 6) + (signature >> 2);
        for (double value : options.radial_grid) {
            signature ^= std::hash<double>{}(value) + (signature << 6) + (signature >> 2);
        }
        for (double value : options.radial_values) {
            signature ^= std::hash<double>{}(value) + (signature << 6) + (signature >> 2);
        }
        plan.polynomial_cache_signature = signature;
        plan.polynomial_weighted_values_qn.resize(grid_size * static_cast<std::size_t>(options.n_max));
        plan.polynomial_quadrature.resize(grid_size);
        for (std::size_t q = 0; q < grid_size; ++q) {
            const double radial = options.radial_grid[q];
            plan.polynomial_quadrature[q] = options.radial_weights[q] * radial * radial;
            for (int n = 0; n < options.n_max; ++n) {
                plan.polynomial_weighted_values_qn[
                    q * static_cast<std::size_t>(options.n_max) + static_cast<std::size_t>(n)]
                    = plan.polynomial_quadrature[q] * options.radial_values[
                        static_cast<std::size_t>(n) * grid_size + q];
            }
        }
    }
    return plan;
}

const double* gto_radial_values(
    const SoapOptions& options,
    const SoapPlan& plan,
    double distance2,
    SoapWorkspace& workspace) {
    const std::size_t radial_count = static_cast<std::size_t>(options.l_max + 1) * options.n_max;
    std::uint64_t distance_key = 0;
    std::memcpy(&distance_key, &distance2, sizeof(distance_key));
    const auto cached = workspace.gto_distance_cache.find(distance_key);
    if (cached != workspace.gto_distance_cache.end()) {
        return workspace.gto_distance_cache_values.data() + cached->second * radial_count;
    }
    const std::size_t cache_index = workspace.gto_distance_cache_values.size() / radial_count;
    workspace.gto_distance_cache_values.resize(
        workspace.gto_distance_cache_values.size() + radial_count);
    double* values = workspace.gto_distance_cache_values.data() + cache_index * radial_count;
    for (std::size_t radial_index = 0; radial_index < radial_count; ++radial_index) {
        values[radial_index] = plan.radial_prefactors[radial_index]
            * std::exp(-plan.radial_exponents[radial_index] * distance2);
    }
    workspace.gto_distance_cache.emplace(distance_key, cache_index);
    return values;
}

void compute_soap_structure(
    const StructureBatchView& batch,
    const SoapOptions& options,
    const NeighborGraph& neighbor_graph,
    std::int64_t structure,
    double* output,
    const std::shared_ptr<ComputeControl>& control,
    bool parallel_centers,
    const SoapPlan& plan
) {
    const std::int64_t begin = batch.offsets[structure];
    const std::int64_t end = batch.offsets[structure + 1];
    const std::int64_t atom_count = end - begin;
    const std::int64_t features = soap_features(options);
    const int width = options.l_max + 1;
    const std::size_t coefficient_types = options.compression == 1 ? 1 : options.species.size();
    const std::size_t coefficient_size = coefficient_types * static_cast<std::size_t>(options.n_max) * static_cast<std::size_t>(width * width);
    const std::size_t harmonic_count = static_cast<std::size_t>(width * width);
    std::vector<std::int32_t> atom_types(static_cast<std::size_t>(atom_count));
    for (std::int64_t local = 0; local < atom_count; ++local) {
        const auto type_it = plan.mapping.find(batch.numbers[begin + local]);
        if (type_it == plan.mapping.end()) {
            throw std::invalid_argument("batch contains an atomic number outside calculator species");
        }
        atom_types[static_cast<std::size_t>(local)] = type_it->second;
    }
#ifdef _OPENMP
    const int workspace_count = parallel_centers
        ? (options.num_threads > 0 ? options.num_threads : omp_get_max_threads())
        : 1;
#else
    const int workspace_count = 1;
#endif
    auto prepare_workspace = [&](SoapWorkspace& workspace) {
        workspace.harmonics.resize(harmonic_count);
        workspace.legendre.resize(harmonic_count);
        workspace.harmonic_sums.resize(static_cast<std::size_t>(2 * width + 1));
        workspace.type_offsets.resize(options.species.size() + 1);
        workspace.type_cursors.resize(options.species.size());
        workspace.local_coefficients.resize(coefficient_size);
        if (options.compression == 2) {
            workspace.summed_coefficients.resize(
                static_cast<std::size_t>(options.n_max) * harmonic_count);
        }
        if (options.radial_basis == 1) {
            workspace.polynomial_integrand.resize(
                coefficient_types * harmonic_count * options.radial_grid.size());
        }
        if (options.outer_average) {
            workspace.power_values.resize(static_cast<std::size_t>(features));
        }
    };
    thread_local SoapWorkspace serial_workspace;
    std::vector<SoapWorkspace> workspaces;
    if (parallel_centers) {
        workspaces.resize(static_cast<std::size_t>(workspace_count));
        for (SoapWorkspace& workspace : workspaces) {
            prepare_workspace(workspace);
        }
    } else {
        prepare_workspace(serial_workspace);
    }
    if (options.radial_basis == 1) {
        auto reset_cache_if_needed = [&](SoapWorkspace& workspace) {
            if (workspace.polynomial_cache_signature != plan.polynomial_cache_signature) {
                workspace.polynomial_distance_cache.clear();
                workspace.polynomial_distance_cache_values.clear();
                workspace.polynomial_cache_signature = plan.polynomial_cache_signature;
            }
        };
        reset_cache_if_needed(serial_workspace);
        for (SoapWorkspace& workspace : workspaces) {
            reset_cache_if_needed(workspace);
        }
    } else if (options.compression != 0 || options.inner_average) {
        auto reset_cache_if_needed = [&](SoapWorkspace& workspace) {
            if (workspace.gto_cache_signature != plan.gto_cache_signature) {
                workspace.gto_distance_cache.clear();
                workspace.gto_distance_cache_values.clear();
                workspace.gto_cache_signature = plan.gto_cache_signature;
            }
        };
        reset_cache_if_needed(serial_workspace);
        for (SoapWorkspace& workspace : workspaces) {
            reset_cache_if_needed(workspace);
        }
    }
    std::vector<std::vector<double>> average_workspaces;
    if (parallel_centers && options.inner_average) {
        average_workspaces.assign(
            static_cast<std::size_t>(workspace_count),
            std::vector<double>(coefficient_size, 0.0));
    }
    std::vector<std::vector<double>> average_power_workspaces;
    if (parallel_centers && options.outer_average) {
        average_power_workspaces.assign(
            static_cast<std::size_t>(workspace_count),
            std::vector<double>(static_cast<std::size_t>(features), 0.0));
    }
    auto workspace_at = [&](int index) -> SoapWorkspace& {
        return parallel_centers ? workspaces[static_cast<std::size_t>(index)] : serial_workspace;
    };
    std::vector<double> averaged_coefficients(coefficient_size, 0.0);
    auto calculate_coefficients = [&](std::int64_t center, SoapWorkspace& workspace) {
        std::vector<double>& coefficients = workspace.local_coefficients;
        std::fill(coefficients.begin(), coefficients.end(), 0.0);
        const NeighborView neighbors = neighbor_graph.for_center(center);
        const std::size_t neighbor_count = neighbors.size;
        workspace.neighbor_harmonics.resize(neighbor_count * harmonic_count);
        workspace.neighbor_x.resize(neighbor_count);
        workspace.neighbor_y.resize(neighbor_count);
        workspace.neighbor_z.resize(neighbor_count);
        workspace.neighbor_distance2.resize(neighbor_count);
        workspace.neighbor_distance.resize(neighbor_count);
        workspace.neighbor_self.resize(neighbor_count);
        if (options.compression == 1) {
            workspace.neighbor_types.resize(neighbor_count);
        }
        workspace.neighbor_weights.resize(neighbor_count);
        if (options.compression == 1 || options.species.size() == 1) {
            // mu2 treats all neighbors as one density and does not need the
            // species sort. Keeping the graph order avoids a count/prefix/
            // scatter pass on the hottest compression path.
            workspace.type_offsets[0] = 0;
            workspace.type_offsets[1] = neighbor_count;
            for (std::size_t index = 0; index < neighbor_count; ++index) {
                const auto atom = neighbors.atoms[index];
                const std::int32_t type = options.compression == 1
                    ? atom_types[static_cast<std::size_t>(atom - begin)] : 0;
                workspace.neighbor_x[index] = neighbors.displacements[index * 3 + 0];
                workspace.neighbor_y[index] = neighbors.displacements[index * 3 + 1];
                workspace.neighbor_z[index] = neighbors.displacements[index * 3 + 2];
                workspace.neighbor_distance2[index] = neighbors.distance2[index];
                workspace.neighbor_distance[index] = std::sqrt(neighbors.distance2[index]);
                workspace.neighbor_self[index] = neighbors.exact_self(index, center) ? 1 : 0;
                if (options.compression == 1) {
                    workspace.neighbor_types[index] = type;
                }
            }
        } else {
            std::fill(workspace.type_cursors.begin(), workspace.type_cursors.end(), 0);
            for (std::size_t index = 0; index < neighbor_count; ++index) {
                const auto atom = neighbors.atoms[index];
                const std::size_t type = static_cast<std::size_t>(atom_types[static_cast<std::size_t>(atom - begin)]);
                ++workspace.type_cursors[type];
            }
            workspace.type_offsets[0] = 0;
            for (std::size_t type = 0; type < options.species.size(); ++type) {
                workspace.type_offsets[type + 1] = workspace.type_offsets[type] + workspace.type_cursors[type];
                workspace.type_cursors[type] = workspace.type_offsets[type];
            }
            for (std::size_t index = 0; index < neighbor_count; ++index) {
                const auto atom = neighbors.atoms[index];
                const std::size_t type = static_cast<std::size_t>(atom_types[static_cast<std::size_t>(atom - begin)]);
                const std::size_t destination = workspace.type_cursors[type]++;
                workspace.neighbor_x[destination] = neighbors.displacements[index * 3 + 0];
                workspace.neighbor_y[destination] = neighbors.displacements[index * 3 + 1];
                workspace.neighbor_z[destination] = neighbors.displacements[index * 3 + 2];
                workspace.neighbor_distance2[destination] = neighbors.distance2[index];
                workspace.neighbor_distance[destination] = std::sqrt(neighbors.distance2[index]);
                workspace.neighbor_self[destination] = neighbors.exact_self(index, center) ? 1 : 0;
            }
        }
        if (options.l_max == 3) {
            solid_harmonics_l3_soa(
                workspace.neighbor_x.data(), workspace.neighbor_y.data(), workspace.neighbor_z.data(),
                neighbor_count, workspace.neighbor_harmonics.data());
        } else if (options.l_max == 4) {
            solid_harmonics_l4_soa(
                workspace.neighbor_x.data(), workspace.neighbor_y.data(), workspace.neighbor_z.data(),
                neighbor_count, workspace.neighbor_harmonics.data());
        } else {
            for (std::size_t index = 0; index < neighbor_count; ++index) {
                const Vec3 displacement{
                    workspace.neighbor_x[index], workspace.neighbor_y[index], workspace.neighbor_z[index]};
                solid_harmonics(
                    displacement, options.l_max, plan.harmonic_normalization,
                    workspace.harmonics, workspace.legendre);
                for (std::size_t harmonic = 0; harmonic < harmonic_count; ++harmonic) {
                    workspace.neighbor_harmonics[harmonic * neighbor_count + index] = workspace.harmonics[harmonic];
                }
            }
        }
        if (options.compression == 1) {
            for (std::size_t index = 0; index < neighbor_count; ++index) {
                workspace.neighbor_weights[index] = soap_weight(
                    options, workspace.neighbor_distance[index],
                    workspace.neighbor_self[index] != 0,
                    static_cast<std::size_t>(workspace.neighbor_types[index]));
            }
        } else {
            for (std::size_t type = 0; type < options.species.size(); ++type) {
                const std::size_t type_begin = workspace.type_offsets[type];
                const std::size_t type_end = workspace.type_offsets[type + 1];
                for (std::size_t index = type_begin; index < type_end; ++index) {
                    workspace.neighbor_weights[index] = soap_weight(
                        options, workspace.neighbor_distance[index],
                        workspace.neighbor_self[index] != 0, type);
                }
            }
        }
        if (options.radial_basis == 1) {
            const std::size_t grid_size = options.radial_grid.size();
            const std::size_t integrand_stride = harmonic_count * grid_size;
            std::fill(workspace.polynomial_integrand.begin(), workspace.polynomial_integrand.end(), 0.0);
            for (std::size_t type = 0; type < coefficient_types; ++type) {
                const std::size_t type_begin = options.compression == 1 ? 0 : workspace.type_offsets[type];
                const std::size_t type_end = options.compression == 1
                    ? neighbor_count : workspace.type_offsets[type + 1];
                for (std::size_t index = type_begin; index < type_end; ++index) {
                    const double radius = workspace.neighbor_distance[index];
                    const double* flir = polynomial_flir(options, radius, workspace);
                    const double inverse_radius = radius <= 1e-14 ? 0.0 : 1.0 / radius;
                    double inverse_radius_power = 1.0;
                    for (int l = 0; l <= options.l_max; ++l) {
                        const double scale = radius <= 1e-14
                            ? (l == 0 ? 4.0 * kPi * workspace.neighbor_weights[index] : 0.0)
                            : 4.0 * kPi * inverse_radius_power * workspace.neighbor_weights[index];
                        const std::size_t l_offset = static_cast<std::size_t>(l * l);
                        const int m_count = 2 * l + 1;
                        const double* source = flir + static_cast<std::size_t>(l) * grid_size;
                        for (int m = 0; m < m_count; ++m) {
                            const double harmonic = workspace.neighbor_harmonics[
                                l_offset * neighbor_count
                                + static_cast<std::size_t>(m) * neighbor_count + index];
                            double* target = workspace.polynomial_integrand.data()
                                + type * integrand_stride
                                + (l_offset + static_cast<std::size_t>(m)) * grid_size;
                            const double factor = scale * harmonic;
#ifdef _OPENMP
#pragma omp simd
#endif
                            for (std::size_t q = 0; q < grid_size; ++q) {
                                target[q] += factor * source[q];
                            }
                        }
                        inverse_radius_power *= inverse_radius;
                    }
                }
            }
            for (std::size_t type = 0; type < coefficient_types; ++type) {
                for (int l = 0; l <= options.l_max; ++l) {
                    const std::size_t l_offset = static_cast<std::size_t>(l * l);
                    const int m_count = 2 * l + 1;
                    for (int n = 0; n < options.n_max; ++n) {
                        double* destination = coefficients.data()
                            + (type * options.n_max + static_cast<std::size_t>(n)) * harmonic_count
                            + l_offset;
                        for (int m = 0; m < m_count; ++m) {
                            double sum = 0.0;
                            const double* integrand = workspace.polynomial_integrand.data()
                                + type * integrand_stride
                                + (l_offset + static_cast<std::size_t>(m)) * grid_size;
                            const double* basis = plan.polynomial_weighted_values_qn.data()
                                + static_cast<std::size_t>(n);
#ifdef _OPENMP
#pragma omp simd reduction(+:sum)
#endif
                            for (std::size_t q = 0; q < grid_size; ++q) {
                                sum += integrand[q] * basis[q * static_cast<std::size_t>(options.n_max)];
                            }
                            destination[m] += sum;
                        }
                    }
                }
            }
        } else {
            // DScribe groups neighbors by species, accumulates each radial
            // channel over that group, and applies beta afterwards.
            const bool use_gto_cache = options.compression != 0 || options.inner_average;
            const std::size_t radial_count = static_cast<std::size_t>(options.l_max + 1) * options.n_max;
            if (use_gto_cache) {
                workspace.radial_values.resize(radial_count * neighbor_count);
                for (std::size_t index = 0; index < neighbor_count; ++index) {
                    const double* cached = gto_radial_values(
                        options, plan, workspace.neighbor_distance2[index], workspace);
                    for (std::size_t radial_index = 0; radial_index < radial_count; ++radial_index) {
                        workspace.radial_values[radial_index * neighbor_count + index] = cached[radial_index];
                    }
                }
            } else {
                workspace.radial_values.resize(neighbor_count);
            }
            for (std::size_t type = 0; type < coefficient_types; ++type) {
                const std::size_t type_begin = options.compression == 1 ? 0 : workspace.type_offsets[type];
                const std::size_t type_end = options.compression == 1
                    ? neighbor_count : workspace.type_offsets[type + 1];
                for (int l = 0; l <= options.l_max; ++l) {
                    const std::size_t l_offset = static_cast<std::size_t>(l * l);
                    const int m_count = 2 * l + 1;
                    for (int k = 0; k < options.n_max; ++k) {
                        const std::size_t radial_index = static_cast<std::size_t>(l * options.n_max + k);
                        std::fill(workspace.harmonic_sums.begin(), workspace.harmonic_sums.begin() + m_count, 0.0);
                        const double radial_prefactor = plan.radial_prefactors[radial_index];
                        const double radial_exponent = plan.radial_exponents[radial_index];
                        if (!use_gto_cache) {
                            for (std::size_t index = type_begin; index < type_end; ++index) {
                                workspace.radial_values[index] = radial_prefactor
                                    * std::exp(-radial_exponent * workspace.neighbor_distance2[index]);
                            }
                        }
                        const double* radial = use_gto_cache
                            ? workspace.radial_values.data() + radial_index * neighbor_count + type_begin
                            : workspace.radial_values.data() + type_begin;
                        for (int m = 0; m < m_count; ++m) {
                            double sum = 0.0;
                            const double* harmonics = workspace.neighbor_harmonics.data()
                                + l_offset * neighbor_count + static_cast<std::size_t>(m) * neighbor_count + type_begin;
#ifdef _OPENMP
#pragma omp simd reduction(+:sum)
#endif
                            for (std::size_t index = 0; index < type_end - type_begin; ++index) {
                                const std::size_t absolute = type_begin + index;
                                sum += radial[index] * harmonics[index] * workspace.neighbor_weights[absolute];
                            }
                            workspace.harmonic_sums[static_cast<std::size_t>(m)] = sum;
                        }
                        for (int n = 0; n < options.n_max; ++n) {
                            const double beta = options.betas[static_cast<std::size_t>((l * options.n_max + n) * options.n_max + k)];
                            double* destination = coefficients.data()
                                + (type * options.n_max + static_cast<std::size_t>(n)) * harmonic_count
                                + l_offset;
#ifdef _OPENMP
#pragma omp simd
#endif
                            for (int m = 0; m < m_count; ++m) {
                                destination[m] += plan.soap_scale * beta * workspace.harmonic_sums[static_cast<std::size_t>(m)];
                            }
                        }
                    }
                }
            }
        }
    };

    auto power_spectrum = [&](const std::vector<double>& coefficients, double* destination,
                              std::vector<double>* scratch = nullptr) {
        std::int64_t feature = 0;
        const std::size_t harmonic_count = static_cast<std::size_t>(width * width);
        auto dot_channels = [&](const double* first, const double* second, int l) {
            double sum = 0.0;
            for (int m = 0; m < 2 * l + 1; ++m) {
                sum += first[static_cast<std::size_t>(l * l + m)]
                    * second[static_cast<std::size_t>(l * l + m)];
            }
            return sum;
        };
        auto base = [&](std::size_t type, int n) {
            return (type * static_cast<std::size_t>(options.n_max) + static_cast<std::size_t>(n)) * harmonic_count;
        };
        if (options.compression == 1) {
            for (int l = 0; l <= options.l_max; ++l) {
                const double prefactor = plan.power_prefactors[static_cast<std::size_t>(l)];
                for (int n1 = 0; n1 < options.n_max; ++n1) {
                    for (int n2 = n1; n2 < options.n_max; ++n2) {
                        destination[feature++] = prefactor * dot_channels(
                            coefficients.data() + static_cast<std::size_t>(n1) * harmonic_count,
                            coefficients.data() + static_cast<std::size_t>(n2) * harmonic_count, l);
                    }
                }
            }
        } else if (options.compression == 2) {
            std::vector<double> local_summed;
            std::vector<double>& summed = scratch == nullptr ? local_summed : *scratch;
            summed.assign(static_cast<std::size_t>(options.n_max) * harmonic_count, 0.0);
            for (std::size_t type = 0; type < options.species.size(); ++type) {
                for (int n = 0; n < options.n_max; ++n) {
                    const std::size_t source = base(type, n);
                    const std::size_t target = static_cast<std::size_t>(n) * harmonic_count;
                    for (std::size_t i = 0; i < harmonic_count; ++i) {
                        summed[target + i] += coefficients[source + i];
                    }
                }
            }
            for (std::size_t type = 0; type < options.species.size(); ++type) {
                for (int l = 0; l <= options.l_max; ++l) {
                    const double prefactor = plan.power_prefactors[static_cast<std::size_t>(l)];
                    for (int n1 = 0; n1 < options.n_max; ++n1) {
                        for (int n2 = 0; n2 < options.n_max; ++n2) {
                            destination[feature++] = prefactor * dot_channels(
                                coefficients.data() + base(type, n1),
                                summed.data() + static_cast<std::size_t>(n2) * harmonic_count, l);
                        }
                    }
                }
            }
        } else if (options.compression == 3) {
            for (std::size_t type = 0; type < options.species.size(); ++type) {
                for (int l = 0; l <= options.l_max; ++l) {
                    const double prefactor = plan.power_prefactors[static_cast<std::size_t>(l)];
                    for (int n1 = 0; n1 < options.n_max; ++n1) {
                        for (int n2 = n1; n2 < options.n_max; ++n2) {
                            destination[feature++] = prefactor * dot_channels(
                                coefficients.data() + base(type, n1),
                                coefficients.data() + base(type, n2), l);
                        }
                    }
                }
            }
        } else {
            for (std::size_t type1 = 0; type1 < options.species.size(); ++type1) {
                for (std::size_t type2 = type1; type2 < options.species.size(); ++type2) {
                    for (int l = 0; l <= options.l_max; ++l) {
                        const double prefactor = plan.power_prefactors[static_cast<std::size_t>(l)];
                        for (int n1 = 0; n1 < options.n_max; ++n1) {
                            const int n2_begin = type1 == type2 ? n1 : 0;
                            for (int n2 = n2_begin; n2 < options.n_max; ++n2) {
                                destination[feature++] = prefactor * dot_channels(
                                    coefficients.data() + base(type1, n1),
                                    coefficients.data() + base(type2, n2), l);
                            }
                        }
                    }
                }
            }
        }
        if (feature != features) {
            throw std::logic_error("internal SOAP feature layout mismatch");
        }
    };

    if (options.inner_average) {
        std::fill(averaged_coefficients.begin(), averaged_coefficients.end(), 0.0);
        if (atom_count > 0) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options.num_threads > 0 ? options.num_threads : omp_get_max_threads()) if(parallel_centers)
#endif
            for (std::int64_t center = begin; center < end; ++center) {
                if (cancelled(control)) {
                    continue;
                }
                int workspace_index = 0;
#ifdef _OPENMP
                if (parallel_centers) {
                    workspace_index = omp_get_thread_num();
                }
#endif
                SoapWorkspace& workspace = workspace_at(workspace_index);
                calculate_coefficients(center, workspace);
                if (parallel_centers) {
                    std::vector<double>& partial = average_workspaces[static_cast<std::size_t>(workspace_index)];
                    for (std::size_t i = 0; i < coefficient_size; ++i) {
                        partial[i] += workspace.local_coefficients[i];
                    }
                } else {
                    for (std::size_t i = 0; i < coefficient_size; ++i) {
                        averaged_coefficients[i] += workspace.local_coefficients[i];
                    }
                }
            }
        }
        if (parallel_centers && atom_count > 0) {
            if (cancelled(control)) {
                throw CancelledError();
            }
            const double inverse_atom_count = 1.0 / static_cast<double>(atom_count);
            for (std::size_t i = 0; i < coefficient_size; ++i) {
                double sum = 0.0;
                for (const std::vector<double>& partial : average_workspaces) {
                    sum += partial[i];
                }
                averaged_coefficients[i] = sum * inverse_atom_count;
            }
        } else if (atom_count > 0) {
            const double inverse_atom_count = 1.0 / static_cast<double>(atom_count);
            for (double& value : averaged_coefficients) {
                value *= inverse_atom_count;
            }
        }
        power_spectrum(averaged_coefficients, output, &workspace_at(0).summed_coefficients);
    } else if (options.outer_average) {
        std::vector<double> averaged_power(static_cast<std::size_t>(features), 0.0);
        if (atom_count > 0) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options.num_threads > 0 ? options.num_threads : omp_get_max_threads()) if(parallel_centers)
#endif
            for (std::int64_t center = begin; center < end; ++center) {
                if (cancelled(control)) {
                    continue;
                }
                int workspace_index = 0;
#ifdef _OPENMP
                if (parallel_centers) {
                    workspace_index = omp_get_thread_num();
                }
#endif
                SoapWorkspace& workspace = workspace_at(workspace_index);
                calculate_coefficients(center, workspace);
                power_spectrum(
                    workspace.local_coefficients, workspace.power_values.data(),
                    &workspace.summed_coefficients);
                if (parallel_centers) {
                    std::vector<double>& partial = average_power_workspaces[
                        static_cast<std::size_t>(workspace_index)];
                    for (std::size_t i = 0; i < static_cast<std::size_t>(features); ++i) {
                        partial[i] += workspace.power_values[i];
                    }
                } else {
                    for (std::size_t i = 0; i < static_cast<std::size_t>(features); ++i) {
                        averaged_power[i] += workspace.power_values[i];
                    }
                }
            }
        }
        if (parallel_centers && atom_count > 0) {
            if (cancelled(control)) {
                throw CancelledError();
            }
            for (std::size_t i = 0; i < static_cast<std::size_t>(features); ++i) {
                double sum = 0.0;
                for (const std::vector<double>& partial : average_power_workspaces) {
                    sum += partial[i];
                }
                output[i] = sum / static_cast<double>(atom_count);
            }
        } else if (atom_count > 0) {
            const double inverse_atom_count = 1.0 / static_cast<double>(atom_count);
            for (std::size_t i = 0; i < static_cast<std::size_t>(features); ++i) {
                output[i] = averaged_power[i] * inverse_atom_count;
            }
        } else {
            std::fill(output, output + features, 0.0);
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options.num_threads > 0 ? options.num_threads : omp_get_max_threads()) if(parallel_centers)
#endif
        for (std::int64_t center = begin; center < end; ++center) {
            if (cancelled(control)) {
                continue;
            }
            int workspace_index = 0;
#ifdef _OPENMP
            if (parallel_centers) {
                workspace_index = omp_get_thread_num();
            }
#endif
            SoapWorkspace& workspace = workspace_at(workspace_index);
            calculate_coefficients(center, workspace);
            power_spectrum(
                workspace.local_coefficients,
                output + (center - begin) * features,
                &workspace.summed_coefficients);
        }
    }
    if (parallel_centers && cancelled(control)) {
        throw CancelledError();
    }
}
} // namespace

void compute_soap(const StructureBatchView& batch, const SoapOptions& options, double* output, const std::shared_ptr<ComputeControl>& control) {
    validate_common(batch);
    validate_species(batch, options.species);
    if (!std::isfinite(options.r_cut) || !std::isfinite(options.sigma)
        || options.r_cut <= 0.0 || options.sigma <= 0.0 || options.n_max < 1 || options.l_max < 0
        || options.l_max > 20 || options.num_threads < 0 || options.radial_basis < 0 || options.radial_basis > 1
        || options.compression < 0 || options.compression > 3
        || options.species_weights.size() != options.species.size()) {
        throw std::invalid_argument("invalid SOAP parameters");
    }
    if (options.radial_basis == 0) {
        if (options.alphas.size() != static_cast<std::size_t>((options.l_max + 1) * options.n_max)
            || options.betas.size() != static_cast<std::size_t>((options.l_max + 1) * options.n_max * options.n_max)) {
            throw std::invalid_argument("invalid SOAP radial basis size");
        }
    } else if (options.radial_grid.size() < 2
        || options.radial_weights.size() != options.radial_grid.size()
        || options.radial_values.size() != static_cast<std::size_t>(options.n_max) * options.radial_grid.size()) {
        throw std::invalid_argument("invalid SOAP polynomial radial basis size");
    }
    for (double value : options.alphas) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("SOAP radial basis must be finite");
        }
    }
    for (double value : options.betas) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("SOAP radial basis must be finite");
        }
    }
    for (double value : options.radial_grid) {
        if (!std::isfinite(value) || value < 0.0 || value > options.r_cut) {
            throw std::invalid_argument("SOAP polynomial radial grid must be finite and in range");
        }
    }
    for (double value : options.radial_weights) {
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::invalid_argument("SOAP polynomial radial weights must be positive and finite");
        }
    }
    for (double value : options.radial_values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("SOAP polynomial radial basis must be finite");
        }
    }
    for (double value : options.species_weights) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("SOAP species weights must be finite");
        }
    }
    if (options.weighting_has_function) {
        if (options.weighting_function < 1 || options.weighting_function > 3
            || !std::isfinite(options.weighting_r0) || options.weighting_r0 <= 0.0
            || !std::isfinite(options.weighting_c) || options.weighting_c < 0.0
            || !std::isfinite(options.weighting_d) || options.weighting_d < 0.0
            || !std::isfinite(options.weighting_m) || options.weighting_m < 0.0) {
            throw std::invalid_argument("invalid SOAP weighting parameters");
        }
    }
    if (options.weighting_has_w0 && (!std::isfinite(options.weighting_w0) || options.weighting_w0 < 0.0)) {
        throw std::invalid_argument("invalid SOAP w0 weighting");
    }
    if (control) {
        control->reset(batch.structures);
    }
    const double padding = options.sigma * std::sqrt(-2.0 * std::log(1e-3));
    const SoapPlan plan = prepare_soap(options);
    const bool structure_average = options.inner_average || options.outer_average;
    const std::int64_t stride = structure_average ? soap_features(options) : 0;
    if (batch.structures == 1) {
        const NeighborGraph neighbor_graph = build_neighbor_graph(
            batch, options.r_cut + padding, control, options.num_threads);
        compute_soap_structure(batch, options, neighbor_graph, 0, output, control, true, plan);
        if (control && control->cancelled()) {
            throw CancelledError();
        }
        if (control) {
            control->mark_completed();
        }
        return;
    }

    // For many small structures, building one global graph creates a large
    // intermediate allocation and then copies every local graph into it.
    // Build and consume each structure's graph in the same worker instead.
    const double structure_cutoff = options.r_cut + padding;
    run_parallel_structures(batch.structures, options.num_threads, control, [&](std::int64_t s) {
        const std::int64_t begin = batch.offsets[s];
        const std::int64_t end = batch.offsets[s + 1];
        const std::int64_t offsets[2] = {0, end - begin};
        const StructureBatchView structure_batch{
            batch.numbers + begin,
            batch.positions + begin * 3,
            batch.cells + s * 9,
            batch.pbc + s * 3,
            offsets,
            1,
            end - begin,
        };
        const NeighborGraph structure_graph = build_neighbor_graph(
            structure_batch, structure_cutoff, control, 1);
        const std::int64_t out_row = structure_average ? s : batch.offsets[s];
        compute_soap_structure(
            structure_batch, options, structure_graph, 0,
            output + out_row * (structure_average ? stride : soap_features(options)),
            control, false, plan);
    });
}

SoapCalculator::SoapCalculator(SoapOptions options) : options_(std::move(options)) {}
std::int64_t SoapCalculator::feature_count() const noexcept { return soap_features(options_); }
const std::vector<std::int32_t>& SoapCalculator::species() const noexcept { return options_.species; }
void SoapCalculator::close() noexcept { closed_.store(true, std::memory_order_release); }
bool SoapCalculator::closed() const noexcept { return closed_.load(std::memory_order_acquire); }
void SoapCalculator::compute(const StructureBatchView& batch, double* output, const std::shared_ptr<ComputeControl>& control) const {
    std::lock_guard<std::mutex> lock(compute_mutex_);
    if (closed()) {
        throw std::runtime_error("SOAP calculator is closed");
    }
    compute_soap(batch, options_, output, control);
}
} // namespace mdescriptor
