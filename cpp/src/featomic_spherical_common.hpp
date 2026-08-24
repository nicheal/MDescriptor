#pragma once

#include "featomic_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mdescriptor::detail {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kSqrt2 = 1.414213562373095048801688724209698079;

inline double gamma_value(double x) {
    const double value = std::tgamma(x);
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::invalid_argument("Featomic radial basis is numerically singular");
    }
    return value;
}

inline double hyp1f1_series(double a, double b, double x) {
    double sum = 1.0;
    double term = 1.0;
    for (int k = 1; k < 10000; ++k) {
        term *= (a + k - 1.0) * x / ((b + k - 1.0) * k);
        sum += term;
        if (std::abs(term) <= std::abs(sum) * 2e-15) {
            break;
        }
        if (!std::isfinite(sum)) {
            throw std::invalid_argument("Featomic radial integral overflow");
        }
    }
    return sum;
}

// Direct summation of M(a,b,-x) loses digits through alternating cancellation
// at the reciprocal-space x values used by LODE. The following ODE evaluates
// the same function without that cancellation. It is only used for negative
// arguments; the ordinary SOAP path retains its fast series/asymptotic split.
inline double hyp1f1_negative_ode(double a, double b, double x) {
    constexpr double start = 0.01;
    constexpr double absolute_tolerance = 2e-15;
    constexpr double relative_tolerance = 2e-13;
    double y = 1.0;
    double derivative = 0.0;
    double term = 1.0;
    for (int k = 1; k < 80; ++k) {
        term *= -start * (a + k - 1.0) / ((b + k - 1.0) * k);
        y += term;
        derivative += k * term / start;
        if (std::abs(term) <= std::abs(y) * 2e-18) {
            break;
        }
    }
    if (x <= start) {
        return y;
    }

    auto derivative_at = [a, b](double position, double value, double slope) {
        return std::array<double, 2>{slope, -((b + position) * slope + a * value) / position};
    };
    double position = start;
    double step = std::min(0.25, x - position);
    while (position < x) {
        step = std::min(step, x - position);
        const auto k1 = derivative_at(position, y, derivative);
        const auto k2 = derivative_at(
            position + step * (1.0 / 5.0),
            y + step * (1.0 / 5.0) * k1[0],
            derivative + step * (1.0 / 5.0) * k1[1]);
        const auto k3 = derivative_at(
            position + step * (3.0 / 10.0),
            y + step * (3.0 / 40.0 * k1[0] + 9.0 / 40.0 * k2[0]),
            derivative + step * (3.0 / 40.0 * k1[1] + 9.0 / 40.0 * k2[1]));
        const auto k4 = derivative_at(
            position + step * (4.0 / 5.0),
            y + step * (44.0 / 45.0 * k1[0] - 56.0 / 15.0 * k2[0] + 32.0 / 9.0 * k3[0]),
            derivative + step * (44.0 / 45.0 * k1[1] - 56.0 / 15.0 * k2[1] + 32.0 / 9.0 * k3[1]));
        const auto k5 = derivative_at(
            position + step * (8.0 / 9.0),
            y + step * (19372.0 / 6561.0 * k1[0] - 25360.0 / 2187.0 * k2[0]
                + 64448.0 / 6561.0 * k3[0] - 212.0 / 729.0 * k4[0]),
            derivative + step * (19372.0 / 6561.0 * k1[1] - 25360.0 / 2187.0 * k2[1]
                + 64448.0 / 6561.0 * k3[1] - 212.0 / 729.0 * k4[1]));
        const auto k6 = derivative_at(
            position + step,
            y + step * (9017.0 / 3168.0 * k1[0] - 355.0 / 33.0 * k2[0]
                + 46732.0 / 5247.0 * k3[0] + 49.0 / 176.0 * k4[0] - 5103.0 / 18656.0 * k5[0]),
            derivative + step * (9017.0 / 3168.0 * k1[1] - 355.0 / 33.0 * k2[1]
                + 46732.0 / 5247.0 * k3[1] + 49.0 / 176.0 * k4[1] - 5103.0 / 18656.0 * k5[1]));
        const auto k7 = derivative_at(
            position + step,
            y + step * (35.0 / 384.0 * k1[0] + 500.0 / 1113.0 * k3[0]
                + 125.0 / 192.0 * k4[0] - 2187.0 / 6784.0 * k5[0] + 11.0 / 84.0 * k6[0]),
            derivative + step * (35.0 / 384.0 * k1[1] + 500.0 / 1113.0 * k3[1]
                + 125.0 / 192.0 * k4[1] - 2187.0 / 6784.0 * k5[1] + 11.0 / 84.0 * k6[1]));

        const double next_y = y + step * (35.0 / 384.0 * k1[0] + 500.0 / 1113.0 * k3[0]
            + 125.0 / 192.0 * k4[0] - 2187.0 / 6784.0 * k5[0] + 11.0 / 84.0 * k6[0]);
        const double next_derivative = derivative + step * (35.0 / 384.0 * k1[1]
            + 500.0 / 1113.0 * k3[1] + 125.0 / 192.0 * k4[1]
            - 2187.0 / 6784.0 * k5[1] + 11.0 / 84.0 * k6[1]);
        const double fourth_y = y + step * (5179.0 / 57600.0 * k1[0] + 7571.0 / 16695.0 * k3[0]
            + 393.0 / 640.0 * k4[0] - 92097.0 / 339200.0 * k5[0]
            + 187.0 / 2100.0 * k6[0] + 1.0 / 40.0 * k7[0]);
        const double fourth_derivative = derivative + step * (5179.0 / 57600.0 * k1[1]
            + 7571.0 / 16695.0 * k3[1] + 393.0 / 640.0 * k4[1]
            - 92097.0 / 339200.0 * k5[1] + 187.0 / 2100.0 * k6[1] + 1.0 / 40.0 * k7[1]);
        const double error = std::max(
            std::abs(next_y - fourth_y) / (absolute_tolerance + relative_tolerance * std::abs(next_y)),
            std::abs(next_derivative - fourth_derivative)
                / (absolute_tolerance + relative_tolerance * std::abs(next_derivative)));
        if (error <= 1.0 || step <= 1e-9) {
            position += step;
            y = next_y;
            derivative = next_derivative;
            const double scale = error == 0.0 ? 5.0 : std::min(5.0, std::max(0.2, 0.9 * std::pow(error, -0.2)));
            step *= scale;
        } else {
            step *= std::max(0.1, 0.9 * std::pow(error, -0.2));
        }
    }
    return y;
}

inline double hyp1f1(double a, double b, double x) {
    if (x < 0.0) {
        const double magnitude = -x;
        const double difference = b - a;
        const double rounded = std::round(difference);
        if (difference <= 0.0 && std::abs(difference - rounded) <= 1e-12) {
            const int degree = static_cast<int>(-rounded);
            double polynomial = 1.0;
            double term = 1.0;
            for (int k = 1; k <= degree; ++k) {
                term *= (-degree + k - 1.0) * magnitude / ((b + k - 1.0) * k);
                polynomial += term;
            }
            return std::exp(-magnitude) * polynomial;
        }
        if (magnitude > 8.0) {
            // Kummer's transformation followed by the CHGM asymptotic
            // branches avoids the adaptive ODE used by the first LODE
            // implementation. The latter was accurate but very expensive:
            // it integrated seven Runge--Kutta stages for every radial
            // channel and reciprocal vector.
            double transformed = b - a;
            const double transformed_original = transformed;
            double positive_x = magnitude;
            int recurrence_count = 0;
            int recurrence_start = 0;
            if (transformed >= 2.0) {
                recurrence_count = 1;
                recurrence_start = static_cast<int>(transformed);
                transformed -= static_cast<double>(recurrence_start) + 1.0;
            }

            double y0 = 0.0;
            double y1 = 0.0;
            double result = 0.0;
            for (int branch = 0; branch <= recurrence_count; ++branch) {
                if (transformed_original >= 2.0) {
                    transformed += 1.0;
                }
                if (positive_x <= std::abs(b) + 30.0 || transformed < 0.0) {
                    double term = 1.0;
                    result = 1.0;
                    for (int j = 1; j <= 500; ++j) {
                        term *= (transformed + j - 1.0) * positive_x
                            / (j * (b + j - 1.0));
                        result += term;
                        if (result != 0.0 && std::abs(term / result) < 1e-15) {
                            break;
                        }
                    }
                    result *= std::exp(-positive_x);
                } else {
                    double sum_1 = 1.0;
                    double sum_2 = 1.0;
                    double term_1 = 1.0;
                    double term_2 = 1.0;
                    for (int i = 1; i <= 30; ++i) {
                        term_1 = -term_1 * (transformed + i - 1.0)
                            * (transformed - b + i) / (positive_x * i);
                        term_2 = -term_2 * (b - transformed + i - 1.0)
                            * (transformed - i) / (positive_x * i);
                        sum_1 += term_1;
                        sum_2 += term_2;
                    }
                    const double first = std::exp(
                        std::lgamma(b) - std::lgamma(b - transformed) - positive_x)
                        * std::pow(positive_x, -transformed)
                        * std::cos(kPi * transformed) * sum_1;
                    const double second = std::exp(
                        std::lgamma(b) - std::lgamma(transformed))
                        * std::pow(positive_x, transformed - b) * sum_2;
                    result = first + second;
                }
                if (branch == 0) {
                    y0 = result;
                } else {
                    y1 = result;
                }
            }
            if (transformed_original >= 2.0) {
                for (int step = 1; step < recurrence_start; ++step) {
                    result = ((transformed * 2.0 - b + positive_x) * y1
                        + (b - transformed) * y0) / transformed;
                    y0 = y1;
                    y1 = result;
                    transformed += 1.0;
                }
            }
            return result;
        }
    }
    return hyp1f1_series(a, b, x);
}

// The direct series loses most significant digits for positive x once the
// exponentially growing branch dominates. This is the same large-x branch as
// Featomic's CHGM implementation, reduced to the regularized combination used
// by the GTO radial integral. It avoids forming Gamma(a)/Gamma(b) * exp(x)
// separately and therefore remains stable for narrow densities.
inline double regularized_hyp1f1_positive(double a, double b, double x) {
    double sum = 1.0;
    double term = 1.0;
    for (int index = 1; index <= 30; ++index) {
        term = -term * (b - a + index - 1.0) * (a - index) / (x * index);
        sum += term;
    }
    return sum;
}

struct SymmetricMatrix {
    int size = 0;
    std::vector<double> values;

    double& at(int row, int column) { return values[static_cast<std::size_t>(row * size + column)]; }
    double at(int row, int column) const { return values[static_cast<std::size_t>(row * size + column)]; }
};

// Small Jacobi eigensolver used only to build the orthonormal GTO radial basis.
// ponytail: this O(n^3) setup is amortized across all atoms; replace with a
// cached eigensolver only if calculators are constructed at high frequency.
inline SymmetricMatrix inverse_sqrt(const SymmetricMatrix& input) {
    const int n = input.size;
    SymmetricMatrix matrix{n, input.values};
    SymmetricMatrix eigenvectors{n, std::vector<double>(static_cast<std::size_t>(n * n), 0.0)};
    for (int i = 0; i < n; ++i) {
        eigenvectors.at(i, i) = 1.0;
    }
    for (int iteration = 0; iteration < 50 * n * n; ++iteration) {
        int p = 0;
        int q = 1;
        double maximum = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (std::abs(matrix.at(i, j)) > maximum) {
                    maximum = std::abs(matrix.at(i, j));
                    p = i;
                    q = j;
                }
            }
        }
        if (maximum <= 1e-14) {
            break;
        }
        const double phi = 0.5 * std::atan2(2.0 * matrix.at(p, q), matrix.at(q, q) - matrix.at(p, p));
        const double cosine = std::cos(phi);
        const double sine = std::sin(phi);
        const double app = matrix.at(p, p);
        const double aqq = matrix.at(q, q);
        const double apq = matrix.at(p, q);
        matrix.at(p, p) = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
        matrix.at(q, q) = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
        matrix.at(p, q) = matrix.at(q, p) = 0.0;
        for (int k = 0; k < n; ++k) {
            if (k == p || k == q) {
                continue;
            }
            const double akp = matrix.at(k, p);
            const double akq = matrix.at(k, q);
            matrix.at(k, p) = matrix.at(p, k) = cosine * akp - sine * akq;
            matrix.at(k, q) = matrix.at(q, k) = sine * akp + cosine * akq;
        }
        for (int k = 0; k < n; ++k) {
            const double vkp = eigenvectors.at(k, p);
            const double vkq = eigenvectors.at(k, q);
            eigenvectors.at(k, p) = cosine * vkp - sine * vkq;
            eigenvectors.at(k, q) = sine * vkp + cosine * vkq;
        }
    }

    std::vector<int> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int left, int right) {
        return matrix.at(left, left) < matrix.at(right, right);
    });
    SymmetricMatrix result{n, std::vector<double>(static_cast<std::size_t>(n * n), 0.0)};
    for (int row = 0; row < n; ++row) {
        for (int column = 0; column < n; ++column) {
            double value = 0.0;
            for (int eigen = 0; eigen < n; ++eigen) {
                const double eigenvalue = matrix.at(order[static_cast<std::size_t>(eigen)], order[static_cast<std::size_t>(eigen)]);
                if (eigenvalue <= std::numeric_limits<double>::epsilon()) {
                    throw std::invalid_argument("radial overlap matrix is singular, lower max_radial");
                }
                value += eigenvectors.at(row, order[static_cast<std::size_t>(eigen)])
                    * eigenvectors.at(column, order[static_cast<std::size_t>(eigen)]) / std::sqrt(eigenvalue);
            }
            result.at(row, column) = value;
        }
    }
    return result;
}

struct GtoRadialBasis {
    int size = 0;
    double radius = 0.0;
    int angular_channel = 0;
    std::vector<double> widths;
    std::vector<double> gto_constants;
    std::vector<double> gamma_a;
    std::vector<double> lode_prefactors;
    double gamma_b = 0.0;
    std::vector<std::vector<double>> orthonormalization;

    GtoRadialBasis(int size_, double radius_, int angular_ = 0)
        : size(size_), radius(radius_), angular_channel(angular_) {
        widths.resize(static_cast<std::size_t>(size));
        gto_constants.resize(static_cast<std::size_t>(size));
        gamma_a.resize(static_cast<std::size_t>(size));
        lode_prefactors.resize(static_cast<std::size_t>(size));
        std::vector<double> normalization(static_cast<std::size_t>(size));
        for (int n = 0; n < size; ++n) {
            const double index = static_cast<double>(n);
            widths[static_cast<std::size_t>(n)] = radius * std::max(std::sqrt(index), 1.0) / size;
            gto_constants[static_cast<std::size_t>(n)] = 1.0 / (2.0 * widths[static_cast<std::size_t>(n)] * widths[static_cast<std::size_t>(n)]);
            gamma_a[static_cast<std::size_t>(n)] = gamma_value(0.5 * (n + angular_channel + 3.0));
            lode_prefactors[static_cast<std::size_t>(n)] = std::pow(widths[static_cast<std::size_t>(n)], n + 3.0)
                * std::pow(kSqrt2, n);
            normalization[static_cast<std::size_t>(n)] = std::sqrt(
                2.0 / (std::pow(widths[static_cast<std::size_t>(n)], 2 * n + 3) * gamma_value(n + 1.5)));
        }
        gamma_b = gamma_value(angular_channel + 1.5);
        SymmetricMatrix overlap{size, std::vector<double>(static_cast<std::size_t>(size * size), 0.0)};
        for (int n1 = 0; n1 < size; ++n1) {
            for (int n2 = n1; n2 < size; ++n2) {
                const double sigma1 = widths[static_cast<std::size_t>(n1)];
                const double sigma2 = widths[static_cast<std::size_t>(n2)];
                const double exponent = 0.5 * (3.0 + n1 + n2);
                const double value = std::pow(0.5 / (sigma1 * sigma1) + 0.5 / (sigma2 * sigma2), -exponent)
                    / (std::pow(sigma1, n1) * std::pow(sigma2, n2))
                    * gamma_value(exponent)
                    / (std::pow(sigma1 * sigma2, 1.5)
                       * std::sqrt(gamma_value(n1 + 1.5) * gamma_value(n2 + 1.5)));
                overlap.at(n1, n2) = overlap.at(n2, n1) = value;
            }
        }
        const SymmetricMatrix inverse_overlap = inverse_sqrt(overlap);
        orthonormalization.assign(static_cast<std::size_t>(size), std::vector<double>(static_cast<std::size_t>(size), 0.0));
        for (int n = 0; n < size; ++n) {
            for (int target = 0; target < size; ++target) {
                orthonormalization[static_cast<std::size_t>(n)][static_cast<std::size_t>(target)]
                    = normalization[static_cast<std::size_t>(n)] * inverse_overlap.at(n, target);
            }
        }
    }

    void radial_integral_into(double distance, int angular, double density_width, std::vector<double>& result) const {
        std::vector<double> raw;
        radial_integral_into(distance, angular, density_width, result, raw);
    }

    void radial_integral_into(
        double distance, int angular, double density_width,
        std::vector<double>& result, std::vector<double>& raw) const {
        result.assign(static_cast<std::size_t>(size), 0.0);
        raw.assign(static_cast<std::size_t>(size), 0.0);
        const double density_width2 = density_width * density_width;
        const double density_constant = 1.0 / (2.0 * density_width2);
        const double global_factor = std::pow(kPi / density_width2, 0.75);
        const double c_r = density_constant * distance;
        const double factor = global_factor * std::exp(-distance * c_r) * std::pow(c_r, angular);
        for (int n = 0; n < size; ++n) {
            const double gto_constant = gto_constants[static_cast<std::size_t>(n)];
            const double z = c_r * c_r / (density_constant + gto_constant);
            const double a = 0.5 * (n + angular + 3.0);
            const double b = angular + 1.5;
            if (z > 30.0) {
                const double asymptotic = regularized_hyp1f1_positive(a, b, z);
                const double logarithm = std::log(global_factor) - distance * c_r
                    + static_cast<double>(angular) * std::log(c_r)
                    - a * std::log(density_constant + gto_constant)
                    + z + (a - b) * std::log(z);
                raw[static_cast<std::size_t>(n)] = std::exp(logarithm) * asymptotic;
            } else {
                const double gamma_a_value = angular == angular_channel
                    ? gamma_a[static_cast<std::size_t>(n)] : gamma_value(a);
                const double gamma_b_value = angular == angular_channel ? gamma_b : gamma_value(b);
                raw[static_cast<std::size_t>(n)] = gamma_a_value / gamma_b_value * hyp1f1(a, b, z)
                    * std::pow(density_constant + gto_constant, -a) * factor;
            }
        }
        for (int target = 0; target < size; ++target) {
            for (int n = 0; n < size; ++n) {
                result[static_cast<std::size_t>(target)] += raw[static_cast<std::size_t>(n)]
                    * orthonormalization[static_cast<std::size_t>(n)][static_cast<std::size_t>(target)];
            }
        }
    }

    std::vector<double> radial_integral(double distance, int angular, double density_width) const {
        std::vector<double> result;
        radial_integral_into(distance, angular, density_width, result);
        return result;
    }

    void lode_radial_integral_into(double k_norm, int angular, std::vector<double>& result) const {
        std::vector<double> raw;
        lode_radial_integral_into(k_norm, angular, result, raw);
    }

    void lode_radial_integral_into(
        double k_norm, int angular, std::vector<double>& result,
        std::vector<double>& raw) const {
        result.assign(static_cast<std::size_t>(size), 0.0);
        raw.assign(static_cast<std::size_t>(size), 0.0);
        const double global_factor = std::sqrt(kPi) / kSqrt2;
        for (int n = 0; n < size; ++n) {
            const double sigma = widths[static_cast<std::size_t>(n)];
            const double k_sigma_sqrt2 = k_norm * sigma / kSqrt2;
            double angular_factor = 1.0;
            if (angular == 1) {
                angular_factor = k_sigma_sqrt2;
            } else if (angular == 2) {
                angular_factor = k_sigma_sqrt2 * k_sigma_sqrt2;
            } else if (angular == 3) {
                angular_factor = k_sigma_sqrt2 * k_sigma_sqrt2 * k_sigma_sqrt2;
            } else if (angular > 3) {
                angular_factor = std::pow(k_sigma_sqrt2, angular);
            }
            const double factor = global_factor * lode_prefactors[static_cast<std::size_t>(n)] * angular_factor;
            const double z = -0.5 * k_norm * k_norm * sigma * sigma;
            const double a = 0.5 * (n + angular + 3.0);
            const double b = angular + 1.5;
            const double gamma_a_value = angular == angular_channel
                ? gamma_a[static_cast<std::size_t>(n)] : gamma_value(a);
            const double gamma_b_value = angular == angular_channel ? gamma_b : gamma_value(b);
            raw[static_cast<std::size_t>(n)] = gamma_a_value / gamma_b_value * hyp1f1(a, b, z) * factor;
        }
        for (int target = 0; target < size; ++target) {
            for (int n = 0; n < size; ++n) {
                result[static_cast<std::size_t>(target)] += raw[static_cast<std::size_t>(n)]
                    * orthonormalization[static_cast<std::size_t>(n)][static_cast<std::size_t>(target)];
            }
        }
    }

    std::vector<double> lode_radial_integral(double k_norm, int angular) const {
        std::vector<double> result;
        lode_radial_integral_into(k_norm, angular, result);
        return result;
    }
};

inline void real_spherical_harmonics_into(
    const std::array<double, 3>& vector,
    int max_angular,
    std::vector<double>& output,
    std::vector<double>& legendre) {
    output.assign(static_cast<std::size_t>((max_angular + 1) * (max_angular + 1)), 0.0);
    const double norm = std::sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
    std::array<double, 3> direction = vector;
    if (norm < 1e-6) {
        direction = {0.0, 0.0, 1.0};
    } else {
        for (double& value : direction) {
            value /= norm;
        }
    }
    legendre.assign(static_cast<std::size_t>((max_angular + 1) * (max_angular + 2) / 2), 0.0);
    auto legendre_at = [max_angular, &legendre](int l, int m) -> double& {
        (void)max_angular;
        return legendre[static_cast<std::size_t>(m + l * (l + 1) / 2)];
    };
    constexpr double sqrt_1_over_2pi = 0.398942280401432677939946059934;
    constexpr double sqrt_3 = 1.732050807568877293527446341505872;
    constexpr double sqrt_3_over_2 = 1.224744871391589049098642;
    const double xy = std::hypot(direction[0], direction[1]);
    const double cos_theta = direction[2];
    const double sin_theta = xy;
    legendre_at(0, 0) = sqrt_1_over_2pi;
    if (max_angular > 0) {
        legendre_at(1, 0) = cos_theta * sqrt_3 * sqrt_1_over_2pi;
        double value = -sqrt_3_over_2 * sin_theta * sqrt_1_over_2pi;
        legendre_at(1, 1) = value;
        for (int l = 2; l <= max_angular; ++l) {
            for (int m = 0; m < l - 1; ++m) {
                const double ls = static_cast<double>(l * l);
                const double lm1s = static_cast<double>((l - 1) * (l - 1));
                const double ms = static_cast<double>(m * m);
                const double a = std::sqrt((4.0 * ls - 1.0) / (ls - ms));
                const double b = -std::sqrt((lm1s - ms) / (4.0 * lm1s - 1.0));
                legendre_at(l, m) = a * (cos_theta * legendre_at(l - 1, m) + b * legendre_at(l - 2, m));
            }
            legendre_at(l, l - 1) = cos_theta * std::sqrt(2.0 * l + 1.0) * value;
            value *= -std::sqrt(1.0 + 0.5 / l) * sin_theta;
            legendre_at(l, l) = value;
        }
    }
    for (int l = 0; l <= max_angular; ++l) {
        output[static_cast<std::size_t>(l * l + l)] = legendre_at(l, 0) / kSqrt2;
    }
    const double cos_phi = xy > std::numeric_limits<double>::epsilon() ? direction[0] / xy : 1.0;
    const double sin_phi = xy > std::numeric_limits<double>::epsilon() ? direction[1] / xy : 0.0;
    double cos_previous = 1.0;
    double sin_previous = 0.0;
    double cos_current = -cos_phi;
    double sin_current = sin_phi;
    const double minus_two_cos = -2.0 * cos_phi;
    for (int m = 1; m <= max_angular; ++m) {
        const double sin_m = minus_two_cos * sin_previous - sin_current;
        const double cos_m = minus_two_cos * cos_previous - cos_current;
        sin_current = sin_previous;
        sin_previous = sin_m;
        cos_current = cos_previous;
        cos_previous = cos_m;
        for (int l = m; l <= max_angular; ++l) {
            output[static_cast<std::size_t>(l * l + l + m)] = legendre_at(l, m) * cos_m;
            output[static_cast<std::size_t>(l * l + l - m)] = legendre_at(l, m) * sin_m;
        }
    }
}

inline void real_spherical_harmonics(const std::array<double, 3>& vector, int max_angular, std::vector<double>& output) {
    std::vector<double> legendre;
    real_spherical_harmonics_into(vector, max_angular, output, legendre);
}

inline double cutoff_value(double distance, double cutoff) {
    if (distance >= cutoff) {
        return 0.0;
    }
    const double width = std::min(0.5, cutoff);
    if (distance <= cutoff - width) {
        return 1.0;
    }
    return 0.5 * (1.0 + std::cos(kPi * (distance - cutoff + width) / width));
}

inline std::size_t coefficient_index(std::size_t species, int radial, int angular, int m, int n_radial, int max_angular) {
    return ((species * static_cast<std::size_t>(n_radial) + static_cast<std::size_t>(radial))
        * static_cast<std::size_t>(max_angular + 1) + static_cast<std::size_t>(angular))
        * static_cast<std::size_t>(2 * max_angular + 1) + static_cast<std::size_t>(max_angular + m);
}

inline void compute_coefficients_into(
    const StructureBatchView& batch,
    const NeighborGraph& graph,
    std::int64_t center,
    const FeatomicOptions& options,
    const std::vector<std::int32_t>& atom_types,
    const std::vector<GtoRadialBasis>& radial_bases,
    bool lode,
    std::vector<double>& coefficients,
    std::vector<double>& harmonics,
    std::vector<double>& legendre,
    std::vector<double>& radial,
    std::vector<double>& radial_raw) {
    const int n_radial = options.max_radial + 1;
    const int width = 2 * options.max_angular + 1;
    const std::size_t size = options.species.size() * static_cast<std::size_t>(n_radial)
        * static_cast<std::size_t>(options.max_angular + 1) * static_cast<std::size_t>(width);
    coefficients.assign(size, 0.0);
    radial.resize(static_cast<std::size_t>(n_radial));
    radial_raw.resize(static_cast<std::size_t>(n_radial));
    const NeighborView neighbors = graph.for_center(center);
    for (std::size_t index = 0; index < neighbors.size; ++index) {
        const auto type = static_cast<std::size_t>(atom_types[static_cast<std::size_t>(neighbors.atoms[index])]);
        const double distance = std::sqrt(std::max(0.0, neighbors.distance2[index]));
        const double scaling = cutoff_value(distance, options.cutoff);
        if (scaling == 0.0) {
            continue;
        }
        std::array<double, 3> displacement{
            neighbors.displacements[index * 3 + 0],
            neighbors.displacements[index * 3 + 1],
            neighbors.displacements[index * 3 + 2],
        };
        real_spherical_harmonics_into(displacement, options.max_angular, harmonics, legendre);
        for (int l = 0; l <= options.max_angular; ++l) {
            if (lode) {
                const double density = 1.0 / std::max(
                    std::pow(distance * distance + options.density_width * options.density_width, 0.5 * options.exponent),
                    1e-12);
                for (int n = 0; n < n_radial; ++n) {
                    radial[static_cast<std::size_t>(n)] = density * scaling
                        * std::exp(-(n + 1.0) * distance * distance / (options.radial_radius * options.radial_radius));
                }
            } else {
                radial_bases[static_cast<std::size_t>(l)].radial_integral_into(
                    distance, l, options.density_width, radial, radial_raw);
                for (double& value : radial) {
                    value *= scaling;
                }
            }
            for (int n = 0; n < n_radial; ++n) {
                for (int m = -l; m <= l; ++m) {
                    coefficients[coefficient_index(type, n, l, m, n_radial, options.max_angular)]
                        += radial[static_cast<std::size_t>(n)]
                        * harmonics[static_cast<std::size_t>(l * l + l + m)];
                }
            }
        }
    }
}

inline std::int64_t local_feature_count(const FeatomicOptions& options, FeatomicKind kind) {
    const std::int64_t species = static_cast<std::int64_t>(options.species.size());
    const std::int64_t radial = static_cast<std::int64_t>(options.max_radial + 1);
    const std::int64_t angular = static_cast<std::int64_t>(options.max_angular + 1);
    switch (kind) {
    case FeatomicKind::SoapRadialSpectrum:
        return species * species * radial;
    case FeatomicKind::SoapPowerSpectrum:
        return species * (species + 1) / 2 * species * angular * radial * radial;
    default:
        return species * species * radial * angular * angular;
    }
}

struct Matrix3 {
    double value[3][3]{};
};

inline double determinant(const Matrix3& matrix) {
    return matrix.value[0][0] * (matrix.value[1][1] * matrix.value[2][2] - matrix.value[1][2] * matrix.value[2][1])
        - matrix.value[0][1] * (matrix.value[1][0] * matrix.value[2][2] - matrix.value[1][2] * matrix.value[2][0])
        + matrix.value[0][2] * (matrix.value[1][0] * matrix.value[2][1] - matrix.value[1][1] * matrix.value[2][0]);
}

inline Matrix3 inverse(const Matrix3& matrix) {
    const double det = determinant(matrix);
    if (!std::isfinite(det) || std::abs(det) < 1e-14) {
        throw std::invalid_argument("cell matrix is singular");
    }
    Matrix3 result;
    result.value[0][0] = (matrix.value[1][1] * matrix.value[2][2] - matrix.value[1][2] * matrix.value[2][1]) / det;
    result.value[0][1] = (matrix.value[0][2] * matrix.value[2][1] - matrix.value[0][1] * matrix.value[2][2]) / det;
    result.value[0][2] = (matrix.value[0][1] * matrix.value[1][2] - matrix.value[0][2] * matrix.value[1][1]) / det;
    result.value[1][0] = (matrix.value[1][2] * matrix.value[2][0] - matrix.value[1][0] * matrix.value[2][2]) / det;
    result.value[1][1] = (matrix.value[0][0] * matrix.value[2][2] - matrix.value[0][2] * matrix.value[2][0]) / det;
    result.value[1][2] = (matrix.value[0][2] * matrix.value[1][0] - matrix.value[0][0] * matrix.value[1][2]) / det;
    result.value[2][0] = (matrix.value[1][0] * matrix.value[2][1] - matrix.value[1][1] * matrix.value[2][0]) / det;
    result.value[2][1] = (matrix.value[0][1] * matrix.value[2][0] - matrix.value[0][0] * matrix.value[2][1]) / det;
    result.value[2][2] = (matrix.value[0][0] * matrix.value[1][1] - matrix.value[0][1] * matrix.value[1][0]) / det;
    return result;
}

inline std::array<double, 3> cross(const std::array<double, 3>& left, const std::array<double, 3>& right) {
    return {
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    };
}

inline double dot(const std::array<double, 3>& left, const std::array<double, 3>& right) {
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

inline double norm(const std::array<double, 3>& value) {
    return std::sqrt(dot(value, value));
}

struct KVector {
    std::array<double, 3> vector{};
    std::array<double, 3> direction{};
    double norm = 0.0;
};

inline std::vector<KVector> make_k_vectors(const double* cell_data, double cutoff) {
    Matrix3 cell;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            cell.value[row][column] = cell_data[row * 3 + column];
        }
    }
    const Matrix3 inverse_cell = inverse(cell);
    std::array<std::array<double, 3>, 3> reciprocal{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            reciprocal[static_cast<std::size_t>(row)][static_cast<std::size_t>(column)]
                = 2.0 * kPi * inverse_cell.value[column][row];
        }
    }
    const std::array<double, 3> b0 = reciprocal[0];
    const std::array<double, 3> b1 = reciprocal[1];
    const std::array<double, 3> b2 = reciprocal[2];
    const double reciprocal_volume = std::abs(dot(b0, cross(b1, b2)));
    const int n0_max = std::max(1, static_cast<int>(std::ceil(norm(cross(b1, b2)) / reciprocal_volume * cutoff)));
    const int n1_max = std::max(1, static_cast<int>(std::ceil(norm(cross(b2, b0)) / reciprocal_volume * cutoff)));
    const int n2_max = std::max(1, static_cast<int>(std::ceil(norm(cross(b0, b1)) / reciprocal_volume * cutoff)));
    const double cutoff2 = cutoff * cutoff;
    auto make_vector = [](const std::array<double, 3>& value) -> KVector {
        const double length = norm(value);
        return {value, std::array<double, 3>{value[0] / length, value[1] / length, value[2] / length}, length};
    };
    std::vector<KVector> result;
    for (int n2 = 1; n2 <= n2_max; ++n2) {
        const std::array<double, 3> value{n2 * b2[0], n2 * b2[1], n2 * b2[2]};
        if (dot(value, value) < cutoff2) {
            result.push_back(make_vector(value));
        }
    }
    for (int n1 = 1; n1 <= n1_max; ++n1) {
        for (int n2 = -n2_max; n2 <= n2_max; ++n2) {
            const std::array<double, 3> value{
                n1 * b1[0] + n2 * b2[0],
                n1 * b1[1] + n2 * b2[1],
                n1 * b1[2] + n2 * b2[2],
            };
            if (dot(value, value) < cutoff2) {
                result.push_back(make_vector(value));
            }
        }
    }
    for (int n1 = 1; n1 <= n0_max; ++n1) {
        for (int n2 = -n1_max; n2 <= n1_max; ++n2) {
            for (int n3 = -n2_max; n3 <= n2_max; ++n3) {
                const std::array<double, 3> value{
                    n1 * b0[0] + n2 * b1[0] + n3 * b2[0],
                    n1 * b0[1] + n2 * b1[1] + n3 * b2[1],
                    n1 * b0[2] + n2 * b1[2] + n3 * b2[2],
                };
                if (dot(value, value) < cutoff2) {
                    result.push_back(make_vector(value));
                }
            }
        }
    }
    return result;
}

inline double exponential_integral_e1(double x) {
    if (x <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    if (x < 1.0) {
        double series = 0.0;
        double power = 1.0;
        for (int k = 1; k < 200; ++k) {
            power *= -x;
            const double term = power / (static_cast<double>(k) * k);
            series += term;
            if (std::abs(term) < 2e-16 * std::max(1.0, std::abs(series))) {
                break;
            }
        }
        return -0.5772156649015329 - std::log(x) - series;
    }
    double term = 1.0;
    double sum = 1.0;
    for (int k = 1; k < 100; ++k) {
        term *= -static_cast<double>(k) / x;
        sum += term;
        if (std::abs(term) > std::abs(sum)) {
            break;
        }
    }
    return std::exp(-x) * sum / x;
}

inline double lode_density_fourier(double k_norm, double smearing, int exponent) {
    const double sigma2 = smearing * smearing;
    const double x = 0.5 * k_norm * k_norm * sigma2;
    if (exponent == 0) {
        return std::pow(4.0 * kPi * sigma2, 0.75) * std::exp(-x);
    }
    if (exponent == 1) {
        return 4.0 * kPi * std::exp(-x) / (k_norm * k_norm);
    }
    const double p_eff = 3.0 - exponent;
    const double factor = std::pow(kPi, 1.5) * std::pow(2.0 * sigma2, 0.5 * p_eff) / gamma_value(0.5 * exponent);
    double value = 0.0;
    if (exponent == 2) {
        value = std::sqrt(kPi / x) * std::erfc(std::sqrt(x));
    } else if (exponent == 3) {
        value = exponential_integral_e1(x);
    } else if (exponent == 4) {
        value = 2.0 * (std::exp(-x) - std::sqrt(kPi * x) * std::erfc(std::sqrt(x)));
    } else if (exponent == 5) {
        value = std::exp(-x) - x * exponential_integral_e1(x);
    } else if (exponent == 6) {
        value = ((2.0 - 4.0 * x) * std::exp(-x) + 4.0 * std::sqrt(kPi) * std::pow(x, 1.5) * std::erfc(std::sqrt(x))) / 3.0;
    } else if (exponent == 7) {
        value = (1.0 - x) * std::exp(-x) / 2.0 + x * x * exponential_integral_e1(x) / 2.0;
    } else if (exponent == 8) {
        value = -2.0 / 15.0 * ((-3.0 + 2.0 * x - 4.0 * x * x) * std::exp(-x) + 4.0 * std::sqrt(kPi) * std::pow(x, 2.5) * std::erfc(std::sqrt(x)));
    } else if (exponent == 9) {
        value = (x * x - x + 2.0) * std::exp(-x) / 6.0 - x * x * x * exponential_integral_e1(x) / 6.0;
    } else {
        throw std::invalid_argument("LODE exponent must be between 0 and 9");
    }
    return factor * value;
}


inline void compute_lode_values(
    const StructureBatchView& batch,
    const FeatomicOptions& options,
    const std::vector<std::int32_t>& atom_types,
    double* output,
    const std::shared_ptr<ComputeControl>& control) {
    const int n_radial = options.max_radial + 1;
    const std::int64_t features = local_feature_count(options, FeatomicKind::LodeSphericalExpansion);
    std::fill(output, output + batch.atoms * features, 0.0);
    static thread_local int cached_max_radial = -1;
    static thread_local int cached_max_angular = -1;
    static thread_local double cached_radius = 0.0;
    static thread_local std::vector<GtoRadialBasis> cached_radial_bases;
    if (cached_max_radial != options.max_radial
        || cached_max_angular != options.max_angular
        || cached_radius != options.radial_radius) {
        cached_radial_bases.clear();
        cached_radial_bases.reserve(static_cast<std::size_t>(options.max_angular + 1));
        for (int l = 0; l <= options.max_angular; ++l) {
            cached_radial_bases.emplace_back(n_radial, options.radial_radius, l);
        }
        cached_max_radial = options.max_radial;
        cached_max_angular = options.max_angular;
        cached_radius = options.radial_radius;
    }
    const auto& radial_bases = cached_radial_bases;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options.num_threads > 0 ? options.num_threads : omp_get_max_threads())
#endif
    for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
        if (control && control->cancelled()) {
            continue;
        }
        const std::int64_t begin = batch.offsets[structure];
        const std::int64_t end = batch.offsets[structure + 1];
        const auto k_vectors = make_k_vectors(batch.cells + structure * 9, options.k_cutoff);
        if (k_vectors.empty()) {
            throw std::invalid_argument("no LODE reciprocal vectors for the current cell and k_cutoff");
        }
        const double volume = std::abs(
            batch.cells[structure * 9 + 0] * (batch.cells[structure * 9 + 4] * batch.cells[structure * 9 + 8] - batch.cells[structure * 9 + 5] * batch.cells[structure * 9 + 7])
            - batch.cells[structure * 9 + 1] * (batch.cells[structure * 9 + 3] * batch.cells[structure * 9 + 8] - batch.cells[structure * 9 + 5] * batch.cells[structure * 9 + 6])
            + batch.cells[structure * 9 + 2] * (batch.cells[structure * 9 + 3] * batch.cells[structure * 9 + 7] - batch.cells[structure * 9 + 4] * batch.cells[structure * 9 + 6]));
        const double global_factor = 4.0 * kPi / volume;
        const std::size_t local_count = static_cast<std::size_t>(end - begin);
        const std::size_t k_count = k_vectors.size();
        std::vector<double> cosines(local_count * k_count, 0.0);
        std::vector<double> sines(local_count * k_count, 0.0);
        std::vector<double> sum_cos(options.species.size() * k_count, 0.0);
        std::vector<double> sum_sin(options.species.size() * k_count, 0.0);
        for (std::int64_t local = 0; local < end - begin; ++local) {
            const double* position = batch.positions + (begin + local) * 3;
            for (std::size_t ik = 0; ik < k_vectors.size(); ++ik) {
                const auto& k = k_vectors[ik];
                const double phase = k.vector[0] * position[0] + k.vector[1] * position[1] + k.vector[2] * position[2];
                cosines[static_cast<std::size_t>(local) * k_count + ik] = std::cos(phase);
                sines[static_cast<std::size_t>(local) * k_count + ik] = std::sin(phase);
            }
            const auto type = static_cast<std::size_t>(atom_types[static_cast<std::size_t>(begin + local)]);
            for (std::size_t ik = 0; ik < k_vectors.size(); ++ik) {
                sum_cos[type * k_count + ik] += cosines[static_cast<std::size_t>(local) * k_count + ik];
                sum_sin[type * k_count + ik] += sines[static_cast<std::size_t>(local) * k_count + ik];
            }
        }
        std::vector<char> species_present(options.species.size(), 0);
        std::vector<std::size_t> active_neighbor_types;
        active_neighbor_types.reserve(options.species.size());
        for (std::int64_t local = 0; local < end - begin; ++local) {
            const std::size_t type = static_cast<std::size_t>(atom_types[static_cast<std::size_t>(begin + local)]);
            if (!species_present[type]) {
                species_present[type] = 1;
                active_neighbor_types.push_back(type);
            }
        }
        std::vector<std::vector<double>> projected(static_cast<std::size_t>(options.max_angular + 1));
        std::vector<double> density_fourier(k_vectors.size(), 0.0);
        for (std::size_t ik = 0; ik < k_vectors.size(); ++ik) {
            density_fourier[ik] = lode_density_fourier(
                k_vectors[ik].norm, options.density_width, options.exponent);
        }
        const std::size_t harmonic_count = static_cast<std::size_t>((options.max_angular + 1) * (options.max_angular + 1));
        std::vector<double> k_harmonics(k_vectors.size() * harmonic_count, 0.0);
        std::vector<double> harmonics;
        std::vector<double> harmonic_legendre;
        for (std::size_t ik = 0; ik < k_vectors.size(); ++ik) {
            real_spherical_harmonics_into(
                k_vectors[ik].direction, options.max_angular, harmonics, harmonic_legendre);
            std::copy(harmonics.begin(), harmonics.end(), k_harmonics.begin() + ik * harmonic_count);
        }
        static thread_local int cached_lode_max_radial = -1;
        static thread_local int cached_lode_max_angular = -1;
        static thread_local double cached_lode_radius = 0.0;
        static thread_local std::unordered_map<double, std::vector<double>> lode_radial_cache;
        if (cached_lode_max_radial != options.max_radial
            || cached_lode_max_angular != options.max_angular
            || cached_lode_radius != options.radial_radius) {
            lode_radial_cache.clear();
            lode_radial_cache.reserve(512);
            cached_lode_max_radial = options.max_radial;
            cached_lode_max_angular = options.max_angular;
            cached_lode_radius = options.radial_radius;
        }
        std::vector<const std::vector<double>*> cached_radial_by_k(k_vectors.size(), nullptr);
        std::vector<double> radial(static_cast<std::size_t>(n_radial));
        std::vector<double> radial_raw(static_cast<std::size_t>(n_radial));
        for (std::size_t ik = 0; ik < k_vectors.size(); ++ik) {
            const double norm = k_vectors[ik].norm;
            auto found = lode_radial_cache.find(norm);
            if (found == lode_radial_cache.end()) {
                std::vector<double> cached_radial(static_cast<std::size_t>(options.max_angular + 1) * n_radial);
                for (int l = 0; l <= options.max_angular; ++l) {
                    radial_bases[static_cast<std::size_t>(l)].lode_radial_integral_into(
                        norm, l, radial, radial_raw);
                    std::copy(radial.begin(), radial.end(), cached_radial.begin() + static_cast<std::size_t>(l) * n_radial);
                }
                found = lode_radial_cache.emplace(norm, std::move(cached_radial)).first;
            }
            cached_radial_by_k[ik] = &found->second;
        }
        for (int l = 0; l <= options.max_angular; ++l) {
            const std::size_t projection_size = static_cast<std::size_t>((2 * l + 1) * n_radial) * k_vectors.size();
            projected[static_cast<std::size_t>(l)].assign(projection_size, 0.0);
            for (std::size_t ik = 0; ik < k_vectors.size(); ++ik) {
                const auto& cached_radial = *cached_radial_by_k[ik];
                const double* radial_values = cached_radial.data() + static_cast<std::size_t>(l) * n_radial;
                for (int m = -l; m <= l; ++m) {
                    for (int n = 0; n < n_radial; ++n) {
                        projected[static_cast<std::size_t>(l)][
                            (static_cast<std::size_t>((m + l) * n_radial + n) * k_vectors.size()) + ik]
                            = k_harmonics[ik * harmonic_count + static_cast<std::size_t>(l * l + l + m)]
                                * radial_values[static_cast<std::size_t>(n)];
                    }
                }
            }
        }
        const std::size_t angular_channels = static_cast<std::size_t>(options.max_angular + 1);
        const std::size_t channel_stride = angular_channels * angular_channels * static_cast<std::size_t>(n_radial);
        std::vector<double> k_weights(2 * k_count, 0.0);
        constexpr double phases[] = {1.0, -1.0, -1.0, 1.0};
        for (std::int64_t local = 0; local < end - begin; ++local) {
            const auto center_type = static_cast<std::size_t>(atom_types[static_cast<std::size_t>(begin + local)]);
            double* row = output + (begin + local) * features;
            const std::size_t center_offset = static_cast<std::size_t>(center_type) * options.species.size() * channel_stride;
            for (const std::size_t neighbor_type : active_neighbor_types) {
                const std::size_t neighbor_offset = center_offset + neighbor_type * channel_stride;
                const std::size_t cosine_offset = static_cast<std::size_t>(local) * k_count;
                const double* cosine_values = cosines.data() + cosine_offset;
                const double* sine_values = sines.data() + cosine_offset;
                const double* sum_cosine_values = sum_cos.data() + neighbor_type * k_count;
                const double* sum_sine_values = sum_sin.data() + neighbor_type * k_count;
                double* even_weights = k_weights.data();
                double* odd_weights = k_weights.data() + k_count;
#ifdef _OPENMP
#pragma omp simd
#endif
                for (std::size_t ik = 0; ik < k_count; ++ik) {
                    const double density = global_factor * density_fourier[ik];
                    const double cosine = cosine_values[ik];
                    const double sine = sine_values[ik];
                    const double sum_cosine = sum_cosine_values[ik];
                    const double sum_sine = sum_sine_values[ik];
                    even_weights[ik] = density * 2.0 * (cosine * sum_cosine + sine * sum_sine);
                    odd_weights[ik] = density * 2.0 * (sine * sum_cosine - cosine * sum_sine);
                }
                for (int l = 0; l <= options.max_angular; ++l) {
                    const double phase = phases[l % 4];
                    const std::size_t angular_offset = neighbor_offset + static_cast<std::size_t>(l * l) * static_cast<std::size_t>(n_radial);
                    const double* weights = k_weights.data() + static_cast<std::size_t>(l % 2) * k_count;
                    const auto& l_projected = projected[static_cast<std::size_t>(l)];
                    for (int m = -l; m <= l; ++m) {
                        const std::size_t m_offset = angular_offset + static_cast<std::size_t>(l + m) * static_cast<std::size_t>(n_radial);
                        for (int n = 0; n < n_radial; ++n) {
                            const double* projected_values = l_projected.data()
                                + static_cast<std::size_t>((m + l) * n_radial + n) * k_count;
                            // Both arrays are contiguous in k. The old compensated
                            // sum serialized this hot loop; the fixed reduction is
                            // reproducible for a given build and enables SIMD.
                            double value = 0.0;
#ifdef _OPENMP
#pragma omp simd reduction(+:value)
#endif
                            for (std::size_t ik = 0; ik < k_count; ++ik) {
                                value += weights[ik] * projected_values[ik];
                            }
                            row[m_offset + static_cast<std::size_t>(n)] = phase * value;
                        }
                    }
                }
            }
        }
        mark_completed(control);
    }
}

} // namespace mdescriptor::detail
