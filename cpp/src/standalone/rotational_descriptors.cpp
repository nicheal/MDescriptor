#include "mdescriptor/extra.hpp"
#include "mdescriptor/neighbor.hpp"
#include "descriptor_common.hpp"
#include "extra_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <complex>
#include <limits>
#include <vector>

namespace mdescriptor {
using namespace detail;

namespace {
using Complex = std::complex<double>;

std::vector<Complex> complex_spherical_harmonics(Vec3 vector, int max_angular) {
    const std::size_t count = static_cast<std::size_t>((max_angular + 1) * (max_angular + 1));
    std::vector<Complex> output(count, Complex{0.0, 0.0});
    const double radius = norm(vector);
    if (radius <= std::numeric_limits<double>::epsilon()) {
        output[0] = Complex{0.5 / std::sqrt(kPi), 0.0};
        return output;
    }

    const double cos_theta = vector.z / radius;
    const double sin_theta = std::hypot(vector.x, vector.y) / radius;
    const double phi = std::atan2(vector.y, vector.x);
    std::vector<std::vector<double>> legendre(
        static_cast<std::size_t>(max_angular + 1),
        std::vector<double>(static_cast<std::size_t>(max_angular + 1), 0.0));
    legendre[0][0] = 1.0;
    for (int m = 1; m <= max_angular; ++m) {
        legendre[static_cast<std::size_t>(m)][static_cast<std::size_t>(m)]
            = -(2.0 * m - 1.0) * sin_theta
                * legendre[static_cast<std::size_t>(m - 1)][static_cast<std::size_t>(m - 1)];
    }
    for (int m = 0; m < max_angular; ++m) {
        legendre[static_cast<std::size_t>(m + 1)][static_cast<std::size_t>(m)]
            = (2.0 * m + 1.0) * cos_theta * legendre[static_cast<std::size_t>(m)][static_cast<std::size_t>(m)];
        for (int l = m + 2; l <= max_angular; ++l) {
            legendre[static_cast<std::size_t>(l)][static_cast<std::size_t>(m)] = (
                (2.0 * l - 1.0) * cos_theta * legendre[static_cast<std::size_t>(l - 1)][static_cast<std::size_t>(m)]
                - (l + m - 1.0) * legendre[static_cast<std::size_t>(l - 2)][static_cast<std::size_t>(m)]) / (l - m);
        }
    }

    for (int l = 0; l <= max_angular; ++l) {
        for (int m = 0; m <= l; ++m) {
            const double normalization = std::sqrt(
                (2.0 * l + 1.0) / (4.0 * kPi)
                * std::exp(std::lgamma(l - m + 1.0) - std::lgamma(l + m + 1.0)));
            const Complex phase = std::polar(1.0, m * phi);
            const Complex positive = normalization * legendre[static_cast<std::size_t>(l)][static_cast<std::size_t>(m)] * phase;
            output[static_cast<std::size_t>(l * l + l + m)] = positive;
            if (m > 0) {
                output[static_cast<std::size_t>(l * l + l - m)] = (m % 2 == 0 ? 1.0 : -1.0) * std::conj(positive);
            }
        }
    }
    return output;
}

std::vector<double> modified_spherical_bessel(double x, int max_angular) {
    std::vector<double> result(static_cast<std::size_t>(max_angular + 1), 0.0);
    const double ax = std::abs(x);
    if (ax < 1.0) {
        for (int l = 0; l <= max_angular; ++l) {
            double term = std::exp(
                l * std::log(std::max(ax, std::numeric_limits<double>::min()))
                + 0.5 * std::log(kPi)
                - (l + 1.0) * std::log(2.0)
                - std::lgamma(l + 1.5));
            double sum = term;
            for (int k = 0; k < 80; ++k) {
                term *= (ax * ax) / (4.0 * (k + 1.0) * (k + l + 1.5));
                sum += term;
                if (std::abs(term) <= std::abs(sum) * 1e-16) {
                    break;
                }
            }
            result[static_cast<std::size_t>(l)] = sum;
        }
        return result;
    }

    result[0] = std::sinh(ax) / ax;
    if (max_angular == 0) {
        return result;
    }
    result[1] = (ax * std::cosh(ax) - std::sinh(ax)) / (ax * ax);
    for (int l = 1; l < max_angular; ++l) {
        result[static_cast<std::size_t>(l + 1)] = result[static_cast<std::size_t>(l - 1)]
            - (2.0 * l + 1.0) / ax * result[static_cast<std::size_t>(l)];
    }
    return result;
}

std::vector<double> inverse_symmetric_sqrt(const std::vector<double>& matrix, int size) {
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

std::vector<double> so3_radial_basis(int n_max, int l_max, double cutoff, double alpha) {
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

std::vector<Complex> so3_coefficients(
    const StructureBatchView& batch,
    std::int64_t structure,
    int center,
    int n_max,
    int l_max,
    double cutoff,
    double alpha,
    bool weight_on,
    const NeighborGraph& graph,
    const std::vector<double>& basis) {
    const int quadrature_count = (n_max + l_max + 1) * 10;
    const std::size_t size = static_cast<std::size_t>(n_max * (l_max + 1) * (2 * l_max + 1));
    std::vector<Complex> output(size, Complex{0.0, 0.0});
    const std::int64_t begin = batch.offsets[structure];
    const std::int64_t center_atom = begin + center;
    const NeighborView neighbors = graph.for_center(center_atom);
    for (std::size_t index = 0; index < neighbors.size; ++index) {
        const Vec3 vector{neighbors.displacements[index * 3], neighbors.displacements[index * 3 + 1], neighbors.displacements[index * 3 + 2]};
        const double radius = std::sqrt(std::max(0.0, neighbors.distance2[index]));
        if (radius <= 0.0 || radius >= cutoff) {
            continue;
        }
        const auto harmonics = complex_spherical_harmonics(vector, l_max);
        std::vector<double> bessel(static_cast<std::size_t>((l_max + 1) * quadrature_count), 0.0);
        for (int q_index = 0; q_index < quadrature_count; ++q_index) {
            const double x = std::cos((2.0 * (q_index + 1) - 1.0) * kPi / (2.0 * quadrature_count));
            const double q = cutoff * 0.5 * (x + 1.0);
            const auto values = modified_spherical_bessel(2.0 * alpha * radius * q, l_max);
            for (int l = 0; l <= l_max; ++l) {
                bessel[static_cast<std::size_t>(q_index * (l_max + 1) + l)] = values[static_cast<std::size_t>(l)];
            }
        }
        const double cutoff_value = 0.5 * (std::cos(kPi * radius / cutoff) + 1.0);
        const int sign = weight_on && batch.numbers[neighbors.atoms[index]] != batch.numbers[center_atom] ? -1 : 1;
        const double atom_weight = sign * static_cast<double>(batch.numbers[neighbors.atoms[index]])
            * 4.0 * kPi * std::exp(-alpha * radius * radius) * cutoff_value;
        for (int n = 0; n < n_max; ++n) {
            for (int l = 0; l <= l_max; ++l) {
                double radial = 0.0;
                for (int q_index = 0; q_index < quadrature_count; ++q_index) {
                    radial += basis[static_cast<std::size_t>(n * quadrature_count + q_index)]
                        * bessel[static_cast<std::size_t>(q_index * (l_max + 1) + l)];
                }
                const double angular_normalization = std::sqrt(
                    2.0 * std::sqrt(2.0) * kPi / std::sqrt(2.0 * l + 1.0));
                for (int m = -l; m <= l; ++m) {
                    const std::size_t index = (static_cast<std::size_t>(n * (l_max + 1) + l) * (2 * l_max + 1))
                        + static_cast<std::size_t>(l_max + m);
                    output[index] += atom_weight * radial * angular_normalization
                        * harmonics[static_cast<std::size_t>(l * l + l + m)];
                }
            }
        }
    }
    return output;
}

double bispectrum_cutoff(double radius, double cutoff, double rmin0) {
    if (radius <= rmin0) {
        return 1.0;
    }
    return radius <= cutoff
        ? 0.5 * (std::cos(kPi * (radius - rmin0) / (cutoff - rmin0)) + 1.0)
        : 0.0;
}

std::vector<std::size_t> u_offsets(int max_order) {
    std::vector<std::size_t> offsets(static_cast<std::size_t>(max_order + 1), 0);
    for (int l = 1; l <= max_order; ++l) {
        offsets[static_cast<std::size_t>(l)] = offsets[static_cast<std::size_t>(l - 1)]
            + static_cast<std::size_t>(l * l);
    }
    return offsets;
}

std::vector<Complex> hyperspherical_u(
    Vec3 vector, int max_order, const std::vector<std::size_t>& offsets,
    double cutoff, double rfac0, double rmin0) {
    const double radius = norm(vector);
    std::vector<Complex> output(offsets.back() + static_cast<std::size_t>((max_order + 1) * (max_order + 1)), Complex{0.0, 0.0});
    if (radius <= 1e-14) {
        output[0] = Complex{1.0, 0.0};
        return output;
    }
    const double psi = rfac0 * kPi * (radius - rmin0) / (cutoff - rmin0);
    const double cos_psi = std::cos(psi);
    const double sin_psi = std::sin(psi);
    const Complex a = Complex{cos_psi, -sin_psi * vector.z / radius};
    const Complex b = (sin_psi / radius) * Complex{vector.y, -vector.x};
    output[0] = Complex{1.0, 0.0};
    for (int l = 1; l <= max_order; ++l) {
        const std::size_t base = offsets[static_cast<std::size_t>(l)];
        const std::size_t previous = offsets[static_cast<std::size_t>(l - 1)];
        int mb = 0;
        while (2 * mb <= l) {
            const std::size_t row = base + static_cast<std::size_t>(mb * (l + 1));
            const std::size_t previous_row = previous + static_cast<std::size_t>(mb * l);
            output[row] = Complex{0.0, 0.0};
            for (int ma = 0; ma < l; ++ma) {
                const double first_root = std::sqrt(static_cast<double>(l - ma) / static_cast<double>(l - mb));
                const double second_root = std::sqrt(static_cast<double>(ma + 1) / static_cast<double>(l - mb));
                output[row + static_cast<std::size_t>(ma)] += first_root * std::conj(a)
                    * output[previous_row + static_cast<std::size_t>(ma)];
                output[row + static_cast<std::size_t>(ma + 1)] += -second_root * std::conj(b)
                    * output[previous_row + static_cast<std::size_t>(ma)];
            }
            ++mb;
        }
        std::size_t left = base;
        std::size_t right = base + static_cast<std::size_t>((l + 1) * (l + 1) - 1);
        int mbpar = 1;
        mb = 0;
        while (2 * mb <= l) {
            int mapar = mbpar;
            for (int ma = 0; ma <= l; ++ma) {
                output[right] = mapar == 1 ? std::conj(output[left]) : -std::conj(output[left]);
                mapar = -mapar;
                ++left;
                --right;
            }
            mbpar = -mbpar;
            ++mb;
        }
    }
    return output;
}

double factorial_value(int value) {
    return std::tgamma(static_cast<double>(value) + 1.0);
}

double clebsch_gordan(int l1, int l2, int l, int m1, int m2) {
    const int aa2 = 2 * m1 - l1;
    const int bb2 = 2 * m2 - l2;
    const int numerator = aa2 + bb2 + l;
    if (numerator % 2 != 0) {
        return 0.0;
    }
    const int m = numerator / 2;
    if (m < 0 || m > l) {
        return 0.0;
    }
    const int z_min = std::max(0, std::max(
        -(l - l2 + aa2) / 2,
        -(l - l1 - bb2) / 2));
    const int z_max = std::min(
        (l1 + l2 - l) / 2,
        std::min((l1 - aa2) / 2, (l2 + bb2) / 2));
    double sum = 0.0;
    for (int z = z_min; z <= z_max; ++z) {
        sum += (z % 2 == 0 ? 1.0 : -1.0) / (
            factorial_value(z)
            * factorial_value((l1 + l2 - l) / 2 - z)
            * factorial_value((l1 - aa2) / 2 - z)
            * factorial_value((l2 + bb2) / 2 - z)
            * factorial_value((l - l2 + aa2) / 2 + z)
            * factorial_value((l - l1 - bb2) / 2 + z));
    }
    const double delta = std::sqrt(
        factorial_value((l1 + l2 - l) / 2)
        * factorial_value((l1 - l2 + l) / 2)
        * factorial_value((-l1 + l2 + l) / 2)
        / factorial_value((l1 + l2 + l) / 2 + 1));
    const double scale = std::sqrt(
        factorial_value((l1 + aa2) / 2)
        * factorial_value((l1 - aa2) / 2)
        * factorial_value((l2 + bb2) / 2)
        * factorial_value((l2 - bb2) / 2)
        * factorial_value((l + 2 * m - l) / 2)
        * factorial_value((l - (2 * m - l)) / 2)
        * (l + 1.0));
    return sum * delta * scale;
}

struct BispectrumComponent {
    int l1 = 0;
    int l2 = 0;
    int l = 0;
};

std::vector<BispectrumComponent> bispectrum_components(int max_order, int diagonal, bool l_bispectrum) {
    std::vector<BispectrumComponent> components;
    for (int l1 = 0; l1 <= max_order; ++l1) {
        if (l_bispectrum && diagonal == 2) {
            components.push_back({l1, l1, l1});
            continue;
        }
        for (int l2 = 0; l2 <= l1; ++l2) {
            if (l_bispectrum && diagonal == 2 && l2 != l1) {
                continue;
            }
            for (int l = l1 - l2; l <= std::min(max_order, l1 + l2); l += 2) {
                if (l_bispectrum) {
                    if (diagonal == 1 && l2 != l1) {
                        continue;
                    }
                    if (diagonal == 2 && l != l1) {
                        continue;
                    }
                    if (diagonal == 3 && l < l1) {
                        continue;
                    }
                } else if (l < l1) {
                    continue;
                }
                components.push_back({l1, l2, l});
            }
        }
    }
    return components;
}

double rotational_neighbor_weight(
    const StructureBatchView& batch,
    std::int32_t atom,
    RotationalDescriptorKind kind,
    double weight_scale,
    const std::vector<double>& neighbor_weights) {
    if (kind == RotationalDescriptorKind::SO4) {
        return static_cast<double>(batch.numbers[atom]);
    }
    if (!neighbor_weights.empty()) {
        return neighbor_weights[static_cast<std::size_t>(atom)];
    }
    return weight_scale;
}

std::vector<double> compute_bispectrum_center(
    const StructureBatchView& batch,
    std::int64_t center,
    int max_order,
    double cutoff,
    RotationalDescriptorKind kind,
    bool normalize_u,
    double weight_scale,
    double rfac0,
    double rmin0,
    double rcutfac,
    const std::vector<double>& neighbor_weights,
    const std::vector<double>& neighbor_radii,
    int diagonal,
    const NeighborGraph& graph) {
    const auto offsets = u_offsets(max_order);
    const auto components = bispectrum_components(max_order, diagonal, kind == RotationalDescriptorKind::LBispectrum);
    std::vector<Complex> total(offsets.back() + static_cast<std::size_t>((max_order + 1) * (max_order + 1)), Complex{0.0, 0.0});
    const double center_weight = kind == RotationalDescriptorKind::SO4 ? static_cast<double>(batch.numbers[center]) : 1.0;
    for (int l = 0; l <= max_order; ++l) {
        for (int m = 0; m <= l; ++m) {
            total[offsets[static_cast<std::size_t>(l)] + static_cast<std::size_t>(m * (l + 1) + m)] = center_weight;
        }
    }
    const NeighborView neighbors = graph.for_center(center);
    for (std::size_t index = 0; index < neighbors.size; ++index) {
        if (neighbors.exact_self(index, center)) {
            continue;
        }
        const Vec3 vector{neighbors.displacements[index * 3], neighbors.displacements[index * 3 + 1], neighbors.displacements[index * 3 + 2]};
        const double radius = std::sqrt(std::max(0.0, neighbors.distance2[index]));
        if (radius <= 1e-8) {
            continue;
        }
        const double neighbor_cutoff = neighbor_radii.empty()
            ? cutoff
            : (neighbor_radii[static_cast<std::size_t>(center)]
                + neighbor_radii[static_cast<std::size_t>(neighbors.atoms[index])]) * rcutfac;
        if (radius > neighbor_cutoff) {
            continue;
        }
        const auto values = hyperspherical_u(vector, max_order, offsets, neighbor_cutoff, rfac0, rmin0);
        const double scale = bispectrum_cutoff(radius, neighbor_cutoff, rmin0)
            * rotational_neighbor_weight(batch, neighbors.atoms[index], kind, weight_scale, neighbor_weights);
        for (std::size_t value = 0; value < total.size(); ++value) {
            total[value] += scale * values[value];
        }
    }
    if (normalize_u) {
        for (int l = 0; l <= max_order; ++l) {
            const double scale = 4.0 * kPi / std::sqrt(l + 1.0);
            for (int mb = 0; mb <= l; ++mb) {
                for (int ma = 0; ma <= l; ++ma) {
                    total[offsets[static_cast<std::size_t>(l)] + static_cast<std::size_t>(mb * (l + 1) + ma)] *= scale;
                }
            }
        }
    }
    std::vector<double> result;
    result.reserve(components.size());
    for (const auto component : components) {
        if ((component.l1 + component.l2 + component.l) % 2 != 0) {
            result.push_back(0.0);
            continue;
        }
        std::vector<Complex> z;
        for (int mb = 0; 2 * mb <= component.l; ++mb) {
            for (int ma = 0; ma <= component.l; ++ma) {
                const int ma1_min = std::max(0, (2 * ma - component.l - component.l2 + component.l1) / 2);
                const int ma2_max = (2 * ma - component.l - (2 * ma1_min - component.l1) + component.l2) / 2;
                const int na = std::min(component.l1, (2 * ma - component.l + component.l2 + component.l1) / 2) - ma1_min + 1;
                const int mb1_min = std::max(0, (2 * mb - component.l - component.l2 + component.l1) / 2);
                const int mb2_max = (2 * mb - component.l - (2 * mb1_min - component.l1) + component.l2) / 2;
                const int nb = std::min(component.l1, (2 * mb - component.l + component.l2 + component.l1) / 2) - mb1_min + 1;
                Complex value{0.0, 0.0};
                for (int ib = 0; ib < nb; ++ib) {
                    Complex inner{0.0, 0.0};
                    const int mb1 = mb1_min + ib;
                    const int mb2 = mb2_max - ib;
                    for (int ia = 0; ia < na; ++ia) {
                        const int ma1 = ma1_min + ia;
                        const int ma2 = ma2_max - ia;
                        inner += clebsch_gordan(component.l1, component.l2, component.l, ma1, ma2)
                            * total[offsets[static_cast<std::size_t>(component.l1)] + static_cast<std::size_t>(
                                mb1 * (component.l1 + 1) + ma1)]
                            * total[offsets[static_cast<std::size_t>(component.l2)] + static_cast<std::size_t>(
                                mb2 * (component.l2 + 1) + ma2)];
                    }
                    value += clebsch_gordan(component.l1, component.l2, component.l, mb1, mb2) * inner;
                }
                z.push_back(value);
            }
        }
        Complex bispectrum{0.0, 0.0};
        std::size_t z_index = 0;
        for (int mb = 0; 2 * mb < component.l; ++mb) {
            for (int ma = 0; ma <= component.l; ++ma) {
                const Complex u = total[offsets[static_cast<std::size_t>(component.l)]
                    + static_cast<std::size_t>(mb * (component.l + 1) + ma)];
                bispectrum += std::conj(u) * z[z_index++];
            }
        }
        if (component.l % 2 == 0) {
            const int mb = component.l / 2;
            for (int ma = 0; ma < mb; ++ma) {
                const Complex u = total[offsets[static_cast<std::size_t>(component.l)]
                    + static_cast<std::size_t>(mb * (component.l + 1) + ma)];
                bispectrum += std::conj(u) * z[z_index++];
            }
            const Complex u = total[offsets[static_cast<std::size_t>(component.l)]
                + static_cast<std::size_t>(mb * (component.l + 1) + mb)];
            bispectrum += 0.5 * std::conj(u) * z[z_index++];
        }
        result.push_back(2.0 * std::real(bispectrum));
    }
    return result;
}
} // namespace

std::int64_t rotational_feature_count(const RotationalDescriptorOptions& options) {
    const int l_max = options.kind == RotationalDescriptorKind::SO3 ? options.l_max
        : options.kind == RotationalDescriptorKind::LBispectrum ? std::max(0, options.twojmax) : 2 * options.l_max;
    if (options.kind == RotationalDescriptorKind::SO3) {
        return static_cast<std::int64_t>(l_max + 1) * options.n_max * (options.n_max + 1) / 2;
    }
    return static_cast<std::int64_t>(bispectrum_components(
        l_max, options.diagonal, options.kind == RotationalDescriptorKind::LBispectrum).size());
}

void compute_rotational_descriptors(
    const StructureBatchView& batch,
    const RotationalDescriptorOptions& options,
    double* output,
    const std::shared_ptr<ComputeControl>& control) {
    validate_batch(batch);
    if (!std::isfinite(options.cutoff) || !std::isfinite(options.alpha) || !std::isfinite(options.rfac0)
        || !std::isfinite(options.rmin0) || !std::isfinite(options.rcutfac)
        || !std::isfinite(options.weight_scale) || options.cutoff <= 0.0 || options.l_max < 0
        || options.n_max < 1 || options.alpha <= 0.0 || options.rfac0 <= 0.0
        || options.rmin0 < 0.0 || options.rcutfac <= 0.0
        || options.num_threads < 0
        || (options.kind == RotationalDescriptorKind::LBispectrum && options.twojmax < 0)) {
        throw std::invalid_argument("invalid rotational descriptor parameters");
    }
    if (!options.neighbor_weights.empty() && options.neighbor_weights.size() != static_cast<std::size_t>(batch.atoms)) {
        throw std::invalid_argument("neighbor_weights must contain one value per atom");
    }
    if (!options.neighbor_radii.empty() && options.neighbor_radii.size() != static_cast<std::size_t>(batch.atoms)) {
        throw std::invalid_argument("neighbor_radii must contain one value per atom");
    }
    for (const double radius : options.neighbor_radii) {
        if (!std::isfinite(radius) || radius <= 0.0) {
            throw std::invalid_argument("neighbor_radii must be finite and positive");
        }
    }
    if (!options.neighbor_radii.empty()
        && 2.0 * *std::min_element(options.neighbor_radii.begin(), options.neighbor_radii.end()) * options.rcutfac <= options.rmin0) {
        throw std::invalid_argument("neighbor radii and rcutfac must define a cutoff larger than rmin0");
    }
    if (options.neighbor_radii.empty() && options.cutoff <= options.rmin0) {
        throw std::invalid_argument("cutoff must be larger than rmin0");
    }
    for (const double weight : options.neighbor_weights) {
        if (!std::isfinite(weight)) {
            throw std::invalid_argument("neighbor_weights must be finite");
        }
    }
    const auto features = rotational_feature_count(options);
    std::fill(output, output + batch.atoms * features, 0.0);
    check_cancelled(control);
    if (control) {
        control->reset(batch.structures);
    }
    double graph_cutoff = options.cutoff;
    if (options.kind == RotationalDescriptorKind::LBispectrum && !options.neighbor_radii.empty()) {
        graph_cutoff = 2.0 * *std::max_element(options.neighbor_radii.begin(), options.neighbor_radii.end()) * options.rcutfac;
    }
    const NeighborGraph graph = build_neighbor_graph(batch, graph_cutoff, control, options.num_threads);
    auto compute_structure = [&](std::int64_t structure) {
        const std::int64_t begin = batch.offsets[structure];
        const std::int64_t end = batch.offsets[structure + 1];
        const bool so3 = options.kind == RotationalDescriptorKind::SO3;
        const int l_max = options.kind == RotationalDescriptorKind::LBispectrum ? std::max(0, options.twojmax / 2) : options.l_max;
        const auto so3_basis = so3 ? so3_radial_basis(options.n_max, l_max, options.cutoff, options.alpha)
                                   : std::vector<double>{};
        const int so3_width = 2 * l_max + 1;
        if (!so3) {
            const int expansion_order = options.kind == RotationalDescriptorKind::LBispectrum
                ? std::max(0, options.twojmax) : 2 * options.l_max;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options.num_threads > 0 ? options.num_threads : omp_get_max_threads()) if(!omp_in_parallel())
#endif
            for (std::int64_t center = begin; center < end; ++center) {
                if (cancelled(control)) {
                    continue;
                }
                const auto values = compute_bispectrum_center(
                    batch, center, expansion_order, options.cutoff, options.kind,
                    options.normalize_u, options.weight_scale, options.rfac0,
                    options.rmin0, options.rcutfac, options.neighbor_weights,
                    options.neighbor_radii, options.diagonal, graph);
                std::copy(values.begin(), values.end(), output + center * features);
            }
            return;
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options.num_threads > 0 ? options.num_threads : omp_get_max_threads()) if(!omp_in_parallel())
#endif
        for (std::int64_t center = begin; center < end; ++center) {
            if (cancelled(control)) {
                continue;
            }
            if (so3) {
                const auto coefficients = so3_coefficients(
                    batch, structure, static_cast<int>(center - begin), options.n_max, l_max,
                    options.cutoff, options.alpha, options.weight_on, graph, so3_basis);
                double* target = output + center * features;
                std::int64_t offset = 0;
                for (int n1 = 0; n1 < options.n_max; ++n1) {
                    for (int n2 = 0; n2 <= n1; ++n2) {
                        for (int l = 0; l <= l_max; ++l) {
                            double value = 0.0;
                            for (int m = -l; m <= l; ++m) {
                                const auto first = (static_cast<std::size_t>(n2 * (l_max + 1) + l) * so3_width)
                                    + static_cast<std::size_t>(l_max + m);
                                const auto second = (static_cast<std::size_t>(n1 * (l_max + 1) + l) * so3_width)
                                    + static_cast<std::size_t>(l_max + m);
                                value += std::real(coefficients[first] * std::conj(coefficients[second]));
                            }
                            target[offset++] = value;
                        }
                    }
                }
                continue;
            }
        }
    };
    if (batch.structures == 1) {
        compute_structure(0);
        check_cancelled(control);
        mark_completed(control);
        return;
    }
    run_parallel_structures(batch.structures, options.num_threads, control, compute_structure);
}
} // namespace mdescriptor
