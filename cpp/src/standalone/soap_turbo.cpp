#include "mdescriptor/descriptor.hpp"
#include "mdescriptor/neighbor.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "local_spherical_common.hpp"

namespace mdescriptor {
using namespace detail;

namespace {

constexpr double kSoapPi = 3.141592653589793238462643383279502884;
constexpr double kSoapSqrt2 = 1.414213562373095048801688724209698079;
using Complex = std::complex<double>;

std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string normalized_compression(const std::string& value) {
    auto mode = lower_copy(value);
    if (mode == "none" || mode == "off") {
        mode.clear();
    }
    return mode;
}

bool is_compression_mode(const std::string& mode) {
    if (mode.empty() || mode == "trivial") {
        return true;
    }
    return mode.size() == 3 && mode[1] == '_'
        && mode[0] >= '0' && mode[0] <= '2'
        && mode[2] >= '0' && mode[2] <= '2';
}

std::int64_t uncompressed_feature_count(const SoapTurboOptions& options) {
    const std::int64_t channels = std::accumulate(
        options.alpha_max.begin(), options.alpha_max.end(), std::int64_t{0});
    return channels * (channels + 1) / 2 * (options.l_max + 1);
}

std::int64_t compression_feature_count(const SoapTurboOptions& options) {
    const auto mode = normalized_compression(options.compression);
    if (mode.empty()) {
        return uncompressed_feature_count(options);
    }
    if (mode == "trivial") {
        const int channels = static_cast<int>(
            std::accumulate(options.alpha_max.begin(), options.alpha_max.end(), 0));
        std::vector<int> pivots;
        int pivot = 0;
        for (const auto alpha : options.alpha_max) {
            pivots.push_back(pivot);
            pivot += alpha;
        }
        std::int64_t count = 0;
        for (int first = 0; first < channels; ++first) {
            for (int second = first; second < channels; ++second) {
                if (std::find(pivots.begin(), pivots.end(), first) != pivots.end()
                    || std::find(pivots.begin(), pivots.end(), second) != pivots.end()) {
                    count += options.l_max + 1;
                }
            }
        }
        return count;
    }
    const int nu_r = mode[0] - '0';
    const int nu_s = mode[2] - '0';
    if (options.alpha_max.empty()) {
        return 0;
    }
    const int alpha = options.alpha_max.front();
    const int species = static_cast<int>(options.alpha_max.size());
    const int n1 = nu_r > 0 ? alpha : 1;
    const int n2 = nu_r == 2 ? alpha : 1;
    const int s1 = nu_s > 0 ? species : 1;
    const int s2 = nu_s == 2 ? species : 1;
    if ((nu_r % 2 == 0) && (nu_s % 2 == 0)) {
        const int channels = n1 * s1;
        return static_cast<std::int64_t>(channels) * (channels + 1) / 2
            * (options.l_max + 1);
    }
    return static_cast<std::int64_t>(n1) * s1 * n2 * s2 * (options.l_max + 1);
}

void validate_options(const SoapTurboOptions& options) {
    if (options.species.empty() || options.alpha_max.size() != options.species.size()) {
        throw std::invalid_argument("SOAPTurbo species and alpha_max must have the same non-zero length");
    }
    if (options.l_max < 0 || options.l_max > 20
        || !std::isfinite(options.rcut_hard) || options.rcut_hard <= 0.0
        || !std::isfinite(options.rcut_soft) || options.rcut_soft <= 0.0
        || options.rcut_soft > options.rcut_hard
        || !std::isfinite(options.nf) || options.nf <= 0.0
        || (options.basis != 0 && options.basis != 1)
        || options.radial_enhancement < 0 || options.radial_enhancement > 2) {
        throw std::invalid_argument("invalid SOAPTurbo parameters");
    }
    const auto expected = options.species.size();
    const auto check = [expected](const std::vector<double>& values, const char* name) {
        if (values.size() != expected) {
            throw std::invalid_argument(std::string("SOAPTurbo ") + name + " must have one value per species");
        }
        for (const double value : values) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument(std::string("SOAPTurbo ") + name + " must be finite");
            }
        }
    };
    for (const auto alpha : options.alpha_max) {
        if (alpha <= 0 || alpha > (options.basis == 0 ? 10 : 11)) {
            throw std::invalid_argument("SOAPTurbo alpha_max exceeds the upstream stable range");
        }
    }
    check(options.atom_sigma_r, "atom_sigma_r");
    check(options.atom_sigma_r_scaling, "atom_sigma_r_scaling");
    check(options.atom_sigma_t, "atom_sigma_t");
    check(options.atom_sigma_t_scaling, "atom_sigma_t_scaling");
    check(options.amplitude_scaling, "amplitude_scaling");
    check(options.central_weight, "central_weight");
    for (std::size_t type = 0; type < expected; ++type) {
        if (options.atom_sigma_r[type] <= 0.0 || options.atom_sigma_t[type] <= 0.0
            || options.atom_sigma_r[type] + options.atom_sigma_r_scaling[type] * options.rcut_hard <= 0.0
            || options.atom_sigma_t[type] + options.atom_sigma_t_scaling[type] * options.rcut_hard <= 0.0
            || options.amplitude_scaling[type] < 0.0) {
            throw std::invalid_argument("SOAPTurbo species parameters must remain positive and finite");
        }
    }
    const auto mode = normalized_compression(options.compression);
    if (!is_compression_mode(mode)) {
        throw std::invalid_argument("SOAPTurbo compression must be empty, trivial, or 0_0 through 2_2");
    }
    if (!mode.empty() && mode != "trivial") {
        for (const auto alpha : options.alpha_max) {
            if (alpha != options.alpha_max.front()) {
                throw std::invalid_argument("SOAPTurbo 0_0 through 2_2 compression requires equal alpha_max");
            }
        }
    }
    const auto mapping = make_type_map(options.species);
    for (const auto species : options.central_species) {
        if (mapping.find(species) == mapping.end()) {
            throw std::invalid_argument("SOAPTurbo central_species must be contained in species");
        }
    }
}

double radial_normalization(int n) {
    return std::sqrt(1.0 / static_cast<double>(2 * n + 5));
}

struct RadialBasis {
    int size = 0;
    bool gaussian_origin = false;
    double sigma = 0.5;
    std::vector<double> transform;
    std::vector<double> gaussian_column;
};

std::vector<double> gaussian_overlap_column(const RadialBasis& basis) {
    std::vector<double> result(static_cast<std::size_t>(basis.size), 0.0);
    const int last = basis.size - 1;
    result[static_cast<std::size_t>(last)] = 1.0;
    const double sigma2 = basis.sigma * basis.sigma;
    double integral_n = 0.0;
    double norm_n = 1.0;
    double norm_np1 = radial_normalization(-2);
    double integral_np1 = std::sqrt(kSoapPi / 2.0) * basis.sigma
        * std::erf(1.0 / (std::sqrt(2.0) * basis.sigma)) / norm_np1;
    double correction = sigma2;
    for (int n = -1; n <= basis.size - 1; ++n) {
        const double norm_np2 = radial_normalization(n);
        const double integral_np2 = sigma2 * static_cast<double>(n + 1) * norm_n / norm_np2 * integral_n
            + norm_np1 / norm_np2 * integral_np1 - correction / norm_np2;
        if (n > 0) {
            result[static_cast<std::size_t>(n - 1)] = integral_np2
                * kSoapSqrt2 / std::sqrt(basis.sigma) / std::pow(kSoapPi, 0.25);
        }
        correction *= 1.0;
        norm_n = norm_np1;
        norm_np1 = norm_np2;
        integral_n = integral_np1;
        integral_np1 = integral_np2;
    }
    return result;
}

RadialBasis make_radial_basis(int size, int basis, double sigma) {
    RadialBasis result;
    result.size = size;
    result.gaussian_origin = basis == 1;
    result.sigma = sigma;
    SymmetricMatrix overlap{size, std::vector<double>(static_cast<std::size_t>(size * size), 0.0)};
    const int polynomial_count = result.gaussian_origin ? size - 1 : size;
    for (int i = 0; i < polynomial_count; ++i) {
        for (int j = i; j < polynomial_count; ++j) {
            const double value = i == j
                ? 1.0
                : std::sqrt(static_cast<double>((2 * (i + 1) + 5) * (2 * (j + 1) + 5)))
                    / static_cast<double>(i + j + 7);
            overlap.at(i, j) = overlap.at(j, i) = value;
        }
    }
    if (result.gaussian_origin) {
        const auto column = gaussian_overlap_column(result);
        for (int index = 0; index < size; ++index) {
            overlap.at(index, size - 1) = overlap.at(size - 1, index) = column[static_cast<std::size_t>(index)];
        }
    }
    result.transform = inverse_sqrt(overlap).values;
    if (result.gaussian_origin) {
        result.gaussian_column = gaussian_overlap_column(result);
    }
    return result;
}

double smoothing_prefactor(double rj, double sigma, double soft, double hard, double nf) {
    if (hard == soft || (soft - rj) >= 4.0 * sigma) {
        return 0.0;
    }
    const double dr = hard - soft;
    const double sigma2 = sigma * sigma;
    return std::exp(-0.5 * (soft - rj) * (soft - rj)
        / (sigma2 + dr * dr / (nf * nf)));
}

void radial_coefficients(
    const RadialBasis& basis,
    const SoapTurboOptions& options,
    std::size_t type,
    double distance,
    bool central,
    double* result,
    double* primitive,
    double* filtered,
    double* raw) {
    std::fill_n(result, basis.size, 0.0);
    if (distance >= options.rcut_hard || (basis.gaussian_origin && central)) {
        return;
    }
    const double hard = 1.0;
    const double soft = options.rcut_soft / options.rcut_hard;
    const double rj = distance / options.rcut_hard;
    const double dr = hard - soft;
    const double atom_sigma = options.atom_sigma_r[type] / options.rcut_hard;
    const double atom_sigma_scaled = atom_sigma + options.atom_sigma_r_scaling[type] * rj;
    const double sigma2 = atom_sigma_scaled * atom_sigma_scaled;
    double amplitude = 1.0 / atom_sigma_scaled;
    const double amplitude_power = options.amplitude_scaling[type];
    if (amplitude_power != 0.0) {
        const double polynomial = 1.0 + 2.0 * rj * rj * rj - 3.0 * rj * rj;
        if (polynomial <= 1e-10) {
            return;
        }
        amplitude *= std::pow(polynomial, amplitude_power);
    }
    if (central) {
        amplitude *= options.central_weight[type];
    }
    if (options.radial_enhancement == 1) {
        amplitude *= rj + std::sqrt(2.0 / kSoapPi) * atom_sigma_scaled;
    } else if (options.radial_enhancement == 2) {
        amplitude *= rj * rj + sigma2 + std::sqrt(8.0 / kSoapPi) * atom_sigma_scaled * rj;
    }
    if (amplitude == 0.0) {
        return;
    }

    const int expansion_count = basis.gaussian_origin ? basis.size - 1 : basis.size;
    std::fill_n(primitive, basis.size, 0.0);
    std::fill_n(filtered, basis.size, 0.0);
    double integral_n = 0.0;
    double norm_n = 1.0;
    double norm_np1 = radial_normalization(-2);
    double integral_np1 = std::sqrt(kSoapPi / 2.0) * atom_sigma_scaled
        * (std::erf((soft - rj) / (std::sqrt(2.0) * atom_sigma_scaled))
            - std::erf(-rj / (std::sqrt(2.0) * atom_sigma_scaled))) / norm_np1;
    double correction_soft = hard == soft ? 0.0
        : sigma2 / dr * std::exp(-0.5 * (soft - rj) * (soft - rj) / sigma2);
    double correction_hard = sigma2 * std::exp(-0.5 * rj * rj / sigma2);
    for (int n = -1; n <= expansion_count; ++n) {
        correction_soft *= dr;
        correction_hard *= hard;
        const double norm_np2 = radial_normalization(n);
        const double integral_np2 = sigma2 * static_cast<double>(n + 1) * norm_n / norm_np2 * integral_n
            - norm_np1 * (rj - hard) / norm_np2 * integral_np1
            + correction_soft / norm_np2 - correction_hard / norm_np2;
        if (n > 0) {
            primitive[static_cast<std::size_t>(n - 1)] = integral_np2;
        }
        norm_n = norm_np1;
        norm_np1 = norm_np2;
        integral_n = integral_np1;
        integral_np1 = integral_np2;
    }

    const double prefactor = smoothing_prefactor(rj, atom_sigma_scaled, soft, hard, options.nf);
    if (prefactor != 0.0) {
        const double nf_width2 = dr * dr / (options.nf * options.nf);
        const double filtered_sigma = atom_sigma_scaled * dr / options.nf
            / std::sqrt(sigma2 + nf_width2);
        const double filtered_sigma2 = filtered_sigma * filtered_sigma;
        const double filtered_center = (sigma2 * soft + nf_width2 * rj)
            / (sigma2 + nf_width2);
        integral_n = 0.0;
        norm_n = 1.0;
        norm_np1 = radial_normalization(-2);
        integral_np1 = std::sqrt(kSoapPi / 2.0) * filtered_sigma
            * (std::erf((hard - filtered_center) / (std::sqrt(2.0) * filtered_sigma))
                - std::erf((soft - filtered_center) / (std::sqrt(2.0) * filtered_sigma))) / norm_np1;
        double filtered_correction = filtered_sigma2 / dr
            * std::exp(-0.5 * (soft - filtered_center) * (soft - filtered_center) / filtered_sigma2);
        for (int n = -1; n <= expansion_count; ++n) {
            filtered_correction *= dr;
            const double norm_np2 = radial_normalization(n);
            const double integral_np2 = filtered_sigma2 * static_cast<double>(n + 1) * norm_n / norm_np2 * integral_n
                - norm_np1 * (filtered_center - hard) / norm_np2 * integral_np1
                - filtered_correction / norm_np2;
            if (n > 0) {
                filtered[static_cast<std::size_t>(n - 1)] = integral_np2;
            }
            norm_n = norm_np1;
            norm_np1 = norm_np2;
            integral_n = integral_np1;
            integral_np1 = integral_np2;
        }
    }

    if (basis.gaussian_origin) {
        const double sigma_star = std::sqrt(basis.sigma * basis.sigma + sigma2);
        primitive[static_cast<std::size_t>(basis.size - 1)] = std::exp(
            -0.5 * rj * rj / (sigma_star * sigma_star))
            * std::sqrt(kSoapPi / 2.0) * atom_sigma_scaled * basis.sigma / sigma_star
            * (1.0 + std::erf(basis.sigma / atom_sigma_scaled * rj
                / (std::sqrt(2.0) * sigma_star)))
            * std::sqrt(2.0 / basis.sigma) / std::pow(kSoapPi, 0.25);
    }

    for (int index = 0; index < basis.size; ++index) {
        raw[static_cast<std::size_t>(index)] = amplitude
            * (primitive[static_cast<std::size_t>(index)]
                + prefactor * filtered[static_cast<std::size_t>(index)]);
    }
    for (int target = 0; target < basis.size; ++target) {
        for (int source = 0; source < basis.size; ++source) {
            result[static_cast<std::size_t>(target)] += basis.transform[
                static_cast<std::size_t>(target * basis.size + source)]
                * raw[static_cast<std::size_t>(source)];
        }
    }
    for (int index = 0; index < basis.size; ++index) {
        result[static_cast<std::size_t>(index)] *= std::sqrt(options.rcut_hard);
    }
}

double associated_legendre(int l, int m, double x) {
    double pmm = 1.0;
    const double root = std::sqrt(std::max(0.0, 1.0 - x * x));
    for (int order = 1; order <= m; ++order) {
        pmm *= -(2.0 * order - 1.0) * root;
    }
    if (l == m) {
        return pmm;
    }
    double pmp1m = x * (2.0 * m + 1.0) * pmm;
    if (l == m + 1) {
        return pmp1m;
    }
    for (int degree = m + 2; degree <= l; ++degree) {
        const double plm = (x * (2.0 * degree - 1.0) * pmp1m
            - (degree + m - 1.0) * pmm) / (degree - m);
        pmm = pmp1m;
        pmp1m = plm;
    }
    return pmp1m;
}

struct AngularTerm {
    int l = 0;
    int m = 0;
    double prefactor = 0.0;
};

std::vector<AngularTerm> make_angular_terms(int l_max) {
    std::vector<AngularTerm> result;
    result.reserve(static_cast<std::size_t>(1 + l_max * (l_max + 1) / 2 + l_max));
    for (int l = 0; l <= l_max; ++l) {
        double factorial_l_minus_m = 1.0;
        for (int i = 1; i <= l; ++i) {
            factorial_l_minus_m *= i;
        }
        for (int m = 0; m <= l; ++m) {
            if (m > 0) {
                factorial_l_minus_m /= static_cast<double>(l + 1 - m);
            }
            double factorial_l_plus_m = 1.0;
            for (int i = 1; i <= l + m; ++i) {
                factorial_l_plus_m *= i;
            }
            result.push_back({
                l,
                m,
                std::sqrt((2.0 * l + 1.0) / (4.0 * kSoapPi)
                    * factorial_l_minus_m / factorial_l_plus_m),
            });
        }
    }
    return result;
}

void modified_spherical_bessel(
    int l_max, double x, double* values, double* semifactorial) {
    std::fill_n(values, l_max + 1, 0.0);
    const double x2 = x * x;
    const double x4 = x2 * x2;
    constexpr double xcut = 1e-7;
    std::fill_n(semifactorial, l_max + 1, 1.0);
    for (int l = 1; l <= l_max; ++l) {
        semifactorial[static_cast<std::size_t>(l)] = semifactorial[static_cast<std::size_t>(l - 1)]
            * (2.0 * l + 1.0);
    }
    double flm2 = 1.0;
    double flm1 = 0.0;
    if (x > 0.0) {
        flm2 = std::abs((1.0 - std::exp(-2.0 * x2)) / (2.0 * x2));
        flm1 = std::abs((x2 - 1.0 + std::exp(-2.0 * x2) * (x2 + 1.0)) / (2.0 * x4));
    }
    for (int l = 0; l <= l_max; ++l) {
        if (l == 0) {
            values[0] = x < xcut ? 1.0 - x2 : flm2;
        } else if (l == 1) {
            values[1] = x2 / 1000.0 < xcut ? (x2 - x4) / semifactorial[1] : flm1;
        } else if (std::pow(x2, l) / semifactorial[static_cast<std::size_t>(l)] * l < xcut) {
            values[static_cast<std::size_t>(l)] = std::pow(x2, l)
                / semifactorial[static_cast<std::size_t>(l)];
        } else {
            values[static_cast<std::size_t>(l)] = std::abs(
                flm2 - (2.0 * l - 1.0) / x2 * flm1);
        }
        // Keep flm2/flm1 as the independently evaluated l=0/l=1 values;
        // the upstream recurrence starts updating them only after l=2.
        if (l >= 2) {
            flm2 = flm1;
            flm1 = values[static_cast<std::size_t>(l)];
        }
    }
}

void angular_coefficients(
    const SoapTurboOptions& options,
    std::size_t type,
    double distance,
    const std::array<double, 3>& displacement,
    const std::vector<AngularTerm>& terms,
    Complex* result,
    double* ilexp,
    double* semifactorial) {
    const int l_max = options.l_max;
    const int packed_count = 1 + l_max * (l_max + 1) / 2 + l_max;
    std::fill_n(result, packed_count, Complex{0.0, 0.0});
    if (distance >= options.rcut_hard) {
        return;
    }
    const double sigma = options.atom_sigma_t[type]
        + options.atom_sigma_t_scaling[type] * distance;
    const double x = distance < 1e-14 ? 1.0
        : std::clamp(displacement[2] / distance, -1.0, 1.0);
    const double phi = distance < 1e-14 ? 0.0 : std::atan2(displacement[1], displacement[0]);
    modified_spherical_bessel(l_max, distance / sigma, ilexp, semifactorial);
    const double amplitude = options.rcut_hard * options.rcut_hard / (sigma * sigma);
    const Complex phase_step = phi == 0.0
        ? Complex{1.0, 0.0} : std::exp(Complex{0.0, -phi});
    int packed = 0;
    for (int l = 0; l <= l_max; ++l) {
        Complex phase{1.0, 0.0};
        for (int m = 0; m <= l; ++m) {
            if (m > 0) {
                phase *= phase_step;
            }
            const auto& term = terms[static_cast<std::size_t>(packed)];
            result[static_cast<std::size_t>(packed++)] = amplitude * term.prefactor
                * associated_legendre(term.l, term.m, x)
                * ilexp[static_cast<std::size_t>(term.l)] * phase;
        }
    }
}

struct CompressionMap {
    std::int64_t dimension = 0;
    bool identity = false;
    std::vector<std::vector<std::pair<std::int64_t, double>>> rows;
};

CompressionMap make_compression_map(const SoapTurboOptions& options) {
    const auto mode = normalized_compression(options.compression);
    const std::int64_t dense_count = uncompressed_feature_count(options);
    if (mode.empty()) {
        CompressionMap result;
        result.dimension = dense_count;
        result.identity = true;
        return result;
    }

    CompressionMap result;
    result.dimension = compression_feature_count(options);
    result.rows.resize(static_cast<std::size_t>(result.dimension));
    if (mode == "trivial") {
        const int channels = static_cast<int>(
            std::accumulate(options.alpha_max.begin(), options.alpha_max.end(), 0));
        std::vector<int> pivots;
        int pivot = 0;
        for (const auto alpha : options.alpha_max) {
            pivots.push_back(pivot);
            pivot += alpha;
        }
        std::int64_t dense = 0;
        std::int64_t compressed = 0;
        for (int first = 0; first < channels; ++first) {
            for (int second = first; second < channels; ++second) {
                const bool retain = std::find(pivots.begin(), pivots.end(), first) != pivots.end()
                    || std::find(pivots.begin(), pivots.end(), second) != pivots.end();
                for (int l = 0; l <= options.l_max; ++l, ++dense) {
                    if (retain) {
                        result.rows[static_cast<std::size_t>(compressed++)].push_back({dense, 1.0});
                    }
                }
            }
        }
        return result;
    }

    const int nu_r = mode[0] - '0';
    const int nu_s = mode[2] - '0';
    const int alpha = options.alpha_max.front();
    const int species = static_cast<int>(options.alpha_max.size());
    const int n1 = nu_r > 0 ? alpha : 1;
    const int n2 = nu_r == 2 ? alpha : 1;
    const int s1 = nu_s > 0 ? species : 1;
    const int s2 = nu_s == 2 ? species : 1;
    const bool symmetric = (nu_r % 2 == 0) && (nu_s % 2 == 0);
    const int channel_count = alpha * species;
    std::int64_t dense = 0;
    auto add = [&](int compressed, std::int64_t dense_index, double factor) {
        result.rows[static_cast<std::size_t>(compressed)].push_back({dense_index, factor});
    };
    for (int first = 0; first < channel_count; ++first) {
        const int z1 = first / alpha;
        const int n_first = first % alpha;
        const int a1 = n_first % n1;
        const int compressed_first = (z1 % s1) * n1 + a1;
        for (int second = first; second < channel_count; ++second) {
            const int z2 = second / alpha;
            const int n_second = second % alpha;
            const int a2 = n_second % n2;
            const int compressed_second = (z2 % s2) * n2 + a2;
            for (int l = 0; l <= options.l_max; ++l, ++dense) {
                if (symmetric) {
                    const int first_compressed = std::min(compressed_first, compressed_second);
                    const int second_compressed = std::max(compressed_first, compressed_second);
                    const int index = (first_compressed * (2 * n1 * s1 + 1 - first_compressed))
                        * (options.l_max + 1) / 2
                        + (second_compressed - first_compressed) * (options.l_max + 1) + l;
                    add(index, dense,
                        first != second && first_compressed == second_compressed ? kSoapSqrt2 : 1.0);
                } else {
                    const int index = (compressed_first * s2 * n2 + compressed_second)
                        * (options.l_max + 1) + l;
                    const int swapped_first = (z2 % s1) * n1 + (n_second % n1);
                    const int swapped_second = (z1 % s2) * n2 + (n_first % n2);
                    const int swapped_index = (swapped_first * s2 * n2 + swapped_second)
                        * (options.l_max + 1) + l;
                    if (first == second) {
                        add(index, dense, 1.0);
                    } else if (index == swapped_index) {
                        add(index, dense, kSoapSqrt2);
                    } else {
                        add(index, dense, 1.0 / kSoapSqrt2);
                        add(swapped_index, dense, 1.0 / kSoapSqrt2);
                    }
                }
            }
        }
    }
    return result;
}

int effective_thread_count(std::int64_t structures, int requested_threads) {
    int available = requested_threads;
    if (available <= 0) {
#ifdef _OPENMP
        available = omp_get_max_threads();
#else
        available = 1;
#endif
    }
    return std::max(1, static_cast<int>(std::min<std::int64_t>(structures, available)));
}

template <typename Function>
void run_structures(
    std::int64_t structures,
    int requested_threads,
    const std::shared_ptr<ComputeControl>& control,
    Function&& function) {
    const int threads = effective_thread_count(structures, requested_threads);
    if (threads == 1) {
        for (std::int64_t structure = 0; structure < structures; ++structure) {
            if (control && control->cancelled()) {
                continue;
            }
            function(structure);
            mark_completed(control);
        }
        if (control && control->cancelled()) {
            throw CancelledError();
        }
        return;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (std::int64_t structure = 0; structure < structures; ++structure) {
        if (control && control->cancelled()) {
            continue;
        }
        function(structure);
        mark_completed(control);
    }
    if (control && control->cancelled()) {
        throw CancelledError();
    }
}

} // namespace

struct SoapTurboPrepared {
    CompressionMap compression;
    std::vector<RadialBasis> bases;
    std::vector<AngularTerm> angular_terms;
    std::vector<std::size_t> channel_offsets;
    detail::TypeMap mapping;
    std::vector<char> central_allowed;
    std::size_t max_basis_size = 0;
    std::int64_t dense_features = 0;
    int packed_count = 0;
};

std::shared_ptr<SoapTurboPrepared> make_soap_turbo_prepared(
    const SoapTurboOptions& options) {
    auto result = std::make_shared<SoapTurboPrepared>();
    result->compression = make_compression_map(options);
    result->mapping = make_type_map(options.species);
    result->central_allowed.assign(
        options.species.size(), options.central_species.empty() ? 1 : 0);
    for (const auto species : options.central_species) {
        result->central_allowed[result->mapping.at(species)] = 1;
    }
    result->bases.reserve(options.species.size());
    for (std::size_t type = 0; type < options.species.size(); ++type) {
        result->bases.push_back(make_radial_basis(
            options.alpha_max[type], options.basis,
            options.atom_sigma_r[type] / options.rcut_hard));
        result->max_basis_size = std::max(
            result->max_basis_size, result->bases.back().size > 0
                ? static_cast<std::size_t>(result->bases.back().size) : std::size_t{0});
    }
    result->channel_offsets.assign(options.species.size() + 1, 0);
    for (std::size_t type = 0; type < options.species.size(); ++type) {
        result->channel_offsets[type + 1] = result->channel_offsets[type]
            + static_cast<std::size_t>(options.alpha_max[type]);
    }
    result->dense_features = uncompressed_feature_count(options);
    result->packed_count = 1 + options.l_max * (options.l_max + 1) / 2 + options.l_max;
    result->angular_terms = make_angular_terms(options.l_max);
    return result;
}

std::int64_t soap_turbo_feature_count(const SoapTurboOptions& options) {
    validate_options(options);
    return compression_feature_count(options);
}

void compute_soap_turbo(
    const StructureBatchView& batch,
    const SoapTurboOptions& options,
    const SoapTurboPrepared& prepared,
    double* output,
    const std::shared_ptr<ComputeControl>& control) {
    const auto& compression = prepared.compression;
    const std::int64_t features = compression.dimension;
    std::fill(output, output + batch.atoms * features, 0.0);
    if (batch.atoms == 0) {
        return;
    }
    if (control) {
        control->reset(batch.structures);
    }

    std::vector<std::size_t> atom_types(static_cast<std::size_t>(batch.atoms));
    for (std::int64_t atom = 0; atom < batch.atoms; ++atom) {
        if (batch.numbers[atom] <= 0) {
            throw std::invalid_argument("atomic numbers must be positive");
        }
        const auto found = prepared.mapping.find(batch.numbers[atom]);
        if (found == prepared.mapping.end()) {
            throw std::invalid_argument("batch contains an atomic number outside calculator species");
        }
        atom_types[static_cast<std::size_t>(atom)] = found->second;
    }
    const std::size_t channel_count = prepared.channel_offsets.back();
    const auto& bases = prepared.bases;
    const auto& channel_offsets = prepared.channel_offsets;
    const int packed_count = prepared.packed_count;
    const int threads = effective_thread_count(batch.structures, options.num_threads);
    const auto graph = build_neighbor_graph(batch, options.rcut_hard, control, threads);

    run_structures(batch.structures, threads, control, [&](std::int64_t structure) {
        const std::int64_t begin = batch.offsets[structure];
        const std::int64_t end = batch.offsets[structure + 1];
        std::vector<Complex> coefficients(channel_count * static_cast<std::size_t>(packed_count));
        std::vector<double> radial(prepared.max_basis_size);
        std::vector<double> primitive(prepared.max_basis_size);
        std::vector<double> filtered(prepared.max_basis_size);
        std::vector<double> raw(prepared.max_basis_size);
        std::vector<Complex> angular(static_cast<std::size_t>(packed_count));
        std::vector<double> ilexp(static_cast<std::size_t>(options.l_max + 1));
        std::vector<double> semifactorial(static_cast<std::size_t>(options.l_max + 1));
        std::vector<double> dense(
            compression.identity ? 0 : static_cast<std::size_t>(prepared.dense_features));
        for (std::int64_t center = begin; center < end; ++center) {
            const std::size_t center_type = atom_types[static_cast<std::size_t>(center)];
            if (!prepared.central_allowed[center_type]) {
                continue;
            }
            std::fill(coefficients.begin(), coefficients.end(), Complex{0.0, 0.0});

            auto add_atom = [&](std::size_t type, double distance,
                                const std::array<double, 3>& displacement, bool central) {
                radial_coefficients(
                    bases[type], options, type, distance, central,
                    radial.data(), primitive.data(), filtered.data(), raw.data());
                angular_coefficients(
                    options, type, distance, displacement,
                    prepared.angular_terms,
                    angular.data(), ilexp.data(), semifactorial.data());
                const std::size_t offset = channel_offsets[type];
                for (int n = 0; n < options.alpha_max[type]; ++n) {
                    for (int k = 0; k < packed_count; ++k) {
                        coefficients[(offset + static_cast<std::size_t>(n)) * packed_count
                            + static_cast<std::size_t>(k)] += 4.0 * kSoapPi
                            * radial[static_cast<std::size_t>(n)]
                            * angular[static_cast<std::size_t>(k)];
                    }
                }
            };

            add_atom(center_type, 0.0, {0.0, 0.0, 0.0}, true);
            const NeighborView neighbors = graph.for_center(center);
            for (std::size_t index = 0; index < neighbors.size; ++index) {
                if (neighbors.exact_self(index, center)) {
                    continue;
                }
                const auto type = atom_types[static_cast<std::size_t>(neighbors.atoms[index])];
                add_atom(
                    type,
                    std::sqrt(std::max(0.0, neighbors.distance2[index])),
                    {neighbors.displacements[index * 3], neighbors.displacements[index * 3 + 1],
                     neighbors.displacements[index * 3 + 2]},
                    false);
            }

            if (options.basis == 1 && options.central_weight[center_type] != 0.0) {
                const auto& basis = bases[center_type];
                const auto& gaussian_column = basis.gaussian_column;
                const double sigma_r = options.atom_sigma_r[center_type];
                const double sigma_t = options.atom_sigma_t[center_type];
                const double enhancement = options.radial_enhancement == 1
                    ? std::sqrt(2.0 / kSoapPi) * sigma_r / options.rcut_hard
                    : options.radial_enhancement == 2
                        ? sigma_r * sigma_r / (options.rcut_hard * options.rcut_hard) : 1.0;
                const double prefactor = enhancement * options.central_weight[center_type]
                    * std::sqrt(4.0 * kSoapPi) * std::pow(kSoapPi, 0.25)
                    * std::sqrt(sigma_r / 2.0) * std::pow(options.rcut_hard, 3.0)
                    / (sigma_t * sigma_t * sigma_r);
                for (int n = 0; n < basis.size; ++n) {
                    double value = 0.0;
                    for (int source = 0; source < basis.size; ++source) {
                        value += basis.transform[static_cast<std::size_t>(n * basis.size + source)]
                            * gaussian_column[static_cast<std::size_t>(source)];
                    }
                    coefficients[(channel_offsets[center_type] + static_cast<std::size_t>(n))
                        * packed_count] += prefactor * value;
                }
            }

            double* row = output + center * features;
            double* dense_values = compression.identity ? row : dense.data();
            std::int64_t dense_index = 0;
            for (std::size_t first = 0; first < channel_count; ++first) {
                for (std::size_t second = first; second < channel_count; ++second) {
                    for (int l = 0; l <= options.l_max; ++l) {
                        double value = 0.0;
                        const int packed_offset = l * (l + 1) / 2;
                        for (int m = 0; m <= l; ++m) {
                            const double multiplicity = (first != second ? kSoapSqrt2 : 1.0)
                                * (m > 0 ? 2.0 : 1.0);
                            const auto first_value = coefficients[first * packed_count
                                + static_cast<std::size_t>(packed_offset + m)];
                            const auto second_value = coefficients[second * packed_count
                                + static_cast<std::size_t>(packed_offset + m)];
                            value += multiplicity * std::real(first_value * std::conj(second_value));
                        }
                        dense_values[static_cast<std::size_t>(dense_index++)] = value;
                    }
                }
            }

            if (!compression.identity) {
                for (std::int64_t compressed = 0; compressed < features; ++compressed) {
                    double value = 0.0;
                    for (const auto [source, factor] : compression.rows[static_cast<std::size_t>(compressed)]) {
                        value += factor * dense[static_cast<std::size_t>(source)];
                    }
                    row[compressed] = value;
                }
            }
            double norm = 0.0;
            for (std::int64_t index = 0; index < features; ++index) {
                norm += row[index] * row[index];
            }
            norm = std::sqrt(norm);
            if (norm < 1e-5) {
                norm = 1.0;
            }
            for (std::int64_t index = 0; index < features; ++index) {
                row[index] /= norm;
            }
        }
    });
}

SoapTurboCalculator::SoapTurboCalculator(SoapTurboOptions options) : options_(std::move(options)) {
    // The CUDA adapter consumes the normalized radial basis and compression
    // map before the first compute call.  Prepare them eagerly so validation
    // remains lazy with respect to the CUDA driver while the CPU and CUDA
    // paths share exactly the same basis construction.
    validate_options(options_);
    prepared_ = make_soap_turbo_prepared(options_);
    prepared_alpha_max_ = options_.alpha_max;
    prepared_channel_offsets_.reserve(prepared_->channel_offsets.size());
    for (const auto value : prepared_->channel_offsets) {
        prepared_channel_offsets_.push_back(static_cast<std::int32_t>(value));
    }
    prepared_central_allowed_.reserve(prepared_->central_allowed.size());
    for (const auto value : prepared_->central_allowed) {
        prepared_central_allowed_.push_back(static_cast<std::int32_t>(value));
    }
    prepared_basis_offsets_.assign(options_.species.size() + 1, 0);
    for (std::size_t type = 0; type < prepared_->bases.size(); ++type) {
        prepared_basis_offsets_[type + 1] = prepared_basis_offsets_[type]
            + static_cast<std::int32_t>(prepared_->bases[type].size);
    }
    for (const auto& basis : prepared_->bases) {
        prepared_basis_transforms_.insert(
            prepared_basis_transforms_.end(), basis.transform.begin(), basis.transform.end());
        prepared_gaussian_columns_.insert(
            prepared_gaussian_columns_.end(), basis.gaussian_column.begin(), basis.gaussian_column.end());
    }
    if (prepared_->compression.identity) {
        prepared_compression_offsets_ = {0};
    } else {
        prepared_compression_offsets_.reserve(prepared_->compression.rows.size() + 1);
        prepared_compression_offsets_.push_back(0);
        for (const auto& row : prepared_->compression.rows) {
            for (const auto [source, factor] : row) {
                prepared_compression_sources_.push_back(source);
                prepared_compression_factors_.push_back(factor);
            }
            prepared_compression_offsets_.push_back(
                static_cast<std::int64_t>(prepared_compression_sources_.size()));
        }
    }
}
std::int64_t SoapTurboCalculator::feature_count() const noexcept {
    return compression_feature_count(options_);
}
const std::vector<std::int32_t>& SoapTurboCalculator::species() const noexcept { return options_.species; }
const std::vector<std::int32_t>& SoapTurboCalculator::alpha_max() const noexcept { return prepared_alpha_max_; }
const std::vector<std::int32_t>& SoapTurboCalculator::channel_offsets() const noexcept { return prepared_channel_offsets_; }
const std::vector<std::int32_t>& SoapTurboCalculator::central_allowed() const noexcept { return prepared_central_allowed_; }
const std::vector<std::int32_t>& SoapTurboCalculator::basis_offsets() const noexcept { return prepared_basis_offsets_; }
const std::vector<double>& SoapTurboCalculator::basis_transforms() const noexcept { return prepared_basis_transforms_; }
const std::vector<double>& SoapTurboCalculator::gaussian_columns() const noexcept { return prepared_gaussian_columns_; }
const std::vector<std::int64_t>& SoapTurboCalculator::compression_offsets() const noexcept { return prepared_compression_offsets_; }
const std::vector<std::int64_t>& SoapTurboCalculator::compression_sources() const noexcept { return prepared_compression_sources_; }
const std::vector<double>& SoapTurboCalculator::compression_factors() const noexcept { return prepared_compression_factors_; }
std::int32_t SoapTurboCalculator::packed_count() const noexcept { return prepared_->packed_count; }
std::int64_t SoapTurboCalculator::dense_feature_count() const noexcept { return prepared_->dense_features; }
void SoapTurboCalculator::close() noexcept { closed_.store(true, std::memory_order_release); }
bool SoapTurboCalculator::closed() const noexcept { return closed_.load(std::memory_order_acquire); }
void SoapTurboCalculator::compute(
    const StructureBatchView& batch,
    double* output,
    const std::shared_ptr<ComputeControl>& control) const {
    if (closed()) {
        throw std::runtime_error("SOAPTurbo calculator is closed");
    }
    std::lock_guard<std::mutex> lock(compute_mutex_);
    if (!prepared_) prepared_ = make_soap_turbo_prepared(options_);
    compute_soap_turbo(batch, options_, *prepared_, output, control);
}

} // namespace mdescriptor
