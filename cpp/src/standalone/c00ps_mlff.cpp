#include "mdescriptor/descriptor.hpp"
#include "mdescriptor/neighbor.hpp"
#include "descriptor_common.hpp"
#include "local_spherical_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mdescriptor {
namespace {

using detail::Vec3;
using detail::cancelled;
using detail::run_parallel_structures;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kSqrtFourPi = 3.544907701811032054596334966682290365;
// VASP 6.6.0 constructs the radial basis on NR equally spaced points and
// normalizes it with the corresponding right-endpoint sum.  The same grid is
// used here for the one-time basis setup.
constexpr int kVaspRadialGridPoints = 10000;
constexpr int kRadialQuadraturePoints = 240;

double spherical_bessel(int l, double x) {
    const double ax = std::abs(x);
    if (ax < 1e-4) {
        if (l == 0) {
            const double x2 = x * x;
            return 1.0 - x2 / 6.0 + x2 * x2 / 120.0;
        }
        double denominator = 1.0;
        for (int value = 1; value <= l; ++value) {
            denominator *= static_cast<double>(2 * value + 1);
        }
        return std::pow(x, l) / denominator
            * (1.0 - x * x / (2.0 * static_cast<double>(2 * l + 3)));
    }
    const double j0 = std::sin(x) / x;
    if (l == 0) {
        return j0;
    }
    const double j1 = std::sin(x) / (x * x) - std::cos(x) / x;
    if (l == 1) {
        return j1;
    }
    double previous = j0;
    double current = j1;
    for (int n = 1; n < l; ++n) {
        const double next = (2.0 * n + 1.0) * current / x - previous;
        previous = current;
        current = next;
    }
    return current;
}

std::vector<double> bessel_zeros(int l, int count) {
    std::vector<double> roots;
    roots.reserve(static_cast<std::size_t>(count));
    const double step = kPi / 8.0;
    double x_left = 1e-4;
    double y_left = spherical_bessel(l, x_left);
    double x = x_left + step;
    while (static_cast<int>(roots.size()) < count
           && x < static_cast<double>(count + l + 8) * kPi) {
        const double y = spherical_bessel(l, x);
        if (y_left == 0.0 || y_left * y < 0.0) {
            double lo = x - step;
            double hi = x;
            for (int iteration = 0; iteration < 64; ++iteration) {
                const double mid = 0.5 * (lo + hi);
                const double y_mid = spherical_bessel(l, mid);
                if (spherical_bessel(l, lo) * y_mid <= 0.0) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            roots.push_back(0.5 * (lo + hi));
        }
        x_left = x;
        y_left = y;
        x += step;
    }
    if (static_cast<int>(roots.size()) != count) {
        throw std::runtime_error("could not construct C00PS-MLFF radial basis zeros");
    }
    return roots;
}

void gauss_legendre(int count, std::vector<double>& nodes, std::vector<double>& weights) {
    nodes.resize(static_cast<std::size_t>(count));
    weights.resize(static_cast<std::size_t>(count));
    const double tolerance = 3e-14;
    const int half = (count + 1) / 2;
    for (int i = 0; i < half; ++i) {
        double z = std::cos(kPi * (static_cast<double>(i) + 0.75) / (static_cast<double>(count) + 0.5));
        double derivative = 0.0;
        for (int iteration = 0; iteration < 100; ++iteration) {
            double p_previous = 1.0;
            double p_current = z;
            for (int order = 2; order <= count; ++order) {
                const double p_next = (
                    (2.0 * order - 1.0) * z * p_current
                    - (order - 1.0) * p_previous
                ) / static_cast<double>(order);
                p_previous = p_current;
                p_current = p_next;
            }
            derivative = static_cast<double>(count) * (z * p_current - p_previous) / (z * z - 1.0);
            const double next = z - p_current / derivative;
            if (std::abs(next - z) < tolerance) {
                z = next;
                break;
            }
            z = next;
        }
        nodes[static_cast<std::size_t>(i)] = -z;
        nodes[static_cast<std::size_t>(count - 1 - i)] = z;
        const double weight = 2.0 / ((1.0 - z * z) * derivative * derivative);
        weights[static_cast<std::size_t>(i)] = weight;
        weights[static_cast<std::size_t>(count - 1 - i)] = weight;
    }
}

double double_factorial_odd(int l) {
    double value = 1.0;
    for (int factor = 1; factor <= l; ++factor) {
        value *= static_cast<double>(2 * factor + 1);
    }
    return value;
}

// VASP's IL0 returns i_l(x) multiplied by exp(-W*s^2-W*r^2), evaluated in a
// form that remains finite for the Gaussian broadening integral.
double scaled_modified_spherical_bessel(int l, double x, double w, double sample, double radius) {
    const double gaussian = std::exp(-w * (sample * sample + radius * radius));
    if (x * x <= 1e-8) {
        return gaussian * std::pow(x, l) / double_factorial_odd(l);
    }
    const double plus = std::exp(-w * (sample - radius) * (sample - radius));
    const double minus = std::exp(-w * (sample + radius) * (sample + radius));
    double previous = 0.5 * (plus - minus) / x;
    if (l == 0) {
        return previous;
    }
    double current = (0.5 * x * (plus + minus) - 0.5 * (plus - minus)) / (x * x);
    if (l == 1) {
        return current;
    }
    for (int degree = 2; degree <= l; ++degree) {
        const double next = previous - (2.0 * degree - 1.0) / x * current;
        previous = current;
        current = next;
    }
    return current;
}

double legendre(int l, double x) {
    x = std::max(-1.0, std::min(1.0, x));
    if (l == 0) {
        return 1.0;
    }
    if (l == 1) {
        return x;
    }
    double previous = 1.0;
    double current = x;
    for (int n = 1; n < l; ++n) {
        const double next = (
            (2.0 * n + 1.0) * x * current - n * previous
        ) / static_cast<double>(n + 1);
        previous = current;
        current = next;
    }
    return current;
}

double cutoff_value(int kind, double distance, double r_cut) {
    if (distance > r_cut) {
        return 0.0;
    }
    if (kind == 0) {
        return 0.5 * (std::cos(kPi * distance / r_cut) + 1.0);
    }
    if (kind == 1) {
        const double x = 4.0 * distance / r_cut - 3.0;
        if (x < -1.0) {
            return 1.0;
        }
        if (x < 1.0) {
            return 0.25 * (x * x * x - 3.0 * x + 2.0);
        }
        return 0.0;
    }
    if (kind == 2 || kind == 3) {
        constexpr double delta = 0.5; // Reference cutoff's 0.5 Angstrom transition width.
        const double r1 = r_cut > delta ? r_cut - delta : 0.5 * r_cut;
        double value = 1.0;
        if (distance > r1) {
            const double rc2 = r_cut * r_cut;
            const double r2 = distance * distance;
            value = (rc2 - r2) * (rc2 - r2)
                * (rc2 + 2.0 * r2 - 3.0 * r1 * r1)
                / std::pow(rc2 - r1 * r1, 3.0);
        }
        if (kind == 3) {
            value /= 1.0 + std::pow(distance / 2.0, 7.0); // Reference cutoff's R0 = 2 Angstrom.
        }
        return value;
    }
    throw std::invalid_argument("unknown C00PS-MLFF cutoff function");
}

void validate_options(const C00PSMlffOptions& options) {
    if (options.species.empty()) {
        throw std::invalid_argument("C00PS-MLFF species must not be empty");
    }
    if (!std::isfinite(options.r_cut) || options.r_cut <= 0.0
        || options.n_radial <= 0 || options.l_max < 0) {
        throw std::invalid_argument("invalid C00PS-MLFF radial or angular parameters");
    }
    if (!std::isfinite(options.radial_sigma) || options.radial_sigma < 0.0) {
        throw std::invalid_argument("C00PS-MLFF radial_sigma must be finite and non-negative");
    }
    if (options.cutoff_function < 0 || options.cutoff_function > 3) {
        throw std::invalid_argument("invalid C00PS-MLFF cutoff function");
    }
    if (options.radial_weight < 0.0 || options.angular_weight < 0.0
        || !std::isfinite(options.radial_weight) || !std::isfinite(options.angular_weight)) {
        throw std::invalid_argument("C00PS-MLFF weights must be finite and non-negative");
    }
    for (const auto species : options.species) {
        if (species <= 0) {
            throw std::invalid_argument("C00PS-MLFF species must be positive");
        }
    }
    (void)detail::species_map(options.species);
    if (!options.include_radial && !options.include_angular) {
        throw std::invalid_argument("at least one C00PS-MLFF descriptor block must be enabled");
    }
}

void prepare_basis(
    const C00PSMlffOptions& options,
    std::vector<std::vector<double>>& zeros,
    std::vector<std::vector<double>>& norms,
    std::vector<std::vector<double>>& radial_values,
    std::vector<std::int32_t>& radial_counts
) {
    if (!zeros.empty()) {
        return;
    }
    zeros.resize(static_cast<std::size_t>(options.l_max + 1));
    norms.resize(static_cast<std::size_t>(options.l_max + 1));
    radial_values.clear();
    radial_values.resize(static_cast<std::size_t>(options.l_max + 1));
    for (int l = 0; l <= options.l_max; ++l) {
        zeros[static_cast<std::size_t>(l)] = bessel_zeros(l, options.n_radial);
    }
    // C00PS-MLFF uses one common q-grid upper bound for all angular channels. The
    // resulting nrb(l) can be smaller than MRB for higher angular momentum.
    radial_counts.resize(static_cast<std::size_t>(options.l_max + 1));
    double qmax = std::numeric_limits<double>::infinity();
    for (const auto& l_zeros : zeros) {
        qmax = std::min(qmax, l_zeros.back());
    }
    for (int l = 0; l <= options.l_max; ++l) {
        auto& l_zeros = zeros[static_cast<std::size_t>(l)];
        auto& l_norms = norms[static_cast<std::size_t>(l)];
        const auto count = static_cast<std::int32_t>(std::count_if(
            l_zeros.begin(), l_zeros.end(), [qmax](double root) { return root <= qmax; }));
        radial_counts[static_cast<std::size_t>(l)] = std::max<std::int32_t>(1, count);
        l_norms.resize(static_cast<std::size_t>(radial_counts[static_cast<std::size_t>(l)]));
        for (int n = 0; n < radial_counts[static_cast<std::size_t>(l)]; ++n) {
            const double root = zeros[static_cast<std::size_t>(l)][static_cast<std::size_t>(n)];
            double norm2 = 0.0;
            const double dr = options.r_cut / static_cast<double>(kVaspRadialGridPoints);
            for (int point = 1; point <= kVaspRadialGridPoints; ++point) {
                const double radius = static_cast<double>(point) * dr;
                const double basis = spherical_bessel(l, root * radius / options.r_cut);
                norm2 += radius * radius * basis * basis * dr;
            }
            l_norms[static_cast<std::size_t>(n)] = std::sqrt(norm2);
        }
    }

    if (options.radial_sigma <= 0.0) {
        return;
    }

    // RAD_FUNC in VASP first convolves every normalized spherical Bessel
    // function with the Gaussian atom distribution and then tabulates the
    // result on the same radial mesh.  The integrand vanishes at both mesh
    // endpoints for the Bessel basis, so high-order Gauss integration matches
    // VASP's right-endpoint grid sum to the precision needed here while
    // avoiding an O(NR^2) setup.
    std::vector<double> nodes;
    std::vector<double> weights;
    gauss_legendre(kRadialQuadraturePoints, nodes, weights);
    const double width_parameter = 0.5 / (options.radial_sigma * options.radial_sigma);
    const double prefactor = 4.0 * kPi * std::pow(width_parameter / kPi, 1.5);
    const double quadrature_scale = 0.5 * options.r_cut;
    const std::size_t table_width = static_cast<std::size_t>(kVaspRadialGridPoints + 1);
    for (int l = 0; l <= options.l_max; ++l) {
        const std::int32_t count = radial_counts[static_cast<std::size_t>(l)];
        auto& table = radial_values[static_cast<std::size_t>(l)];
        table.assign(static_cast<std::size_t>(count) * table_width, 0.0);
        std::vector<std::vector<double>> basis_values(
            static_cast<std::size_t>(count),
            std::vector<double>(nodes.size(), 0.0));
        for (int n = 0; n < count; ++n) {
            const double root = zeros[static_cast<std::size_t>(l)][static_cast<std::size_t>(n)];
            const double norm = norms[static_cast<std::size_t>(l)][static_cast<std::size_t>(n)];
            for (std::size_t q = 0; q < nodes.size(); ++q) {
                const double radius = quadrature_scale * (nodes[q] + 1.0);
                basis_values[static_cast<std::size_t>(n)][q]
                    = spherical_bessel(l, root * radius / options.r_cut) / norm;
            }
        }
        for (int point = 0; point <= kVaspRadialGridPoints; ++point) {
            const double sample = options.r_cut * static_cast<double>(point)
                / static_cast<double>(kVaspRadialGridPoints);
            const double cutoff = cutoff_value(options.cutoff_function, sample, options.r_cut);
            for (int n = 0; n < count; ++n) {
                double integral = 0.0;
                for (std::size_t q = 0; q < nodes.size(); ++q) {
                    const double radius = quadrature_scale * (nodes[q] + 1.0);
                    const double x = 2.0 * width_parameter * sample * radius;
                    integral += weights[q] * basis_values[static_cast<std::size_t>(n)][q]
                        * scaled_modified_spherical_bessel(l, x, width_parameter, sample, radius)
                        * radius * radius;
                }
                table[static_cast<std::size_t>(n) * table_width + static_cast<std::size_t>(point)]
                    = cutoff * prefactor * quadrature_scale * integral;
            }
        }
    }
}

double radial_value(
    const C00PSMlffOptions& options,
    const std::vector<std::vector<double>>& zeros,
    const std::vector<std::vector<double>>& norms,
    const std::vector<std::vector<double>>& radial_values,
    int l,
    int n,
    double distance
) {
    if (options.radial_sigma > 0.0) {
        const std::size_t table_width = static_cast<std::size_t>(kVaspRadialGridPoints + 1);
        const auto& table = radial_values[static_cast<std::size_t>(l)];
        const double coordinate = std::max(0.0, std::min(1.0, distance / options.r_cut))
            * static_cast<double>(kVaspRadialGridPoints);
        const int left = std::min(kVaspRadialGridPoints - 1, static_cast<int>(coordinate));
        const double fraction = coordinate - static_cast<double>(left);
        const auto value = [&](int point) {
            const int bounded = std::max(0, std::min(kVaspRadialGridPoints, point));
            return table[static_cast<std::size_t>(n) * table_width + static_cast<std::size_t>(bounded)];
        };
        // Four-point cubic interpolation keeps the tabulated path smooth and
        // mirrors the cubic spline evaluation used by VASP's SPLVAL_ML_NEW.
        const double p0 = value(left - 1);
        const double p1 = value(left);
        const double p2 = value(left + 1);
        const double p3 = value(left + 2);
        return p1 + 0.5 * fraction * (
            p2 - p0 + fraction * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
                + fraction * (3.0 * (p1 - p2) + p3 - p0)));
    }
    const double basis = spherical_bessel(
        l,
        zeros[static_cast<std::size_t>(l)][static_cast<std::size_t>(n)] * distance / options.r_cut);
    const double norm = norms[static_cast<std::size_t>(l)][static_cast<std::size_t>(n)];
    return cutoff_value(options.cutoff_function, distance, options.r_cut) * basis / norm;
}

} // namespace

std::int64_t c00ps_mlff_feature_count(const C00PSMlffOptions& options) {
    validate_options(options);
    std::vector<std::vector<double>> zeros;
    std::vector<std::vector<double>> norms;
    std::vector<std::vector<double>> radial_values;
    std::vector<std::int32_t> radial_counts;
    prepare_basis(options, zeros, norms, radial_values, radial_counts);
    const std::int64_t radial = options.include_radial
        ? static_cast<std::int64_t>(options.species.size()) * radial_counts[0]
        : 0;
    std::int64_t angular = 0;
    if (options.include_angular) {
        for (const std::int32_t count : radial_counts) {
            const std::int64_t channels = static_cast<std::int64_t>(options.species.size()) * count;
            angular += channels * (channels + 1) / 2;
        }
    }
    return radial + angular;
}

 C00PSMlffCalculator::C00PSMlffCalculator(C00PSMlffOptions options)
    : options_(std::move(options)) {
    validate_options(options_);
    prepare_basis(options_, zeros_, norms_, radial_values_, radial_counts_);
    basis_ready_ = true;
}

std::int64_t C00PSMlffCalculator::feature_count() const noexcept {
    const std::int64_t radial = options_.include_radial
        ? static_cast<std::int64_t>(options_.species.size()) * radial_counts_[0]
        : 0;
    std::int64_t angular = 0;
    if (options_.include_angular) {
        for (const std::int32_t count : radial_counts_) {
            const std::int64_t channels = static_cast<std::int64_t>(options_.species.size()) * count;
            angular += channels * (channels + 1) / 2;
        }
    }
    return radial + angular;
}

const std::vector<std::int32_t>& C00PSMlffCalculator::species() const noexcept {
    return options_.species;
}

const std::vector<std::int32_t>& C00PSMlffCalculator::radial_counts() const noexcept {
    return radial_counts_;
}

void C00PSMlffCalculator::close() noexcept {
    closed_.store(true, std::memory_order_release);
}

bool C00PSMlffCalculator::closed() const noexcept {
    return closed_.load(std::memory_order_acquire);
}

void C00PSMlffCalculator::compute(
    const StructureBatchView& batch,
    double* output,
    const std::shared_ptr<ComputeControl>& control
) const {
    std::lock_guard<std::mutex> lock(compute_mutex_);
    if (closed()) {
        throw std::runtime_error("C00PS-MLFF calculator is closed");
    }
    if (output == nullptr) {
        throw std::invalid_argument("C00PS-MLFF output must not be null");
    }
    detail::validate_batch(batch);
    detail::validate_species(batch, options_.species);
    if (control) {
        control->reset(batch.structures);
    }
    if (!basis_ready_) {
        prepare_basis(options_, zeros_, norms_, radial_values_, radial_counts_);
        basis_ready_ = true;
    }

    const std::int64_t features = feature_count();
    std::fill(output, output + batch.atoms * features, 0.0);
    const auto mapping = detail::species_map(options_.species);
    const std::int64_t radial_channels = static_cast<std::int64_t>(options_.species.size()) * radial_counts_[0];
    std::int64_t angular_features = 0;
    if (options_.include_angular) {
        for (const std::int32_t count : radial_counts_) {
            const std::int64_t channels = static_cast<std::int64_t>(options_.species.size()) * count;
            angular_features += channels * (channels + 1) / 2;
        }
    }
    const NeighborGraph graph = build_neighbor_graph(batch, options_.r_cut, control, options_.num_threads);

    run_parallel_structures(batch.structures, options_.num_threads, control, [&](std::int64_t structure) {
        const std::int64_t begin = batch.offsets[structure];
        const std::int64_t end = batch.offsets[structure + 1];
        for (std::int64_t center = begin; center < end; ++center) {
            if (cancelled(control)) {
                continue;
            }
            const NeighborView neighbors = graph.for_center(center);
            const std::int64_t center_type = mapping.at(batch.numbers[center]);
            std::vector<Vec3> vectors;
            std::vector<double> distances;
            std::vector<std::int64_t> types;
            vectors.reserve(neighbors.size);
            distances.reserve(neighbors.size);
            types.reserve(neighbors.size);
            for (std::size_t index = 0; index < neighbors.size; ++index) {
                const double distance2 = neighbors.distance2[index];
                if (distance2 <= 1e-24) {
                    continue;
                }
                const auto type_it = mapping.find(batch.numbers[neighbors.atoms[index]]);
                if (type_it == mapping.end()) {
                    throw std::invalid_argument("batch contains an atomic number outside calculator species");
                }
                vectors.push_back({
                    neighbors.displacements[index * 3 + 0],
                    neighbors.displacements[index * 3 + 1],
                    neighbors.displacements[index * 3 + 2],
                });
                distances.push_back(std::sqrt(distance2));
                types.push_back(type_it->second);
            }
            if (distances.empty()) {
                continue;
            }

            double* row = output + center * features;
            const std::size_t neighbor_count = distances.size();
            const int coefficient_l_max = options_.include_angular ? options_.l_max : 0;
            std::vector<std::size_t> coefficient_offsets(
                static_cast<std::size_t>(coefficient_l_max + 1), 0);
            std::vector<std::size_t> radial_offsets(
                static_cast<std::size_t>(coefficient_l_max + 1), 0);
            std::size_t coefficient_size = 0;
            std::size_t radial_value_size = 0;
            for (int l = 0; l <= coefficient_l_max; ++l) {
                const std::int32_t radial_count = radial_counts_[static_cast<std::size_t>(l)];
                const std::size_t channels = options_.species.size()
                    * static_cast<std::size_t>(radial_count);
                coefficient_offsets[static_cast<std::size_t>(l)] = coefficient_size;
                coefficient_size += channels * static_cast<std::size_t>(2 * l + 1);
                radial_offsets[static_cast<std::size_t>(l)] = radial_value_size;
                radial_value_size += neighbor_count * static_cast<std::size_t>(radial_count);
            }
            std::vector<double> coefficients(coefficient_size, 0.0);
            std::vector<double> neighbor_radial_values(radial_value_size, 0.0);
            std::vector<double> radial_c00(static_cast<std::size_t>(radial_channels), 0.0);
            std::vector<double> harmonics;
            std::vector<double> harmonic_legendre;
            for (std::size_t neighbor = 0; neighbor < neighbor_count; ++neighbor) {
                if (options_.include_angular) {
                    const Vec3 vector = vectors[neighbor];
                    const std::array<double, 3> displacement{vector.x, vector.y, vector.z};
                    detail::real_spherical_harmonics_into(
                        displacement, options_.l_max, harmonics, harmonic_legendre);
                }
                for (int l = 0; l <= coefficient_l_max; ++l) {
                    const std::int32_t radial_count = radial_counts_[static_cast<std::size_t>(l)];
                    const std::size_t base = static_cast<std::size_t>(types[neighbor])
                        * static_cast<std::size_t>(radial_count);
                    const std::size_t harmonic_width = static_cast<std::size_t>(2 * l + 1);
                    for (int n = 0; n < radial_count; ++n) {
                        const double value = radial_value(
                            options_, zeros_, norms_, radial_values_, l, n, distances[neighbor]);
                        neighbor_radial_values[
                            radial_offsets[static_cast<std::size_t>(l)]
                            + neighbor * static_cast<std::size_t>(radial_count)
                            + static_cast<std::size_t>(n)] = value;
                        if (options_.include_angular) {
                            const std::size_t channel = base + static_cast<std::size_t>(n);
                            double* coefficient = coefficients.data()
                                + coefficient_offsets[static_cast<std::size_t>(l)]
                                + channel * harmonic_width;
                            for (std::size_t m = 0; m < harmonic_width; ++m) {
                                coefficient[m] += value * harmonics[
                                    static_cast<std::size_t>(l * l) + m];
                            }
                        }
                        if (l == 0) {
                            radial_c00[base + static_cast<std::size_t>(n)] += value / kSqrtFourPi;
                        }
                    }
                }
            }

            if (options_.include_radial) {
                std::copy(radial_c00.begin(), radial_c00.end(), row);
                if (options_.normalize_radial) {
                    double norm2 = 0.0;
                    for (const double value : radial_c00) norm2 += value * value;
                    if (norm2 > 1e-20) {
                        const double scale = 1.0 / std::sqrt(norm2);
                        for (std::int64_t channel = 0; channel < radial_channels; ++channel) {
                            row[channel] *= scale;
                        }
                    }
                }
            }

            if (options_.include_angular) {
                const std::size_t angular_offset = options_.include_radial
                    ? static_cast<std::size_t>(radial_channels) : 0;
                std::vector<std::size_t> self_power_offsets(
                    static_cast<std::size_t>(options_.l_max + 1), 0);
                std::vector<double> self_power(static_cast<std::size_t>(angular_features), 0.0);
                std::size_t self_power_size = 0;
                for (int l = 0; l <= options_.l_max; ++l) {
                    const std::int64_t channels = static_cast<std::int64_t>(options_.species.size())
                        * radial_counts_[static_cast<std::size_t>(l)];
                    self_power_offsets[static_cast<std::size_t>(l)] = self_power_size;
                    self_power_size += static_cast<std::size_t>(channels * (channels + 1) / 2);
                }
                if (options_.exclude_self_interaction) {
                    for (int l = 0; l <= options_.l_max; ++l) {
                        const std::int32_t radial_count = radial_counts_[static_cast<std::size_t>(l)];
                        const std::int64_t channels = static_cast<std::int64_t>(options_.species.size())
                            * radial_count;
                        const double addition = (2.0 * l + 1.0) / (4.0 * kPi);
                        for (std::size_t neighbor = 0; neighbor < neighbor_count; ++neighbor) {
                            // Match VASP 6.6.0 MLFF LSIC: only the centre
                            // species receives a self-interaction correction.
                            if (types[neighbor] != center_type) {
                                continue;
                            }
                            const std::int64_t first_base = types[neighbor] * radial_count;
                            const double* values = neighbor_radial_values.data()
                                + radial_offsets[static_cast<std::size_t>(l)]
                                + neighbor * static_cast<std::size_t>(radial_count);
                            for (int first = 0; first < radial_count; ++first) {
                                const std::int64_t first_channel = first_base + first;
                                for (int second = first; second < radial_count; ++second) {
                                    const std::int64_t second_channel = first_base + second;
                                    const std::size_t pair_index = static_cast<std::size_t>(
                                        first_channel * channels
                                        - first_channel * (first_channel - 1) / 2
                                        + second_channel - first_channel);
                                    self_power[self_power_offsets[static_cast<std::size_t>(l)] + pair_index]
                                        += addition * values[first] * values[second];
                                }
                            }
                        }
                    }
                }
                std::size_t angular_index = 0;
                for (int l = 0; l <= options_.l_max; ++l) {
                    const std::int32_t radial_count = radial_counts_[static_cast<std::size_t>(l)];
                    const std::int64_t channels = static_cast<std::int64_t>(options_.species.size()) * radial_count;
                    const double prefactor = std::sqrt(8.0 * kPi * kPi / (2.0 * l + 1.0));
                    const std::size_t harmonic_width = static_cast<std::size_t>(2 * l + 1);
                    for (std::int64_t first = 0; first < channels; ++first) {
                        for (std::int64_t second = first; second < channels; ++second) {
                            double total = 0.0;
                            const double* first_coeff = coefficients.data()
                                + coefficient_offsets[static_cast<std::size_t>(l)]
                                + static_cast<std::size_t>(first) * harmonic_width;
                            const double* second_coeff = coefficients.data()
                                + coefficient_offsets[static_cast<std::size_t>(l)]
                                + static_cast<std::size_t>(second) * harmonic_width;
                            for (std::size_t m = 0; m < harmonic_width; ++m) {
                                total += first_coeff[m] * second_coeff[m];
                            }
                            const std::size_t pair_index = static_cast<std::size_t>(
                                first * channels - first * (first - 1) / 2 + second - first);
                            if (options_.exclude_self_interaction) {
                                total -= self_power[
                                    self_power_offsets[static_cast<std::size_t>(l)] + pair_index];
                            }
                            // VASP's WVAR distinguishes radial indices, not
                            // flattened species/radial channels.  Therefore
                            // cross-species channels with equal radial index
                            // retain weight 1.0.
                            const int first_radial = static_cast<int>(first % radial_count);
                            const int second_radial = static_cast<int>(second % radial_count);
                            const double radial_pair_weight = first_radial == second_radial
                                ? 1.0 : std::sqrt(2.0);
                            row[angular_offset + angular_index++] = radial_pair_weight * prefactor * total;
                        }
                    }
                }
                if (options_.normalize_angular) {
                    double norm2 = 0.0;
                    for (std::int64_t index = 0; index < angular_features; ++index) {
                        const double value = row[angular_offset + static_cast<std::size_t>(index)];
                        norm2 += value * value;
                    }
                    if (norm2 > 1e-20) {
                        const double scale = 1.0 / std::sqrt(norm2);
                        for (std::int64_t index = 0; index < angular_features; ++index) {
                            row[angular_offset + static_cast<std::size_t>(index)] *= scale;
                        }
                    }
                }
            }

            if (options_.super_vector) {
                const std::int64_t radial_end = options_.include_radial ? radial_channels : 0;
                double norm2 = 0.0;
                if (options_.include_radial) {
                    const double scale = std::sqrt(options_.radial_weight);
                    for (std::int64_t index = 0; index < radial_end; ++index) {
                        row[index] *= scale;
                    }
                }
                if (options_.include_angular) {
                    const double scale = std::sqrt(options_.angular_weight);
                    for (std::int64_t index = radial_end; index < features; ++index) {
                        row[index] *= scale;
                    }
                }
                for (std::int64_t index = 0; index < features; ++index) {
                    norm2 += row[index] * row[index];
                }
                if (norm2 > 1e-20) {
                    const double scale = 1.0 / std::sqrt(norm2);
                    for (std::int64_t index = 0; index < features; ++index) {
                        row[index] *= scale;
                    }
                }
            }
        }
    });
}

} // namespace mdescriptor
