#include "mdescriptor/local_descriptors.hpp"
#include "mdescriptor/neighbor.hpp"
#include "local_spherical_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace mdescriptor {
using namespace detail;

namespace {

void assemble_spherical_expansion_row(
    double* row,
    std::size_t center_type,
    const std::vector<double>& coefficients,
    std::size_t species_count,
    std::size_t radial_count,
    int max_angular) {
    const std::size_t angular_count = static_cast<std::size_t>(max_angular + 1);
    const std::size_t angular_block = angular_count * angular_count;
    const std::size_t species_block = radial_count * angular_block;
    const std::size_t center_block = species_count * radial_count
        * angular_block;
    row += center_type * center_block;
    std::size_t offset = 0;
    for (std::size_t neighbor_type = 0; neighbor_type < species_count; ++neighbor_type) {
        const double* species_coefficients = coefficients.data() + neighbor_type * species_block;
        for (int angular = 0; angular <= max_angular; ++angular) {
            for (int m = -angular; m <= angular; ++m) {
                const std::size_t harmonic = static_cast<std::size_t>(
                    angular * angular + angular + m);
                const double* harmonic_coefficients = species_coefficients + harmonic;
                for (int radial = 0; radial < static_cast<int>(radial_count); ++radial) {
                    row[offset++] = harmonic_coefficients[
                        static_cast<std::size_t>(radial) * angular_block];
                }
            }
        }
    }
}

void assemble_soap_radial_spectrum_row(
    double* row,
    std::size_t center_type,
    const std::vector<double>& coefficients,
    std::size_t species_count,
    std::size_t radial_count) {
    row += center_type * species_count * radial_count;
    for (std::size_t neighbor_type = 0; neighbor_type < species_count; ++neighbor_type) {
        std::copy_n(
            coefficients.data() + neighbor_type * radial_count,
            radial_count,
            row + neighbor_type * radial_count);
    }
}

void assemble_soap_power_spectrum_row(
    double* row,
    std::size_t center_type,
    const std::vector<double>& coefficients,
    std::size_t species_count,
    std::size_t radial_count,
    int max_angular,
    const std::vector<double>& angular_scales) {
    const std::size_t angular_count = static_cast<std::size_t>(max_angular + 1);
    const std::size_t angular_block = angular_count * angular_count;
    const std::size_t species_block = radial_count * angular_block;
    const std::size_t pair_count = species_count * (species_count + 1) / 2;
    const std::size_t center_block = pair_count * angular_count
        * radial_count * radial_count;
    row += center_type * center_block;
    std::size_t offset = 0;
    const int radial_count_int = static_cast<int>(radial_count);
    for (std::size_t first = 0; first < species_count; ++first) {
        for (std::size_t second = first; second < species_count; ++second) {
            const double* first_coefficients = coefficients.data() + first * species_block;
            const double* second_coefficients = coefficients.data() + second * species_block;
            const double pair_scale = first == second ? 1.0 : kSqrt2;
            for (int angular = 0; angular <= max_angular; ++angular) {
                const double scale = pair_scale
                    * angular_scales[static_cast<std::size_t>(angular)];
                const std::size_t harmonic_start = static_cast<std::size_t>(angular * angular);
                for (int first_radial = 0; first_radial < radial_count_int; ++first_radial) {
                    const double* first_harmonics = first_coefficients
                        + static_cast<std::size_t>(first_radial) * angular_block
                        + harmonic_start;
                    for (int second_radial = 0; second_radial < radial_count_int; ++second_radial) {
                        const double* second_harmonics = second_coefficients
                            + static_cast<std::size_t>(second_radial) * angular_block
                            + harmonic_start;
                        double value = 0.0;
                        for (int m = 0; m < 2 * angular + 1; ++m) {
                            value += first_harmonics[static_cast<std::size_t>(m)]
                                * second_harmonics[static_cast<std::size_t>(m)];
                        }
                        row[offset++] = scale * value;
                    }
                }
            }
        }
    }
}

} // namespace

std::int64_t local_descriptor_feature_count(const LocalDescriptorOptions& options, LocalDescriptorKind kind) {
    validate_options(options);
    if (options.species.empty()) {
        throw std::invalid_argument("species must not be empty");
    }
    return detail::local_feature_count(options, kind);
}

void compute_spherical_expansion(
    const StructureBatchView& batch,
    const LocalDescriptorOptions& options,
    LocalDescriptorKind kind,
    double* output,
    const std::shared_ptr<ComputeControl>& control) {
    validate_options(options);
    validate_species(batch, options.species);
    if (control) {
        control->reset(batch.structures);
    }
    const TypeMap mapping = make_type_map(options.species);
    const auto atom_types = make_atom_types(batch, mapping);
    if (kind == LocalDescriptorKind::LodeSphericalExpansion) {
        if (options.exponent > 9) {
            throw std::invalid_argument("LODE exponent must be between 1 and 9");
        }
        for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
            if (make_k_vectors(batch.cells + structure * 9, options.k_cutoff).empty()) {
                throw std::invalid_argument("no LODE reciprocal vectors for the current cell and k_cutoff");
            }
        }
        compute_lode_values(batch, options, atom_types, output, control);
        check_cancelled(control);
        return;
    }
    const NeighborGraph graph = build_neighbor_graph(batch, options.cutoff, control, options.num_threads);
    const int n_radial = options.max_radial + 1;
    const int coefficient_max_angular = local_coefficient_max_angular(options, kind);
    const std::int64_t features = local_feature_count(options, kind);
    LocalDescriptorOptions coefficient_options = options;
    coefficient_options.max_angular = coefficient_max_angular;
    std::fill(output, output + batch.atoms * features, 0.0);
    static thread_local RadialBasisSet cached_radial_bases;
    if (!cached_radial_bases.matches(
            options.max_radial, coefficient_max_angular, options.cutoff)) {
        cached_radial_bases.reset(
            options.max_radial, coefficient_max_angular, options.cutoff);
    }
    cached_radial_bases.prepare_density(options.density_width);
    const auto& radial_bases = cached_radial_bases;
    const std::size_t species_count = options.species.size();
    const std::size_t radial_count = static_cast<std::size_t>(n_radial);
    std::vector<double> power_l_scale;
    if (kind == LocalDescriptorKind::SoapPowerSpectrum) {
        power_l_scale.resize(static_cast<std::size_t>(options.max_angular + 1));
        for (int angular = 0; angular <= options.max_angular; ++angular) {
            power_l_scale[static_cast<std::size_t>(angular)]
                = (angular % 2 == 0 ? 1.0 : -1.0)
                / std::sqrt(2.0 * angular + 1.0);
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options.num_threads > 0 ? options.num_threads : omp_get_max_threads())
#endif
    for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
        if (control && control->cancelled()) {
            continue;
        }
        const std::int64_t begin = batch.offsets[structure];
        const std::int64_t end = batch.offsets[structure + 1];
        std::vector<double> coefficients;
        std::vector<double> harmonics;
        std::vector<double> legendre;
        std::vector<double> radial;
        std::vector<double> radial_raw;
        coefficients.reserve(static_cast<std::size_t>(local_coefficient_count(
            coefficient_options, LocalDescriptorKind::SphericalExpansion)));
        for (std::int64_t center = begin; center < end; ++center) {
            const auto center_type = static_cast<std::size_t>(atom_types[static_cast<std::size_t>(center)]);
            compute_coefficients_into(
                graph, center, coefficient_options, atom_types, radial_bases,
                coefficients, harmonics, legendre, radial, radial_raw);
            double* row = output + center * features;
            if (kind == LocalDescriptorKind::SoapRadialSpectrum) {
                assemble_soap_radial_spectrum_row(
                    row, center_type, coefficients, species_count, radial_count);
            } else if (kind == LocalDescriptorKind::SoapPowerSpectrum) {
                assemble_soap_power_spectrum_row(
                    row, center_type, coefficients, species_count, radial_count,
                    options.max_angular, power_l_scale);
            } else {
                assemble_spherical_expansion_row(
                    row, center_type, coefficients, species_count, radial_count,
                    options.max_angular);
            }
        }
        mark_completed(control);
    }
    check_cancelled(control);
}
} // namespace mdescriptor
