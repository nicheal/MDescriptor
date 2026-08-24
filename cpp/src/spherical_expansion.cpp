#include "mdescriptor/featomic.hpp"
#include "mdescriptor/neighbor.hpp"
#include "featomic_spherical_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace mdescriptor {
using namespace detail;

std::int64_t featomic_feature_count(const FeatomicOptions& options, FeatomicKind kind) {
    validate_options(options);
    if (options.species.empty()) {
        throw std::invalid_argument("species must not be empty");
    }
    return local_feature_count(options, kind);
}

void compute_featomic_spherical(
    const StructureBatchView& batch,
    const FeatomicOptions& options,
    FeatomicKind kind,
    double* output,
    const std::shared_ptr<ComputeControl>& control) {
    validate_options(options);
    validate_species(batch, options.species);
    if (control) {
        control->reset(batch.structures);
    }
    const TypeMap mapping = make_type_map(options.species);
    const auto atom_types = make_atom_types(batch, mapping);
    if (kind == FeatomicKind::LodeSphericalExpansion) {
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
    const int max_angular = kind == FeatomicKind::SoapRadialSpectrum ? 0 : options.max_angular;
    const std::int64_t features = local_feature_count(options, kind);
    FeatomicOptions coefficient_options = options;
    coefficient_options.max_angular = max_angular;
    std::fill(output, output + batch.atoms * features, 0.0);
    static thread_local int cached_max_radial = -1;
    static thread_local int cached_max_angular = -1;
    static thread_local double cached_radius = 0.0;
    static thread_local std::vector<GtoRadialBasis> cached_radial_bases;
    if (cached_max_radial != options.max_radial
        || cached_max_angular != options.max_angular
        || cached_radius != options.cutoff) {
        cached_radial_bases.clear();
        cached_radial_bases.reserve(static_cast<std::size_t>(options.max_angular + 1));
        for (int l = 0; l <= options.max_angular; ++l) {
            cached_radial_bases.emplace_back(n_radial, options.cutoff, l);
        }
        cached_max_radial = options.max_radial;
        cached_max_angular = options.max_angular;
        cached_radius = options.cutoff;
    }
    const auto& radial_bases = cached_radial_bases;
    const std::size_t species_count = options.species.size();
    const std::size_t radial_count = static_cast<std::size_t>(n_radial);
    const std::size_t angular_count = static_cast<std::size_t>(options.max_angular + 1);
    const std::size_t power_pair_count = species_count * (species_count + 1) / 2;
    const std::size_t power_pair_stride = angular_count * radial_count * radial_count;
    const std::size_t power_center_stride = power_pair_count * power_pair_stride;
    std::vector<double> power_l_scale(angular_count, 0.0);
    for (int l = 0; l <= options.max_angular; ++l) {
        power_l_scale[static_cast<std::size_t>(l)]
            = (l % 2 == 0 ? 1.0 : -1.0) / std::sqrt(2.0 * l + 1.0);
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
        const std::size_t coefficient_size = species_count * radial_count
            * angular_count * static_cast<std::size_t>(2 * options.max_angular + 1);
        coefficients.reserve(coefficient_size);
        for (std::int64_t center = begin; center < end; ++center) {
            const auto center_type = static_cast<std::size_t>(atom_types[static_cast<std::size_t>(center)]);
            compute_coefficients_into(
                batch, graph, center, coefficient_options, atom_types, radial_bases, false,
                coefficients, harmonics, legendre, radial, radial_raw);
            double* row = output + center * features;
            if (kind == FeatomicKind::SoapRadialSpectrum) {
                row += center_type * species_count * radial_count;
                for (std::size_t neighbor_type = 0; neighbor_type < species_count; ++neighbor_type) {
                    for (int n = 0; n < n_radial; ++n) {
                        row[neighbor_type * radial_count + static_cast<std::size_t>(n)]
                            = coefficients[coefficient_index(neighbor_type, n, 0, 0, n_radial, coefficient_options.max_angular)];
                        }
                    }
            } else if (kind == FeatomicKind::SoapPowerSpectrum) {
                row += center_type * power_center_stride;
                std::size_t offset = 0;
                const int coefficient_width = 2 * options.max_angular + 1;
                for (std::size_t first = 0; first < species_count; ++first) {
                    for (std::size_t second = first; second < species_count; ++second) {
                        const double pair_scale = first == second ? 1.0 : kSqrt2;
                        for (int l = 0; l <= options.max_angular; ++l) {
                            const double scale = pair_scale * power_l_scale[static_cast<std::size_t>(l)];
                            const std::size_t m_offset = static_cast<std::size_t>(options.max_angular - l);
                            for (int n1 = 0; n1 < n_radial; ++n1) {
                                const double* first_coeff = coefficients.data()
                                    + (((first * radial_count + static_cast<std::size_t>(n1)) * angular_count
                                        + static_cast<std::size_t>(l)) * static_cast<std::size_t>(coefficient_width)
                                       + m_offset);
                                for (int n2 = 0; n2 < n_radial; ++n2) {
                                    const double* second_coeff = coefficients.data()
                                        + (((second * radial_count + static_cast<std::size_t>(n2)) * angular_count
                                            + static_cast<std::size_t>(l)) * static_cast<std::size_t>(coefficient_width)
                                           + m_offset);
                                    double value = 0.0;
                                    for (int m = 0; m < 2 * l + 1; ++m) {
                                        value += first_coeff[m] * second_coeff[m];
                                    }
                                    row[offset++] = scale * value;
                                }
                            }
                        }
                    }
                }
            } else {
                const std::size_t center_block = species_count * radial_count
                    * static_cast<std::size_t>(max_angular + 1) * static_cast<std::size_t>(max_angular + 1);
                row += center_type * center_block;
                std::size_t offset = 0;
                for (std::size_t neighbor_type = 0; neighbor_type < species_count; ++neighbor_type) {
                    for (int l = 0; l <= max_angular; ++l) {
                        for (int m = -l; m <= l; ++m) {
                            for (int n = 0; n < n_radial; ++n) {
                                row[offset++] = coefficients[coefficient_index(neighbor_type, n, l, m, n_radial, options.max_angular)];
                            }
                        }
                    }
                }
            }
        }
        mark_completed(control);
    }
    check_cancelled(control);
}
} // namespace mdescriptor
