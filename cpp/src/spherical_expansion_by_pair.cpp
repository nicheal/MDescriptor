#include "mdescriptor/local_descriptors.hpp"
#include "mdescriptor/neighbor.hpp"
#include "local_spherical_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace mdescriptor {
using namespace detail;

DescriptorPairTable compute_spherical_expansion_by_pair(
    const StructureBatchView& batch,
    const LocalDescriptorOptions& options,
    const std::shared_ptr<ComputeControl>& control) {
    validate_options(options);
    validate_species(batch, options.species);
    if (control) {
        control->reset(batch.structures);
    }
    const int n_radial = options.max_radial + 1;
    // A pair has exactly one center/neighbor type channel. Keep only that
    // channel in the pair table; the TensorMap reference also stores pair
    // channels separately, and materializing all species-pair zero channels
    // made the dense adapter memory-bound for large feature sets.
    const std::int64_t features = static_cast<std::int64_t>(
        (options.max_angular + 1) * (options.max_angular + 1) * n_radial);
    const NeighborGraph graph = build_neighbor_graph(batch, options.cutoff, control, options.num_threads);
    std::vector<GtoRadialBasis> radial_bases;
    radial_bases.reserve(static_cast<std::size_t>(options.max_angular + 1));
    for (int l = 0; l <= options.max_angular; ++l) {
        radial_bases.emplace_back(n_radial, options.cutoff, l);
    }
    DescriptorPairTable result;
    const std::size_t pair_count = graph.offsets().empty()
        ? 0
        : static_cast<std::size_t>(graph.offsets().back());
    result.values.reserve(pair_count * static_cast<std::size_t>(9 + features));
    result.offsets.push_back(0);
    std::vector<double> harmonics;
    std::vector<double> radial(static_cast<std::size_t>(n_radial));
    std::vector<double> radial_raw(static_cast<std::size_t>(n_radial));
    for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
        check_cancelled(control);
        const std::int64_t begin = batch.offsets[structure];
        const std::int64_t end = batch.offsets[structure + 1];
        for (std::int64_t center = begin; center < end; ++center) {
            const NeighborView neighbors = graph.for_center(center);
            for (std::size_t index = 0; index < neighbors.size; ++index) {
                const std::int64_t neighbor = neighbors.atoms[index];
                const double distance = std::sqrt(std::max(0.0, neighbors.distance2[index]));
                const double scaling = cutoff_value(distance, options.cutoff);
                const std::size_t row_start = result.values.size();
                result.values.resize(row_start + static_cast<std::size_t>(9 + features), 0.0);
                double* row = result.values.data() + row_start;
                row[0] = static_cast<double>(center);
                row[1] = static_cast<double>(neighbor);
                row[2] = static_cast<double>(neighbors.shifts[index * 3 + 0]);
                row[3] = static_cast<double>(neighbors.shifts[index * 3 + 1]);
                row[4] = static_cast<double>(neighbors.shifts[index * 3 + 2]);
                row[5] = neighbors.displacements[index * 3 + 0];
                row[6] = neighbors.displacements[index * 3 + 1];
                row[7] = neighbors.displacements[index * 3 + 2];
                row[8] = distance;
                if (scaling != 0.0) {
                    std::array<double, 3> displacement{
                        neighbors.displacements[index * 3 + 0],
                        neighbors.displacements[index * 3 + 1],
                        neighbors.displacements[index * 3 + 2],
                    };
                    real_spherical_harmonics(displacement, options.max_angular, harmonics);
                    for (int l = 0; l <= options.max_angular; ++l) {
                        radial_bases[static_cast<std::size_t>(l)].radial_integral_into(
                            distance, l, options.density_width, radial, radial_raw);
                        for (int m = -l; m <= l; ++m) {
                            const std::size_t offset = static_cast<std::size_t>(l * l + l + m)
                                * static_cast<std::size_t>(n_radial);
                            for (int n = 0; n < n_radial; ++n) {
                                row[9 + offset + static_cast<std::size_t>(n)] = scaling * radial[static_cast<std::size_t>(n)]
                                    * harmonics[static_cast<std::size_t>(l * l + l + m)];
                            }
                        }
                    }
                }
            }
        }
        result.offsets.push_back(static_cast<std::int64_t>(result.values.size() / static_cast<std::size_t>(9 + features)));
        mark_completed(control);
    }
    check_cancelled(control);
    return result;
}
} // namespace mdescriptor
