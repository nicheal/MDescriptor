#include "mdescriptor/local_descriptors.hpp"
#include "mdescriptor/neighbor.hpp"
#include "local_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mdescriptor {
using namespace detail;

void compute_sorted_distances(
    const StructureBatchView& batch,
    const LocalDescriptorOptions& options,
    int max_neighbors,
    bool separate_neighbor_types,
    double* output,
    const std::shared_ptr<ComputeControl>& control) {
    validate_options(options);
    validate_species(batch, options.species);
    if (max_neighbors <= 0) {
        throw std::invalid_argument("max_neighbors must be positive");
    }
    if (control) {
        control->reset(batch.structures);
    }
    const TypeMap mapping = make_type_map(options.species);
    const std::int64_t columns = separate_neighbor_types
        ? static_cast<std::int64_t>(options.species.size()) * max_neighbors
        : max_neighbors;
    std::fill(output, output + batch.atoms * columns, 0.0);
    const NeighborGraph graph = build_neighbor_graph(batch, options.cutoff, control, options.num_threads);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options.num_threads > 0 ? options.num_threads : omp_get_max_threads())
#endif
    for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
        if (control && control->cancelled()) {
            continue;
        }
        const std::int64_t begin = batch.offsets[structure];
        const std::int64_t end = batch.offsets[structure + 1];
        for (std::int64_t center = begin; center < end; ++center) {
            std::vector<std::vector<double>> distances(options.species.size());
            const NeighborView neighbors = graph.for_center(center);
            for (std::size_t index = 0; index < neighbors.size; ++index) {
                if (neighbors.exact_self(index, center)) {
                    continue;
                }
                const auto type = mapping.find(batch.numbers[neighbors.atoms[index]])->second;
                distances[type].push_back(std::sqrt(std::max(0.0, neighbors.distance2[index])));
            }
            if (separate_neighbor_types) {
                for (std::size_t type = 0; type < distances.size(); ++type) {
                    auto& values = distances[type];
                    std::sort(values.begin(), values.end());
                    const auto offset = center * columns + static_cast<std::int64_t>(type) * max_neighbors;
                    for (int index = 0; index < max_neighbors; ++index) {
                        if (index < static_cast<int>(values.size())) {
                            output[offset + index] = values[static_cast<std::size_t>(index)];
                        } else if (!values.empty()) {
                            output[offset + index] = options.cutoff;
                        }
                    }
                }
            } else {
                std::vector<double> values;
                for (auto& part : distances) {
                    values.insert(values.end(), part.begin(), part.end());
                }
                std::sort(values.begin(), values.end());
                const auto offset = center * columns;
                for (int index = 0; index < max_neighbors; ++index) {
                    if (index < static_cast<int>(values.size())) {
                        output[offset + index] = values[static_cast<std::size_t>(index)];
                    } else if (!values.empty()) {
                        output[offset + index] = options.cutoff;
                    }
                }
            }
        }
        mark_completed(control);
    }
    check_cancelled(control);
}
} // namespace mdescriptor
