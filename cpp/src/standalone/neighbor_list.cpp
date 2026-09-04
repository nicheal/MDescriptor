#include "mdescriptor/local_descriptors.hpp"
#include "mdescriptor/neighbor.hpp"
#include "mdescriptor/detail/neighbor_filter.hpp"
#include "descriptor_common.hpp"
#include "local_common.hpp"

#include <cmath>
#include <cstddef>
#include <iterator>
#include <vector>

namespace mdescriptor {
using namespace detail;

DescriptorPairTable compute_neighbor_list(
    const StructureBatchView& batch,
    double cutoff,
    bool full_neighbor_list,
    bool self_pairs,
    int num_threads,
    const std::shared_ptr<ComputeControl>& control) {
    if (!std::isfinite(cutoff) || cutoff <= 0.0) {
        throw std::invalid_argument("neighbor cutoff must be finite and positive");
    }
    if (num_threads < 0) {
        throw std::invalid_argument("num_threads must be non-negative");
    }
    if (control) {
        control->reset(batch.structures);
    }
    const NeighborGraph graph = build_neighbor_graph(batch, cutoff, control, num_threads);
    std::vector<std::vector<double>> per_structure(static_cast<std::size_t>(batch.structures));
    run_parallel_structures(batch.structures, num_threads, control, [&](std::int64_t structure) {
        std::vector<double>& values = per_structure[static_cast<std::size_t>(structure)];
        const std::int64_t begin = batch.offsets[structure];
        const std::int64_t end = batch.offsets[structure + 1];
        for (std::int64_t center = begin; center < end; ++center) {
            const NeighborView neighbors = graph.for_center(center);
            for (std::size_t index = 0; index < neighbors.size; ++index) {
                if (!self_pairs && neighbors.exact_self(index, center)) {
                    continue;
                }
                const std::int64_t atom = neighbors.atoms[index];
                if (!full_neighbor_list && !keep_half_neighbor(
                    center, atom, neighbors.shifts[index * 3 + 0],
                    neighbors.shifts[index * 3 + 1], neighbors.shifts[index * 3 + 2])) {
                    continue;
                }
                values.push_back(static_cast<double>(center));
                values.push_back(static_cast<double>(atom));
                values.push_back(static_cast<double>(neighbors.shifts[index * 3 + 0]));
                values.push_back(static_cast<double>(neighbors.shifts[index * 3 + 1]));
                values.push_back(static_cast<double>(neighbors.shifts[index * 3 + 2]));
                values.push_back(neighbors.displacements[index * 3 + 0]);
                values.push_back(neighbors.displacements[index * 3 + 1]);
                values.push_back(neighbors.displacements[index * 3 + 2]);
                values.push_back(std::sqrt(std::max(0.0, neighbors.distance2[index])));
            }
        }
    });

    DescriptorPairTable result;
    result.offsets.resize(static_cast<std::size_t>(batch.structures) + 1, 0);
    for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
        result.offsets[static_cast<std::size_t>(structure + 1)]
            = result.offsets[static_cast<std::size_t>(structure)]
            + static_cast<std::int64_t>(per_structure[static_cast<std::size_t>(structure)].size() / 9);
    }
    result.values.reserve(static_cast<std::size_t>(result.offsets.back()) * 9);
    for (auto& values : per_structure) {
        result.values.insert(
            result.values.end(),
            std::make_move_iterator(values.begin()),
            std::make_move_iterator(values.end()));
    }
    check_cancelled(control);
    return result;
}
} // namespace mdescriptor
