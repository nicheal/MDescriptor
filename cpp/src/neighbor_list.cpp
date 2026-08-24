#include "mdescriptor/local_descriptors.hpp"
#include "mdescriptor/neighbor.hpp"
#include "local_common.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace mdescriptor {
using namespace detail;

DescriptorPairTable compute_neighbor_list(
    const StructureBatchView& batch,
    double cutoff,
    bool full_neighbor_list,
    bool self_pairs,
    const std::shared_ptr<ComputeControl>& control) {
    if (!std::isfinite(cutoff) || cutoff <= 0.0) {
        throw std::invalid_argument("neighbor cutoff must be finite and positive");
    }
    if (control) {
        control->reset(batch.structures);
    }
    const NeighborGraph graph = build_neighbor_graph(batch, cutoff, control);
    DescriptorPairTable result;
    result.offsets.push_back(0);
    for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
        check_cancelled(control);
        const std::int64_t begin = batch.offsets[structure];
        const std::int64_t end = batch.offsets[structure + 1];
        for (std::int64_t center = begin; center < end; ++center) {
            const NeighborView neighbors = graph.for_center(center);
            for (std::size_t index = 0; index < neighbors.size; ++index) {
                if (!self_pairs && neighbors.exact_self(index, center)) {
                    continue;
                }
                const std::int64_t atom = neighbors.atoms[index];
                if (!full_neighbor_list && atom < center) {
                    continue;
                }
                result.values.push_back(static_cast<double>(center));
                result.values.push_back(static_cast<double>(atom));
                result.values.push_back(static_cast<double>(neighbors.shifts[index * 3 + 0]));
                result.values.push_back(static_cast<double>(neighbors.shifts[index * 3 + 1]));
                result.values.push_back(static_cast<double>(neighbors.shifts[index * 3 + 2]));
                result.values.push_back(neighbors.displacements[index * 3 + 0]);
                result.values.push_back(neighbors.displacements[index * 3 + 1]);
                result.values.push_back(neighbors.displacements[index * 3 + 2]);
                result.values.push_back(std::sqrt(std::max(0.0, neighbors.distance2[index])));
            }
        }
        result.offsets.push_back(static_cast<std::int64_t>(result.values.size() / 9));
        mark_completed(control);
    }
    check_cancelled(control);
    return result;
}
} // namespace mdescriptor
