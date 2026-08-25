#include "mdescriptor/local_descriptors.hpp"
#include "local_common.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace mdescriptor {
using namespace detail;

void compute_atomic_composition(
    const StructureBatchView& batch,
    const std::vector<std::int32_t>& species,
    bool per_system,
    double* output,
    const std::shared_ptr<ComputeControl>& control) {
    validate_species(batch, species);
    if (control) {
        control->reset(batch.structures);
    }
    const TypeMap mapping = make_type_map(species);
    const std::int64_t rows = per_system ? batch.structures : batch.atoms;
    std::fill(output, output + rows * static_cast<std::int64_t>(species.size()), 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
        if (control && control->cancelled()) {
            continue;
        }
        const std::int64_t begin = batch.offsets[structure];
        const std::int64_t end = batch.offsets[structure + 1];
        for (std::int64_t atom = begin; atom < end; ++atom) {
            const auto type = mapping.find(batch.numbers[atom])->second;
            const std::int64_t row = per_system ? structure : atom;
            output[row * static_cast<std::int64_t>(species.size()) + static_cast<std::int64_t>(type)] += 1.0;
        }
        mark_completed(control);
    }
    check_cancelled(control);
}
} // namespace mdescriptor
