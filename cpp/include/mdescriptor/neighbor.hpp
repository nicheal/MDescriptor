#pragma once

#include "descriptor.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mdescriptor {

struct NeighborView {
    const std::int32_t* atoms = nullptr;
    const std::int32_t* shifts = nullptr; // row-major (neighbor_count, 3)
    const double* displacements = nullptr; // row-major (neighbor_count, 3)
    const double* distance2 = nullptr;
    std::size_t size = 0;

    bool exact_self(std::size_t index, std::int64_t center) const noexcept;
};

class NeighborGraph {
public:
    NeighborGraph() = default;

    NeighborView for_center(std::int64_t center) const noexcept;
    std::int64_t atoms() const noexcept { return static_cast<std::int64_t>(offsets_.size()) - 1; }
    double cutoff() const noexcept { return cutoff_; }

    const std::vector<std::int64_t>& offsets() const noexcept { return offsets_; }
    const std::vector<std::int32_t>& atoms_data() const noexcept { return atoms_; }
    const std::vector<std::int32_t>& shifts() const noexcept { return shifts_; }
    const std::vector<double>& displacements() const noexcept { return displacements_; }
    const std::vector<double>& distance2() const noexcept { return distance2_; }

private:
    friend NeighborGraph build_neighbor_graph(
        const StructureBatchView&,
        double,
        const std::shared_ptr<ComputeControl>&,
        int,
        bool,
        bool,
        bool);

    double cutoff_ = 0.0;
    std::vector<std::int64_t> offsets_;
    std::vector<std::int32_t> atoms_;
    std::vector<std::int32_t> shifts_;
    std::vector<double> displacements_;
    std::vector<double> distance2_;
};

NeighborGraph build_neighbor_graph(
    const StructureBatchView& batch,
    double cutoff,
    const std::shared_ptr<ComputeControl>& control = nullptr,
    int num_threads = 0,
    bool include_boundary = true,
    bool use_scaled_periodic_images = false,
    bool store_shifts = true);

} // namespace mdescriptor
