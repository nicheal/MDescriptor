#pragma once

#include "mdescriptor/cuda/batch.hpp"
#include "mdescriptor/cuda/context.hpp"
#include "mdescriptor/detail/batch.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mdescriptor::cuda {

enum class NeighborGraphOrdering {
    Distance,
    Canonical,
};

// CSR + SoA representation shared by the CUDA descriptor family.  The graph
// can either be uploaded by an adapter that already owns a host graph or be
// constructed from a device batch.  DPA4/DPA4C use the latter path: only
// compact per-structure geometry planning crosses the host seam, while
// coordinate normalization, pair enumeration, CSR scans, and ordering run on
// the CUDA stream.
class DeviceNeighborGraph {
public:
    DeviceNeighborGraph() = default;
    DeviceNeighborGraph(const DeviceNeighborGraph&) = delete;
    DeviceNeighborGraph& operator=(const DeviceNeighborGraph&) = delete;
    ~DeviceNeighborGraph() noexcept;

    void upload(
        CudaExecutionContext& context,
        const std::vector<std::int64_t>& offsets,
        const std::vector<std::int32_t>& atoms,
        const std::vector<std::int32_t>& shifts,
        const std::vector<double>& displacements,
        const std::vector<double>& distance2);
    // Build the DPA graph from device-resident coordinates.  The host batch
    // is metadata used to validate PBC and upload compact inverse cells/image
    // bounds; it is never used to enumerate or materialize pairs.
    void build_dpa(
        CudaExecutionContext& context,
        DeviceBatch& batch,
        const detail::StructureBatchView& host_batch,
        double cutoff,
        bool tie_break_shifts,
        bool round_edge_endpoints,
        bool include_exact_self = false,
        bool normalize_periodic_positions = true,
        bool include_boundary = true,
        NeighborGraphOrdering ordering = NeighborGraphOrdering::Distance);
    // NEP's periodic path can construct the cell list and its CSR neighbor
    // graph on the device.  The host batch is used only for the small amount
    // of per-structure grid planning; positions and all pair enumeration stay
    // on the CUDA stream.
    void build_nep(
        CudaExecutionContext& context,
        DeviceBatch& batch,
        const detail::StructureBatchView& host_batch,
        double cutoff);
    void clear() noexcept;

    std::size_t pairs() const noexcept { return pairs_; }
    std::int64_t max_neighbors() const noexcept { return max_neighbors_; }
    bool slot_major() const noexcept { return slot_major_; }
    std::int64_t neighbor_stride() const noexcept { return neighbor_stride_; }
    const std::int32_t* neighbor_counts() const noexcept { return neighbor_counts_; }
    const std::int64_t* offsets() const noexcept { return offsets_; }
    const std::int32_t* atoms() const noexcept { return atoms_; }
    const std::int32_t* shifts() const noexcept { return shifts_; }
    const double* displacements() const noexcept { return displacements_; }
    const double* distance2() const noexcept { return distance2_; }
private:
    void build_canonical_graph(
        CudaExecutionContext& context,
        DeviceBatch& batch,
        const detail::StructureBatchView& host_batch,
        double cutoff,
        bool include_exact_self,
        bool include_boundary,
        const std::vector<std::int32_t>& image_bounds,
        const std::vector<double>& grid_minimum,
        const std::vector<double>& grid_spacing,
        const std::vector<std::int32_t>& grid_dimensions);

    template <typename Value>
    void ensure_capacity(Value** pointer, std::size_t* capacity, std::size_t count);

    std::int64_t* offsets_ = nullptr;
    std::int32_t* atoms_ = nullptr;
    std::int32_t* shifts_ = nullptr;
    double* displacements_ = nullptr;
    double* distance2_ = nullptr;
    std::size_t pairs_ = 0;
    std::int64_t max_neighbors_ = 0;
    bool slot_major_ = false;
    std::int64_t neighbor_stride_ = 0;
    std::size_t offsets_capacity_ = 0;
    std::size_t atoms_capacity_ = 0;
    std::size_t shifts_capacity_ = 0;
    std::size_t displacements_capacity_ = 0;
    std::size_t distance2_capacity_ = 0;
    double* dpa_positions_ = nullptr;
    std::int32_t* dpa_image_bounds_ = nullptr;
    double* dpa_reference_inverses_ = nullptr;
    double* canonical_grid_min_ = nullptr;
    double* canonical_grid_spacing_ = nullptr;
    std::int32_t* canonical_grid_dimensions_ = nullptr;
    std::size_t dpa_positions_capacity_ = 0;
    std::size_t dpa_image_bounds_capacity_ = 0;
    std::size_t dpa_reference_inverses_capacity_ = 0;
    std::size_t canonical_grid_min_capacity_ = 0;
    std::size_t canonical_grid_spacing_capacity_ = 0;
    std::size_t canonical_grid_dimensions_capacity_ = 0;
    std::int64_t* canonical_extended_offsets_ = nullptr;
    std::int32_t* canonical_extended_atoms_ = nullptr;
    std::int32_t* canonical_extended_shifts_ = nullptr;
    double* canonical_extended_positions_ = nullptr;
    std::size_t canonical_extended_offsets_capacity_ = 0;
    std::size_t canonical_extended_atoms_capacity_ = 0;
    std::size_t canonical_extended_shifts_capacity_ = 0;
    std::size_t canonical_extended_positions_capacity_ = 0;

    std::int32_t* atom_to_structure_ = nullptr;
    std::int32_t* cell_counts_ = nullptr;
    std::int32_t* cell_offsets_ = nullptr;
    std::int32_t* cell_fill_ = nullptr;
    std::int32_t* cell_atoms_ = nullptr;
    std::int32_t* cell_sort_keys_ = nullptr;
    std::int32_t* atom_cells_ = nullptr;
    std::int32_t* neighbor_counts_ = nullptr;
    std::int32_t* neighbor_overflow_ = nullptr;
    std::int32_t* structure_cell_offsets_ = nullptr;
    std::int32_t* structure_cell_dims_ = nullptr;
    double* reference_cells_ = nullptr;
    double* reference_cell_inverses_ = nullptr;
    std::size_t atom_to_structure_capacity_ = 0;
    std::size_t atom_cells_capacity_ = 0;
    std::size_t neighbor_counts_capacity_ = 0;
    std::size_t neighbor_overflow_capacity_ = 0;
    std::size_t cell_atoms_capacity_ = 0;
    std::size_t cell_sort_keys_capacity_ = 0;
    std::size_t structure_cell_offsets_capacity_ = 0;
    std::size_t structure_cell_dims_capacity_ = 0;
    std::size_t reference_cells_capacity_ = 0;
    std::size_t reference_cell_inverses_capacity_ = 0;
    std::size_t cell_counts_capacity_ = 0;
    std::size_t cell_offsets_capacity_ = 0;
    std::size_t cell_fill_capacity_ = 0;
};

} // namespace mdescriptor::cuda
