#pragma once

#include "mdescriptor/cuda/batch.hpp"
#include "mdescriptor/cuda/context.hpp"
#include "mdescriptor/cuda/neighbor_graph.hpp"
#include "mdescriptor/detail/batch.hpp"
#include "mdescriptor/detail/rotational_bispectrum.hpp"

#include <pybind11/pytypes.h>

#include <cstdint>
#include <memory>
#include <string>

namespace mdescriptor::cuda {

// Device-resident view of the immutable flattened Clebsch--Gordan plan used
// by SO4, SNAP, and L-Bispectrum.  The owning cache is kept behind this small
// interface so the backend can reuse the allocations across compute() calls.
struct RotationalPlanDeviceView {
    const std::int64_t* z_inner_offsets = nullptr;
    const std::int64_t* inner_term_offsets = nullptr;
    const double* inner_outer_coefficients = nullptr;
    const std::int64_t* term_first_indices = nullptr;
    const std::int64_t* term_second_indices = nullptr;
    const double* term_coefficients = nullptr;
    const std::int64_t* projection_offsets = nullptr;
    const std::int64_t* projection_u_indices = nullptr;
    const std::int64_t* projection_z_indices = nullptr;
    const double* projection_scales = nullptr;
    std::int64_t features = 0;
};

class RotationalPlanCache {
public:
    RotationalPlanCache();
    RotationalPlanCache(const RotationalPlanCache&) = delete;
    RotationalPlanCache& operator=(const RotationalPlanCache&) = delete;
    ~RotationalPlanCache() noexcept;

    RotationalPlanDeviceView prepare(
        CudaExecutionContext& context,
        int expansion_order,
        int diagonal,
        bool l_bispectrum);
    void clear() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Compute one of the descriptor families that are not part of the original
// CUDA plugin.  The function deliberately returns the same small raw-result
// mapping as the existing backend: all numeric work is performed against the
// device batch/graph, and only the final public arrays are copied back to the
// host for DescriptorResult normalization in Python.
pybind11::dict compute_extended_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const std::string& name,
    const pybind11::dict& options,
    const pybind11::object& control,
    RotationalPlanCache* rotational_plan = nullptr);

} // namespace mdescriptor::cuda
