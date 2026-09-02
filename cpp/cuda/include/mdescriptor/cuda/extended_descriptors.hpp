#pragma once

#include "mdescriptor/cuda/batch.hpp"
#include "mdescriptor/cuda/context.hpp"
#include "mdescriptor/cuda/neighbor_graph.hpp"
#include "mdescriptor/detail/batch.hpp"

#include <pybind11/pytypes.h>

#include <string>

namespace mdescriptor::cuda {

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
    const pybind11::object& control);

} // namespace mdescriptor::cuda
