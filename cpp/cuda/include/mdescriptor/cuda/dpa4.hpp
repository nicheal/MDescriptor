#pragma once

#include "mdescriptor/cuda/batch.hpp"
#include "mdescriptor/cuda/context.hpp"
#include "mdescriptor/cuda/neighbor_graph.hpp"

#include <pybind11/pytypes.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace mdescriptor::cuda {

// Device-side owner for the fixed, graph-native DPA4 payload.  The Python
// checkpoint reader remains the source of truth for the .pt format; this
// object owns the typed device weights and the private CUDA workspace used by
// the inference path.
class DeviceDpa4Model {
public:
    struct DeviceArray;
    struct Model;

    DeviceDpa4Model(CudaExecutionContext& context, pybind11::dict payload);
    DeviceDpa4Model(const DeviceDpa4Model&) = delete;
    DeviceDpa4Model& operator=(const DeviceDpa4Model&) = delete;
    ~DeviceDpa4Model() noexcept;

    std::int64_t feature_count() const noexcept;
    double cutoff() const noexcept;

    std::vector<double> compute(
        CudaExecutionContext& context,
        const DeviceBatch& batch,
        const DeviceNeighborGraph& graph,
        const std::vector<std::int32_t>& type_indices) const;

private:
    void release() noexcept;

    std::unique_ptr<Model> model_;
};

} // namespace mdescriptor::cuda
