#pragma once

#include "mdescriptor/cuda/context.hpp"
#include "mdescriptor/cuda/batch.hpp"
#include "mdescriptor/cuda/dpa4.hpp"
#include "mdescriptor/cuda/dpa4c.hpp"
#include "mdescriptor/cuda/extended_descriptors.hpp"
#include "mdescriptor/cuda/nep.hpp"
#include "mdescriptor/cuda/neighbor_graph.hpp"

#include <pybind11/pytypes.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace mdescriptor::cuda {

class Backend {
public:
    Backend(std::string name, pybind11::dict options);
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;
    ~Backend() noexcept;

    std::int64_t feature_count() const noexcept { return feature_count_; }
    pybind11::object compute(pybind11::object batch, pybind11::object control);
    pybind11::dict metadata() const;
    void close() noexcept;

private:
    std::string name_;
    pybind11::dict options_;
    std::int64_t feature_count_ = 0;
    std::unique_ptr<CudaExecutionContext> context_;
    std::unique_ptr<DeviceNepModel> nep_model_;
    std::unique_ptr<DeviceDpa4Model> dpa4_model_;
    std::unique_ptr<DeviceDpa4cModel> dpa4c_model_;
    DeviceBatch device_batch_;
    DeviceBatch nep_expanded_batch_;
    DeviceNeighborGraph device_graph_;
    // A context owns one stream and reusable buffers.  Serialize calls per
    // backend instance so concurrent public compute() calls cannot resize or
    // overwrite those resources while another launch is in flight.
    mutable std::mutex compute_mutex_;
    bool closed_ = false;
};

} // namespace mdescriptor::cuda
