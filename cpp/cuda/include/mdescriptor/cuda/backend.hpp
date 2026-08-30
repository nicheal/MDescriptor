#pragma once

#include "mdescriptor/cuda/context.hpp"
#include "mdescriptor/cuda/batch.hpp"
#include "mdescriptor/cuda/neighbor_graph.hpp"

#include <pybind11/pytypes.h>

#include <cstdint>
#include <memory>
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
    DeviceBatch device_batch_;
    DeviceNeighborGraph device_graph_;
    bool closed_ = false;
};

} // namespace mdescriptor::cuda
