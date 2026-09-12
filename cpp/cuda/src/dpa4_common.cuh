#pragma once

// Host-side payload parsing and device upload glue shared by the DPA4 and
// DPA4C descriptor bindings.  Error messages stay backend-specific; callers
// pass their own strings.

#include "mdescriptor/cuda/error.hpp"

#include "device_memory.cuh"

#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mdescriptor::cuda::dpa4_common {

namespace py = pybind11;

inline py::handle required(
    const py::dict& payload, const char* name, const char* backend) {
    if (!payload.contains(name)) {
        throw std::invalid_argument(
            std::string(backend) + " CUDA payload is missing " + name);
    }
    return payload[name];
}

template <typename Value>
std::vector<Value> payload_array(
    py::handle value,
    const char* name,
    const char* invalid_error,
    bool reject_zero_dimension) {
    using Array = py::array_t<Value, py::array::c_style | py::array::forcecast>;
    const Array array = Array::ensure(value);
    if (!array || (reject_zero_dimension && array.ndim() == 0)) {
        throw std::invalid_argument(invalid_error);
    }
    const auto info = array.request();
    const auto* data = static_cast<const Value*>(info.ptr);
    return std::vector<Value>(data, data + info.size);
}

template <typename Array, typename Value>
std::unique_ptr<Array> upload_array(
    CudaExecutionContext& context,
    const std::vector<Value>& values,
    const char* operation,
    const char* device_error,
    bool synchronous_copy) {
    auto result = std::make_unique<Array>();
    result->bytes = values.size() * sizeof(Value);
    if (result->bytes == 0) {
        return result;
    }
    check_cuda(cudaSetDevice(context.device()), device_error);
    try {
        check_cuda(cudaMalloc(&result->pointer, result->bytes), operation);
        // DPA4C copies asynchronously on the context stream, while DPA4 parses
        // into temporary parser buffers and must not leave an asynchronous DMA
        // operation borrowing memory that has already gone out of scope.
        if (synchronous_copy) {
            check_cuda(
                cudaMemcpy(
                    result->pointer, values.data(), result->bytes,
                    cudaMemcpyHostToDevice),
                operation);
        } else {
            check_cuda(
                cudaMemcpyAsync(
                    result->pointer, values.data(), result->bytes,
                    cudaMemcpyHostToDevice, context.stream()),
                operation);
        }
    } catch (...) {
        (void)cudaFree(result->pointer);
        result->pointer = nullptr;
        throw;
    }
    return result;
}

template <typename Value, typename Array>
Value* device_data(const std::unique_ptr<Array>& value) {
    return value == nullptr ? nullptr : static_cast<Value*>(value->pointer);
}

inline std::size_t align_bytes(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1U) / alignment * alignment;
}

} // namespace mdescriptor::cuda::dpa4_common
