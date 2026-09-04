#pragma once

#include "mdescriptor/cuda/context.hpp"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace mdescriptor::cuda {

// All CUDA translation units use the same status-to-error policy. Keeping
// this at the CUDA seam prevents subtle differences in unavailable-device and
// allocation handling as new descriptor adapters are added.
inline void check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return;
    }
    const std::string message = std::string(operation) + ": " + cudaGetErrorString(status);
    if (status == cudaErrorMemoryAllocation) {
        throw CudaOutOfMemory(message.c_str());
    }
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver
        || status == cudaErrorSystemDriverMismatch || status == cudaErrorUnknown) {
        throw CudaUnavailable(message.c_str());
    }
    throw std::runtime_error(message);
}

} // namespace mdescriptor::cuda
