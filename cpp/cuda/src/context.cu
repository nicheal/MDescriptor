#include "mdescriptor/cuda/context.hpp"
#include "mdescriptor/cuda/error.hpp"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace mdescriptor::cuda {

CudaExecutionContext::CudaExecutionContext(int device) : device_(device) {
    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess || device_count <= 0) {
        throw CudaUnavailable("CUDA device is unavailable");
    }
    if (device < 0 || device >= device_count) {
        throw CudaUnavailable("requested CUDA device is unavailable");
    }
    check_cuda(cudaSetDevice(device_), "could not select the CUDA device");
    check_cuda(
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
        "could not create the CUDA stream");
}

CudaExecutionContext::~CudaExecutionContext() noexcept {
    close();
}

void CudaExecutionContext::ensure_capacity(std::size_t count) {
    if (count <= output_capacity_) {
        return;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
        throw CudaOutOfMemory("requested CUDA output buffer is too large");
    }
    check_cuda(cudaSetDevice(device_), "could not select the CUDA device");
    if (output_ != nullptr) {
        check_cuda(cudaFree(output_), "could not release the CUDA output buffer");
        output_ = nullptr;
    }
    output_capacity_ = 0;
    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&output_), count * sizeof(double)),
        "could not allocate the CUDA output buffer");
    output_capacity_ = count;
}

void CudaExecutionContext::ensure_workspace_capacity(std::size_t count) {
    if (count <= workspace_capacity_) {
        return;
    }
    check_cuda(cudaSetDevice(device_), "could not select the CUDA device");
    if (workspace_ != nullptr) {
        check_cuda(cudaFree(workspace_), "could not release the CUDA workspace");
        workspace_ = nullptr;
    }
    workspace_capacity_ = 0;
    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&workspace_), count),
        "could not allocate the CUDA workspace");
    workspace_capacity_ = count;
}

double* CudaExecutionContext::output_buffer(std::size_t count) {
    if (closed_) {
        throw std::runtime_error("CUDA execution context is closed");
    }
    check_cuda(cudaSetDevice(device_), "could not select the CUDA device");
    ensure_capacity(count);
    return output_;
}

void* CudaExecutionContext::workspace_buffer(std::size_t count) {
    if (closed_) {
        throw std::runtime_error("CUDA execution context is closed");
    }
    if (count == 0) {
        return nullptr;
    }
    check_cuda(cudaSetDevice(device_), "could not select the CUDA device");
    ensure_workspace_capacity(count);
    return workspace_;
}

std::vector<double> CudaExecutionContext::download_output(std::size_t count) {
    return download_output_slice(0, count);
}

std::vector<double> CudaExecutionContext::download_output_slice(
    std::size_t offset,
    std::size_t count) {
    if (count == 0) {
        return {};
    }
    if (output_ == nullptr || offset > output_capacity_
        || count > output_capacity_ - offset) {
        throw std::runtime_error("CUDA output buffer is not large enough");
    }
    std::vector<double> result(count, 0.0);
    check_cuda(
        cudaMemcpyAsync(
            result.data(), output_ + offset, count * sizeof(double),
            cudaMemcpyDeviceToHost, stream_),
        "could not copy descriptor data from the CUDA device");
    synchronize();
    return result;
}

void CudaExecutionContext::download_output_into(double* destination, std::size_t count) {
    download_output_slice_into(0, destination, count);
}

void CudaExecutionContext::download_output_slice_into(
    std::size_t offset,
    double* destination,
    std::size_t count) {
    if (closed_) {
        throw std::runtime_error("CUDA execution context is closed");
    }
    if (count == 0) {
        return;
    }
    if (destination == nullptr) {
        throw std::invalid_argument("CUDA output destination must not be null");
    }
    if (output_ == nullptr || offset > output_capacity_
        || count > output_capacity_ - offset) {
        throw std::runtime_error("CUDA output buffer is not large enough");
    }
    check_cuda(
        cudaMemcpyAsync(
            destination, output_ + offset, count * sizeof(double),
            cudaMemcpyDeviceToHost, stream_),
        "could not copy descriptor data from the CUDA device");
    synchronize();
}

void* CudaExecutionContext::static_payload_buffer(
    std::size_t slot,
    const void* source,
    std::size_t bytes,
    const char* operation) {
    if (closed_) {
        throw std::runtime_error("CUDA execution context is closed");
    }
    if (bytes == 0) {
        return nullptr;
    }
    if (source == nullptr) {
        throw std::invalid_argument("CUDA static payload source must not be null");
    }
    if (slot >= static_payloads_.size()) {
        static_payloads_.resize(slot + 1, nullptr);
        static_payload_sizes_.resize(slot + 1, 0);
    }
    if (static_payloads_[slot] != nullptr) {
        if (static_payload_sizes_[slot] != bytes) {
            throw std::invalid_argument("CUDA static payload slot changed size");
        }
        return static_payloads_[slot];
    }
    check_cuda(cudaSetDevice(device_), "could not select the CUDA device");
    void* allocation = nullptr;
    check_cuda(cudaMalloc(&allocation, bytes), operation);
    try {
        check_cuda(
            cudaMemcpy(allocation, source, bytes, cudaMemcpyHostToDevice),
            operation);
    } catch (...) {
        (void)cudaFree(allocation);
        throw;
    }
    static_payloads_[slot] = allocation;
    static_payload_sizes_[slot] = bytes;
    return allocation;
}

void CudaExecutionContext::synchronize() {
    if (!closed_ && stream_ != nullptr) {
        check_cuda(cudaStreamSynchronize(stream_), "CUDA stream synchronization failed");
    }
}

void CudaExecutionContext::close() noexcept {
    if (closed_) {
        return;
    }
    closed_ = true;
    if (stream_ != nullptr) {
        (void)cudaStreamSynchronize(stream_);
    }
    if (output_ != nullptr) {
        (void)cudaFree(output_);
        output_ = nullptr;
    }
    output_capacity_ = 0;
    if (workspace_ != nullptr) {
        (void)cudaFree(workspace_);
        workspace_ = nullptr;
    }
    workspace_capacity_ = 0;
    for (void* payload : static_payloads_) {
        if (payload != nullptr) (void)cudaFree(payload);
    }
    static_payloads_.clear();
    static_payload_sizes_.clear();
    if (stream_ != nullptr) {
        (void)cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

} // namespace mdescriptor::cuda
