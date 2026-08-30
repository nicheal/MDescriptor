#include "mdescriptor/cuda/context.hpp"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace mdescriptor::cuda {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return;
    }
    if (status == cudaErrorMemoryAllocation) {
        throw CudaOutOfMemory(operation);
    }
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver
        || status == cudaErrorSystemDriverMismatch) {
        throw CudaUnavailable(operation);
    }
    throw std::runtime_error(operation);
}

} // namespace

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

double* CudaExecutionContext::output_buffer(std::size_t count) {
    if (closed_) {
        throw std::runtime_error("CUDA execution context is closed");
    }
    check_cuda(cudaSetDevice(device_), "could not select the CUDA device");
    ensure_capacity(count);
    return output_;
}

std::vector<double> CudaExecutionContext::download_output(std::size_t count) {
    if (count == 0) {
        return {};
    }
    if (count > output_capacity_ || output_ == nullptr) {
        throw std::runtime_error("CUDA output buffer is not large enough");
    }
    std::vector<double> result(count, 0.0);
    check_cuda(
        cudaMemcpyAsync(
            result.data(), output_, count * sizeof(double),
            cudaMemcpyDeviceToHost, stream_),
        "could not copy descriptor data from the CUDA device");
    synchronize();
    return result;
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
    if (stream_ != nullptr) {
        (void)cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

} // namespace mdescriptor::cuda
