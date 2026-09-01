#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace mdescriptor::cuda {

class CudaUnavailable : public std::runtime_error {
public:
    explicit CudaUnavailable(const char* message) : std::runtime_error(message) {}
};

class CudaOutOfMemory : public std::runtime_error {
public:
    explicit CudaOutOfMemory(const char* message) : std::runtime_error(message) {}
};

class CudaExecutionContext {
public:
    explicit CudaExecutionContext(int device = 0);
    CudaExecutionContext(const CudaExecutionContext&) = delete;
    CudaExecutionContext& operator=(const CudaExecutionContext&) = delete;
    ~CudaExecutionContext() noexcept;

    int device() const noexcept { return device_; }
    cudaStream_t stream() const noexcept { return stream_; }

    // Descriptor kernels use this stream and therefore close() can provide the
    // one lifecycle synchronization point promised by the backend seam.
    double* output_buffer(std::size_t count);
    // Reusable byte-addressable temporary storage for descriptor kernels.
    // The previous computation is synchronized before a resize can occur.
    void* workspace_buffer(std::size_t count);
    std::vector<double> download_output(std::size_t count);
    std::vector<double> download_output_slice(std::size_t offset, std::size_t count);

    void synchronize();
    void close() noexcept;

private:
    void ensure_capacity(std::size_t count);
    void ensure_workspace_capacity(std::size_t count);

    int device_ = 0;
    cudaStream_t stream_ = nullptr;
    double* output_ = nullptr;
    std::size_t output_capacity_ = 0;
    unsigned char* workspace_ = nullptr;
    std::size_t workspace_capacity_ = 0;
    bool closed_ = false;
};

} // namespace mdescriptor::cuda
