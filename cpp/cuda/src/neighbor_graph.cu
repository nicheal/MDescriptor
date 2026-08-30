#include "mdescriptor/cuda/neighbor_graph.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace mdescriptor::cuda {
namespace {

template <typename Value>
void release(Value*& pointer) noexcept {
    if (pointer != nullptr) {
        (void)cudaFree(pointer);
        pointer = nullptr;
    }
}

template <typename Value>
void ensure_and_upload(
    Value** destination,
    std::size_t* capacity,
    const Value* source,
    std::size_t count,
    cudaStream_t stream) {
    if (count > *capacity) {
        if (*destination != nullptr) {
            if (cudaFree(*destination) != cudaSuccess) {
                throw std::runtime_error("could not release the CUDA neighbor graph");
            }
            *destination = nullptr;
        }
        const auto allocation = cudaMalloc(
            reinterpret_cast<void**>(destination), count * sizeof(Value));
        if (allocation == cudaErrorMemoryAllocation) {
            throw CudaOutOfMemory("could not allocate the CUDA neighbor graph");
        }
        if (allocation != cudaSuccess) {
            throw std::runtime_error("could not allocate the CUDA neighbor graph");
        }
        *capacity = count;
    }
    if (count != 0 && cudaMemcpyAsync(
        *destination, source, count * sizeof(Value), cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        throw std::runtime_error("could not upload the CUDA neighbor graph");
    }
}

} // namespace

DeviceNeighborGraph::~DeviceNeighborGraph() noexcept {
    clear();
}

void DeviceNeighborGraph::upload(
    CudaExecutionContext& context,
    const std::vector<std::int64_t>& offsets,
    const std::vector<std::int32_t>& atoms,
    const std::vector<std::int32_t>& shifts,
    const std::vector<double>& displacements,
    const std::vector<double>& distance2) {
    if (offsets.empty() || shifts.size() != atoms.size() * 3
        || displacements.size() != atoms.size() * 3
        || distance2.size() != atoms.size()) {
        throw std::invalid_argument("invalid CUDA neighbor graph arrays");
    }
    if (cudaSetDevice(context.device()) != cudaSuccess) {
        throw std::runtime_error("could not select the CUDA device");
    }
    ensure_and_upload(&offsets_, &offsets_capacity_, offsets.data(), offsets.size(), context.stream());
    try {
        ensure_and_upload(&atoms_, &atoms_capacity_, atoms.data(), atoms.size(), context.stream());
        ensure_and_upload(&shifts_, &shifts_capacity_, shifts.data(), shifts.size(), context.stream());
        ensure_and_upload(
            &displacements_, &displacements_capacity_, displacements.data(),
            displacements.size(), context.stream());
        ensure_and_upload(
            &distance2_, &distance2_capacity_, distance2.data(),
            distance2.size(), context.stream());
        context.synchronize();
    } catch (...) {
        clear();
        throw;
    }
    pairs_ = atoms.size();
}

void DeviceNeighborGraph::clear() noexcept {
    release(offsets_);
    release(atoms_);
    release(shifts_);
    release(displacements_);
    release(distance2_);
    pairs_ = 0;
    offsets_capacity_ = 0;
    atoms_capacity_ = 0;
    shifts_capacity_ = 0;
    displacements_capacity_ = 0;
    distance2_capacity_ = 0;
}

} // namespace mdescriptor::cuda
