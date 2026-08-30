#include "mdescriptor/cuda/batch.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace mdescriptor::cuda {
namespace {

void check_copy(cudaError_t status, const char* operation) {
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

template <typename Value>
void ensure_and_copy(
    Value** destination,
    std::size_t* capacity,
    const Value* source,
    std::size_t count,
    cudaStream_t stream,
    const char* operation) {
    if (count > *capacity) {
        if (*destination != nullptr) {
            check_copy(cudaFree(*destination), operation);
            *destination = nullptr;
        }
        *capacity = 0;
        check_copy(
            cudaMalloc(reinterpret_cast<void**>(destination), count * sizeof(Value)),
            operation);
        *capacity = count;
    }
    if (count != 0) {
        check_copy(
            cudaMemcpyAsync(
                *destination, source, count * sizeof(Value),
                cudaMemcpyHostToDevice, stream),
            operation);
    }
}

} // namespace

DeviceBatch::~DeviceBatch() noexcept {
    clear();
}

template <typename Value>
void DeviceBatch::release(Value*& pointer) noexcept {
    if (pointer != nullptr) {
        (void)cudaFree(pointer);
        pointer = nullptr;
    }
}

void DeviceBatch::upload(
    CudaExecutionContext& context,
    const detail::StructureBatchView& batch) {
    structures_ = batch.structures;
    atoms_ = batch.atoms;
    check_copy(cudaSetDevice(context.device()), "could not select the CUDA device");
    ensure_and_copy(
        &numbers_, &numbers_capacity_, batch.numbers,
        static_cast<std::size_t>(atoms_), context.stream(), "could not upload numbers");
    ensure_and_copy(
        &positions_, &positions_capacity_, batch.positions,
        static_cast<std::size_t>(atoms_) * 3, context.stream(), "could not upload positions");
    ensure_and_copy(
        &cells_, &cells_capacity_, batch.cells,
        static_cast<std::size_t>(structures_) * 9, context.stream(), "could not upload cells");
    ensure_and_copy(
        &pbc_, &pbc_capacity_, batch.pbc,
        static_cast<std::size_t>(structures_) * 3, context.stream(), "could not upload pbc");
    ensure_and_copy(
        &offsets_, &offsets_capacity_, batch.offsets,
        static_cast<std::size_t>(structures_ + 1), context.stream(), "could not upload offsets");
    context.synchronize();
}

void DeviceBatch::clear() noexcept {
    release(numbers_);
    release(positions_);
    release(cells_);
    release(pbc_);
    release(offsets_);
    structures_ = 0;
    atoms_ = 0;
    numbers_capacity_ = 0;
    positions_capacity_ = 0;
    cells_capacity_ = 0;
    pbc_capacity_ = 0;
    offsets_capacity_ = 0;
}

} // namespace mdescriptor::cuda
