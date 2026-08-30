#pragma once

#include "mdescriptor/detail/batch.hpp"
#include "mdescriptor/cuda/context.hpp"

#include <cstddef>
#include <cstdint>

namespace mdescriptor::cuda {

// A device-owned copy of the stable CPU batch layout.  The view is private to
// the CUDA plugin; no CUDA pointer is added to the public StructureBatch or
// the CPU headers.
class DeviceBatch {
public:
    DeviceBatch() = default;
    DeviceBatch(const DeviceBatch&) = delete;
    DeviceBatch& operator=(const DeviceBatch&) = delete;
    ~DeviceBatch() noexcept;

    void upload(
        CudaExecutionContext& context,
        const detail::StructureBatchView& batch);
    void clear() noexcept;

    std::int64_t structures() const noexcept { return structures_; }
    std::int64_t atoms() const noexcept { return atoms_; }
    const std::int32_t* numbers() const noexcept { return numbers_; }
    const double* positions() const noexcept { return positions_; }
    const double* cells() const noexcept { return cells_; }
    const std::int32_t* pbc() const noexcept { return pbc_; }
    const std::int64_t* offsets() const noexcept { return offsets_; }

private:
    template <typename Value>
    static void release(Value*& pointer) noexcept;

    std::int32_t* numbers_ = nullptr;
    double* positions_ = nullptr;
    double* cells_ = nullptr;
    std::int32_t* pbc_ = nullptr;
    std::int64_t* offsets_ = nullptr;
    std::int64_t structures_ = 0;
    std::int64_t atoms_ = 0;
    std::size_t numbers_capacity_ = 0;
    std::size_t positions_capacity_ = 0;
    std::size_t cells_capacity_ = 0;
    std::size_t pbc_capacity_ = 0;
    std::size_t offsets_capacity_ = 0;
};

} // namespace mdescriptor::cuda
