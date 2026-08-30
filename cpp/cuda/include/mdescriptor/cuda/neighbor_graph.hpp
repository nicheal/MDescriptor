#pragma once

#include "mdescriptor/cuda/context.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mdescriptor::cuda {

// CSR + SoA representation shared by the first CUDA descriptor family.  The
// host canonical graph is uploaded in deterministic order; later kernels can
// consume it without relying on atomic append order.
class DeviceNeighborGraph {
public:
    DeviceNeighborGraph() = default;
    DeviceNeighborGraph(const DeviceNeighborGraph&) = delete;
    DeviceNeighborGraph& operator=(const DeviceNeighborGraph&) = delete;
    ~DeviceNeighborGraph() noexcept;

    void upload(
        CudaExecutionContext& context,
        const std::vector<std::int64_t>& offsets,
        const std::vector<std::int32_t>& atoms,
        const std::vector<std::int32_t>& shifts,
        const std::vector<double>& displacements,
        const std::vector<double>& distance2);
    void clear() noexcept;

    std::size_t pairs() const noexcept { return pairs_; }
    const std::int64_t* offsets() const noexcept { return offsets_; }
    const std::int32_t* atoms() const noexcept { return atoms_; }
    const std::int32_t* shifts() const noexcept { return shifts_; }
    const double* displacements() const noexcept { return displacements_; }
    const double* distance2() const noexcept { return distance2_; }

private:
    std::int64_t* offsets_ = nullptr;
    std::int32_t* atoms_ = nullptr;
    std::int32_t* shifts_ = nullptr;
    double* displacements_ = nullptr;
    double* distance2_ = nullptr;
    std::size_t pairs_ = 0;
    std::size_t offsets_capacity_ = 0;
    std::size_t atoms_capacity_ = 0;
    std::size_t shifts_capacity_ = 0;
    std::size_t displacements_capacity_ = 0;
    std::size_t distance2_capacity_ = 0;
};

} // namespace mdescriptor::cuda
