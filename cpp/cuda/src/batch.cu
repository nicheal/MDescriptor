#include "mdescriptor/cuda/batch.hpp"
#include "mdescriptor/cuda/error.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mdescriptor::cuda {
namespace {

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
            check_cuda(cudaFree(*destination), operation);
            *destination = nullptr;
        }
        *capacity = 0;
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(destination), count * sizeof(Value)),
            operation);
        *capacity = count;
    }
    if (count != 0) {
        check_cuda(
            cudaMemcpyAsync(
                *destination, source, count * sizeof(Value),
                cudaMemcpyHostToDevice, stream),
            operation);
    }
}

template <typename Value>
void ensure_capacity(Value** destination, std::size_t* capacity, std::size_t count) {
    if (count <= *capacity) {
        return;
    }
    if (*destination != nullptr) {
        check_cuda(cudaFree(*destination), "could not release CUDA batch storage");
        *destination = nullptr;
    }
    *capacity = 0;
    if (count == 0) {
        return;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(Value)) {
        throw CudaOutOfMemory("requested CUDA batch storage is too large");
    }
    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(destination), count * sizeof(Value)),
        "could not allocate CUDA batch storage");
    *capacity = count;
}

__global__ void stage_positions_aos_to_soa_kernel(
    int atoms,
    int stride,
    const double* positions_aos,
    double* positions_soa) {
    const std::int64_t atom = static_cast<std::int64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    if (atom >= atoms) return;
    positions_soa[atom] = positions_aos[3 * atom + 0];
    positions_soa[stride + atom] = positions_aos[3 * atom + 1];
    positions_soa[2 * stride + atom] = positions_aos[3 * atom + 2];
}

bool inverse_row_major3(const double* matrix, double* inverse) {
    const double determinant =
        matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
        - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
        + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
    if (!std::isfinite(determinant) || std::abs(determinant) <= 1.0e-12) {
        return false;
    }
    const double inverse_determinant = 1.0 / determinant;
    inverse[0] = (matrix[4] * matrix[8] - matrix[5] * matrix[7]) * inverse_determinant;
    inverse[1] = (matrix[2] * matrix[7] - matrix[1] * matrix[8]) * inverse_determinant;
    inverse[2] = (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * inverse_determinant;
    inverse[3] = (matrix[5] * matrix[6] - matrix[3] * matrix[8]) * inverse_determinant;
    inverse[4] = (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * inverse_determinant;
    inverse[5] = (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * inverse_determinant;
    inverse[6] = (matrix[3] * matrix[7] - matrix[4] * matrix[6]) * inverse_determinant;
    inverse[7] = (matrix[1] * matrix[6] - matrix[0] * matrix[7]) * inverse_determinant;
    inverse[8] = (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * inverse_determinant;
    return true;
}

std::array<std::int32_t, 3> nep_replication_counts(
    const double* source_cell,
    double cutoff) {
    double reference_cell[9] = {};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            reference_cell[row * 3 + column] = source_cell[column * 3 + row];
        }
    }
    double inverse[9] = {};
    if (!inverse_row_major3(reference_cell, inverse)) {
        throw std::invalid_argument("cannot expand a singular periodic NEP cell");
    }
    std::array<std::int32_t, 3> counts{1, 1, 1};
    for (int axis = 0; axis < 3; ++axis) {
        const double x = inverse[axis * 3 + 0];
        const double y = inverse[axis * 3 + 1];
        const double z = inverse[axis * 3 + 2];
        const double reciprocal_norm = std::sqrt(x * x + y * y + z * z);
        const double required = 2.0 * cutoff * reciprocal_norm;
        if (!std::isfinite(required)
            || required > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument("CUDA NEP periodic image range is too large");
        }
        counts[static_cast<std::size_t>(axis)] = std::max<std::int32_t>(
            1, static_cast<std::int32_t>(std::ceil(required - 1.0e-12)));
    }
    return counts;
}

__global__ void expand_nep_batch_kernel(
    int source_structures,
    const std::int32_t* source_numbers,
    const double* source_positions,
    const double* source_cells,
    const std::int64_t* source_offsets,
    const std::int64_t* expanded_offsets,
    const std::int32_t* replication_counts,
    std::int32_t* expanded_numbers,
    double* expanded_positions) {
    const std::int64_t atom = static_cast<std::int64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    if (atom >= expanded_offsets[source_structures]) return;

    int low = 0;
    int high = source_structures;
    while (low + 1 < high) {
        const int middle = low + (high - low) / 2;
        if (expanded_offsets[middle] <= atom) {
            low = middle;
        } else {
            high = middle;
        }
    }
    const int structure = low;
    const std::int64_t structure_begin = expanded_offsets[structure];
    const std::int64_t source_begin = source_offsets[structure];
    const std::int64_t source_count = source_offsets[structure + 1] - source_begin;
    if (source_count <= 0) return;
    const std::int32_t* counts = replication_counts + structure * 3;
    const std::int64_t replica = (atom - structure_begin) / source_count;
    const std::int64_t local = (atom - structure_begin) % source_count;
    const std::int64_t source_atom = source_begin + local;
    const std::int64_t replicas_xy = static_cast<std::int64_t>(counts[1]) * counts[2];
    const std::int32_t ia = static_cast<std::int32_t>(replica / replicas_xy);
    const std::int64_t remainder = replica % replicas_xy;
    const std::int32_t ib = static_cast<std::int32_t>(remainder / counts[2]);
    const std::int32_t ic = static_cast<std::int32_t>(remainder % counts[2]);
    const double* cell = source_cells + structure * 9;
    const double tx = static_cast<double>(ia) * cell[0]
        + static_cast<double>(ib) * cell[3]
        + static_cast<double>(ic) * cell[6];
    const double ty = static_cast<double>(ia) * cell[1]
        + static_cast<double>(ib) * cell[4]
        + static_cast<double>(ic) * cell[7];
    const double tz = static_cast<double>(ia) * cell[2]
        + static_cast<double>(ib) * cell[5]
        + static_cast<double>(ic) * cell[8];
    expanded_numbers[atom] = source_numbers[source_atom];
    expanded_positions[atom * 3 + 0] = source_positions[source_atom * 3 + 0] + tx;
    expanded_positions[atom * 3 + 1] = source_positions[source_atom * 3 + 1] + ty;
    expanded_positions[atom * 3 + 2] = source_positions[source_atom * 3 + 2] + tz;
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
    expanded_ = false;
    original_atom_count_ = 0;
    host_cells_.clear();
    host_pbc_.clear();
    host_offsets_.clear();
    structures_ = batch.structures;
    atoms_ = batch.atoms;
    check_cuda(cudaSetDevice(context.device()), "could not select the CUDA device");
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
}

bool DeviceBatch::expand_nep(
    CudaExecutionContext& context,
    const DeviceBatch& source,
    const detail::StructureBatchView& source_host,
    double cutoff) {
    if (source_host.structures != source.structures()
        || source_host.atoms != source.atoms()
        || source_host.offsets == nullptr || source_host.cells == nullptr
        || source_host.pbc == nullptr) {
        throw std::invalid_argument("invalid source batch for CUDA NEP expansion");
    }
    if (!std::isfinite(cutoff) || cutoff <= 0.0) {
        throw std::invalid_argument("CUDA NEP expansion cutoff must be positive");
    }
    if (source_host.structures > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())
        || source_host.atoms > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw CudaOutOfMemory("CUDA NEP expansion exceeds int32 index capacity");
    }

    const std::size_t structure_count = static_cast<std::size_t>(source_host.structures);
    const std::size_t source_atom_count = static_cast<std::size_t>(source_host.atoms);
    std::vector<std::array<std::int32_t, 3>> replication(
        structure_count, {1, 1, 1});
    std::vector<std::int64_t> expanded_offsets(structure_count + 1U, 0);
    std::vector<double> expanded_cells(structure_count * 9U, 0.0);
    std::vector<std::int32_t> expanded_pbc(structure_count * 3U, 0);
    std::size_t expanded_atom_count = 0;
    bool expanded = false;

    for (std::size_t structure = 0; structure < structure_count; ++structure) {
        const std::int32_t* pbc = source_host.pbc + structure * 3U;
        const bool periodic = pbc[0] == 1 && pbc[1] == 1 && pbc[2] == 1;
        const bool isolated = pbc[0] == 0 && pbc[1] == 0 && pbc[2] == 0;
        if (!periodic && !isolated) {
            throw std::invalid_argument(
                "CUDA NEP supports all-zero or all-one pbc per structure");
        }
        if (periodic) {
            replication[structure] = nep_replication_counts(
                source_host.cells + structure * 9U, cutoff);
        }
        const auto counts = replication[structure];
        const std::int64_t replicas = static_cast<std::int64_t>(counts[0])
            * counts[1] * counts[2];
        if (replicas > std::numeric_limits<std::int32_t>::max()
            || source_host.offsets[structure + 1] - source_host.offsets[structure]
                > (std::numeric_limits<std::int64_t>::max() -
                   static_cast<std::int64_t>(expanded_atom_count)) / replicas) {
            throw CudaOutOfMemory("CUDA NEP expanded batch is too large");
        }
        const std::int64_t structure_atoms =
            (source_host.offsets[structure + 1] - source_host.offsets[structure]) * replicas;
        if (expanded_atom_count > static_cast<std::size_t>(
                std::numeric_limits<std::int32_t>::max())
            || structure_atoms > static_cast<std::int64_t>(
                std::numeric_limits<std::int32_t>::max())
            || expanded_atom_count + static_cast<std::size_t>(structure_atoms)
                > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw CudaOutOfMemory("CUDA NEP expanded batch exceeds int32 index capacity");
        }
        expanded_atom_count += static_cast<std::size_t>(structure_atoms);
        expanded_offsets[structure + 1] = static_cast<std::int64_t>(expanded_atom_count);
        expanded = expanded || replicas > 1;
        for (int axis = 0; axis < 3; ++axis) {
            expanded_pbc[structure * 3U + static_cast<std::size_t>(axis)] = pbc[axis];
        }
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                expanded_cells[structure * 9U + row * 3 + column] =
                    source_host.cells[structure * 9U + row * 3 + column]
                    * static_cast<double>(counts[row]);
            }
        }
    }
    if (!expanded) {
        expanded_ = false;
        original_atom_count_ = 0;
        host_cells_.clear();
        host_pbc_.clear();
        host_offsets_.clear();
        return false;
    }

    if (cudaSetDevice(context.device()) != cudaSuccess) {
        throw std::runtime_error("could not select the CUDA device");
    }
    structures_ = source.structures();
    atoms_ = static_cast<std::int64_t>(expanded_atom_count);
    expanded_ = true;
    original_atom_count_ = source.atoms();
    host_cells_ = std::move(expanded_cells);
    host_pbc_ = std::move(expanded_pbc);
    host_offsets_ = std::move(expanded_offsets);

    std::vector<std::int64_t> expansion_first(source_atom_count, 0);
    std::vector<std::int64_t> expansion_stride(source_atom_count, 0);
    std::vector<std::int32_t> expansion_replicas(source_atom_count, 1);
    for (std::size_t structure = 0; structure < structure_count; ++structure) {
        const std::int64_t source_begin = source_host.offsets[structure];
        const std::int64_t source_end = source_host.offsets[structure + 1];
        const std::int64_t source_count = source_end - source_begin;
        const std::int32_t* counts = replication[structure].data();
        const std::int32_t replicas = static_cast<std::int32_t>(
            static_cast<std::int64_t>(counts[0]) * counts[1] * counts[2]);
        const std::int64_t expanded_begin = host_offsets_[structure];
        for (std::int64_t local = 0; local < source_count; ++local) {
            const std::size_t atom = static_cast<std::size_t>(source_begin + local);
            expansion_first[atom] = expanded_begin + local;
            expansion_stride[atom] = source_count;
            expansion_replicas[atom] = replicas;
        }
    }

    ensure_capacity(
        &numbers_, &numbers_capacity_, static_cast<std::size_t>(atoms_));
    ensure_capacity(
        &positions_, &positions_capacity_, static_cast<std::size_t>(atoms_) * 3U);
    ensure_capacity(
        &cells_, &cells_capacity_, host_cells_.size());
    ensure_capacity(
        &pbc_, &pbc_capacity_, host_pbc_.size());
    ensure_capacity(
        &offsets_, &offsets_capacity_, host_offsets_.size());
    ensure_capacity(
        &expansion_first_, &expansion_first_capacity_, expansion_first.size());
    ensure_capacity(
        &expansion_stride_, &expansion_stride_capacity_, expansion_stride.size());
    ensure_capacity(
        &expansion_replicas_, &expansion_replicas_capacity_, expansion_replicas.size());
    ensure_capacity(
        &expansion_counts_, &expansion_counts_capacity_, structure_count * 3U);

    const cudaStream_t stream = context.stream();
    ensure_and_copy(
        &cells_, &cells_capacity_, host_cells_.data(), host_cells_.size(), stream,
        "could not upload expanded NEP cells");
    ensure_and_copy(
        &pbc_, &pbc_capacity_, host_pbc_.data(), host_pbc_.size(), stream,
        "could not upload expanded NEP pbc");
    ensure_and_copy(
        &offsets_, &offsets_capacity_, host_offsets_.data(), host_offsets_.size(), stream,
        "could not upload expanded NEP offsets");
    ensure_and_copy(
        &expansion_first_, &expansion_first_capacity_, expansion_first.data(),
        expansion_first.size(), stream, "could not upload NEP expansion mapping");
    ensure_and_copy(
        &expansion_stride_, &expansion_stride_capacity_, expansion_stride.data(),
        expansion_stride.size(), stream, "could not upload NEP expansion strides");
    ensure_and_copy(
        &expansion_replicas_, &expansion_replicas_capacity_, expansion_replicas.data(),
        expansion_replicas.size(), stream, "could not upload NEP expansion replica counts");
    std::vector<std::int32_t> replication_flat(structure_count * 3U, 1);
    for (std::size_t structure = 0; structure < structure_count; ++structure) {
        for (int axis = 0; axis < 3; ++axis) {
            replication_flat[structure * 3U + static_cast<std::size_t>(axis)] =
                replication[structure][static_cast<std::size_t>(axis)];
        }
    }
    ensure_and_copy(
        &expansion_counts_, &expansion_counts_capacity_, replication_flat.data(),
        replication_flat.size(), stream, "could not upload NEP expansion counts");

    const auto blocks = static_cast<unsigned int>(
        (expanded_atom_count + 127U) / 128U);
    expand_nep_batch_kernel<<<blocks, 128, 0, stream>>>(
        static_cast<int>(structure_count), source.numbers(), source.positions(), source.cells(),
        source.offsets(), offsets_, expansion_counts_, numbers_, positions_);
    check_cuda(cudaGetLastError(), "CUDA NEP batch expansion failed");
    return true;
}

void DeviceBatch::ensure_positions_soa(CudaExecutionContext& context) {
    if (atoms_ == 0) {
        positions_soa_stride_ = positions_capacity_ / 3U;
        return;
    }
    if (positions_capacity_ == 0 || positions_capacity_ % 3U != 0) {
        throw std::runtime_error("invalid CUDA position capacity");
    }
    if (positions_capacity_ > positions_soa_capacity_) {
        if (positions_soa_ != nullptr) {
            check_cuda(cudaFree(positions_soa_), "could not release CUDA SoA positions");
            positions_soa_ = nullptr;
        }
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&positions_soa_),
                positions_capacity_ * sizeof(double)),
            "could not allocate CUDA SoA positions");
        positions_soa_capacity_ = positions_capacity_;
    }
    positions_soa_stride_ = positions_capacity_ / 3U;
    constexpr unsigned int block_size = 32;
    const auto blocks = static_cast<unsigned int>(
        (static_cast<std::size_t>(atoms_) + block_size - 1U) / block_size);
    stage_positions_aos_to_soa_kernel<<<blocks, block_size, 0, context.stream()>>>(
        static_cast<int>(atoms_), static_cast<int>(positions_soa_stride_), positions_, positions_soa_);
    check_cuda(cudaGetLastError(), "could not stage CUDA SoA positions");
}

void DeviceBatch::clear() noexcept {
    release(numbers_);
    release(positions_);
    release(positions_soa_);
    release(cells_);
    release(pbc_);
    release(offsets_);
    release(expansion_first_);
    release(expansion_stride_);
    release(expansion_replicas_);
    release(expansion_counts_);
    structures_ = 0;
    atoms_ = 0;
    numbers_capacity_ = 0;
    positions_capacity_ = 0;
    positions_soa_capacity_ = 0;
    positions_soa_stride_ = 0;
    cells_capacity_ = 0;
    pbc_capacity_ = 0;
    offsets_capacity_ = 0;
    expanded_ = false;
    original_atom_count_ = 0;
    host_cells_.clear();
    host_pbc_.clear();
    host_offsets_.clear();
    expansion_first_capacity_ = 0;
    expansion_stride_capacity_ = 0;
    expansion_replicas_capacity_ = 0;
    expansion_counts_capacity_ = 0;
}

} // namespace mdescriptor::cuda
