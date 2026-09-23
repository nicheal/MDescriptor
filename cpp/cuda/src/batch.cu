#include "mdescriptor/cuda/batch.hpp"
#include "mdescriptor/cuda/error.hpp"

#include "device_memory.cuh"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mdescriptor::cuda {
namespace {

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

std::array<std::int32_t, 3> nep_replication_counts(
    const double* source_cell,
    double cutoff) {
    double reference_cell[9] = {};
    double inverse[9] = {};
    const auto norms = reciprocal_row_norms(
        source_cell, reference_cell, inverse,
        "cannot expand a singular periodic NEP cell");
    std::array<std::int32_t, 3> counts{1, 1, 1};
    for (int axis = 0; axis < 3; ++axis) {
        const double required =
            2.0 * cutoff * norms[static_cast<std::size_t>(axis)];
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

bool DeviceBatch::requires_nep_expansion(
    const detail::StructureBatchView& batch,
    double cutoff) const {
    if (batch.structures < 0 || batch.atoms < 0
        || batch.offsets == nullptr || batch.cells == nullptr || batch.pbc == nullptr
        || !std::isfinite(cutoff) || cutoff <= 0.0) {
        throw std::invalid_argument("invalid batch for CUDA NEP expansion planning");
    }
    for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
        const std::int32_t* pbc = batch.pbc + structure * 3;
        const bool periodic = pbc[0] == 1 && pbc[1] == 1 && pbc[2] == 1;
        const bool isolated = pbc[0] == 0 && pbc[1] == 0 && pbc[2] == 0;
        if (!periodic && !isolated) {
            throw std::invalid_argument(
                "CUDA NEP supports all-zero or all-one pbc per structure");
        }
        if (!periodic) continue;
        const auto counts = nep_replication_counts(batch.cells + structure * 9, cutoff);
        if (counts[0] > 1 || counts[1] > 1 || counts[2] > 1) return true;
    }
    return false;
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
    ensure_and_upload(
        &numbers_, &numbers_capacity_, batch.numbers,
        static_cast<std::size_t>(atoms_), context.stream(), "could not upload numbers");
    ensure_and_upload(
        &positions_, &positions_capacity_, batch.positions,
        static_cast<std::size_t>(atoms_) * 3, context.stream(), "could not upload positions");
    ensure_and_upload(
        &cells_, &cells_capacity_, batch.cells,
        static_cast<std::size_t>(structures_) * 9, context.stream(), "could not upload cells");
    ensure_and_upload(
        &pbc_, &pbc_capacity_, batch.pbc,
        static_cast<std::size_t>(structures_) * 3, context.stream(), "could not upload pbc");
    ensure_and_upload(
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
    ensure_and_upload(
        &cells_, &cells_capacity_, host_cells_.data(), host_cells_.size(), stream,
        "could not upload expanded NEP cells");
    ensure_and_upload(
        &pbc_, &pbc_capacity_, host_pbc_.data(), host_pbc_.size(), stream,
        "could not upload expanded NEP pbc");
    ensure_and_upload(
        &offsets_, &offsets_capacity_, host_offsets_.data(), host_offsets_.size(), stream,
        "could not upload expanded NEP offsets");
    ensure_and_upload(
        &expansion_first_, &expansion_first_capacity_, expansion_first.data(),
        expansion_first.size(), stream, "could not upload NEP expansion mapping");
    ensure_and_upload(
        &expansion_stride_, &expansion_stride_capacity_, expansion_stride.data(),
        expansion_stride.size(), stream, "could not upload NEP expansion strides");
    ensure_and_upload(
        &expansion_replicas_, &expansion_replicas_capacity_, expansion_replicas.data(),
        expansion_replicas.size(), stream, "could not upload NEP expansion replica counts");
    std::vector<std::int32_t> replication_flat(structure_count * 3U, 1);
    for (std::size_t structure = 0; structure < structure_count; ++structure) {
        for (int axis = 0; axis < 3; ++axis) {
            replication_flat[structure * 3U + static_cast<std::size_t>(axis)] =
                replication[structure][static_cast<std::size_t>(axis)];
        }
    }
    ensure_and_upload(
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
    device_free(numbers_);
    device_free(positions_);
    device_free(positions_soa_);
    device_free(cells_);
    device_free(pbc_);
    device_free(offsets_);
    device_free(expansion_first_);
    device_free(expansion_stride_);
    device_free(expansion_replicas_);
    device_free(expansion_counts_);
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
