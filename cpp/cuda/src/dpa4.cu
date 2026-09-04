#include "mdescriptor/cuda/dpa4.hpp"
#include "mdescriptor/cuda/error.hpp"

#include "mdescriptor/dpa4_wigner.hpp"

#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace mdescriptor::cuda {

struct DeviceDpa4Model::DeviceArray {
    void* pointer = nullptr;
    std::size_t bytes = 0;

    ~DeviceArray() noexcept {
        if (pointer != nullptr) {
            (void)cudaFree(pointer);
        }
    }
};

namespace {

constexpr int kFullDim = 16;
constexpr int kReducedDim = 10;
constexpr int kChannels = 64;
constexpr int kFrames = 3;
constexpr int kGridSize = 152;
constexpr int kGridCoeff = 48;
constexpr int kGridScratchChannels = 384;
constexpr std::size_t kGridScratchStride =
    static_cast<std::size_t>(kGridSize) * kGridScratchChannels;
// Keep enough nodes in each grid batch to amortize launch and dispatch
// overhead.  The previous 16-node tiles left the 152x48 grid
// contractions launch-bound on small and medium batches.  This scratch tile
// is still bounded independently of the atom count, so increasing it does
// not make the descriptor workspace scale quadratically with a batch.
constexpr int kGridTileNodes = 128;
constexpr float kEpsilon = 1.0e-7F;
constexpr float kNormEpsilon = 1.0e-5F;

__device__ __forceinline__ std::int64_t center_for_edge(
    const std::int64_t* graph_offsets,
    std::int64_t atoms,
    std::int64_t edge) {
    // CSR rows are monotonic.  The previous edge kernels linearly scanned
    // from atom zero for every edge, making center lookup O(edges * atoms)
    // for a large batch.  A lower-bound search keeps the same CSR semantics
    // while reducing the lookup to O(log(atoms)).
    std::int64_t lower = 0;
    std::int64_t upper = atoms;
    while (lower < upper) {
        const std::int64_t middle = lower + (upper - lower) / 2;
        if (graph_offsets[middle + 1] <= edge) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    return lower;
}

enum TopWeight : int {
    kTypeEmbedding = 0,
    kEnvRbf1,
    kEnvRbf2,
    kEnvTypeEmbedding,
    kEnvG1,
    kEnvG2,
    kEnvOutput,
    kFilmScaleNorm,
    kFilmShiftNorm,
    kRadialFreqs,
    kRadialLayer1,
    kRadialNormScale,
    kRadialLayer2,
    kWignerL2,
    kWignerL3,
    kWignerL3Exponents,
    kGieRows,
    kGieM0,
    kGieRadial,
    kGridTo,
    kGridFrom,
    kOutputLinear1,
    kOutputLinear2,
    kOutputScalarGate,
    kOutputGridLeft,
    kOutputGridRight,
    kOutputGridOut,
    kTopWeightCount,
};

enum BlockWeight : int {
    kPreNormScale = 0,
    kPreNormBias,
    kPreNormBalance,
    kPostNormScale,
    kPostNormBias,
    kPostNormBalance,
    kFfnNormScale,
    kFfnNormBias,
    kFfnNormBalance,
    kPreFocus,
    kPostFocus,
    kRadialMixer,
    kRadialChannelBasis,
    kSo2M0_0,
    kSo2M0_1,
    kSo2M0_2,
    kSo2M0_3,
    kSo2M1_0,
    kSo2M1_1,
    kSo2M1_2,
    kSo2M1_3,
    kSo2Gate_0,
    kSo2Gate_1,
    kSo2Gate_2,
    kAttnQkScale,
    kAttnQ,
    kAttnK,
    kAttnOutputGateScale,
    kAttnLogit,
    kAttnZBias,
    kAttnGate,
    kMessageScalarGate,
    kMessageFrameExpand,
    kMessageFrameContract,
    kMessageResidualScale,
    kFfnLinear1,
    kFfnLinear2,
    kFfnScalarGate,
    kFfnGridLeft,
    kFfnGridRight,
    kFfnGridRouter,
    kFfnGridOut,
    kBlockWeightCount,
};

template <typename Value>
std::vector<Value> payload_array(py::handle value, const char* name) {
    using Array = py::array_t<Value, py::array::c_style | py::array::forcecast>;
    const Array array = Array::ensure(value);
    if (!array || array.ndim() == 0) {
        throw std::invalid_argument(
            std::string("DPA4 payload field ") + name + " must be an array");
    }
    const auto info = array.request();
    if (info.size < 0) {
        throw std::invalid_argument(
            std::string("DPA4 payload field ") + name + " has an invalid size");
    }
    const auto* data = static_cast<const Value*>(info.ptr);
    return std::vector<Value>(data, data + info.size);
}

py::handle required(const py::dict& payload, const char* name) {
    if (!payload.contains(name)) {
        throw std::invalid_argument(
            std::string("DPA4 CUDA payload is missing ") + name);
    }
    return payload[name];
}

template <typename Value>
std::vector<Value> read_exact(
    const py::dict& payload,
    const char* name,
    std::size_t expected) {
    auto values = payload_array<Value>(required(payload, name), name);
    if (values.size() != expected) {
        throw std::invalid_argument(
            std::string("DPA4 CUDA payload has unexpected size for ") + name);
    }
    return values;
}

bool read_bool(const py::dict& payload, const char* name, bool default_value) {
    if (!payload.contains(name) || payload[name].is_none()) {
        return default_value;
    }
    if (!py::isinstance<py::bool_>(payload[name])) {
        throw std::invalid_argument(
            std::string("DPA4 CUDA payload field ") + name + " must be boolean");
    }
    return py::cast<bool>(payload[name]);
}

template <typename Value>
std::unique_ptr<DeviceDpa4Model::DeviceArray> upload_array(
    CudaExecutionContext& context,
    const std::vector<Value>& values,
    const char* operation) {
    auto result = std::make_unique<DeviceDpa4Model::DeviceArray>();
    result->bytes = values.size() * sizeof(Value);
    if (result->bytes == 0) {
        return result;
    }
    check_cuda(cudaSetDevice(context.device()), "could not select the DPA4 CUDA device");
    check_cuda(cudaMalloc(&result->pointer, result->bytes), operation);
    // A blocking copy is intentional here.  The input vector is a temporary
    // parser buffer, and model construction must not leave an asynchronous
    // DMA operation borrowing memory that has already gone out of scope.
    try {
        check_cuda(
            cudaMemcpy(
                result->pointer, values.data(), result->bytes,
                cudaMemcpyHostToDevice),
            operation);
    } catch (...) {
        (void)cudaFree(result->pointer);
        result->pointer = nullptr;
        throw;
    }
    return result;
}

template <typename Value>
const Value* device_data(
    const std::unique_ptr<DeviceDpa4Model::DeviceArray>& value) {
    return value == nullptr ? nullptr : static_cast<const Value*>(value->pointer);
}

struct HostBlock {
    bool pre_norm_enabled = false;
    bool post_norm_enabled = true;
    bool ffn_norm_enabled = true;
    std::array<std::vector<float>, kBlockWeightCount> values;
};

struct DeviceBlock {
    int pre_norm_enabled = 0;
    int post_norm_enabled = 0;
    int ffn_norm_enabled = 0;
    const float* pre_norm_scale = nullptr;
    const float* pre_norm_bias = nullptr;
    const float* pre_norm_balance = nullptr;
    const float* post_norm_scale = nullptr;
    const float* post_norm_bias = nullptr;
    const float* post_norm_balance = nullptr;
    const float* ffn_norm_scale = nullptr;
    const float* ffn_norm_bias = nullptr;
    const float* ffn_norm_balance = nullptr;
    const float* pre_focus = nullptr;
    const float* post_focus = nullptr;
    const float* radial_mixer = nullptr;
    const float* radial_channel_basis = nullptr;
    const float* so2_m0[4] = {};
    const float* so2_m1[4] = {};
    const float* so2_gate[3] = {};
    const float* attn_qk_scale = nullptr;
    const float* attn_q = nullptr;
    const float* attn_k = nullptr;
    const float* attn_output_gate_scale = nullptr;
    const float* attn_logit = nullptr;
    const float* attn_z_bias = nullptr;
    const float* attn_gate = nullptr;
    const float* message_scalar_gate = nullptr;
    const float* message_frame_expand = nullptr;
    const float* message_frame_contract = nullptr;
    const float* message_residual_scale = nullptr;
    const float* ffn_linear1 = nullptr;
    const float* ffn_linear2 = nullptr;
    const float* ffn_scalar_gate = nullptr;
    const float* ffn_grid_left = nullptr;
    const float* ffn_grid_right = nullptr;
    const float* ffn_grid_router = nullptr;
    const float* ffn_grid_out = nullptr;
};

struct DeviceModel {
    double rcut = 0.0;
    int ntypes = 0;
    int channels = 0;
    int feature_count = 0;
    const float* type_embedding = nullptr;
    const float* env_rbf1 = nullptr;
    const float* env_rbf2 = nullptr;
    const float* env_type_embedding = nullptr;
    const float* env_g1 = nullptr;
    const float* env_g2 = nullptr;
    const float* env_output = nullptr;
    const float* film_scale_norm = nullptr;
    const float* film_shift_norm = nullptr;
    float film_scale_strength_log = 0.0F;
    float film_shift_strength_log = 0.0F;
    const float* radial_freqs = nullptr;
    const float* radial_layer1 = nullptr;
    const float* radial_norm_scale = nullptr;
    const float* radial_layer2 = nullptr;
    const float* wigner_l2 = nullptr;
    const float* wigner_l3 = nullptr;
    const std::int64_t* wigner_l3_exponents = nullptr;
    const std::int64_t* gie_rows = nullptr;
    const std::int64_t* gie_m0 = nullptr;
    const std::int64_t* gie_radial = nullptr;
    const float* grid_to = nullptr;
    const float* grid_from = nullptr;
    const DeviceBlock* blocks = nullptr;
};

} // namespace

struct DeviceDpa4Model::Model {
    int ntypes = 0;
    int feature_count = 0;
    double rcut = 0.0;
    DeviceModel device;
    std::vector<std::int32_t> type_numbers;
    std::vector<float> wigner_l2;
    std::vector<float> wigner_l3;
    std::vector<std::int64_t> wigner_l3_exponents;
    std::array<std::unique_ptr<DeviceDpa4Model::DeviceArray>, kTopWeightCount>
        top;
    std::array<
        std::array<
            std::unique_ptr<DeviceDpa4Model::DeviceArray>, kBlockWeightCount>,
        3>
        blocks;
    std::array<DeviceBlock, 3> block_views;
    // These arrays are batch-shaped rather than model-shaped.  Keep them
    // with the model so repeated descriptor calls do not pay two
    // cudaMalloc/cudaFree pairs per batch.  Backend serializes compute calls,
    // so a single cached buffer is sufficient.
    mutable std::unique_ptr<DeviceDpa4Model::DeviceArray> type_indices_device;
    mutable std::unique_ptr<DeviceDpa4Model::DeviceArray> active_nodes_device;
};

namespace {

struct WorkspaceLayout {
    std::size_t state0 = 0;
    std::size_t state1 = 0;
    std::size_t pre_focus = 0;
    std::size_t aggregate = 0;
    std::size_t message = 0;
    std::size_t hidden = 0;
    std::size_t activation = 0;
    std::size_t radial = 0;
    std::size_t radial_compact = 0;
    std::size_t radial_input = 0;
    std::size_t radial_sign = 0;
    std::size_t radial_projection = 0;
    std::size_t radial_m1_output = 0;
    std::size_t envelope = 0;
    std::size_t radial_bias = 0;
    std::size_t local = 0;
    std::size_t edge_message = 0;
    std::size_t rotation = 0;
    std::size_t gie = 0;
    std::size_t scratch0 = 0;
    std::size_t scratch1 = 0;
    std::size_t scratch2 = 0;
    std::size_t grid_input = 0;
    std::size_t bytes = 0;
};

std::size_t align_bytes(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1U) / alignment * alignment;
}

WorkspaceLayout make_workspace_layout(
    std::size_t atoms,
    std::size_t edges) {
    WorkspaceLayout result;
    std::size_t cursor = 0;
    auto reserve = [&](std::size_t size, std::size_t alignment) {
        cursor = align_bytes(cursor, alignment);
        const std::size_t offset = cursor;
        cursor += size;
        return offset;
    };
    const std::size_t node_state = atoms * kFullDim * kChannels * sizeof(float);
    const std::size_t node_hidden = atoms * kFullDim * 1152U * sizeof(float);
    const std::size_t node_activation = atoms * kFullDim * 576U * sizeof(float);
    const std::size_t node_features = atoms * kFullDim * kChannels * sizeof(float);
    result.state0 = reserve(node_state, alignof(float));
    result.state1 = reserve(node_state, alignof(float));
    result.pre_focus = reserve(node_features, alignof(float));
    result.aggregate = reserve(node_features, alignof(float));
    result.message = reserve(node_features, alignof(float));
    result.hidden = reserve(node_hidden, alignof(float));
    result.activation = reserve(node_activation, alignof(float));
    result.radial = reserve(edges * 256U * sizeof(float), alignof(float));
    result.radial_compact = reserve(edges * 25U * sizeof(float), alignof(float));
    result.radial_input = reserve(edges * 256U * sizeof(float), alignof(float));
    result.radial_sign = reserve(edges * 2U * 192U * sizeof(float), alignof(float));
    result.radial_projection = reserve(edges * 256U * sizeof(float), alignof(float));
    result.radial_m1_output = reserve(edges * 2U * 384U * sizeof(float), alignof(float));
    result.envelope = reserve(edges * sizeof(float), alignof(float));
    result.radial_bias = reserve(edges * kChannels * sizeof(float), alignof(float));
    result.local = reserve(edges * kReducedDim * kChannels * sizeof(float), alignof(float));
    result.edge_message = reserve(edges * kFullDim * kChannels * sizeof(float), alignof(float));
    result.rotation = reserve(edges * kReducedDim * kFullDim * sizeof(float), alignof(float));
    result.gie = reserve(edges * 15U * sizeof(float), alignof(float));
    const std::size_t scratch =
        kGridScratchStride * static_cast<std::size_t>(kGridTileNodes)
        * sizeof(float);
    result.scratch0 = reserve(scratch, alignof(float));
    result.scratch1 = reserve(scratch, alignof(float));
    result.scratch2 = reserve(scratch, alignof(float));
    result.grid_input = reserve(
        static_cast<std::size_t>(kGridTileNodes) * kGridCoeff * 384U
            * sizeof(float),
        alignof(float));
    result.bytes = align_bytes(cursor, alignof(double));
    return result;
}

template <typename Value>
Value* workspace_data(void* workspace, std::size_t offset) {
    return reinterpret_cast<Value*>(
        static_cast<unsigned char*>(workspace) + offset);
}

constexpr int kMatmulTile = 16;
constexpr int kLargeMatmulTile = 64;
constexpr int kMatmulThreadTile = 4;

// Row-major GEMM used by the fixed DPA4 graph.  The descriptor only needs a
// small, known set of FP32 projections, so a tiled kernel avoids carrying an
// external BLAS runtime into the CUDA wheel while keeping the same FP32
// accumulation boundary as the previous implementation.
__global__ void row_major_strided_gemm_kernel(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ product,
    std::int64_t rows,
    std::int64_t columns,
    std::int64_t inner,
    std::int64_t left_row_stride,
    std::int64_t right_row_stride,
    std::int64_t product_row_stride,
    std::int64_t left_batch_stride,
    std::int64_t right_batch_stride,
    std::int64_t product_batch_stride,
    std::int64_t batch_count) {
    __shared__ float left_tile[kMatmulTile][kMatmulTile];
    __shared__ float right_tile[kMatmulTile][kMatmulTile];

    const std::int64_t row =
        static_cast<std::int64_t>(blockIdx.y) * kMatmulTile + threadIdx.y;
    const std::int64_t column =
        static_cast<std::int64_t>(blockIdx.x) * kMatmulTile + threadIdx.x;
    const std::int64_t batch = static_cast<std::int64_t>(blockIdx.z);
    if (batch >= batch_count) {
        return;
    }

    const float* left_batch = left + batch * left_batch_stride;
    const float* right_batch = right + batch * right_batch_stride;
    float* product_batch = product + batch * product_batch_stride;
    float value = 0.0F;
    for (std::int64_t start = 0; start < inner; start += kMatmulTile) {
        const std::int64_t left_column = start + threadIdx.x;
        const std::int64_t right_row = start + threadIdx.y;
        left_tile[threadIdx.y][threadIdx.x] =
            row < rows && left_column < inner
            ? left_batch[row * left_row_stride + left_column]
            : 0.0F;
        right_tile[threadIdx.y][threadIdx.x] =
            right_row < inner && column < columns
            ? right_batch[right_row * right_row_stride + column]
            : 0.0F;
        __syncthreads();
        // The tile loaders already zero-pad the last K tile.  Always walking
        // the fixed tile lets nvcc fully unroll this hot loop and avoids a
        // per-element bounds branch in the wide projections.
#pragma unroll
        for (int index = 0; index < kMatmulTile; ++index) {
            value = fmaf(
                left_tile[threadIdx.y][index],
                right_tile[index][threadIdx.x],
                value);
        }
        __syncthreads();
    }
    if (row < rows && column < columns) {
        product_batch[row * product_row_stride + column] = value;
    }
}

// The wide DPA4 projections have at least 64 output columns.  Giving each
// thread a 4x4 output tile cuts the number of blocks and reuses the same
// weight tile across sixteen accumulators.  The small kernel above remains
// the better choice for the 25/32-column radial projections and tiny batches.
__global__ void row_major_wide_gemm_kernel(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ product,
    int rows,
    int columns,
    int inner,
    int left_row_stride,
    int right_row_stride,
    int product_row_stride,
    int left_batch_stride,
    int right_batch_stride,
    int product_batch_stride,
    int batch_count) {
    constexpr int kThreadsPerRow = kLargeMatmulTile / kMatmulThreadTile;
    __shared__ float left_tile[kLargeMatmulTile][kMatmulTile];
    __shared__ float right_tile[kMatmulTile][kLargeMatmulTile];

    const int thread = static_cast<int>(threadIdx.x);
    const int thread_row = thread / kThreadsPerRow;
    const int thread_column = thread % kThreadsPerRow;
    const int row =
        static_cast<int>(blockIdx.y) * kLargeMatmulTile
        + thread_row * kMatmulThreadTile;
    const int column =
        static_cast<int>(blockIdx.x) * kLargeMatmulTile
        + thread_column * kMatmulThreadTile;
    const int batch = static_cast<int>(blockIdx.z);
    if (batch >= batch_count) {
        return;
    }

    const float* left_batch = left + batch * left_batch_stride;
    const float* right_batch = right + batch * right_batch_stride;
    float* product_batch = product + batch * product_batch_stride;
    float values[kMatmulThreadTile][kMatmulThreadTile] = {};

    for (int start = 0; start < inner; start += kMatmulTile) {
        for (int index = thread; index < kLargeMatmulTile * kMatmulTile;
             index += blockDim.x) {
            const int tile_row = index / kMatmulTile;
            const int tile_column = index % kMatmulTile;
            const int source_row =
                static_cast<int>(blockIdx.y) * kLargeMatmulTile
                + tile_row;
            const int source_column = start + tile_column;
            left_tile[tile_row][tile_column] =
                source_row < rows && source_column < inner
                ? left_batch[source_row * left_row_stride + source_column]
                : 0.0F;
        }
        for (int index = thread; index < kMatmulTile * kLargeMatmulTile;
             index += blockDim.x) {
            const int tile_row = index / kLargeMatmulTile;
            const int tile_column = index % kLargeMatmulTile;
            const int source_row = start + tile_row;
            const int source_column =
                static_cast<int>(blockIdx.x) * kLargeMatmulTile
                + tile_column;
            right_tile[tile_row][tile_column] =
                source_row < inner && source_column < columns
                ? right_batch[source_row * right_row_stride + source_column]
                : 0.0F;
        }
        __syncthreads();
        // As above, the last K tile is zero-padded, so keep the reduction
        // fixed-width and let nvcc unroll it.
#pragma unroll
        for (int index = 0; index < kMatmulTile; ++index) {
#pragma unroll
            for (int output_row = 0; output_row < kMatmulThreadTile;
                 ++output_row) {
                const float left_value = left_tile[thread_row * kMatmulThreadTile
                    + output_row][index];
#pragma unroll
                for (int output_column = 0;
                     output_column < kMatmulThreadTile; ++output_column) {
                    values[output_row][output_column] = fmaf(
                        left_value,
                        right_tile[index][thread_column * kMatmulThreadTile
                            + output_column],
                        values[output_row][output_column]);
                }
            }
        }
        __syncthreads();
    }
    for (int output_row = 0; output_row < kMatmulThreadTile; ++output_row) {
        for (int output_column = 0;
             output_column < kMatmulThreadTile; ++output_column) {
            const int destination_row = row + output_row;
            const int destination_column = column + output_column;
            if (destination_row < rows && destination_column < columns) {
                product_batch[destination_row * product_row_stride
                    + destination_column] = values[output_row][output_column];
            }
        }
    }
}

// A 128-column tile is a better fit for the large radial projections and
// grid/readout matrices: each block reuses the weight tile over twice as many
// output columns without increasing the 32 accumulators held by a thread.
template <int kRowTile, int kColumnTile, int kThreads>
__global__ void row_major_wide_columns_gemm_kernel(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ product,
    int rows,
    int columns,
    int inner,
    int left_row_stride,
    int right_row_stride,
    int product_row_stride,
    int left_batch_stride,
    int right_batch_stride,
    int product_batch_stride,
    int batch_count) {
    constexpr int kThreadsPerRow = kColumnTile / kMatmulThreadTile;
    constexpr int kThreadRows = kThreads / kThreadsPerRow;
    constexpr int kThreadTileRows =
        kRowTile / kThreadRows;
    __shared__ float left_tile[kRowTile][kMatmulTile];
    __shared__ float right_tile[kMatmulTile][kColumnTile];

    const int thread = static_cast<int>(threadIdx.x);
    const int thread_row = thread / kThreadsPerRow;
    const int thread_column = thread % kThreadsPerRow;
    const int row =
        static_cast<int>(blockIdx.y) * kRowTile
        + thread_row * kThreadTileRows;
    const int column =
        static_cast<int>(blockIdx.x) * kColumnTile
        + thread_column * kMatmulThreadTile;
    const int batch = static_cast<int>(blockIdx.z);
    if (batch >= batch_count) {
        return;
    }

    const float* left_batch = left + batch * left_batch_stride;
    const float* right_batch = right + batch * right_batch_stride;
    float* product_batch = product + batch * product_batch_stride;
    float values[kThreadTileRows][kMatmulThreadTile] = {};

    for (int start = 0; start < inner; start += kMatmulTile) {
        for (int index = thread; index < kRowTile * kMatmulTile;
             index += blockDim.x) {
            const int tile_row = index / kMatmulTile;
            const int tile_column = index % kMatmulTile;
            const int source_row =
                static_cast<int>(blockIdx.y) * kRowTile
                + tile_row;
            const int source_column = start + tile_column;
            left_tile[tile_row][tile_column] =
                source_row < rows && source_column < inner
                ? left_batch[source_row * left_row_stride + source_column]
                : 0.0F;
        }
        for (int index = thread; index < kMatmulTile * kColumnTile;
             index += blockDim.x) {
            const int tile_row = index / kColumnTile;
            const int tile_column = index % kColumnTile;
            const int source_row = start + tile_row;
            const int source_column =
                static_cast<int>(blockIdx.x) * kColumnTile
                + tile_column;
            right_tile[tile_row][tile_column] =
                source_row < inner && source_column < columns
                ? right_batch[source_row * right_row_stride + source_column]
                : 0.0F;
        }
        __syncthreads();
#pragma unroll
        for (int index = 0; index < kMatmulTile; ++index) {
#pragma unroll
            for (int output_row = 0; output_row < kThreadTileRows;
                 ++output_row) {
                const float left_value = left_tile[
                    thread_row * kThreadTileRows + output_row][index];
#pragma unroll
                for (int output_column = 0;
                     output_column < kMatmulThreadTile; ++output_column) {
                    values[output_row][output_column] = fmaf(
                        left_value,
                        right_tile[index][thread_column * kMatmulThreadTile
                            + output_column],
                        values[output_row][output_column]);
                }
            }
        }
        __syncthreads();
    }
    for (int output_row = 0; output_row < kThreadTileRows; ++output_row) {
        for (int output_column = 0;
             output_column < kMatmulThreadTile; ++output_column) {
            const int destination_row = row + output_row;
            const int destination_column = column + output_column;
            if (destination_row < rows && destination_column < columns) {
                product_batch[destination_row * product_row_stride
                    + destination_column] = values[output_row][output_column];
            }
        }
    }
}

void launch_row_major_gemm(
    const float* left,
    const float* right,
    float* product,
    std::int64_t rows,
    std::int64_t columns,
    std::int64_t inner,
    std::int64_t left_row_stride,
    std::int64_t right_row_stride,
    std::int64_t product_row_stride,
    std::int64_t left_batch_stride,
    std::int64_t right_batch_stride,
    std::int64_t product_batch_stride,
    std::int64_t batch_count,
    cudaStream_t stream,
    const char* operation) {
    if (rows <= 0 || columns <= 0 || inner <= 0 || batch_count <= 0) {
        return;
    }
    const auto fits_fast_index = [](std::int64_t value) {
        return value <= static_cast<std::int64_t>(
            std::numeric_limits<int>::max());
    };
    const bool fast_indexable =
        fits_fast_index(rows) && fits_fast_index(columns)
        && fits_fast_index(inner) && fits_fast_index(left_row_stride)
        && fits_fast_index(right_row_stride)
        && fits_fast_index(product_row_stride)
        && fits_fast_index(left_batch_stride)
        && fits_fast_index(right_batch_stride)
        && fits_fast_index(product_batch_stride)
        && fits_fast_index(batch_count);
    if (fast_indexable && columns >= 128 && rows >= 32) {
        constexpr int kColumnTile = 128;
        const dim3 grid(
            static_cast<unsigned int>(
                (columns + kColumnTile - 1) / kColumnTile),
            static_cast<unsigned int>(
                (rows + kLargeMatmulTile - 1) / kLargeMatmulTile),
            static_cast<unsigned int>(batch_count));
        row_major_wide_columns_gemm_kernel<
            kLargeMatmulTile, kColumnTile, 256><<<
            grid, dim3(16 * 16, 1, 1), 0, stream>>>(
            left, right, product,
            static_cast<int>(rows), static_cast<int>(columns),
            static_cast<int>(inner), static_cast<int>(left_row_stride),
            static_cast<int>(right_row_stride),
            static_cast<int>(product_row_stride),
            static_cast<int>(left_batch_stride),
            static_cast<int>(right_batch_stride),
            static_cast<int>(product_batch_stride),
            static_cast<int>(batch_count));
    } else if (fast_indexable && columns >= kLargeMatmulTile && rows >= 32) {
        const dim3 grid(
            static_cast<unsigned int>(
                (columns + kLargeMatmulTile - 1) / kLargeMatmulTile),
            static_cast<unsigned int>(
                (rows + kLargeMatmulTile - 1) / kLargeMatmulTile),
            static_cast<unsigned int>(batch_count));
        row_major_wide_gemm_kernel<<<
            grid, dim3(16 * 16, 1, 1), 0, stream>>>(
            left, right, product,
            static_cast<int>(rows), static_cast<int>(columns),
            static_cast<int>(inner), static_cast<int>(left_row_stride),
            static_cast<int>(right_row_stride),
            static_cast<int>(product_row_stride),
            static_cast<int>(left_batch_stride),
            static_cast<int>(right_batch_stride),
            static_cast<int>(product_batch_stride),
            static_cast<int>(batch_count));
    } else {
        const dim3 grid(
            static_cast<unsigned int>((columns + kMatmulTile - 1) / kMatmulTile),
            static_cast<unsigned int>((rows + kMatmulTile - 1) / kMatmulTile),
            static_cast<unsigned int>(batch_count));
        row_major_strided_gemm_kernel<<<
            grid, dim3(kMatmulTile, kMatmulTile, 1), 0, stream>>>(
            left, right, product, rows, columns, inner,
            left_row_stride, right_row_stride, product_row_stride,
            left_batch_stride, right_batch_stride, product_batch_stride,
            batch_count);
    }
    check_cuda(cudaGetLastError(), operation);
}

__device__ __forceinline__ int degree_for_row(int row) {
    return row == 0 ? 0 : row < 4 ? 1 : row < 9 ? 2 : 3;
}

__device__ __forceinline__ int reduced_degree(int row) {
    return row < 4 ? row : row < 7 ? row - 3 : row - 6;
}

__device__ __forceinline__ int reduced_group(int row) {
    return row < 4 ? 0 : 1;
}

__device__ __forceinline__ float d_sigmoid(float value) {
    return 1.0F / (1.0F + expf(-value));
}

__device__ __forceinline__ float d_silu(float value) {
    return value * d_sigmoid(value);
}

__device__ __forceinline__ float d_softplus(float value) {
    return value > 20.0F ? value : log1pf(expf(value));
}

__device__ __forceinline__ float d_affine(
    const float* weights,
    int input_width,
    int output_width,
    const float* input,
    int output) {
    // Match the native DPA4 contraction boundary.  Its wide projections use
    // fp64 accumulation before the checkpoint's fp32 storage boundary, while
    // the short projections stay on the fp32/FMA path.  This is important for
    // the 512-channel environment projection and 576-channel readout, where
    // cancellation otherwise moves small descriptor components noticeably.
    if (input_width >= 512) {
        double value = 0.0;
        for (int index = 0; index < input_width; ++index) {
            value += static_cast<double>(input[index])
                * static_cast<double>(weights[index * output_width + output]);
        }
        return static_cast<float>(value);
    }
    float value = 0.0F;
    for (int index = 0; index < input_width; ++index) {
        value += input[index] * weights[index * output_width + output];
    }
    return value;
}

__device__ __forceinline__ Dpa4Quaternion d_normalize_quaternion(
    const Dpa4Quaternion& quaternion) {
    // Keep the same double-precision norm used by the host Wigner helper;
    // only the normalized quaternion itself crosses back to fp32.
    const double w = static_cast<double>(quaternion.w);
    const double x = static_cast<double>(quaternion.x);
    const double y = static_cast<double>(quaternion.y);
    const double z = static_cast<double>(quaternion.z);
    const double eps = static_cast<double>(kEpsilon);
    const float divisor = static_cast<float>(sqrt(
        w * w + x * x + y * y + z * z + eps * eps));
    return {
        quaternion.w / divisor,
        quaternion.x / divisor,
        quaternion.y / divisor,
        quaternion.z / divisor,
    };
}

__device__ __forceinline__ float d_smooth_step_cinf(float value) {
    const float clamped = fmaxf(0.0F, fminf(1.0F, value));
    if (clamped <= 0.0F) {
        return 0.0F;
    }
    if (clamped >= 1.0F) {
        return 1.0F;
    }
    constexpr float dtype_epsilon = 1.1920928955078125e-7F;
    const float left = expf(-1.0F / fmaxf(clamped, dtype_epsilon));
    const float right = expf(-1.0F / fmaxf(1.0F - clamped, dtype_epsilon));
    return left / (left + right);
}

__device__ __forceinline__ Dpa4Quaternion d_edge_quaternion(
    float x,
    float y,
    float z) {
    const float length = sqrtf(x * x + y * y + z * z + kEpsilon * kEpsilon);
    const float edge_length = sqrtf(length * length + kEpsilon * kEpsilon);
    const float ux = x / edge_length;
    const float uy = y / edge_length;
    const float uz = z / edge_length;
    const Dpa4Quaternion q_pos = d_normalize_quaternion({
        1.0F + uz, uy, -ux, 0.0F,
    });
    const Dpa4Quaternion q_neg = d_normalize_quaternion({
        -ux, 0.0F, 1.0F - uz, uy,
    });
    const float weight = d_smooth_step_cinf(0.5F * (uz + 1.0F));
    const float dot = q_neg.w * q_pos.w + q_neg.x * q_pos.x
        + q_neg.y * q_pos.y + q_neg.z * q_pos.z;
    const float sign = dot < 0.0F ? -1.0F : 1.0F;
    const Dpa4Quaternion blended = {
        (1.0F - weight) * q_neg.w + weight * sign * q_pos.w,
        (1.0F - weight) * q_neg.x + weight * sign * q_pos.x,
        (1.0F - weight) * q_neg.y + weight * sign * q_pos.y,
        (1.0F - weight) * q_neg.z + weight * sign * q_pos.z,
    };
    return d_normalize_quaternion(blended);
}

__device__ __forceinline__ void d_l1_block(
    const Dpa4Quaternion& quaternion,
    float* output) {
    const float w = quaternion.w;
    const float x = quaternion.x;
    const float y = quaternion.y;
    const float z = quaternion.z;
    const float x2 = x * x;
    const float y2 = y * y;
    const float z2 = z * z;
    const float xy = x * y;
    const float xz = x * z;
    const float yz = y * z;
    const float wx = w * x;
    const float wy = w * y;
    const float wz = w * z;
    const float cartesian[9] = {
        1.0F - 2.0F * (y2 + z2), 2.0F * (xy - wz), 2.0F * (xz + wy),
        2.0F * (xy + wz), 1.0F - 2.0F * (x2 + z2), 2.0F * (yz - wx),
        2.0F * (xz - wy), 2.0F * (yz + wx), 1.0F - 2.0F * (x2 + y2),
    };
    constexpr int permutation[3] = {1, 2, 0};
    constexpr float signs[3] = {-1.0F, -1.0F, 1.0F};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            output[row * 3 + column] =
                cartesian[permutation[row] * 3 + permutation[column]]
                * signs[row] * signs[column];
        }
    }
}

__device__ void d_l2_block(
    const Dpa4Quaternion& quaternion,
    const DeviceModel& model,
    float* output) {
    const float components[4] = {
        quaternion.w, quaternion.x, quaternion.y, quaternion.z,
    };
    float q2[16];
    for (int first = 0; first < 4; ++first) {
        for (int second = 0; second < 4; ++second) {
            q2[first * 4 + second] = components[first] * components[second];
        }
    }
    for (int row = 0; row < 5; ++row) {
        for (int column = 0; column < 5; ++column) {
            float value = 0.0F;
            const int coefficient_offset = (row * 5 + column) * 256;
            for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < 4; ++b) {
                    for (int c = 0; c < 4; ++c) {
                        for (int d = 0; d < 4; ++d) {
                            const float q4 = q2[a * 4 + b] * q2[c * 4 + d];
                            const int q4_index = ((a * 4 + b) * 4 + c) * 4 + d;
                            value += model.wigner_l2[coefficient_offset + q4_index] * q4;
                        }
                    }
                }
            }
            output[row * 5 + column] = value;
        }
    }
}

__device__ void d_l3_block(
    const Dpa4Quaternion& quaternion,
    const DeviceModel& model,
    float* output) {
    float powers[4][7];
    const float components[4] = {
        quaternion.w, quaternion.x, quaternion.y, quaternion.z,
    };
    for (int component = 0; component < 4; ++component) {
        powers[component][0] = 1.0F;
        for (int power = 1; power <= 6; ++power) {
            powers[component][power] =
                powers[component][power - 1] * components[component];
        }
    }
    for (int row = 0; row < 7; ++row) {
        for (int column = 0; column < 7; ++column) {
            float value = 0.0F;
            const int coefficient_offset = (row * 7 + column) * 84;
            for (int monomial = 0; monomial < 84; ++monomial) {
                const int exponent_offset = monomial * 4;
                float term = powers[0][static_cast<int>(
                    model.wigner_l3_exponents[exponent_offset])]
                    * powers[1][static_cast<int>(
                        model.wigner_l3_exponents[exponent_offset + 1])];
                term *= powers[2][static_cast<int>(
                    model.wigner_l3_exponents[exponent_offset + 2])]
                    * powers[3][static_cast<int>(
                        model.wigner_l3_exponents[exponent_offset + 3])];
                value += model.wigner_l3[coefficient_offset + monomial] * term;
            }
            output[row * 7 + column] = value;
        }
    }
}

__global__ void build_rotation_gie_kernel(
    const std::int64_t* graph_offsets,
    const std::int32_t* graph_atoms,
    const std::int32_t* graph_shifts,
    const double* displacements,
    std::int64_t atoms,
    DeviceModel model,
    float* rotation,
    float* gie) {
    const std::size_t edge = static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    const std::int64_t total = graph_offsets[atoms];
    if (edge >= static_cast<std::size_t>(total)) {
        return;
    }
    const std::int64_t center = center_for_edge(
        graph_offsets, atoms, static_cast<std::int64_t>(edge));
    const std::int32_t neighbor = graph_atoms[edge];
    const bool exact_self = neighbor == center
        && graph_shifts != nullptr
        && graph_shifts[edge * 3] == 0
        && graph_shifts[edge * 3 + 1] == 0
        && graph_shifts[edge * 3 + 2] == 0;
    float* rotation_destination = rotation + edge * kReducedDim * kFullDim;
    float* gie_destination = gie + edge * 15;
    for (int index = 0; index < kReducedDim * kFullDim; ++index) {
        rotation_destination[index] = 0.0F;
    }
    for (int index = 0; index < 15; ++index) {
        gie_destination[index] = 0.0F;
    }
    if (exact_self) {
        return;
    }
    const float x = static_cast<float>(displacements[edge * 3]);
    const float y = static_cast<float>(displacements[edge * 3 + 1]);
    const float z = static_cast<float>(displacements[edge * 3 + 2]);
    const Dpa4Quaternion quaternion = d_edge_quaternion(x, y, z);
    float l1[9];
    float l2[25];
    float l3[49];
    d_l1_block(quaternion, l1);
    d_l2_block(quaternion, model, l2);
    d_l3_block(quaternion, model, l3);
    constexpr int selected_rows[kReducedDim] = {
        0, 2, 6, 12, 1, 5, 11, 3, 7, 13,
    };
    for (int reduced = 0; reduced < kReducedDim; ++reduced) {
        const int row = selected_rows[reduced];
        if (row == 0) {
            rotation_destination[reduced * kFullDim] = 1.0F;
        } else if (row < 4) {
            for (int column = 0; column < 3; ++column) {
                rotation_destination[reduced * kFullDim + column + 1] =
                    l1[(row - 1) * 3 + column];
            }
        } else if (row < 9) {
            for (int column = 0; column < 5; ++column) {
                rotation_destination[reduced * kFullDim + column + 4] =
                    l2[(row - 4) * 5 + column];
            }
        } else {
            for (int column = 0; column < 7; ++column) {
                rotation_destination[reduced * kFullDim + column + 9] =
                    l3[(row - 9) * 7 + column];
            }
        }
    }
    for (int item = 0; item < 15; ++item) {
        const int row = item + 1;
        const int degree = static_cast<int>(sqrtf(static_cast<float>(row)));
        const int column = degree * (degree + 1);
        if (row < 4) {
            gie_destination[item] = l1[(column - 1) * 3 + row - 1];
        } else if (row < 9) {
            gie_destination[item] = l2[(column - 4) * 5 + row - 4];
        } else {
            gie_destination[item] = l3[(column - 9) * 7 + row - 9];
        }
    }
}

__global__ void prepare_film_kernel(
    std::int64_t atoms,
    const float* env_matrix,
    DeviceModel model,
    float* film) {
    const std::int64_t center = static_cast<std::int64_t>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (center >= atoms) {
        return;
    }
    __shared__ float matrix[512];
    const float* source = env_matrix + center * 1024;
    for (int index = lane; index < 512; index += blockDim.x) {
        matrix[index] = source[index];
    }
    __syncthreads();
    float* destination = film + center * 1024;
    for (int value = lane; value < 128; value += blockDim.x) {
        double result = 0.0;
        for (int index = 0; index < 512; ++index) {
            result += static_cast<double>(matrix[index])
                * static_cast<double>(model.env_output[index * 128 + value]);
        }
        destination[value] = static_cast<float>(result);
    }
}

// The environment MLP is shared by every edge, so doing all of its matrix
// products inside one thread per center leaves the GPU mostly idle.  Keep the
// geometry/radial work as a light edge kernel, then use the same edge-major
// layout as the rest of the descriptor for batched tiled projections.
__global__ void prepare_geometry_kernel(
    const std::int64_t* graph_offsets,
    const std::int32_t* graph_atoms,
    const std::int32_t* graph_shifts,
    const double* displacements,
    std::int64_t atoms,
    std::int64_t edges,
    DeviceModel model,
    std::int32_t* active_nodes,
    float* radial_compact,
    float* envelopes) {
    const std::int64_t edge =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (edge >= edges) {
        return;
    }
    const std::int64_t center = center_for_edge(graph_offsets, atoms, edge);
    const std::int32_t neighbor = graph_atoms[edge];
    const bool exact_self = neighbor == center
        && graph_shifts != nullptr
        && graph_shifts[edge * 3] == 0
        && graph_shifts[edge * 3 + 1] == 0
        && graph_shifts[edge * 3 + 2] == 0;
    float* compact = radial_compact + edge * 25;
    for (int value = 0; value < 25; ++value) {
        compact[value] = 0.0F;
    }
    envelopes[edge] = 0.0F;
    if (exact_self) {
        return;
    }
    atomicExch(active_nodes + center, 1);

    const float dx = static_cast<float>(displacements[edge * 3]);
    const float dy = static_cast<float>(displacements[edge * 3 + 1]);
    const float dz = static_cast<float>(displacements[edge * 3 + 2]);
    const float distance = sqrtf(
        dx * dx + dy * dy + dz * dz + kEpsilon * kEpsilon);
    const float inv_r = 1.0F / distance;
    float u = (static_cast<float>(model.rcut) - distance)
        / static_cast<float>(model.rcut);
    u = fmaxf(0.0F, fminf(1.0F, u));
    const float x = 1.0F - u;
    float series = 35.0F;
    series = 20.0F + x * series;
    series = 10.0F + x * series;
    series = 4.0F + x * series;
    series = 1.0F + x * series;
    const float envelope = u * u * u * u * series;
    envelopes[edge] = envelope;

    float radial_series = 84.0F;
    radial_series = 56.0F + x * radial_series;
    radial_series = 35.0F + x * radial_series;
    radial_series = 20.0F + x * radial_series;
    radial_series = 10.0F + x * radial_series;
    radial_series = 4.0F + x * radial_series;
    radial_series = 1.0F + x * radial_series;
    const float radial_envelope = u * u * u * u * radial_series;
    constexpr float pi = 3.1415927410125732422F;
    for (int radial = 0; radial < 16; ++radial) {
        const float argument = distance * model.radial_freqs[radial];
        const float zarg = argument / pi;
        const float sinc = zarg == 0.0F
            ? 1.0F : sinf(pi * zarg) / (pi * zarg);
        compact[radial] = model.radial_freqs[radial] * sinc * radial_envelope;
    }
    (void)inv_r;
}

__global__ void environment_silu_kernel(
    std::int64_t edges,
    float* values,
    int offset,
    int width) {
    const std::int64_t flat =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= edges * static_cast<std::int64_t>(width)) {
        return;
    }
    const std::int64_t edge = flat / width;
    const int index = static_cast<int>(flat % width);
    values[edge * 256 + offset + index] =
        d_silu(values[edge * 256 + offset + index]);
}

__global__ void prepare_environment_input_kernel(
    const std::int64_t* graph_offsets,
    const std::int32_t* graph_atoms,
    const std::int32_t* graph_shifts,
    const std::int32_t* type_indices,
    std::int64_t atoms,
    std::int64_t edges,
    const float* radial_input,
    DeviceModel model,
    float* environment_input) {
    const std::int64_t edge =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (edge >= edges) {
        return;
    }
    const std::int64_t center = center_for_edge(graph_offsets, atoms, edge);
    const std::int32_t neighbor = graph_atoms[edge];
    const bool exact_self = neighbor == center
        && graph_shifts != nullptr
        && graph_shifts[edge * 3] == 0
        && graph_shifts[edge * 3 + 1] == 0
        && graph_shifts[edge * 3 + 2] == 0;
    float* destination = environment_input + edge * 256;
    if (exact_self) {
        for (int index = 0; index < 64; ++index) {
            destination[index] = 0.0F;
        }
        return;
    }
    const int center_type = type_indices[center];
    const int neighbor_type = type_indices[neighbor];
    for (int index = 0; index < 32; ++index) {
        destination[index] = radial_input[edge * 256 + 32 + index];
    }
    for (int index = 0; index < 16; ++index) {
        destination[32 + index] =
            model.env_type_embedding[neighbor_type * 16 + index];
        destination[48 + index] =
            model.env_type_embedding[center_type * 16 + index];
    }
}

__global__ void prepare_environment_aggregate_kernel(
    const std::int64_t* graph_offsets,
    const double* displacements,
    std::int64_t atoms,
    const float* envelopes,
    const float* radial_bias,
    float* env_matrix,
    float* degree_inverse) {
    const std::int64_t center = static_cast<std::int64_t>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (center >= atoms) {
        return;
    }
    const std::int64_t begin = graph_offsets[center];
    const std::int64_t end = graph_offsets[center + 1];
    __shared__ double env_agg[256];
    __shared__ double inverse;
    if (lane == 0) {
        double degree_sum = 0.0;
        for (std::int64_t edge = begin; edge < end; ++edge) {
            const double envelope = static_cast<double>(envelopes[edge]);
            degree_sum += envelope * envelope;
        }
        inverse = 1.0 / sqrt(degree_sum + 0.25);
        degree_inverse[center * 1024] = static_cast<float>(inverse);
    }
    for (int index = lane; index < 256; index += blockDim.x) {
        const int coordinate = index / 64;
        const int channel = index % 64;
        double value = 0.0;
        for (std::int64_t edge = begin; edge < end; ++edge) {
            const float envelope = envelopes[edge];
            const float dx = static_cast<float>(displacements[edge * 3]);
            const float dy = static_cast<float>(displacements[edge * 3 + 1]);
            const float dz = static_cast<float>(displacements[edge * 3 + 2]);
            const float distance = sqrtf(
                dx * dx + dy * dy + dz * dz + kEpsilon * kEpsilon);
            const float inv_r = 1.0F / distance;
            const float s = envelope * inv_r;
            const float component = coordinate == 0
                ? s
                : coordinate == 1
                ? s * dx * inv_r
                : coordinate == 2
                ? s * dy * inv_r
                : s * dz * inv_r;
            value += static_cast<double>(component)
                * static_cast<double>(radial_bias[edge * 64 + channel]);
        }
        env_agg[index] = value;
    }
    __syncthreads();
    for (int index = lane; index < 256; index += blockDim.x) {
        env_agg[index] *= inverse;
    }
    __syncthreads();
    float* destination = env_matrix + center * 1024;
    for (int flat = lane; flat < 512; flat += blockDim.x) {
        const int row = flat / 8;
        const int column = flat % 8;
        float value = 0.0F;
        for (int coordinate = 0; coordinate < 4; ++coordinate) {
            value += static_cast<float>(
                env_agg[coordinate * 64 + row]
                * env_agg[coordinate * 64 + column]);
        }
        destination[flat] = value;
    }
}

__global__ void prepare_finalize_kernel(
    const std::int64_t* graph_offsets,
    const std::int32_t* graph_atoms,
    const std::int32_t* graph_shifts,
    const std::int32_t* type_indices,
    std::int64_t atoms,
    const float* film,
    const float* degree_inverse,
    const float* gie,
    DeviceModel model,
    const std::int32_t* active_nodes,
    float* radial_output,
    float* state) {
    const std::int64_t center = static_cast<std::int64_t>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (center >= atoms) {
        return;
    }
    const int center_type = type_indices[center];
    const float* center_film = film + center * 1024;
    __shared__ float scale_inverse;
    __shared__ float shift_inverse;
    if (lane == 0) {
        double scale_sq = 0.0;
        double shift_sq = 0.0;
        for (int channel = 0; channel < 64; ++channel) {
            scale_sq += static_cast<double>(center_film[channel])
                * center_film[channel];
            shift_sq += static_cast<double>(center_film[64 + channel])
                * center_film[64 + channel];
        }
        scale_inverse = static_cast<float>(
            1.0 / sqrt(scale_sq / 64.0 + static_cast<double>(kEpsilon)));
        shift_inverse = static_cast<float>(
            1.0 / sqrt(shift_sq / 64.0 + static_cast<double>(kEpsilon)));
    }
    __syncthreads();
    const float scale_strength = expf(model.film_scale_strength_log);
    const float shift_strength = expf(model.film_shift_strength_log);
    float* center_state = state + center * kFullDim * kChannels;
    for (int index = lane; index < kFullDim * kChannels; index += blockDim.x) {
        center_state[index] = 0.0F;
    }
    __syncthreads();
    for (int channel = lane; channel < 64; channel += blockDim.x) {
        const float scale = 1.0F + scale_strength * tanhf(
            center_film[channel] * scale_inverse * model.film_scale_norm[channel]);
        const float shift = shift_strength * tanhf(
            center_film[64 + channel] * shift_inverse
            * model.film_shift_norm[channel]);
        center_state[channel] =
            model.type_embedding[center_type * 64 + channel] * scale + shift;
    }
    __syncthreads();
    if (lane != 0) {
        return;
    }
    const std::int64_t begin = graph_offsets[center];
    const std::int64_t end = graph_offsets[center + 1];
    const float inv_degree = degree_inverse[center * 1024];
    for (std::int64_t edge = begin; edge < end; ++edge) {
        const std::int32_t neighbor = graph_atoms[edge];
        const bool exact_self = neighbor == center
            && graph_shifts != nullptr
            && graph_shifts[edge * 3] == 0
            && graph_shifts[edge * 3 + 1] == 0
            && graph_shifts[edge * 3 + 2] == 0;
        if (exact_self) {
            continue;
        }
        for (int item = 0; item < 15; ++item) {
            const int row = static_cast<int>(model.gie_rows[item]);
            const int radial_slot = static_cast<int>(model.gie_radial[item]);
            if (row <= 0 || row >= kFullDim || radial_slot < 0 || radial_slot >= 3) {
                continue;
            }
            const float coupling = gie[edge * 15 + item];
            for (int channel = 0; channel < 64; ++channel) {
                center_state[row * 64 + channel] +=
                    coupling
                    * radial_output[edge * 256 + (radial_slot + 1) * 64 + channel]
                    * inv_degree;
            }
        }
        const int neighbor_type = type_indices[neighbor];
        for (int degree = 0; degree < 4; ++degree) {
            for (int channel = 0; channel < 64; ++channel) {
                radial_output[edge * 256 + degree * 64 + channel] +=
                    model.type_embedding[center_type * 64 + channel]
                    + model.type_embedding[neighbor_type * 64 + channel];
            }
        }
    }
    (void)active_nodes;
}

__global__ void equivariant_copy_or_norm_kernel(
    const float* input,
    std::int64_t atoms,
    int enabled,
    const float* scale,
    const float* bias,
    const float* balance,
    float* output) {
    const std::int64_t node =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node >= atoms) {
        return;
    }
    const float* source = input + node * kFullDim * kChannels;
    float* destination = output + node * kFullDim * kChannels;
    if (!enabled) {
        for (int index = 0; index < kFullDim * kChannels; ++index) {
            destination[index] = source[index];
        }
        return;
    }
    double mean = 0.0;
    for (int channel = 0; channel < 64; ++channel) {
        mean += source[channel];
    }
    mean /= 64.0;
    double variance = 0.0;
    for (int row = 0; row < kFullDim; ++row) {
        const int degree = degree_for_row(row);
        for (int channel = 0; channel < 64; ++channel) {
            const double value = row == 0
                ? static_cast<double>(source[row * 64 + channel]) - mean
                : source[row * 64 + channel];
            variance += value * value * balance[row];
        }
        (void)degree;
    }
    const float inverse = static_cast<float>(
        1.0 / sqrt(variance + static_cast<double>(kNormEpsilon)));
    for (int row = 0; row < kFullDim; ++row) {
        const int degree = degree_for_row(row);
        for (int channel = 0; channel < 64; ++channel) {
            const float value = row == 0
                ? static_cast<float>(
                    static_cast<double>(source[row * 64 + channel]) - mean)
                : source[row * 64 + channel];
            destination[row * 64 + channel] =
                value * inverse * scale[degree * 64 + channel]
                + (row == 0 ? bias[channel] : 0.0F);
        }
    }
}

__global__ void copy_state_kernel(
    const float* source,
    std::int64_t atoms,
    float* destination) {
    const std::int64_t node = static_cast<std::int64_t>(blockIdx.x);
    if (node >= atoms) {
        return;
    }
    for (int index = static_cast<int>(threadIdx.x);
         index < kFullDim * kChannels; index += blockDim.x) {
        destination[node * kFullDim * kChannels + index] =
            source[node * kFullDim * kChannels + index];
    }
}

__global__ void restore_inactive_state_kernel(
    const float* snapshot,
    const std::int32_t* active,
    std::int64_t atoms,
    float* state) {
    const std::int64_t node = static_cast<std::int64_t>(blockIdx.x);
    if (node >= atoms) {
        return;
    }
    if (active[node] != 0) {
        return;
    }
    for (int index = static_cast<int>(threadIdx.x);
         index < kFullDim * kChannels; index += blockDim.x) {
        state[node * kFullDim * kChannels + index] =
            snapshot[node * kFullDim * kChannels + index];
    }
}

__global__ void edge_local_from_state_kernel(
    const std::int64_t* graph_offsets,
    const std::int32_t* graph_atoms,
    const std::int32_t* graph_shifts,
    std::int64_t atoms,
    const float* rotation,
    const float* pre_focus,
    const float* radial,
    float* local,
    float* radial_bias) {
    // This is another independent per-edge matrix product.  A cooperative
    // block loads the neighbor feature once into shared memory and computes
    // all 10*64 outputs in parallel instead of making one thread perform the
    // whole 10*64*16 contraction.
    const std::size_t edge = static_cast<std::size_t>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const std::int64_t total = graph_offsets[atoms];
    if (edge >= static_cast<std::size_t>(total)) {
        return;
    }
    const std::int64_t center = center_for_edge(
        graph_offsets, atoms, static_cast<std::int64_t>(edge));
    const std::int32_t neighbor = graph_atoms[edge];
    const bool exact_self = neighbor == center
        && graph_shifts != nullptr
        && graph_shifts[edge * 3] == 0
        && graph_shifts[edge * 3 + 1] == 0
        && graph_shifts[edge * 3 + 2] == 0;
    float* destination = local + edge * kReducedDim * kChannels;
    float* bias_destination = radial_bias + edge * kChannels;
    if (exact_self) {
        for (int index = lane; index < kReducedDim * kChannels; index += blockDim.x) {
            destination[index] = 0.0F;
        }
        for (int channel = lane; channel < kChannels; channel += blockDim.x) {
            bias_destination[channel] = 0.0F;
        }
        return;
    }
    for (int channel = lane; channel < kChannels; channel += blockDim.x) {
        bias_destination[channel] = radial[edge * 256 + channel];
    }
    __shared__ float source_values[1024];
    const float* source = pre_focus + neighbor * kFullDim * kChannels;
    for (int index = lane; index < kFullDim * kChannels; index += blockDim.x) {
        source_values[index] = source[index];
    }
    __syncthreads();
    for (int index = lane; index < kReducedDim * kChannels; index += blockDim.x) {
        const int reduced = index / kChannels;
        const int channel = index % kChannels;
        // The native CPU ABI accumulates this small 16-term rotation in
        // fp64 before storing the fp32 local feature.  The cancellation is
        // substantial for axis-aligned edges, so retaining fp32 here causes
        // orientation-dependent errors of several output units.
        double value = 0.0;
        const int degree = reduced_degree(reduced);
        const int first_global = degree * degree;
        const int width = 2 * degree + 1;
        for (int component = 0; component < width; ++component) {
            const int global = first_global + component;
            value += static_cast<double>(rotation[
                (edge * kReducedDim + reduced) * kFullDim + global])
                * static_cast<double>(source_values[global * kChannels + channel]);
        }
        destination[index] = static_cast<float>(value);
    }
}

__global__ void radial_mix_initial_kernel(
    std::int64_t edges,
    const float* compact,
    const float* radial_channel_basis,
    float* local) {
    const std::size_t edge = static_cast<std::size_t>(blockIdx.x);
    if (edge >= static_cast<std::size_t>(edges)) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x);
    __shared__ float source[640];
    const float* source_global = local + edge * kReducedDim * kChannels;
    for (int index = lane; index < kReducedDim * kChannels; index += blockDim.x) {
        source[index] = source_global[index];
    }
    __syncthreads();
    for (int index = lane; index < kReducedDim * kChannels; index += blockDim.x) {
        const int row = index / kChannels;
        const int channel = index % kChannels;
        const int group = reduced_group(row);
        const int degree = reduced_degree(row);
        const int group_size = group == 0 ? 4 : 3;
        const int compact_offset = group == 0 ? 0 : 16;
        const int output_local = group == 0 ? degree : degree - 1;
        float value = 0.0F;
        for (int input_local = 0; input_local < group_size; ++input_local) {
            const int input_row = group == 0
                ? input_local
                : (row < 7 ? 4 + input_local : 7 + input_local);
            const int coefficient =
                compact_offset + input_local * group_size + output_local;
            value += compact[edge * 25 + coefficient]
                * source[input_row * kChannels + channel];
        }
        local[edge * kReducedDim * kChannels + index] =
            value * radial_channel_basis[channel];
    }
}

__global__ void pack_radial_values_kernel(
    std::int64_t edges,
    const float* local,
    float* packed) {
    const std::size_t edge = static_cast<std::size_t>(blockIdx.x);
    if (edge >= static_cast<std::size_t>(edges)) {
        return;
    }
    for (int index = static_cast<int>(threadIdx.x); index < 256;
         index += blockDim.x) {
        packed[edge * 256 + index] =
            local[edge * kReducedDim * kChannels + index];
    }
}

__global__ void pack_radial_m1_inputs_kernel(
    std::int64_t edges,
    const float* local,
    float* packed) {
    const std::size_t edge = static_cast<std::size_t>(blockIdx.x);
    if (edge >= static_cast<std::size_t>(edges)) {
        return;
    }
    for (int channel = static_cast<int>(threadIdx.x); channel < 192;
         channel += blockDim.x) {
        const std::size_t local_offset = edge * kReducedDim * kChannels;
        packed[edge * 192 + channel] =
            local[local_offset + 4 * kChannels + channel];
        packed[(static_cast<std::size_t>(edges) + edge) * 192 + channel] =
            local[local_offset + 7 * kChannels + channel];
    }
}

__global__ void radial_apply_kernel(
    const float* radial_m0,
    const float* radial_m1,
    const float* gate,
    std::int64_t edges,
    int final_layer,
    float* local,
    float* edge_message_output) {
    const std::size_t edge = static_cast<std::size_t>(blockIdx.x);
    if (edge >= static_cast<std::size_t>(edges)) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x);
    const float* m0 = radial_m0 + edge * 256;
    const float* negative = radial_m1 + edge * 384;
    const float* positive = radial_m1
        + (static_cast<std::size_t>(edges) + edge) * 384;
    float* edge_message = edge_message_output + edge * 1024;
    __shared__ float logits[192];
    if (!final_layer) {
        for (int output = lane; output < 192; output += blockDim.x) {
            float value = 0.0F;
            for (int input = 0; input < 64; ++input) {
                value += m0[input] * gate[input * 192 + output];
            }
            logits[output] = value;
        }
    }
    __syncthreads();
    __shared__ float values[640];
    float* destination = local + edge * 640;
    for (int index = lane; index < 640; index += blockDim.x) {
        values[index] = destination[index];
    }
    __syncthreads();
    for (int index = lane; index < 640; index += blockDim.x) {
        const int row = index / 64;
        const int channel = index % 64;
        float value = 0.0F;
        if (row < 4) {
            value = m0[index];
            edge_message[index] = value;
        } else if (row < 7) {
            const int m1_index = (row - 4) * 64 + channel;
            value = negative[m1_index] - positive[192 + m1_index];
            edge_message[256 + m1_index] = value;
        } else {
            const int m1_index = (row - 7) * 64 + channel;
            value = negative[192 + m1_index] + positive[m1_index];
            edge_message[448 + m1_index] = value;
        }
        if (final_layer) {
            values[index] += value;
        } else if (row == 0) {
            values[index] += d_silu(value);
        } else {
            const int degree = row < 4 ? row : row < 7 ? row - 3 : row - 6;
            values[index] += value
                * d_sigmoid(logits[(degree - 1) * 64 + channel]);
        }
    }
    __syncthreads();
    for (int index = lane; index < 640; index += blockDim.x) {
        destination[index] = values[index];
    }
}

__global__ void radial_pre_norm_kernel(
    std::int64_t edges,
    const float* radial_pre,
    const float* radial_norm_scale,
    float* radial_hidden) {
    const std::size_t edge = static_cast<std::size_t>(blockIdx.x);
    if (edge >= static_cast<std::size_t>(edges)) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x);
    const float* source = radial_pre + edge * 256;
    __shared__ float inverse;
    if (lane == 0) {
        float variance = 0.0F;
        for (int channel = 0; channel < 64; ++channel) {
            variance += source[channel] * source[channel];
        }
        inverse = 1.0F / sqrtf(variance / 64.0F + kEpsilon);
    }
    __syncthreads();
    float* destination = radial_hidden + edge * 256;
    for (int channel = lane; channel < 64; channel += blockDim.x) {
        destination[channel] = d_silu(
            source[channel] * inverse * radial_norm_scale[channel]);
    }
}

__global__ void radial_envelope_kernel(
    std::int64_t edges,
    const float* envelopes,
    float* radial) {
    const std::size_t edge = static_cast<std::size_t>(blockIdx.x);
    if (edge >= static_cast<std::size_t>(edges)) {
        return;
    }
    const float envelope = envelopes[edge];
    float* values = radial + edge * 256;
    for (int index = static_cast<int>(threadIdx.x); index < 256;
         index += blockDim.x) {
        values[index] *= envelope;
    }
}

void radial_initial(
    std::int64_t edges,
    const float* radial_basis,
    const DeviceModel& model,
    float* radial_input,
    float* radial,
    const float* envelopes,
    cudaStream_t stream) {
    if (edges <= 0) {
        return;
    }
    launch_row_major_gemm(
        radial_basis, model.radial_layer1, radial_input,
        edges, 64, 16, 25, 64, 256, 0, 0, 0, 1, stream,
        "DPA4 radial layer1 projection launch failed");
    radial_pre_norm_kernel<<<
        static_cast<unsigned int>(edges), 256, 0, stream>>>(
        edges, radial_input, model.radial_norm_scale, radial_input);
    check_cuda(cudaGetLastError(), "DPA4 radial normalization launch failed");
    launch_row_major_gemm(
        radial_input, model.radial_layer2, radial,
        edges, 256, 64, 256, 256, 256, 0, 0, 0, 1, stream,
        "DPA4 radial layer2 projection launch failed");
    radial_envelope_kernel<<<
        static_cast<unsigned int>(edges), 256, 0, stream>>>(
        edges, envelopes, radial);
    check_cuda(cudaGetLastError(), "DPA4 radial envelope launch failed");
}

void environment_rbf(
    std::int64_t edges,
    const float* radial_compact,
    const DeviceModel& model,
    float* radial_input,
    cudaStream_t stream) {
    if (edges <= 0) {
        return;
    }
    launch_row_major_gemm(
        radial_compact, model.env_rbf1, radial_input,
        edges, 32, 16, 25, 32, 256, 0, 0, 0, 1, stream,
        "DPA4 environment radial layer1 projection launch failed");
    const std::int64_t radial_hidden_values = edges * 32;
    environment_silu_kernel<<<
        static_cast<unsigned int>((radial_hidden_values + 255) / 256), 256, 0, stream>>>(
        edges, radial_input, 0, 32);
    check_cuda(
        cudaGetLastError(),
        "DPA4 environment radial activation launch failed");
    launch_row_major_gemm(
        radial_input, model.env_rbf2, radial_input + 32,
        edges, 32, 32, 256, 32, 256, 0, 0, 0, 1, stream,
        "DPA4 environment radial layer2 projection launch failed");
}

void environment_g(
    std::int64_t edges,
    const DeviceModel& model,
    float* environment_input,
    float* radial_bias,
    cudaStream_t stream) {
    if (edges <= 0) {
        return;
    }
    launch_row_major_gemm(
        environment_input, model.env_g1, environment_input + 64,
        edges, 128, 64, 256, 128, 256, 0, 0, 0, 1, stream,
        "DPA4 environment layer1 projection launch failed");
    const std::int64_t hidden_values = edges * 128;
    environment_silu_kernel<<<
        static_cast<unsigned int>((hidden_values + 255) / 256), 256, 0, stream>>>(
        edges, environment_input, 64, 128);
    check_cuda(
        cudaGetLastError(),
        "DPA4 environment activation launch failed");
    launch_row_major_gemm(
        environment_input + 64, model.env_g2, radial_bias,
        edges, 64, 128, 256, 64, 64, 0, 0, 0, 1, stream,
        "DPA4 environment layer2 projection launch failed");
}

__global__ void rotate_edge_kernel(
    const std::int64_t* graph_offsets,
    std::int64_t atoms,
    const float* rotation,
    const float* local,
    float* edge_message) {
    // The rotation is an independent tiny GEMM for every edge.  Cooperating
    // over one edge keeps the reduction in registers while reusing its local
    // input from shared memory; the former one-thread-per-edge loop left most
    // of the SM idle and repeatedly loaded the same 640-element input.
    const std::size_t edge = static_cast<std::size_t>(blockIdx.x);
    if (edge >= static_cast<std::size_t>(graph_offsets[atoms])) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x);
    constexpr float rescale[16] = {
        1.0F, 1.0F, 1.0F, 1.0F,
        1.2909944487358056F, 1.2909944487358056F, 1.2909944487358056F,
        1.2909944487358056F, 1.2909944487358056F,
        1.5275252316519468F, 1.5275252316519468F, 1.5275252316519468F,
        1.5275252316519468F, 1.5275252316519468F, 1.5275252316519468F,
        1.5275252316519468F,
    };
    __shared__ float local_values[640];
    const float* source = local + edge * kReducedDim * kChannels;
    for (int index = lane; index < kReducedDim * kChannels; index += blockDim.x) {
        local_values[index] = source[index];
    }
    __syncthreads();
    for (int flat = lane; flat < kFullDim * kChannels; flat += blockDim.x) {
        const int row = flat / kChannels;
        const int channel = flat % kChannels;
        const int degree = degree_for_row(row);
        const int reduced_count = degree == 0 ? 1 : 3;
        float value = 0.0F;
        for (int component = 0; component < reduced_count; ++component) {
            const int reduced = degree == 0 ? 0 : degree + 3 * component;
            value += rotation[(edge * kReducedDim + reduced) * kFullDim + row]
                * local_values[reduced * kChannels + channel];
        }
        edge_message[edge * kFullDim * kChannels
            + flat] = value * rescale[row];
    }
}

__global__ void qk_kernel(
    const float* pre_focus,
    std::int64_t atoms,
    DeviceBlock block,
    float* query,
    float* key) {
    const std::int64_t node = static_cast<std::int64_t>(blockIdx.x);
    if (node >= atoms) {
        return;
    }
    const float* input = pre_focus + node * kFullDim * kChannels;
    __shared__ float normalized[64];
    if (threadIdx.x == 0) {
        float mean_square = 0.0F;
        for (int channel = 0; channel < 64; ++channel) {
            mean_square += input[channel] * input[channel];
        }
        const float inverse =
            1.0F / sqrtf(mean_square / 64.0F + kEpsilon);
        for (int channel = 0; channel < 64; ++channel) {
            normalized[channel] =
                input[channel] * inverse * block.attn_qk_scale[channel];
        }
    }
    __syncthreads();
    for (int out = static_cast<int>(threadIdx.x); out < 64;
         out += blockDim.x) {
        query[node * 64 + out] =
            d_affine(block.attn_q, 64, 64, normalized, out);
        key[node * 64 + out] =
            d_affine(block.attn_k, 64, 64, normalized, out);
    }
}

__global__ void attention_kernel(
    const std::int64_t* graph_offsets,
    const std::int32_t* graph_atoms,
    const std::int32_t* graph_shifts,
    std::int64_t atoms,
    const float* envelopes,
    const float* radial_bias,
    const float* edge_message,
    const float* pre_focus,
    const float* query,
    const float* key,
    DeviceBlock block,
    float* aggregate) {
    const std::int64_t node =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node >= atoms) {
        return;
    }
    // The aggregate is a float32 model tensor.  Keeping this per-node
    // accumulator in double spills twice as much local memory and invokes the
    // consumer GPU's slow FP64 pipeline for every edge contribution.
    float accumulator[kFullDim * kChannels] = {};
    const std::int64_t begin = graph_offsets[node];
    const std::int64_t end = graph_offsets[node + 1];
    const double null_logit = log(
        static_cast<double>(d_softplus(block.attn_z_bias[0]))
        + static_cast<double>(kEpsilon));
    double max_logit = null_logit;
    for (std::int64_t edge = begin; edge < end; ++edge) {
        const std::int32_t source = graph_atoms[edge];
        const bool exact_self = source == node
            && graph_shifts != nullptr
            && graph_shifts[edge * 3] == 0
            && graph_shifts[edge * 3 + 1] == 0
            && graph_shifts[edge * 3 + 2] == 0;
        const float envelope = envelopes[edge];
        if (exact_self || envelope <= 0.0F) {
            continue;
        }
        double dot = 0.0;
        for (int channel = 0; channel < 64; ++channel) {
            dot += static_cast<double>(query[node * 64 + channel])
                * key[source * 64 + channel];
        }
        double radial = 0.0;
        for (int channel = 0; channel < 64; ++channel) {
            radial += static_cast<double>(radial_bias[edge * 64 + channel])
                * block.attn_logit[channel];
        }
        const double logit = dot / 8.0 + radial
            + 2.0 * log(static_cast<double>(envelope));
        max_logit = fmax(max_logit, logit);
    }
    double denominator = exp(null_logit - max_logit);
    for (std::int64_t edge = begin; edge < end; ++edge) {
        const std::int32_t source = graph_atoms[edge];
        const bool exact_self = source == node
            && graph_shifts != nullptr
            && graph_shifts[edge * 3] == 0
            && graph_shifts[edge * 3 + 1] == 0
            && graph_shifts[edge * 3 + 2] == 0;
        const float envelope = envelopes[edge];
        if (exact_self || envelope <= 0.0F) {
            continue;
        }
        double dot = 0.0;
        for (int channel = 0; channel < 64; ++channel) {
            dot += static_cast<double>(query[node * 64 + channel])
                * key[source * 64 + channel];
        }
        double radial = 0.0;
        for (int channel = 0; channel < 64; ++channel) {
            radial += static_cast<double>(radial_bias[edge * 64 + channel])
                * block.attn_logit[channel];
        }
        denominator += exp(dot / 8.0 + radial
            + 2.0 * log(static_cast<double>(envelope)) - max_logit);
    }
    const double inverse = 1.0 / denominator;
    for (std::int64_t edge = begin; edge < end; ++edge) {
        const std::int32_t source = graph_atoms[edge];
        const bool exact_self = source == node
            && graph_shifts != nullptr
            && graph_shifts[edge * 3] == 0
            && graph_shifts[edge * 3 + 1] == 0
            && graph_shifts[edge * 3 + 2] == 0;
        const float envelope = envelopes[edge];
        if (exact_self || envelope <= 0.0F) {
            continue;
        }
        double dot = 0.0;
        for (int channel = 0; channel < 64; ++channel) {
            dot += static_cast<double>(query[node * 64 + channel])
                * key[source * 64 + channel];
        }
        double radial = 0.0;
        for (int channel = 0; channel < 64; ++channel) {
            radial += static_cast<double>(radial_bias[edge * 64 + channel])
                * block.attn_logit[channel];
        }
        const double alpha = exp(dot / 8.0 + radial
            + 2.0 * log(static_cast<double>(envelope)) - max_logit) * inverse;
        for (int row = 0; row < kFullDim; ++row) {
            for (int channel = 0; channel < 64; ++channel) {
                accumulator[row * 64 + channel] += static_cast<float>(alpha)
                    * edge_message[edge * kFullDim * kChannels
                        + row * kChannels + channel];
            }
        }
    }
    const float* node_input = pre_focus + node * kFullDim * kChannels;
    float gate_input[64];
    float gate_mean_square = 0.0F;
    for (int channel = 0; channel < 64; ++channel) {
        gate_mean_square += node_input[channel] * node_input[channel];
    }
    const float gate_inverse =
        1.0F / sqrtf(gate_mean_square / 64.0F + kEpsilon);
    double gate_logit = 0.0;
    for (int channel = 0; channel < 64; ++channel) {
        gate_input[channel] =
            node_input[channel] * gate_inverse * block.attn_output_gate_scale[channel];
        gate_logit += static_cast<double>(gate_input[channel])
            * block.attn_gate[channel];
    }
    const float gate = d_sigmoid(static_cast<float>(gate_logit));
    for (int index = 0; index < kFullDim * kChannels; ++index) {
        aggregate[node * kFullDim * kChannels + index] =
            accumulator[index] * gate;
    }
}

__global__ void message_grid_kernel(
    const float* aggregate,
    const float* context,
    std::int64_t node_begin,
    std::int64_t tile_nodes,
    DeviceBlock block,
    const float* grid_to,
    const float* grid_from,
    float* scratch_q,
    float* scratch_c,
    float* scratch_product,
    float* output) {
    // One block owns one atom.  The old launch used one thread per atom and
    // serialized the two grid projections and their contractions inside that
    // thread.  These are batched GEMMs in deepmd-kit, so distribute the
    // independent output elements over the block instead.
    const std::int64_t local_node = static_cast<std::int64_t>(blockIdx.x);
    if (local_node >= tile_nodes) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x);
    const std::int64_t node = node_begin + local_node;
    float* query_projection = scratch_q + local_node * kGridScratchStride;
    float* context_projection = scratch_c + local_node * kGridScratchStride;
    float* product = scratch_product + local_node * kGridScratchStride;
    const float* query = aggregate + node * kFullDim * kChannels;
    const float* ctx = context + node * kFullDim * kChannels;

    // First form the coefficient projections once.  The previous loop did
    // this same contraction independently for every grid row.
    for (int flat = lane; flat < kGridCoeff * 64; flat += blockDim.x) {
        const int coefficient = flat / 64;
        const int channel = flat % 64;
        const int row = coefficient / kFrames;
        const int frame = coefficient % kFrames;
        const int degree = degree_for_row(row);
        float q_coeff = 0.0F;
        float c_coeff = 0.0F;
        for (int input_channel = 0; input_channel < 64; ++input_channel) {
            const float weight = block.message_frame_expand[
                degree * 64 * 192 + input_channel * 192
                + frame * 64 + channel];
            q_coeff += query[row * 64 + input_channel] * weight;
            c_coeff += ctx[row * 64 + input_channel] * weight;
        }
        query_projection[flat] = q_coeff;
        context_projection[flat] = c_coeff;
    }
    __syncthreads();

    for (int flat = lane; flat < kGridSize * 64; flat += blockDim.x) {
        const int grid_row = flat / 64;
        const int channel = flat % 64;
        float query_value = 0.0F;
        float context_value = 0.0F;
        for (int coefficient = 0; coefficient < kGridCoeff; ++coefficient) {
            const float projector = grid_to[grid_row * kGridCoeff + coefficient];
            query_value += projector * query_projection[coefficient * 64 + channel];
            context_value += projector * context_projection[coefficient * 64 + channel];
        }
        product[flat] = query_value * context_value;
    }
    __syncthreads();

    __shared__ float scalar_pair[128];
    __shared__ float scalar_out[64];
    __shared__ float scalar_gate[64];
    for (int index = lane; index < 128; index += blockDim.x) {
        scalar_pair[index] = index < 64 ? query[index] : ctx[index - 64];
    }
    __syncthreads();
    for (int index = lane; index < 64; index += blockDim.x) {
        scalar_out[index] =
            d_silu(scalar_pair[index]) * scalar_pair[64 + index];
        float value = 0.0F;
        for (int input = 0; input < 128; ++input) {
            value += scalar_pair[input]
                * block.message_scalar_gate[input * 64 + index];
        }
        scalar_gate[index] = d_sigmoid(value);
    }
    __syncthreads();

    // Cache the grid-to-coefficient contraction once per packed coefficient
    // and input channel instead of recomputing it for every output channel.
    for (int flat = lane; flat < kGridCoeff * 64; flat += blockDim.x) {
        const int coefficient = flat / 64;
        const int input = flat % 64;
        float value = 0.0F;
        for (int grid_row = 0; grid_row < kGridSize; ++grid_row) {
            value += grid_from[coefficient * kGridSize + grid_row]
                * product[grid_row * 64 + input];
        }
        query_projection[flat] = value;
    }
    __syncthreads();

    float* destination = output + node * kFullDim * kChannels;
    for (int flat = lane; flat < kFullDim * kChannels; flat += blockDim.x) {
        const int row = flat / kChannels;
        const int out = flat % kChannels;
        const int degree = degree_for_row(row);
        float value = 0.0F;
        for (int frame = 0; frame < kFrames; ++frame) {
            for (int input = 0; input < 64; ++input) {
                const int packed = row * kFrames + frame;
                value += query_projection[packed * 64 + input]
                    * scalar_gate[input]
                    * block.message_frame_contract[
                        degree * 192 * 64
                        + (frame * 64 + input) * 64 + out];
            }
        }
        if (row == 0) {
            for (int input = 0; input < 64; ++input) {
                value += scalar_out[input]
                    * block.message_frame_contract[input * 64 + out];
            }
        }
        destination[flat] = value * block.message_residual_scale[out];
    }
}

__global__ void post_state_kernel(
    const float* state,
    const float* aggregate,
    const float* message,
    std::int64_t atoms,
    DeviceBlock block,
    float* next_state) {
    const std::int64_t node = static_cast<std::int64_t>(blockIdx.x);
    if (node >= atoms) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x);
    __shared__ float post[1024];
    const float* input = aggregate + node * 1024;
    const float* message_input = message + node * 1024;
    for (int flat = lane; flat < 1024; flat += blockDim.x) {
        const int packed_row = flat / 64;
        const int out = flat % 64;
        const int degree = degree_for_row(packed_row);
        const std::size_t weight_offset =
            static_cast<std::size_t>(degree) * 64U * 64U;
        float value = 0.0F;
        for (int input_channel = 0; input_channel < 64; ++input_channel) {
            const float weight = block.post_focus[
                weight_offset + input_channel * 64 + out];
            value += (
                input[packed_row * 64 + input_channel]
                + message_input[packed_row * 64 + input_channel]) * weight;
        }
        post[flat] = value;
    }
    __syncthreads();

    float* destination = next_state + node * 1024;
    const float* original = state + node * 1024;
    __shared__ double norm_mean;
    __shared__ float norm_inverse;
    if (lane == 0) {
        if (block.post_norm_enabled) {
            double mean = 0.0;
            for (int channel = 0; channel < 64; ++channel) {
                mean += post[channel];
            }
            mean /= 64.0;
            double variance = 0.0;
            for (int row = 0; row < 16; ++row) {
                for (int channel = 0; channel < 64; ++channel) {
                    const double value = row == 0
                        ? static_cast<double>(post[row * 64 + channel]) - mean
                        : post[row * 64 + channel];
                    variance += value * value * block.post_norm_balance[row];
                }
            }
            const float inverse = static_cast<float>(
                1.0 / sqrt(variance + static_cast<double>(kNormEpsilon)));
            norm_mean = mean;
            norm_inverse = inverse;
        } else {
            norm_mean = 0.0;
            norm_inverse = 1.0F;
        }
    }
    __syncthreads();
    if (block.post_norm_enabled) {
        for (int flat = lane; flat < 1024; flat += blockDim.x) {
            const int row = flat / 64;
            const int channel = flat % 64;
            const int degree = degree_for_row(row);
            const float value = row == 0
                ? static_cast<float>(
                    static_cast<double>(post[flat]) - norm_mean)
                : post[flat];
            destination[flat] =
                original[flat]
                + value * norm_inverse * block.post_norm_scale[
                    degree * 64 + channel]
                + (row == 0 ? block.post_norm_bias[channel] : 0.0F);
        }
    } else {
        for (int flat = lane; flat < 1024; flat += blockDim.x) {
            destination[flat] = original[flat] + post[flat];
        }
    }
}

__global__ void ffn_normalize_kernel(
    const float* input,
    std::int64_t atoms,
    DeviceBlock block,
    float* output) {
    const std::int64_t node = static_cast<std::int64_t>(blockIdx.x);
    if (node >= atoms) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x);
    const float* source = input + node * 1024;
    __shared__ double mean;
    __shared__ float inverse;
    if (lane == 0) {
        if (block.ffn_norm_enabled) {
            double mean_value = 0.0;
            for (int channel = 0; channel < 64; ++channel) {
                mean_value += source[channel];
            }
            mean_value /= 64.0;
            double variance = 0.0;
            for (int row = 0; row < 16; ++row) {
                for (int channel = 0; channel < 64; ++channel) {
                    const double value = row == 0
                        ? static_cast<double>(source[row * 64 + channel])
                            - mean_value
                        : source[row * 64 + channel];
                    variance += value * value * block.ffn_norm_balance[row];
                }
            }
            mean = mean_value;
            inverse = static_cast<float>(
                1.0 / sqrt(variance + static_cast<double>(kNormEpsilon)));
        } else {
            mean = 0.0;
            inverse = 1.0F;
        }
    }
    __syncthreads();
    float* destination = output + node * 1024;
    for (int flat = lane; flat < 1024; flat += blockDim.x) {
        const int row = flat / 64;
        const int channel = flat % 64;
        if (block.ffn_norm_enabled) {
            const float value = row == 0
                ? static_cast<float>(static_cast<double>(source[flat]) - mean)
                : source[flat];
            destination[flat] =
                value * inverse * block.ffn_norm_scale[
                    degree_for_row(row) * 64 + channel]
                + (row == 0 ? block.ffn_norm_bias[channel] : 0.0F);
        } else {
            destination[flat] = source[flat];
        }
    }
}

__global__ void linear2_residual_kernel(
    const float* old_state,
    const float* activation,
    std::int64_t atoms,
    DeviceBlock block,
    float* new_state) {
    const std::int64_t node = static_cast<std::int64_t>(blockIdx.x);
    if (node >= atoms) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x);
    const float* old_values = old_state + node * 1024;
    const float* act = activation + node * kFullDim * 576;
    float* destination = new_state + node * 1024;
    for (int flat = lane; flat < 1024; flat += blockDim.x) {
        const int packed_row = flat / 64;
        const int out = flat % 64;
        const int degree = degree_for_row(packed_row);
        const float* weights = block.ffn_linear2
            + static_cast<std::size_t>(degree) * 576U * 64U;
        const float value = d_affine(
            weights, 576, 64, act + packed_row * 576, out);
        destination[flat] = old_values[flat] + value;
    }
}

__global__ void pack_output_grid_input_kernel(
    const float* hidden,
    std::int64_t node_begin,
    std::int64_t tile_nodes,
    float* packed) {
    const std::int64_t local_node = static_cast<std::int64_t>(blockIdx.x);
    if (local_node >= tile_nodes) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x);
    const float* input = hidden
        + (node_begin + local_node) * kFullDim * 1152;
    float* destination = packed
        + local_node * kGridCoeff * 384;
    for (int flat = lane; flat < kGridCoeff * 384; flat += blockDim.x) {
        const int coefficient = flat / 384;
        const int input_channel = flat % 384;
        const int row = coefficient / kFrames;
        const int frame = coefficient % kFrames;
        const int source_channel = input_channel < 192
            ? frame * 192 + input_channel
            : 576 + frame * 192 + input_channel - 192;
        destination[flat] = input[row * 1152 + source_channel];
    }
}

__global__ void pack_block_grid_input_kernel(
    const float* hidden,
    std::int64_t node_begin,
    std::int64_t tile_nodes,
    float* packed) {
    const std::int64_t local_node = static_cast<std::int64_t>(blockIdx.x);
    if (local_node >= tile_nodes) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x);
    const float* input = hidden
        + (node_begin + local_node) * kFullDim * 1152;
    float* destination = packed;
    constexpr int channel_stride = kGridCoeff * 192;
    for (int flat = lane; flat < 2 * channel_stride; flat += blockDim.x) {
        const bool right = flat >= channel_stride;
        const int local = right ? flat - channel_stride : flat;
        const int coefficient = local / 192;
        const int channel = local % 192;
        const int row = coefficient / kFrames;
        const int frame = coefficient % kFrames;
        float* batch_destination = right
            ? destination + tile_nodes * channel_stride
                + local_node * channel_stride
            : destination + local_node * channel_stride;
        batch_destination[local] = input[
            row * 1152 + (right ? 576 : 0) + frame * 192 + channel];
    }
}

__global__ void grid_product_kernel(
    const float* left,
    const float* right,
    std::size_t count,
    float* product) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    if (index < count) {
        product[index] = left[index] * right[index];
    }
}

__global__ void output_grid_post_kernel(
    const float* packed,
    std::int64_t node_begin,
    std::int64_t tile_nodes,
    const float* scalar_weight,
    float* activation) {
    const std::int64_t local_node = static_cast<std::int64_t>(blockIdx.x);
    if (local_node >= tile_nodes) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x);
    const float* scalar_pair = packed
        + local_node * kGridCoeff * 384;
    float* destination = activation
        + (node_begin + local_node) * kGridCoeff * 192;
    for (int output = lane; output < 192; output += blockDim.x) {
        const float gate_input = scalar_pair[output];
        const float value_input = scalar_pair[192 + output];
        const float scalar_output = d_silu(gate_input) * value_input;
        float gate_logit = 0.0F;
        for (int input = 0; input < 384; ++input) {
            gate_logit += scalar_pair[input]
                * scalar_weight[input * 192 + output];
        }
        const float scalar_gate = d_sigmoid(gate_logit);
        for (int coefficient = 0; coefficient < kGridCoeff; ++coefficient) {
            destination[coefficient * 192 + output] *= scalar_gate;
        }
        destination[output] += scalar_output;
    }
}

__global__ void block_grid_post_kernel(
    const float* packed,
    std::int64_t node_begin,
    std::int64_t tile_nodes,
    const float* scalar_weight,
    float* activation) {
    const std::int64_t local_node = static_cast<std::int64_t>(blockIdx.x);
    if (local_node >= tile_nodes) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x);
    constexpr int half_stride = kGridCoeff * 192;
    const float* scalar_pair = packed + local_node * half_stride;
    const float* value_pair = packed
        + tile_nodes * half_stride + local_node * half_stride;
    float* destination = activation
        + (node_begin + local_node) * kGridCoeff * 192;
    for (int output = lane; output < 192; output += blockDim.x) {
        const float gate_input = scalar_pair[output];
        const float value_input = value_pair[output];
        const float scalar_output = d_silu(gate_input) * value_input;
        float gate_logit = 0.0F;
        for (int input = 0; input < 2 * 192; ++input) {
            const float scalar = input < 192
                ? scalar_pair[input]
                : value_pair[input - 192];
            gate_logit += scalar * scalar_weight[input * 192 + output];
        }
        const float scalar_gate = d_sigmoid(gate_logit);
        for (int coefficient = 0; coefficient < kGridCoeff; ++coefficient) {
            destination[coefficient * 192 + output] *= scalar_gate;
        }
        destination[output] += scalar_output;
    }
}

__global__ void output_linear2_kernel(
    const float* state,
    const float* activation,
    std::int64_t atoms,
    const float* weights,
    double* output) {
    const std::int64_t node = static_cast<std::int64_t>(blockIdx.x);
    if (node >= atoms) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x);
    for (int out = lane; out < 64; out += blockDim.x) {
        float value = 0.0F;
        for (int input = 0; input < 576; ++input) {
            value += activation[node * 16 * 576 + input]
                * weights[input * 64 + out];
        }
        output[node * 64 + out] =
            static_cast<double>(state[node * 1024 + out])
            + static_cast<double>(value);
    }
}

void launch_check(cudaError_t status, const char* operation) {
    check_cuda(status, operation);
    check_cuda(cudaGetLastError(), operation);
}

void so3_linear(
    const float* input,
    std::int64_t atoms,
    int input_channels,
    int output_channels,
    const float* weights,
    float* output,
    cudaStream_t stream) {
    if (atoms <= 0) {
        return;
    }
    constexpr int row_starts[4] = {0, 1, 4, 9};
    constexpr int row_counts[4] = {1, 3, 5, 7};
    const long long input_node_stride =
        static_cast<long long>(kFullDim) * input_channels;
    const long long output_node_stride =
        static_cast<long long>(kFullDim) * output_channels;
    const long long weight_degree_stride =
        static_cast<long long>(input_channels) * output_channels;
    for (int degree = 0; degree < 4; ++degree) {
        const int row_count = row_counts[degree];
        launch_row_major_gemm(
            input + row_starts[degree] * input_channels,
            weights + degree * weight_degree_stride,
            output + row_starts[degree] * output_channels,
            atoms, output_channels, input_channels,
            input_node_stride, output_channels, output_node_stride,
            input_channels, 0, output_channels, row_count,
            stream,
            "DPA4 SO(3) linear projection launch failed");
    }
}

void output_grid(
    const float* hidden,
    std::int64_t node_begin,
    std::int64_t tile_nodes,
    const float* grid_to,
    const float* grid_from,
    const float* left_weight,
    const float* right_weight,
    const float* output_weight,
    const float* scalar_weight,
    float* packed,
    float* scratch0,
    float* scratch1,
    float* scratch2,
    float* activation,
    cudaStream_t stream) {
    if (tile_nodes <= 0) {
        return;
    }
    pack_output_grid_input_kernel<<<
        static_cast<unsigned int>(tile_nodes), 128, 0, stream>>>(
        hidden, node_begin, tile_nodes, packed);
    launch_check(cudaGetLastError(), "DPA4 output grid pack launch failed");

    const long long grid_stride = 152LL * 384LL;
    const long long coefficient_stride = 48LL * 384LL;

    // The node batches are contiguous in memory.  Flatten the three
    // (input-batched) projections into one GEMM so a weight tile is reused
    // across nodes instead of being loaded once for every blockIdx.z batch.
    const std::int64_t flattened_rows = tile_nodes * 48;
    launch_row_major_gemm(
        packed, left_weight, scratch0,
        flattened_rows, 384, 384, 384, 384, 384,
        0, 0, 0, 1, stream,
        "DPA4 output grid left projection launch failed");
    launch_row_major_gemm(
        packed, right_weight, scratch1,
        flattened_rows, 384, 384, 384, 384, 384,
        0, 0, 0, 1, stream,
        "DPA4 output grid right projection launch failed");
    launch_row_major_gemm(
        grid_to, scratch0, scratch2,
        152, 384, 48, 48, 384, 384,
        0, coefficient_stride, grid_stride, tile_nodes, stream,
        "DPA4 output grid left-to-grid projection launch failed");
    launch_row_major_gemm(
        grid_to, scratch1, scratch0,
        152, 384, 48, 48, 384, 384,
        0, coefficient_stride, grid_stride, tile_nodes, stream,
        "DPA4 output grid right-to-grid projection launch failed");

    const std::size_t product_count = static_cast<std::size_t>(tile_nodes)
        * kGridSize * 384U;
    grid_product_kernel<<<
        static_cast<unsigned int>((product_count + 255U) / 256U), 256, 0, stream>>>(
        scratch2, scratch0, product_count, scratch1);
    launch_check(cudaGetLastError(), "DPA4 output grid product launch failed");

    launch_row_major_gemm(
        grid_from, scratch1, scratch2,
        48, 384, 152, 152, 384, 384,
        0, grid_stride, coefficient_stride, tile_nodes, stream,
        "DPA4 output grid grid-to-coefficient projection launch failed");
    launch_row_major_gemm(
        scratch2, output_weight,
        activation + static_cast<std::size_t>(node_begin) * 48U * 192U,
        flattened_rows, 192, 384, 384, 192, 192,
        0, 0, 0, 1, stream,
        "DPA4 output grid output projection launch failed");
    output_grid_post_kernel<<<
        static_cast<unsigned int>(tile_nodes), 128, 0, stream>>>(
        packed, node_begin, tile_nodes, scalar_weight, activation);
    launch_check(cudaGetLastError(), "DPA4 output grid gate launch failed");
}

void block_grid(
    const float* hidden,
    std::int64_t node_begin,
    std::int64_t tile_nodes,
    const float* grid_to,
    const float* grid_from,
    const float* left_weight,
    const float* right_weight,
    const float* output_weight,
    const float* scalar_weight,
    float* packed,
    float* scratch0,
    float* scratch1,
    float* scratch2,
    float* activation,
    cudaStream_t stream) {
    if (tile_nodes <= 0) {
        return;
    }
    pack_block_grid_input_kernel<<<
        static_cast<unsigned int>(tile_nodes), 128, 0, stream>>>(
        hidden, node_begin, tile_nodes, packed);
    launch_check(cudaGetLastError(), "DPA4 block grid pack launch failed");

    const long long half_stride = 48LL * 192LL;
    const long long grid_stride = 152LL * 192LL;
    const long long coefficient_stride = 48LL * 192LL;
    const std::int64_t flattened_rows = tile_nodes * 48;

    launch_row_major_gemm(
        packed, left_weight, scratch0,
        flattened_rows, 192, 192, 192, 192, 192,
        0, 0, 0, 1, stream,
        "DPA4 block grid left projection launch failed");
    launch_row_major_gemm(
        packed + tile_nodes * half_stride, right_weight, scratch1,
        flattened_rows, 192, 192, 192, 192, 192,
        0, 0, 0, 1, stream,
        "DPA4 block grid right projection launch failed");
    launch_row_major_gemm(
        grid_to, scratch0, scratch2,
        152, 192, 48, 48, 192, 192,
        0, coefficient_stride, grid_stride, tile_nodes, stream,
        "DPA4 block grid left-to-grid projection launch failed");
    launch_row_major_gemm(
        grid_to, scratch1, scratch0,
        152, 192, 48, 48, 192, 192,
        0, coefficient_stride, grid_stride, tile_nodes, stream,
        "DPA4 block grid right-to-grid projection launch failed");

    const std::size_t product_count = static_cast<std::size_t>(tile_nodes)
        * kGridSize * 192U;
    grid_product_kernel<<<
        static_cast<unsigned int>((product_count + 255U) / 256U), 256, 0, stream>>>(
        scratch2, scratch0, product_count, scratch1);
    launch_check(cudaGetLastError(), "DPA4 block grid product launch failed");

    launch_row_major_gemm(
        grid_from, scratch1, scratch2,
        48, 192, 152, 152, 192, 192,
        0, grid_stride, coefficient_stride, tile_nodes, stream,
        "DPA4 block grid grid-to-coefficient projection launch failed");
    launch_row_major_gemm(
        scratch2, output_weight,
        activation + static_cast<std::size_t>(node_begin) * 48U * 192U,
        flattened_rows, 192, 192, 192, 192, 192,
        0, 0, 0, 1, stream,
        "DPA4 block grid output projection launch failed");
    block_grid_post_kernel<<<
        static_cast<unsigned int>(tile_nodes), 128, 0, stream>>>(
        packed, node_begin, tile_nodes, scalar_weight, activation);
    launch_check(cudaGetLastError(), "DPA4 block grid gate launch failed");
}

void radial_so2(
    std::int64_t edges,
    const float* radial,
    const DeviceBlock& block,
    float* radial_compact,
    float* radial_input,
    float* radial_sign,
    float* radial_projection,
    float* radial_m1_output,
    float* local,
    float* edge_message,
    cudaStream_t stream) {
    if (edges <= 0) {
        return;
    }

    // The radial mixer is shared by every edge.  Flattening the edge
    // dimension into the row-major projection below.
    launch_row_major_gemm(
        radial, block.radial_mixer, radial_compact,
        edges, 25, 256, 256, 25, 25, 0, 0, 0, 1, stream,
        "DPA4 radial mixer projection launch failed");
    radial_mix_initial_kernel<<<
        static_cast<unsigned int>(edges), 256, 0, stream>>>(
        edges, radial_compact, block.radial_channel_basis, local);
    launch_check(cudaGetLastError(), "DPA4 initial radial mix launch failed");

    for (int layer = 0; layer < 4; ++layer) {
        pack_radial_values_kernel<<<
            static_cast<unsigned int>(edges), 256, 0, stream>>>(
            edges, local, radial_input);
        launch_check(cudaGetLastError(), "DPA4 radial m0 pack launch failed");
        launch_row_major_gemm(
            radial_input, block.so2_m0[layer], radial_projection,
            edges, 256, 256, 256, 256, 256, 0, 0, 0, 1, stream,
            "DPA4 radial m0 projection launch failed");

        pack_radial_m1_inputs_kernel<<<
            static_cast<unsigned int>(edges), 256, 0, stream>>>(
            edges, local, radial_sign);
        launch_check(cudaGetLastError(), "DPA4 radial m1 pack launch failed");
        launch_row_major_gemm(
            radial_sign, block.so2_m1[layer], radial_m1_output,
            2 * edges, 384, 192, 192, 384, 384, 0, 0, 0, 1, stream,
            "DPA4 radial m1 projection launch failed");
        radial_apply_kernel<<<
            static_cast<unsigned int>(edges), 256, 0, stream>>>(
            radial_projection, radial_m1_output,
            layer < 3 ? block.so2_gate[layer] : nullptr,
            edges, layer == 3 ? 1 : 0, local, edge_message);
        launch_check(cudaGetLastError(), "DPA4 radial activation launch failed");
    }
}

} // namespace

DeviceDpa4Model::DeviceDpa4Model(
    CudaExecutionContext& context,
    py::dict payload) {
    const py::dict model_payload = payload.contains("model")
        ? py::cast<py::dict>(payload["model"]) : payload;
    auto model = std::make_unique<Model>();
    model->rcut = py::cast<double>(required(model_payload, "rcut"));
    model->ntypes = py::cast<int>(required(model_payload, "ntypes"));
    const int channels = py::cast<int>(required(model_payload, "channels"));
    const int n_radial = py::cast<int>(required(model_payload, "n_radial"));
    model->feature_count = py::cast<int>(required(payload, "feature_count"));
    if (!std::isfinite(model->rcut) || model->rcut <= 0.0
        || model->ntypes <= 0 || channels != kChannels
        || n_radial != 16 || model->feature_count != kChannels) {
        throw std::invalid_argument(
            "unsupported DPA4 CUDA configuration; expected the default "
            "lmax=3, channels=64 network");
    }
    if (!payload.contains("type_numbers")) {
        throw std::invalid_argument("DPA4 CUDA payload is missing type_numbers");
    }
    const auto type_numbers =
        payload_array<std::int32_t>(payload["type_numbers"], "type_numbers");
    if (type_numbers.size() != static_cast<std::size_t>(model->ntypes)) {
        throw std::invalid_argument("DPA4 CUDA type_numbers has unexpected size");
    }
    model->type_numbers = type_numbers;
    std::vector<int> seen(119, 0);
    for (std::int32_t number : type_numbers) {
        if (number <= 0 || number >= static_cast<std::int32_t>(seen.size())
            || seen[static_cast<std::size_t>(number)] != 0) {
            throw std::invalid_argument(
                "DPA4 CUDA type_numbers contains an invalid or duplicate atomic number");
        }
        seen[static_cast<std::size_t>(number)] = 1;
    }

    const std::size_t type_rows = static_cast<std::size_t>(model->ntypes + 1);
    model->wigner_l2 = read_exact<float>(
        model_payload, "wigner_l2_tensor", 25U * 256U);
    model->wigner_l3 = read_exact<float>(
        model_payload, "wigner_l3_coefficients", 49U * 84U);
    model->wigner_l3_exponents = read_exact<std::int64_t>(
        model_payload, "wigner_l3_exponents", 84U * 4U);
    const auto gie_rows = read_exact<std::int64_t>(
        model_payload, "gie_row_index", 15U);
    const auto gie_m0 = read_exact<std::int64_t>(
        model_payload, "gie_m0_index", 15U);
    const auto gie_radial = read_exact<std::int64_t>(
        model_payload, "gie_radial_index", 15U);

    std::array<std::vector<float>, kTopWeightCount> host_top;
    host_top[kTypeEmbedding] = read_exact<float>(
        model_payload, "type_embedding", type_rows * 64U);
    host_top[kEnvRbf1] = read_exact<float>(model_payload, "env_rbf_layer1", 16U * 32U);
    host_top[kEnvRbf2] = read_exact<float>(model_payload, "env_rbf_layer2", 32U * 32U);
    host_top[kEnvTypeEmbedding] = read_exact<float>(
        model_payload, "env_type_embedding", type_rows * 16U);
    host_top[kEnvG1] = read_exact<float>(model_payload, "env_g_layer1", 64U * 128U);
    host_top[kEnvG2] = read_exact<float>(model_payload, "env_g_layer2", 128U * 64U);
    host_top[kEnvOutput] = read_exact<float>(
        model_payload, "env_output_projection", 512U * 128U);
    host_top[kFilmScaleNorm] = read_exact<float>(
        model_payload, "film_scale_norm", 64U);
    host_top[kFilmShiftNorm] = read_exact<float>(
        model_payload, "film_shift_norm", 64U);
    host_top[kRadialFreqs] = read_exact<float>(
        model_payload, "radial_freqs", 16U);
    host_top[kRadialLayer1] = read_exact<float>(
        model_payload, "radial_layer1", 16U * 64U);
    host_top[kRadialNormScale] = read_exact<float>(
        model_payload, "radial_norm_scale", 64U);
    host_top[kRadialLayer2] = read_exact<float>(
        model_payload, "radial_layer2", 64U * 256U);
    host_top[kWignerL2] = model->wigner_l2;
    host_top[kWignerL3] = model->wigner_l3;
    host_top[kWignerL3Exponents] = {};
    host_top[kGieRows] = {};
    host_top[kGieM0] = {};
    host_top[kGieRadial] = {};
    host_top[kGridTo] = read_exact<float>(
        model_payload, "grid_to", static_cast<std::size_t>(kGridSize) * kGridCoeff);
    host_top[kGridFrom] = read_exact<float>(
        model_payload, "grid_from", static_cast<std::size_t>(kGridCoeff) * kGridSize);
    host_top[kOutputLinear1] = read_exact<float>(
        model_payload, "output_linear1", 4U * 64U * 1152U);
    host_top[kOutputLinear2] = read_exact<float>(
        model_payload, "output_linear2", 4U * 576U * 64U);
    host_top[kOutputScalarGate] = read_exact<float>(
        model_payload, "output_scalar_gate", 384U * 192U);
    host_top[kOutputGridLeft] = read_exact<float>(
        model_payload, "output_grid_left", 384U * 384U);
    host_top[kOutputGridRight] = read_exact<float>(
        model_payload, "output_grid_right", 384U * 384U);
    host_top[kOutputGridOut] = read_exact<float>(
        model_payload, "output_grid_out", 384U * 192U);
    model->top[kWignerL3Exponents] =
        upload_array(context, model->wigner_l3_exponents, "could not upload DPA4 Wigner exponents");
    model->top[kGieRows] =
        upload_array(context, gie_rows, "could not upload DPA4 GIE rows");
    model->top[kGieM0] =
        upload_array(context, gie_m0, "could not upload DPA4 GIE m0 indices");
    model->top[kGieRadial] =
        upload_array(context, gie_radial, "could not upload DPA4 GIE radial indices");
    for (int index = 0; index < kTopWeightCount; ++index) {
        if (index == kWignerL3Exponents || index == kGieRows
            || index == kGieM0 || index == kGieRadial) {
            continue;
        }
        model->top[index] = upload_array(
            context, host_top[index], "could not upload DPA4 model weight");
    }

    if (!model_payload.contains("blocks")) {
        throw std::invalid_argument("DPA4 CUDA payload is missing blocks");
    }
    const py::list blocks = py::cast<py::list>(model_payload["blocks"]);
    if (blocks.size() != 3) {
        throw std::invalid_argument("DPA4 CUDA payload requires three blocks");
    }
    for (std::size_t block_index = 0; block_index < 3; ++block_index) {
        const py::dict source = py::cast<py::dict>(blocks[block_index]);
        HostBlock host;
        host.pre_norm_enabled = read_bool(source, "pre_norm_enabled", false);
        host.post_norm_enabled = read_bool(source, "post_norm_enabled", true);
        host.ffn_norm_enabled = read_bool(source, "ffn_norm_enabled", true);
        const auto read_block = [&](int slot, const char* name, std::size_t size) {
            host.values[slot] = read_exact<float>(source, name, size);
        };
        const auto read_norm = [&](int scale_slot, int bias_slot, int balance_slot,
                                   const char* scale, const char* bias,
                                   const char* balance, bool enabled) {
            if (enabled) {
                read_block(scale_slot, scale, 4U * 64U);
                read_block(bias_slot, bias, 64U);
                read_block(balance_slot, balance, 16U);
            } else {
                host.values[scale_slot] = {};
                host.values[bias_slot] = {};
                host.values[balance_slot] = {};
            }
        };
        read_norm(kPreNormScale, kPreNormBias, kPreNormBalance,
                  "pre_norm_scale", "pre_norm_bias", "pre_norm_balance",
                  host.pre_norm_enabled);
        read_norm(kPostNormScale, kPostNormBias, kPostNormBalance,
                  "post_norm_scale", "post_norm_bias", "post_norm_balance",
                  host.post_norm_enabled);
        read_norm(kFfnNormScale, kFfnNormBias, kFfnNormBalance,
                  "ffn_norm_scale", "ffn_norm_bias", "ffn_norm_balance",
                  host.ffn_norm_enabled);
        read_block(kPreFocus, "pre_focus_weight", 4U * 64U * 64U);
        read_block(kPostFocus, "post_focus_weight", 4U * 64U * 64U);
        read_block(kRadialMixer, "radial_mixer_weight", 256U * 25U);
        read_block(kRadialChannelBasis, "radial_channel_basis", 64U);
        const py::list so2_m0 = py::cast<py::list>(required(source, "so2_weight_m0"));
        const py::list so2_m1 = py::cast<py::list>(required(source, "so2_weight_m1"));
        const py::list so2_gate = py::cast<py::list>(required(source, "so2_gate_weight"));
        if (so2_m0.size() != 4 || so2_m1.size() != 4 || so2_gate.size() != 3) {
            throw std::invalid_argument("DPA4 CUDA SO(2) weight lists have invalid lengths");
        }
        for (int index = 0; index < 4; ++index) {
            host.values[kSo2M0_0 + index] = payload_array<float>(
                so2_m0[index], "so2_weight_m0");
            host.values[kSo2M1_0 + index] = payload_array<float>(
                so2_m1[index], "so2_weight_m1");
            if (host.values[kSo2M0_0 + index].size() != 256U * 256U
                || host.values[kSo2M1_0 + index].size() != 192U * 384U) {
                throw std::invalid_argument("DPA4 CUDA SO(2) weight has unexpected size");
            }
        }
        for (int index = 0; index < 3; ++index) {
            host.values[kSo2Gate_0 + index] = payload_array<float>(
                so2_gate[index], "so2_gate_weight");
            if (host.values[kSo2Gate_0 + index].size() != 64U * 192U) {
                throw std::invalid_argument("DPA4 CUDA SO(2) gate weight has unexpected size");
            }
        }
        read_block(kAttnQkScale, "attn_qk_scale", 64U);
        read_block(kAttnQ, "attn_q_weight", 64U * 64U);
        read_block(kAttnK, "attn_k_weight", 64U * 64U);
        read_block(kAttnOutputGateScale, "attn_output_gate_scale", 64U);
        read_block(kAttnLogit, "attn_logit_weight", 64U);
        read_block(kAttnZBias, "attn_z_bias_raw", 1U);
        read_block(kAttnGate, "attn_gate_weight", 64U);
        read_block(kMessageScalarGate, "message_scalar_gate", 128U * 64U);
        read_block(kMessageFrameExpand, "message_frame_expand", 4U * 64U * 192U);
        read_block(kMessageFrameContract, "message_frame_contract", 4U * 192U * 64U);
        read_block(kMessageResidualScale, "message_residual_scale", 64U);
        read_block(kFfnLinear1, "ffn_linear1", 4U * 64U * 1152U);
        read_block(kFfnLinear2, "ffn_linear2", 4U * 576U * 64U);
        read_block(kFfnScalarGate, "ffn_scalar_gate", 384U * 192U);
        read_block(kFfnGridLeft, "ffn_grid_left", 192U * 192U);
        read_block(kFfnGridRight, "ffn_grid_right", 192U * 192U);
        read_block(kFfnGridRouter, "ffn_grid_router", 384U);
        read_block(kFfnGridOut, "ffn_grid_out", 192U * 192U);
        for (int slot = 0; slot < kBlockWeightCount; ++slot) {
            model->blocks[block_index][slot] = upload_array(
                context, host.values[slot], "could not upload DPA4 block weight");
        }
        DeviceBlock& view = model->block_views[block_index];
        view.pre_norm_enabled = host.pre_norm_enabled;
        view.post_norm_enabled = host.post_norm_enabled;
        view.ffn_norm_enabled = host.ffn_norm_enabled;
        view.pre_norm_scale = device_data<float>(model->blocks[block_index][kPreNormScale]);
        view.pre_norm_bias = device_data<float>(model->blocks[block_index][kPreNormBias]);
        view.pre_norm_balance = device_data<float>(model->blocks[block_index][kPreNormBalance]);
        view.post_norm_scale = device_data<float>(model->blocks[block_index][kPostNormScale]);
        view.post_norm_bias = device_data<float>(model->blocks[block_index][kPostNormBias]);
        view.post_norm_balance = device_data<float>(model->blocks[block_index][kPostNormBalance]);
        view.ffn_norm_scale = device_data<float>(model->blocks[block_index][kFfnNormScale]);
        view.ffn_norm_bias = device_data<float>(model->blocks[block_index][kFfnNormBias]);
        view.ffn_norm_balance = device_data<float>(model->blocks[block_index][kFfnNormBalance]);
        view.pre_focus = device_data<float>(model->blocks[block_index][kPreFocus]);
        view.post_focus = device_data<float>(model->blocks[block_index][kPostFocus]);
        view.radial_mixer = device_data<float>(model->blocks[block_index][kRadialMixer]);
        view.radial_channel_basis = device_data<float>(
            model->blocks[block_index][kRadialChannelBasis]);
        for (int index = 0; index < 4; ++index) {
            view.so2_m0[index] = device_data<float>(
                model->blocks[block_index][kSo2M0_0 + index]);
            view.so2_m1[index] = device_data<float>(
                model->blocks[block_index][kSo2M1_0 + index]);
        }
        for (int index = 0; index < 3; ++index) {
            view.so2_gate[index] = device_data<float>(
                model->blocks[block_index][kSo2Gate_0 + index]);
        }
        view.attn_qk_scale = device_data<float>(
            model->blocks[block_index][kAttnQkScale]);
        view.attn_q = device_data<float>(model->blocks[block_index][kAttnQ]);
        view.attn_k = device_data<float>(model->blocks[block_index][kAttnK]);
        view.attn_output_gate_scale = device_data<float>(
            model->blocks[block_index][kAttnOutputGateScale]);
        view.attn_logit = device_data<float>(
            model->blocks[block_index][kAttnLogit]);
        view.attn_z_bias = device_data<float>(
            model->blocks[block_index][kAttnZBias]);
        view.attn_gate = device_data<float>(
            model->blocks[block_index][kAttnGate]);
        view.message_scalar_gate = device_data<float>(
            model->blocks[block_index][kMessageScalarGate]);
        view.message_frame_expand = device_data<float>(
            model->blocks[block_index][kMessageFrameExpand]);
        view.message_frame_contract = device_data<float>(
            model->blocks[block_index][kMessageFrameContract]);
        view.message_residual_scale = device_data<float>(
            model->blocks[block_index][kMessageResidualScale]);
        view.ffn_linear1 = device_data<float>(
            model->blocks[block_index][kFfnLinear1]);
        view.ffn_linear2 = device_data<float>(
            model->blocks[block_index][kFfnLinear2]);
        view.ffn_scalar_gate = device_data<float>(
            model->blocks[block_index][kFfnScalarGate]);
        view.ffn_grid_left = device_data<float>(
            model->blocks[block_index][kFfnGridLeft]);
        view.ffn_grid_right = device_data<float>(
            model->blocks[block_index][kFfnGridRight]);
        view.ffn_grid_router = device_data<float>(
            model->blocks[block_index][kFfnGridRouter]);
        view.ffn_grid_out = device_data<float>(
            model->blocks[block_index][kFfnGridOut]);
    }

    model->device.rcut = model->rcut;
    model->device.ntypes = model->ntypes;
    model->device.channels = channels;
    model->device.feature_count = model->feature_count;
    model->device.type_embedding = device_data<float>(model->top[kTypeEmbedding]);
    model->device.env_rbf1 = device_data<float>(model->top[kEnvRbf1]);
    model->device.env_rbf2 = device_data<float>(model->top[kEnvRbf2]);
    model->device.env_type_embedding = device_data<float>(
        model->top[kEnvTypeEmbedding]);
    model->device.env_g1 = device_data<float>(model->top[kEnvG1]);
    model->device.env_g2 = device_data<float>(model->top[kEnvG2]);
    model->device.env_output = device_data<float>(model->top[kEnvOutput]);
    model->device.film_scale_norm = device_data<float>(
        model->top[kFilmScaleNorm]);
    model->device.film_shift_norm = device_data<float>(
        model->top[kFilmShiftNorm]);
    const auto scale_log = payload_array<float>(
        required(model_payload, "film_scale_strength_log"),
        "film_scale_strength_log");
    const auto shift_log = payload_array<float>(
        required(model_payload, "film_shift_strength_log"),
        "film_shift_strength_log");
    if (scale_log.size() != 1 || shift_log.size() != 1) {
        throw std::invalid_argument("DPA4 CUDA FiLM strength fields are malformed");
    }
    model->device.film_scale_strength_log = scale_log[0];
    model->device.film_shift_strength_log = shift_log[0];
    model->device.radial_freqs = device_data<float>(model->top[kRadialFreqs]);
    model->device.radial_layer1 = device_data<float>(model->top[kRadialLayer1]);
    model->device.radial_norm_scale = device_data<float>(
        model->top[kRadialNormScale]);
    model->device.radial_layer2 = device_data<float>(model->top[kRadialLayer2]);
    model->device.wigner_l2 = device_data<float>(model->top[kWignerL2]);
    model->device.wigner_l3 = device_data<float>(model->top[kWignerL3]);
    model->device.wigner_l3_exponents = device_data<std::int64_t>(
        model->top[kWignerL3Exponents]);
    model->device.gie_rows = device_data<std::int64_t>(model->top[kGieRows]);
    model->device.gie_m0 = device_data<std::int64_t>(model->top[kGieM0]);
    model->device.gie_radial = device_data<std::int64_t>(model->top[kGieRadial]);
    model->device.grid_to = device_data<float>(model->top[kGridTo]);
    model->device.grid_from = device_data<float>(model->top[kGridFrom]);
    model->device.blocks = model->block_views.data();
    model_ = std::move(model);
}

DeviceDpa4Model::~DeviceDpa4Model() noexcept {
    release();
}

void DeviceDpa4Model::release() noexcept {
    model_.reset();
}

std::int64_t DeviceDpa4Model::feature_count() const noexcept {
    return model_ == nullptr ? 0 : model_->feature_count;
}

double DeviceDpa4Model::cutoff() const noexcept {
    return model_ == nullptr ? 0.0 : model_->rcut;
}

std::vector<double> DeviceDpa4Model::compute(
    CudaExecutionContext& context,
    const DeviceBatch& batch,
    const DeviceNeighborGraph& graph,
    const std::vector<std::int32_t>& type_indices) const {
    if (model_ == nullptr) {
        throw std::runtime_error("DPA4 CUDA model is closed");
    }
    if (type_indices.size() != static_cast<std::size_t>(batch.atoms())) {
        throw std::invalid_argument(
            "DPA4 CUDA type index count does not match the batch");
    }
    for (std::int32_t type : type_indices) {
        if (type < 0 || type >= model_->ntypes) {
            throw std::invalid_argument(
                "DPA4 CUDA type index is outside the checkpoint type map");
        }
    }
    if (batch.atoms() == 0) {
        return {};
    }
    check_cuda(cudaSetDevice(context.device()), "could not select the DPA4 CUDA device");
    const std::size_t atoms = static_cast<std::size_t>(batch.atoms());
    const std::size_t edges = graph.pairs();
    const bool has_real_edges = edges != 0;
    const WorkspaceLayout layout = make_workspace_layout(atoms, edges);
    void* workspace = context.workspace_buffer(layout.bytes);
    float* state0 = workspace_data<float>(workspace, layout.state0);
    float* state1 = workspace_data<float>(workspace, layout.state1);
    float* pre_focus = workspace_data<float>(workspace, layout.pre_focus);
    float* aggregate = workspace_data<float>(workspace, layout.aggregate);
    float* message = workspace_data<float>(workspace, layout.message);
    float* hidden = workspace_data<float>(workspace, layout.hidden);
    float* activation = workspace_data<float>(workspace, layout.activation);
    float* radial = workspace_data<float>(workspace, layout.radial);
    float* radial_compact = workspace_data<float>(workspace, layout.radial_compact);
    float* radial_input = workspace_data<float>(workspace, layout.radial_input);
    float* radial_sign = workspace_data<float>(workspace, layout.radial_sign);
    float* radial_projection = workspace_data<float>(workspace, layout.radial_projection);
    float* radial_m1_output = workspace_data<float>(workspace, layout.radial_m1_output);
    float* envelopes = workspace_data<float>(workspace, layout.envelope);
    float* radial_bias = workspace_data<float>(workspace, layout.radial_bias);
    float* local = workspace_data<float>(workspace, layout.local);
    float* edge_message = workspace_data<float>(workspace, layout.edge_message);
    float* rotation = workspace_data<float>(workspace, layout.rotation);
    float* gie = workspace_data<float>(workspace, layout.gie);
    float* scratch0 = workspace_data<float>(workspace, layout.scratch0);
    float* scratch1 = workspace_data<float>(workspace, layout.scratch1);
    float* scratch2 = workspace_data<float>(workspace, layout.scratch2);
    float* grid_input = workspace_data<float>(workspace, layout.grid_input);
    auto ensure_cached_buffer = [](
        std::unique_ptr<DeviceArray>& buffer,
        std::size_t bytes,
        const char* operation) {
        if (buffer != nullptr && buffer->bytes >= bytes) {
            return;
        }
        auto replacement = std::make_unique<DeviceArray>();
        replacement->bytes = bytes;
        check_cuda(cudaMalloc(&replacement->pointer, bytes), operation);
        buffer = std::move(replacement);
    };
    ensure_cached_buffer(
        model_->type_indices_device,
        type_indices.size() * sizeof(std::int32_t),
        "could not allocate DPA4 CUDA type indices");
    check_cuda(
        cudaMemcpyAsync(
            model_->type_indices_device->pointer, type_indices.data(),
            type_indices.size() * sizeof(std::int32_t),
            cudaMemcpyHostToDevice, context.stream()),
        "could not upload DPA4 CUDA type indices");
    ensure_cached_buffer(
        model_->active_nodes_device,
        atoms * sizeof(std::int32_t),
        "could not allocate DPA4 CUDA active-node flags");
    check_cuda(
        cudaMemsetAsync(
            model_->active_nodes_device->pointer, 0,
            atoms * sizeof(std::int32_t), context.stream()),
        "could not clear DPA4 CUDA active-node flags");
    const auto* device_active_nodes =
        static_cast<const std::int32_t*>(model_->active_nodes_device->pointer);
    const auto* device_type_indices =
        static_cast<const std::int32_t*>(model_->type_indices_device->pointer);
    const unsigned int node_blocks =
        static_cast<unsigned int>((atoms + 127U) / 128U);
    const unsigned int edge_blocks =
        static_cast<unsigned int>((edges + 127U) / 128U);
    if (edges != 0) {
        build_rotation_gie_kernel<<<edge_blocks, 128, 0, context.stream()>>>(
            graph.offsets(), graph.atoms(), graph.shifts(), graph.displacements(),
            batch.atoms(), model_->device, rotation, gie);
        launch_check(cudaGetLastError(), "DPA4 rotation kernel launch failed");
    }
    if (edges != 0) {
        prepare_geometry_kernel<<<edge_blocks, 128, 0, context.stream()>>>(
            graph.offsets(), graph.atoms(), graph.shifts(), graph.displacements(),
            batch.atoms(), static_cast<std::int64_t>(edges), model_->device,
            static_cast<std::int32_t*>(model_->active_nodes_device->pointer),
            radial_compact, envelopes);
        launch_check(cudaGetLastError(), "DPA4 geometry kernel launch failed");
        environment_rbf(
            static_cast<std::int64_t>(edges), radial_compact,
            model_->device, radial_input, context.stream());
        prepare_environment_input_kernel<<<edge_blocks, 128, 0, context.stream()>>>(
            graph.offsets(), graph.atoms(), graph.shifts(), device_type_indices,
            batch.atoms(), static_cast<std::int64_t>(edges), radial_input,
            model_->device, radial_projection);
        launch_check(
            cudaGetLastError(),
            "DPA4 environment input kernel launch failed");
        environment_g(
            static_cast<std::int64_t>(edges), model_->device,
            radial_projection, radial_bias, context.stream());
    }
    prepare_environment_aggregate_kernel<<<
        static_cast<unsigned int>(atoms), 128, 0, context.stream()>>>(
        graph.offsets(), graph.displacements(), batch.atoms(), envelopes,
        radial_bias, state1 + 128, state1 + 640);
    launch_check(
        cudaGetLastError(),
        "DPA4 environment aggregation kernel launch failed");
    radial_initial(
        static_cast<std::int64_t>(edges), radial_compact,
        model_->device, radial_input, radial, envelopes, context.stream());
    prepare_film_kernel<<<static_cast<unsigned int>(atoms), 128, 0, context.stream()>>>(
        batch.atoms(), state1 + 128, model_->device, state1);
    launch_check(cudaGetLastError(), "DPA4 FiLM projection launch failed");
    prepare_finalize_kernel<<<static_cast<unsigned int>(atoms), 128, 0, context.stream()>>>(
        graph.offsets(), graph.atoms(), graph.shifts(), device_type_indices,
        batch.atoms(), state1, state1 + 640, gie, model_->device,
        static_cast<const std::int32_t*>(model_->active_nodes_device->pointer),
        radial, state0);
    launch_check(cudaGetLastError(), "DPA4 prepare finalize launch failed");
    for (int block_index = 0; block_index < 3 && has_real_edges; ++block_index) {
        const DeviceBlock block = model_->block_views[block_index];
        copy_state_kernel<<<static_cast<unsigned int>(atoms), 128, 0, context.stream()>>>(
            state0, batch.atoms(), message);
        launch_check(cudaGetLastError(), "DPA4 block snapshot launch failed");
        equivariant_copy_or_norm_kernel<<<node_blocks, 128, 0, context.stream()>>>(
            state0, batch.atoms(), block.pre_norm_enabled,
            block.pre_norm_scale, block.pre_norm_bias, block.pre_norm_balance,
            state1);
        launch_check(cudaGetLastError(), "DPA4 pre-norm kernel launch failed");
        so3_linear(
            state1, static_cast<std::int64_t>(atoms), 64, 64,
            block.pre_focus, pre_focus, context.stream());
        if (edges != 0) {
            constexpr unsigned int edge_block_size = 256;
            edge_local_from_state_kernel<<<static_cast<unsigned int>(edges), edge_block_size, 0, context.stream()>>>(
                graph.offsets(), graph.atoms(), graph.shifts(), batch.atoms(),
                rotation, pre_focus, radial, local, radial_bias);
            launch_check(cudaGetLastError(), "DPA4 local rotation launch failed");
            radial_so2(
                static_cast<std::int64_t>(edges), radial, block,
                radial_compact, radial_input, radial_sign, radial_projection,
                radial_m1_output, local, edge_message, context.stream());
            rotate_edge_kernel<<<static_cast<unsigned int>(edges), edge_block_size, 0, context.stream()>>>(
                graph.offsets(), batch.atoms(), rotation, local, edge_message);
            launch_check(cudaGetLastError(), "DPA4 edge message launch failed");
        } else {
            check_cuda(
                cudaMemsetAsync(
                    aggregate, 0, atoms * kFullDim * kChannels * sizeof(float),
                    context.stream()),
                "could not clear DPA4 aggregate");
        }
        // The pre-normalized state is dead after pre_focus.  Reuse its first
        // two scalar-width slices for Q and K rather than growing the
        // per-node workspace just for these projections.
        float* query = state1;
        float* key = state1 + atoms * kChannels;
        qk_kernel<<<static_cast<unsigned int>(atoms), 128, 0, context.stream()>>>(
            pre_focus, batch.atoms(), block, query, key);
        launch_check(cudaGetLastError(), "DPA4 QK kernel launch failed");
        attention_kernel<<<node_blocks, 128, 0, context.stream()>>>(
            graph.offsets(), graph.atoms(), graph.shifts(), batch.atoms(),
            envelopes, radial_bias, edge_message, pre_focus,
            query, key,
            block, aggregate);
        launch_check(cudaGetLastError(), "DPA4 attention kernel launch failed");
        for (std::size_t start = 0; start < atoms; start += kGridTileNodes) {
            const std::size_t tile = std::min(
                static_cast<std::size_t>(kGridTileNodes), atoms - start);
            message_grid_kernel<<<static_cast<unsigned int>(tile), 128, 0, context.stream()>>>(
                aggregate, pre_focus, static_cast<std::int64_t>(start),
                static_cast<std::int64_t>(tile), block, model_->device.grid_to,
                model_->device.grid_from, scratch0, scratch1, scratch2, message);
            launch_check(cudaGetLastError(), "DPA4 message grid launch failed");
        }
        post_state_kernel<<<static_cast<unsigned int>(atoms), 128, 0, context.stream()>>>(
            state0, aggregate, message, batch.atoms(), block, state1);
        launch_check(cudaGetLastError(), "DPA4 post-focus launch failed");
        // post_state leaves pre_focus dead until the next block.  Reuse it as
        // the normalized FFN input, then use four row-grouped GEMMs for the
        // degree-specific 64->1152 projection.
        ffn_normalize_kernel<<<static_cast<unsigned int>(atoms), 128, 0, context.stream()>>>(
            state1, batch.atoms(), block, pre_focus);
        launch_check(cudaGetLastError(), "DPA4 FFN normalization launch failed");
        so3_linear(
            pre_focus, static_cast<std::int64_t>(atoms), 64, 1152,
            block.ffn_linear1, hidden, context.stream());
        for (std::size_t start = 0; start < atoms; start += kGridTileNodes) {
            const std::size_t tile = std::min(
                static_cast<std::size_t>(kGridTileNodes), atoms - start);
            block_grid(
                hidden, static_cast<std::int64_t>(start),
                static_cast<std::int64_t>(tile), model_->device.grid_to,
                model_->device.grid_from, block.ffn_grid_left,
                block.ffn_grid_right, block.ffn_grid_out,
            block.ffn_scalar_gate, grid_input, scratch0, scratch1,
                scratch2, activation, context.stream());
        }
        linear2_residual_kernel<<<static_cast<unsigned int>(atoms), 128, 0, context.stream()>>>(
            state1, activation, batch.atoms(), block, state0);
        launch_check(cudaGetLastError(), "DPA4 FFN output launch failed");
        restore_inactive_state_kernel<<<static_cast<unsigned int>(atoms), 128, 0, context.stream()>>>(
            message, device_active_nodes, batch.atoms(), state0);
        launch_check(cudaGetLastError(), "DPA4 inactive-state restore failed");
    }
    so3_linear(
        state0, static_cast<std::int64_t>(atoms), 64, 1152,
        device_data<float>(model_->top[kOutputLinear1]), hidden,
        context.stream());
    for (std::size_t start = 0; start < atoms; start += kGridTileNodes) {
        const std::size_t tile = std::min(
            static_cast<std::size_t>(kGridTileNodes), atoms - start);
        output_grid(
            hidden, static_cast<std::int64_t>(start),
            static_cast<std::int64_t>(tile), model_->device.grid_to,
            model_->device.grid_from,
            device_data<float>(model_->top[kOutputGridLeft]),
            device_data<float>(model_->top[kOutputGridRight]),
            device_data<float>(model_->top[kOutputGridOut]),
            device_data<float>(model_->top[kOutputScalarGate]), grid_input,
            scratch0, scratch1, scratch2, activation, context.stream());
    }
    double* output = context.output_buffer(atoms * kChannels);
    output_linear2_kernel<<<static_cast<unsigned int>(atoms), 128, 0, context.stream()>>>(
        state0, activation, batch.atoms(),
        device_data<float>(model_->top[kOutputLinear2]), output);
    launch_check(cudaGetLastError(), "DPA4 output head launch failed");
    auto result = context.download_output(atoms * kChannels);
    return result;
}

} // namespace mdescriptor::cuda
