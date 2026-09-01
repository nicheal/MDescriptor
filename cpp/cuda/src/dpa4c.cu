#include "mdescriptor/cuda/dpa4c.hpp"

#include "mdescriptor/dpa4c.hpp"

#include <cuda_runtime.h>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
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

struct DeviceDpa4cModel::DeviceArray {
    void* pointer = nullptr;
    std::size_t bytes = 0;
    ~DeviceArray() noexcept { if (pointer != nullptr) (void)cudaFree(pointer); }
};

namespace {

constexpr float kSqrt2 = 1.41421356237309504880F;
constexpr float kSqrt3 = 1.73205080756887729353F;
constexpr float kSqrt5 = 2.23606797749978969641F;
constexpr float kSqrt6 = 2.44948974278317809820F;
constexpr float kEpsilon = 1.0e-7F;
constexpr float kNormFloor = 0.25F;

void check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    const std::string message = std::string(operation) + ": " + cudaGetErrorString(status);
    if (status == cudaErrorMemoryAllocation) throw CudaOutOfMemory(message.c_str());
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver
        || status == cudaErrorSystemDriverMismatch || status == cudaErrorUnknown) {
        throw CudaUnavailable(message.c_str());
    }
    throw std::runtime_error(message);
}

template <typename Value>
std::vector<Value> payload_array(py::handle value, const char* name) {
    using Array = py::array_t<Value, py::array::c_style | py::array::forcecast>;
    const Array array = Array::ensure(value);
    if (!array) throw std::invalid_argument(std::string("DPA4C ") + name + " must be a numeric array");
    const auto info = array.request();
    if (info.size == 0) return {};
    const auto* data = static_cast<const Value*>(info.ptr);
    return std::vector<Value>(data, data + info.size);
}

py::handle required(const py::dict& payload, const char* name) {
    if (!payload.contains(name)) {
        throw std::invalid_argument(std::string("DPA4C CUDA payload is missing ") + name);
    }
    return payload[name];
}

template <typename Value>
void expect_size(const std::vector<Value>& value, std::size_t expected, const char* name) {
    if (value.size() != expected) {
        throw std::invalid_argument(std::string("DPA4C CUDA ") + name + " has unexpected size");
    }
}

void expect_offsets(const std::vector<std::int64_t>& offsets, std::size_t count, const char* name) {
    expect_size(offsets, count, name);
    if (offsets.empty() || offsets.front() != 0 || offsets.back() < 0) {
        throw std::invalid_argument(std::string("DPA4C CUDA ") + name + " is invalid");
    }
    for (std::size_t index = 1; index < offsets.size(); ++index) {
        if (offsets[index] < offsets[index - 1]) {
            throw std::invalid_argument(std::string("DPA4C CUDA ") + name + " is not monotonic");
        }
    }
}

struct HostPayload {
    double rcut = 0.0;
    int ntypes = 0;
    int channels = 0;
    int lmax = 0;
    int n_radial = 0;
    int radial_modes = 0;
    int radial_hidden = 0;
    int pair_hidden = 0;
    bool calibrate = true;
    std::vector<float> type_embedding;
    std::vector<float> radial_freqs;
    std::vector<float> radial_w0;
    std::vector<float> radial_w1;
    std::vector<float> radial_mode_w;
    std::vector<float> pair_w0;
    std::vector<float> pair_w1;
    std::vector<int> degree_channels;
    std::vector<int> bispectrum_ranks;
    std::vector<float> alignment;
    std::vector<std::int64_t> alignment_offsets;
    std::vector<float> projections;
    std::vector<std::int64_t> projection_offsets;
    std::vector<float> coupling;
    std::vector<std::int64_t> coupling_offsets;
    std::vector<int> degree_triples;
    std::vector<std::int64_t> probe_offsets;
    std::vector<std::int64_t> probe_index;
    std::vector<float> probe_scale;
    std::vector<float> output_mean;
    std::vector<float> output_stddev;
    std::vector<std::int32_t> type_numbers;
};

HostPayload parse_payload(const py::dict& payload) {
    auto unsupported_flag = [](py::handle value) {
        if (value.is_none()) return false;
        if (py::isinstance<py::bool_>(value)) return py::cast<bool>(value);
        return true;
    };
    for (const char* key : {"compress", "compression", "use_spin", "spin", "charge_spin_embedding"}) {
        if (payload.contains(key) && unsupported_flag(payload[key])) {
            throw std::invalid_argument(
                std::string("DPA4C CUDA does not support ") + key
                + "; use the uncompressed spin-free native payload");
        }
    }
    HostPayload p;
    p.rcut = py::cast<double>(required(payload, "rcut"));
    p.ntypes = py::cast<int>(required(payload, "ntypes"));
    p.channels = py::cast<int>(required(payload, "channels"));
    p.lmax = py::cast<int>(required(payload, "lmax"));
    p.n_radial = py::cast<int>(required(payload, "n_radial"));
    p.radial_modes = py::cast<int>(required(payload, "radial_modes"));
    p.radial_hidden = py::cast<int>(required(payload, "radial_hidden"));
    p.pair_hidden = py::cast<int>(required(payload, "pair_hidden"));
    if (payload.contains("calibrate")) p.calibrate = py::cast<bool>(payload["calibrate"]);
    p.type_embedding = payload_array<float>(required(payload, "type_embedding"), "type_embedding");
    p.radial_freqs = payload_array<float>(required(payload, "radial_freqs"), "radial_freqs");
    p.radial_w0 = payload_array<float>(required(payload, "radial_w0"), "radial_w0");
    p.radial_w1 = payload_array<float>(required(payload, "radial_w1"), "radial_w1");
    p.radial_mode_w = payload_array<float>(required(payload, "radial_mode_w"), "radial_mode_w");
    p.pair_w0 = payload_array<float>(required(payload, "pair_w0"), "pair_w0");
    p.pair_w1 = payload_array<float>(required(payload, "pair_w1"), "pair_w1");
    p.degree_channels = py::cast<std::vector<int>>(required(payload, "degree_channels"));
    p.bispectrum_ranks = py::cast<std::vector<int>>(required(payload, "bispectrum_ranks"));
    p.alignment = payload_array<float>(required(payload, "readout_alignment"), "readout_alignment");
    p.alignment_offsets = payload_array<std::int64_t>(
        required(payload, "readout_alignment_offsets"), "readout_alignment_offsets");
    p.projections = payload_array<float>(required(payload, "readout_projections"), "readout_projections");
    p.projection_offsets = payload_array<std::int64_t>(
        required(payload, "readout_projection_offsets"), "readout_projection_offsets");
    p.coupling = payload_array<float>(required(payload, "bispectrum_coupling"), "bispectrum_coupling");
    p.coupling_offsets = payload_array<std::int64_t>(required(payload, "coupling_offsets"), "coupling_offsets");
    p.degree_triples = py::cast<std::vector<int>>(required(payload, "degree_triples"));
    p.probe_offsets = payload_array<std::int64_t>(required(payload, "probe_offsets"), "probe_offsets");
    p.probe_index = payload_array<std::int64_t>(required(payload, "probe_index"), "probe_index");
    p.probe_scale = payload_array<float>(required(payload, "probe_scale"), "probe_scale");
    p.output_mean = payload_array<float>(required(payload, "output_mean"), "output_mean");
    p.output_stddev = payload_array<float>(required(payload, "output_stddev"), "output_stddev");
    if (payload.contains("type_numbers")) {
        try {
            p.type_numbers = payload_array<std::int32_t>(payload["type_numbers"], "type_numbers");
        } catch (const std::invalid_argument&) {
            p.type_numbers = py::cast<std::vector<std::int32_t>>(payload["type_numbers"]);
        }
    }

    if (!std::isfinite(p.rcut) || p.rcut <= 0.0 || p.ntypes <= 0 || p.channels <= 0
        || p.lmax < 2 || p.lmax > 4 || p.n_radial <= 0 || p.radial_modes < 0
        || p.radial_hidden <= 0 || p.pair_hidden <= 0) {
        throw std::invalid_argument("DPA4C CUDA has an invalid structural configuration");
    }
    expect_size(p.degree_channels, static_cast<std::size_t>(p.lmax + 1), "degree_channels");
    expect_size(p.bispectrum_ranks, static_cast<std::size_t>(p.lmax), "bispectrum_ranks");
    if (p.degree_channels[0] != p.channels) {
        throw std::invalid_argument("DPA4C CUDA degree-zero width does not match channels");
    }
    for (int width : p.degree_channels) if (width <= 0) throw std::invalid_argument("DPA4C CUDA degree widths must be positive");
    for (int rank : p.bispectrum_ranks) if (rank <= 0) throw std::invalid_argument("DPA4C CUDA probe ranks must be positive");
    const std::size_t type_rows = static_cast<std::size_t>(p.ntypes + 1);
    expect_size(p.type_embedding, type_rows * static_cast<std::size_t>(p.channels), "type_embedding");
    expect_size(p.radial_freqs, static_cast<std::size_t>(p.n_radial), "radial_freqs");
    expect_size(p.radial_w0, static_cast<std::size_t>(p.n_radial) * 2 * p.radial_hidden, "radial_w0");
    expect_size(p.radial_w1, static_cast<std::size_t>(p.radial_hidden) * p.channels, "radial_w1");
    expect_size(p.radial_mode_w, static_cast<std::size_t>(p.radial_hidden) * p.radial_modes, "radial_mode_w");
    expect_size(p.pair_w0, static_cast<std::size_t>(2 * p.channels) * 2 * p.pair_hidden, "pair_w0");
    expect_size(p.pair_w1, static_cast<std::size_t>(p.pair_hidden) * p.channels * (2 + p.radial_modes), "pair_w1");
    expect_offsets(p.alignment_offsets, 3, "readout_alignment_offsets");
    expect_offsets(p.projection_offsets, static_cast<std::size_t>(p.lmax + 1), "readout_projection_offsets");
    expect_size(p.alignment, static_cast<std::size_t>(p.alignment_offsets.back()), "readout_alignment");
    expect_size(p.projections, static_cast<std::size_t>(p.projection_offsets.back()), "readout_projections");
    for (int degree = 1; degree <= 2; ++degree) {
        const std::int64_t size = p.alignment_offsets[degree] - p.alignment_offsets[degree - 1];
        if (size != static_cast<std::int64_t>(p.degree_channels[degree] * p.degree_channels[degree]))
            throw std::invalid_argument("DPA4C CUDA alignment matrix has unexpected shape");
    }
    for (int degree = 1; degree <= p.lmax; ++degree) {
        const std::int64_t size = p.projection_offsets[degree] - p.projection_offsets[degree - 1];
        const int expected = p.degree_channels[degree] * p.bispectrum_ranks[degree - 1];
        if (size != 0 && size != expected) throw std::invalid_argument("DPA4C CUDA projection matrix has unexpected shape");
        if (size == 0 && p.bispectrum_ranks[degree - 1] != p.degree_channels[degree])
            throw std::invalid_argument("DPA4C CUDA non-full-rank probe is missing");
    }
    if (p.degree_triples.size() % 3 != 0) throw std::invalid_argument("DPA4C CUDA degree triples are malformed");
    const std::size_t triples = p.degree_triples.size() / 3;
    expect_offsets(p.coupling_offsets, triples + 1, "coupling_offsets");
    expect_offsets(p.probe_offsets, triples + 1, "probe_offsets");
    expect_size(p.coupling, static_cast<std::size_t>(p.coupling_offsets.back()), "bispectrum_coupling");
    expect_size(p.probe_index, static_cast<std::size_t>(p.probe_offsets.back()), "probe_index");
    expect_size(p.probe_scale, p.probe_index.size(), "probe_scale");
    std::int64_t moment_count = 0;
    std::int64_t gram_count = 0;
    for (int degree = 0; degree <= p.lmax; ++degree) {
        moment_count += static_cast<std::int64_t>(2 * degree + 1) * p.degree_channels[degree];
        if (degree > 0) gram_count += static_cast<std::int64_t>(p.degree_channels[degree]) * (p.degree_channels[degree] + 1) / 2;
    }
    const std::int64_t expected_features = p.channels + gram_count + p.probe_offsets.back()
        + static_cast<std::int64_t>(p.bispectrum_ranks[0]) * p.bispectrum_ranks[1] + 2 + p.channels;
    if (expected_features <= 0 || p.output_mean.size() != static_cast<std::size_t>(expected_features)
        || p.output_stddev.size() != static_cast<std::size_t>(expected_features))
        throw std::invalid_argument("DPA4C CUDA output calibration has unexpected shape");
    for (float value : p.output_stddev) if (!std::isfinite(value) || value <= 0.0F)
        throw std::invalid_argument("DPA4C CUDA output standard deviations must be positive");
    if (!p.type_numbers.empty() && p.type_numbers.size() != static_cast<std::size_t>(p.ntypes))
        throw std::invalid_argument("DPA4C CUDA type_numbers must have ntypes entries");
    for (std::size_t index = 0; index < triples; ++index) {
        std::int64_t full_size = 1;
        int degrees[3] = {};
        for (int component = 0; component < 3; ++component) {
            const int degree = p.degree_triples[index * 3 + component];
            if (degree < 1 || degree > p.lmax) throw std::invalid_argument("DPA4C CUDA degree triple is outside lmax");
            degrees[component] = degree;
            full_size *= p.bispectrum_ranks[degree - 1];
        }
        const bool special_112 = degrees[0] == 1 && degrees[1] == 1 && degrees[2] == 2;
        const std::int64_t coupling_size = p.coupling_offsets[index + 1] - p.coupling_offsets[index];
        const std::int64_t expected_coupling_size = static_cast<std::int64_t>(2 * degrees[0] + 1)
            * (2 * degrees[1] + 1) * (2 * degrees[2] + 1);
        if (!special_112 && coupling_size != expected_coupling_size) {
            throw std::invalid_argument("DPA4C CUDA coupling tensor has unexpected shape");
        }
        for (std::int64_t probe = p.probe_offsets[index]; probe < p.probe_offsets[index + 1]; ++probe) {
            if (p.probe_index[static_cast<std::size_t>(probe)] < 0
                || p.probe_index[static_cast<std::size_t>(probe)] >= full_size) {
                throw std::invalid_argument("DPA4C CUDA probe index is outside its contraction");
            }
        }
    }
    return p;
}

struct Dpa4cCudaLayout {
    std::int64_t stride = 0;
    std::int64_t fixed_bytes = 0;
    std::int64_t reduced = 0;
    std::int64_t radial_basis = 0;
    std::int64_t radial_pre = 0;
    std::int64_t radial_hidden = 0;
    std::int64_t radial = 0;
    std::int64_t modes = 0;
    std::int64_t basis = 0;
    std::int64_t amplitudes = 0;
    std::int64_t moments = 0;
    std::int64_t blocks = 0;
    std::int64_t projected = 0;
    std::int64_t descriptor = 0;
    std::int64_t full = 0;
    std::int64_t matrices = 0;
};

std::size_t align_bytes(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

template <typename Value>
std::unique_ptr<DeviceDpa4cModel::DeviceArray> upload_array(
    CudaExecutionContext& context, const std::vector<Value>& values, const char* operation) {
    auto result = std::make_unique<DeviceDpa4cModel::DeviceArray>();
    result->bytes = values.size() * sizeof(Value);
    if (values.empty()) return result;
    check_cuda(cudaSetDevice(context.device()), "could not select the CUDA device for DPA4C");
    try {
        check_cuda(cudaMalloc(&result->pointer, result->bytes), operation);
        check_cuda(cudaMemcpyAsync(
            result->pointer, values.data(), result->bytes,
            cudaMemcpyHostToDevice, context.stream()), operation);
    } catch (...) {
        result.reset();
        throw;
    }
    return result;
}

template <typename Value>
Value* device_data(const std::unique_ptr<DeviceDpa4cModel::DeviceArray>& value) {
    return value == nullptr ? nullptr : static_cast<Value*>(value->pointer);
}

struct KernelModel {
    double rcut;
    int ntypes, channels, lmax, n_radial, radial_modes, radial_hidden, pair_hidden;
    bool calibrate;
    std::int64_t feature_count, moment_count, triple_count;
    const float* type_embedding;
    const float* radial_freqs;
    const float* radial_w0;
    const float* radial_w1;
    const float* radial_mode_w;
    const float* pair_scale;
    const float* pair_shift;
    const float* pair_mixing;
    const float* alignment;
    const std::int64_t* alignment_offsets;
    const float* projections;
    const std::int64_t* projection_offsets;
    const float* coupling;
    const std::int64_t* coupling_offsets;
    const int* degree_triples;
    const std::int64_t* probe_offsets;
    const std::int64_t* probe_index;
    const float* probe_scale;
    const float* output_mean;
    const float* output_stddev;
    const std::int32_t* gram_index;
    const float* gram_scale;
    const int* degree_channels;
    const int* bispectrum_ranks;
};

__device__ __forceinline__ float affine(const float* weights, int input_width, int output_width, const float* input, int output) {
    // The model and the deepmd-kit GPU reference use float32 GEMMs.  Double
    // accumulation is disproportionately slow on consumer GPUs and is not
    // needed for the descriptor's float32 intermediate tensors.
    float value = 0.0F;
    for (int index = 0; index < input_width; ++index) value += input[index] * weights[index * output_width + output];
    return value;
}

__device__ __forceinline__ float sigmoid(float value) { return 1.0F / (1.0F + expf(-value)); }

__device__ void angular_basis(float x, float y, float z, int lmax, float* result) {
    const float norm2 = x * x + y * y + z * z;
    const float x2 = x * x, y2 = y * y, z2 = z * z;
    for (int index = 0; index < 25; ++index) result[index] = 0.0F;
    result[0] = 1.0F;
    if (lmax >= 1) { result[1] = x; result[2] = y; result[3] = z; }
    if (lmax >= 2) {
        result[4] = kSqrt3 * x * y; result[5] = kSqrt3 * y * z;
        result[6] = 0.5F * (3.0F * z2 - norm2); result[7] = kSqrt3 * x * z;
        result[8] = 0.5F * kSqrt3 * (x2 - y2);
    }
    if (lmax >= 3) {
        result[9] = sqrtf(5.0F / 8.0F) * y * (3.0F * x2 - y2);
        result[10] = sqrtf(15.0F) * x * y * z;
        result[11] = sqrtf(3.0F / 8.0F) * y * (5.0F * z2 - norm2);
        result[12] = 0.5F * z * (5.0F * z2 - 3.0F * norm2);
        result[13] = sqrtf(3.0F / 8.0F) * x * (5.0F * z2 - norm2);
        result[14] = 0.5F * sqrtf(15.0F) * z * (x2 - y2);
        result[15] = sqrtf(5.0F / 8.0F) * x * (x2 - 3.0F * y2);
    }
    if (lmax >= 4) {
        const float xdiff = x2 - y2, z4 = z2 * z2, norm22 = norm2 * norm2;
        result[16] = 0.5F * sqrtf(35.0F) * x * y * xdiff;
        result[17] = 0.25F * sqrtf(70.0F) * y * z * (3.0F * x2 - y2);
        result[18] = 0.5F * sqrtf(5.0F) * x * y * (7.0F * z2 - norm2);
        result[19] = 0.25F * sqrtf(10.0F) * y * z * (7.0F * z2 - 3.0F * norm2);
        result[20] = 0.125F * (35.0F * z4 - 30.0F * z2 * norm2 + 3.0F * norm22);
        result[21] = 0.25F * sqrtf(10.0F) * x * z * (7.0F * z2 - 3.0F * norm2);
        result[22] = 0.25F * sqrtf(5.0F) * xdiff * (7.0F * z2 - norm2);
        result[23] = 0.25F * sqrtf(70.0F) * x * z * (x2 - 3.0F * y2);
        result[24] = 0.125F * sqrtf(35.0F) * (x2 * x2 - 6.0F * x2 * y2 + y2 * y2);
    }
}

// The projection widths are encoded by the offset arrays and the degree
// profile is passed separately in the kernel as a compact constant-sized array.
__device__ void packed_l2_to_stf(const float* packed, int rank, float* matrices) {
    for (int channel = 0; channel < rank; ++channel) {
        const float q0 = packed[channel], q1 = packed[rank + channel], q2 = packed[2 * rank + channel];
        const float q3 = packed[3 * rank + channel], q4 = packed[4 * rank + channel];
        const float qxy = q0 / kSqrt2, qyz = q1 / kSqrt2, qxz = q3 / kSqrt2;
        float* matrix = matrices + channel * 9;
        matrix[0] = -q2 / kSqrt6 + q4 / kSqrt2; matrix[1] = qxy; matrix[2] = qxz;
        matrix[3] = qxy; matrix[4] = -q2 / kSqrt6 - q4 / kSqrt2; matrix[5] = qyz;
        matrix[6] = qxz; matrix[7] = qyz; matrix[8] = 2.0F * q2 / kSqrt6;
    }
}

__device__ __forceinline__ int moment_offset_for_degree(const KernelModel& m, int degree) {
    int offset = m.channels;
    for (int current = 1; current < degree; ++current)
        offset += (2 * current + 1) * m.degree_channels[current];
    return offset;
}

__device__ __forceinline__ int projected_offset_for_degree(const KernelModel& m, int degree) {
    int offset = 0;
    for (int current = 1; current < degree; ++current)
        offset += (2 * current + 1) * m.bispectrum_ranks[current - 1];
    return offset;
}

__global__ void dpa4c_kernel(
    const std::int64_t* graph_offsets, const std::int32_t* graph_atoms,
    const std::int32_t* graph_shifts, const double* displacements,
    const std::int32_t* type_indices,
    std::int64_t atoms, unsigned char* workspace, Dpa4cCudaLayout layout,
    KernelModel m,
    double* output) {
    const std::int64_t center = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms) return;
    unsigned char* base = workspace + center * layout.stride;
    auto* reduced = reinterpret_cast<double*>(base + layout.reduced);
    auto* radial_basis = reinterpret_cast<float*>(base + layout.radial_basis);
    auto* radial_pre = reinterpret_cast<float*>(base + layout.radial_pre);
    auto* radial_hidden = reinterpret_cast<float*>(base + layout.radial_hidden);
    auto* radial = reinterpret_cast<float*>(base + layout.radial);
    auto* modes = reinterpret_cast<float*>(base + layout.modes);
    auto* basis = reinterpret_cast<float*>(base + layout.basis);
    auto* amplitudes = reinterpret_cast<double*>(base + layout.amplitudes);
    auto* moments = reinterpret_cast<float*>(base + layout.moments);
    auto* blocks = reinterpret_cast<float*>(base + layout.blocks);
    auto* projected = reinterpret_cast<float*>(base + layout.projected);
    auto* descriptor = reinterpret_cast<float*>(base + layout.descriptor);
    auto* full = reinterpret_cast<float*>(base + layout.full);
    auto* matrices = reinterpret_cast<float*>(base + layout.matrices);
    const int center_type = type_indices[center];
    const std::int64_t begin = graph_offsets[center], end = graph_offsets[center + 1];
    for (std::int64_t index = 0; index < 2 + m.moment_count; ++index) reduced[index] = 0.0;
    // DeviceNeighborGraph::build_dpa orders every row by distance before
    // this kernel launches.  Reusing that order avoids a second per-center
    // insertion sort here; it also preserves the graph's deterministic
    // tie-break order.
    for (std::int64_t edge = begin; edge < end; ++edge) {
        const std::int32_t neighbor = graph_atoms[edge];
        if (neighbor == center && graph_shifts != nullptr
            && graph_shifts[edge * 3] == 0 && graph_shifts[edge * 3 + 1] == 0 && graph_shifts[edge * 3 + 2] == 0) continue;
        const int neighbor_type = type_indices[neighbor];
        const float dx = static_cast<float>(displacements[edge * 3]);
        const float dy = static_cast<float>(displacements[edge * 3 + 1]);
        const float dz = static_cast<float>(displacements[edge * 3 + 2]);
        const float d2 = dx * dx + dy * dy + dz * dz;
        const float distance = sqrtf(d2 + kEpsilon * kEpsilon);
        const float ux = dx / distance, uy = dy / distance, uz = dz / distance;
        float cutoff_coordinate = (static_cast<float>(m.rcut) - distance) / static_cast<float>(m.rcut);
        cutoff_coordinate = fmaxf(0.0F, fminf(1.0F, cutoff_coordinate));
        const float x = 1.0F - cutoff_coordinate;
        // Keep the five scalar operations in the same order as the CPU
        // reference.  A compact Horner expression lets nvcc reassociate or
        // contract operations and changes the last float ulps of the cutoff.
        float series = 35.0F;
        series = 20.0F + x * series;
        series = 10.0F + x * series;
        series = 4.0F + x * series;
        series = 1.0F + x * series;
        const float envelope = cutoff_coordinate * cutoff_coordinate * cutoff_coordinate * cutoff_coordinate * series;
        for (int radial_index = 0; radial_index < m.n_radial; ++radial_index) {
            const float argument = distance * m.radial_freqs[radial_index];
            // This is the same dtype-level sequence as the vendored
            // RadialBasis: torch.sinc(argument / pi), with the zero limit
            // made explicit for the CUDA scalar path.
            constexpr float pi = 3.1415927410125732422F;
            const float sinc_argument = argument / pi;
            const float sinc = sinc_argument == 0.0F
                ? 1.0F : sinf(pi * sinc_argument) / (pi * sinc_argument);
            radial_basis[radial_index] = m.radial_freqs[radial_index] * sinc;
        }
        for (int hidden = 0; hidden < 2 * m.radial_hidden; ++hidden)
            radial_pre[hidden] = affine(m.radial_w0, m.n_radial, 2 * m.radial_hidden, radial_basis, hidden);
        for (int hidden = 0; hidden < m.radial_hidden; ++hidden)
            radial_hidden[hidden] = radial_pre[hidden] * sigmoid(radial_pre[hidden]) * radial_pre[m.radial_hidden + hidden];
        for (int channel = 0; channel < m.channels; ++channel)
            radial[channel] = affine(m.radial_w1, m.radial_hidden, m.channels, radial_hidden, channel);
        for (int mode = 0; mode < m.radial_modes; ++mode)
            modes[mode] = affine(m.radial_mode_w, m.radial_hidden, m.radial_modes, radial_hidden, mode);
        const int pair = center_type * (m.ntypes + 1) + neighbor_type;
        angular_basis(ux, uy, uz, m.lmax, basis);
        reduced[0] += static_cast<double>(envelope) * envelope;
        const float envelope2 = envelope * envelope;
        reduced[1] += static_cast<double>(envelope2) * envelope2;
        for (int channel = 0; channel < m.channels; ++channel) {
            double value = static_cast<double>(radial[channel]) * m.pair_scale[pair * m.channels + channel]
                + m.pair_shift[pair * m.channels + channel];
            for (int mode = 0; mode < m.radial_modes; ++mode)
                value += static_cast<double>(m.pair_mixing[(pair * m.channels + channel) * m.radial_modes + mode]) * modes[mode];
            amplitudes[channel] = value;
            reduced[2 + channel] += value * envelope;
        }
        int moment_offset = 0;
        for (int degree = 0; degree <= m.lmax; ++degree) {
            const int width = m.degree_channels[degree];
            if (degree > 0) {
                const int basis_offset = degree * degree;
                for (int component = 0; component < 2 * degree + 1; ++component)
                    for (int channel = 0; channel < width; ++channel)
                        reduced[2 + moment_offset + component * width + channel] += amplitudes[channel] * envelope2 * basis[basis_offset + component];
            }
            moment_offset += (2 * degree + 1) * width;
        }
    }
    const double divisor_scalar = sqrt(reduced[0] + static_cast<double>(kNormFloor));
    const double divisor_angular = sqrt(reduced[1] + static_cast<double>(kNormFloor));
    for (int index = 0; index < m.moment_count; ++index) moments[index] = 0.0F;
    for (int channel = 0; channel < m.channels; ++channel) moments[channel] = static_cast<float>(reduced[2 + channel] / divisor_scalar);
    int moment_offset = m.channels;
    for (int degree = 1; degree <= m.lmax; ++degree) {
        const int width = m.degree_channels[degree];
        for (int component = 0; component < 2 * degree + 1; ++component)
            for (int channel = 0; channel < width; ++channel)
                moments[moment_offset + component * width + channel] = static_cast<float>(reduced[2 + moment_offset + component * width + channel] / divisor_angular);
        moment_offset += (2 * degree + 1) * width;
    }
    int block_offset = 0;
    for (int degree = 0; degree <= m.lmax; ++degree) {
        const int width = m.degree_channels[degree], dimension = 2 * degree + 1;
        for (int index = 0; index < dimension * width; ++index) blocks[block_offset + index] = moments[block_offset + index];
        if (degree == 1 || degree == 2) {
            for (int index = 0; index < dimension * width; ++index) reinterpret_cast<float*>(base + layout.full)[index] = blocks[block_offset + index];
            const float* matrix = m.alignment + m.alignment_offsets[degree - 1];
            for (int component = 0; component < dimension; ++component) for (int out = 0; out < width; ++out) {
                double value = blocks[block_offset + component * width + out];
                for (int in = 0; in < width; ++in) value += static_cast<double>(reinterpret_cast<float*>(base + layout.full)[component * width + in]) * matrix[in * width + out];
                blocks[block_offset + component * width + out] = static_cast<float>(value);
            }
        }
        block_offset += dimension * width;
    }
    int projected_offset = 0;
    for (int degree = 1; degree <= m.lmax; ++degree) {
        const int width = m.degree_channels[degree], rank = m.bispectrum_ranks[degree - 1], dimension = 2 * degree + 1;
        const std::int64_t matrix_begin = m.projection_offsets[degree - 1], matrix_end = m.projection_offsets[degree];
        const int source_offset = moment_offset_for_degree(m, degree);
        if (matrix_begin == matrix_end) {
            for (int index = 0; index < dimension * rank; ++index) projected[projected_offset + index] = blocks[source_offset + index];
        } else {
            const float* matrix = m.projections + matrix_begin;
            for (int component = 0; component < dimension; ++component) for (int out = 0; out < rank; ++out) {
                double value = 0.0;
                for (int in = 0; in < width; ++in) value += static_cast<double>(blocks[source_offset + component * width + in]) * matrix[in * rank + out];
                projected[projected_offset + component * rank + out] = static_cast<float>(value);
            }
        }
        projected_offset += dimension * rank;
    }
    int descriptor_offset = 0;
    for (int index = 0; index < m.channels; ++index) descriptor[descriptor_offset++] = blocks[index];
    int gram_cursor = 0;
    block_offset = m.channels;
    for (int degree = 1; degree <= m.lmax; ++degree) {
        const int width = m.degree_channels[degree], dimension = 2 * degree + 1;
        const int gram_count = width * (width + 1) / 2;
        for (int gram = 0; gram < gram_count; ++gram) {
            const int flat = m.gram_index[gram_cursor];
            const int row = flat / width, column = flat % width;
            double value = 0.0;
            for (int component = 0; component < dimension; ++component) value += static_cast<double>(blocks[block_offset + component * width + row]) * blocks[block_offset + component * width + column];
            descriptor[descriptor_offset++] = static_cast<float>(value * m.gram_scale[gram_cursor++]);
        }
        block_offset += dimension * width;
    }
    projected_offset = 0;
    for (std::int64_t triple = 0; triple < m.triple_count; ++triple) {
        const int degree1 = m.degree_triples[triple * 3], degree2 = m.degree_triples[triple * 3 + 1], degree3 = m.degree_triples[triple * 3 + 2];
        const int rank1 = m.bispectrum_ranks[degree1 - 1], rank2 = m.bispectrum_ranks[degree2 - 1], rank3 = m.bispectrum_ranks[degree3 - 1];
        const int dim1 = 2 * degree1 + 1, dim2 = 2 * degree2 + 1, dim3 = 2 * degree3 + 1;
        const int proj1 = projected_offset_for_degree(m, degree1);
        const int proj2 = projected_offset_for_degree(m, degree2);
        const int proj3 = projected_offset_for_degree(m, degree3);
        const std::int64_t probe_begin = m.probe_offsets[triple], probe_end = m.probe_offsets[triple + 1];
        const int full_count = rank1 * rank2 * rank3;
        for (int index = 0; index < full_count; ++index) full[index] = 0.0F;
        if (degree1 == 1 && degree2 == 1 && degree3 == 2) {
            packed_l2_to_stf(projected + proj3, rank3, matrices);
            for (int first = 0; first < rank1; ++first) for (int second = 0; second < rank2; ++second) for (int tensor = 0; tensor < rank3; ++tensor) {
                const float* matrix = matrices + tensor * 9;
                const float vx = projected[proj1 + first], vy = projected[proj1 + rank1 + first], vz = projected[proj1 + 2 * rank1 + first];
                const float wx = projected[proj2 + second], wy = projected[proj2 + rank2 + second], wz = projected[proj2 + 2 * rank2 + second];
                const double mx = static_cast<double>(matrix[0]) * wx + static_cast<double>(matrix[1]) * wy + static_cast<double>(matrix[2]) * wz;
                const double my = static_cast<double>(matrix[3]) * wx + static_cast<double>(matrix[4]) * wy + static_cast<double>(matrix[5]) * wz;
                const double mz = static_cast<double>(matrix[6]) * wx + static_cast<double>(matrix[7]) * wy + static_cast<double>(matrix[8]) * wz;
                full[(first * rank2 + second) * rank3 + tensor] = static_cast<float>(-(vx * mx + vy * my + vz * mz) / kSqrt5);
            }
        } else {
            const float* coupling = m.coupling + m.coupling_offsets[triple];
            for (int first = 0; first < rank1; ++first) for (int second = 0; second < rank2; ++second) for (int third = 0; third < rank3; ++third) {
                double value = 0.0;
                for (int i = 0; i < dim1; ++i) for (int j = 0; j < dim2; ++j) for (int k = 0; k < dim3; ++k)
                    value += static_cast<double>(coupling[(i * dim2 + j) * dim3 + k]) * projected[proj1 + i * rank1 + first] * projected[proj2 + j * rank2 + second] * projected[proj3 + k * rank3 + third];
                full[(first * rank2 + second) * rank3 + third] = static_cast<float>(value);
            }
        }
        for (std::int64_t probe = probe_begin; probe < probe_end; ++probe) descriptor[descriptor_offset++] = full[m.probe_index[probe]] * m.probe_scale[probe];
    }
    const int vector_rank = m.bispectrum_ranks[0], tensor_rank = m.bispectrum_ranks[1];
    const int tensor_offset = projected_offset_for_degree(m, 2);
    packed_l2_to_stf(projected + tensor_offset, tensor_rank, matrices);
    for (int tensor = 0; tensor < tensor_rank; ++tensor) for (int vector = 0; vector < vector_rank; ++vector) {
        const float* matrix = matrices + tensor * 9;
        const float vx = projected[vector], vy = projected[vector_rank + vector], vz = projected[2 * vector_rank + vector];
        const double wx = static_cast<double>(matrix[0]) * vx + static_cast<double>(matrix[1]) * vy + static_cast<double>(matrix[2]) * vz;
        const double wy = static_cast<double>(matrix[3]) * vx + static_cast<double>(matrix[4]) * vy + static_cast<double>(matrix[5]) * vz;
        const double wz = static_cast<double>(matrix[6]) * vx + static_cast<double>(matrix[7]) * vy + static_cast<double>(matrix[8]) * vz;
        descriptor[descriptor_offset++] = static_cast<float>(wx * wx + wy * wy + wz * wz);
    }
    descriptor[descriptor_offset++] = static_cast<float>(divisor_scalar);
    descriptor[descriptor_offset++] = static_cast<float>(divisor_angular);
    for (int channel = 0; channel < m.channels; ++channel) descriptor[descriptor_offset++] = m.type_embedding[center_type * m.channels + channel];
    for (std::int64_t feature = 0; feature < m.feature_count; ++feature) {
        double value = descriptor[feature];
        if (m.calibrate) value = (value - m.output_mean[feature]) / m.output_stddev[feature];
        output[center * m.feature_count + feature] = value;
    }
}

} // namespace

struct DeviceDpa4cModel::Layout : Dpa4cCudaLayout {};

DeviceDpa4cModel::DeviceDpa4cModel(CudaExecutionContext& context, py::dict payload) {
    const py::str model_key("model");
    const py::dict model = payload.contains(model_key)
        ? py::cast<py::dict>(payload[model_key])
        : payload;
    HostPayload p = parse_payload(model);
    // The public Python handoff keeps the model tensors nested and carries the
    // atomic-number map beside them.  Keep parsing model-specific fields
    // isolated from that envelope, then attach the validated map used for the
    // device-side type lookup.
    if (p.type_numbers.empty() && payload.contains("type_numbers")) {
        p.type_numbers = payload_array<std::int32_t>(
            payload["type_numbers"], "type_numbers");
    }
    device_ = context.device(); rcut_ = p.rcut; ntypes_ = p.ntypes; channels_ = p.channels; lmax_ = p.lmax;
    n_radial_ = p.n_radial; radial_modes_ = p.radial_modes; radial_hidden_ = p.radial_hidden;
    pair_hidden_ = p.pair_hidden; calibrate_ = p.calibrate; degree_channels_ = p.degree_channels;
    bispectrum_ranks_ = p.bispectrum_ranks; type_numbers_ = p.type_numbers;
    host_type_lookup_.assign(119, -1);
    for (int index = 0; index < static_cast<int>(type_numbers_.size()); ++index) {
        const int number = type_numbers_[index];
        if (number <= 0 || number >= static_cast<int>(host_type_lookup_.size()) || host_type_lookup_[number] >= 0)
            throw std::invalid_argument("DPA4C CUDA type_numbers contains an invalid or duplicate atomic number");
        host_type_lookup_[number] = index;
    }
    degree_offsets_.assign(static_cast<std::size_t>(lmax_ + 2), 0);
    for (int degree = 0; degree <= lmax_; ++degree) degree_offsets_[degree + 1] = degree_offsets_[degree] + (2 * degree + 1) * degree_channels_[degree];
    moment_count_ = degree_offsets_.back();
    gram_offsets_.push_back(0);
    for (int degree = 1; degree <= lmax_; ++degree) {
        const int width = degree_channels_[degree];
        for (int row = 0; row < width; ++row) for (int column = row; column < width; ++column) {
            gram_index_.push_back(row * width + column); gram_scale_.push_back(row == column ? 1.0F : kSqrt2);
        }
        gram_offsets_.push_back(static_cast<std::int64_t>(gram_index_.size()));
    }
    feature_count_ = static_cast<std::int64_t>(p.output_mean.size()); triple_count_ = static_cast<std::int64_t>(p.degree_triples.size() / 3);
    std::int64_t max_full = 0, projected_count = 0, max_block = 0;
    for (int degree = 1; degree <= lmax_; ++degree) {
        projected_count += (2 * degree + 1) * bispectrum_ranks_[degree - 1];
        max_block = std::max<std::int64_t>(max_block, degree_offsets_[degree + 1] - degree_offsets_[degree]);
    }
    for (std::int64_t triple = 0; triple < triple_count_; ++triple) {
        max_full = std::max(max_full, static_cast<std::int64_t>(bispectrum_ranks_[p.degree_triples[triple * 3] - 1]) * bispectrum_ranks_[p.degree_triples[triple * 3 + 1] - 1] * bispectrum_ranks_[p.degree_triples[triple * 3 + 2] - 1]);
    }
    auto layout = std::make_unique<Layout>();
    std::size_t bytes = 0;
    auto reserve = [&](std::size_t size, std::size_t alignment) { bytes = align_bytes(bytes, alignment); const std::size_t result = bytes; bytes += size; return static_cast<std::int64_t>(result); };
    layout->reduced = reserve(static_cast<std::size_t>(2 + moment_count_) * sizeof(double), alignof(double));
    layout->radial_basis = reserve(static_cast<std::size_t>(n_radial_) * sizeof(float), alignof(float));
    layout->radial_pre = reserve(static_cast<std::size_t>(2 * radial_hidden_) * sizeof(float), alignof(float));
    layout->radial_hidden = reserve(static_cast<std::size_t>(radial_hidden_) * sizeof(float), alignof(float));
    layout->radial = reserve(static_cast<std::size_t>(channels_) * sizeof(float), alignof(float));
    layout->modes = reserve(static_cast<std::size_t>(radial_modes_) * sizeof(float), alignof(float));
    layout->basis = reserve(25 * sizeof(float), alignof(float));
    layout->amplitudes = reserve(static_cast<std::size_t>(channels_) * sizeof(double), alignof(double));
    layout->moments = reserve(static_cast<std::size_t>(moment_count_) * sizeof(float), alignof(float));
    layout->blocks = reserve(static_cast<std::size_t>(moment_count_) * sizeof(float), alignof(float));
    layout->projected = reserve(static_cast<std::size_t>(projected_count) * sizeof(float), alignof(float));
    layout->descriptor = reserve(static_cast<std::size_t>(feature_count_) * sizeof(float), alignof(float));
    layout->full = reserve(static_cast<std::size_t>(std::max<std::int64_t>(1, std::max(max_full, max_block))) * sizeof(float), alignof(float));
    layout->matrices = reserve(static_cast<std::size_t>(std::max(1, bispectrum_ranks_[1]) * 9) * sizeof(float), alignof(float));
    bytes = align_bytes(bytes, alignof(double)); layout->fixed_bytes = static_cast<std::int64_t>(bytes);
    layout_ = std::move(layout);

    const std::size_t type_rows = static_cast<std::size_t>(ntypes_ + 1);
    std::vector<float> pair_scale(type_rows * type_rows * channels_, 0.0F), pair_shift(pair_scale.size(), 0.0F), pair_mixing(type_rows * type_rows * channels_ * radial_modes_, 0.0F);
    std::vector<float> input(static_cast<std::size_t>(2 * channels_)), pre(static_cast<std::size_t>(2 * pair_hidden_)), hidden(static_cast<std::size_t>(pair_hidden_)), logits(static_cast<std::size_t>(channels_ * (2 + radial_modes_)));
    for (int center = 0; center < ntypes_; ++center) for (int neighbor = 0; neighbor < ntypes_; ++neighbor) {
        for (int channel = 0; channel < channels_; ++channel) { input[channel] = p.type_embedding[center * channels_ + channel]; input[channels_ + channel] = p.type_embedding[neighbor * channels_ + channel]; }
        for (int out = 0; out < 2 * pair_hidden_; ++out) { double value = 0.0; for (int in = 0; in < 2 * channels_; ++in) value += static_cast<double>(input[in]) * p.pair_w0[in * (2 * pair_hidden_) + out]; pre[out] = static_cast<float>(value); }
        for (int index = 0; index < pair_hidden_; ++index) hidden[index] = pre[index] * (1.0F / (1.0F + std::exp(-pre[index]))) * pre[pair_hidden_ + index];
        for (int out = 0; out < channels_ * (2 + radial_modes_); ++out) { double value = 0.0; for (int in = 0; in < pair_hidden_; ++in) value += static_cast<double>(hidden[in]) * p.pair_w1[in * channels_ * (2 + radial_modes_) + out]; logits[out] = 0.1F * static_cast<float>(value); }
        const std::size_t pair = static_cast<std::size_t>(center * (ntypes_ + 1) + neighbor);
        for (int channel = 0; channel < channels_; ++channel) {
            pair_scale[pair * channels_ + channel] = 1.0F + std::tanh(logits[channel]);
            pair_shift[pair * channels_ + channel] = p.type_embedding[center * channels_ + channel] + p.type_embedding[neighbor * channels_ + channel] + std::tanh(logits[channels_ + channel]);
            for (int mode = 0; mode < radial_modes_; ++mode) pair_mixing[(pair * channels_ + channel) * radial_modes_ + mode] = std::tanh(logits[2 * channels_ + channel * radial_modes_ + mode]);
        }
    }
    try {
        type_embedding_ = upload_array(context, p.type_embedding, "could not upload DPA4C type embedding");
        degree_channels_device_ = upload_array(context, p.degree_channels, "could not upload DPA4C degree channels");
        bispectrum_ranks_device_ = upload_array(context, p.bispectrum_ranks, "could not upload DPA4C bispectrum ranks");
        radial_freqs_ = upload_array(context, p.radial_freqs, "could not upload DPA4C radial frequencies");
        radial_w0_ = upload_array(context, p.radial_w0, "could not upload DPA4C radial first layer");
        radial_w1_ = upload_array(context, p.radial_w1, "could not upload DPA4C radial output layer");
        radial_mode_w_ = upload_array(context, p.radial_mode_w, "could not upload DPA4C radial mode layer");
        pair_scale_ = upload_array(context, pair_scale, "could not upload DPA4C pair scales");
        pair_shift_ = upload_array(context, pair_shift, "could not upload DPA4C pair shifts");
        pair_mixing_ = upload_array(context, pair_mixing, "could not upload DPA4C pair mixing");
        alignment_ = upload_array(context, p.alignment, "could not upload DPA4C alignment");
        alignment_offsets_ = upload_array(context, p.alignment_offsets, "could not upload DPA4C alignment offsets");
        projections_ = upload_array(context, p.projections, "could not upload DPA4C projections");
        projection_offsets_ = upload_array(context, p.projection_offsets, "could not upload DPA4C projection offsets");
        coupling_ = upload_array(context, p.coupling, "could not upload DPA4C coupling");
        coupling_offsets_ = upload_array(context, p.coupling_offsets, "could not upload DPA4C coupling offsets");
        degree_triples_ = upload_array(context, p.degree_triples, "could not upload DPA4C degree triples");
        probe_offsets_ = upload_array(context, p.probe_offsets, "could not upload DPA4C probe offsets");
        probe_index_ = upload_array(context, p.probe_index, "could not upload DPA4C probe indices");
        probe_scale_ = upload_array(context, p.probe_scale, "could not upload DPA4C probe scales");
        output_mean_ = upload_array(context, p.output_mean, "could not upload DPA4C output means");
        output_stddev_ = upload_array(context, p.output_stddev, "could not upload DPA4C output standard deviations");
    } catch (...) { release(); throw; }
}

DeviceDpa4cModel::~DeviceDpa4cModel() noexcept { release(); }

void DeviceDpa4cModel::release() noexcept {
    type_embedding_.reset(); degree_channels_device_.reset(); bispectrum_ranks_device_.reset(); radial_freqs_.reset(); radial_w0_.reset(); radial_w1_.reset(); radial_mode_w_.reset();
    pair_scale_.reset(); pair_shift_.reset(); pair_mixing_.reset(); alignment_.reset(); alignment_offsets_.reset();
    projections_.reset(); projection_offsets_.reset(); coupling_.reset(); coupling_offsets_.reset(); degree_triples_.reset();
    probe_offsets_.reset(); probe_index_.reset(); probe_scale_.reset(); output_mean_.reset(); output_stddev_.reset();
    layout_.reset();
}

int DeviceDpa4cModel::type_index_for_number(std::int32_t number) const noexcept {
    return number > 0 && number < static_cast<std::int32_t>(host_type_lookup_.size())
        ? host_type_lookup_[static_cast<std::size_t>(number)] : -1;
}

std::vector<double> DeviceDpa4cModel::compute(
    CudaExecutionContext& context, const DeviceBatch& batch,
    const DeviceNeighborGraph& graph, const std::vector<std::int32_t>& type_indices) const {
    if (context.device() != device_) throw std::invalid_argument("DPA4C CUDA model and execution context use different devices");
    if (batch.atoms() < 0) throw std::invalid_argument("DPA4C CUDA received an invalid batch");
    if (type_indices.size() != static_cast<std::size_t>(batch.atoms())) throw std::invalid_argument("DPA4C CUDA type_indices must have one entry per atom");
    for (std::int32_t value : type_indices) if (value < 0 || value >= ntypes_) throw std::invalid_argument("DPA4C CUDA type index is outside the checkpoint type map");
    if (batch.atoms() == 0) return {};
    if (graph.offsets() == nullptr) throw std::invalid_argument("DPA4C CUDA received an invalid neighbor graph");
    // The graph rows are already ordered by DeviceNeighborGraph::build_dpa;
    // the per-atom workspace therefore only needs the fixed descriptor state.
    const std::size_t fixed_bytes = static_cast<std::size_t>(layout_->fixed_bytes);
    const std::size_t stride = align_bytes(fixed_bytes, alignof(double));
    if (stride == 0 || static_cast<std::size_t>(batch.atoms()) > std::numeric_limits<std::size_t>::max() / stride)
        throw CudaOutOfMemory("DPA4C CUDA workspace is too large");
    Dpa4cCudaLayout layout = *layout_;
    layout.stride = static_cast<std::int64_t>(stride);
    const std::size_t atom_count = static_cast<std::size_t>(batch.atoms());
    const std::size_t feature_count = static_cast<std::size_t>(feature_count_);
    if (feature_count != 0 && atom_count > std::numeric_limits<std::size_t>::max() / feature_count)
        throw CudaOutOfMemory("DPA4C CUDA output is too large");
    const std::size_t output_count = atom_count * feature_count;
    double* output = context.output_buffer(output_count);
    DeviceArray type_indices_device;
    check_cuda(cudaSetDevice(context.device()), "could not select the CUDA device for DPA4C compute");
    check_cuda(cudaMalloc(&type_indices_device.pointer, type_indices.size() * sizeof(std::int32_t)), "could not allocate DPA4C CUDA type indices");
    auto* device_types = static_cast<std::int32_t*>(type_indices_device.pointer);
    check_cuda(cudaMemcpyAsync(device_types, type_indices.data(), type_indices.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice, context.stream()), "could not upload DPA4C CUDA type indices");
    KernelModel model{
        0.0, ntypes_, channels_, lmax_, n_radial_, radial_modes_, radial_hidden_, pair_hidden_, calibrate_, feature_count_, moment_count_, triple_count_,
        device_data<float>(type_embedding_), device_data<float>(radial_freqs_), device_data<float>(radial_w0_), device_data<float>(radial_w1_), device_data<float>(radial_mode_w_),
        device_data<float>(pair_scale_), device_data<float>(pair_shift_), device_data<float>(pair_mixing_), device_data<float>(alignment_), device_data<std::int64_t>(alignment_offsets_),
        device_data<float>(projections_), device_data<std::int64_t>(projection_offsets_), device_data<float>(coupling_), device_data<std::int64_t>(coupling_offsets_),
        device_data<int>(degree_triples_), device_data<std::int64_t>(probe_offsets_), device_data<std::int64_t>(probe_index_), device_data<float>(probe_scale_),
        device_data<float>(output_mean_), device_data<float>(output_stddev_), nullptr, nullptr,
        device_data<int>(degree_channels_device_), device_data<int>(bispectrum_ranks_device_)};
    model.rcut = rcut_;
    // The compact gram metadata is copied for this launch through the context
    // workspace tail; it is small and avoids another model-owned allocation.
    std::vector<std::int32_t> gram_index = gram_index_;
    std::vector<float> gram_scale = gram_scale_;
    const std::size_t metadata_bytes = gram_index.size() * sizeof(std::int32_t) + gram_scale.size() * sizeof(float);
    if (metadata_bytes > std::numeric_limits<std::size_t>::max() - stride * static_cast<std::size_t>(batch.atoms()))
        throw CudaOutOfMemory("DPA4C CUDA metadata is too large");
    auto* workspace = static_cast<unsigned char*>(context.workspace_buffer(
        stride * static_cast<std::size_t>(batch.atoms()) + metadata_bytes));
    auto* device_gram_index = reinterpret_cast<std::int32_t*>(workspace + stride * static_cast<std::size_t>(batch.atoms()));
    auto* device_gram_scale = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(device_gram_index) + gram_index.size() * sizeof(std::int32_t));
    check_cuda(cudaMemcpyAsync(device_gram_index, gram_index.data(), gram_index.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice, context.stream()), "could not upload DPA4C gram indices");
    check_cuda(cudaMemcpyAsync(device_gram_scale, gram_scale.data(), gram_scale.size() * sizeof(float), cudaMemcpyHostToDevice, context.stream()), "could not upload DPA4C gram scales");
    model.gram_index = device_gram_index; model.gram_scale = device_gram_scale;
    const auto blocks = static_cast<unsigned int>((static_cast<std::size_t>(batch.atoms()) + 127) / 128);
    dpa4c_kernel<<<blocks, 128, 0, context.stream()>>>(
        graph.offsets(), graph.atoms(), graph.shifts(), graph.displacements(), device_types,
        batch.atoms(), workspace, layout, model, output);
    check_cuda(cudaGetLastError(), "DPA4C CUDA descriptor kernel launch failed");
    std::vector<double> result = context.download_output(output_count);
    return result;
}

} // namespace mdescriptor::cuda
