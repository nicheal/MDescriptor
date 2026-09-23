#include "mdescriptor/cuda/dpa4c.hpp"
#include "mdescriptor/cuda/error.hpp"

#include "dpa4_common.cuh"
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

struct DeviceDpa4cModel::DeviceArray : mdescriptor::cuda::DeviceArray {};

namespace {

constexpr float kSqrt2 = 1.41421356237309504880F;
constexpr float kSqrt3 = 1.73205080756887729353F;
constexpr float kSqrt5 = 2.23606797749978969641F;
constexpr float kSqrt6 = 2.44948974278317809820F;
constexpr float kEpsilon = 1.0e-7F;
constexpr float kNormFloor = 0.25F;

template <typename Value>
std::vector<Value> payload_array(py::handle value, const char* name) {
    const std::string error = std::string("DPA4C ") + name + " must be a numeric array";
    return dpa4_common::payload_array<Value>(value, name, error.c_str(), false);
}

py::handle required(const py::dict& payload, const char* name) {
    return dpa4_common::required(payload, name, "DPA4C");
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
    std::vector<int> fitting_neurons;
    std::vector<float> fitting_weights;
    std::vector<float> fitting_biases;
    std::vector<double> fitting_atom_bias;
    std::vector<double> output_bias;
    bool has_fitting = false;
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
    if (payload.contains("fitting_neurons")) {
        p.fitting_neurons = py::cast<std::vector<int>>(payload["fitting_neurons"]);
        p.fitting_weights = payload_array<float>(
            required(payload, "fitting_weights"), "fitting_weights");
        p.fitting_biases = payload_array<float>(
            required(payload, "fitting_biases"), "fitting_biases");
        p.fitting_atom_bias = payload_array<double>(
            required(payload, "fitting_atom_bias"), "fitting_atom_bias");
        p.output_bias = payload_array<double>(
            required(payload, "output_bias"), "output_bias");
        const std::string activation = py::cast<std::string>(
            required(payload, "fitting_activation"));
        if (activation != "silu") {
            throw std::invalid_argument("DPA4C CUDA supports only SiLU energy fitting");
        }
        p.has_fitting = true;
    }
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
    if (p.has_fitting) {
        if (p.fitting_atom_bias.size() != static_cast<std::size_t>(p.ntypes)
            || p.output_bias.size() != static_cast<std::size_t>(p.ntypes)) {
            throw std::invalid_argument("DPA4C CUDA fitting type arrays have unexpected shape");
        }
        std::size_t weight_count = 0;
        std::size_t bias_count = 1;
        std::size_t width = static_cast<std::size_t>(expected_features);
        for (int next_width : p.fitting_neurons) {
            if (next_width <= 0) {
                throw std::invalid_argument("DPA4C CUDA fitting widths must be positive");
            }
            weight_count += width * static_cast<std::size_t>(next_width);
            bias_count += static_cast<std::size_t>(next_width);
            width = static_cast<std::size_t>(next_width);
        }
        weight_count += width;
        expect_size(p.fitting_weights, weight_count, "fitting_weights");
        expect_size(p.fitting_biases, bias_count, "fitting_biases");
    }
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
    std::int64_t fitting_activations = 0;
    std::int64_t fitting_pre = 0;
    std::int64_t fitting_scratch = 0;
    std::int64_t feature_gradient = 0;
    std::int64_t block_gradient = 0;
    std::int64_t projected_gradient = 0;
    std::int64_t readout_scratch = 0;
    std::int64_t scalar_adjoint = 0;
    std::int64_t angular_adjoint = 0;
    std::int64_t divisor_adjoint = 0;
    std::int64_t atom_energy = 0;
};

using dpa4_common::align_bytes;

template <typename Value>
std::unique_ptr<DeviceDpa4cModel::DeviceArray> upload_array(
    CudaExecutionContext& context, const std::vector<Value>& values, const char* operation) {
    return dpa4_common::upload_array<DeviceDpa4cModel::DeviceArray>(
        context, values, operation, "could not select the CUDA device for DPA4C",
        false);
}

template <typename Value>
Value* device_data(const std::unique_ptr<DeviceDpa4cModel::DeviceArray>& value) {
    return dpa4_common::device_data<Value>(value);
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
    int fitting_layer_count;
    int fitting_max_width;
    const int* fitting_neurons;
    const std::int64_t* fitting_activation_offsets;
    const float* fitting_weights;
    const float* fitting_biases;
    const double* fitting_atom_bias;
    const double* output_bias;
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

// The projection widths are encoded by the per-degree offset arrays.
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

__device__ __forceinline__ void packed_l2_tensor_to_stf(
    const float* packed, int rank, int tensor, float* matrix) {
    const float q0 = packed[tensor];
    const float q1 = packed[rank + tensor];
    const float q2 = packed[2 * rank + tensor];
    const float q3 = packed[3 * rank + tensor];
    const float q4 = packed[4 * rank + tensor];
    const float qxy = q0 / kSqrt2;
    const float qyz = q1 / kSqrt2;
    const float qxz = q3 / kSqrt2;
    matrix[0] = -q2 / kSqrt6 + q4 / kSqrt2;
    matrix[1] = qxy;
    matrix[2] = qxz;
    matrix[3] = qxy;
    matrix[4] = -q2 / kSqrt6 - q4 / kSqrt2;
    matrix[5] = qyz;
    matrix[6] = qxz;
    matrix[7] = qyz;
    matrix[8] = 2.0F * q2 / kSqrt6;
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
        if (exact_self_edge(center, neighbor, graph_shifts, edge)) continue;
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
    if (output != nullptr) {
        for (std::int64_t feature = 0; feature < m.feature_count; ++feature) {
            double value = descriptor[feature];
            if (m.calibrate) value = (value - m.output_mean[feature]) / m.output_stddev[feature];
            output[center * m.feature_count + feature] = value;
        }
    }
}

__device__ __forceinline__ float silu(float value) {
    return value / (1.0F + expf(-value));
}

__device__ __forceinline__ float silu_derivative(float value) {
    const float gate = 1.0F / (1.0F + expf(-value));
    return gate * (1.0F + value * (1.0F - gate));
}

__device__ __forceinline__ int projected_count(const KernelModel& m) {
    int count = 0;
    for (int degree = 1; degree <= m.lmax; ++degree)
        count += (2 * degree + 1) * m.bispectrum_ranks[degree - 1];
    return count;
}

__device__ void packed_stf_gradient_one(
    const double* matrix, int tensor, int rank, double* packed_gradient) {
    packed_gradient[tensor] += (matrix[1] + matrix[3]) / kSqrt2;
    packed_gradient[rank + tensor] += (matrix[5] + matrix[7]) / kSqrt2;
    packed_gradient[2 * rank + tensor] +=
        (-matrix[0] - matrix[4] + 2.0 * matrix[8]) / sqrt(6.0);
    packed_gradient[3 * rank + tensor] += (matrix[2] + matrix[6]) / kSqrt2;
    packed_gradient[4 * rank + tensor] += (matrix[0] - matrix[4]) / kSqrt2;
}

__device__ void dpa4c_readout_backward(
    const KernelModel& m,
    unsigned char* base,
    const Dpa4cCudaLayout& layout) {
    const auto* reduced = reinterpret_cast<const double*>(base + layout.reduced);
    const auto* blocks = reinterpret_cast<const float*>(base + layout.blocks);
    const auto* projected = reinterpret_cast<const float*>(base + layout.projected);
    const auto* feature_gradient = reinterpret_cast<const double*>(base + layout.feature_gradient);
    auto* block_gradient = reinterpret_cast<double*>(base + layout.block_gradient);
    auto* projected_gradient = reinterpret_cast<double*>(base + layout.projected_gradient);
    auto* scratch = reinterpret_cast<double*>(base + layout.readout_scratch);
    auto* scalar_adjoint = reinterpret_cast<double*>(base + layout.scalar_adjoint);
    auto* angular_adjoint = reinterpret_cast<double*>(base + layout.angular_adjoint);
    auto* divisor_adjoint = reinterpret_cast<double*>(base + layout.divisor_adjoint);
    divisor_adjoint[0] = 0.0;
    divisor_adjoint[1] = 0.0;
    const int pcount = projected_count(m);
    for (int index = 0; index < m.moment_count; ++index) block_gradient[index] = 0.0;
    for (int index = 0; index < pcount; ++index) projected_gradient[index] = 0.0;

    int feature = 0;
    for (int channel = 0; channel < m.channels; ++channel)
        block_gradient[channel] = feature_gradient[feature++];
    int gram_cursor = 0;
    for (int degree = 1; degree <= m.lmax; ++degree) {
        const int width = m.degree_channels[degree];
        const int dim = 2 * degree + 1;
        const int offset = moment_offset_for_degree(m, degree);
        const int gram_count = width * (width + 1) / 2;
        for (int gram = 0; gram < gram_count; ++gram) {
            const int flat = m.gram_index[gram_cursor];
            const int row = flat / width;
            const int column = flat % width;
            const double gradient = feature_gradient[feature++]
                * m.gram_scale[gram_cursor];
            for (int component = 0; component < dim; ++component) {
                const int row_index = offset + component * width + row;
                const int column_index = offset + component * width + column;
                block_gradient[row_index] += gradient * blocks[column_index]
                    * (row == column ? 2.0 : 1.0);
                if (row != column)
                    block_gradient[column_index] += gradient * blocks[row_index];
            }
            ++gram_cursor;
        }
    }

    for (int triple = 0; triple < m.triple_count; ++triple) {
        const int degree1 = m.degree_triples[triple * 3];
        const int degree2 = m.degree_triples[triple * 3 + 1];
        const int degree3 = m.degree_triples[triple * 3 + 2];
        const int rank1 = m.bispectrum_ranks[degree1 - 1];
        const int rank2 = m.bispectrum_ranks[degree2 - 1];
        const int rank3 = m.bispectrum_ranks[degree3 - 1];
        const int dim1 = 2 * degree1 + 1;
        const int dim2 = 2 * degree2 + 1;
        const int dim3 = 2 * degree3 + 1;
        const int proj1 = projected_offset_for_degree(m, degree1);
        const int proj2 = projected_offset_for_degree(m, degree2);
        const int proj3 = projected_offset_for_degree(m, degree3);
        const float* coupling = m.coupling + m.coupling_offsets[triple];
        for (std::int64_t probe = m.probe_offsets[triple];
             probe < m.probe_offsets[triple + 1]; ++probe) {
            const int index = static_cast<int>(m.probe_index[probe]);
            const int third = index % rank3;
            const int second = (index / rank3) % rank2;
            const int first = index / (rank2 * rank3);
            const double gradient = feature_gradient[feature++] * m.probe_scale[probe];
            if (degree1 == 1 && degree2 == 1 && degree3 == 2) {
                float matrix[9];
                packed_l2_tensor_to_stf(projected + proj3, rank3, third, matrix);
                const double left[3] = {
                    projected[proj1 + first],
                    projected[proj1 + rank1 + first],
                    projected[proj1 + 2 * rank1 + first]};
                const double right[3] = {
                    projected[proj2 + second],
                    projected[proj2 + rank2 + second],
                    projected[proj2 + 2 * rank2 + second]};
                const double scale = -gradient / kSqrt5;
                double matrix_gradient[9] = {};
                for (int row = 0; row < 3; ++row) {
                    double left_grad = 0.0;
                    double right_grad = 0.0;
                    for (int column = 0; column < 3; ++column) {
                        left_grad += static_cast<double>(matrix[row * 3 + column]) * right[column];
                        right_grad += static_cast<double>(matrix[column * 3 + row]) * left[column];
                        matrix_gradient[row * 3 + column] += scale * left[row] * right[column];
                    }
                    projected_gradient[proj1 + row * rank1 + first] += scale * left_grad;
                    projected_gradient[proj2 + row * rank2 + second] += scale * right_grad;
                }
                packed_stf_gradient_one(matrix_gradient, third, rank3,
                    projected_gradient + proj3);
            } else {
                for (int i = 0; i < dim1; ++i) {
                    const double left = projected[proj1 + i * rank1 + first];
                    for (int j = 0; j < dim2; ++j) {
                        const double middle = projected[proj2 + j * rank2 + second];
                        for (int k = 0; k < dim3; ++k) {
                            const double right = projected[proj3 + k * rank3 + third];
                            const double weight = coupling[(i * dim2 + j) * dim3 + k];
                            projected_gradient[proj1 + i * rank1 + first] +=
                                gradient * weight * middle * right;
                            projected_gradient[proj2 + j * rank2 + second] +=
                                gradient * weight * left * right;
                            projected_gradient[proj3 + k * rank3 + third] +=
                                gradient * weight * left * middle;
                        }
                    }
                }
            }
        }
    }

    const int vector_rank = m.bispectrum_ranks[0];
    const int tensor_rank = m.bispectrum_ranks[1];
    const int tensor_offset = projected_offset_for_degree(m, 2);
    for (int tensor = 0; tensor < tensor_rank; ++tensor) {
        float matrix[9];
        packed_l2_tensor_to_stf(projected + tensor_offset, tensor_rank, tensor, matrix);
        double tensor_matrix_gradient[9] = {};
        for (int vector = 0; vector < vector_rank; ++vector) {
            const double v[3] = {
                projected[vector], projected[vector_rank + vector],
                projected[2 * vector_rank + vector]};
            double qv[3] = {};
            for (int row = 0; row < 3; ++row)
                for (int column = 0; column < 3; ++column)
                    qv[row] += matrix[row * 3 + column] * v[column];
            const double gradient = feature_gradient[feature++];
            for (int column = 0; column < 3; ++column) {
                double q2v = 0.0;
                for (int row = 0; row < 3; ++row) {
                    q2v += matrix[row * 3 + column] * qv[row];
                    tensor_matrix_gradient[row * 3 + column] +=
                        2.0 * gradient * qv[row] * v[column];
                }
                projected_gradient[column * vector_rank + vector] +=
                    2.0 * gradient * q2v;
            }
        }
        packed_stf_gradient_one(tensor_matrix_gradient, tensor, tensor_rank,
            projected_gradient + tensor_offset);
    }

    const double direct_divisor_scalar = feature_gradient[feature++];
    const double direct_divisor_angular = feature_gradient[feature++];
    const double divisor_scalar = sqrt(reduced[0] + static_cast<double>(kNormFloor));
    const double divisor_angular = sqrt(reduced[1] + static_cast<double>(kNormFloor));
    const double divisor_scalar_squared = divisor_scalar * divisor_scalar;
    const double divisor_angular_squared = divisor_angular * divisor_angular;

    // Reverse the low-rank readout projections, then undo the learned degree
    // alignments.  The scalar block has no projection or alignment.
    for (int degree = 1; degree <= m.lmax; ++degree) {
        const int width = m.degree_channels[degree];
        const int rank = m.bispectrum_ranks[degree - 1];
        const int dim = 2 * degree + 1;
        const int block_offset = moment_offset_for_degree(m, degree);
        const int proj_offset = projected_offset_for_degree(m, degree);
        const std::int64_t projection_begin = m.projection_offsets[degree - 1];
        const std::int64_t projection_end = m.projection_offsets[degree];
        if (projection_begin == projection_end) {
            for (int index = 0; index < dim * width; ++index)
                block_gradient[block_offset + index] += projected_gradient[proj_offset + index];
        } else {
            const float* matrix = m.projections + projection_begin;
            for (int component = 0; component < dim; ++component)
                for (int input = 0; input < width; ++input) {
                    double value = 0.0;
                    for (int output = 0; output < rank; ++output)
                        value += projected_gradient[proj_offset + component * rank + output]
                            * matrix[input * rank + output];
                    block_gradient[block_offset + component * width + input] += value;
                }
        }
        if (degree == 1 || degree == 2) {
            const float* alignment = m.alignment + m.alignment_offsets[degree - 1];
            for (int component = 0; component < dim; ++component) {
                for (int input = 0; input < width; ++input) {
                    double value = block_gradient[block_offset + component * width + input];
                    for (int output = 0; output < width; ++output)
                        value += block_gradient[block_offset + component * width + output]
                            * alignment[input * width + output];
                    scratch[input] = value;
                }
                for (int channel = 0; channel < width; ++channel)
                    block_gradient[block_offset + component * width + channel] = scratch[channel];
            }
        }
    }

    for (int channel = 0; channel < m.channels; ++channel) {
        const double gradient = block_gradient[channel];
        scalar_adjoint[channel] = gradient / divisor_scalar;
        divisor_adjoint[0] -= gradient * reduced[2 + channel] / divisor_scalar_squared;
    }
    divisor_adjoint[0] += direct_divisor_scalar;
    for (int degree = 1; degree <= m.lmax; ++degree) {
        const int width = m.degree_channels[degree];
        const int dim = 2 * degree + 1;
        const int block_offset = moment_offset_for_degree(m, degree);
        for (int index = 0; index < dim * width; ++index) {
            const double gradient = block_gradient[block_offset + index];
            angular_adjoint[block_offset + index] = gradient / divisor_angular;
            divisor_adjoint[1] -= gradient * reduced[2 + block_offset + index]
                / divisor_angular_squared;
        }
    }
    divisor_adjoint[1] += direct_divisor_angular;
}

__global__ void dpa4c_fit_backward_kernel(
    std::int64_t atoms,
    const std::int32_t* type_indices,
    unsigned char* workspace,
    Dpa4cCudaLayout layout,
    KernelModel m) {
    const std::int64_t center = static_cast<std::int64_t>(blockIdx.x);
    if (center >= atoms) return;
    const int thread = threadIdx.x;
    unsigned char* base = workspace + center * layout.stride;
    auto* activation = reinterpret_cast<float*>(base + layout.fitting_activations);
    auto* pre = reinterpret_cast<float*>(base + layout.fitting_pre);
    auto* scratch = reinterpret_cast<float*>(base + layout.fitting_scratch);
    auto* feature_gradient = reinterpret_cast<double*>(base + layout.feature_gradient);
    const auto* descriptor = reinterpret_cast<const float*>(base + layout.descriptor);
    for (std::int64_t feature = thread; feature < m.feature_count; feature += blockDim.x) {
        double value = descriptor[feature];
        if (m.calibrate) value = (value - m.output_mean[feature]) / m.output_stddev[feature];
        activation[feature] = static_cast<float>(value);
    }
    __syncthreads();

    std::size_t weight_offset = 0;
    std::size_t bias_offset = 0;
    int previous_width = static_cast<int>(m.feature_count);
    for (int layer = 0; layer < m.fitting_layer_count; ++layer) {
        const int width = m.fitting_neurons[layer];
        const float* previous = activation + (layer == 0
            ? 0 : m.fitting_activation_offsets[layer - 1]);
        float* next = activation + m.fitting_activation_offsets[layer];
        float* layer_pre = pre + m.fitting_activation_offsets[layer] - m.feature_count;
        const float* weights = m.fitting_weights + weight_offset;
        const float* biases = m.fitting_biases + bias_offset;
        for (int out = thread; out < width; out += blockDim.x) {
            double value = biases[out];
            for (int in = 0; in < previous_width; ++in)
                value += static_cast<double>(previous[in]) * weights[in * width + out];
            layer_pre[out] = static_cast<float>(value);
            float activated = silu(layer_pre[out]);
            if (width == previous_width) activated += previous[out];
            next[out] = activated;
        }
        __syncthreads();
        weight_offset += static_cast<std::size_t>(previous_width) * width;
        bias_offset += static_cast<std::size_t>(width);
        previous_width = width;
    }

    const std::int64_t final_input_offset = m.fitting_layer_count == 0 ? 0
        : m.fitting_activation_offsets[m.fitting_layer_count - 1];
    const float* final_input = activation + final_input_offset;
    const float* final_weights = m.fitting_weights + weight_offset;
    if (thread == 0) {
        double raw_value = m.fitting_biases[bias_offset];
        for (int in = 0; in < previous_width; ++in)
            raw_value += static_cast<double>(final_input[in]) * final_weights[in];
        const int type = type_indices[center];
        const float energy_float = static_cast<float>(static_cast<float>(raw_value)
            + static_cast<float>(m.fitting_atom_bias[type]));
        *reinterpret_cast<double*>(base + layout.atom_energy) =
            static_cast<double>(energy_float) + m.output_bias[type];
    }

    float* gradient = scratch;
    float* previous_gradient = scratch + m.fitting_max_width;
    for (int in = thread; in < previous_width; in += blockDim.x)
        gradient[in] = final_weights[in];
    __syncthreads();
    for (int layer = m.fitting_layer_count - 1; layer >= 0; --layer) {
        const int width = m.fitting_neurons[layer];
        const int prior_width = layer == 0 ? static_cast<int>(m.feature_count)
            : m.fitting_neurons[layer - 1];
        std::int64_t weight_begin = 0;
        int in_width = static_cast<int>(m.feature_count);
        for (int index = 0; index < layer; ++index) {
            weight_begin += static_cast<std::int64_t>(in_width) * m.fitting_neurons[index];
            in_width = m.fitting_neurons[index];
        }
        const float* weights = m.fitting_weights + weight_begin;
        const float* layer_pre = pre + m.fitting_activation_offsets[layer] - m.feature_count;
        constexpr int kWarpSize = 32;
        const int lane = thread & (kWarpSize - 1);
        const int warp = thread / kWarpSize;
        const int warp_count = blockDim.x / kWarpSize;
        for (int in = warp; in < prior_width; in += warp_count) {
            double value = 0.0;
            for (int output_base = 0; output_base < width; output_base += kWarpSize) {
                const int out = output_base + lane;
                if (out < width) {
                    value += static_cast<double>(gradient[out])
                        * silu_derivative(layer_pre[out])
                        * weights[in * width + out];
                }
            }
            for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
                value += __shfl_down_sync(0xffffffffU, value, offset);
            if (lane == 0) {
                if (width == prior_width) value += gradient[in];
                previous_gradient[in] = static_cast<float>(value);
            }
        }
        __syncthreads();
        float* temporary = gradient;
        gradient = previous_gradient;
        previous_gradient = temporary;
    }
    for (std::int64_t feature = thread; feature < m.feature_count; feature += blockDim.x) {
        double value = gradient[feature];
        if (m.calibrate) value /= m.output_stddev[feature];
        feature_gradient[feature] = value;
    }
}

__global__ void dpa4c_readout_backward_kernel(
    std::int64_t atoms,
    unsigned char* workspace,
    Dpa4cCudaLayout layout,
    KernelModel m) {
    const std::int64_t center = static_cast<std::int64_t>(blockIdx.x);
    if (center < atoms && threadIdx.x == 0) {
        unsigned char* base = workspace + center * layout.stride;
        dpa4c_readout_backward(m, base, layout);
    }
}

struct DeviceDual3 {
    float value;
    float derivative[3];
};

__device__ __forceinline__ DeviceDual3 dual_constant(float value) {
    return {value, {0.0F, 0.0F, 0.0F}};
}

__device__ __forceinline__ DeviceDual3 operator+(
    DeviceDual3 lhs, DeviceDual3 rhs) {
    DeviceDual3 out{lhs.value + rhs.value, {}};
    for (int axis = 0; axis < 3; ++axis)
        out.derivative[axis] = lhs.derivative[axis] + rhs.derivative[axis];
    return out;
}

__device__ __forceinline__ DeviceDual3 operator-(
    DeviceDual3 lhs, DeviceDual3 rhs) {
    DeviceDual3 out{lhs.value - rhs.value, {}};
    for (int axis = 0; axis < 3; ++axis)
        out.derivative[axis] = lhs.derivative[axis] - rhs.derivative[axis];
    return out;
}

__device__ __forceinline__ DeviceDual3 operator*(
    DeviceDual3 lhs, DeviceDual3 rhs) {
    DeviceDual3 out{lhs.value * rhs.value, {}};
    for (int axis = 0; axis < 3; ++axis)
        out.derivative[axis] = lhs.derivative[axis] * rhs.value
            + lhs.value * rhs.derivative[axis];
    return out;
}

__device__ __forceinline__ DeviceDual3 operator*(
    DeviceDual3 lhs, float rhs) { return lhs * dual_constant(rhs); }

__device__ __forceinline__ DeviceDual3 operator/(
    DeviceDual3 lhs, DeviceDual3 rhs) {
    DeviceDual3 out{lhs.value / rhs.value, {}};
    const float scale = 1.0F / (rhs.value * rhs.value);
    for (int axis = 0; axis < 3; ++axis)
        out.derivative[axis] = (lhs.derivative[axis] * rhs.value
            - lhs.value * rhs.derivative[axis]) * scale;
    return out;
}

__device__ __forceinline__ DeviceDual3 dual_sqrt(DeviceDual3 value) {
    DeviceDual3 out{sqrtf(value.value), {}};
    const float scale = 0.5F / out.value;
    for (int axis = 0; axis < 3; ++axis)
        out.derivative[axis] = value.derivative[axis] * scale;
    return out;
}

__device__ void angular_basis_dual(
    DeviceDual3 x, DeviceDual3 y, DeviceDual3 z, int lmax,
    DeviceDual3* result) {
    const DeviceDual3 norm2 = x * x + y * y + z * z;
    const DeviceDual3 x2 = x * x, y2 = y * y, z2 = z * z;
    for (int index = 0; index < 25; ++index) result[index] = dual_constant(0.0F);
    result[0] = dual_constant(1.0F);
    if (lmax >= 1) { result[1] = x; result[2] = y; result[3] = z; }
    if (lmax >= 2) {
        result[4] = (x * y) * kSqrt3;
        result[5] = (y * z) * kSqrt3;
        result[6] = (z2 * 3.0F - norm2) * 0.5F;
        result[7] = (x * z) * kSqrt3;
        result[8] = (x2 - y2) * (0.5F * kSqrt3);
    }
    if (lmax >= 3) {
        result[9] = y * (x2 * 3.0F - y2) * sqrtf(5.0F / 8.0F);
        result[10] = (x * y * z) * sqrtf(15.0F);
        result[11] = y * (z2 * 5.0F - norm2) * sqrtf(3.0F / 8.0F);
        result[12] = z * (z2 * 5.0F - norm2 * 3.0F) * 0.5F;
        result[13] = x * (z2 * 5.0F - norm2) * sqrtf(3.0F / 8.0F);
        result[14] = z * (x2 - y2) * (0.5F * sqrtf(15.0F));
        result[15] = x * (x2 - y2 * 3.0F) * sqrtf(5.0F / 8.0F);
    }
    if (lmax >= 4) {
        const DeviceDual3 difference = x2 - y2;
        const DeviceDual3 z4 = z2 * z2;
        const DeviceDual3 norm4 = norm2 * norm2;
        result[16] = (x * y * difference) * (0.5F * sqrtf(35.0F));
        result[17] = y * z * (x2 * 3.0F - y2) * (0.25F * sqrtf(70.0F));
        result[18] = x * y * (z2 * 7.0F - norm2) * (0.5F * sqrtf(5.0F));
        result[19] = y * z * (z2 * 7.0F - norm2 * 3.0F) * (0.25F * sqrtf(10.0F));
        result[20] = (z4 * 35.0F - z2 * norm2 * 30.0F
            + norm4 * 3.0F) * 0.125F;
        result[21] = x * z * (z2 * 7.0F - norm2 * 3.0F) * (0.25F * sqrtf(10.0F));
        result[22] = difference * (z2 * 7.0F - norm2) * (0.25F * sqrtf(5.0F));
        result[23] = x * z * (x2 - y2 * 3.0F) * (0.25F * sqrtf(70.0F));
        result[24] = (x2 * x2 - x2 * y2 * 6.0F + y2 * y2)
            * (0.125F * sqrtf(35.0F));
    }
}

__device__ void dpa4c_edge_gradient(
    const KernelModel& m,
    unsigned char* base,
    const Dpa4cCudaLayout& layout,
    int center_type,
    int neighbor_type,
    double dx,
    double dy,
    double dz,
    double* result) {
    const auto* scalar_adjoint = reinterpret_cast<const double*>(base + layout.scalar_adjoint);
    const auto* angular_adjoint = reinterpret_cast<const double*>(base + layout.angular_adjoint);
    const auto* divisor_adjoint = reinterpret_cast<const double*>(base + layout.divisor_adjoint);
    const auto* reduced = reinterpret_cast<const double*>(base + layout.reduced);
    const double divisor_scalar = sqrt(reduced[0] + static_cast<double>(kNormFloor));
    const double divisor_angular = sqrt(reduced[1] + static_cast<double>(kNormFloor));
    const std::int64_t pair = static_cast<std::int64_t>(center_type) * (m.ntypes + 1) + neighbor_type;
    const float* pair_scale = m.pair_scale + pair * m.channels;
    const float* pair_shift = m.pair_shift + pair * m.channels;
    const float* pair_mixing = m.pair_mixing
        + static_cast<std::int64_t>(pair) * m.channels * m.radial_modes;

    const DeviceDual3 x{static_cast<float>(dx), {1.0F, 0.0F, 0.0F}};
    const DeviceDual3 y{static_cast<float>(dy), {0.0F, 1.0F, 0.0F}};
    const DeviceDual3 z{static_cast<float>(dz), {0.0F, 0.0F, 1.0F}};
    const DeviceDual3 distance = dual_sqrt(
        x * x + y * y + z * z + dual_constant(kEpsilon * kEpsilon));
    const float rcut = static_cast<float>(m.rcut);
    DeviceDual3 cutoff = (dual_constant(rcut) - distance) / dual_constant(rcut);
    if (cutoff.value <= 0.0F) cutoff = dual_constant(0.0F);
    else if (cutoff.value >= 1.0F) cutoff = dual_constant(1.0F);
    const DeviceDual3 ux = x / distance, uy = y / distance, uz = z / distance;
    const DeviceDual3 tx = dual_constant(1.0F) - cutoff;
    DeviceDual3 series = dual_constant(35.0F);
    series = dual_constant(20.0F) + tx * series;
    series = dual_constant(10.0F) + tx * series;
    series = dual_constant(4.0F) + tx * series;
    series = dual_constant(1.0F) + tx * series;
    const DeviceDual3 envelope = cutoff * cutoff * cutoff * cutoff * series;
    DeviceDual3 basis[25];
    angular_basis_dual(ux, uy, uz, m.lmax, basis);

    double envelope_gradient = divisor_adjoint[0] / divisor_scalar * envelope.value
        + 2.0 * divisor_adjoint[1] / divisor_angular
            * envelope.value * envelope.value * envelope.value;
    double amplitude_gradient[256] = {};
    double basis_gradient[25] = {};
    float radial_basis[256] = {};
    float radial_pre[512] = {};
    float radial_hidden[256] = {};
    float radial[256] = {};
    float modes[256] = {};
    double amplitudes[256] = {};
    double radial_basis_derivative[256] = {};

    for (int index = 0; index < m.n_radial; ++index) {
        const float frequency = m.radial_freqs[index];
        const float argument = distance.value * frequency;
        const float sinc = fabsf(argument) < 1.0e-7F
            ? 1.0F : sinf(argument) / argument;
        radial_basis[index] = frequency * sinc;
        if (fabsf(argument) < 1.0e-3F) {
            radial_basis_derivative[index] = -frequency * frequency * frequency
                * distance.value / 3.0F;
        } else {
            radial_basis_derivative[index] = frequency * frequency
                * (argument * cosf(argument) - sinf(argument))
                / (argument * argument);
        }
    }
    for (int out = 0; out < 2 * m.radial_hidden; ++out) {
        double value = 0.0;
        for (int in = 0; in < m.n_radial; ++in)
            value += static_cast<double>(radial_basis[in])
                * m.radial_w0[in * (2 * m.radial_hidden) + out];
        radial_pre[out] = static_cast<float>(value);
    }
    for (int index = 0; index < m.radial_hidden; ++index) {
        const float gate = radial_pre[index];
        const float value = radial_pre[m.radial_hidden + index];
        radial_hidden[index] = gate * sigmoid(gate) * value;
    }
    for (int out = 0; out < m.channels; ++out) {
        double value = 0.0;
        for (int in = 0; in < m.radial_hidden; ++in)
            value += static_cast<double>(radial_hidden[in])
                * m.radial_w1[in * m.channels + out];
        radial[out] = static_cast<float>(value);
    }
    for (int out = 0; out < m.radial_modes; ++out) {
        double value = 0.0;
        for (int in = 0; in < m.radial_hidden; ++in)
            value += static_cast<double>(radial_hidden[in])
                * m.radial_mode_w[in * m.radial_modes + out];
        modes[out] = static_cast<float>(value);
    }
    for (int channel = 0; channel < m.channels; ++channel) {
        double amplitude = static_cast<double>(radial[channel]) * pair_scale[channel]
            + pair_shift[channel];
        for (int mode = 0; mode < m.radial_modes; ++mode)
            amplitude += pair_mixing[channel * m.radial_modes + mode] * modes[mode];
        amplitudes[channel] = amplitude;
        amplitude_gradient[channel] = scalar_adjoint[channel] * envelope.value;
        envelope_gradient += scalar_adjoint[channel] * amplitude;
    }
    int moment_offset = m.channels;
    for (int degree = 1; degree <= m.lmax; ++degree) {
        const int width = m.degree_channels[degree];
        for (int component = 0; component < 2 * degree + 1; ++component) {
            const int basis_index = degree * degree + component;
            const double actual_basis = basis[basis_index].value;
            for (int channel = 0; channel < width; ++channel) {
                const double gradient = angular_adjoint[
                    moment_offset + component * width + channel];
                amplitude_gradient[channel] += gradient * envelope.value
                    * envelope.value * actual_basis;
                envelope_gradient += gradient * amplitudes[channel]
                    * 2.0 * envelope.value * actual_basis;
                basis_gradient[basis_index] += gradient * amplitudes[channel]
                    * envelope.value * envelope.value;
            }
        }
        moment_offset += (2 * degree + 1) * width;
    }

    double mode_gradient[256] = {};
    double radial_gradient[256] = {};
    for (int channel = 0; channel < m.channels; ++channel) {
        radial_gradient[channel] = amplitude_gradient[channel] * pair_scale[channel];
        for (int mode = 0; mode < m.radial_modes; ++mode)
            mode_gradient[mode] += amplitude_gradient[channel]
                * pair_mixing[channel * m.radial_modes + mode];
    }
    double radial_hidden_gradient[256] = {};
    for (int hidden = 0; hidden < m.radial_hidden; ++hidden) {
        for (int channel = 0; channel < m.channels; ++channel)
            radial_hidden_gradient[hidden] += radial_gradient[channel]
                * m.radial_w1[hidden * m.channels + channel];
        for (int mode = 0; mode < m.radial_modes; ++mode)
            radial_hidden_gradient[hidden] += mode_gradient[mode]
                * m.radial_mode_w[hidden * m.radial_modes + mode];
    }
    double radial_value_gradient[256] = {};
    double radial_gate_gradient[256] = {};
    for (int hidden = 0; hidden < m.radial_hidden; ++hidden) {
        const float gate = radial_pre[hidden];
        const float value = radial_pre[m.radial_hidden + hidden];
        const float sigmoid_gate = sigmoid(gate);
        radial_gate_gradient[hidden] = radial_hidden_gradient[hidden] * value
            * sigmoid_gate * (1.0F + gate * (1.0F - sigmoid_gate));
        radial_value_gradient[hidden] = radial_hidden_gradient[hidden]
            * gate * sigmoid_gate;
    }
    double basis_radial_gradient[256] = {};
    for (int radial_index = 0; radial_index < m.n_radial; ++radial_index) {
        for (int hidden = 0; hidden < m.radial_hidden; ++hidden) {
            basis_radial_gradient[radial_index] += radial_gate_gradient[hidden]
                * m.radial_w0[radial_index * (2 * m.radial_hidden) + hidden]
                + radial_value_gradient[hidden]
                    * m.radial_w0[radial_index * (2 * m.radial_hidden)
                        + m.radial_hidden + hidden];
        }
    }
    double radial_derivative = 0.0;
    for (int index = 0; index < m.n_radial; ++index)
        radial_derivative += basis_radial_gradient[index]
            * radial_basis_derivative[index];

    for (int axis = 0; axis < 3; ++axis) {
        double value = envelope_gradient * envelope.derivative[axis]
            + radial_derivative * distance.derivative[axis];
        for (int basis_index = 0; basis_index < (m.lmax + 1) * (m.lmax + 1); ++basis_index)
            value += basis_gradient[basis_index] * basis[basis_index].derivative[axis];
        result[axis] = value;
    }
}

__global__ void dpa4c_force_kernel(
    const std::int64_t* graph_offsets,
    const std::int32_t* graph_atoms,
    const std::int32_t* graph_shifts,
    const double* displacements,
    const std::int32_t* type_indices,
    std::int64_t atoms,
    unsigned char* workspace,
    Dpa4cCudaLayout layout,
    KernelModel m,
    double* forces) {
    const std::int64_t center = static_cast<std::int64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    if (center >= atoms) return;
    unsigned char* base = workspace + center * layout.stride;
    const int center_type = type_indices[center];
    for (std::int64_t edge = graph_offsets[center]; edge < graph_offsets[center + 1]; ++edge) {
        const std::int32_t neighbor = graph_atoms[edge];
        if (exact_self_edge(center, neighbor, graph_shifts, edge)) continue;
        double gradient[3] = {};
        dpa4c_edge_gradient(m, base, layout, center_type, type_indices[neighbor],
            displacements[edge * 3], displacements[edge * 3 + 1],
            displacements[edge * 3 + 2], gradient);
        for (int axis = 0; axis < 3; ++axis) {
            atomicAdd(&forces[center * 3 + axis], gradient[axis]);
            atomicAdd(&forces[static_cast<std::int64_t>(neighbor) * 3 + axis], -gradient[axis]);
        }
    }
}

__global__ void dpa4c_energy_copy_and_sum_kernel(
    const std::int64_t* offsets,
    std::int64_t structures,
    std::int64_t atoms,
    const unsigned char* workspace,
    Dpa4cCudaLayout layout,
    double* output) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    if (index < atoms) {
        const auto* base = workspace + index * layout.stride;
        output[index] = *reinterpret_cast<const double*>(base + layout.atom_energy);
    }
    if (index < structures) {
        double total = 0.0;
        for (std::int64_t atom = offsets[index]; atom < offsets[index + 1]; ++atom) {
            const auto* base = workspace + atom * layout.stride;
            total += *reinterpret_cast<const double*>(base + layout.atom_energy);
        }
        output[atoms + index] = total;
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
    bispectrum_ranks_ = p.bispectrum_ranks;
    has_fitting_ = p.has_fitting;
    host_fitting_neurons_ = p.fitting_neurons;
    fitting_max_width_ = std::max(channels_, static_cast<int>(p.output_mean.size()));
    fitting_activation_offsets_.assign(
        host_fitting_neurons_.size() + 1, 0);
    fitting_activation_offsets_[0] = static_cast<std::int64_t>(p.output_mean.size());
    for (std::size_t layer = 0; layer < host_fitting_neurons_.size(); ++layer) {
        fitting_max_width_ = std::max(fitting_max_width_, host_fitting_neurons_[layer]);
        fitting_activation_offsets_[layer + 1] = fitting_activation_offsets_[layer]
            + host_fitting_neurons_[layer];
    }
    degree_offsets_.assign(static_cast<std::size_t>(lmax_ + 2), 0);
    for (int degree = 0; degree <= lmax_; ++degree) degree_offsets_[degree + 1] = degree_offsets_[degree] + (2 * degree + 1) * degree_channels_[degree];
    moment_count_ = degree_offsets_.back();
    for (int degree = 1; degree <= lmax_; ++degree) {
        const int width = degree_channels_[degree];
        for (int row = 0; row < width; ++row) for (int column = row; column < width; ++column) {
            gram_index_.push_back(row * width + column); gram_scale_.push_back(row == column ? 1.0F : kSqrt2);
        }
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
    if (has_fitting_) {
        layout->fitting_activations = reserve(
            static_cast<std::size_t>(fitting_activation_offsets_.back()) * sizeof(float),
            alignof(float));
        std::int64_t pre_count = 0;
        for (int width : host_fitting_neurons_) pre_count += width;
        layout->fitting_pre = reserve(
            static_cast<std::size_t>(pre_count) * sizeof(float), alignof(float));
        layout->fitting_scratch = reserve(
            static_cast<std::size_t>(2 * fitting_max_width_) * sizeof(float), alignof(float));
        layout->feature_gradient = reserve(
            static_cast<std::size_t>(feature_count_) * sizeof(double), alignof(double));
        layout->block_gradient = reserve(
            static_cast<std::size_t>(moment_count_) * sizeof(double), alignof(double));
        layout->projected_gradient = reserve(
            static_cast<std::size_t>(projected_count) * sizeof(double), alignof(double));
        layout->readout_scratch = reserve(
            static_cast<std::size_t>(channels_) * sizeof(double), alignof(double));
        layout->scalar_adjoint = reserve(
            static_cast<std::size_t>(channels_) * sizeof(double), alignof(double));
        layout->angular_adjoint = reserve(
            static_cast<std::size_t>(moment_count_) * sizeof(double), alignof(double));
        layout->divisor_adjoint = reserve(2 * sizeof(double), alignof(double));
        layout->atom_energy = reserve(sizeof(double), alignof(double));
    }
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
        gram_index_device_ = upload_array(context, gram_index_, "could not upload DPA4C gram indices");
        gram_scale_device_ = upload_array(context, gram_scale_, "could not upload DPA4C gram scales");
        if (has_fitting_) {
            fitting_neurons_ = upload_array(context, p.fitting_neurons,
                "could not upload DPA4C fitting widths");
            fitting_activation_offsets_device_ = upload_array(
                context, fitting_activation_offsets_,
                "could not upload DPA4C fitting activation offsets");
            fitting_weights_ = upload_array(context, p.fitting_weights,
                "could not upload DPA4C fitting weights");
            fitting_biases_ = upload_array(context, p.fitting_biases,
                "could not upload DPA4C fitting biases");
            fitting_atom_bias_ = upload_array(context, p.fitting_atom_bias,
                "could not upload DPA4C atom biases");
            output_bias_ = upload_array(context, p.output_bias,
                "could not upload DPA4C output biases");
        }
    } catch (...) { release(); throw; }
}

DeviceDpa4cModel::~DeviceDpa4cModel() noexcept { release(); }

void DeviceDpa4cModel::release() noexcept {
    type_embedding_.reset(); degree_channels_device_.reset(); bispectrum_ranks_device_.reset(); radial_freqs_.reset(); radial_w0_.reset(); radial_w1_.reset(); radial_mode_w_.reset();
    pair_scale_.reset(); pair_shift_.reset(); pair_mixing_.reset(); alignment_.reset(); alignment_offsets_.reset();
    projections_.reset(); projection_offsets_.reset(); coupling_.reset(); coupling_offsets_.reset(); degree_triples_.reset();
    probe_offsets_.reset(); probe_index_.reset(); probe_scale_.reset(); output_mean_.reset(); output_stddev_.reset();
    fitting_neurons_.reset(); fitting_activation_offsets_device_.reset();
    fitting_weights_.reset(); fitting_biases_.reset();
    fitting_atom_bias_.reset(); output_bias_.reset();
    gram_index_device_.reset(); gram_scale_device_.reset();
    layout_.reset();
}

void DeviceDpa4cModel::compute_into(
    CudaExecutionContext& context, const DeviceBatch& batch,
    const DeviceNeighborGraph& graph, const std::vector<std::int32_t>& type_indices,
    double* host_output) const {
    if (context.device() != device_) throw std::invalid_argument("DPA4C CUDA model and execution context use different devices");
    if (batch.atoms() < 0) throw std::invalid_argument("DPA4C CUDA received an invalid batch");
    if (type_indices.size() != static_cast<std::size_t>(batch.atoms())) throw std::invalid_argument("DPA4C CUDA type_indices must have one entry per atom");
    for (std::int32_t value : type_indices) if (value < 0 || value >= ntypes_) throw std::invalid_argument("DPA4C CUDA type index is outside the checkpoint type map");
    if (batch.atoms() == 0) return;
    if (host_output == nullptr) {
        throw std::invalid_argument("DPA4C CUDA output destination must not be null");
    }
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
    const auto type_indices_device = upload_array(
        context, type_indices, "could not upload DPA4C CUDA type indices");
    auto* device_types = device_data<std::int32_t>(type_indices_device);
    KernelModel model{};
    model.rcut = rcut_; model.ntypes = ntypes_; model.channels = channels_; model.lmax = lmax_;
    model.n_radial = n_radial_; model.radial_modes = radial_modes_;
    model.radial_hidden = radial_hidden_; model.pair_hidden = pair_hidden_;
    model.calibrate = calibrate_; model.feature_count = feature_count_;
    model.moment_count = moment_count_; model.triple_count = triple_count_;
    model.type_embedding = device_data<float>(type_embedding_);
    model.radial_freqs = device_data<float>(radial_freqs_);
    model.radial_w0 = device_data<float>(radial_w0_);
    model.radial_w1 = device_data<float>(radial_w1_);
    model.radial_mode_w = device_data<float>(radial_mode_w_);
    model.pair_scale = device_data<float>(pair_scale_);
    model.pair_shift = device_data<float>(pair_shift_);
    model.pair_mixing = device_data<float>(pair_mixing_);
    model.alignment = device_data<float>(alignment_);
    model.alignment_offsets = device_data<std::int64_t>(alignment_offsets_);
    model.projections = device_data<float>(projections_);
    model.projection_offsets = device_data<std::int64_t>(projection_offsets_);
    model.coupling = device_data<float>(coupling_);
    model.coupling_offsets = device_data<std::int64_t>(coupling_offsets_);
    model.degree_triples = device_data<int>(degree_triples_);
    model.probe_offsets = device_data<std::int64_t>(probe_offsets_);
    model.probe_index = device_data<std::int64_t>(probe_index_);
    model.probe_scale = device_data<float>(probe_scale_);
    model.output_mean = device_data<float>(output_mean_);
    model.output_stddev = device_data<float>(output_stddev_);
    model.degree_channels = device_data<int>(degree_channels_device_);
    model.bispectrum_ranks = device_data<int>(bispectrum_ranks_device_);
    model.fitting_layer_count = static_cast<int>(host_fitting_neurons_.size());
    model.fitting_max_width = fitting_max_width_;
    model.fitting_neurons = device_data<int>(fitting_neurons_);
    model.fitting_activation_offsets = device_data<std::int64_t>(
        fitting_activation_offsets_device_);
    model.fitting_weights = device_data<float>(fitting_weights_);
    model.fitting_biases = device_data<float>(fitting_biases_);
    model.fitting_atom_bias = device_data<double>(fitting_atom_bias_);
    model.output_bias = device_data<double>(output_bias_);
    // The compact gram metadata is constant per model and was uploaded once
    // at construction; no per-call workspace tail is needed.
    model.gram_index = device_data<std::int32_t>(gram_index_device_);
    model.gram_scale = device_data<float>(gram_scale_device_);
    auto* workspace = static_cast<unsigned char*>(context.workspace_buffer(
        stride * static_cast<std::size_t>(batch.atoms())));
    const auto blocks = static_cast<unsigned int>((static_cast<std::size_t>(batch.atoms()) + 127) / 128);
    dpa4c_kernel<<<blocks, 128, 0, context.stream()>>>(
        graph.offsets(), graph.atoms(), graph.shifts(), graph.displacements(), device_types,
        batch.atoms(), workspace, layout, model, output);
    check_cuda(cudaGetLastError(), "DPA4C CUDA descriptor kernel launch failed");
    context.download_output_into(host_output, output_count);
}

void DeviceDpa4cModel::predict_into(
    CudaExecutionContext& context, const DeviceBatch& batch,
    const DeviceNeighborGraph& graph, const std::vector<std::int32_t>& type_indices,
    double* energy, double* atom_energy, double* forces) const {
    if (context.device() != device_ || !has_fitting_) {
        throw std::invalid_argument("DPA4C CUDA prediction requires a fitted model on the active device");
    }
    const auto atoms = batch.atoms();
    const auto structures = batch.structures();
    if (atoms < 0 || structures < 0 || graph.offsets() == nullptr
        || type_indices.size() != static_cast<std::size_t>(atoms)
        || (atoms > 0 && (atom_energy == nullptr || forces == nullptr))
        || (structures > 0 && energy == nullptr)) {
        throw std::invalid_argument("DPA4C CUDA prediction received an invalid batch or output");
    }
    for (std::int32_t type : type_indices) {
        if (type < 0 || type >= ntypes_) {
            throw std::invalid_argument("DPA4C CUDA type index is outside the checkpoint type map");
        }
    }
    if (atoms == 0) {
        std::fill_n(energy, structures, 0.0);
        return;
    }

    const std::size_t atom_count = static_cast<std::size_t>(atoms);
    const std::size_t structure_count = static_cast<std::size_t>(structures);
    const std::size_t stride = align_bytes(
        static_cast<std::size_t>(layout_->fixed_bytes), alignof(double));
    if (stride == 0 || atom_count > std::numeric_limits<std::size_t>::max() / stride
        || atom_count > (std::numeric_limits<std::size_t>::max() - structure_count) / 4) {
        throw CudaOutOfMemory("DPA4C CUDA prediction buffers are too large");
    }
    Dpa4cCudaLayout layout = *layout_;
    layout.stride = static_cast<std::int64_t>(stride);
    auto* workspace = static_cast<unsigned char*>(context.workspace_buffer(atom_count * stride));
    const std::size_t output_count = atom_count * 4 + structure_count;
    double* output = context.output_buffer(output_count);
    double* force_output = output + atom_count + structure_count;
    check_cuda(cudaMemsetAsync(force_output, 0, atom_count * 3 * sizeof(double),
        context.stream()), "could not clear DPA4C CUDA forces");
    const auto type_device = upload_array(
        context, type_indices, "could not upload DPA4C CUDA type indices");
    const auto* types = device_data<std::int32_t>(type_device);

    KernelModel model{};
    model.rcut = rcut_; model.ntypes = ntypes_; model.channels = channels_; model.lmax = lmax_;
    model.n_radial = n_radial_; model.radial_modes = radial_modes_;
    model.radial_hidden = radial_hidden_; model.pair_hidden = pair_hidden_;
    model.calibrate = calibrate_; model.feature_count = feature_count_;
    model.moment_count = moment_count_; model.triple_count = triple_count_;
    model.type_embedding = device_data<float>(type_embedding_);
    model.radial_freqs = device_data<float>(radial_freqs_);
    model.radial_w0 = device_data<float>(radial_w0_);
    model.radial_w1 = device_data<float>(radial_w1_);
    model.radial_mode_w = device_data<float>(radial_mode_w_);
    model.pair_scale = device_data<float>(pair_scale_);
    model.pair_shift = device_data<float>(pair_shift_);
    model.pair_mixing = device_data<float>(pair_mixing_);
    model.alignment = device_data<float>(alignment_);
    model.alignment_offsets = device_data<std::int64_t>(alignment_offsets_);
    model.projections = device_data<float>(projections_);
    model.projection_offsets = device_data<std::int64_t>(projection_offsets_);
    model.coupling = device_data<float>(coupling_);
    model.coupling_offsets = device_data<std::int64_t>(coupling_offsets_);
    model.degree_triples = device_data<int>(degree_triples_);
    model.probe_offsets = device_data<std::int64_t>(probe_offsets_);
    model.probe_index = device_data<std::int64_t>(probe_index_);
    model.probe_scale = device_data<float>(probe_scale_);
    model.output_mean = device_data<float>(output_mean_);
    model.output_stddev = device_data<float>(output_stddev_);
    model.degree_channels = device_data<int>(degree_channels_device_);
    model.bispectrum_ranks = device_data<int>(bispectrum_ranks_device_);
    model.gram_index = device_data<std::int32_t>(gram_index_device_);
    model.gram_scale = device_data<float>(gram_scale_device_);
    model.fitting_layer_count = static_cast<int>(host_fitting_neurons_.size());
    model.fitting_max_width = fitting_max_width_;
    model.fitting_neurons = device_data<int>(fitting_neurons_);
    model.fitting_activation_offsets = device_data<std::int64_t>(
        fitting_activation_offsets_device_);
    model.fitting_weights = device_data<float>(fitting_weights_);
    model.fitting_biases = device_data<float>(fitting_biases_);
    model.fitting_atom_bias = device_data<double>(fitting_atom_bias_);
    model.output_bias = device_data<double>(output_bias_);

    const auto atom_blocks = static_cast<unsigned int>((atom_count + 127) / 128);
    dpa4c_kernel<<<atom_blocks, 128, 0, context.stream()>>>(
        graph.offsets(), graph.atoms(), graph.shifts(), graph.displacements(),
        types, atoms, workspace, layout, model, nullptr);
    check_cuda(cudaGetLastError(), "DPA4C CUDA descriptor kernel launch failed");
    dpa4c_fit_backward_kernel<<<static_cast<unsigned int>(atom_count), 256, 0, context.stream()>>>(
        atoms, types, workspace, layout, model);
    check_cuda(cudaGetLastError(), "DPA4C CUDA fitting kernel launch failed");
    dpa4c_readout_backward_kernel<<<static_cast<unsigned int>(atom_count), 1, 0, context.stream()>>>(
        atoms, workspace, layout, model);
    check_cuda(cudaGetLastError(), "DPA4C CUDA readout backward kernel launch failed");
    dpa4c_force_kernel<<<atom_blocks, 128, 0, context.stream()>>>(
        graph.offsets(), graph.atoms(), graph.shifts(), graph.displacements(),
        types, atoms, workspace, layout, model, force_output);
    check_cuda(cudaGetLastError(), "DPA4C CUDA force kernel launch failed");
    const auto output_blocks = static_cast<unsigned int>(
        (std::max(atom_count, structure_count) + 127) / 128);
    dpa4c_energy_copy_and_sum_kernel<<<output_blocks, 128, 0, context.stream()>>>(
        batch.offsets(), structures, atoms, workspace, layout, output);
    check_cuda(cudaGetLastError(), "DPA4C CUDA energy kernel launch failed");

    const auto host_output = context.download_output(output_count);
    std::copy_n(host_output.data(), atom_count, atom_energy);
    std::copy_n(host_output.data() + atom_count, structure_count, energy);
    std::copy_n(host_output.data() + atom_count + structure_count,
        atom_count * 3, forces);
}

} // namespace mdescriptor::cuda
