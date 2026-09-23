#include "mdescriptor/dpa4c.hpp"

#include "mdescriptor/detail/math3.hpp"
#include "mdescriptor/neighbor.hpp"
#include "dpa_common.hpp"
#include "descriptor_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mdescriptor {
namespace {

using detail::Mat3;
using detail::Vec3;

constexpr float kPi = 3.14159265358979323846F;
constexpr float kSqrt2 = 1.41421356237309504880F;
constexpr float kSqrt3 = 1.73205080756887729353F;
constexpr float kSqrt5 = 2.23606797749978969641F;
constexpr float kSqrt6 = 2.44948974278317809820F;
constexpr float kEpsilon = 1.0e-7F;
constexpr float kNormFloor = 0.25F;

void packed_l2_to_stf(const float* packed, int rank, std::vector<float>& matrices);

float sigmoid(float value) {
    return 1.0F / (1.0F + std::exp(-value));
}

void affine_values(
    const std::vector<float>& weights,
    int input_width,
    int output_width,
    const float* input,
    float* output,
    double* accumulators
) {
    if (output_width <= 0) {
        return;
    }
    // Weights are row-major [input, output].  Traverse each input row once
    // so the inner loop is contiguous while each output keeps its original
    // input accumulation order.
    std::fill(accumulators, accumulators + output_width, 0.0);
    for (int input_index = 0; input_index < input_width; ++input_index) {
        const double input_value = static_cast<double>(input[input_index]);
        const float* row = weights.data()
            + static_cast<std::size_t>(input_index * output_width);
        for (int output_index = 0; output_index < output_width; ++output_index) {
            accumulators[output_index] += input_value
                * static_cast<double>(row[output_index]);
        }
    }
    for (int output_index = 0; output_index < output_width; ++output_index) {
        output[output_index] = static_cast<float>(accumulators[output_index]);
    }
}

void validate_vector_size(
    const std::vector<float>& value,
    std::size_t expected,
    const char* name) {
    if (value.size() != expected) {
        throw std::invalid_argument(
            std::string(name) + " has unexpected size");
    }
}

void validate_vector_size(
    const std::vector<std::int64_t>& value,
    std::size_t expected,
    const char* name) {
    if (value.size() != expected) {
        throw std::invalid_argument(
            std::string(name) + " has unexpected size");
    }
}

void validate_options(const Dpa4cOptions& options) {
    if (!std::isfinite(options.rcut) || options.rcut <= 0.0) {
        throw std::invalid_argument("DPA4C rcut must be finite and positive");
    }
    if (options.ntypes <= 0 || options.channels <= 0 || options.lmax < 2 || options.lmax > 4
        || options.n_radial <= 0 || options.radial_modes < 0 || options.radial_hidden <= 0
        || options.pair_hidden <= 0) {
        throw std::invalid_argument("invalid DPA4C structural configuration");
    }
    if (options.degree_channels.size() != static_cast<std::size_t>(options.lmax + 1)
        || options.bispectrum_ranks.size() != static_cast<std::size_t>(options.lmax)) {
        throw std::invalid_argument("invalid DPA4C degree profile");
    }
    if (options.degree_channels[0] != options.channels) {
        throw std::invalid_argument("DPA4C degree-zero width does not match channels");
    }
    for (int width : options.degree_channels) {
        if (width <= 0) {
            throw std::invalid_argument("DPA4C degree widths must be positive");
        }
    }
    for (int rank : options.bispectrum_ranks) {
        if (rank <= 0) {
            throw std::invalid_argument("DPA4C probe ranks must be positive");
        }
    }

    const std::size_t type_rows = static_cast<std::size_t>(options.ntypes + 1);
    validate_vector_size(
        options.type_embedding,
        type_rows * static_cast<std::size_t>(options.channels),
        "DPA4C type embedding");
    validate_vector_size(
        options.radial_freqs,
        static_cast<std::size_t>(options.n_radial),
        "DPA4C radial frequencies");
    validate_vector_size(
        options.radial_w0,
        static_cast<std::size_t>(options.n_radial)
            * static_cast<std::size_t>(2 * options.radial_hidden),
        "DPA4C radial first layer");
    validate_vector_size(
        options.radial_w1,
        static_cast<std::size_t>(options.radial_hidden)
            * static_cast<std::size_t>(options.channels),
        "DPA4C radial output layer");
    validate_vector_size(
        options.radial_mode_w,
        static_cast<std::size_t>(options.radial_hidden)
            * static_cast<std::size_t>(options.radial_modes),
        "DPA4C radial mode layer");

    const int pair_output = options.channels * (2 + options.radial_modes);
    validate_vector_size(
        options.pair_w0,
        static_cast<std::size_t>(2 * options.channels)
            * static_cast<std::size_t>(2 * options.pair_hidden),
        "DPA4C pair first layer");
    validate_vector_size(
        options.pair_w1,
        static_cast<std::size_t>(options.pair_hidden)
            * static_cast<std::size_t>(pair_output),
        "DPA4C pair output layer");

    if (options.readout_alignment_offsets.size() != 3
        || options.readout_projection_offsets.size()
            != static_cast<std::size_t>(options.lmax + 1)) {
        throw std::invalid_argument("invalid DPA4C readout offsets");
    }
    const std::size_t alignment_size = static_cast<std::size_t>(
        options.readout_alignment_offsets.back());
    validate_vector_size(options.readout_alignment, alignment_size, "DPA4C alignment");
    for (int degree = 1; degree <= 2; ++degree) {
        const int width = options.degree_channels[static_cast<std::size_t>(degree)];
        const auto begin = options.readout_alignment_offsets[static_cast<std::size_t>(degree - 1)];
        const auto end = options.readout_alignment_offsets[static_cast<std::size_t>(degree)];
        if (end - begin != static_cast<std::int64_t>(width * width)) {
            throw std::invalid_argument("DPA4C alignment matrix has unexpected shape");
        }
    }
    const std::size_t projection_size = static_cast<std::size_t>(
        options.readout_projection_offsets.back());
    validate_vector_size(options.readout_projections, projection_size, "DPA4C projections");
    for (int degree = 1; degree <= options.lmax; ++degree) {
        const int width = options.degree_channels[static_cast<std::size_t>(degree)];
        const int rank = options.bispectrum_ranks[static_cast<std::size_t>(degree - 1)];
        const auto begin = options.readout_projection_offsets[static_cast<std::size_t>(degree - 1)];
        const auto end = options.readout_projection_offsets[static_cast<std::size_t>(degree)];
        const auto size = end - begin;
        if (size != 0 && size != static_cast<std::int64_t>(width * rank)) {
            throw std::invalid_argument("DPA4C probe matrix has unexpected shape");
        }
        if (size == 0 && rank != width) {
            throw std::invalid_argument("DPA4C non-full-rank probe is missing");
        }
    }

    if (options.degree_triples.size() % 3 != 0) {
        throw std::invalid_argument("DPA4C degree triples must have three entries");
    }
    // Every triple degree indexes ``bispectrum_ranks[degree - 1]`` and sizes a
    // (2*degree+1) band; a malformed checkpoint must be rejected here rather
    // than at compute time.
    for (const int degree : options.degree_triples) {
        if (degree < 1 || degree > options.lmax) {
            throw std::invalid_argument("DPA4C degree triples must stay within 1..lmax");
        }
    }
    // Probe indices address each triple's rank product; validating them here
    // keeps a malformed checkpoint from throwing mid-computation.
    for (std::size_t triple_index = 0;
         triple_index < options.degree_triples.size() / 3; ++triple_index) {
        const std::int64_t rank_product = static_cast<std::int64_t>(
            options.bispectrum_ranks[static_cast<std::size_t>(
                options.degree_triples[triple_index * 3 + 0] - 1)])
            * options.bispectrum_ranks[static_cast<std::size_t>(
                options.degree_triples[triple_index * 3 + 1] - 1)]
            * options.bispectrum_ranks[static_cast<std::size_t>(
                options.degree_triples[triple_index * 3 + 2] - 1)];
        for (auto probe = options.probe_offsets[triple_index];
             probe < options.probe_offsets[triple_index + 1]; ++probe) {
            const auto index = options.probe_index[static_cast<std::size_t>(probe)];
            if (index < 0 || index >= rank_product) {
                throw std::invalid_argument(
                    "DPA4C probe index is outside its contraction");
            }
        }
    }
    const std::size_t triple_count = options.degree_triples.size() / 3;
    validate_vector_size(options.coupling_offsets, triple_count + 1, "DPA4C coupling offsets");
    validate_vector_size(options.probe_offsets, triple_count + 1, "DPA4C probe offsets");
    if (options.coupling_offsets.front() != 0 || options.probe_offsets.front() != 0
        || options.coupling_offsets.back() < 0 || options.probe_offsets.back() < 0) {
        throw std::invalid_argument("invalid DPA4C readout offsets");
    }
    validate_vector_size(
        options.bispectrum_coupling,
        static_cast<std::size_t>(options.coupling_offsets.back()),
        "DPA4C bispectrum coupling");
    validate_vector_size(
        options.probe_index,
        static_cast<std::size_t>(options.probe_offsets.back()),
        "DPA4C probe index");
    validate_vector_size(options.probe_scale, options.probe_index.size(), "DPA4C probe scale");

    std::int64_t moment_count = 0;
    std::int64_t gram_count = 0;
    for (int degree = 0; degree <= options.lmax; ++degree) {
        const int width = options.degree_channels[static_cast<std::size_t>(degree)];
        moment_count += static_cast<std::int64_t>(2 * degree + 1) * width;
        if (degree > 0) {
            gram_count += static_cast<std::int64_t>(width) * (width + 1) / 2;
        }
    }
    const std::int64_t quartic_count = static_cast<std::int64_t>(
        options.bispectrum_ranks[0] * options.bispectrum_ranks[1]);
    const std::int64_t expected_features = options.channels + gram_count
        + options.probe_offsets.back() + quartic_count + 2 + options.channels;
    if (expected_features <= 0
        || options.output_mean.size() != static_cast<std::size_t>(expected_features)
        || options.output_stddev.size() != static_cast<std::size_t>(expected_features)) {
        throw std::invalid_argument("DPA4C output calibration has unexpected shape");
    }
    for (float value : options.output_stddev) {
        if (!std::isfinite(value) || value <= 0.0F) {
            throw std::invalid_argument("DPA4C output standard deviations must be positive");
        }
    }
}

void validate_fitting_options(const Dpa4cOptions& options, std::int64_t feature_count) {
    if (options.fitting_weights.empty() || options.fitting_activation != "silu") {
        throw std::invalid_argument(
            "DPA4C prediction requires a supported energy fitting network");
    }
    if (options.fitting_atom_bias.size() != static_cast<std::size_t>(options.ntypes)
        || options.output_bias.size() != static_cast<std::size_t>(options.ntypes)) {
        throw std::invalid_argument("DPA4C fitting type biases have unexpected size");
    }
    std::int64_t input_width = feature_count;
    std::size_t expected_weights = 0;
    std::size_t expected_biases = 1;
    for (int width : options.fitting_neurons) {
        if (width <= 0) {
            throw std::invalid_argument("DPA4C fitting widths must be positive");
        }
        expected_weights += static_cast<std::size_t>(input_width)
            * static_cast<std::size_t>(width);
        expected_biases += static_cast<std::size_t>(width);
        input_width = width;
    }
    expected_weights += static_cast<std::size_t>(input_width);
    if (options.fitting_weights.size() != expected_weights
        || options.fitting_biases.size() != expected_biases) {
        throw std::invalid_argument("DPA4C fitting network tensors have unexpected shape");
    }
}

inline float silu(float value) {
    return value / (1.0F + std::exp(-value));
}

inline float silu_derivative(float value) {
    const float gate = 1.0F / (1.0F + std::exp(-value));
    return gate * (1.0F + value * (1.0F - gate));
}

double fitting_energy_and_gradient(
    const Dpa4cOptions& options,
    const float* input,
    float* input_gradient,
    int type_index
) {
    const int layer_count = static_cast<int>(options.fitting_neurons.size());
    std::vector<std::vector<float>> activations(static_cast<std::size_t>(layer_count + 1));
    std::vector<std::vector<float>> pre_activations(static_cast<std::size_t>(layer_count));
    activations[0].assign(input, input + options.output_mean.size());
    std::vector<std::size_t> weight_offsets(static_cast<std::size_t>(layer_count + 1));
    std::vector<std::size_t> bias_offsets(static_cast<std::size_t>(layer_count + 1));
    int current_width = static_cast<int>(options.output_mean.size());
    std::size_t weight_offset = 0;
    std::size_t bias_offset = 0;
    for (int layer = 0; layer < layer_count; ++layer) {
        const int width = options.fitting_neurons[static_cast<std::size_t>(layer)];
        weight_offsets[static_cast<std::size_t>(layer)] = weight_offset;
        bias_offsets[static_cast<std::size_t>(layer)] = bias_offset;
        weight_offset += static_cast<std::size_t>(current_width)
            * static_cast<std::size_t>(width);
        bias_offset += static_cast<std::size_t>(width);
        std::vector<float>& pre = pre_activations[static_cast<std::size_t>(layer)];
        const std::vector<float>& previous = activations[static_cast<std::size_t>(layer)];
        const float* weights = options.fitting_weights.data()
            + weight_offsets[static_cast<std::size_t>(layer)];
        const float* biases = options.fitting_biases.data()
            + bias_offsets[static_cast<std::size_t>(layer)];
        pre.resize(static_cast<std::size_t>(width));
        for (int out = 0; out < width; ++out) {
            double value = static_cast<double>(biases[out]);
            for (int in = 0; in < current_width; ++in) {
                value += static_cast<double>(previous[static_cast<std::size_t>(in)])
                    * static_cast<double>(weights[in * width + out]);
            }
            pre[static_cast<std::size_t>(out)] = static_cast<float>(value);
        }
        std::vector<float>& next = activations[static_cast<std::size_t>(layer + 1)];
        next.resize(static_cast<std::size_t>(width));
        for (int out = 0; out < width; ++out) {
            float value = silu(pre[static_cast<std::size_t>(out)]);
            if (width == current_width) {
                value += previous[static_cast<std::size_t>(out)];
            }
            next[static_cast<std::size_t>(out)] = value;
        }
        current_width = width;
    }
    const std::size_t final_weight_offset = weight_offset;
    const std::size_t final_bias_offset = bias_offset;
    const float* final_weights = options.fitting_weights.data() + final_weight_offset;
    double value = static_cast<double>(options.fitting_biases[final_bias_offset]);
    for (int in = 0; in < current_width; ++in) {
        value += static_cast<double>(activations.back()[static_cast<std::size_t>(in)])
            * static_cast<double>(final_weights[in]);
    }

    std::vector<double> gradient(static_cast<std::size_t>(current_width));
    for (int in = 0; in < current_width; ++in) {
        gradient[static_cast<std::size_t>(in)] = static_cast<double>(final_weights[in]);
    }
    for (int layer = layer_count - 1; layer >= 0; --layer) {
        const int width = options.fitting_neurons[static_cast<std::size_t>(layer)];
        const int previous_width = layer == 0
            ? static_cast<int>(options.output_mean.size())
            : options.fitting_neurons[static_cast<std::size_t>(layer - 1)];
        const float* weights = options.fitting_weights.data()
            + weight_offsets[static_cast<std::size_t>(layer)];
        std::vector<double> previous_gradient(static_cast<std::size_t>(previous_width), 0.0);
        for (int in = 0; in < previous_width; ++in) {
            double total = width == previous_width ? gradient[static_cast<std::size_t>(in)] : 0.0;
            for (int out = 0; out < width; ++out) {
                const double activated_gradient = gradient[static_cast<std::size_t>(out)]
                    * static_cast<double>(silu_derivative(
                        pre_activations[static_cast<std::size_t>(layer)][
                            static_cast<std::size_t>(out)]));
                total += activated_gradient * static_cast<double>(weights[in * width + out]);
            }
            previous_gradient[static_cast<std::size_t>(in)] = total;
        }
        gradient = std::move(previous_gradient);
    }
    for (std::size_t feature = 0; feature < gradient.size(); ++feature) {
        input_gradient[feature] = static_cast<float>(gradient[feature]);
    }
    const float biased_energy = static_cast<float>(static_cast<float>(value)
        + static_cast<float>(options.fitting_atom_bias[static_cast<std::size_t>(type_index)]));
    return static_cast<double>(biased_energy)
        + options.output_bias[static_cast<std::size_t>(type_index)];
}

struct Dual3 {
    float value = 0.0F;
    float derivative[3] = {0.0F, 0.0F, 0.0F};
};

Dual3 dual_constant(float value) { return {value, {0.0F, 0.0F, 0.0F}}; }
Dual3 operator+(Dual3 lhs, Dual3 rhs) {
    Dual3 out{lhs.value + rhs.value, {}};
    for (int axis = 0; axis < 3; ++axis) out.derivative[axis] = lhs.derivative[axis] + rhs.derivative[axis];
    return out;
}
Dual3 operator-(Dual3 lhs, Dual3 rhs) {
    Dual3 out{lhs.value - rhs.value, {}};
    for (int axis = 0; axis < 3; ++axis) out.derivative[axis] = lhs.derivative[axis] - rhs.derivative[axis];
    return out;
}
Dual3 operator*(Dual3 lhs, Dual3 rhs) {
    Dual3 out{lhs.value * rhs.value, {}};
    for (int axis = 0; axis < 3; ++axis) out.derivative[axis] =
        lhs.derivative[axis] * rhs.value + lhs.value * rhs.derivative[axis];
    return out;
}
Dual3 operator*(Dual3 lhs, float rhs) {
    return lhs * dual_constant(rhs);
}
Dual3 operator*(float lhs, Dual3 rhs) { return rhs * lhs; }
Dual3 operator/(Dual3 lhs, Dual3 rhs) {
    Dual3 out{lhs.value / rhs.value, {}};
    const float scale = 1.0F / (rhs.value * rhs.value);
    for (int axis = 0; axis < 3; ++axis) out.derivative[axis] =
        (lhs.derivative[axis] * rhs.value - lhs.value * rhs.derivative[axis]) * scale;
    return out;
}
Dual3 dual_sqrt(Dual3 value) {
    Dual3 out{std::sqrt(value.value), {}};
    const float scale = 0.5F / out.value;
    for (int axis = 0; axis < 3; ++axis) out.derivative[axis] = value.derivative[axis] * scale;
    return out;
}
Dual3 dual_sin(Dual3 value) {
    Dual3 out{std::sin(value.value), {}};
    const float scale = std::cos(value.value);
    for (int axis = 0; axis < 3; ++axis) out.derivative[axis] = value.derivative[axis] * scale;
    return out;
}
Dual3 dual_sinc(Dual3 value) {
    if (std::abs(value.value) < 1.0e-3F) {
        const Dual3 square = value * value;
        return dual_constant(1.0F) - square * (1.0F / 6.0F)
            + square * square * (1.0F / 120.0F);
    }
    return dual_sin(value) / value;
}
Dual3 dual_silu(Dual3 value) {
    const float gate = 1.0F / (1.0F + std::exp(-value.value));
    Dual3 out{value.value * gate, {}};
    const float scale = gate * (1.0F + value.value * (1.0F - gate));
    for (int axis = 0; axis < 3; ++axis) out.derivative[axis] = value.derivative[axis] * scale;
    return out;
}

void angular_basis_dual(
    const Dual3& x, const Dual3& y, const Dual3& z, int lmax, Dual3* result) {
    const Dual3 squared_norm = x * x + y * y + z * z;
    const Dual3 x2 = x * x, y2 = y * y, z2 = z * z;
    std::fill(result, result + static_cast<std::size_t>((lmax + 1) * (lmax + 1)),
        dual_constant(0.0F));
    result[0] = dual_constant(1.0F);
    if (lmax >= 1) { result[1] = x; result[2] = y; result[3] = z; }
    if (lmax >= 2) {
        result[4] = (x * y) * kSqrt3;
        result[5] = (y * z) * kSqrt3;
        result[6] = (z2 * 3.0F - squared_norm) * 0.5F;
        result[7] = (x * z) * kSqrt3;
        result[8] = (x2 - y2) * (0.5F * kSqrt3);
    }
    if (lmax >= 3) {
        result[9] = y * (x2 * 3.0F - y2) * std::sqrt(5.0F / 8.0F);
        result[10] = (x * y * z) * std::sqrt(15.0F);
        result[11] = y * (z2 * 5.0F - squared_norm) * std::sqrt(3.0F / 8.0F);
        result[12] = z * (z2 * 5.0F - squared_norm * 3.0F) * 0.5F;
        result[13] = x * (z2 * 5.0F - squared_norm) * std::sqrt(3.0F / 8.0F);
        result[14] = z * (x2 - y2) * (0.5F * std::sqrt(15.0F));
        result[15] = x * (x2 - y2 * 3.0F) * std::sqrt(5.0F / 8.0F);
    }
    if (lmax >= 4) {
        const Dual3 x2_minus_y2 = x2 - y2;
        const Dual3 z4 = z2 * z2;
        const Dual3 squared_norm2 = squared_norm * squared_norm;
        result[16] = x * y * x2_minus_y2 * (0.5F * std::sqrt(35.0F));
        result[17] = y * z * (x2 * 3.0F - y2) * (0.25F * std::sqrt(70.0F));
        result[18] = x * y * (z2 * 7.0F - squared_norm) * (0.5F * std::sqrt(5.0F));
        result[19] = y * z * (z2 * 7.0F - squared_norm * 3.0F) * (0.25F * std::sqrt(10.0F));
        result[20] = (z4 * 35.0F - z2 * squared_norm * 30.0F + squared_norm2 * 3.0F) * 0.125F;
        result[21] = x * z * (z2 * 7.0F - squared_norm * 3.0F) * (0.25F * std::sqrt(10.0F));
        result[22] = x2_minus_y2 * (z2 * 7.0F - squared_norm) * (0.25F * std::sqrt(5.0F));
        result[23] = x * z * (x2 - y2 * 3.0F) * (0.25F * std::sqrt(70.0F));
        result[24] = (x2 * x2 - x2 * y2 * 6.0F + y2 * y2) * (0.125F * std::sqrt(35.0F));
    }
}

void edge_derivatives(
    const Dpa4cOptions& options,
    const std::vector<float>& pair_scale,
    const std::vector<float>& pair_shift,
    const std::vector<float>& pair_mixing,
    int center_type,
    int neighbor_type,
    float dx,
    float dy,
    float dz,
    const double* edge_values,
    const double* grad_scalar,
    const double* grad_angular,
    double divisor_scalar,
    double divisor_angular,
    double grad_divisor_scalar,
    double grad_divisor_angular,
    float* result
) {
    Dual3 displacement[3] = {{dx, {1.0F, 0.0F, 0.0F}},
        {dy, {0.0F, 1.0F, 0.0F}}, {dz, {0.0F, 0.0F, 1.0F}}};
    const Dual3 distance = dual_sqrt(
        displacement[0] * displacement[0] + displacement[1] * displacement[1]
        + displacement[2] * displacement[2] + dual_constant(kEpsilon * kEpsilon));
    const float rcut = static_cast<float>(options.rcut);
    Dual3 cutoff = (dual_constant(rcut) - distance) / dual_constant(rcut);
    if (cutoff.value <= 0.0F) cutoff = dual_constant(0.0F);
    else if (cutoff.value >= 1.0F) cutoff = dual_constant(1.0F);
    const Dual3 x = dual_constant(1.0F) - cutoff;
    Dual3 series = dual_constant(35.0F);
    series = dual_constant(20.0F) + x * series;
    series = dual_constant(10.0F) + x * series;
    series = dual_constant(4.0F) + x * series;
    series = dual_constant(1.0F) + x * series;
    const Dual3 envelope = cutoff * cutoff * cutoff * cutoff * series;
    const Dual3 unit[3] = {displacement[0] / distance,
        displacement[1] / distance, displacement[2] / distance};
    Dual3 basis[25];
    angular_basis_dual(unit[0], unit[1], unit[2], options.lmax, basis);

    std::vector<Dual3> radial_basis(static_cast<std::size_t>(options.n_radial));
    std::vector<Dual3> radial_pre(static_cast<std::size_t>(2 * options.radial_hidden));
    std::vector<Dual3> radial_hidden(static_cast<std::size_t>(options.radial_hidden));
    std::vector<Dual3> radial(static_cast<std::size_t>(options.channels));
    std::vector<Dual3> modes(static_cast<std::size_t>(options.radial_modes));
    for (int index = 0; index < options.n_radial; ++index) {
        const float frequency = options.radial_freqs[static_cast<std::size_t>(index)];
        radial_basis[static_cast<std::size_t>(index)] = dual_sinc(distance * frequency) * frequency;
    }
    for (int out = 0; out < 2 * options.radial_hidden; ++out) {
        Dual3 value = dual_constant(0.0F);
        for (int in = 0; in < options.n_radial; ++in) {
            value = value + radial_basis[static_cast<std::size_t>(in)]
                * options.radial_w0[static_cast<std::size_t>(in * 2 * options.radial_hidden + out)];
        }
        radial_pre[static_cast<std::size_t>(out)] = value;
    }
    for (int hidden = 0; hidden < options.radial_hidden; ++hidden) {
        radial_hidden[static_cast<std::size_t>(hidden)] = dual_silu(
            radial_pre[static_cast<std::size_t>(hidden)])
            * radial_pre[static_cast<std::size_t>(options.radial_hidden + hidden)];
    }
    for (int out = 0; out < options.channels; ++out) {
        Dual3 value = dual_constant(0.0F);
        for (int in = 0; in < options.radial_hidden; ++in) {
            value = value + radial_hidden[static_cast<std::size_t>(in)]
                * options.radial_w1[static_cast<std::size_t>(in * options.channels + out)];
        }
        radial[static_cast<std::size_t>(out)] = value;
    }
    for (int out = 0; out < options.radial_modes; ++out) {
        Dual3 value = dual_constant(0.0F);
        for (int in = 0; in < options.radial_hidden; ++in) {
            value = value + radial_hidden[static_cast<std::size_t>(in)]
                * options.radial_mode_w[static_cast<std::size_t>(in * options.radial_modes + out)];
        }
        modes[static_cast<std::size_t>(out)] = value;
    }
    (void)center_type;
    (void)neighbor_type;
    std::vector<Dual3> amplitude(static_cast<std::size_t>(options.channels));
    for (int channel = 0; channel < options.channels; ++channel) {
        Dual3 value = radial[static_cast<std::size_t>(channel)]
            * pair_scale[static_cast<std::size_t>(channel)]
            + dual_constant(pair_shift[static_cast<std::size_t>(channel)]);
        for (int mode = 0; mode < options.radial_modes; ++mode) {
            value = value + modes[static_cast<std::size_t>(mode)]
                * pair_mixing[static_cast<std::size_t>(
                    channel * options.radial_modes + mode)];
        }
        amplitude[static_cast<std::size_t>(channel)] = value;
    }

    const double actual_envelope = static_cast<double>(edge_values[3]);
    double envelope_gradient = grad_divisor_scalar / divisor_scalar * actual_envelope
        + 2.0 * grad_divisor_angular / divisor_angular
            * actual_envelope * actual_envelope * actual_envelope;
    std::vector<double> amplitude_gradient(static_cast<std::size_t>(options.channels), 0.0);
    for (int channel = 0; channel < options.channels; ++channel) {
        const double actual_amplitude = static_cast<double>(edge_values[4 + channel]);
        amplitude_gradient[static_cast<std::size_t>(channel)] =
            grad_scalar[channel] * actual_envelope;
        envelope_gradient += grad_scalar[channel] * actual_amplitude;
        for (int degree = 1; degree <= options.lmax; ++degree) {
            const int width = options.degree_channels[static_cast<std::size_t>(degree)];
            if (channel >= width) {
                continue;
            }
            const int moment_offset = static_cast<int>(options.channels);
            int degree_offset = moment_offset;
            for (int previous = 1; previous < degree; ++previous) {
                degree_offset += (2 * previous + 1)
                    * options.degree_channels[static_cast<std::size_t>(previous)];
            }
            for (int component = 0; component < 2 * degree + 1; ++component) {
                const std::size_t gradient_index = static_cast<std::size_t>(
                    degree_offset + component * width + channel);
                const double gradient = grad_angular[gradient_index];
                const double actual_basis = static_cast<double>(edge_values[
                    4 + options.channels + degree * degree + component]);
                amplitude_gradient[static_cast<std::size_t>(channel)] += gradient
                    * actual_envelope * actual_envelope * actual_basis;
                envelope_gradient += gradient * actual_amplitude
                    * 2.0 * actual_envelope * actual_basis;
            }
        }
    }

    for (int axis = 0; axis < 3; ++axis) {
        double gradient = envelope_gradient * envelope.derivative[axis];
        for (int channel = 0; channel < options.channels; ++channel) {
            gradient += amplitude_gradient[static_cast<std::size_t>(channel)]
                * amplitude[static_cast<std::size_t>(channel)].derivative[axis];
        }
        for (int degree = 1; degree <= options.lmax; ++degree) {
            const int width = options.degree_channels[static_cast<std::size_t>(degree)];
            int degree_offset = static_cast<int>(options.channels);
            for (int previous = 1; previous < degree; ++previous) {
                degree_offset += (2 * previous + 1)
                    * options.degree_channels[static_cast<std::size_t>(previous)];
            }
            for (int component = 0; component < 2 * degree + 1; ++component) {
                const int basis_index = degree * degree + component;
                for (int channel = 0; channel < width; ++channel) {
                    const double gradient_basis = grad_angular[static_cast<std::size_t>(
                        degree_offset + component * width + channel)]
                        * static_cast<double>(edge_values[3])
                        * static_cast<double>(edge_values[3])
                        * static_cast<double>(edge_values[4 + channel]);
                    gradient += gradient_basis * basis[basis_index].derivative[axis];
                }
            }
        }
        result[axis] = static_cast<float>(gradient);
    }
}

struct Dpa4cReadoutAdjoints {
    std::vector<double> scalar;
    std::vector<double> angular;
    double divisor_scalar = 0.0;
    double divisor_angular = 0.0;
};

Dpa4cReadoutAdjoints backprop_readout(
    const Dpa4cOptions& options,
    const std::vector<int>& degree_offsets,
    const std::vector<std::int64_t>& gram_offsets,
    const std::vector<std::int64_t>& gram_index,
    const std::vector<float>& gram_scale,
    const std::vector<double>& feature_gradient,
    const std::vector<std::vector<float>>& blocks,
    const std::vector<std::vector<float>>& projected,
    const std::vector<double>& reduced,
    double divisor_scalar,
    double divisor_angular
) {
    std::vector<std::vector<double>> block_gradient(blocks.size());
    for (std::size_t degree = 0; degree < blocks.size(); ++degree) {
        block_gradient[degree].assign(blocks[degree].size(), 0.0);
    }
    std::vector<std::vector<double>> projected_gradient(projected.size());
    for (std::size_t degree = 0; degree < projected.size(); ++degree) {
        projected_gradient[degree].assign(projected[degree].size(), 0.0);
    }

    std::size_t feature = 0;
    for (int channel = 0; channel < options.channels; ++channel) {
        block_gradient[0][static_cast<std::size_t>(channel)] = feature_gradient[feature++];
    }
    for (int degree = 1; degree <= options.lmax; ++degree) {
        const int width = options.degree_channels[static_cast<std::size_t>(degree)];
        const int dimension = 2 * degree + 1;
        for (std::int64_t gram = gram_offsets[static_cast<std::size_t>(degree - 1)];
             gram < gram_offsets[static_cast<std::size_t>(degree)]; ++gram) {
            const int flat = static_cast<int>(gram_index[static_cast<std::size_t>(gram)]);
            const int row = flat / width;
            const int column = flat % width;
            const double gradient = feature_gradient[feature++]
                * static_cast<double>(gram_scale[static_cast<std::size_t>(gram)]);
            for (int component = 0; component < dimension; ++component) {
                const std::size_t row_index = static_cast<std::size_t>(
                    component * width + row);
                const std::size_t column_index = static_cast<std::size_t>(
                    component * width + column);
                block_gradient[static_cast<std::size_t>(degree)][row_index] +=
                    gradient * static_cast<double>(blocks[static_cast<std::size_t>(degree)][column_index])
                    * (row == column ? 2.0 : 1.0);
                if (row != column) {
                    block_gradient[static_cast<std::size_t>(degree)][column_index] +=
                        gradient * static_cast<double>(blocks[static_cast<std::size_t>(degree)][row_index]);
                }
            }
        }
    }

    const std::size_t triple_count = options.degree_triples.size() / 3;
    for (std::size_t triple = 0; triple < triple_count; ++triple) {
        const int degree_1 = options.degree_triples[triple * 3];
        const int degree_2 = options.degree_triples[triple * 3 + 1];
        const int degree_3 = options.degree_triples[triple * 3 + 2];
        const int rank_1 = options.bispectrum_ranks[static_cast<std::size_t>(degree_1 - 1)];
        const int rank_2 = options.bispectrum_ranks[static_cast<std::size_t>(degree_2 - 1)];
        const int rank_3 = options.bispectrum_ranks[static_cast<std::size_t>(degree_3 - 1)];
        std::vector<double> full_gradient(static_cast<std::size_t>(rank_1 * rank_2 * rank_3), 0.0);
        for (std::int64_t probe = options.probe_offsets[triple];
             probe < options.probe_offsets[triple + 1]; ++probe) {
            const std::size_t probe_index = static_cast<std::size_t>(probe);
            full_gradient[static_cast<std::size_t>(options.probe_index[probe_index])] +=
                feature_gradient[feature++]
                * static_cast<double>(options.probe_scale[probe_index]);
        }
        const std::size_t projected_1_offset = static_cast<std::size_t>(
            degree_offsets[static_cast<std::size_t>(degree_1)] - options.channels);
        const std::size_t projected_2_offset = static_cast<std::size_t>(
            degree_offsets[static_cast<std::size_t>(degree_2)] - options.channels);
        const std::size_t projected_3_offset = static_cast<std::size_t>(
            degree_offsets[static_cast<std::size_t>(degree_3)] - options.channels);
        if (degree_1 == 1 && degree_2 == 1 && degree_3 == 2) {
            std::vector<float> matrices;
            packed_l2_to_stf(projected[1].data(), rank_3, matrices);
            std::vector<double> matrix_gradient(static_cast<std::size_t>(rank_3 * 9), 0.0);
            const double normalization = -1.0 / static_cast<double>(kSqrt5);
            for (int first = 0; first < rank_1; ++first) {
                for (int second = 0; second < rank_2; ++second) {
                    for (int tensor = 0; tensor < rank_3; ++tensor) {
                        const double gradient = full_gradient[static_cast<std::size_t>(
                            (first * rank_2 + second) * rank_3 + tensor)] * normalization;
                        const float* matrix = matrices.data() + static_cast<std::size_t>(tensor * 9);
                        const double left[3] = {
                            projected[0][static_cast<std::size_t>(first)],
                            projected[0][static_cast<std::size_t>(rank_1 + first)],
                            projected[0][static_cast<std::size_t>(2 * rank_1 + first)],
                        };
                        const double right[3] = {
                            projected[0][static_cast<std::size_t>(second)],
                            projected[0][static_cast<std::size_t>(rank_1 + second)],
                            projected[0][static_cast<std::size_t>(2 * rank_1 + second)],
                        };
                        for (int row = 0; row < 3; ++row) {
                            double left_gradient = 0.0;
                            double right_gradient = 0.0;
                            for (int column = 0; column < 3; ++column) {
                                left_gradient += static_cast<double>(matrix[row * 3 + column])
                                    * right[column];
                                right_gradient += static_cast<double>(matrix[column * 3 + row])
                                    * left[column];
                                matrix_gradient[static_cast<std::size_t>(tensor * 9 + row * 3 + column)] +=
                                    gradient * left[row] * right[column];
                            }
                            projected_gradient[0][static_cast<std::size_t>(row * rank_1 + first)] +=
                                gradient * left_gradient;
                            projected_gradient[0][static_cast<std::size_t>(row * rank_2 + second)] +=
                                gradient * right_gradient;
                        }
                    }
                }
            }
            for (int tensor = 0; tensor < rank_3; ++tensor) {
                const double* matrix = matrix_gradient.data() + static_cast<std::size_t>(tensor * 9);
                projected_gradient[1][static_cast<std::size_t>(tensor)] +=
                    (matrix[1] + matrix[3]) / static_cast<double>(kSqrt2);
                projected_gradient[1][static_cast<std::size_t>(rank_3 + tensor)] +=
                    (matrix[5] + matrix[7]) / static_cast<double>(kSqrt2);
                projected_gradient[1][static_cast<std::size_t>(2 * rank_3 + tensor)] +=
                    (-matrix[0] - matrix[4] + 2.0 * matrix[8])
                    / std::sqrt(6.0);
                projected_gradient[1][static_cast<std::size_t>(3 * rank_3 + tensor)] +=
                    (matrix[2] + matrix[6]) / static_cast<double>(kSqrt2);
                projected_gradient[1][static_cast<std::size_t>(4 * rank_3 + tensor)] +=
                    (matrix[0] - matrix[4]) / static_cast<double>(kSqrt2);
            }
        } else {
            const int dim_1 = 2 * degree_1 + 1;
            const int dim_2 = 2 * degree_2 + 1;
            const int dim_3 = 2 * degree_3 + 1;
            const float* coupling = options.bispectrum_coupling.data()
                + options.coupling_offsets[triple];
            for (int first = 0; first < rank_1; ++first) {
                for (int second = 0; second < rank_2; ++second) {
                    for (int third = 0; third < rank_3; ++third) {
                        const double gradient = full_gradient[static_cast<std::size_t>(
                            (first * rank_2 + second) * rank_3 + third)];
                        for (int i = 0; i < dim_1; ++i) {
                            for (int j = 0; j < dim_2; ++j) {
                                for (int k = 0; k < dim_3; ++k) {
                                    const double weight = static_cast<double>(coupling[
                                        (i * dim_2 + j) * dim_3 + k]);
                                    const double first_value = projected[
                                        static_cast<std::size_t>(degree_1 - 1)][
                                        static_cast<std::size_t>(i * rank_1 + first)];
                                    const double second_value = projected[
                                        static_cast<std::size_t>(degree_2 - 1)][
                                        static_cast<std::size_t>(j * rank_2 + second)];
                                    const double third_value = projected[
                                        static_cast<std::size_t>(degree_3 - 1)][
                                        static_cast<std::size_t>(k * rank_3 + third)];
                                    projected_gradient[static_cast<std::size_t>(degree_1 - 1)][
                                        static_cast<std::size_t>(i * rank_1 + first)] +=
                                        gradient * weight * second_value * third_value;
                                    projected_gradient[static_cast<std::size_t>(degree_2 - 1)][
                                        static_cast<std::size_t>(j * rank_2 + second)] +=
                                        gradient * weight * first_value * third_value;
                                    projected_gradient[static_cast<std::size_t>(degree_3 - 1)][
                                        static_cast<std::size_t>(k * rank_3 + third)] +=
                                        gradient * weight * first_value * second_value;
                                }
                            }
                        }
                    }
                }
            }
        }
        (void)projected_1_offset;
        (void)projected_2_offset;
        (void)projected_3_offset;
    }

    const int vector_rank = options.bispectrum_ranks[0];
    const int tensor_rank = options.bispectrum_ranks[1];
    const std::size_t quartic_offset = feature;
    std::vector<float> tensor_matrices;
    packed_l2_to_stf(projected[1].data(), tensor_rank, tensor_matrices);
    std::vector<double> quartic_tensor_gradient(static_cast<std::size_t>(tensor_rank * 9), 0.0);
    for (int tensor = 0; tensor < tensor_rank; ++tensor) {
        const float* matrix = tensor_matrices.data() + static_cast<std::size_t>(tensor * 9);
        for (int vector = 0; vector < vector_rank; ++vector) {
            const double v[3] = {
                projected[0][static_cast<std::size_t>(vector)],
                projected[0][static_cast<std::size_t>(vector_rank + vector)],
                projected[0][static_cast<std::size_t>(2 * vector_rank + vector)],
            };
            double qv[3] = {};
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    qv[row] += static_cast<double>(matrix[row * 3 + column]) * v[column];
                }
            }
            const double gradient = feature_gradient[quartic_offset
                + static_cast<std::size_t>(tensor * vector_rank + vector)];
            for (int column = 0; column < 3; ++column) {
                double value = 0.0;
                for (int row = 0; row < 3; ++row) {
                    value += static_cast<double>(matrix[row * 3 + column]) * qv[row];
                    quartic_tensor_gradient[static_cast<std::size_t>(tensor * 9
                        + row * 3 + column)] += 2.0 * gradient * qv[row] * v[column];
                }
                projected_gradient[0][static_cast<std::size_t>(column * vector_rank + vector)] +=
                    2.0 * gradient * value;
            }
        }
    }
    for (int tensor = 0; tensor < tensor_rank; ++tensor) {
        const double* matrix = quartic_tensor_gradient.data() + static_cast<std::size_t>(tensor * 9);
        projected_gradient[1][static_cast<std::size_t>(tensor)] +=
            (matrix[1] + matrix[3]) / static_cast<double>(kSqrt2);
        projected_gradient[1][static_cast<std::size_t>(tensor_rank + tensor)] +=
            (matrix[5] + matrix[7]) / static_cast<double>(kSqrt2);
        projected_gradient[1][static_cast<std::size_t>(2 * tensor_rank + tensor)] +=
            (-matrix[0] - matrix[4] + 2.0 * matrix[8]) / std::sqrt(6.0);
        projected_gradient[1][static_cast<std::size_t>(3 * tensor_rank + tensor)] +=
            (matrix[2] + matrix[6]) / static_cast<double>(kSqrt2);
        projected_gradient[1][static_cast<std::size_t>(4 * tensor_rank + tensor)] +=
            (matrix[0] - matrix[4]) / static_cast<double>(kSqrt2);
    }
    feature = quartic_offset + static_cast<std::size_t>(vector_rank * tensor_rank);
    const double direct_divisor_scalar = feature_gradient[feature++];
    const double direct_divisor_angular = feature_gradient[feature++];
    if (feature + static_cast<std::size_t>(options.channels) != feature_gradient.size()) {
        throw std::runtime_error("DPA4C prediction readout gradient has an unexpected width");
    }

    std::vector<std::vector<double>> normalized_moment_gradient(blocks.size());
    for (std::size_t degree = 0; degree < blocks.size(); ++degree) {
        normalized_moment_gradient[degree].assign(blocks[degree].size(), 0.0);
    }
    for (int degree = 1; degree <= options.lmax; ++degree) {
        const int width = options.degree_channels[static_cast<std::size_t>(degree)];
        const int dimension = 2 * degree + 1;
        const int rank = options.bispectrum_ranks[static_cast<std::size_t>(degree - 1)];
        const auto matrix_begin = options.readout_projection_offsets[
            static_cast<std::size_t>(degree - 1)];
        const auto matrix_end = options.readout_projection_offsets[
            static_cast<std::size_t>(degree)];
        std::vector<double>& grad_block = block_gradient[static_cast<std::size_t>(degree)];
        const std::vector<double>& grad_projected = projected_gradient[
            static_cast<std::size_t>(degree - 1)];
        if (matrix_begin == matrix_end) {
            for (std::size_t index = 0; index < grad_block.size(); ++index) {
                grad_block[index] += grad_projected[index];
            }
        } else {
            const float* matrix = options.readout_projections.data() + matrix_begin;
            for (int component = 0; component < dimension; ++component) {
                for (int input_channel = 0; input_channel < width; ++input_channel) {
                    double value = 0.0;
                    for (int output_channel = 0; output_channel < rank; ++output_channel) {
                        value += grad_projected[static_cast<std::size_t>(
                            component * rank + output_channel)]
                            * static_cast<double>(matrix[input_channel * rank + output_channel]);
                    }
                    grad_block[static_cast<std::size_t>(component * width + input_channel)] += value;
                }
            }
        }
        if (degree == 1 || degree == 2) {
            const auto alignment_begin = options.readout_alignment_offsets[
                static_cast<std::size_t>(degree - 1)];
            const float* matrix = options.readout_alignment.data() + alignment_begin;
            const std::vector<double> aligned_gradient = grad_block;
            const std::vector<float>& source = blocks[static_cast<std::size_t>(degree)];
            std::fill(grad_block.begin(), grad_block.end(), 0.0);
            for (int component = 0; component < dimension; ++component) {
                for (int input_channel = 0; input_channel < width; ++input_channel) {
                    double value = aligned_gradient[static_cast<std::size_t>(
                        component * width + input_channel)];
                    for (int output_channel = 0; output_channel < width; ++output_channel) {
                        value += aligned_gradient[static_cast<std::size_t>(
                            component * width + output_channel)]
                            * static_cast<double>(matrix[input_channel * width + output_channel]);
                    }
                    grad_block[static_cast<std::size_t>(component * width + input_channel)] = value;
                }
            }
            (void)source;
        }
        const int moment_offset = degree_offsets[static_cast<std::size_t>(degree)];
        for (std::size_t index = 0; index < grad_block.size(); ++index) {
            normalized_moment_gradient[static_cast<std::size_t>(degree)][index] = grad_block[index];
        }
        (void)moment_offset;
    }
    normalized_moment_gradient[0] = block_gradient[0];

    Dpa4cReadoutAdjoints adjoints;
    adjoints.scalar.resize(static_cast<std::size_t>(options.channels));
    adjoints.angular.assign(static_cast<std::size_t>(degree_offsets.back()), 0.0);
    adjoints.divisor_scalar = direct_divisor_scalar;
    adjoints.divisor_angular = direct_divisor_angular;
    const double divisor_scalar_squared = divisor_scalar * divisor_scalar;
    for (int channel = 0; channel < options.channels; ++channel) {
        const double gradient = normalized_moment_gradient[0][static_cast<std::size_t>(channel)];
        adjoints.scalar[static_cast<std::size_t>(channel)] = gradient / divisor_scalar;
        adjoints.divisor_scalar -= gradient * reduced[static_cast<std::size_t>(2 + channel)]
            / divisor_scalar_squared;
    }
    const double divisor_angular_squared = divisor_angular * divisor_angular;
    for (int degree = 1; degree <= options.lmax; ++degree) {
        const int width = options.degree_channels[static_cast<std::size_t>(degree)];
        const int dimension = 2 * degree + 1;
        const int moment_offset = degree_offsets[static_cast<std::size_t>(degree)];
        for (int component = 0; component < dimension; ++component) {
            for (int channel = 0; channel < width; ++channel) {
                const std::size_t index = static_cast<std::size_t>(
                    component * width + channel);
                const double gradient = normalized_moment_gradient[
                    static_cast<std::size_t>(degree)][index];
                adjoints.angular[static_cast<std::size_t>(
                    moment_offset + static_cast<int>(index))] = gradient / divisor_angular;
                adjoints.divisor_angular -= gradient * reduced[static_cast<std::size_t>(
                    2 + moment_offset + static_cast<int>(index))]
                    / divisor_angular_squared;
            }
        }
    }
    return adjoints;
}

void angular_basis(const float x, const float y, const float z, int lmax, float* result) {
    const float squared_norm = x * x + y * y + z * z;
    const float x2 = x * x;
    const float y2 = y * y;
    const float z2 = z * z;
    std::fill(result, result + static_cast<std::size_t>((lmax + 1) * (lmax + 1)), 0.0F);
    result[0] = 1.0F;
    if (lmax >= 1) {
        result[1] = x;
        result[2] = y;
        result[3] = z;
    }
    if (lmax >= 2) {
        result[4] = kSqrt3 * x * y;
        result[5] = kSqrt3 * y * z;
        result[6] = 0.5F * (3.0F * z2 - squared_norm);
        result[7] = kSqrt3 * x * z;
        result[8] = 0.5F * kSqrt3 * (x2 - y2);
    }
    if (lmax >= 3) {
        result[9] = std::sqrt(5.0F / 8.0F) * y * (3.0F * x2 - y2);
        result[10] = std::sqrt(15.0F) * x * y * z;
        result[11] = std::sqrt(3.0F / 8.0F) * y * (5.0F * z2 - squared_norm);
        result[12] = 0.5F * z * (5.0F * z2 - 3.0F * squared_norm);
        result[13] = std::sqrt(3.0F / 8.0F) * x * (5.0F * z2 - squared_norm);
        result[14] = 0.5F * std::sqrt(15.0F) * z * (x2 - y2);
        result[15] = std::sqrt(5.0F / 8.0F) * x * (x2 - 3.0F * y2);
    }
    if (lmax >= 4) {
        const float x2_minus_y2 = x2 - y2;
        const float z4 = z2 * z2;
        const float squared_norm2 = squared_norm * squared_norm;
        result[16] = 0.5F * std::sqrt(35.0F) * x * y * x2_minus_y2;
        result[17] = 0.25F * std::sqrt(70.0F) * y * z * (3.0F * x2 - y2);
        result[18] = 0.5F * std::sqrt(5.0F) * x * y * (7.0F * z2 - squared_norm);
        result[19] = 0.25F * std::sqrt(10.0F) * y * z
            * (7.0F * z2 - 3.0F * squared_norm);
        result[20] = 0.125F * (35.0F * z4 - 30.0F * z2 * squared_norm
            + 3.0F * squared_norm2);
        result[21] = 0.25F * std::sqrt(10.0F) * x * z
            * (7.0F * z2 - 3.0F * squared_norm);
        result[22] = 0.25F * std::sqrt(5.0F) * x2_minus_y2
            * (7.0F * z2 - squared_norm);
        result[23] = 0.25F * std::sqrt(70.0F) * x * z * (x2 - 3.0F * y2);
        result[24] = 0.125F * std::sqrt(35.0F)
            * (x2 * x2 - 6.0F * x2 * y2 + y2 * y2);
    }
}

void packed_l2_to_stf(const float* packed, int rank, std::vector<float>& matrices) {
    matrices.resize(static_cast<std::size_t>(rank) * 9);
    for (int channel = 0; channel < rank; ++channel) {
        const float q0 = packed[channel];
        const float q1 = packed[rank + channel];
        const float q2 = packed[2 * rank + channel];
        const float q3 = packed[3 * rank + channel];
        const float q4 = packed[4 * rank + channel];
        const float qxy = q0 / kSqrt2;
        const float qyz = q1 / kSqrt2;
        const float qxz = q3 / kSqrt2;
        const float qxx = -q2 / kSqrt6 + q4 / kSqrt2;
        const float qyy = -q2 / kSqrt6 - q4 / kSqrt2;
        const float qzz = 2.0F * q2 / kSqrt6;
        float* matrix = matrices.data() + static_cast<std::size_t>(channel) * 9;
        matrix[0] = qxx;
        matrix[1] = qxy;
        matrix[2] = qxz;
        matrix[3] = qxy;
        matrix[4] = qyy;
        matrix[5] = qyz;
        matrix[6] = qxz;
        matrix[7] = qyz;
        matrix[8] = qzz;
    }
}

} // namespace

Dpa4cCalculator::Dpa4cCalculator(Dpa4cOptions options)
    : options_(std::move(options)) {
    validate_options(options_);

    degree_offsets_.assign(options_.degree_channels.size() + 1, 0);
    for (std::size_t degree = 0; degree < options_.degree_channels.size(); ++degree) {
        degree_offsets_[degree + 1] = degree_offsets_[degree]
            + (2 * static_cast<int>(degree) + 1) * options_.degree_channels[degree];
    }
    moment_count_ = degree_offsets_.back();

    gram_offsets_.push_back(0);
    for (int degree = 1; degree <= options_.lmax; ++degree) {
        const int width = options_.degree_channels[static_cast<std::size_t>(degree)];
        for (int row = 0; row < width; ++row) {
            for (int column = row; column < width; ++column) {
                gram_index_.push_back(static_cast<std::int64_t>(row * width + column));
                gram_scale_.push_back(row == column ? 1.0F : kSqrt2);
            }
        }
        gram_offsets_.push_back(static_cast<std::int64_t>(gram_index_.size()));
    }

    feature_count_ = static_cast<std::int64_t>(options_.output_mean.size());

    const int type_rows = options_.ntypes + 1;
    const std::size_t pair_count = static_cast<std::size_t>(type_rows) * type_rows;
    pair_cache_.resize(pair_count);
}

void Dpa4cCalculator::fill_pair_cache(
    const std::vector<std::size_t>& pair_indices) const {
    if (pair_indices.empty()) {
        return;
    }
    const int type_rows = options_.ntypes + 1;
    const int pair_output = options_.channels * (2 + options_.radial_modes);
    std::vector<float> input(static_cast<std::size_t>(2 * options_.channels));
    std::vector<float> pre_activation(static_cast<std::size_t>(2 * options_.pair_hidden));
    std::vector<float> hidden(static_cast<std::size_t>(options_.pair_hidden));
    std::vector<float> logits(static_cast<std::size_t>(pair_output));
    std::vector<double> affine_accumulators(static_cast<std::size_t>(
        std::max(2 * options_.pair_hidden, pair_output)));
    for (const std::size_t pair_index : pair_indices) {
        if (pair_index >= pair_cache_.size() || pair_cache_[pair_index]) {
            continue;
        }
        const int center_type = static_cast<int>(pair_index / static_cast<std::size_t>(type_rows));
        const int neighbor_type = static_cast<int>(pair_index % static_cast<std::size_t>(type_rows));
        const float* center = options_.type_embedding.data()
            + static_cast<std::size_t>(center_type * options_.channels);
        const float* neighbor = options_.type_embedding.data()
            + static_cast<std::size_t>(neighbor_type * options_.channels);
        std::copy(center, center + options_.channels, input.begin());
        std::copy(
            neighbor,
            neighbor + options_.channels,
            input.begin() + options_.channels);
        affine_values(
            options_.pair_w0,
            2 * options_.channels,
            2 * options_.pair_hidden,
            input.data(),
            pre_activation.data(),
            affine_accumulators.data());
        for (int index = 0; index < options_.pair_hidden; ++index) {
            const float gate = pre_activation[static_cast<std::size_t>(index)];
            const float value = pre_activation[static_cast<std::size_t>(
                options_.pair_hidden + index)];
            hidden[static_cast<std::size_t>(index)] = gate * sigmoid(gate) * value;
        }
        affine_values(
            options_.pair_w1,
            options_.pair_hidden,
            pair_output,
            hidden.data(),
            logits.data(),
            affine_accumulators.data());
        for (float& value : logits) {
            value *= 0.1F;
        }
        auto coefficients = std::make_unique<PairCoefficients>();
        coefficients->scale.resize(static_cast<std::size_t>(options_.channels));
        coefficients->shift.resize(static_cast<std::size_t>(options_.channels));
        coefficients->mixing.resize(
            static_cast<std::size_t>(options_.channels)
            * static_cast<std::size_t>(options_.radial_modes));
        for (int channel = 0; channel < options_.channels; ++channel) {
            coefficients->scale[static_cast<std::size_t>(channel)] =
                1.0F + std::tanh(logits[static_cast<std::size_t>(channel)]);
            coefficients->shift[static_cast<std::size_t>(channel)] =
                center[channel] + neighbor[channel]
                + std::tanh(logits[static_cast<std::size_t>(options_.channels + channel)]);
        }
        for (int channel = 0; channel < options_.channels; ++channel) {
            for (int mode = 0; mode < options_.radial_modes; ++mode) {
                const int logit_index = 2 * options_.channels
                    + channel * options_.radial_modes + mode;
                coefficients->mixing[static_cast<std::size_t>(
                    channel * options_.radial_modes + mode)] =
                    std::tanh(logits[static_cast<std::size_t>(logit_index)]);
            }
        }
        pair_cache_[pair_index] = std::move(coefficients);
    }
}

std::int64_t Dpa4cCalculator::feature_count() const noexcept {
    return feature_count_;
}

void Dpa4cCalculator::compute(
    const StructureBatchView& batch,
    const std::int32_t* type_indices,
    double* output,
    const std::shared_ptr<ComputeControl>& control) const {
    compute_impl(batch, type_indices, output, nullptr, nullptr, nullptr, control);
}

void Dpa4cCalculator::predict(
    const StructureBatchView& batch,
    const std::int32_t* type_indices,
    double* energy,
    double* atom_energy,
    double* forces,
    const std::shared_ptr<ComputeControl>& control) const {
    if (energy == nullptr && batch.structures > 0) {
        throw std::invalid_argument("DPA4C energy output cannot be null");
    }
    if (batch.atoms > 0 && (atom_energy == nullptr || forces == nullptr)) {
        throw std::invalid_argument("DPA4C atomic energy and force outputs cannot be null");
    }
    compute_impl(batch, type_indices, nullptr, energy, atom_energy, forces, control);
}

void Dpa4cCalculator::compute_impl(
    const StructureBatchView& batch,
    const std::int32_t* type_indices,
    double* output,
    double* energy,
    double* atom_energy,
    double* forces,
    const std::shared_ptr<ComputeControl>& control) const {
    const bool predicting = energy != nullptr || atom_energy != nullptr || forces != nullptr;
    std::lock_guard<std::mutex> lock(compute_mutex_);
    assert_open(predicting ? "DPA4C predictor" : "DPA4C descriptor");
    detail::validate_batch(batch);
    if (type_indices == nullptr && batch.atoms > 0) {
        throw std::invalid_argument("DPA4C type indices cannot be null");
    }
    for (std::int64_t atom = 0; atom < batch.atoms; ++atom) {
        const std::int32_t type = type_indices[atom];
        if (type < 0 || type >= options_.ntypes) {
            throw std::invalid_argument("DPA4C type index is outside the checkpoint type map");
        }
    }
    if (predicting) {
        validate_fitting_options(options_, feature_count_);
        if (batch.atoms > 0 && (atom_energy == nullptr || forces == nullptr)) {
            throw std::invalid_argument("DPA4C atomic energy and force outputs cannot be null");
        }
        if (batch.atoms > 0) {
            std::fill(forces, forces + static_cast<std::size_t>(batch.atoms) * 3U, 0.0);
        }
    }
    if (batch.atoms == 0) {
        if (energy != nullptr && batch.structures > 0) {
            std::fill(energy, energy + batch.structures, 0.0);
        }
        detail::mark_completed_structures(control, batch.structures);
        return;
    }

    const std::vector<double> wrapped = detail::normalized_positions(batch);
    StructureBatchView normalized_batch = batch;
    normalized_batch.positions = wrapped.data();
    const NeighborGraph graph = build_neighbor_graph(
        normalized_batch,
        options_.rcut,
        control,
        options_.num_threads,
        true,
        false,
        true);

    // Most inputs use a small subset of the checkpoint type map.  Resolve
    // only the ordered type pairs that are present in this graph, in a stable
    // order, before entering the parallel reduction below.
    const std::size_t type_rows = static_cast<std::size_t>(options_.ntypes + 1);
    std::vector<std::size_t> used_pair_indices;
    for (std::int64_t center_atom = 0; center_atom < batch.atoms; ++center_atom) {
        const NeighborView neighbors = graph.for_center(center_atom);
        const std::size_t center_type = static_cast<std::size_t>(type_indices[center_atom]);
        for (std::size_t edge = 0; edge < neighbors.size; ++edge) {
            if (neighbors.exact_self(edge, center_atom)) {
                continue;
            }
            const std::size_t neighbor_type = static_cast<std::size_t>(
                type_indices[neighbors.atoms[edge]]);
            used_pair_indices.push_back(center_type * type_rows + neighbor_type);
        }
    }
    std::sort(used_pair_indices.begin(), used_pair_indices.end());
    used_pair_indices.erase(
        std::unique(used_pair_indices.begin(), used_pair_indices.end()),
        used_pair_indices.end());
    fill_pair_cache(used_pair_indices);

    const int angular_width = (options_.lmax + 1) * (options_.lmax + 1);
    std::vector<std::int64_t> structure_for_atom;
    std::unique_ptr<std::atomic<std::int64_t>[]> remaining_atoms;
    if (control) {
        structure_for_atom.resize(static_cast<std::size_t>(batch.atoms));
        remaining_atoms = std::make_unique<std::atomic<std::int64_t>[]>(
            static_cast<std::size_t>(batch.structures));
        for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
            const std::int64_t begin = batch.offsets[structure];
            const std::int64_t end = batch.offsets[structure + 1];
            remaining_atoms[static_cast<std::size_t>(structure)].store(
                end - begin, std::memory_order_relaxed);
            for (std::int64_t atom = begin; atom < end; ++atom) {
                structure_for_atom[static_cast<std::size_t>(atom)] = structure;
            }
            if (begin == end) {
                detail::check_cancelled(control);
                control->mark_completed();
            }
        }
    }
#ifdef _OPENMP
#pragma omp parallel num_threads(detail::resolved_thread_count(options_.num_threads))
#endif
    {
        // Per-thread work buffers, allocated once and reused for every
        // atom.  Only `reduced` accumulates across the edge loop and needs
        // a per-atom reset; every other buffer is fully rewritten before it
        // is read (affine_values zeroes its own accumulators, and the
        // moment writes cover every entry).
        std::vector<double> reduced(static_cast<std::size_t>(2 + moment_count_), 0.0);
        std::vector<float> radial_basis(static_cast<std::size_t>(options_.n_radial));
        std::vector<float> radial_pre(static_cast<std::size_t>(2 * options_.radial_hidden));
        std::vector<float> radial_hidden(static_cast<std::size_t>(options_.radial_hidden));
        std::vector<float> radial(static_cast<std::size_t>(options_.channels));
        std::vector<float> modes(static_cast<std::size_t>(options_.radial_modes));
        std::vector<double> affine_accumulators(static_cast<std::size_t>(std::max(
            2 * options_.radial_hidden,
            std::max(options_.channels, options_.radial_modes))));
        std::vector<float> basis(static_cast<std::size_t>(angular_width));
        std::vector<double> amplitudes(static_cast<std::size_t>(options_.channels));
        std::vector<float> moments(static_cast<std::size_t>(moment_count_), 0.0F);
        std::vector<std::size_t> edge_order;
        std::vector<std::int32_t> edge_atoms;
        const std::size_t edge_state_stride = static_cast<std::size_t>(
            4 + options_.channels + angular_width);
        std::vector<double> edge_states;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (std::int64_t center_atom = 0; center_atom < batch.atoms; ++center_atom) {
        if (control && control->cancelled()) {
            continue;
        }
        const int center_type = type_indices[center_atom];
        // Accumulate destination reductions in fp64 while keeping the
        // checkpoint/activation storage in fp32.  A center can have many
        // neighbors; rounding every contribution in fp32 was the dominant
        // source of the residual DPA4C parity error.
        std::fill(reduced.begin(), reduced.end(), 0.0);
        edge_atoms.clear();
        edge_states.clear();
        const NeighborView neighbors = graph.for_center(center_atom);
        // The Python dense builder presents each destination row in ascending
        // distance order.  Match that order before the moment reduction so
        // fp32 edge features and the final gram/bispectrum contractions do not
        // depend on the cell-list traversal order.
        edge_order.resize(neighbors.size);
        std::iota(edge_order.begin(), edge_order.end(), std::size_t{0});
        std::stable_sort(edge_order.begin(), edge_order.end(), [&neighbors](
            std::size_t lhs, std::size_t rhs) {
            return neighbors.distance2[lhs] < neighbors.distance2[rhs];
        });
        for (const std::size_t edge : edge_order) {
            if (neighbors.exact_self(edge, center_atom)) {
                continue;
            }
            const std::int32_t neighbor_atom = neighbors.atoms[edge];
            const int neighbor_type = type_indices[neighbor_atom];
            const float dx = static_cast<float>(neighbors.displacements[edge * 3 + 0]);
            const float dy = static_cast<float>(neighbors.displacements[edge * 3 + 1]);
            const float dz = static_cast<float>(neighbors.displacements[edge * 3 + 2]);
            const float distance_squared = dx * dx + dy * dy + dz * dz;
            const float distance = std::sqrt(distance_squared + kEpsilon * kEpsilon);
            const float ux = dx / distance;
            const float uy = dy / distance;
            const float uz = dz / distance;

            float cutoff_coordinate = (static_cast<float>(options_.rcut) - distance)
                / static_cast<float>(options_.rcut);
            cutoff_coordinate = std::max(0.0F, std::min(1.0F, cutoff_coordinate));
            const float x = 1.0F - cutoff_coordinate;
            float series = 35.0F;
            series = 20.0F + x * series;
            series = 10.0F + x * series;
            series = 4.0F + x * series;
            series = 1.0F + x * series;
            const float envelope = cutoff_coordinate * cutoff_coordinate
                * cutoff_coordinate * cutoff_coordinate * series;

            for (int radial_index = 0; radial_index < options_.n_radial; ++radial_index) {
                const float frequency = options_.radial_freqs[static_cast<std::size_t>(radial_index)];
                const float argument = distance * frequency;
                constexpr float pi = 3.1415927410125732422F;
                const float sinc_argument = argument / pi;
                const float sinc = sinc_argument == 0.0F
                    ? 1.0F : std::sin(pi * sinc_argument) / (pi * sinc_argument);
                radial_basis[static_cast<std::size_t>(radial_index)] = frequency * sinc;
            }
            affine_values(
                options_.radial_w0,
                options_.n_radial,
                2 * options_.radial_hidden,
                radial_basis.data(),
                radial_pre.data(),
                affine_accumulators.data());
            for (int hidden_index = 0; hidden_index < options_.radial_hidden; ++hidden_index) {
                const float gate = radial_pre[static_cast<std::size_t>(hidden_index)];
                const float value = radial_pre[static_cast<std::size_t>(
                    options_.radial_hidden + hidden_index)];
                radial_hidden[static_cast<std::size_t>(hidden_index)] =
                    gate * sigmoid(gate) * value;
            }
            affine_values(
                options_.radial_w1,
                options_.radial_hidden,
                options_.channels,
                radial_hidden.data(),
                radial.data(),
                affine_accumulators.data());
            affine_values(
                options_.radial_mode_w,
                options_.radial_hidden,
                options_.radial_modes,
                radial_hidden.data(),
                modes.data(),
                affine_accumulators.data());

            const std::size_t pair_index = static_cast<std::size_t>(
                center_type * (options_.ntypes + 1) + neighbor_type);
            const PairCoefficients& pair = *pair_cache_[pair_index];
            angular_basis(ux, uy, uz, options_.lmax, basis.data());

            reduced[0] += static_cast<double>(envelope) * static_cast<double>(envelope);
            const float envelope_squared = envelope * envelope;
            reduced[1] += static_cast<double>(envelope_squared)
                * static_cast<double>(envelope_squared);
            for (int channel = 0; channel < options_.channels; ++channel) {
                double raw_amplitude = static_cast<double>(radial[
                    static_cast<std::size_t>(channel)])
                    * static_cast<double>(pair.scale[static_cast<std::size_t>(channel)])
                    + static_cast<double>(pair.shift[static_cast<std::size_t>(channel)]);
                for (int mode = 0; mode < options_.radial_modes; ++mode) {
                    raw_amplitude += static_cast<double>(pair.mixing[
                        static_cast<std::size_t>(channel * options_.radial_modes + mode)])
                        * static_cast<double>(modes[static_cast<std::size_t>(mode)]);
                }
                amplitudes[static_cast<std::size_t>(channel)] = raw_amplitude;
                reduced[static_cast<std::size_t>(2 + channel)] +=
                    raw_amplitude * static_cast<double>(envelope);
            }
            for (int degree = 1; degree <= options_.lmax; ++degree) {
                const int width = options_.degree_channels[static_cast<std::size_t>(degree)];
                const int basis_offset = degree * degree;
                const int moment_offset = degree_offsets_[static_cast<std::size_t>(degree)];
                for (int component = 0; component < 2 * degree + 1; ++component) {
                    for (int channel = 0; channel < width; ++channel) {
                        reduced[static_cast<std::size_t>(2 + moment_offset
                            + component * width + channel)] += amplitudes[
                                static_cast<std::size_t>(channel)] * envelope_squared
                            * static_cast<double>(basis[
                                static_cast<std::size_t>(basis_offset + component)]);
                    }
                }
            }
            if (predicting) {
                edge_atoms.push_back(neighbor_atom);
                const std::size_t offset = edge_states.size();
                edge_states.resize(offset + edge_state_stride);
                edge_states[offset + 0] = dx;
                edge_states[offset + 1] = dy;
                edge_states[offset + 2] = dz;
                edge_states[offset + 3] = envelope;
                for (int channel = 0; channel < options_.channels; ++channel) {
                    edge_states[offset + static_cast<std::size_t>(4 + channel)] =
                        amplitudes[static_cast<std::size_t>(channel)];
                }
                for (int basis_index = 0; basis_index < angular_width; ++basis_index) {
                    edge_states[offset + static_cast<std::size_t>(
                        4 + options_.channels + basis_index)] =
                        basis[static_cast<std::size_t>(basis_index)];
                }
            }
        }

        const double divisor_scalar = std::sqrt(reduced[0] + static_cast<double>(kNormFloor));
        const double divisor_angular = std::sqrt(reduced[1] + static_cast<double>(kNormFloor));
        const int scalar_width = options_.channels;
        for (int channel = 0; channel < scalar_width; ++channel) {
            moments[static_cast<std::size_t>(channel)] = static_cast<float>(
                reduced[static_cast<std::size_t>(2 + channel)] / divisor_scalar);
        }
        for (int degree = 1; degree <= options_.lmax; ++degree) {
            const int width = options_.degree_channels[static_cast<std::size_t>(degree)];
            const int offset = degree_offsets_[static_cast<std::size_t>(degree)];
            for (int component = 0; component < 2 * degree + 1; ++component) {
                for (int channel = 0; channel < width; ++channel) {
                    moments[static_cast<std::size_t>(offset + component * width + channel)] =
                        static_cast<float>(reduced[static_cast<std::size_t>(
                            2 + offset + component * width + channel)] / divisor_angular);
                }
            }
        }

        std::vector<std::vector<float>> blocks;
        std::vector<std::vector<float>> projected;
        blocks.reserve(static_cast<std::size_t>(options_.lmax + 1));
        projected.reserve(static_cast<std::size_t>(options_.lmax));
        for (int degree = 0; degree <= options_.lmax; ++degree) {
            const int width = options_.degree_channels[static_cast<std::size_t>(degree)];
            const int dimension = 2 * degree + 1;
            std::vector<float> block(static_cast<std::size_t>(dimension * width));
            const int offset = degree_offsets_[static_cast<std::size_t>(degree)];
            std::copy(
                moments.begin() + offset,
                moments.begin() + offset + dimension * width,
                block.begin());
            if (degree == 1 || degree == 2) {
                const auto matrix_begin = options_.readout_alignment_offsets[
                    static_cast<std::size_t>(degree - 1)];
                const float* matrix = options_.readout_alignment.data() + matrix_begin;
                std::vector<float> aligned = block;
                for (int component = 0; component < dimension; ++component) {
                    for (int output_channel = 0; output_channel < width; ++output_channel) {
                        double value = static_cast<double>(block[
                            static_cast<std::size_t>(component * width + output_channel)]);
                        for (int input_channel = 0; input_channel < width; ++input_channel) {
                            value += static_cast<double>(block[static_cast<std::size_t>(
                                component * width + input_channel)])
                                * static_cast<double>(matrix[static_cast<std::size_t>(
                                    input_channel * width + output_channel)]);
                        }
                        aligned[static_cast<std::size_t>(component * width + output_channel)] =
                            static_cast<float>(value);
                    }
                }
                block = std::move(aligned);
            }
            blocks.push_back(std::move(block));
        }
        for (int degree = 1; degree <= options_.lmax; ++degree) {
            const int width = options_.degree_channels[static_cast<std::size_t>(degree)];
            const int rank = options_.bispectrum_ranks[static_cast<std::size_t>(degree - 1)];
            const int dimension = 2 * degree + 1;
            const auto matrix_begin = options_.readout_projection_offsets[
                static_cast<std::size_t>(degree - 1)];
            const auto matrix_end = options_.readout_projection_offsets[
                static_cast<std::size_t>(degree)];
            std::vector<float> block(static_cast<std::size_t>(dimension * rank), 0.0F);
            if (matrix_begin == matrix_end) {
                block = blocks[static_cast<std::size_t>(degree)];
            } else {
                const float* matrix = options_.readout_projections.data() + matrix_begin;
                for (int component = 0; component < dimension; ++component) {
                    for (int output_channel = 0; output_channel < rank; ++output_channel) {
                        double value = 0.0;
                        for (int input_channel = 0; input_channel < width; ++input_channel) {
                            value += static_cast<double>(blocks[static_cast<std::size_t>(degree)][
                                static_cast<std::size_t>(component * width + input_channel)])
                                * static_cast<double>(matrix[static_cast<std::size_t>(
                                    input_channel * rank + output_channel)]);
                        }
                        block[static_cast<std::size_t>(component * rank + output_channel)] =
                            static_cast<float>(value);
                    }
                }
            }
            projected.push_back(std::move(block));
        }

        std::vector<float> descriptor;
        descriptor.reserve(static_cast<std::size_t>(feature_count_));
        descriptor.insert(descriptor.end(), blocks[0].begin(), blocks[0].end());
        for (int degree = 1; degree <= options_.lmax; ++degree) {
            const int width = options_.degree_channels[static_cast<std::size_t>(degree)];
            const int dimension = 2 * degree + 1;
            const auto gram_begin = gram_offsets_[static_cast<std::size_t>(degree - 1)];
            const auto gram_end = gram_offsets_[static_cast<std::size_t>(degree)];
            for (std::int64_t gram = gram_begin; gram < gram_end; ++gram) {
                const int flat = static_cast<int>(gram_index_[static_cast<std::size_t>(gram)]);
                const int row = flat / width;
                const int column = flat % width;
                double value = 0.0;
                for (int component = 0; component < dimension; ++component) {
                    value += static_cast<double>(blocks[static_cast<std::size_t>(degree)][
                        static_cast<std::size_t>(component * width + row)])
                        * static_cast<double>(blocks[static_cast<std::size_t>(degree)][
                            static_cast<std::size_t>(component * width + column)]);
                }
                descriptor.push_back(static_cast<float>(value * static_cast<double>(
                    gram_scale_[static_cast<std::size_t>(gram)])));
            }
        }

        for (std::size_t triple_index = 0;
             triple_index < options_.degree_triples.size() / 3;
             ++triple_index) {
            const int degree_1 = options_.degree_triples[triple_index * 3 + 0];
            const int degree_2 = options_.degree_triples[triple_index * 3 + 1];
            const int degree_3 = options_.degree_triples[triple_index * 3 + 2];
            const int dimension_1 = 2 * degree_1 + 1;
            const int dimension_2 = 2 * degree_2 + 1;
            const int dimension_3 = 2 * degree_3 + 1;
            const int rank_1 = options_.bispectrum_ranks[static_cast<std::size_t>(degree_1 - 1)];
            const int rank_2 = options_.bispectrum_ranks[static_cast<std::size_t>(degree_2 - 1)];
            const int rank_3 = options_.bispectrum_ranks[static_cast<std::size_t>(degree_3 - 1)];
            const auto coupling_begin = options_.coupling_offsets[triple_index];
            const auto probe_begin = options_.probe_offsets[triple_index];
            const auto probe_end = options_.probe_offsets[triple_index + 1];
            std::vector<float> full(static_cast<std::size_t>(rank_1 * rank_2 * rank_3), 0.0F);
            if (degree_1 == 1 && degree_2 == 1 && degree_3 == 2) {
                const std::vector<float>& vector_block = projected[0];
                const std::vector<float>& tensor_block = projected[1];
                std::vector<float> matrices;
                packed_l2_to_stf(tensor_block.data(), rank_3, matrices);
                for (int first = 0; first < rank_1; ++first) {
                    for (int second = 0; second < rank_2; ++second) {
                        for (int tensor = 0; tensor < rank_3; ++tensor) {
                            const float* matrix = matrices.data() + static_cast<std::size_t>(tensor * 9);
                            const float vx = vector_block[static_cast<std::size_t>(first)];
                            const float vy = vector_block[static_cast<std::size_t>(rank_1 + first)];
                            const float vz = vector_block[static_cast<std::size_t>(2 * rank_1 + first)];
                            const float wx = vector_block[static_cast<std::size_t>(second)];
                            const float wy = vector_block[static_cast<std::size_t>(rank_1 + second)];
                            const float wz = vector_block[static_cast<std::size_t>(2 * rank_1 + second)];
                            const double mx = static_cast<double>(matrix[0]) * wx
                                + static_cast<double>(matrix[1]) * wy
                                + static_cast<double>(matrix[2]) * wz;
                            const double my = static_cast<double>(matrix[3]) * wx
                                + static_cast<double>(matrix[4]) * wy
                                + static_cast<double>(matrix[5]) * wz;
                            const double mz = static_cast<double>(matrix[6]) * wx
                                + static_cast<double>(matrix[7]) * wy
                                + static_cast<double>(matrix[8]) * wz;
                            full[static_cast<std::size_t>((first * rank_2 + second) * rank_3 + tensor)] =
                                static_cast<float>(-(static_cast<double>(vx) * mx
                                    + static_cast<double>(vy) * my
                                    + static_cast<double>(vz) * mz)
                                    / static_cast<double>(kSqrt5));
                        }
                    }
                }
            } else {
                const float* coupling = options_.bispectrum_coupling.data() + coupling_begin;
                for (int first = 0; first < rank_1; ++first) {
                    for (int second = 0; second < rank_2; ++second) {
                        for (int third = 0; third < rank_3; ++third) {
                            double value = 0.0;
                            for (int i = 0; i < dimension_1; ++i) {
                                for (int j = 0; j < dimension_2; ++j) {
                                    for (int k = 0; k < dimension_3; ++k) {
                                        value += static_cast<double>(coupling[
                                            (i * dimension_2 + j) * dimension_3 + k])
                                            * static_cast<double>(projected[
                                                static_cast<std::size_t>(degree_1 - 1)][
                                                static_cast<std::size_t>(i * rank_1 + first)])
                                            * static_cast<double>(projected[
                                                static_cast<std::size_t>(degree_2 - 1)][
                                                static_cast<std::size_t>(j * rank_2 + second)])
                                            * static_cast<double>(projected[
                                                static_cast<std::size_t>(degree_3 - 1)][
                                                static_cast<std::size_t>(k * rank_3 + third)]);
                                    }
                                }
                            }
                            full[static_cast<std::size_t>((first * rank_2 + second) * rank_3 + third)] =
                                static_cast<float>(value);
                        }
                    }
                }
            }
            for (std::int64_t probe = probe_begin; probe < probe_end; ++probe) {
                const auto index = options_.probe_index[static_cast<std::size_t>(probe)];
                if (index < 0 || index >= static_cast<std::int64_t>(full.size())) {
                    throw std::runtime_error("DPA4C probe index is outside its contraction");
                }
                descriptor.push_back(
                    full[static_cast<std::size_t>(index)]
                    * options_.probe_scale[static_cast<std::size_t>(probe)]);
            }
        }

        const std::vector<float>& vector_block = projected[0];
        const std::vector<float>& tensor_block = projected[1];
        const int vector_rank = options_.bispectrum_ranks[0];
        const int tensor_rank = options_.bispectrum_ranks[1];
        std::vector<float> matrices;
        packed_l2_to_stf(tensor_block.data(), tensor_rank, matrices);
        for (int tensor = 0; tensor < tensor_rank; ++tensor) {
            const float* matrix = matrices.data() + static_cast<std::size_t>(tensor * 9);
            for (int vector = 0; vector < vector_rank; ++vector) {
                const float vx = vector_block[static_cast<std::size_t>(vector)];
                const float vy = vector_block[static_cast<std::size_t>(vector_rank + vector)];
                const float vz = vector_block[static_cast<std::size_t>(2 * vector_rank + vector)];
                const double mx = static_cast<double>(matrix[0]) * vx
                    + static_cast<double>(matrix[1]) * vy
                    + static_cast<double>(matrix[2]) * vz;
                const double my = static_cast<double>(matrix[3]) * vx
                    + static_cast<double>(matrix[4]) * vy
                    + static_cast<double>(matrix[5]) * vz;
                const double mz = static_cast<double>(matrix[6]) * vx
                    + static_cast<double>(matrix[7]) * vy
                    + static_cast<double>(matrix[8]) * vz;
                descriptor.push_back(static_cast<float>(mx * mx + my * my + mz * mz));
            }
        }
        descriptor.push_back(divisor_scalar);
        descriptor.push_back(divisor_angular);
        const float* center_embedding = options_.type_embedding.data()
            + static_cast<std::size_t>(center_type * options_.channels);
        descriptor.insert(descriptor.end(), center_embedding, center_embedding + options_.channels);
        if (descriptor.size() != static_cast<std::size_t>(feature_count_)) {
            throw std::runtime_error("DPA4C native readout produced an unexpected feature count");
        }
        if (output != nullptr) {
            double* destination = output + center_atom * feature_count_;
            for (std::size_t feature = 0; feature < descriptor.size(); ++feature) {
                double value = static_cast<double>(descriptor[feature]);
                if (options_.calibrate) {
                    value = (value - static_cast<double>(options_.output_mean[feature]))
                        / static_cast<double>(options_.output_stddev[feature]);
                }
                destination[feature] = value;
            }
        }
        if (predicting) {
            std::vector<float> fitting_input(descriptor.size());
            std::vector<float> fitting_gradient(descriptor.size());
            std::vector<double> feature_gradient(descriptor.size());
            for (std::size_t index = 0; index < descriptor.size(); ++index) {
                double value = static_cast<double>(descriptor[index]);
                if (options_.calibrate) {
                    value = (value - static_cast<double>(options_.output_mean[index]))
                        / static_cast<double>(options_.output_stddev[index]);
                }
                fitting_input[index] = static_cast<float>(value);
            }
            atom_energy[center_atom] = fitting_energy_and_gradient(
                options_, fitting_input.data(), fitting_gradient.data(), center_type);
            for (std::size_t index = 0; index < descriptor.size(); ++index) {
                feature_gradient[index] = static_cast<double>(fitting_gradient[index]);
                if (options_.calibrate) {
                    feature_gradient[index] /= static_cast<double>(options_.output_stddev[index]);
                }
            }
            const Dpa4cReadoutAdjoints adjoints = backprop_readout(
                options_, degree_offsets_, gram_offsets_, gram_index_, gram_scale_,
                feature_gradient, blocks, projected, reduced, divisor_scalar,
                divisor_angular);
            std::vector<float> edge_gradient(3);
            for (std::size_t edge = 0; edge < edge_atoms.size(); ++edge) {
                const std::size_t state_offset = edge * edge_state_stride;
                const std::int32_t neighbor_atom = edge_atoms[edge];
                const std::size_t pair_index = static_cast<std::size_t>(
                    center_type * (options_.ntypes + 1) + type_indices[neighbor_atom]);
                const PairCoefficients& pair = *pair_cache_[pair_index];
                edge_derivatives(
                    options_, pair.scale, pair.shift, pair.mixing, center_type,
                    type_indices[neighbor_atom],
                    static_cast<float>(edge_states[state_offset + 0]),
                    static_cast<float>(edge_states[state_offset + 1]),
                    static_cast<float>(edge_states[state_offset + 2]),
                    edge_states.data() + state_offset, adjoints.scalar.data(),
                    adjoints.angular.data(), divisor_scalar, divisor_angular,
                    adjoints.divisor_scalar, adjoints.divisor_angular,
                    edge_gradient.data());
                for (int axis = 0; axis < 3; ++axis) {
                    const double gradient = static_cast<double>(edge_gradient[
                        static_cast<std::size_t>(axis)]);
#ifdef _OPENMP
#pragma omp atomic update
#endif
                    forces[static_cast<std::size_t>(neighbor_atom) * 3U
                        + static_cast<std::size_t>(axis)] -= gradient;
#ifdef _OPENMP
#pragma omp atomic update
#endif
                    forces[static_cast<std::size_t>(center_atom) * 3U
                        + static_cast<std::size_t>(axis)] += gradient;
                }
            }
        }
        if (control) {
            const std::int64_t structure = structure_for_atom[
                static_cast<std::size_t>(center_atom)];
            if (remaining_atoms[static_cast<std::size_t>(structure)].fetch_sub(
                    1, std::memory_order_acq_rel) == 1) {
                control->mark_completed();
            }
        }
    }
    }
    if (control && control->cancelled()) {
        throw CancelledError();
    }
    if (predicting) {
        for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
            double total = 0.0;
            for (std::int64_t atom = batch.offsets[structure];
                 atom < batch.offsets[structure + 1]; ++atom) {
                total += atom_energy[atom];
            }
            energy[structure] = total;
        }
    }
}

} // namespace mdescriptor
