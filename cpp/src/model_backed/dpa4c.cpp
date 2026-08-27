#include "mdescriptor/dpa4c.hpp"

#include "mdescriptor/detail/math3.hpp"
#include "mdescriptor/neighbor.hpp"

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

float sigmoid(float value) {
    return 1.0F / (1.0F + std::exp(-value));
}

float affine_value(
    const std::vector<float>& weights,
    int input_width,
    int output_width,
    const float* input,
    int output
) {
    double result = 0.0;
    for (int input_index = 0; input_index < input_width; ++input_index) {
        result += static_cast<double>(input[input_index])
            * static_cast<double>(weights[static_cast<std::size_t>(
                input_index * output_width + output)]);
    }
    return static_cast<float>(result);
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

Mat3 load_cell(const StructureBatchView& batch, std::int64_t structure) {
    Mat3 cell;
    const double* source = batch.cells + structure * 9;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            cell.a[row][column] = source[row * 3 + column];
        }
    }
    return cell;
}

Vec3 load_position(const double* positions, std::int64_t atom) {
    const double* source = positions + atom * 3;
    return {source[0], source[1], source[2]};
}

std::vector<double> normalized_positions(const StructureBatchView& batch) {
    std::vector<double> positions(
        static_cast<std::size_t>(batch.atoms) * 3,
        0.0);
    if (batch.atoms > 0) {
        std::copy(
            batch.positions,
            batch.positions + static_cast<std::size_t>(batch.atoms) * 3,
            positions.begin());
    }
    for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
        bool periodic = true;
        for (int axis = 0; axis < 3; ++axis) {
            periodic = periodic && batch.pbc[structure * 3 + axis] == 1;
        }
        if (!periodic) {
            continue;
        }
        const Mat3 cell = load_cell(batch, structure);
        Mat3 inverse;
        const bool diagonal = cell.a[0][1] == 0.0 && cell.a[0][2] == 0.0
            && cell.a[1][0] == 0.0 && cell.a[1][2] == 0.0
            && cell.a[2][0] == 0.0 && cell.a[2][1] == 0.0;
        if (diagonal) {
            for (int axis = 0; axis < 3; ++axis) {
                inverse.a[axis][axis] = 1.0 / cell.a[axis][axis];
            }
        } else {
            inverse = detail::inverse(cell);
        }
        const std::int64_t begin = batch.offsets[structure];
        const std::int64_t end = batch.offsets[structure + 1];
        for (std::int64_t atom = begin; atom < end; ++atom) {
            const Vec3 point = load_position(batch.positions, atom);
            const Vec3 fractional{
                point.x * inverse.a[0][0] + point.y * inverse.a[1][0]
                    + point.z * inverse.a[2][0],
                point.x * inverse.a[0][1] + point.y * inverse.a[1][1]
                    + point.z * inverse.a[2][1],
                point.x * inverse.a[0][2] + point.y * inverse.a[1][2]
                    + point.z * inverse.a[2][2],
            };
            const Vec3 wrapped{
                fractional.x - std::floor(fractional.x),
                fractional.y - std::floor(fractional.y),
                fractional.z - std::floor(fractional.z),
            };
            const Vec3 cartesian = wrapped.x * detail::row(cell, 0)
                + wrapped.y * detail::row(cell, 1)
                + wrapped.z * detail::row(cell, 2);
            positions[static_cast<std::size_t>(atom * 3 + 0)] = cartesian.x;
            positions[static_cast<std::size_t>(atom * 3 + 1)] = cartesian.y;
            positions[static_cast<std::size_t>(atom * 3 + 2)] = cartesian.z;
        }
    }
    return positions;
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

void Dpa4cCalculator::ensure_pair_cache(
    const std::vector<std::size_t>& pair_indices) const {
    if (pair_indices.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(compute_mutex_);
    const int type_rows = options_.ntypes + 1;
    const int pair_output = options_.channels * (2 + options_.radial_modes);
    std::vector<float> input(static_cast<std::size_t>(2 * options_.channels));
    std::vector<float> pre_activation(static_cast<std::size_t>(2 * options_.pair_hidden));
    std::vector<float> hidden(static_cast<std::size_t>(options_.pair_hidden));
    std::vector<float> logits(static_cast<std::size_t>(pair_output));
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
        for (int output = 0; output < 2 * options_.pair_hidden; ++output) {
            pre_activation[static_cast<std::size_t>(output)] = affine_value(
                options_.pair_w0,
                2 * options_.channels,
                2 * options_.pair_hidden,
                input.data(),
                output);
        }
        for (int index = 0; index < options_.pair_hidden; ++index) {
            const float gate = pre_activation[static_cast<std::size_t>(index)];
            const float value = pre_activation[static_cast<std::size_t>(
                options_.pair_hidden + index)];
            hidden[static_cast<std::size_t>(index)] = gate * sigmoid(gate) * value;
        }
        for (int output = 0; output < pair_output; ++output) {
            logits[static_cast<std::size_t>(output)] = 0.1F * affine_value(
                options_.pair_w1,
                options_.pair_hidden,
                pair_output,
                hidden.data(),
                output);
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

void Dpa4cCalculator::close() noexcept {
    closed_.store(true, std::memory_order_release);
}

bool Dpa4cCalculator::closed() const noexcept {
    return closed_.load(std::memory_order_acquire);
}

void Dpa4cCalculator::compute(
    const StructureBatchView& batch,
    const std::int32_t* type_indices,
    double* output,
    const std::shared_ptr<ComputeControl>& control) const {
    if (closed()) {
        throw std::runtime_error("DPA4C descriptor is closed");
    }
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
    if (batch.atoms == 0) {
        return;
    }

    const std::vector<double> wrapped = normalized_positions(batch);
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
    ensure_pair_cache(used_pair_indices);

    const int angular_width = (options_.lmax + 1) * (options_.lmax + 1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options_.num_threads > 0 ? options_.num_threads : omp_get_max_threads())
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
        std::vector<double> reduced(static_cast<std::size_t>(2 + moment_count_), 0.0);
        std::vector<float> radial_basis(static_cast<std::size_t>(options_.n_radial));
        std::vector<float> radial_pre(static_cast<std::size_t>(2 * options_.radial_hidden));
        std::vector<float> radial_hidden(static_cast<std::size_t>(options_.radial_hidden));
        std::vector<float> radial(static_cast<std::size_t>(options_.channels));
        std::vector<float> modes(static_cast<std::size_t>(options_.radial_modes));
        std::vector<float> basis(static_cast<std::size_t>(angular_width));
        std::vector<double> amplitudes(static_cast<std::size_t>(options_.channels));
        const NeighborView neighbors = graph.for_center(center_atom);
        // The Python dense builder presents each destination row in ascending
        // distance order.  Match that order before the moment reduction so
        // fp32 edge features and the final gram/bispectrum contractions do not
        // depend on the cell-list traversal order.
        std::vector<std::size_t> edge_order(neighbors.size);
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
                const float sinc = argument == 0.0F ? 1.0F : std::sin(argument) / argument;
                radial_basis[static_cast<std::size_t>(radial_index)] = frequency * sinc;
            }
            for (int hidden_index = 0; hidden_index < 2 * options_.radial_hidden; ++hidden_index) {
                radial_pre[static_cast<std::size_t>(hidden_index)] = affine_value(
                    options_.radial_w0,
                    options_.n_radial,
                    2 * options_.radial_hidden,
                    radial_basis.data(),
                    hidden_index);
            }
            for (int hidden_index = 0; hidden_index < options_.radial_hidden; ++hidden_index) {
                const float gate = radial_pre[static_cast<std::size_t>(hidden_index)];
                const float value = radial_pre[static_cast<std::size_t>(
                    options_.radial_hidden + hidden_index)];
                radial_hidden[static_cast<std::size_t>(hidden_index)] =
                    gate * sigmoid(gate) * value;
            }
            for (int channel = 0; channel < options_.channels; ++channel) {
                radial[static_cast<std::size_t>(channel)] = affine_value(
                    options_.radial_w1,
                    options_.radial_hidden,
                    options_.channels,
                    radial_hidden.data(),
                    channel);
            }
            for (int mode = 0; mode < options_.radial_modes; ++mode) {
                modes[static_cast<std::size_t>(mode)] = affine_value(
                    options_.radial_mode_w,
                    options_.radial_hidden,
                    options_.radial_modes,
                    radial_hidden.data(),
                    mode);
            }

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
        }

        const double divisor_scalar = std::sqrt(reduced[0] + static_cast<double>(kNormFloor));
        const double divisor_angular = std::sqrt(reduced[1] + static_cast<double>(kNormFloor));
        std::vector<float> moments(static_cast<std::size_t>(moment_count_), 0.0F);
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
        double* destination = output + center_atom * feature_count_;
        for (std::size_t feature = 0; feature < descriptor.size(); ++feature) {
            double value = static_cast<double>(descriptor[feature]);
            if (options_.calibrate) {
                value = (value - static_cast<double>(options_.output_mean[feature]))
                    / static_cast<double>(options_.output_stddev[feature]);
            }
            destination[feature] = value;
        }
        if (control) {
            control->mark_completed();
        }
    }
    if (control && control->cancelled()) {
        throw CancelledError();
    }
}

} // namespace mdescriptor
