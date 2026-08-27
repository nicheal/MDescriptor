#include "mdescriptor/dpa4.hpp"

#include "mdescriptor/detail/batch.hpp"
#include "mdescriptor/detail/math3.hpp"
#include "mdescriptor/neighbor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mdescriptor {
namespace {

constexpr int kLmax = 3;
constexpr int kDegrees = 4;
constexpr int kFullDim = 16;
constexpr int kReducedDim = 10;
constexpr int kChannels = 64;
constexpr int kFrames = 3;
constexpr int kGridSize = 152;
constexpr int kGridCoeff = kFullDim * kFrames;
constexpr float kEpsilon = 1.0e-7F;
constexpr float kEquivariantNormEpsilon = 1.0e-5F;

inline std::size_t node_index(std::int64_t node, int degree, int channel) {
    return (static_cast<std::size_t>(node) * kFullDim + static_cast<std::size_t>(degree))
        * kChannels + static_cast<std::size_t>(channel);
}

inline std::size_t edge_index(std::size_t edge, int degree, int channel) {
    return (edge * kFullDim + static_cast<std::size_t>(degree)) * kChannels
        + static_cast<std::size_t>(channel);
}

inline std::size_t reduced_index(std::size_t edge, int degree, int channel) {
    return (edge * kReducedDim + static_cast<std::size_t>(degree)) * kChannels
        + static_cast<std::size_t>(channel);
}

inline float sigmoid(float value) {
    return 1.0F / (1.0F + std::exp(-value));
}

inline float silu(float value) {
    return value * sigmoid(value);
}

inline float softplus(float value) {
    if (value > 20.0F) {
        return value;
    }
    return std::log1p(std::exp(value));
}

template <typename Value>
void require_size(const std::vector<Value>& values, std::size_t expected, const char* name) {
    if (values.size() != expected) {
        throw std::invalid_argument(
            std::string("DPA4 ") + name + " has unexpected size");
    }
}

void validate_options(const Dpa4Options& options) {
    if (!std::isfinite(options.rcut) || options.rcut <= 0.0
        || options.ntypes <= 0 || options.channels != kChannels
        || options.n_radial != 16 || options.num_threads <= 0) {
        throw std::invalid_argument("unsupported DPA4 native configuration");
    }
    require_size(options.type_embedding, static_cast<std::size_t>(options.ntypes + 1) * kChannels,
                 "type embedding");
    require_size(options.env_rbf_layer1, 16U * 32U, "environment radial layer 1");
    require_size(options.env_rbf_layer2, 32U * 32U, "environment radial layer 2");
    require_size(options.env_type_embedding, static_cast<std::size_t>(options.ntypes + 1) * 16U,
                 "environment type embedding");
    require_size(options.env_g_layer1, 64U * 128U, "environment G layer 1");
    require_size(options.env_g_layer2, 128U * 64U, "environment G layer 2");
    require_size(options.env_output_projection, 512U * 128U, "environment output projection");
    require_size(options.film_scale_norm, 64U, "FiLM scale norm");
    require_size(options.film_shift_norm, 64U, "FiLM shift norm");
    require_size(options.radial_freqs, 16U, "radial frequencies");
    require_size(options.radial_layer1, 16U * 64U, "radial layer 1");
    require_size(options.radial_norm_scale, 64U, "radial norm scale");
    require_size(options.radial_layer2, 64U * 256U, "radial layer 2");
    require_size(options.gie_row_index, 15U, "GIE row index");
    require_size(options.gie_m0_index, 15U, "GIE m0 index");
    require_size(options.gie_radial_index, 15U, "GIE radial index");
    require_size(options.grid_to, static_cast<std::size_t>(kGridSize) * kGridCoeff,
                 "SO(3) grid to matrix");
    require_size(options.grid_from, static_cast<std::size_t>(kGridCoeff) * kGridSize,
                 "SO(3) grid from matrix");
    if (options.blocks.size() != 3U) {
        throw std::invalid_argument("DPA4 native backend requires three blocks");
    }
    for (const Dpa4BlockOptions& block : options.blocks) {
        if (block.pre_norm_enabled) {
            require_size(block.pre_norm_scale, 4U * 64U, "block pre-norm scale");
            require_size(block.pre_norm_bias, 64U, "block pre-norm bias");
            require_size(block.pre_norm_balance, 16U, "block pre-norm balance");
        }
        if (block.post_norm_enabled) {
            require_size(block.post_norm_scale, 4U * 64U, "block post-norm scale");
            require_size(block.post_norm_bias, 64U, "block post-norm bias");
            require_size(block.post_norm_balance, 16U, "block post-norm balance");
        }
        if (block.ffn_norm_enabled) {
            require_size(block.ffn_norm_scale, 4U * 64U, "FFN pre-norm scale");
            require_size(block.ffn_norm_bias, 64U, "FFN pre-norm bias");
            require_size(block.ffn_norm_balance, 16U, "FFN pre-norm balance");
        }
        require_size(block.pre_focus_weight, 4U * 64U * 64U, "block pre-focus weight");
        require_size(block.post_focus_weight, 4U * 64U * 64U, "block post-focus weight");
        require_size(block.radial_mixer_weight, 256U * 25U, "radial degree mixer weight");
        require_size(block.radial_channel_basis, 64U, "radial degree mixer channel basis");
        for (const auto& value : block.so2_weight_m0) {
            require_size(value, 256U * 256U, "SO(2) m0 weight");
        }
        for (const auto& value : block.so2_weight_m1) {
            require_size(value, 192U * 384U, "SO(2) m1 weight");
        }
        for (const auto& value : block.so2_gate_weight) {
            require_size(value, 64U * 192U, "SO(2) gate weight");
        }
        require_size(block.attn_qk_scale, 64U, "attention QK norm scale");
        require_size(block.attn_q_weight, 64U * 64U, "attention Q weight");
        require_size(block.attn_k_weight, 64U * 64U, "attention K weight");
        require_size(block.attn_output_gate_scale, 64U, "attention output norm scale");
        require_size(block.attn_logit_weight, 64U, "attention logit weight");
        require_size(block.attn_z_bias_raw, 1U, "attention null bias");
        require_size(block.attn_gate_weight, 64U, "attention output gate weight");
        require_size(block.message_scalar_gate, 128U * 64U, "message scalar gate");
        require_size(block.message_frame_expand, 4U * 64U * 192U, "message frame expand");
        require_size(block.message_frame_contract, 4U * 192U * 64U, "message frame contract");
        require_size(block.message_residual_scale, 64U, "message residual scale");
        require_size(block.ffn_linear1, 4U * 64U * 1152U, "FFN linear 1");
        require_size(block.ffn_linear2, 4U * 576U * 64U, "FFN linear 2");
        require_size(block.ffn_scalar_gate, 384U * 192U, "FFN scalar gate");
        require_size(block.ffn_grid_left, 192U * 192U, "FFN grid left");
        require_size(block.ffn_grid_right, 192U * 192U, "FFN grid right");
        require_size(block.ffn_grid_router, 384U, "FFN grid router");
        require_size(block.ffn_grid_out, 192U * 192U, "FFN grid output");
    }
    require_size(options.output_linear1, 4U * 64U * 1152U, "output linear 1");
    require_size(options.output_linear2, 4U * 576U * 64U, "output linear 2");
    require_size(options.output_scalar_gate, 384U * 192U, "output scalar gate");
    require_size(options.output_grid_left, 384U * 384U, "output grid left");
    require_size(options.output_grid_right, 384U * 384U, "output grid right");
    require_size(options.output_grid_out, 384U * 192U, "output grid output");
    validate_dpa4_wigner_payload(options.wigner);
}

Dpa4WignerPayload make_wigner_payload(Dpa4Options& options) {
    options.wigner.l2_tensor.coefficients = options.wigner_l2_tensor.data();
    options.wigner.l2_tensor.coefficient_count = options.wigner_l2_tensor.size();
    options.wigner.l3.coefficients = options.wigner_l3_coefficients.data();
    options.wigner.l3.exponents = options.wigner_l3_exponents.data();
    options.wigner.l3.monomial_count = kDpa4WignerL3MonomialCount;
    options.wigner.l3.degree = 3;
    options.wigner.l3.monomial_degree = 6;
    return options.wigner;
}

struct EdgeData {
    std::vector<std::int32_t> src;
    std::vector<std::int32_t> dst;
    std::vector<std::int64_t> offsets;
    std::vector<float> vector;
    std::vector<float> length;
    std::vector<float> envelope;
    std::vector<float> radial_basis;
    // For each edge, rows of D_full selected by coeff_index_m.  The same
    // values are used as columns of D_full^T during rotate-back.
    std::vector<float> rotation_to_m; // [E, 10, 16]
    std::vector<float> gie_zonal;      // [E, 15]
};

// The dense DeepMD input adapter wraps periodic coordinates into the primary
// cell before constructing the extended image list.  Do the same in the C++
// path so a translated periodic frame follows exactly the same geometry and
// neighbor ordering as the reference implementation.
std::vector<double> normalized_positions(const StructureBatchView& batch) {
    std::vector<double> positions(
        static_cast<std::size_t>(batch.atoms) * 3U,
        0.0);
    if (batch.atoms > 0) {
        std::copy(
            batch.positions,
            batch.positions + static_cast<std::size_t>(batch.atoms) * 3U,
            positions.begin());
    }
    for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
        const std::int32_t* pbc = batch.pbc + structure * 3;
        const bool periodic = pbc[0] == 1 && pbc[1] == 1 && pbc[2] == 1;
        if (!periodic) {
            continue;
        }
        detail::Mat3 cell;
        const double* cell_data = batch.cells + structure * 9U;
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                cell.a[row][column] = cell_data[row * 3 + column];
            }
        }
        detail::Mat3 inverse;
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
            const double* point = batch.positions + atom * 3;
            const detail::Vec3 fractional{
                point[0] * inverse.a[0][0] + point[1] * inverse.a[1][0]
                    + point[2] * inverse.a[2][0],
                point[0] * inverse.a[0][1] + point[1] * inverse.a[1][1]
                    + point[2] * inverse.a[2][1],
                point[0] * inverse.a[0][2] + point[1] * inverse.a[1][2]
                    + point[2] * inverse.a[2][2],
            };
            const detail::Vec3 wrapped{
                fractional.x - std::floor(fractional.x),
                fractional.y - std::floor(fractional.y),
                fractional.z - std::floor(fractional.z),
            };
            const detail::Vec3 cartesian = wrapped.x * detail::row(cell, 0)
                + wrapped.y * detail::row(cell, 1)
                + wrapped.z * detail::row(cell, 2);
            positions[static_cast<std::size_t>(atom * 3 + 0)] = cartesian.x;
            positions[static_cast<std::size_t>(atom * 3 + 1)] = cartesian.y;
            positions[static_cast<std::size_t>(atom * 3 + 2)] = cartesian.z;
        }
    }
    return positions;
}

float c3_envelope(float distance, float rcut, int exponent) {
    float u = (rcut - distance) / rcut;
    u = std::max(0.0F, std::min(1.0F, u));
    const float x = 1.0F - u;
    float series = 0.0F;
    // C3CutoffEnvelope uses coefficients comb(k+3,3), evaluated by Horner.
    for (int k = exponent - 1; k >= 0; --k) {
        const float coefficient = static_cast<float>((k + 3) * (k + 2) * (k + 1) / 6);
        series = coefficient + x * series;
    }
    return std::pow(u, 4.0F) * series;
}

float edge_coordinate(
    const StructureBatchView& batch,
    std::int32_t atom,
    const std::int32_t* shift,
    std::int64_t structure,
    int component) {
    const double* position = batch.positions + static_cast<std::size_t>(atom) * 3U;
    double value = position[component];
    const std::int32_t* pbc = batch.pbc + structure * 3;
    if (pbc[0] == 1 || pbc[1] == 1 || pbc[2] == 1) {
        const double* cell = batch.cells + structure * 9U;
        for (int axis = 0; axis < 3; ++axis) {
            value += static_cast<double>(shift[axis]) * cell[axis * 3 + component];
        }
    }
    return static_cast<float>(value);
}

EdgeData build_edges(
    const StructureBatchView& batch,
    double cutoff,
    int num_threads,
    const std::shared_ptr<ComputeControl>& control) {
    // The common graph builder supplies the same image enumeration and cutoff
    // convention as the other native descriptors.  DPA4's nlist excludes the
    // exact self pair, so remove only the original-cell self edge here; periodic
    // self images remain valid neighbors.
    const std::vector<double> wrapped = normalized_positions(batch);
    StructureBatchView normalized_batch = batch;
    normalized_batch.positions = wrapped.data();
    const NeighborGraph graph = build_neighbor_graph(
        normalized_batch, cutoff, control, num_threads, true, false, true);
    EdgeData edges;
    edges.offsets.assign(static_cast<std::size_t>(batch.atoms) + 1U, 0);
    edges.src.reserve(graph.atoms_data().size());
    edges.dst.reserve(graph.atoms_data().size());
    edges.vector.reserve(graph.displacements().size());
    for (std::int64_t center = 0; center < batch.atoms; ++center) {
        const NeighborView view = graph.for_center(center);
        const std::int64_t structure = [&]() {
            const auto it = std::upper_bound(
                batch.offsets, batch.offsets + batch.structures, center);
            return static_cast<std::int64_t>(std::max<std::ptrdiff_t>(0, (it - batch.offsets) - 1));
        }();
        // ``_build_nlist`` used by the reference descriptor emits each row in
        // ascending distance order.  The shared cell-list graph is deliberately
        // order-agnostic and visits spatial cells first, which can reverse two
        // neighbors in the same destination segment.  The attention reduction
        // is mathematically permutation invariant, but its fp32 scatter order
        // is not; reproduce the dense ABI's stable distance ordering here.
        std::vector<std::size_t> order(view.size);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::stable_sort(order.begin(), order.end(), [&view](std::size_t lhs, std::size_t rhs) {
            const double left_distance = std::sqrt(view.distance2[lhs]);
            const double right_distance = std::sqrt(view.distance2[rhs]);
            if (left_distance != right_distance) {
                return left_distance < right_distance;
            }
            // The dense reference first orders periodic image shifts by their
            // integer-vector norm (stable), then keeps source atoms in their
            // original order.  This is the tie policy used by
            // ``array_api_compat.numpy.argsort`` when equal distances occur.
            if (view.shifts != nullptr) {
                const std::int32_t* left = view.shifts + lhs * 3U;
                const std::int32_t* right = view.shifts + rhs * 3U;
                const std::int64_t left_norm = static_cast<std::int64_t>(left[0]) * left[0]
                    + static_cast<std::int64_t>(left[1]) * left[1]
                    + static_cast<std::int64_t>(left[2]) * left[2];
                const std::int64_t right_norm = static_cast<std::int64_t>(right[0]) * right[0]
                    + static_cast<std::int64_t>(right[1]) * right[1]
                    + static_cast<std::int64_t>(right[2]) * right[2];
                if (left_norm != right_norm) {
                    return left_norm < right_norm;
                }
                for (int axis = 0; axis < 3; ++axis) {
                    if (left[axis] != right[axis]) {
                        return left[axis] < right[axis];
                    }
                }
            }
            return view.atoms[lhs] < view.atoms[rhs];
        });
        for (const std::size_t local : order) {
            if (view.exact_self(local, center)) {
                continue;
            }
            const std::int32_t source = view.atoms[local];
            const std::int32_t* shift = view.shifts == nullptr
                ? nullptr : view.shifts + local * 3U;
            edges.src.push_back(source);
            edges.dst.push_back(static_cast<std::int32_t>(center));
            for (int component = 0; component < 3; ++component) {
                const float source_value = shift == nullptr
                    ? static_cast<float>(normalized_batch.positions[
                        static_cast<std::size_t>(source) * 3U + component])
                    : edge_coordinate(normalized_batch, source, shift, structure, component);
                const float center_value = static_cast<float>(
                    normalized_batch.positions[static_cast<std::size_t>(center) * 3U + component]);
                edges.vector.push_back(source_value - center_value);
            }
        }
        edges.offsets[static_cast<std::size_t>(center + 1)] =
            static_cast<std::int64_t>(edges.src.size());
    }
    const std::size_t count = edges.src.size();
    edges.length.resize(count);
    edges.envelope.resize(count);
    edges.radial_basis.resize(count * 16U);
    edges.rotation_to_m.resize(count * 10U * 16U);
    edges.gie_zonal.resize(count * 15U);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(num_threads)
#endif
    for (std::ptrdiff_t edge = 0; edge < static_cast<std::ptrdiff_t>(count); ++edge) {
        const std::size_t index = static_cast<std::size_t>(edge);
        const float dx = edges.vector[index * 3U + 0U];
        const float dy = edges.vector[index * 3U + 1U];
        const float dz = edges.vector[index * 3U + 2U];
        const float length = std::sqrt(dx * dx + dy * dy + dz * dz + kEpsilon * kEpsilon);
        edges.length[index] = length;
        edges.envelope[index] = c3_envelope(length, static_cast<float>(cutoff), 5);
        const float radial_env = c3_envelope(length, static_cast<float>(cutoff), 7);
        for (int radial = 0; radial < 16; ++radial) {
            // The frequencies are filled by the caller after construction.
            edges.radial_basis[index * 16U + static_cast<std::size_t>(radial)] = radial_env;
        }
    }
    return edges;
}

void fill_radial_basis(EdgeData& edges, const Dpa4Options& options) {
    const std::size_t count = edges.src.size();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options.num_threads)
#endif
    for (std::ptrdiff_t edge = 0; edge < static_cast<std::ptrdiff_t>(count); ++edge) {
        const std::size_t index = static_cast<std::size_t>(edge);
        const float distance = edges.length[index];
        const float radial_env = edges.radial_basis[index * 16U];
        for (int radial = 0; radial < 16; ++radial) {
            const float frequency = options.radial_freqs[static_cast<std::size_t>(radial)];
            const float argument = distance * frequency;
            // The reference uses ``torch.sinc(argument / pi)`` rather than
            // evaluating ``sin(argument) / argument`` directly.  Keep the
            // same float32 divide/multiply sequence before calling sin so
            // the CPU kernel follows that precision boundary.
            constexpr float pi = 3.1415927410125732422F;
            const float sinc_argument = argument / pi;
            const float sinc = sinc_argument == 0.0F
                ? 1.0F : std::sin(pi * sinc_argument) / (pi * sinc_argument);
            edges.radial_basis[index * 16U + static_cast<std::size_t>(radial)] =
                frequency * sinc * radial_env;
        }
    }
}

void build_rotations(EdgeData& edges, const Dpa4Options& options) {
    const std::size_t count = edges.src.size();
    std::vector<Dpa4EdgeVector> vectors(count);
    std::vector<Dpa4Quaternion> quaternions(count);
    for (std::size_t edge = 0; edge < count; ++edge) {
        vectors[edge] = {
            edges.vector[edge * 3U + 0U],
            edges.vector[edge * 3U + 1U],
            edges.vector[edge * 3U + 2U],
        };
        quaternions[edge] = build_dpa4_edge_quaternion_with_length(
            vectors[edge], edges.length[edge], kEpsilon);
    }
    Dpa4WignerLowOrder wigner(options.wigner);
    const std::array<int, 10> coeff_index = {0, 2, 6, 12, 1, 5, 11, 3, 7, 13};
    const std::array<int, 15> gie_rows = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    };
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options.num_threads)
#endif
    for (std::ptrdiff_t edge = 0; edge < static_cast<std::ptrdiff_t>(count); ++edge) {
        const std::size_t index = static_cast<std::size_t>(edge);
        float block[256];
        wigner.compute_blocks(quaternions[index], block, kEpsilon);
        for (int reduced = 0; reduced < kReducedDim; ++reduced) {
            const int row = coeff_index[static_cast<std::size_t>(reduced)];
            for (int column = 0; column < kFullDim; ++column) {
                edges.rotation_to_m[(index * kReducedDim + static_cast<std::size_t>(reduced))
                    * kFullDim + static_cast<std::size_t>(column)] =
                    block[row * kFullDim + column];
            }
        }
        for (std::size_t row_index = 0; row_index < gie_rows.size(); ++row_index) {
            const int row = gie_rows[row_index];
            const int degree = row == 0 ? 0 : static_cast<int>(std::sqrt(static_cast<float>(row)));
            const int column = degree * (degree + 1);
            edges.gie_zonal[index * 15U + row_index] = block[column * kFullDim + row];
        }
    }
}

void row_matmul(
    const float* input,
    int input_width,
    const std::vector<float>& weight,
    int output_width,
    float* output) {
    // Keep short projections on the model's fp32/FMA path.  The two long
    // reductions below are the channel projections used by the DPA4 grid
    // branches.  A small fixed fan-in for width 384 approximates the pairwise
    // reduction used by a batched GEMM without introducing a BLAS dependency;
    // width 512/1152 uses a widened accumulator and crosses back to fp32 once.
    for (int column = 0; column < output_width; ++column) {
        if (input_width == 384) {
            constexpr int lanes = 16;
            std::array<float, lanes> partial{};
            for (int row = 0; row < input_width; ++row) {
                const std::size_t lane = static_cast<std::size_t>(row % lanes);
                partial[lane] = std::fma(
                    input[row],
                    weight[static_cast<std::size_t>(row * output_width + column)],
                    partial[lane]);
            }
            float value = 0.0F;
            for (float part : partial) {
                value += part;
            }
            output[column] = value;
            continue;
        }
        if (input_width >= 512) {
            double value = 0.0;
            for (int row = 0; row < input_width; ++row) {
                value += static_cast<double>(input[row])
                    * static_cast<double>(weight[static_cast<std::size_t>(row * output_width + column)]);
            }
            output[column] = static_cast<float>(value);
            continue;
        }
        float value = 0.0F;
        for (int row = 0; row < input_width; ++row) {
            value = std::fma(
                input[row],
                weight[static_cast<std::size_t>(row * output_width + column)],
                value);
        }
        output[column] = value;
    }
}

void channel_matmul(
    const float* input,
    int input_width,
    const std::vector<float>& weight,
    int output_width,
    float* output) {
    row_matmul(input, input_width, weight, output_width, output);
}

void apply_so3_linear(
    const std::vector<float>& input,
    std::int64_t nodes,
    int input_channels,
    int output_channels,
    const std::vector<float>& weight,
    std::vector<float>& output,
    int num_threads) {
    output.assign(static_cast<std::size_t>(nodes) * kFullDim
                      * static_cast<std::size_t>(output_channels), 0.0F);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(num_threads)
#endif
    for (std::int64_t node = 0; node < nodes; ++node) {
        for (int degree = 0; degree < kDegrees; ++degree) {
            const int width = 2 * degree + 1;
            const std::size_t weight_offset = static_cast<std::size_t>(degree)
                * static_cast<std::size_t>(input_channels * output_channels);
            for (int component = 0; component < width; ++component) {
                const std::size_t input_offset =
                    (static_cast<std::size_t>(node) * kFullDim
                        + static_cast<std::size_t>(degree * degree + component))
                    * static_cast<std::size_t>(input_channels);
                const std::size_t output_offset =
                    (static_cast<std::size_t>(node) * kFullDim
                        + static_cast<std::size_t>(degree * degree + component))
                    * static_cast<std::size_t>(output_channels);
                for (int out = 0; out < output_channels; ++out) {
                    float value = 0.0F;
                    for (int in = 0; in < input_channels; ++in) {
                        value = std::fma(
                            input[input_offset + static_cast<std::size_t>(in)],
                            weight[weight_offset + static_cast<std::size_t>(in * output_channels + out)],
                            value);
                    }
                    output[output_offset + static_cast<std::size_t>(out)] = value;
                }
            }
        }
    }
}

void apply_channel_matrix(
    const float* input,
    int input_width,
    const std::vector<float>& weight,
    int output_width,
    float* output) {
    row_matmul(input, input_width, weight, output_width, output);
}

void compute_environment(
    const EdgeData& edges,
    const Dpa4Options& options,
    const std::int32_t* types,
    std::int64_t nodes,
    std::vector<float>& type_features,
    std::vector<float>& film) {
    type_features.resize(static_cast<std::size_t>(nodes) * 64U);
    for (std::int64_t node = 0; node < nodes; ++node) {
        const int type = types[node];
        std::copy(
            options.type_embedding.begin() + static_cast<std::size_t>(type) * 64U,
            options.type_embedding.begin() + static_cast<std::size_t>(type + 1) * 64U,
            type_features.begin() + static_cast<std::size_t>(node) * 64U);
    }
    std::vector<double> degree(static_cast<std::size_t>(nodes), 0.0);
    for (std::int64_t node = 0; node < nodes; ++node) {
        for (std::int64_t edge = edges.offsets[static_cast<std::size_t>(node)];
             edge < edges.offsets[static_cast<std::size_t>(node + 1)]; ++edge) {
            const float env = edges.envelope[static_cast<std::size_t>(edge)];
            degree[static_cast<std::size_t>(node)] += static_cast<double>(env)
                * static_cast<double>(env);
        }
    }
    std::vector<double> inv_degree(static_cast<std::size_t>(nodes), 0.0);
    for (std::int64_t node = 0; node < nodes; ++node) {
        inv_degree[static_cast<std::size_t>(node)] =
            1.0 / std::sqrt(degree[static_cast<std::size_t>(node)] + 0.25);
    }
    film.assign(static_cast<std::size_t>(nodes) * 128U, 0.0F);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options.num_threads)
#endif
    for (std::int64_t node = 0; node < nodes; ++node) {
        std::array<double, 4U * 64U> env_agg{};
        std::array<float, 32> rbf_hidden{};
        std::array<float, 32> rbf_projected{};
        std::array<float, 128> hidden{};
        std::array<float, 64> g{};
        std::array<float, 64> type_src{};
        std::array<float, 64> type_dst{};
        std::array<float, 16> rbf{};
        for (std::int64_t edge = edges.offsets[static_cast<std::size_t>(node)];
             edge < edges.offsets[static_cast<std::size_t>(node + 1)]; ++edge) {
            const std::size_t e = static_cast<std::size_t>(edge);
            const int source_type = types[edges.src[e]];
            const int destination_type = types[edges.dst[e]];
            std::copy(
                options.env_type_embedding.begin() + static_cast<std::size_t>(source_type) * 16U,
                options.env_type_embedding.begin() + static_cast<std::size_t>(source_type + 1) * 16U,
                type_src.begin());
            std::copy(
                options.env_type_embedding.begin() + static_cast<std::size_t>(destination_type) * 16U,
                options.env_type_embedding.begin() + static_cast<std::size_t>(destination_type + 1) * 16U,
                type_dst.begin());
            for (int radial = 0; radial < 16; ++radial) {
                rbf[static_cast<std::size_t>(radial)] = edges.radial_basis[e * 16U + static_cast<std::size_t>(radial)];
            }
            row_matmul(rbf.data(), 16, options.env_rbf_layer1, 32, rbf_hidden.data());
            for (float& value : rbf_hidden) {
                value = silu(value);
            }
            row_matmul(rbf_hidden.data(), 32, options.env_rbf_layer2, 32, rbf_projected.data());
            std::array<float, 64> g_input{};
            for (int i = 0; i < 32; ++i) {
                g_input[static_cast<std::size_t>(i)] = rbf_projected[static_cast<std::size_t>(i)];
            }
            // The environment type embedding has width 16 per endpoint.
            std::copy(type_src.begin(), type_src.begin() + 16, g_input.begin() + 32);
            std::copy(type_dst.begin(), type_dst.begin() + 16, g_input.begin() + 48);
            row_matmul(g_input.data(), 64, options.env_g_layer1, 128, hidden.data());
            for (float& value : hidden) {
                value = silu(value);
            }
            row_matmul(hidden.data(), 128, options.env_g_layer2, 64, g.data());
            const float dx = edges.vector[e * 3U + 0U];
            const float dy = edges.vector[e * 3U + 1U];
            const float dz = edges.vector[e * 3U + 2U];
            const float inv_r = 1.0F / std::sqrt(dx * dx + dy * dy + dz * dz + kEpsilon * kEpsilon);
            const float s = edges.envelope[e] * inv_r;
            const float rtilde[4] = {s, s * dx * inv_r, s * dy * inv_r, s * dz * inv_r};
            for (int coordinate = 0; coordinate < 4; ++coordinate) {
                for (int channel = 0; channel < 64; ++channel) {
                    env_agg[static_cast<std::size_t>(coordinate * 64 + channel)] +=
                        static_cast<double>(rtilde[coordinate])
                        * static_cast<double>(g[static_cast<std::size_t>(channel)]);
                }
            }
        }
        const double scale = inv_degree[static_cast<std::size_t>(node)];
        for (double& value : env_agg) {
            value *= scale;
        }
        std::array<float, 512> d_matrix{};
        for (int row = 0; row < 64; ++row) {
            for (int column = 0; column < 8; ++column) {
                float value = 0.0F;
                for (int coordinate = 0; coordinate < 4; ++coordinate) {
                    value += env_agg[static_cast<std::size_t>(coordinate * 64 + row)]
                        * env_agg[static_cast<std::size_t>(coordinate * 64 + column)];
                }
                d_matrix[static_cast<std::size_t>(row * 8 + column)] =
                    static_cast<float>(value);
            }
        }
        row_matmul(
            d_matrix.data(), 512, options.env_output_projection, 128,
            film.data() + static_cast<std::size_t>(node) * 128U);
    }

    // ScalarRMSNorm + FiLM scale/shift.  The checkpoint uses edge_norm=True,
    // therefore the two independent norm scales are active.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options.num_threads)
#endif
    for (std::int64_t node = 0; node < nodes; ++node) {
        double scale_sq = 0.0;
        double shift_sq = 0.0;
        for (int channel = 0; channel < 64; ++channel) {
            const float scale_value = film[static_cast<std::size_t>(node) * 128U + static_cast<std::size_t>(channel)];
            const float shift_value = film[static_cast<std::size_t>(node) * 128U + 64U + static_cast<std::size_t>(channel)];
            scale_sq += static_cast<double>(scale_value) * static_cast<double>(scale_value);
            shift_sq += static_cast<double>(shift_value) * static_cast<double>(shift_value);
        }
        const float scale_inv = static_cast<float>(1.0 / std::sqrt(
            scale_sq / 64.0 + static_cast<double>(kEpsilon)));
        const float shift_inv = static_cast<float>(1.0 / std::sqrt(
            shift_sq / 64.0 + static_cast<double>(kEpsilon)));
        const float scale_strength = std::exp(options.film_scale_strength_log);
        const float shift_strength = std::exp(options.film_shift_strength_log);
        for (int channel = 0; channel < 64; ++channel) {
            float& scale_value = film[static_cast<std::size_t>(node) * 128U + static_cast<std::size_t>(channel)];
            float& shift_value = film[static_cast<std::size_t>(node) * 128U + 64U + static_cast<std::size_t>(channel)];
            scale_value = 1.0F + scale_strength
                * std::tanh(scale_value * scale_inv * options.film_scale_norm[static_cast<std::size_t>(channel)]);
            shift_value = shift_strength
                * std::tanh(shift_value * shift_inv * options.film_shift_norm[static_cast<std::size_t>(channel)]);
            const std::size_t type_offset = static_cast<std::size_t>(node) * 64U + static_cast<std::size_t>(channel);
            type_features[type_offset] = type_features[type_offset] * scale_value + shift_value;
        }
    }
}

void radial_features(
    const EdgeData& edges,
    const Dpa4Options& options,
    std::vector<float>& output) {
    const std::size_t count = edges.src.size();
    output.resize(count * 256U);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options.num_threads)
#endif
    for (std::ptrdiff_t edge = 0; edge < static_cast<std::ptrdiff_t>(count); ++edge) {
        const std::size_t e = static_cast<std::size_t>(edge);
        std::array<float, 64> hidden{};
        row_matmul(
            edges.radial_basis.data() + e * 16U,
            16,
            options.radial_layer1,
            64,
            hidden.data());
        float variance = 0.0F;
        for (float value : hidden) {
            variance += value * value;
        }
        const float inv_rms = 1.0F / std::sqrt(variance / 64.0F + kEpsilon);
        for (int channel = 0; channel < 64; ++channel) {
            hidden[static_cast<std::size_t>(channel)] = silu(
                hidden[static_cast<std::size_t>(channel)] * inv_rms
                    * options.radial_norm_scale[static_cast<std::size_t>(channel)]);
        }
        row_matmul(
            hidden.data(), 64, options.radial_layer2, 256,
            output.data() + e * 256U);
        for (int value = 0; value < 256; ++value) {
            output[e * 256U + static_cast<std::size_t>(value)] *= edges.envelope[e];
        }
    }
}

void initial_features(
    const EdgeData& edges,
    const Dpa4Options& options,
    const std::vector<float>& radial,
    std::int64_t nodes,
    std::vector<float>& x) {
    x.assign(static_cast<std::size_t>(nodes) * kFullDim * kChannels, 0.0F);
    // The caller writes the type slice separately; this function only adds GIE.
    std::vector<float> degree_sum(static_cast<std::size_t>(nodes), 0.0F);
    for (std::int64_t node = 0; node < nodes; ++node) {
        for (std::int64_t edge = edges.offsets[static_cast<std::size_t>(node)];
             edge < edges.offsets[static_cast<std::size_t>(node + 1)]; ++edge) {
            const float env = edges.envelope[static_cast<std::size_t>(edge)];
            degree_sum[static_cast<std::size_t>(node)] += env * env;
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options.num_threads)
#endif
    for (std::int64_t node = 0; node < nodes; ++node) {
        const float inv_degree = 1.0F / std::sqrt(
            degree_sum[static_cast<std::size_t>(node)] + 0.25F);
        std::vector<float> node_features(
            static_cast<std::size_t>(kFullDim * kChannels), 0.0F);
        for (std::int64_t edge = edges.offsets[static_cast<std::size_t>(node)];
             edge < edges.offsets[static_cast<std::size_t>(node + 1)]; ++edge) {
            const std::size_t e = static_cast<std::size_t>(edge);
            for (int row_index = 0; row_index < 15; ++row_index) {
                const int row = static_cast<int>(options.gie_row_index[static_cast<std::size_t>(row_index)]);
                const int radial_slot = static_cast<int>(options.gie_radial_index[static_cast<std::size_t>(row_index)]);
                const float coupling = edges.gie_zonal[e * 15U + static_cast<std::size_t>(row_index)];
                for (int channel = 0; channel < 64; ++channel) {
                    node_features[static_cast<std::size_t>(row * kChannels + channel)] +=
                        coupling
                        // GIE receives the radial embedding with the l=0
                        // slice removed (radial_feat[:, 1:, :]); the stored
                        // slot index is therefore relative to l=1.
                        * radial[e * 256U
                            + static_cast<std::size_t>((radial_slot + 1) * 64 + channel)]
                        * inv_degree;
                }
            }
        }
        for (int row = 0; row < kFullDim; ++row) {
            for (int channel = 0; channel < kChannels; ++channel) {
                x[node_index(node, row, channel)] =
                    node_features[static_cast<std::size_t>(row * kChannels + channel)];
            }
        }
    }
}

void add_type_slice(
    std::vector<float>& x,
    const std::vector<float>& type_features,
    std::int64_t nodes,
    int num_threads) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(num_threads)
#endif
    for (std::int64_t node = 0; node < nodes; ++node) {
        std::copy(
            type_features.begin() + static_cast<std::size_t>(node) * 64U,
            type_features.begin() + static_cast<std::size_t>(node + 1) * 64U,
            x.begin() + node_index(node, 0, 0));
    }
}

void equivariant_norm(
    const std::vector<float>& input,
    const std::vector<float>& norm_scale,
    const std::vector<float>& norm_bias,
    const std::vector<float>& norm_balance,
    std::int64_t nodes,
    std::vector<float>& output,
    int num_threads) {
    output.resize(input.size());
    const std::array<int, 16> degree = {0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3};
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(num_threads)
#endif
    for (std::int64_t node = 0; node < nodes; ++node) {
        double mean = 0.0;
        for (int channel = 0; channel < 64; ++channel) {
            mean += static_cast<double>(input[node_index(node, 0, channel)]);
        }
        mean /= 64.0;
        double variance = 0.0;
        for (int row = 0; row < kFullDim; ++row) {
            const float weight = norm_balance[static_cast<std::size_t>(row)];
            for (int channel = 0; channel < 64; ++channel) {
                const double value = row == 0
                    ? static_cast<double>(input[node_index(node, row, channel)]) - mean
                    : input[node_index(node, row, channel)];
                variance += value * value * static_cast<double>(weight);
            }
        }
        const float inverse = static_cast<float>(
            1.0 / std::sqrt(variance + static_cast<double>(kEquivariantNormEpsilon)));
        for (int row = 0; row < kFullDim; ++row) {
            const int l = degree[static_cast<std::size_t>(row)];
            for (int channel = 0; channel < 64; ++channel) {
                const float value = row == 0
                    ? static_cast<float>(static_cast<double>(input[node_index(node, row, channel)]) - mean)
                    : input[node_index(node, row, channel)];
                output[node_index(node, row, channel)] = value * inverse
                    * norm_scale[static_cast<std::size_t>(l * 64 + channel)];
                if (row == 0) {
                    output[node_index(node, row, channel)] += norm_bias[static_cast<std::size_t>(channel)];
                }
            }
        }
    }
}

void apply_so2_linear(
    const std::vector<float>& input,
    const Dpa4BlockOptions& block,
    int layer,
    std::size_t edge_count,
    std::vector<float>& output,
    int num_threads) {
    output.assign(edge_count * kReducedDim * kChannels, 0.0F);
    const auto& weight_m0 = block.so2_weight_m0[static_cast<std::size_t>(layer)];
    const auto& weight_m1 = block.so2_weight_m1[static_cast<std::size_t>(layer)];

    // The m=0 sector contains l=0..3 and is an ordinary dense (256 x 256)
    // channel/degree map.  The two signed m=1 sectors share the SO(2) complex
    // weight, represented by the [Wu, Wv] concatenation in the checkpoint.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(num_threads)
#endif
    for (std::ptrdiff_t edge_index = 0; edge_index < static_cast<std::ptrdiff_t>(edge_count); ++edge_index) {
        const std::size_t edge = static_cast<std::size_t>(edge_index);
        for (int output_degree = 0; output_degree < 4; ++output_degree) {
            for (int output_channel = 0; output_channel < kChannels; ++output_channel) {
                double value = 0.0;
                const int output_index = output_degree * kChannels + output_channel;
                for (int input_degree = 0; input_degree < 4; ++input_degree) {
                    const int input_base = input_degree * kChannels;
                    for (int input_channel = 0; input_channel < kChannels; ++input_channel) {
                        const std::size_t input_offset =
                            edge * kReducedDim * kChannels + static_cast<std::size_t>(input_base + input_channel);
                        value += static_cast<double>(input[input_offset])
                            * static_cast<double>(weight_m0[static_cast<std::size_t>(
                                (input_base + input_channel) * 256 + output_index)]);
                    }
                }
                output[edge * kReducedDim * kChannels + static_cast<std::size_t>(output_index)] =
                    static_cast<float>(value);
            }
        }

        for (int output_degree = 1; output_degree <= 3; ++output_degree) {
            const int output_l = output_degree - 1;
            for (int output_channel = 0; output_channel < kChannels; ++output_channel) {
                double neg_u = 0.0;
                double neg_v = 0.0;
                double pos_u = 0.0;
                double pos_v = 0.0;
                for (int input_degree = 1; input_degree <= 3; ++input_degree) {
                    const int input_l = input_degree - 1;
                    for (int input_channel = 0; input_channel < kChannels; ++input_channel) {
                        const std::size_t weight_offset = static_cast<std::size_t>(
                            (input_l * kChannels + input_channel) * 384);
                        const float coefficient_u = weight_m1[weight_offset + static_cast<std::size_t>(output_l * kChannels + output_channel)];
                        const float coefficient_v = weight_m1[weight_offset + 192U + static_cast<std::size_t>(output_l * kChannels + output_channel)];
                        const std::size_t neg_offset = edge * kReducedDim * kChannels
                            + static_cast<std::size_t>((4 + input_l) * kChannels + input_channel);
                        const std::size_t pos_offset = edge * kReducedDim * kChannels
                            + static_cast<std::size_t>((7 + input_l) * kChannels + input_channel);
                        neg_u += static_cast<double>(input[neg_offset])
                            * static_cast<double>(coefficient_u);
                        neg_v += static_cast<double>(input[neg_offset])
                            * static_cast<double>(coefficient_v);
                        pos_u += static_cast<double>(input[pos_offset])
                            * static_cast<double>(coefficient_u);
                        pos_v += static_cast<double>(input[pos_offset])
                            * static_cast<double>(coefficient_v);
                    }
                }
                const std::size_t neg_output = edge * kReducedDim * kChannels
                    + static_cast<std::size_t>((4 + output_l) * kChannels + output_channel);
                const std::size_t pos_output = edge * kReducedDim * kChannels
                    + static_cast<std::size_t>((7 + output_l) * kChannels + output_channel);
                output[neg_output] = static_cast<float>(neg_u - pos_v);
                output[pos_output] = static_cast<float>(neg_v + pos_u);
            }
        }
    }
}

void apply_so2_gate(
    const std::vector<float>& input,
    const std::vector<float>& gate_weight,
    std::size_t edge_count,
    std::vector<float>& output,
    int num_threads) {
    output.resize(input.size());
    static constexpr std::array<int, 9> kReducedDegree = {
        1, 2, 3, 1, 2, 3, 1, 2, 3,
    };
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(num_threads)
#endif
    for (std::ptrdiff_t edge_index = 0; edge_index < static_cast<std::ptrdiff_t>(edge_count); ++edge_index) {
        const std::size_t edge = static_cast<std::size_t>(edge_index);
        const std::size_t edge_offset = edge * kReducedDim * kChannels;
        for (int channel = 0; channel < kChannels; ++channel) {
            output[edge_offset + static_cast<std::size_t>(channel)] = silu(
                input[edge_offset + static_cast<std::size_t>(channel)]);
        }
        std::array<float, 192> gate{};
        for (int gate_channel = 0; gate_channel < 192; ++gate_channel) {
            double value = 0.0;
            for (int channel = 0; channel < kChannels; ++channel) {
                value += static_cast<double>(input[
                    edge_offset + static_cast<std::size_t>(channel)])
                    * static_cast<double>(gate_weight[
                        static_cast<std::size_t>(channel * 192 + gate_channel)]);
            }
            gate[static_cast<std::size_t>(gate_channel)] = sigmoid(static_cast<float>(value));
        }
        for (int row = 1; row < kReducedDim; ++row) {
            const int degree = kReducedDegree[static_cast<std::size_t>(row - 1)];
            const int gate_offset = (degree - 1) * kChannels;
            for (int channel = 0; channel < kChannels; ++channel) {
                output[edge_offset + static_cast<std::size_t>(row * kChannels + channel)] =
                    input[edge_offset + static_cast<std::size_t>(row * kChannels + channel)]
                    * gate[static_cast<std::size_t>(gate_offset + channel)];
            }
        }
    }
}

void dynamic_radial_mix(
    const std::vector<float>& local,
    const std::vector<float>& radial,
    const Dpa4BlockOptions& block,
    std::size_t edge_count,
    std::vector<float>& output,
    int num_threads) {
    output.assign(local.size(), 0.0F);
    static constexpr std::array<int, 10> kRadialDegree = {
        0, 1, 2, 3, 1, 2, 3, 1, 2, 3,
    };
    static constexpr std::array<int, 10> kRadialGroup = {
        0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
    };
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(num_threads)
#endif
    for (std::ptrdiff_t edge_index = 0; edge_index < static_cast<std::ptrdiff_t>(edge_count); ++edge_index) {
        const std::size_t edge = static_cast<std::size_t>(edge_index);
        std::array<float, 25> compact{};
        row_matmul(
            radial.data() + edge * kReducedDim * kChannels,
            4 * kChannels,
            block.radial_mixer_weight,
            25,
            compact.data());
        const std::size_t edge_offset = edge * kReducedDim * kChannels;
        for (int row = 0; row < kReducedDim; ++row) {
            const int group = kRadialGroup[static_cast<std::size_t>(row)];
            const int output_degree = kRadialDegree[static_cast<std::size_t>(row)];
            const int group_size = group == 0 ? 4 : 3;
            // The +/-m=1 blocks share one radial kernel in the rank-1
            // checkpoint.  Its compact layout is [m=0:16, |m|=1:9], so
            // both signed sectors read the same 9 coefficients.
            const int compact_offset = group == 0 ? 0 : 16;
            const int output_local = group == 0 ? output_degree : output_degree - 1;
            for (int channel = 0; channel < kChannels; ++channel) {
                double value = 0.0;
                for (int input_local = 0; input_local < group_size; ++input_local) {
                    const int input_row = group == 0
                        ? input_local
                        : (row < 7 ? 4 + input_local : 7 + input_local);
                    const int coefficient = compact_offset + input_local * group_size + output_local;
                    value += static_cast<double>(compact[static_cast<std::size_t>(coefficient)])
                        * static_cast<double>(local[
                            edge_offset + static_cast<std::size_t>(input_row * kChannels + channel)]);
                }
                output[edge_offset + static_cast<std::size_t>(row * kChannels + channel)] =
                    value * block.radial_channel_basis[static_cast<std::size_t>(channel)];
            }
        }
    }
}

void rotate_reduced_to_global(
    const std::vector<float>& local,
    const EdgeData& edges,
    const Dpa4BlockOptions& block,
    const Dpa4Options& options,
    std::vector<float>& output,
    int num_threads) {
    const std::size_t edge_count = edges.src.size();
    output.assign(edge_count * kFullDim * kChannels, 0.0F);
    static constexpr std::array<float, 16> kRescale = {
        1.0F, 1.0F, 1.0F, 1.0F,
        1.2909944487358056F, 1.2909944487358056F, 1.2909944487358056F,
        1.2909944487358056F, 1.2909944487358056F,
        1.5275252316519468F, 1.5275252316519468F, 1.5275252316519468F,
        1.5275252316519468F, 1.5275252316519468F, 1.5275252316519468F,
        1.5275252316519468F,
    };
    (void)block;
    (void)options;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(num_threads)
#endif
    for (std::ptrdiff_t edge_index = 0; edge_index < static_cast<std::ptrdiff_t>(edge_count); ++edge_index) {
        const std::size_t edge = static_cast<std::size_t>(edge_index);
        const std::size_t edge_offset = edge * kReducedDim * kChannels;
        for (int global_row = 0; global_row < kFullDim; ++global_row) {
            for (int channel = 0; channel < kChannels; ++channel) {
                float value = 0.0F;
                for (int reduced = 0; reduced < kReducedDim; ++reduced) {
                    value += edges.rotation_to_m[
                        (edge * kReducedDim + static_cast<std::size_t>(reduced)) * kFullDim
                            + static_cast<std::size_t>(global_row)]
                        * local[edge_offset + static_cast<std::size_t>(reduced * kChannels + channel)];
                }
                output[edge * kFullDim * kChannels
                    + static_cast<std::size_t>(global_row * kChannels + channel)] =
                    value * kRescale[static_cast<std::size_t>(global_row)];
            }
        }
    }
}

void apply_channel_projection(
    const std::vector<float>& input,
    int input_channels,
    int output_channels,
    const std::vector<float>& weight,
    std::vector<float>& output) {
    const std::size_t rows = input.size() / static_cast<std::size_t>(input_channels);
    output.assign(rows * static_cast<std::size_t>(output_channels), 0.0F);
    for (std::size_t row = 0; row < rows; ++row) {
        row_matmul(
            input.data() + row * static_cast<std::size_t>(input_channels),
            input_channels,
            weight,
            output_channels,
            output.data() + row * static_cast<std::size_t>(output_channels));
    }
}

void project_to_grid(
    const std::vector<float>& coefficients,
    int channels,
    const std::vector<float>& matrix,
    std::vector<float>& grid) {
    const std::size_t coefficient_rows = coefficients.size() / static_cast<std::size_t>(channels);
    if (coefficient_rows != kGridCoeff) {
        throw std::invalid_argument("DPA4 grid coefficient width is not 48");
    }
    grid.assign(static_cast<std::size_t>(kGridSize) * static_cast<std::size_t>(channels), 0.0F);
    for (int grid_point = 0; grid_point < kGridSize; ++grid_point) {
        for (int channel = 0; channel < channels; ++channel) {
            float value = 0.0F;
            for (int coefficient = 0; coefficient < kGridCoeff; ++coefficient) {
                value = std::fma(
                    matrix[static_cast<std::size_t>(grid_point * kGridCoeff + coefficient)],
                    coefficients[static_cast<std::size_t>(coefficient * channels + channel)],
                    value);
            }
            grid[static_cast<std::size_t>(grid_point * channels + channel)] = value;
        }
    }
}

void project_from_grid(
    const std::vector<float>& grid,
    int channels,
    const std::vector<float>& matrix,
    std::vector<float>& coefficients) {
    if (grid.size() != static_cast<std::size_t>(kGridSize) * static_cast<std::size_t>(channels)) {
        throw std::invalid_argument("DPA4 grid width is not 152");
    }
    coefficients.assign(static_cast<std::size_t>(kGridCoeff) * static_cast<std::size_t>(channels), 0.0F);
    for (int coefficient = 0; coefficient < kGridCoeff; ++coefficient) {
        for (int channel = 0; channel < channels; ++channel) {
            float value = 0.0F;
            for (int grid_point = 0; grid_point < kGridSize; ++grid_point) {
                value = std::fma(
                    matrix[static_cast<std::size_t>(coefficient * kGridSize + grid_point)],
                    grid[static_cast<std::size_t>(grid_point * channels + channel)],
                    value);
            }
            coefficients[static_cast<std::size_t>(coefficient * channels + channel)] = value;
        }
    }
}

void expand_frames(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    std::vector<float>& output) {
    // [packed coefficient, 64] -> [packed coefficient, frame(3), 64].
    output.assign(static_cast<std::size_t>(kGridCoeff) * kChannels, 0.0F);
    static constexpr std::array<int, 16> kDegree = {
        0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
    };
    for (int row = 0; row < kFullDim; ++row) {
        const int degree = kDegree[static_cast<std::size_t>(row)];
        for (int frame = 0; frame < kFrames; ++frame) {
            for (int output_channel = 0; output_channel < kChannels; ++output_channel) {
                double value = 0.0;
                for (int input_channel = 0; input_channel < kChannels; ++input_channel) {
                    value += static_cast<double>(input[
                        static_cast<std::size_t>(row * kChannels + input_channel)])
                        * static_cast<double>(weight[static_cast<std::size_t>(
                            degree * kChannels * (kFrames * kChannels)
                            + input_channel * (kFrames * kChannels) + frame * kChannels + output_channel)]);
                }
                output[static_cast<std::size_t>((row * kFrames + frame) * kChannels + output_channel)] =
                    static_cast<float>(value);
            }
        }
    }
}

void contract_frames(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    std::vector<float>& output) {
    output.assign(static_cast<std::size_t>(kFullDim) * kChannels, 0.0F);
    static constexpr std::array<int, 16> kDegree = {
        0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
    };
    for (int row = 0; row < kFullDim; ++row) {
        const int degree = kDegree[static_cast<std::size_t>(row)];
        for (int output_channel = 0; output_channel < kChannels; ++output_channel) {
            double value = 0.0;
            for (int frame = 0; frame < kFrames; ++frame) {
                for (int input_channel = 0; input_channel < kChannels; ++input_channel) {
                    value += static_cast<double>(input[static_cast<std::size_t>(
                        (row * kFrames + frame) * kChannels + input_channel)])
                        * static_cast<double>(weight[static_cast<std::size_t>(
                            degree * (kFrames * kChannels) * kChannels
                            + (frame * kChannels + input_channel) * kChannels + output_channel)]);
                }
            }
            output[static_cast<std::size_t>(row * kChannels + output_channel)] =
                static_cast<float>(value);
        }
    }
}

void scalar_swiglu(
    const std::vector<float>& input,
    int channels,
    std::vector<float>& output) {
    output.resize(static_cast<std::size_t>(channels));
    for (int channel = 0; channel < channels; ++channel) {
        const float gate = input[static_cast<std::size_t>(channel)];
        const float value = input[static_cast<std::size_t>(channels + channel)];
        output[static_cast<std::size_t>(channel)] = gate * sigmoid(gate) * value;
    }
}

void message_grid_one(
    const std::vector<float>& query,
    const std::vector<float>& context,
    const Dpa4BlockOptions& block,
    const Dpa4Options& options,
    std::vector<float>& output) {
    std::vector<float> query_frame;
    std::vector<float> context_frame;
    expand_frames(query, block.message_frame_expand, query_frame);
    expand_frames(context, block.message_frame_expand, context_frame);

    std::vector<float> query_grid;
    std::vector<float> context_grid;
    project_to_grid(query_frame, kChannels, options.grid_to, query_grid);
    project_to_grid(context_frame, kChannels, options.grid_to, context_grid);
    std::vector<float> product_grid(query_grid.size(), 0.0F);
    for (std::size_t index = 0; index < product_grid.size(); ++index) {
        product_grid[index] = query_grid[index] * context_grid[index];
    }
    std::vector<float> product_coeff;
    project_from_grid(product_grid, kChannels, options.grid_from, product_coeff);

    std::vector<float> scalar_pair(2U * kChannels, 0.0F);
    for (int channel = 0; channel < kChannels; ++channel) {
        scalar_pair[static_cast<std::size_t>(channel)] = query[static_cast<std::size_t>(channel)];
        scalar_pair[static_cast<std::size_t>(kChannels + channel)] = context[static_cast<std::size_t>(channel)];
    }
    std::vector<float> scalar_out;
    scalar_swiglu(scalar_pair, kChannels, scalar_out);
    std::array<float, kChannels> scalar_gate{};
    for (int output_channel = 0; output_channel < kChannels; ++output_channel) {
        double value = 0.0;
        for (int input_channel = 0; input_channel < 2 * kChannels; ++input_channel) {
            value += static_cast<double>(scalar_pair[static_cast<std::size_t>(input_channel)])
                * static_cast<double>(block.message_scalar_gate[
                    static_cast<std::size_t>(input_channel * kChannels + output_channel)]);
        }
        scalar_gate[static_cast<std::size_t>(output_channel)] = sigmoid(static_cast<float>(value));
    }
    for (int coefficient = 0; coefficient < kGridCoeff; ++coefficient) {
        for (int channel = 0; channel < kChannels; ++channel) {
            product_coeff[static_cast<std::size_t>(coefficient * kChannels + channel)]
                *= scalar_gate[static_cast<std::size_t>(channel)];
        }
    }
    for (int channel = 0; channel < kChannels; ++channel) {
        product_coeff[static_cast<std::size_t>(channel)] += scalar_out[static_cast<std::size_t>(channel)];
    }

    contract_frames(product_coeff, block.message_frame_contract, output);
    for (int row = 0; row < kFullDim; ++row) {
        for (int channel = 0; channel < kChannels; ++channel) {
            output[static_cast<std::size_t>(row * kChannels + channel)]
                *= block.message_residual_scale[static_cast<std::size_t>(channel)];
        }
    }
}

void block_grid_branch_one(
    const std::vector<float>& input,
    const Dpa4BlockOptions& block,
    const Dpa4Options& options,
    std::vector<float>& output) {
    std::vector<float> left(static_cast<std::size_t>(kGridCoeff) * 192U, 0.0F);
    std::vector<float> right(left.size(), 0.0F);
    for (int row = 0; row < kFullDim; ++row) {
        for (int frame = 0; frame < kFrames; ++frame) {
            for (int channel = 0; channel < 192; ++channel) {
                const std::size_t frame_offset = static_cast<std::size_t>((row * kFrames + frame) * 192 + channel);
                left[frame_offset] = input[static_cast<std::size_t>(row * 1152 + frame * 192 + channel)];
                right[frame_offset] = input[static_cast<std::size_t>(row * 1152 + 576 + frame * 192 + channel)];
            }
        }
    }
    std::vector<float> left_projected;
    std::vector<float> right_projected;
    apply_channel_projection(left, 192, 192, block.ffn_grid_left, left_projected);
    apply_channel_projection(right, 192, 192, block.ffn_grid_right, right_projected);
    std::vector<float> left_grid;
    std::vector<float> right_grid;
    project_to_grid(left_projected, 192, options.grid_to, left_grid);
    project_to_grid(right_projected, 192, options.grid_to, right_grid);
    std::vector<float> product_grid(left_grid.size(), 0.0F);
    for (std::size_t index = 0; index < product_grid.size(); ++index) {
        product_grid[index] = left_grid[index] * right_grid[index];
    }
    std::vector<float> product_coeff;
    project_from_grid(product_grid, 192, options.grid_from, product_coeff);

    std::vector<float> scalar_pair(384U, 0.0F);
    for (int channel = 0; channel < 192; ++channel) {
        scalar_pair[static_cast<std::size_t>(channel)] = left[static_cast<std::size_t>(channel)];
        scalar_pair[static_cast<std::size_t>(192 + channel)] = right[static_cast<std::size_t>(channel)];
    }
    std::vector<float> scalar_out;
    scalar_swiglu(scalar_pair, 192, scalar_out);
    std::array<float, 192> scalar_gate{};
    for (int output_channel = 0; output_channel < 192; ++output_channel) {
        double value = 0.0;
        for (int input_channel = 0; input_channel < 384; ++input_channel) {
            value += static_cast<double>(scalar_pair[static_cast<std::size_t>(input_channel)])
                * static_cast<double>(block.ffn_scalar_gate[
                    static_cast<std::size_t>(input_channel * 192 + output_channel)]);
        }
        scalar_gate[static_cast<std::size_t>(output_channel)] = sigmoid(static_cast<float>(value));
    }
    // The deployed model has one branch; the softmax over one route is exactly 1.
    (void)block.ffn_grid_router;
    std::vector<float> grid_output;
    apply_channel_projection(product_coeff, 192, 192, block.ffn_grid_out, grid_output);
    for (int coefficient = 0; coefficient < kGridCoeff; ++coefficient) {
        for (int channel = 0; channel < 192; ++channel) {
            grid_output[static_cast<std::size_t>(coefficient * 192 + channel)]
                *= scalar_gate[static_cast<std::size_t>(channel)];
        }
    }
    for (int channel = 0; channel < 192; ++channel) {
        grid_output[static_cast<std::size_t>(channel)] += scalar_out[static_cast<std::size_t>(channel)];
    }
    output = std::move(grid_output);
}

void output_grid_mlp_one(
    const std::vector<float>& input,
    const Dpa4Options& options,
    std::vector<float>& output) {
    std::vector<float> fused(static_cast<std::size_t>(kGridCoeff) * 384U, 0.0F);
    for (int row = 0; row < kFullDim; ++row) {
        for (int frame = 0; frame < kFrames; ++frame) {
            for (int channel = 0; channel < 192; ++channel) {
                const std::size_t out_offset = static_cast<std::size_t>((row * kFrames + frame) * 384 + channel);
                fused[out_offset] = input[static_cast<std::size_t>(row * 1152 + frame * 192 + channel)];
                fused[out_offset + 192U] = input[static_cast<std::size_t>(row * 1152 + 576 + frame * 192 + channel)];
            }
        }
    }
    std::vector<float> left;
    std::vector<float> right;
    apply_channel_projection(fused, 384, 384, options.output_grid_left, left);
    apply_channel_projection(fused, 384, 384, options.output_grid_right, right);
    std::vector<float> left_grid;
    std::vector<float> right_grid;
    project_to_grid(left, 384, options.grid_to, left_grid);
    project_to_grid(right, 384, options.grid_to, right_grid);
    std::vector<float> product_grid(left_grid.size(), 0.0F);
    for (std::size_t index = 0; index < product_grid.size(); ++index) {
        product_grid[index] = left_grid[index] * right_grid[index];
    }
    std::vector<float> product_coeff;
    project_from_grid(product_grid, 384, options.grid_from, product_coeff);
    std::vector<float> projected;
    apply_channel_projection(product_coeff, 384, 192, options.output_grid_out, projected);

    std::vector<float> scalar_pair(384U, 0.0F);
    for (int channel = 0; channel < 192; ++channel) {
        scalar_pair[static_cast<std::size_t>(channel)] = input[static_cast<std::size_t>(channel)];
        scalar_pair[static_cast<std::size_t>(192 + channel)] = input[static_cast<std::size_t>(576 + channel)];
    }
    std::vector<float> scalar_out;
    scalar_swiglu(scalar_pair, 192, scalar_out);
    std::array<float, 192> scalar_gate{};
    for (int output_channel = 0; output_channel < 192; ++output_channel) {
        double value = 0.0;
        for (int input_channel = 0; input_channel < 384; ++input_channel) {
            value += static_cast<double>(scalar_pair[static_cast<std::size_t>(input_channel)])
                * static_cast<double>(options.output_scalar_gate[
                    static_cast<std::size_t>(input_channel * 192 + output_channel)]);
        }
        scalar_gate[static_cast<std::size_t>(output_channel)] = sigmoid(static_cast<float>(value));
    }
    for (int coefficient = 0; coefficient < kGridCoeff; ++coefficient) {
        for (int channel = 0; channel < 192; ++channel) {
            projected[static_cast<std::size_t>(coefficient * 192 + channel)]
                *= scalar_gate[static_cast<std::size_t>(channel)];
        }
    }
    for (int channel = 0; channel < 192; ++channel) {
        projected[static_cast<std::size_t>(channel)] += scalar_out[static_cast<std::size_t>(channel)];
    }
    output = std::move(projected);
}

void scalar_rms_norm(
    const float* input,
    const std::vector<float>& scale,
    float* output) {
    float mean_square = 0.0F;
    for (int channel = 0; channel < kChannels; ++channel) {
        mean_square += input[channel] * input[channel];
    }
    const float inverse = 1.0F / std::sqrt(mean_square / static_cast<float>(kChannels)
        + kEpsilon);
    for (int channel = 0; channel < kChannels; ++channel) {
        output[channel] = input[channel] * inverse * scale[static_cast<std::size_t>(channel)];
    }
}

void run_block(
    const std::vector<float>& input,
    const EdgeData& edges,
    const std::vector<float>& radial_full,
    const Dpa4BlockOptions& block,
    const Dpa4Options& options,
    std::int64_t nodes,
    std::vector<float>& output) {
    std::vector<float> so2_input;
    if (block.pre_norm_enabled) {
        equivariant_norm(
            input,
            block.pre_norm_scale,
            block.pre_norm_bias,
            block.pre_norm_balance,
            nodes,
            so2_input,
            options.num_threads);
    } else {
        so2_input = input;
    }

    std::vector<float> pre_focus;
    apply_so3_linear(
        so2_input, nodes, kChannels, kChannels,
        block.pre_focus_weight, pre_focus, options.num_threads);

    const std::size_t edge_count = edges.src.size();
    std::vector<float> local(edge_count * kReducedDim * kChannels, 0.0F);
    std::vector<float> radial_reduced(edge_count * kReducedDim * kChannels, 0.0F);
    static constexpr std::array<int, 10> kReducedDegree = {
        0, 1, 2, 3, 1, 2, 3, 1, 2, 3,
    };
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options.num_threads)
#endif
    for (std::ptrdiff_t edge = 0; edge < static_cast<std::ptrdiff_t>(edge_count); ++edge) {
        const std::size_t e = static_cast<std::size_t>(edge);
        const std::int32_t source = edges.src[e];
        for (int reduced = 0; reduced < kReducedDim; ++reduced) {
            for (int channel = 0; channel < kChannels; ++channel) {
                double value = 0.0;
                for (int global = 0; global < kFullDim; ++global) {
                    value += static_cast<double>(edges.rotation_to_m[
                        (e * kReducedDim + static_cast<std::size_t>(reduced)) * kFullDim
                            + static_cast<std::size_t>(global)])
                        * static_cast<double>(pre_focus[node_index(source, global, channel)]);
                }
                local[e * kReducedDim * kChannels
                    + static_cast<std::size_t>(reduced * kChannels + channel)] =
                    static_cast<float>(value);
                const int degree = kReducedDegree[static_cast<std::size_t>(reduced)];
                radial_reduced[e * kReducedDim * kChannels
                    + static_cast<std::size_t>(reduced * kChannels + channel)] =
                    radial_full[e * 4U * kChannels + static_cast<std::size_t>(degree * kChannels + channel)];
            }
        }
    }
    std::vector<float> mixed_local;
    dynamic_radial_mix(
        local, radial_reduced, block, edge_count, mixed_local, options.num_threads);
    local.swap(mixed_local);
    for (int layer = 0; layer < 4; ++layer) {
        std::vector<float> linear;
        apply_so2_linear(local, block, layer, edge_count, linear, options.num_threads);
        if (layer < 3) {
            std::vector<float> activated;
            apply_so2_gate(
                linear,
                block.so2_gate_weight[static_cast<std::size_t>(layer)],
                edge_count,
                activated,
                options.num_threads);
            for (std::size_t index = 0; index < local.size(); ++index) {
                local[index] += activated[index];
            }
        } else {
            for (std::size_t index = 0; index < local.size(); ++index) {
                local[index] += linear[index];
            }
        }
    }
    std::vector<float> edge_message;
    rotate_reduced_to_global(
        local, edges, block, options, edge_message, options.num_threads);

    std::vector<float> q(static_cast<std::size_t>(nodes) * kChannels, 0.0F);
    std::vector<float> key(static_cast<std::size_t>(nodes) * kChannels, 0.0F);
    for (std::int64_t node = 0; node < nodes; ++node) {
        std::array<float, kChannels> normalized{};
        scalar_rms_norm(
            pre_focus.data() + node_index(node, 0, 0),
            block.attn_qk_scale,
            normalized.data());
        row_matmul(
            normalized.data(), kChannels, block.attn_q_weight, kChannels,
            q.data() + static_cast<std::size_t>(node) * kChannels);
        row_matmul(
            normalized.data(), kChannels, block.attn_k_weight, kChannels,
            key.data() + static_cast<std::size_t>(node) * kChannels);
    }

    // Attention has a positive learned null mass.  Compute every destination
    // segment independently, which is both race-free and preserves the edge
    // order emitted by NeighborGraph for reproducible reductions.
    std::vector<double> attention_accum(
        static_cast<std::size_t>(nodes) * kFullDim * kChannels, 0.0);
    const double null_logit = std::log(
        static_cast<double>(softplus(block.attn_z_bias_raw[0]))
        + static_cast<double>(kEpsilon));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options.num_threads)
#endif
    for (std::int64_t node = 0; node < nodes; ++node) {
        const std::int64_t begin = edges.offsets[static_cast<std::size_t>(node)];
        const std::int64_t end = edges.offsets[static_cast<std::size_t>(node + 1)];
        double max_logit = null_logit;
        for (std::int64_t edge = begin; edge < end; ++edge) {
            const std::size_t e = static_cast<std::size_t>(edge);
            const float env = edges.envelope[e];
            if (env <= 0.0F) {
                continue;
            }
            const std::int32_t source = edges.src[e];
            double dot = 0.0;
            for (int channel = 0; channel < kChannels; ++channel) {
                dot += static_cast<double>(q[static_cast<std::size_t>(node) * kChannels
                    + static_cast<std::size_t>(channel)]
                    * static_cast<double>(key[static_cast<std::size_t>(source) * kChannels
                        + static_cast<std::size_t>(channel)]));
            }
            double radial_bias = 0.0;
            for (int channel = 0; channel < kChannels; ++channel) {
                radial_bias += static_cast<double>(radial_reduced[
                    e * kReducedDim * kChannels + static_cast<std::size_t>(channel)]
                    * static_cast<double>(block.attn_logit_weight[static_cast<std::size_t>(channel)]));
            }
            const double logit = dot * (1.0 / 8.0) + radial_bias
                + 2.0 * std::log(static_cast<double>(env));
            max_logit = std::max(max_logit, logit);
        }
        double denominator = std::exp(null_logit - max_logit);
        for (std::int64_t edge = begin; edge < end; ++edge) {
            const std::size_t e = static_cast<std::size_t>(edge);
            const float env = edges.envelope[e];
            if (env <= 0.0F) {
                continue;
            }
            const std::int32_t source = edges.src[e];
            double dot = 0.0;
            for (int channel = 0; channel < kChannels; ++channel) {
                dot += static_cast<double>(q[static_cast<std::size_t>(node) * kChannels
                    + static_cast<std::size_t>(channel)]
                    * static_cast<double>(key[static_cast<std::size_t>(source) * kChannels
                        + static_cast<std::size_t>(channel)]));
            }
            double radial_bias = 0.0;
            for (int channel = 0; channel < kChannels; ++channel) {
                radial_bias += static_cast<double>(radial_reduced[
                    e * kReducedDim * kChannels + static_cast<std::size_t>(channel)]
                    * static_cast<double>(block.attn_logit_weight[static_cast<std::size_t>(channel)]));
            }
            const double logit = dot * (1.0 / 8.0) + radial_bias
                + 2.0 * std::log(static_cast<double>(env));
            denominator += std::exp(logit - max_logit);
        }
        const double inverse_denominator = 1.0 / denominator;
        for (std::int64_t edge = begin; edge < end; ++edge) {
            const std::size_t e = static_cast<std::size_t>(edge);
            const float env = edges.envelope[e];
            if (env <= 0.0F) {
                continue;
            }
            const std::int32_t source = edges.src[e];
            double dot = 0.0;
            for (int channel = 0; channel < kChannels; ++channel) {
                dot += static_cast<double>(q[static_cast<std::size_t>(node) * kChannels
                    + static_cast<std::size_t>(channel)]
                    * static_cast<double>(key[static_cast<std::size_t>(source) * kChannels
                        + static_cast<std::size_t>(channel)]));
            }
            double radial_bias = 0.0;
            for (int channel = 0; channel < kChannels; ++channel) {
                radial_bias += static_cast<double>(radial_reduced[
                    e * kReducedDim * kChannels + static_cast<std::size_t>(channel)]
                    * static_cast<double>(block.attn_logit_weight[static_cast<std::size_t>(channel)]));
            }
            const double logit = dot * (1.0 / 8.0) + radial_bias
                + 2.0 * std::log(static_cast<double>(env));
            const double alpha = std::exp(logit - max_logit) * inverse_denominator;
            for (int row = 0; row < kFullDim; ++row) {
                for (int channel = 0; channel < kChannels; ++channel) {
                    attention_accum[node_index(node, row, channel)] +=
                        alpha * edge_message[edge_index(e, row, channel)];
                }
            }
        }

        std::array<float, kChannels> normalized_gate{};
        scalar_rms_norm(
            pre_focus.data() + node_index(node, 0, 0),
            block.attn_output_gate_scale,
            normalized_gate.data());
        double gate_logit = 0.0;
        for (int channel = 0; channel < kChannels; ++channel) {
            gate_logit += static_cast<double>(normalized_gate[static_cast<std::size_t>(channel)])
                * static_cast<double>(block.attn_gate_weight[static_cast<std::size_t>(channel)]);
        }
        const float gate = sigmoid(static_cast<float>(gate_logit));
        for (int row = 0; row < kFullDim; ++row) {
            for (int channel = 0; channel < kChannels; ++channel) {
                attention_accum[node_index(node, row, channel)] *= gate;
            }
        }
    }

    std::vector<float> attention_output(attention_accum.size(), 0.0F);
    for (std::size_t index = 0; index < attention_accum.size(); ++index) {
        attention_output[index] = static_cast<float>(attention_accum[index]);
    }
    // The grid cross-product is a residual branch evaluated against the
    // attention aggregate, then added before the final post-focus mix.
    const std::vector<float>& aggregate = attention_output;
    std::vector<float> post_input(aggregate.size(), 0.0F);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options.num_threads)
#endif
    for (std::int64_t node = 0; node < nodes; ++node) {
        const std::size_t offset = node_index(node, 0, 0);
        std::vector<float> query_node(
            pre_focus.begin() + offset, pre_focus.begin() + offset + kFullDim * kChannels);
        // The vendor call is message_node_grid_product(out, x): the
        // attention aggregate is the query and the pre-focus node feature is
        // the context.
        std::vector<float> context_node(
            aggregate.begin() + offset, aggregate.begin() + offset + kFullDim * kChannels);
        std::vector<float> grid_output;
        message_grid_one(context_node, query_node, block, options, grid_output);
        for (int row = 0; row < kFullDim; ++row) {
            for (int channel = 0; channel < kChannels; ++channel) {
                post_input[offset + static_cast<std::size_t>(row * kChannels + channel)] =
                    context_node[static_cast<std::size_t>(row * kChannels + channel)]
                    + grid_output[static_cast<std::size_t>(row * kChannels + channel)];
            }
        }
    }
    std::vector<float> so2_output;
    apply_so3_linear(
        post_input, nodes, kChannels, kChannels,
        block.post_focus_weight, so2_output, options.num_threads);
    if (block.post_norm_enabled) {
        std::vector<float> normalized_output;
        equivariant_norm(
            so2_output,
            block.post_norm_scale,
            block.post_norm_bias,
            block.post_norm_balance,
            nodes,
            normalized_output,
            options.num_threads);
        so2_output.swap(normalized_output);
    }
    std::vector<float> state(input.size(), 0.0F);
    for (std::size_t index = 0; index < state.size(); ++index) {
        state[index] = input[index] + so2_output[index];
    }
    std::vector<float> ffn_input;
    if (block.ffn_norm_enabled) {
        equivariant_norm(
            state,
            block.ffn_norm_scale,
            block.ffn_norm_bias,
            block.ffn_norm_balance,
            nodes,
            ffn_input,
            options.num_threads);
    } else {
        ffn_input = state;
    }
    std::vector<float> ffn_hidden;
    apply_so3_linear(
        ffn_input, nodes, kChannels, 1152,
        block.ffn_linear1, ffn_hidden, options.num_threads);
    std::vector<float> ffn_act(static_cast<std::size_t>(nodes) * kFullDim * 576U, 0.0F);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options.num_threads)
#endif
    for (std::int64_t node = 0; node < nodes; ++node) {
        const std::size_t wide_offset = static_cast<std::size_t>(node) * kFullDim * 1152U;
        std::vector<float> node_input(
            ffn_hidden.begin() + wide_offset,
            ffn_hidden.begin() + wide_offset + kFullDim * 1152U);
        std::vector<float> node_output;
        block_grid_branch_one(node_input, block, options, node_output);
        std::copy(
            node_output.begin(), node_output.end(),
            ffn_act.begin() + static_cast<std::size_t>(node) * kFullDim * 576U);
    }
    std::vector<float> ffn_output;
    apply_so3_linear(
        ffn_act, nodes, 576, kChannels,
        block.ffn_linear2, ffn_output, options.num_threads);
    output.resize(input.size());
    for (std::size_t index = 0; index < output.size(); ++index) {
        output[index] = state[index] + ffn_output[index];
    }
}

} // namespace

Dpa4Calculator::Dpa4Calculator(Dpa4Options options)
    : options_(std::move(options)),
      wigner_(make_wigner_payload(options_)) {
    validate_options(options_);
}

std::int64_t Dpa4Calculator::feature_count() const noexcept {
    return feature_count_;
}

void Dpa4Calculator::close() noexcept {
    closed_.store(true, std::memory_order_release);
}

bool Dpa4Calculator::closed() const noexcept {
    return closed_.load(std::memory_order_acquire);
}

void Dpa4Calculator::compute(
    const StructureBatchView& batch,
    const std::int32_t* type_indices,
    double* output,
    const std::shared_ptr<ComputeControl>& control) const {
    if (closed()) {
        throw std::runtime_error("DPA4 descriptor is closed");
    }
    detail::validate_batch(batch);
    if (type_indices == nullptr && batch.atoms > 0) {
        throw std::invalid_argument("DPA4 type indices cannot be null");
    }
    if (output == nullptr && batch.atoms > 0) {
        throw std::invalid_argument("DPA4 output cannot be null");
    }
    for (std::int64_t atom = 0; atom < batch.atoms; ++atom) {
        if (type_indices[atom] < 0 || type_indices[atom] >= options_.ntypes) {
            throw std::invalid_argument("DPA4 type index is outside the checkpoint type map");
        }
    }
    if (batch.atoms == 0) {
        return;
    }
    detail::check_cancelled(control);

    EdgeData edges = build_edges(batch, options_.rcut, options_.num_threads, control);
    detail::check_cancelled(control);
    fill_radial_basis(edges, options_);
    build_rotations(edges, options_);
    detail::check_cancelled(control);

    std::vector<float> type_features;
    std::vector<float> film;
    compute_environment(
        edges, options_, type_indices, batch.atoms, type_features, film);
    detail::check_cancelled(control);

    std::vector<float> radial;
    radial_features(edges, options_, radial);
    detail::check_cancelled(control);

    std::vector<float> x;
    initial_features(edges, options_, radial, batch.atoms, x);
    add_type_slice(x, type_features, batch.atoms, options_.num_threads);

    // The edge type term is fused after the geometric initial embedding.  It
    // uses the unfused lookup table, whereas the node l=0 seed above contains
    // the environment FiLM transform.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options_.num_threads)
#endif
    for (std::ptrdiff_t edge = 0; edge < static_cast<std::ptrdiff_t>(edges.src.size()); ++edge) {
        const std::size_t e = static_cast<std::size_t>(edge);
        const int source_type = type_indices[edges.src[e]];
        const int destination_type = type_indices[edges.dst[e]];
        for (int degree = 0; degree < 4; ++degree) {
            for (int channel = 0; channel < kChannels; ++channel) {
                radial[e * 4U * kChannels + static_cast<std::size_t>(degree * kChannels + channel)] +=
                    options_.type_embedding[static_cast<std::size_t>(source_type * kChannels + channel)]
                    + options_.type_embedding[static_cast<std::size_t>(destination_type * kChannels + channel)];
            }
        }
    }
    // ``build_neighbor_graph`` omits padded slots.  The native DeepMD graph
    // lower skips all interaction blocks for a frame with no real edge;
    // executing them against an empty message aggregate changes the
    // descriptor through the residual/FFN path and produces a large,
    // non-numerical mismatch.  A StructureBatch can mix connected and
    // isolated frames, so retain the pre-block state for each inactive frame
    // after running a block for the connected frames.
    std::vector<bool> active_structures(static_cast<std::size_t>(batch.structures), false);
    for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
        const std::int64_t begin = batch.offsets[structure];
        const std::int64_t end = batch.offsets[structure + 1];
        for (std::int64_t atom = begin; atom < end; ++atom) {
            if (edges.offsets[static_cast<std::size_t>(atom + 1)]
                > edges.offsets[static_cast<std::size_t>(atom)]) {
                active_structures[static_cast<std::size_t>(structure)] = true;
                break;
            }
        }
    }
    if (!edges.src.empty()) {
        for (const Dpa4BlockOptions& block : options_.blocks) {
            std::vector<float> block_output;
            run_block(
                x, edges, radial, block, options_, batch.atoms, block_output);
            for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
                if (active_structures[static_cast<std::size_t>(structure)]) {
                    continue;
                }
                const std::int64_t begin = batch.offsets[structure];
                const std::int64_t end = batch.offsets[structure + 1];
                const std::size_t row_width = static_cast<std::size_t>(kFullDim * kChannels);
                for (std::int64_t atom = begin; atom < end; ++atom) {
                    const std::size_t offset = static_cast<std::size_t>(atom) * row_width;
                    std::copy(
                        x.begin() + static_cast<std::ptrdiff_t>(offset),
                        x.begin() + static_cast<std::ptrdiff_t>(offset + row_width),
                        block_output.begin() + static_cast<std::ptrdiff_t>(offset));
                }
            }
            x.swap(block_output);
            detail::check_cancelled(control);
        }
    }

    std::vector<float> output_hidden;
    apply_so3_linear(
        x, batch.atoms, kChannels, 1152,
        options_.output_linear1, output_hidden, options_.num_threads);
    std::vector<float> output_act(
        static_cast<std::size_t>(batch.atoms) * kFullDim * 576U, 0.0F);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options_.num_threads)
#endif
    for (std::int64_t node = 0; node < batch.atoms; ++node) {
        const std::size_t wide_offset = static_cast<std::size_t>(node) * kFullDim * 1152U;
        std::vector<float> node_input(
            output_hidden.begin() + wide_offset,
            output_hidden.begin() + wide_offset + kFullDim * 1152U);
        std::vector<float> node_output;
        output_grid_mlp_one(node_input, options_, node_output);
        std::copy(
            node_output.begin(), node_output.end(),
            output_act.begin() + static_cast<std::size_t>(node) * kFullDim * 576U);
    }
    std::vector<float> output_update;
    apply_so3_linear(
        output_act, batch.atoms, 576, kChannels,
        options_.output_linear2, output_update, options_.num_threads);
    for (std::int64_t node = 0; node < batch.atoms; ++node) {
        for (int channel = 0; channel < kChannels; ++channel) {
            const std::size_t index = node_index(node, 0, channel);
            output[static_cast<std::size_t>(node) * kChannels + static_cast<std::size_t>(channel)] =
                static_cast<double>(x[index] + output_update[index]);
        }
        if (control) {
            control->mark_completed();
        }
    }
    detail::check_cancelled(control);
}

} // namespace mdescriptor
