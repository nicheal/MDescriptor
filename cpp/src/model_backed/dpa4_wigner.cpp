#include "mdescriptor/dpa4_wigner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace mdescriptor {
namespace {

constexpr float kOne = 1.0F;
constexpr float kZero = 0.0F;

float safe_norm(const Dpa4EdgeVector& edge, float eps) {
    return std::sqrt(edge.x * edge.x + edge.y * edge.y + edge.z * edge.z + eps * eps);
}

float quaternion_norm(const Dpa4Quaternion& quaternion, float eps) {
    return std::sqrt(
        quaternion.w * quaternion.w
        + quaternion.x * quaternion.x
        + quaternion.y * quaternion.y
        + quaternion.z * quaternion.z
        + eps * eps);
}

Dpa4Quaternion normalize_quaternion_unchecked(
    const Dpa4Quaternion& quaternion,
    float eps) {
    // Evaluate the norm in double precision before crossing back to the
    // model's fp32 storage boundary.  This avoids a one-ulp drift in the
    // chart blend and in the quaternion passed to the low-order kernels.
    const double w = static_cast<double>(quaternion.w);
    const double x = static_cast<double>(quaternion.x);
    const double y = static_cast<double>(quaternion.y);
    const double z = static_cast<double>(quaternion.z);
    const double e = static_cast<double>(eps);
    const float divisor = static_cast<float>(std::sqrt(w * w + x * x + y * y + z * z + e * e));
    return {
        quaternion.w / divisor,
        quaternion.x / divisor,
        quaternion.y / divisor,
        quaternion.z / divisor,
    };
}

float smooth_step_cinf(float value) {
    const float clamped = std::max(kZero, std::min(kOne, value));
    if (clamped <= kZero) {
        return kZero;
    }
    if (clamped >= kOne) {
        return kOne;
    }

    // This is the dtype-level floor used by the Python implementation, not
    // the quaternion epsilon supplied by the caller.
    const float dtype_eps = std::numeric_limits<float>::epsilon();
    const float left = std::exp(-kOne / std::max(clamped, dtype_eps));
    const float right = std::exp(-kOne / std::max(kOne - clamped, dtype_eps));
    return left / (left + right);
}

Dpa4Quaternion nlerp_shortest(
    const Dpa4Quaternion& q0,
    const Dpa4Quaternion& q1,
    float weight,
    float eps) {
    const float dot = q0.w * q1.w + q0.x * q1.x + q0.y * q1.y + q0.z * q1.z;
    const float sign = dot < kZero ? -kOne : kOne;
    const Dpa4Quaternion aligned{
        sign * q1.w,
        sign * q1.x,
        sign * q1.y,
        sign * q1.z,
    };
    const Dpa4Quaternion blended{
        (kOne - weight) * q0.w + weight * aligned.w,
        (kOne - weight) * q0.x + weight * aligned.x,
        (kOne - weight) * q0.y + weight * aligned.y,
        (kOne - weight) * q0.z + weight * aligned.z,
    };
    return normalize_quaternion_unchecked(blended, eps);
}

Dpa4Quaternion build_edge_quaternion_from_length(
    const Dpa4EdgeVector& edge,
    float edge_length,
    float eps) {
    const Dpa4EdgeVector unit{
        edge.x / edge_length,
        edge.y / edge_length,
        edge.z / edge_length,
    };

    // +Z chart: regular away from the -Z pole.
    const Dpa4Quaternion q_pos = normalize_quaternion_unchecked(
        {kOne + unit.z, unit.y, -unit.x, kZero},
        eps);
    // -Z chart: regular away from the +Z pole.
    const Dpa4Quaternion q_neg = normalize_quaternion_unchecked(
        {-unit.x, kZero, kOne - unit.z, unit.y},
        eps);
    const float blend = smooth_step_cinf(0.5F * (unit.z + kOne));
    return nlerp_shortest(q_neg, q_pos, blend, eps);
}

void rotation_matrix_from_normalized(
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

    output[0] = kOne - 2.0F * (y2 + z2);
    output[1] = 2.0F * (xy - wz);
    output[2] = 2.0F * (xz + wy);
    output[3] = 2.0F * (xy + wz);
    output[4] = kOne - 2.0F * (x2 + z2);
    output[5] = 2.0F * (yz - wx);
    output[6] = 2.0F * (xz - wy);
    output[7] = 2.0F * (yz + wx);
    output[8] = kOne - 2.0F * (x2 + y2);
}

void l1_block_from_normalized(
    const Dpa4Quaternion& quaternion,
    float* output) {
    float rotation[9];
    rotation_matrix_from_normalized(quaternion, rotation);
    constexpr int permutation[3] = {1, 2, 0};
    constexpr float signs[3] = {-kOne, -kOne, kOne};
    for (int row = 0; row < kDpa4WignerL1Dimension; ++row) {
        for (int column = 0; column < kDpa4WignerL1Dimension; ++column) {
            output[row * kDpa4WignerL1Dimension + column] =
                rotation[permutation[row] * kDpa4WignerL1Dimension + permutation[column]]
                * signs[row] * signs[column];
        }
    }
}

void require_output(const float* output, std::size_t size, const char* name) {
    if (size != 0 && output == nullptr) {
        throw std::invalid_argument(std::string(name) + " output is null");
    }
}

std::size_t expected_monomial_count(int degree) {
    if (degree == 2) {
        return kDpa4WignerL2MonomialCount;
    }
    if (degree == 3) {
        return kDpa4WignerL3MonomialCount;
    }
    throw std::invalid_argument("DPA4 low-order Wigner degree must be 2 or 3");
}

void validate_monomial_payload(
    const Dpa4WignerMonomialPayload& payload,
    int expected_degree) {
    if (payload.degree != expected_degree
        || payload.monomial_degree != 2 * expected_degree
        || payload.monomial_count != expected_monomial_count(expected_degree)
        || payload.exponents == nullptr
        || payload.coefficients == nullptr) {
        throw std::invalid_argument("invalid DPA4 Wigner monomial payload");
    }

    for (std::size_t monomial = 0; monomial < payload.monomial_count; ++monomial) {
        std::int64_t exponent_sum = 0;
        for (int component = 0; component < 4; ++component) {
            const std::int64_t exponent = payload.exponents[monomial * 4 + component];
            if (exponent < 0 || exponent > payload.monomial_degree) {
                throw std::invalid_argument("DPA4 Wigner exponent is out of range");
            }
            exponent_sum += exponent;
        }
        if (exponent_sum != payload.monomial_degree) {
            throw std::invalid_argument("DPA4 Wigner exponent has incorrect degree");
        }
    }
}

void validate_l2_tensor(const Dpa4WignerL2TensorPayload& payload) {
    if (payload.coefficients == nullptr
        || payload.coefficient_count != kDpa4WignerL2TensorValues) {
        throw std::invalid_argument("invalid DPA4 Wigner l=2 tensor payload");
    }
}

void compute_l2_tensor_from_normalized(
    const Dpa4Quaternion& quaternion,
    const Dpa4WignerL2TensorPayload& payload,
    float* output) {
    const std::array<float, 4> components = {
        quaternion.w,
        quaternion.x,
        quaternion.y,
        quaternion.z,
    };
    std::array<float, 16> q2{};
    for (int first = 0; first < 4; ++first) {
        for (int second = 0; second < 4; ++second) {
            q2[static_cast<std::size_t>(first * 4 + second)] =
                components[static_cast<std::size_t>(first)]
                * components[static_cast<std::size_t>(second)];
        }
    }

    for (int row = 0; row < kDpa4WignerL2Dimension; ++row) {
        for (int column = 0; column < kDpa4WignerL2Dimension; ++column) {
            float value = kZero;
            const std::size_t coefficient_offset = static_cast<std::size_t>(
                (row * kDpa4WignerL2Dimension + column) * 256);
            for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < 4; ++b) {
                    for (int c = 0; c < 4; ++c) {
                        for (int d = 0; d < 4; ++d) {
                            const float q4 = q2[static_cast<std::size_t>(a * 4 + b)]
                                * q2[static_cast<std::size_t>(c * 4 + d)];
                            const std::size_t q4_index = static_cast<std::size_t>(
                                ((a * 4 + b) * 4 + c) * 4 + d);
                            value += payload.coefficients[coefficient_offset + q4_index] * q4;
                        }
                    }
                }
            }
            output[row * kDpa4WignerL2Dimension + column] = value;
        }
    }
}

void compute_monomial_block_from_normalized(
    const Dpa4Quaternion& quaternion,
    const Dpa4WignerMonomialPayload& payload,
    float* output) {
    const int dimension = 2 * payload.degree + 1;
    std::array<std::array<float, 7>, 4> powers{};
    const std::array<float, 4> components = {
        quaternion.w,
        quaternion.x,
        quaternion.y,
        quaternion.z,
    };
    for (int component = 0; component < 4; ++component) {
        powers[static_cast<std::size_t>(component)][0] = kOne;
        for (int power = 1; power <= payload.monomial_degree; ++power) {
            powers[static_cast<std::size_t>(component)][static_cast<std::size_t>(power)] =
                powers[static_cast<std::size_t>(component)][static_cast<std::size_t>(power - 1)]
                * components[static_cast<std::size_t>(component)];
        }
    }

    for (int row = 0; row < dimension; ++row) {
        for (int column = 0; column < dimension; ++column) {
            float value = kZero;
            const std::size_t coefficient_offset = static_cast<std::size_t>(
                (row * dimension + column) * payload.monomial_count);
            for (std::size_t monomial = 0; monomial < payload.monomial_count; ++monomial) {
                float term = powers[0][static_cast<std::size_t>(
                    payload.exponents[monomial * 4 + 0])]
                    * powers[1][static_cast<std::size_t>(payload.exponents[monomial * 4 + 1])];
                term *= powers[2][static_cast<std::size_t>(payload.exponents[monomial * 4 + 2])]
                    * powers[3][static_cast<std::size_t>(payload.exponents[monomial * 4 + 3])];
                value += payload.coefficients[coefficient_offset + monomial] * term;
            }
            output[row * dimension + column] = value;
        }
    }
}

void copy_block(
    const float* source,
    int dimension,
    int offset,
    float* destination) {
    for (int row = 0; row < dimension; ++row) {
        std::copy(
            source + row * dimension,
            source + (row + 1) * dimension,
            destination + (offset + row) * kDpa4WignerFullDimension + offset);
    }
}

void validate_edge_batch_arguments(
    const Dpa4EdgeVector* edges,
    std::size_t edge_count,
    const Dpa4Quaternion* output,
    int num_threads,
    const char* name) {
    if (edge_count != 0 && (edges == nullptr || output == nullptr)) {
        throw std::invalid_argument(std::string(name) + " buffer is null");
    }
    if (num_threads < 0) {
        throw std::invalid_argument(std::string(name) + " num_threads must be non-negative");
    }
    if (edge_count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::invalid_argument(std::string(name) + " edge count is too large");
    }
}

} // namespace

Dpa4Quaternion normalize_dpa4_quaternion(
    const Dpa4Quaternion& quaternion,
    float eps) {
    return normalize_quaternion_unchecked(quaternion, eps);
}

Dpa4Quaternion build_dpa4_edge_quaternion(
    const Dpa4EdgeVector& edge,
    float eps) {
    return build_edge_quaternion_from_length(edge, safe_norm(edge, eps), eps);
}

Dpa4Quaternion build_dpa4_edge_quaternion_with_length(
    const Dpa4EdgeVector& edge,
    float edge_length,
    float eps) {
    return build_edge_quaternion_from_length(
        edge,
        std::sqrt(edge_length * edge_length + eps * eps),
        eps);
}

void build_dpa4_edge_quaternions(
    const Dpa4EdgeVector* edges,
    std::size_t edge_count,
    Dpa4Quaternion* output,
    float eps,
    int num_threads) {
    validate_edge_batch_arguments(edges, edge_count, output, num_threads, "DPA4 edge quaternion");
    const std::ptrdiff_t signed_count = static_cast<std::ptrdiff_t>(edge_count);

#ifdef _OPENMP
    if (num_threads > 0) {
#pragma omp parallel for schedule(static) num_threads(num_threads)
        for (std::ptrdiff_t index = 0; index < signed_count; ++index) {
            output[index] = build_dpa4_edge_quaternion(edges[index], eps);
        }
    } else {
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t index = 0; index < signed_count; ++index) {
            output[index] = build_dpa4_edge_quaternion(edges[index], eps);
        }
    }
#else
    (void)num_threads;
    for (std::ptrdiff_t index = 0; index < signed_count; ++index) {
        output[index] = build_dpa4_edge_quaternion(edges[index], eps);
    }
#endif
}

void dpa4_quaternion_to_rotation_matrix(
    const Dpa4Quaternion& quaternion,
    float* output,
    float eps) {
    require_output(output, 9, "DPA4 rotation matrix");
    rotation_matrix_from_normalized(normalize_quaternion_unchecked(quaternion, eps), output);
}

void compute_dpa4_l1_block(
    const Dpa4Quaternion& quaternion,
    float* output,
    float eps) {
    require_output(output, 9, "DPA4 l=1 block");
    l1_block_from_normalized(normalize_quaternion_unchecked(quaternion, eps), output);
}

void validate_dpa4_wigner_payload(const Dpa4WignerPayload& payload) {
    if (payload.l2_tensor.coefficients != nullptr) {
        validate_l2_tensor(payload.l2_tensor);
    } else {
        validate_monomial_payload(payload.l2_monomial, 2);
    }
    validate_monomial_payload(payload.l3, 3);
}

void compute_dpa4_monomial_block(
    const Dpa4Quaternion& quaternion,
    const Dpa4WignerMonomialPayload& payload,
    float* output,
    float eps) {
    validate_monomial_payload(payload, payload.degree);
    const int dimension = 2 * payload.degree + 1;
    require_output(
        output,
        static_cast<std::size_t>(dimension * dimension),
        "DPA4 Wigner monomial block");
    compute_monomial_block_from_normalized(
        normalize_quaternion_unchecked(quaternion, eps),
        payload,
        output);
}

Dpa4WignerLowOrder::Dpa4WignerLowOrder(Dpa4WignerPayload payload)
    : payload_(payload) {
    validate_dpa4_wigner_payload(payload_);
}

void Dpa4WignerLowOrder::compute_l2_block(
    const Dpa4Quaternion& quaternion,
    float* output,
    float eps) const {
    require_output(output, 25, "DPA4 l=2 block");
    const Dpa4Quaternion normalized = normalize_quaternion_unchecked(quaternion, eps);
    if (payload_.l2_tensor.coefficients != nullptr) {
        compute_l2_tensor_from_normalized(normalized, payload_.l2_tensor, output);
    } else {
        compute_monomial_block_from_normalized(normalized, payload_.l2_monomial, output);
    }
}

void Dpa4WignerLowOrder::compute_l3_block(
    const Dpa4Quaternion& quaternion,
    float* output,
    float eps) const {
    require_output(output, 49, "DPA4 l=3 block");
    compute_monomial_block_from_normalized(
        normalize_quaternion_unchecked(quaternion, eps),
        payload_.l3,
        output);
}

void Dpa4WignerLowOrder::compute_blocks(
    const Dpa4Quaternion& quaternion,
    float* output,
    float eps) const {
    require_output(output, 16U * 16U, "DPA4 low-order Wigner blocks");
    std::fill(output, output + 16U * 16U, kZero);
    output[0] = kOne;

    const Dpa4Quaternion normalized = normalize_quaternion_unchecked(quaternion, eps);
    float l1[9];
    float l2[25];
    float l3[49];
    l1_block_from_normalized(normalized, l1);
    if (payload_.l2_tensor.coefficients != nullptr) {
        compute_l2_tensor_from_normalized(normalized, payload_.l2_tensor, l2);
    } else {
        compute_monomial_block_from_normalized(normalized, payload_.l2_monomial, l2);
    }
    compute_monomial_block_from_normalized(normalized, payload_.l3, l3);
    copy_block(l1, 3, 1, output);
    copy_block(l2, 5, 4, output);
    copy_block(l3, 7, 9, output);
}

void Dpa4WignerLowOrder::compute_blocks_batch(
    const Dpa4Quaternion* quaternions,
    std::size_t edge_count,
    float* output,
    float eps,
    int num_threads) const {
    if (edge_count != 0 && (quaternions == nullptr || output == nullptr)) {
        throw std::invalid_argument("DPA4 Wigner batch buffer is null");
    }
    if (num_threads < 0) {
        throw std::invalid_argument("DPA4 Wigner batch num_threads must be non-negative");
    }
    if (edge_count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::invalid_argument("DPA4 Wigner batch edge count is too large");
    }
    const std::ptrdiff_t signed_count = static_cast<std::ptrdiff_t>(edge_count);

#ifdef _OPENMP
    if (num_threads > 0) {
#pragma omp parallel for schedule(static) num_threads(num_threads)
        for (std::ptrdiff_t index = 0; index < signed_count; ++index) {
            compute_blocks(
                quaternions[index],
                output + static_cast<std::size_t>(index) * 16U * 16U,
                eps);
        }
    } else {
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t index = 0; index < signed_count; ++index) {
            compute_blocks(
                quaternions[index],
                output + static_cast<std::size_t>(index) * 16U * 16U,
                eps);
        }
    }
#else
    (void)num_threads;
    for (std::ptrdiff_t index = 0; index < signed_count; ++index) {
        compute_blocks(
            quaternions[index],
            output + static_cast<std::size_t>(index) * 16U * 16U,
            eps);
    }
#endif
}

} // namespace mdescriptor
