#pragma once

#include <cstddef>
#include <cstdint>

namespace mdescriptor {

// These constants mirror the low-order paths in the vendored
// dpmodel/descriptor/dpa4_nn/wignerd.py implementation.
constexpr float kDpa4WignerDefaultEpsilon = 1.0e-7F;
constexpr int kDpa4WignerMaxDegree = 3;
constexpr int kDpa4WignerL1Dimension = 3;
constexpr int kDpa4WignerL2Dimension = 5;
constexpr int kDpa4WignerL3Dimension = 7;
constexpr int kDpa4WignerFullDimension = 16;
constexpr std::size_t kDpa4WignerL2TensorValues = 25U * 256U;
constexpr std::size_t kDpa4WignerL2MonomialCount = 35U;
constexpr std::size_t kDpa4WignerL3MonomialCount = 84U;

// Geometry and quaternion values use the DPA4 compute precision (float32), as
// does WignerDCalculator.call after it casts its input and coefficient buffers.
// The quaternion component order is (w, x, y, z).
struct Dpa4EdgeVector {
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
};

struct Dpa4Quaternion {
    float w = 1.0F;
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
};

// Generic monomial coefficient ABI used by C_l3/exp_l3 and, optionally, by
// the pre-expansion C_l2 coefficient table.
//
// exponents is a row-major [monomial_count, 4] int64 buffer in q=(w,x,y,z)
// order. coefficients is a row-major [d*d, monomial_count] float32 buffer,
// where d=2*degree+1 and the first index is the flattened output (row, col).
// For the vendored tables:
//   l=2: degree=2, monomial_degree=4, monomial_count=35;
//   l=3: degree=3, monomial_degree=6, monomial_count=84.
// The payload borrows both buffers; the caller owns them and must keep them
// alive for every computation using the payload.
struct Dpa4WignerMonomialPayload {
    int degree = 0;
    int monomial_degree = 0;
    std::size_t monomial_count = 0;
    const std::int64_t* exponents = nullptr;
    const float* coefficients = nullptr;
};

// Exact layout emitted by WignerSmallOrderCoefficients.C_l2 after
// _build_l2_contraction_tensor(): [5, 5, 4, 4, 4, 4], contiguous C order.
// Flattened index is (((((row * 5 + col) * 4 + a) * 4 + b) * 4 + c) * 4 + d)
// and the term is C[row,col,a,b,c,d] * q[a]q[b]q[c]q[d].
struct Dpa4WignerL2TensorPayload {
    const float* coefficients = nullptr;
    std::size_t coefficient_count = 0;
};

// l2_tensor takes precedence when it is non-null. If it is omitted, l2_monomial
// is consumed instead. l3 is always consumed in monomial form.
struct Dpa4WignerPayload {
    Dpa4WignerL2TensorPayload l2_tensor;
    Dpa4WignerMonomialPayload l2_monomial;
    Dpa4WignerMonomialPayload l3;
};

Dpa4Quaternion normalize_dpa4_quaternion(
    const Dpa4Quaternion& quaternion,
    float eps = kDpa4WignerDefaultEpsilon
);

// Build the global->local edge rotation used by DPA4. Its rotation matrix
// sends the normalized edge direction to local +Z. The implementation follows
// build_edge_quaternion() in the vendored wignerd.py, including the two charts,
// C-infinity blend, shortest-arc nlerp, and epsilon-regularized norms.
Dpa4Quaternion build_dpa4_edge_quaternion(
    const Dpa4EdgeVector& edge,
    float eps = kDpa4WignerDefaultEpsilon
);

// Variant for callers that already computed the edge length. The Python path
// still applies sqrt(edge_length**2 + eps**2), so this is not a raw unit-vector
// shortcut.
Dpa4Quaternion build_dpa4_edge_quaternion_with_length(
    const Dpa4EdgeVector& edge,
    float edge_length,
    float eps = kDpa4WignerDefaultEpsilon
);

void build_dpa4_edge_quaternions(
    const Dpa4EdgeVector* edges,
    std::size_t edge_count,
    Dpa4Quaternion* output,
    float eps = kDpa4WignerDefaultEpsilon,
    int num_threads = 0
);

// Row-major active 3x3 Cartesian rotation matrix for the normalized
// quaternion. This is the matrix called quaternion_to_rotation_matrix() in the
// vendored implementation.
void dpa4_quaternion_to_rotation_matrix(
    const Dpa4Quaternion& quaternion,
    float* output,
    float eps = kDpa4WignerDefaultEpsilon
);

// Row-major packed SeZM l=1 Wigner block. It applies the vendored permutation
// [1, 2, 0] and sign outer-product of [-1, -1, +1] to the Cartesian matrix.
void compute_dpa4_l1_block(
    const Dpa4Quaternion& quaternion,
    float* output,
    float eps = kDpa4WignerDefaultEpsilon
);

void validate_dpa4_wigner_payload(const Dpa4WignerPayload& payload);

// Consume one monomial payload and produce its row-major (2l+1)x(2l+1)
// Wigner block. This is the shared l=2/l=3 evaluator behind the class below.
void compute_dpa4_monomial_block(
    const Dpa4Quaternion& quaternion,
    const Dpa4WignerMonomialPayload& payload,
    float* output,
    float eps = kDpa4WignerDefaultEpsilon
);

class Dpa4WignerLowOrder {
public:
    // The coefficient buffers are borrowed, not copied. Construction validates
    // the shapes and exponent ABI, but does not take ownership of the buffers.
    explicit Dpa4WignerLowOrder(Dpa4WignerPayload payload);

    const Dpa4WignerPayload& payload() const noexcept { return payload_; }

    void compute_l2_block(
        const Dpa4Quaternion& quaternion,
        float* output,
        float eps = kDpa4WignerDefaultEpsilon
    ) const;

    void compute_l3_block(
        const Dpa4Quaternion& quaternion,
        float* output,
        float eps = kDpa4WignerDefaultEpsilon
    ) const;

    // Produce the l<=3 packed block-diagonal matrix with shape [16, 16].
    // Degree l starts at row/column offset l*l, matching D_full in wignerd.py;
    // all off-degree entries are zero and the l=0 block is [1].
    void compute_blocks(
        const Dpa4Quaternion& quaternion,
        float* output,
        float eps = kDpa4WignerDefaultEpsilon
    ) const;

    void compute_blocks_batch(
        const Dpa4Quaternion* quaternions,
        std::size_t edge_count,
        float* output,
        float eps = kDpa4WignerDefaultEpsilon,
        int num_threads = 0
    ) const;

private:
    Dpa4WignerPayload payload_;
};

} // namespace mdescriptor
