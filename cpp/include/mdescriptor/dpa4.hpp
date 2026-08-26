#pragma once

#include "mdescriptor/descriptor.hpp"
#include "mdescriptor/dpa4_wigner.hpp"
#include "mdescriptor/detail/control.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace mdescriptor {

// The Python checkpoint reader owns the .pt format and converts the validated
// default DPA4 graph into this flat inference ABI.  The C++ side deliberately
// specializes the first production configuration (lmax=3, mmax=1, one focus,
// three blocks) while the Python adapter keeps a NumPy fallback for other DPA4
// configurations.
struct Dpa4BlockOptions {
    bool pre_norm_enabled = false;
    bool post_norm_enabled = true;
    bool ffn_norm_enabled = true;
    std::vector<float> pre_norm_scale;       // [4, 1, 64]
    std::vector<float> pre_norm_bias;        // [1, 64]
    std::vector<float> pre_norm_balance;     // [16]
    std::vector<float> post_norm_scale;      // [4, 1, 64]
    std::vector<float> post_norm_bias;       // [1, 64]
    std::vector<float> post_norm_balance;    // [16]
    std::vector<float> ffn_norm_scale;       // [4, 1, 64]
    std::vector<float> ffn_norm_bias;        // [1, 64]
    std::vector<float> ffn_norm_balance;     // [16]

    std::vector<float> pre_focus_weight;      // [4, 64, 64]
    std::vector<float> post_focus_weight;     // [4, 64, 64]

    std::vector<float> radial_mixer_weight;  // [256, 25]
    std::vector<float> radial_channel_basis; // [1, 64]

    std::array<std::vector<float>, 4> so2_weight_m0; // [256, 256]
    std::array<std::vector<float>, 4> so2_weight_m1; // [192, 384]
    std::array<std::vector<float>, 3> so2_gate_weight; // [64, 192]

    std::vector<float> attn_qk_scale;         // [1, 64]
    std::vector<float> attn_q_weight;         // [64, 64]
    std::vector<float> attn_k_weight;         // [64, 64]
    std::vector<float> attn_output_gate_scale; // [1, 64]
    std::vector<float> attn_logit_weight;     // [64, 1, 1]
    std::vector<float> attn_z_bias_raw;       // [1, 1]
    std::vector<float> attn_gate_weight;      // [64, 1, 1]

    // Message-node SO(3) cross grid product.
    std::vector<float> message_scalar_gate;   // [128, 64]
    std::vector<float> message_frame_expand;  // [4, 64, 192]
    std::vector<float> message_frame_contract; // [4, 192, 64]
    std::vector<float> message_residual_scale; // [1, 64]

    // Block FFN: SO3(64 -> 1152), grid branch, SO3(576 -> 64).
    std::vector<float> ffn_linear1;            // [4, 64, 1152]
    std::vector<float> ffn_linear2;            // [4, 576, 64]
    std::vector<float> ffn_scalar_gate;        // [384, 192]
    std::vector<float> ffn_grid_left;          // [192, 192]
    std::vector<float> ffn_grid_right;         // [192, 192]
    std::vector<float> ffn_grid_router;        // [384, 1]
    std::vector<float> ffn_grid_out;           // [192, 192]
};

struct Dpa4Options {
    double rcut = 6.0;
    int ntypes = 0;
    int channels = 64;
    int n_radial = 16;
    int num_threads = 1;

    std::vector<float> type_embedding;        // [ntypes+1, 64]
    std::vector<float> env_rbf_layer1;        // [16, 32]
    std::vector<float> env_rbf_layer2;        // [32, 32]
    std::vector<float> env_type_embedding;    // [ntypes+1, 16]
    std::vector<float> env_g_layer1;          // [64, 128]
    std::vector<float> env_g_layer2;          // [128, 64]
    std::vector<float> env_output_projection; // [512, 128]
    std::vector<float> film_scale_norm;       // [1, 64]
    std::vector<float> film_shift_norm;       // [1, 64]
    float film_scale_strength_log = 0.0F;
    float film_shift_strength_log = 0.0F;

    std::vector<float> radial_freqs;           // [16]
    std::vector<float> radial_layer1;          // [16, 64]
    std::vector<float> radial_norm_scale;      // [64]
    std::vector<float> radial_layer2;          // [64, 256]

    Dpa4WignerPayload wigner;
    std::vector<float> wigner_l2_tensor;       // owns [25, 256]
    std::vector<float> wigner_l3_coefficients; // owns [49, 84]
    std::vector<std::int64_t> wigner_l3_exponents; // owns [84, 4]

    std::vector<std::int64_t> gie_row_index;   // [15]
    std::vector<std::int64_t> gie_m0_index;    // [15]
    std::vector<std::int64_t> gie_radial_index; // [15]

    // All default l=3,k=1 SO(3) grids use the same deterministic projector.
    std::vector<float> grid_to;                // [152, 48]
    std::vector<float> grid_from;              // [48, 152]

    std::array<Dpa4BlockOptions, 3> blocks;

    // Final output FFN uses the same SO(3) shape as the block FFN, but a
    // polynomial grid MLP instead of a branch mixer.
    std::vector<float> output_linear1;          // [4, 64, 1152]
    std::vector<float> output_linear2;          // [4, 576, 64]
    std::vector<float> output_scalar_gate;      // [384, 192]
    std::vector<float> output_grid_left;        // [384, 384]
    std::vector<float> output_grid_right;       // [384, 384]
    std::vector<float> output_grid_out;         // [384, 192]
};

class Dpa4Calculator {
public:
    explicit Dpa4Calculator(Dpa4Options options);

    std::int64_t feature_count() const noexcept;
    void close() noexcept;
    bool closed() const noexcept;

    void compute(
        const StructureBatchView& batch,
        const std::int32_t* type_indices,
        double* output,
        const std::shared_ptr<ComputeControl>& control
    ) const;

private:
    Dpa4Options options_;
    Dpa4WignerLowOrder wigner_;
    std::int64_t feature_count_ = 64;
    std::atomic<bool> closed_{false};
};

} // namespace mdescriptor
