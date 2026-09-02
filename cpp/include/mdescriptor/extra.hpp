#pragma once

#include "descriptor.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace mdescriptor {

enum class MatrixKind : std::int32_t {
    Sine = 0,
    Ewald = 1,
    Coulomb = 2,
};

void compute_matrix(
    const StructureBatchView& batch,
    std::int64_t n_atoms_max,
    const std::string& permutation,
    double exponent,
    MatrixKind kind,
    double accuracy,
    double w,
    double r_cut,
    double g_cut,
    double a,
    int num_threads,
    double* output,
    const std::shared_ptr<ComputeControl>& control);

enum class MBTRGeometry : std::int32_t {
    AtomicNumber = 0,
    Distance = 1,
    InverseDistance = 2,
    Angle = 3,
    Cosine = 4,
};

enum class MBTRWeighting : std::int32_t {
    Unity = 0,
    Exponential = 1,
    InverseSquare = 2,
    SmoothCutoff = 3,
};

enum class MBTRNormalization : std::int32_t {
    None = 0,
    L2 = 1,
    NAtoms = 2,
    ValleOganov = 3,
};

struct MBTROptions {
    std::vector<std::int32_t> species;
    MBTRGeometry geometry = MBTRGeometry::Distance;
    MBTRWeighting weighting = MBTRWeighting::Exponential;
    MBTRNormalization normalization = MBTRNormalization::None;
    double grid_min = 0.0;
    double grid_max = 6.0;
    double grid_sigma = 0.1;
    int grid_n = 50;
    bool normalize_gaussians = true;
    double scale = 0.5;
    double threshold = 1e-3;
    double r_cut = 6.0;
    double sharpness = 2.0;
    bool local = false;
    int num_threads = 0;
};

std::int64_t mbtr_feature_count(const MBTROptions& options);
void compute_mbtr(
    const StructureBatchView& batch,
    const MBTROptions& options,
    double* output,
    const std::shared_ptr<ComputeControl>& control);

struct EadOptions {
    int max_degree = 3;
    double cutoff = 6.0;
    std::vector<double> eta;
    std::vector<double> rs;
    int num_threads = 0;
};

std::int64_t ead_feature_count(const EadOptions& options);
void compute_ead(
    const StructureBatchView& batch,
    const EadOptions& options,
    double* output,
    const std::shared_ptr<ComputeControl>& control);

struct MtpOptions {
    std::vector<std::int32_t> species;
    // When set, use the official MLIP-2/MLMTPR alpha-index basis from this
    // text .mtp potential instead of the generic compact basis below.
    std::string potential_path;
    // Content identity supplied by the model resolver.  The native model
    // cache must not identify mutable files by path alone.
    std::string model_digest;
    double min_dist = 0.0;
    double max_dist = 5.0;
    int radial_basis_size = 4;
    int radial_funcs_count = 1;
    int max_rank = 2;
    int num_threads = 0;
};

struct OfficialMtpModel;

std::int64_t mtp_feature_count(const MtpOptions& options);
void compute_mtp(
    const StructureBatchView& batch,
    const MtpOptions& options,
    double* output,
    const std::shared_ptr<ComputeControl>& control);

class MtpCalculator {
public:
    MtpCalculator(MtpOptions options);

    std::int64_t feature_count() const noexcept;
    const std::vector<std::int32_t>& species() const noexcept;
    bool official_model() const noexcept;
    bool official_mlip4() const noexcept;
    const std::string& official_format() const noexcept;
    const std::vector<int>& official_alpha_moment_mapping() const noexcept;
    double official_min_dist() const noexcept;
    double official_max_dist() const noexcept;
    int official_radial_basis_size() const noexcept;
    int official_radial_funcs_count() const noexcept;
    const std::string& official_radial_basis_type() const noexcept;
    std::int32_t official_species_count() const noexcept;
    double official_scaling() const noexcept;
    std::int32_t official_alpha_moments_count() const noexcept;
    const std::vector<std::int32_t>& official_alpha_index_basic() const noexcept;
    const std::vector<std::int32_t>& official_alpha_index_times() const noexcept;
    const std::vector<double>& official_radial_coefficients() const noexcept;
    const std::vector<std::int32_t>& official_model_species() const noexcept;
    const std::vector<double>& official_model_parameters() const noexcept;
    double official_radial_scaling() const noexcept;
    const std::vector<double>& official_radial_recursive() const noexcept;
    double official_radial_zeroth() const noexcept;
    double official_radial_exp_ratio() const noexcept;
    double official_radial_maxdist_sq() const noexcept;
    double official_radial_maxdist_sq_minus_eps() const noexcept;
    const std::vector<double>& official_radial_vdw_params() const noexcept;
    std::int32_t official_radial_kind() const noexcept;
    const std::vector<std::int32_t>& official_moments() const noexcept;
    const std::vector<std::int32_t>& official_eval_kinds() const noexcept;
    const std::vector<std::int32_t>& official_eval_linear_ids() const noexcept;
    const std::vector<double>& official_eval_linear_coefficients() const noexcept;
    const std::vector<std::int64_t>& official_eval_product_offsets() const noexcept;
    const std::vector<std::int32_t>& official_eval_product_left() const noexcept;
    const std::vector<std::int32_t>& official_eval_product_right() const noexcept;
    const std::vector<double>& official_eval_product_coefficients() const noexcept;
    const std::vector<std::int32_t>& official_scalar_output_ids() const noexcept;
    void close() noexcept;
    bool closed() const noexcept;

    void compute(
        const StructureBatchView& batch,
        double* output,
        const std::shared_ptr<ComputeControl>& control
    ) const;

private:
    MtpOptions options_;
    std::shared_ptr<OfficialMtpModel> official_model_;
    mutable std::mutex compute_mutex_;
    std::atomic<bool> closed_{false};
};

enum class RotationalDescriptorKind : std::int32_t {
    SO3 = 0,
    SO4 = 1,
    SNAP = 2,
    LBispectrum = 3,
};

struct RotationalDescriptorOptions {
    RotationalDescriptorKind kind = RotationalDescriptorKind::SO3;
    int n_max = 3;
    int l_max = 3;
    double cutoff = 3.5;
    double alpha = 2.0;
    bool weight_on = false;
    bool normalize_u = false;
    double weight_scale = 1.0;
    double rfac0 = 1.0;
    double rmin0 = 0.0;
    double rcutfac = 1.0;
    int num_threads = 0;
    std::vector<double> neighbor_weights;
    std::vector<double> neighbor_radii;
    int twojmax = 3;
    int diagonal = 3;
};

std::int64_t rotational_feature_count(const RotationalDescriptorOptions& options);
void compute_rotational_descriptors(
    const StructureBatchView& batch,
    const RotationalDescriptorOptions& options,
    double* output,
    const std::shared_ptr<ComputeControl>& control);

} // namespace mdescriptor
