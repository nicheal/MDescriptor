#pragma once

#include "descriptor.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace mdescriptor {

// The public ACE adapter intentionally exposes the standard ACE1.jl
// ``Utils.rpi_basis`` construction path.  The native option object contains
// the already-normalised scalar/vector degree form; parsing of the JSON-safe
// Python options lives in the Python adapter.
struct AceOptions {
    std::vector<std::int32_t> species;
    int max_order = 3;
    double r0 = 2.5;
    double transform_p = 2.0;
    double transform_a = 1.0;
    double w_l = 1.5;
    double max_degree = 8.0;
    double degree_csp = 1.0;
    double degree_chc = 0.0;
    double degree_ahc = 0.0;
    double degree_bhc = 0.0;
    std::vector<double> degree_by_order;
    std::vector<double> angular_weight_by_order;
    double r_cut = 5.0;
    double r_in = 1.25;
    int p_cut = 2;
    int p_in = 2;
    bool constants = false;
    int num_threads = 0;
};

class AceCalculator {
public:
    struct Impl;

    explicit AceCalculator(AceOptions options);

    std::int64_t feature_count() const noexcept;
    const std::vector<std::int32_t>& species() const noexcept;
    const std::vector<std::int64_t>& feature_counts() const noexcept;
    std::int32_t max_angular() const noexcept;
    std::int32_t max_radial() const noexcept;
    const std::vector<std::int32_t>& base_species() const noexcept;
    const std::vector<std::int32_t>& base_radial() const noexcept;
    const std::vector<std::int32_t>& base_angular() const noexcept;
    const std::vector<std::int32_t>& base_magnetic() const noexcept;
    const std::vector<double>& radial_a() const noexcept;
    const std::vector<double>& radial_b() const noexcept;
    const std::vector<double>& radial_c() const noexcept;
    double radial_t_left() const noexcept;
    double radial_t_right() const noexcept;
    std::int32_t radial_p_left() const noexcept;
    std::int32_t radial_p_right() const noexcept;
    const std::vector<std::int64_t>& center_feature_offsets() const noexcept;
    const std::vector<std::int64_t>& feature_term_offsets() const noexcept;
    const std::vector<std::int64_t>& term_channel_offsets() const noexcept;
    const std::vector<std::int32_t>& term_channels() const noexcept;
    const std::vector<double>& term_coefficients() const noexcept;
    void close() noexcept;
    bool closed() const noexcept;

    void compute(
        const detail::StructureBatchView& batch,
        double* output,
        const std::shared_ptr<ComputeControl>& control
    ) const;

private:
    AceOptions options_;
    std::vector<std::int64_t> feature_counts_;
    std::int64_t feature_count_ = 0;
    std::int32_t max_angular_ = 0;
    std::int32_t max_radial_ = 0;
    std::vector<std::int32_t> base_species_;
    std::vector<std::int32_t> base_radial_;
    std::vector<std::int32_t> base_angular_;
    std::vector<std::int32_t> base_magnetic_;
    std::vector<double> radial_a_;
    std::vector<double> radial_b_;
    std::vector<double> radial_c_;
    double radial_t_left_ = 0.0;
    double radial_t_right_ = 0.0;
    std::int32_t radial_p_left_ = 0;
    std::int32_t radial_p_right_ = 0;
    std::vector<std::int64_t> center_feature_offsets_;
    std::vector<std::int64_t> feature_term_offsets_;
    std::vector<std::int64_t> term_channel_offsets_;
    std::vector<std::int32_t> term_channels_;
    std::vector<double> term_coefficients_;
    mutable std::mutex compute_mutex_;
    std::atomic<bool> closed_{false};

    std::shared_ptr<const Impl> impl_;
};

} // namespace mdescriptor
