#pragma once

#include "mdescriptor/descriptor.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mdescriptor {

// Reader and evaluator for the native MLIP-4 MTP JSON representation.  The
// class deliberately exposes descriptor values only; MLIP-4's energy and
// force parameter paths are outside the descriptor API.
class NativeMtp4Model {
public:
    NativeMtp4Model();
    ~NativeMtp4Model();
    NativeMtp4Model(NativeMtp4Model&&) noexcept;
    NativeMtp4Model& operator=(NativeMtp4Model&&) noexcept;

    NativeMtp4Model(const NativeMtp4Model&) = delete;
    NativeMtp4Model& operator=(const NativeMtp4Model&) = delete;

    void load(const std::string& path);
    std::int64_t feature_count() const noexcept;
    int species_count() const noexcept;
    double min_dist() const noexcept;
    double max_dist() const noexcept;
    int radial_basis_size() const noexcept;
    int radial_funcs_count() const noexcept;
    const std::string& radial_basis_type() const noexcept;
    bool orthogonalized() const noexcept;
    const std::vector<std::int32_t>& species_order() const noexcept;
    const std::vector<double>& parameters() const noexcept;
    double radial_scaling() const noexcept;
    const std::vector<double>& radial_recursive() const noexcept;
    double radial_zeroth() const noexcept;
    double radial_exp_ratio() const noexcept;
    double radial_maxdist_sq() const noexcept;
    double radial_maxdist_sq_minus_eps() const noexcept;
    const std::vector<double>& radial_vdw_params() const noexcept;
    std::int32_t radial_kind() const noexcept;
    const std::vector<std::int32_t>& moments() const noexcept;
    const std::vector<std::int32_t>& eval_kinds() const noexcept;
    const std::vector<std::int32_t>& eval_linear_ids() const noexcept;
    const std::vector<double>& eval_linear_coefficients() const noexcept;
    const std::vector<std::int64_t>& eval_product_offsets() const noexcept;
    const std::vector<std::int32_t>& eval_product_left() const noexcept;
    const std::vector<std::int32_t>& eval_product_right() const noexcept;
    const std::vector<double>& eval_product_coefficients() const noexcept;
    const std::vector<std::int32_t>& scalar_output_ids() const noexcept;

    void compute(
        const StructureBatchView& batch,
        const std::vector<std::int32_t>& species,
        int num_threads,
        double* output,
        const std::shared_ptr<ComputeControl>& control
    ) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mdescriptor
