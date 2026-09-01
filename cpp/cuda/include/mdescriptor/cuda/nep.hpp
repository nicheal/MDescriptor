#pragma once

#include "mdescriptor/cuda/batch.hpp"
#include "mdescriptor/cuda/context.hpp"
#include "mdescriptor/cuda/neighbor_graph.hpp"
#include "mdescriptor/nep.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mdescriptor::cuda {

// Device-resident descriptor portion of one parsed NEP model.  The host
// parser remains the single source of truth for the model protocol; this
// object only owns the arrays needed by the descriptor kernel.
class DeviceNepModel {
public:
    DeviceNepModel(
        CudaExecutionContext& context,
        const mdescriptor::NepDescriptorParameters& parameters);
    DeviceNepModel(const DeviceNepModel&) = delete;
    DeviceNepModel& operator=(const DeviceNepModel&) = delete;
    ~DeviceNepModel() noexcept;

    int version() const noexcept { return version_; }
    int num_types() const noexcept { return num_types_; }
    int n_max_radial() const noexcept { return n_max_radial_; }
    int n_max_angular() const noexcept { return n_max_angular_; }
    int basis_size_radial() const noexcept { return basis_size_radial_; }
    int basis_size_angular() const noexcept { return basis_size_angular_; }
    int l_max() const noexcept { return l_max_; }
    int num_l() const noexcept { return num_l_; }
    int dimension() const noexcept { return dimension_; }
    double radial_cutoff_max() const noexcept { return radial_cutoff_max_; }
    double angular_cutoff_max() const noexcept { return angular_cutoff_max_; }
    bool has_q_222() const noexcept { return has_q_222_; }
    bool has_q_1111() const noexcept { return has_q_1111_; }
    bool has_q_112() const noexcept { return has_q_112_; }
    bool has_q_123() const noexcept { return has_q_123_; }
    bool has_q_233() const noexcept { return has_q_233_; }
    bool has_q_134() const noexcept { return has_q_134_; }

    // Returns false for an atomic number not present in the model.  Keeping
    // this check on the host preserves the CPU descriptor's input error
    // semantics instead of silently writing an all-zero GPU row.
    bool supports_atomic_number(std::int32_t number) const noexcept;

    const std::int32_t* type_lookup() const noexcept { return type_lookup_; }
    const float* radial_cutoff_pair() const noexcept { return radial_cutoff_pair_; }
    const float* angular_cutoff_pair() const noexcept { return angular_cutoff_pair_; }
    const float* radial_pair_coefficients() const noexcept {
        return radial_pair_coefficients_;
    }
    const float* angular_pair_coefficients() const noexcept {
        return angular_pair_coefficients_;
    }
    const float* scalers() const noexcept { return scalers_; }

private:
    void release() noexcept;

    int version_ = 0;
    int num_types_ = 0;
    int n_max_radial_ = 0;
    int n_max_angular_ = 0;
    int basis_size_radial_ = 0;
    int basis_size_angular_ = 0;
    int l_max_ = 0;
    int num_l_ = 0;
    int dimension_ = 0;
    double radial_cutoff_max_ = 0.0;
    double angular_cutoff_max_ = 0.0;
    bool has_q_222_ = false;
    bool has_q_1111_ = false;
    bool has_q_112_ = false;
    bool has_q_123_ = false;
    bool has_q_233_ = false;
    bool has_q_134_ = false;
    std::vector<std::int32_t> host_type_lookup_;

    std::int32_t* type_lookup_ = nullptr;
    float* radial_cutoff_pair_ = nullptr;
    float* angular_cutoff_pair_ = nullptr;
    float* radial_pair_coefficients_ = nullptr;
    float* angular_pair_coefficients_ = nullptr;
    float* scalers_ = nullptr;
};

std::vector<double> compute_nep(
    CudaExecutionContext& context,
    const DeviceBatch& batch,
    const DeviceNeighborGraph& graph,
    const DeviceNepModel& model,
    bool reference_radial_accumulation = false);

} // namespace mdescriptor::cuda
